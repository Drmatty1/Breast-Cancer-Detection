"""
End-to-End Inference Module for the Multi-Task Mammography Agent.

Governed by: docs/breast_cancer___ps1___methodology.pdf (§3.1, §3.4, §8).
Accepts paired ipsilateral CC and MLO views, executes the multi-head ensemble,
generates Grad-CAM saliency, and emits a schema-validated finding card.

SAFETY & COMPLIANCE NOTICE:
This pipeline is a research prototype. Outputs are machine learning estimates
and do NOT constitute clinical diagnoses or validated medical recommendations.
"""
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import os
import json
import numpy as np
import torch
import torch.nn.functional as F

from src.models.mammo_agent import MammographyAgentModel
from src.mammo.saliency import GradCAM, extract_top_regions, overlay_heatmap_on_image
from src.mammo.finding_card import build_finding_card, validate_finding_card


# Locked Stage 5/6 Calibration Temperatures and Primary Threshold
LOCKED_TEMPERATURES = {
    "fold_0": 1.3864682912826538,
    "fold_1": 1.3713736534118652,
    "fold_2": 0.9705718755722046,
    "fold_3": 1.9837770462036133,
    "fold_4": 1.6316295862197876
}
LOCKED_PRIMARY_THRESHOLD = 0.367477005294

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINTS_DIR = PROJECT_ROOT / "reports" / "mammo_agent_training" / "checkpoints"
DEFAULT_SECONDARY_CALIBRATION_PATH = PROJECT_ROOT / "reports" / "rp2_experimental" / "calibration" / "secondary_heads_calibration.json"


DEFAULT_STAGE5_CALIBRATION_PATH = PROJECT_ROOT / "reports" / "stage5_cross_validation_gpu" / "calibration_models.json"


def validate_calibration_artifact(
    artifact: Any,
    loaded_checkpoints: Optional[List[str]] = None
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validates secondary heads calibration artifact schema, positive finite temperatures,
    class counts, cutoff counts, model fingerprint, and label mapping compatibility.

    Returns:
        (is_valid, validation_message, validated_config)
    """
    fallback_config = {
        "status": "uncalibrated_fallback",
        "heads": {
            "margin_head": {"temperature": 1.0, "status": "uncalibrated"},
            "calc_head": {"temperature": 1.0, "status": "uncalibrated"},
            "birads_ordinal_head": {"temperature": 1.0, "status": "uncalibrated"}
        }
    }

    if not isinstance(artifact, dict):
        return False, "Artifact must be a JSON dictionary", fallback_config

    version = artifact.get("version")
    if not isinstance(version, str) or not version.startswith("rp2"):
        return False, f"Incompatible artifact version: expected rp2.*, got {version}", fallback_config

    # Validate model fingerprint
    fingerprint = artifact.get("model_fingerprint")
    if not isinstance(fingerprint, dict):
        return False, "Artifact missing 'model_fingerprint' dictionary", fallback_config
    if fingerprint.get("model_architecture") != "tf_efficientnet_lite0_multitask":
        return False, f"Model architecture mismatch: expected tf_efficientnet_lite0_multitask, got {fingerprint.get('model_architecture')}", fallback_config

    if loaded_checkpoints:
        known_ckpts = fingerprint.get("checkpoint_files", [])
        for ckpt_name in loaded_checkpoints:
            if ckpt_name not in known_ckpts:
                return False, f"Loaded checkpoint '{ckpt_name}' not recognized in calibration lineage", fallback_config

    # Validate label mappings
    label_mappings = artifact.get("label_mappings")
    if not isinstance(label_mappings, dict):
        return False, "Artifact missing 'label_mappings' dictionary", fallback_config

    expected_labels = {
        "margin_head": ["circumscribed", "indistinct", "obscured", "spiculated"],
        "calc_head": ["amorphous", "coarse", "fine_linear_branching", "pleomorphic", "punctate"],
        "birads_ordinal_head": ["2", "3", "4", "5"]
    }
    for h_name, exp_classes in expected_labels.items():
        if h_name not in label_mappings:
            return False, f"Label mappings missing head: {h_name}", fallback_config
        h_map = label_mappings[h_name]
        if not isinstance(h_map, dict) or h_map.get("classes") != exp_classes:
            return False, f"Class mapping mismatch for {h_name}: expected {exp_classes}, got {h_map.get('classes') if isinstance(h_map, dict) else None}", fallback_config

    # Validate heads and temperatures
    heads = artifact.get("heads")
    if not isinstance(heads, dict):
        return False, "Artifact missing 'heads' dictionary", fallback_config

    required_specs = {
        "margin_head": {"class_count": 4},
        "calc_head": {"class_count": 5},
        "birads_ordinal_head": {"cutoff_count": 3, "class_count": 4}
    }

    parsed_heads = {}
    for h_name, expected_spec in required_specs.items():
        if h_name not in heads:
            return False, f"Artifact missing required head: {h_name}", fallback_config

        h_data = heads[h_name]
        if not isinstance(h_data, dict):
            return False, f"Head entry '{h_name}' must be a dictionary", fallback_config

        temp = h_data.get("temperature")
        if not isinstance(temp, (int, float)) or not np.isfinite(temp) or temp <= 0.05 or temp > 100.0:
            return False, f"Invalid temperature for {h_name}: {temp} (must be finite in [0.05, 100.0])", fallback_config

        for spec_k, spec_v in expected_spec.items():
            if h_data.get(spec_k) != spec_v:
                return False, f"Specification mismatch for {h_name}: expected {spec_k}={spec_v}, got {h_data.get(spec_k)}", fallback_config

        status = h_data.get("status", "uncalibrated")
        if status not in ("calibrated", "uncalibrated"):
            return False, f"Invalid status for {h_name}: {status} (must be 'calibrated' or 'uncalibrated')", fallback_config

        parsed_heads[h_name] = {
            "temperature": float(temp),
            "status": status,
            "limitation": h_data.get("limitation")
        }

    valid_config = {
        "status": "valid",
        "version": version,
        "model_fingerprint": fingerprint,
        "label_mappings": label_mappings,
        "heads": parsed_heads
    }
    return True, "Valid", valid_config


def pava_isotonic_non_increasing(cutoff_probs: np.ndarray) -> Tuple[np.ndarray, bool]:
    r"""
    Exact weighted least squares isotonic regression onto the non-increasing cone:
    minimize sum_{i=0}^2 (q_i - \hat{q}_i)^2 subject to 1.0 >= \hat{q}_0 >= \hat{q}_1 >= \hat{q}_2 >= 0.0
    with equal weights w = [1, 1, 1].
    """
    arr = np.asarray(cutoff_probs, dtype=np.float64)
    q0, q1, q2 = float(arr[0]), float(arr[1]), float(arr[2])

    was_violated = (q0 < q1) or (q1 < q2)
    if not was_violated:
        return np.copy(arr), False

    if q0 < q1 and q1 >= q2:
        m01 = (q0 + q1) / 2.0
        if m01 >= q2:
            res = [m01, m01, q2]
        else:
            m012 = (q0 + q1 + q2) / 3.0
            res = [m012, m012, m012]
    elif q0 >= q1 and q1 < q2:
        m12 = (q1 + q2) / 2.0
        if q0 >= m12:
            res = [q0, m12, m12]
        else:
            m012 = (q0 + q1 + q2) / 3.0
            res = [m012, m012, m012]
    else:  # q0 < q1 and q1 < q2
        m012 = (q0 + q1 + q2) / 3.0
        res = [m012, m012, m012]

    return np.clip(np.array(res, dtype=np.float64), 0.0, 1.0), True


def enforce_ordinal_monotonicity(
    cutoff_probs: np.ndarray,
    method: str = "sequential_upper_bound"
) -> Tuple[np.ndarray, bool]:
    """
    Applies a checkpoint-compatible inference safeguard to cumulative ordinal cutoff probabilities.

    Cumulative ordinal link formulation defines:
      q_0 = P(Y > 2)
      q_1 = P(Y > 3)
      q_2 = P(Y > 4)
    Theoretical consistency requires monotonicity: 1.0 >= q_0 >= q_1 >= q_2 >= 0.0.

    Because OrdinalBiRadsHead uses unconstrained linear projections without architectural
    monotonicity constraints, finite-sample logits can produce order inversions (e.g. q_1 > q_0).

    Methods:
      - 'sequential_upper_bound' (default): Clamps downstream cutoffs sequentially:
          q_0* = q_0, q_1* = min(q_0*, q_1), q_2* = min(q_1*, q_2).
      - 'pava': Symmetrical weighted least-squares isotonic regression via Pool Adjacent Violators.

    Note: This is an inference-only projection that restores cumulative validity without
    modifying checkpoint weights. It is NOT equivalent to an architecturally constrained ordinal model.

    Args:
        cutoff_probs: 1D array of cumulative cutoff probabilities in [0.0, 1.0].
        method: Projection method ('sequential_upper_bound' or 'pava').

    Returns:
        (projected_probs, was_violated): Monotonic probabilities and boolean violation flag.
    """
    if method == "pava":
        return pava_isotonic_non_increasing(cutoff_probs)
    elif method == "sequential_upper_bound":
        arr = np.asarray(cutoff_probs, dtype=np.float64)
        was_violated = False
        projected = np.copy(arr)

        for i in range(1, len(projected)):
            if projected[i] > projected[i - 1]:
                was_violated = True
                projected[i] = projected[i - 1]

        return projected, was_violated
    else:
        raise ValueError(f"Unknown monotonicity projection method '{method}'. Expected 'sequential_upper_bound' or 'pava'.")


class MammographyAgentInferencePipeline:
    """
    Unified end-to-end inference engine for the 4-head Mammography Agent.
    Supports single-checkpoint or 5-checkpoint ensemble evaluation with locked calibration.
    """
    def __init__(
        self,
        checkpoint_paths: Optional[List[Union[str, Path]]] = None,
        device: Optional[torch.device] = None,
        secondary_calibration_path: Optional[Union[str, Path]] = None,
        monotonicity_projection_method: str = "sequential_upper_bound"
    ):
        self.monotonicity_projection_method = monotonicity_projection_method
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if checkpoint_paths is None:
            # Default to all 5 trained fold checkpoints
            checkpoint_paths = [
                DEFAULT_CHECKPOINTS_DIR / f"mammo_agent_fold{k}_best.pt"
                for k in range(5)
            ]

        self.models: List[MammographyAgentModel] = []
        self.checkpoint_names: List[str] = []
        self.temperatures: List[float] = []

        for p in checkpoint_paths:
            path_obj = Path(p)
            if not path_obj.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {path_obj}")

            model = MammographyAgentModel(pretrained_backbone=False).to(self.device)
            ckpt = torch.load(path_obj, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)

            # Verify checkpoint key compatibility
            model_keys = set(model.state_dict().keys())
            ckpt_keys = set(state_dict.keys())
            missing = model_keys - ckpt_keys
            if missing:
                raise ValueError(f"Checkpoint {path_obj.name} missing required keys: {list(missing)[:5]}")

            model.load_state_dict(state_dict)
            model.eval()
            self.models.append(model)
            self.checkpoint_names.append(path_obj.name)

            # Identify fold temperature and verify Stage 5 calibration status
            fold_idx = ckpt.get("fold")
            fold_key = f"fold_{fold_idx}" if fold_idx is not None else None
            temp = LOCKED_TEMPERATURES.get(fold_key, 1.0) if fold_key else 1.0
            self.temperatures.append(temp)

        # Audit binary head calibration status across loaded ensemble
        # Binary head is calibrated if and only if every loaded model has a verified Stage 5 calibration temperature
        all_binary_calibrated = (
            len(self.models) > 0
            and all(t != 1.0 for t in self.temperatures)
        )
        self.binary_calib_status: str = "calibrated" if all_binary_calibrated else "uncalibrated"

        # Configurable and portable secondary heads calibration resolution
        if secondary_calibration_path is not None:
            calib_path = Path(secondary_calibration_path)
        elif os.environ.get("RP2_CALIBRATION_PATH"):
            calib_path = Path(os.environ["RP2_CALIBRATION_PATH"])
        else:
            calib_path = DEFAULT_SECONDARY_CALIBRATION_PATH

        self.secondary_calib: Optional[Dict[str, Any]] = None
        self.secondary_calib_status: str = "missing"
        self.secondary_calib_error: Optional[str] = None

        if calib_path.is_file():
            try:
                with open(calib_path, "r", encoding="utf-8") as f:
                    raw_artifact = json.load(f)
                is_valid, msg, validated_config = validate_calibration_artifact(
                    raw_artifact,
                    loaded_checkpoints=self.checkpoint_names
                )
                if is_valid:
                    self.secondary_calib = validated_config
                    self.secondary_calib_status = "loaded"
                else:
                    self.secondary_calib = validated_config
                    self.secondary_calib_status = "invalid_fallback"
                    self.secondary_calib_error = msg
            except Exception as e:
                _, _, fallback_config = validate_calibration_artifact(None)
                self.secondary_calib = fallback_config
                self.secondary_calib_status = "malformed_fallback"
                self.secondary_calib_error = str(e)

    def _prepare_tensor(self, img_input: Union[torch.Tensor, np.ndarray, str, Path]) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Converts input into a standardized float32 tensor [1, 1, 512, 512] in [-1, 1]
        and returns the raw 2D numpy array [512, 512] for overlay generation.
        """
        if isinstance(img_input, (str, Path)):
            p = Path(img_input)
            if not p.is_file():
                raise FileNotFoundError(f"Image input file not found: {p}")

            if p.suffix.lower() == ".npy":
                arr = np.load(p).astype(np.float32)
            elif p.suffix.lower() == ".dcm":
                # Lazy import preprocessing for raw DICOMs
                from src.preprocessing.mammography import process_mammogram
                processed = process_mammogram(p)
                arr = processed.image.astype(np.float32)
            else:
                raise ValueError(f"Unsupported file format: {p.suffix}. Expected .npy or .dcm.")
        elif isinstance(img_input, np.ndarray):
            arr = img_input.astype(np.float32)
        elif isinstance(img_input, torch.Tensor):
            arr = img_input.detach().cpu().numpy().astype(np.float32)
            if arr.ndim == 4:
                arr = arr.squeeze(0).squeeze(0)
            elif arr.ndim == 3:
                arr = arr.squeeze(0)
        else:
            raise TypeError(f"Unsupported input type: {type(img_input)}")

        if arr.ndim != 2 or arr.shape != (512, 512):
            raise ValueError(f"Input image must have shape (512, 512); received shape {arr.shape}.")

        # Ensure normalized to [-1.0, 1.0]
        if arr.min() >= 0.0 and arr.max() <= 1.0:
            norm_arr = arr * 2.0 - 1.0
        else:
            norm_arr = arr

        tensor = torch.from_numpy(norm_arr).unsqueeze(0).unsqueeze(0).float()
        return tensor, arr

    def predict(
        self,
        cc_input: Union[torch.Tensor, np.ndarray, str, Path],
        mlo_input: Union[torch.Tensor, np.ndarray, str, Path],
        generate_saliency: bool = True,
        saliency_output_dir: Optional[Union[str, Path]] = None,
        pair_identifier: str = "sample_pair",
        clinical_header: Optional[Dict[str, Any]] = None,
        abnormality_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end multi-task inference on a paired CC/MLO mammogram.

        Args:
            cc_input: Craniocaudal view (tensor, numpy array, or file path).
            mlo_input: Mediolateral oblique view (tensor, numpy array, or file path).
            generate_saliency: Whether to compute Grad-CAM saliency.
            saliency_output_dir: Directory to save saliency overlay images.
            pair_identifier: Identifier string for filenames and card metadata.
            clinical_header: Dict of clinical metadata and indications (optional).
            abnormality_type: Explicit clinical finding indicator ('mass', 'calcification', 'both', or None).
                              Used to govern secondary finding assertions and prevent hallucinations.

        Returns:
            Validated Finding Card dictionary.
        """
        if cc_input is None or mlo_input is None:
            raise ValueError("Mammo Agent requires both CC and MLO views; single-view processing is not permitted.")

        # Prepare input tensors
        t_cc, raw_cc = self._prepare_tensor(cc_input)
        t_mlo, raw_mlo = self._prepare_tensor(mlo_input)

        t_cc = t_cc.to(self.device)
        t_mlo = t_mlo.to(self.device)

        # Ensemble accumulation
        binary_cal_probs = []
        margin_prob_dist = []
        calc_prob_dist = []
        birads_cutoff_probs = []

        # Check secondary heads calibration status
        if self.secondary_calib and self.secondary_calib_status == "loaded":
            heads_meta = self.secondary_calib.get("heads", {})
            t_margin = float(heads_meta.get("margin_head", {}).get("temperature", 1.0))
            margin_status = heads_meta.get("margin_head", {}).get("status", "uncalibrated")

            t_calc = float(heads_meta.get("calc_head", {}).get("temperature", 1.0))
            calc_status = heads_meta.get("calc_head", {}).get("status", "uncalibrated")

            t_birads = float(heads_meta.get("birads_ordinal_head", {}).get("temperature", 1.0))
            birads_status = heads_meta.get("birads_ordinal_head", {}).get("status", "uncalibrated")
        else:
            t_margin = 1.0
            margin_status = "uncalibrated"
            t_calc = 1.0
            calc_status = "uncalibrated"
            t_birads = 1.0
            birads_status = "uncalibrated"

        with torch.no_grad():
            for i, model in enumerate(self.models):
                preds = model(t_cc, t_mlo)

                # 1. Binary Malignancy: scale with fold locked temperature
                logit_bin = float(preds["binary_logit"].item())
                temp = self.temperatures[i]
                p_bin = 1.0 / (1.0 + np.exp(-logit_bin / temp))
                binary_cal_probs.append(p_bin)

                # 2. Margin: Softmax distribution (temperature-scaled)
                p_margin = F.softmax(preds["margin_logits"] / t_margin, dim=-1).squeeze(0).cpu().numpy()
                margin_prob_dist.append(p_margin)

                # 3. Calcification: Softmax distribution (temperature-scaled)
                p_calc = F.softmax(preds["calc_logits"] / t_calc, dim=-1).squeeze(0).cpu().numpy()
                calc_prob_dist.append(p_calc)

                # 4. Ordinal BI-RADS: Sigmoid of cutoff logits (temperature-scaled)
                q_birads = torch.sigmoid(preds["birads_ordinal_logits"] / t_birads).squeeze(0).cpu().numpy()
                birads_cutoff_probs.append(q_birads)

        # Aggregate ensemble averages
        mean_binary_prob = float(np.mean(binary_cal_probs))
        mean_margin_probs = np.mean(margin_prob_dist, axis=0).tolist()
        mean_calc_probs = np.mean(calc_prob_dist, axis=0).tolist()
        mean_cutoff_probs = np.mean(birads_cutoff_probs, axis=0)

        # Apply inference-only ordinal monotonicity safeguard
        mean_cutoff_probs_mono, was_violated = enforce_ordinal_monotonicity(
            mean_cutoff_probs,
            method=self.monotonicity_projection_method
        )

        # Ordinal level = count of cutoffs > 0.5
        predicted_birads_level = int((mean_cutoff_probs_mono > 0.5).sum())

        if self.secondary_calib and self.secondary_calib_status == "loaded":
            # Discrete probability distribution over BI-RADS levels {2, 3, 4, 5}
            q0 = mean_cutoff_probs_mono[0]
            q1 = mean_cutoff_probs_mono[1]
            q2 = mean_cutoff_probs_mono[2]

            p0 = max(0.0, 1.0 - q0)
            p1 = max(0.0, q0 - q1)
            p2 = max(0.0, q1 - q2)
            p3 = max(0.0, q2)
            sum_p = p0 + p1 + p2 + p3
            if sum_p > 0:
                birads_category_probs = [p0 / sum_p, p1 / sum_p, p2 / sum_p, p3 / sum_p]
            else:
                birads_category_probs = [0.25, 0.25, 0.25, 0.25]

            # Decision confidence score is the probability of the predicted category in [0.25, 1.0]
            birads_conf = float(birads_category_probs[predicted_birads_level])

            # Explicit confidence calibration status: Only marked 'calibrated' if head status is calibrated
            if birads_status == "calibrated":
                confidence_status = "calibrated"
            else:
                confidence_status = "uncalibrated_discrete_probability"
        else:
            # Fallback to legacy uncalibrated distance heuristic if calibration artifact is unavailable or invalid
            birads_conf = float(np.max(np.abs(mean_cutoff_probs_mono - 0.5)) * 2.0)
            birads_category_probs = None
            confidence_status = "uncalibrated_heuristic"

        # Grad-CAM Saliency Generation
        saliency_regions = []
        saliency_path = None
        cc_saliency_path = None
        mlo_saliency_path = None

        if generate_saliency and len(self.models) > 0:
            cam_engine = GradCAM(self.models[0])
            cam_results = cam_engine.generate_cam(t_cc, t_mlo, task="binary")
            cam_cc = cam_results["cam_cc"][0]
            cam_mlo = cam_results["cam_mlo"][0]
            cam_engine.remove_hooks()

            # Extract top sensitive regions on both views
            regions_cc = extract_top_regions(cam_cc, threshold=0.5, max_regions=2)
            regions_mlo = extract_top_regions(cam_mlo, threshold=0.5, max_regions=2)
            saliency_regions = regions_cc + regions_mlo

            # Save overlay images if directory provided
            if saliency_output_dir is not None:
                out_dir = Path(saliency_output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)

                overlay_cc = overlay_heatmap_on_image(raw_cc, cam_cc)
                overlay_mlo = overlay_heatmap_on_image(raw_mlo, cam_mlo)

                cc_path = out_dir / f"{pair_identifier}_gradcam_CC.png"
                mlo_path = out_dir / f"{pair_identifier}_gradcam_MLO.png"

                # Save RGB as BGR via OpenCV
                import cv2
                cv2.imwrite(str(cc_path), cv2.cvtColor(overlay_cc, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(mlo_path), cv2.cvtColor(overlay_mlo, cv2.COLOR_RGB2BGR))

                cc_saliency_path = str(cc_path)
                mlo_saliency_path = str(mlo_path)
                saliency_path = cc_saliency_path

        # Build Finding Card
        checkpoint_summary = f"{len(self.models)}-model ensemble ({', '.join(self.checkpoint_names)})"
        card = build_finding_card(
            binary_prob_calibrated=mean_binary_prob,
            margin_probs=mean_margin_probs,
            calc_probs=mean_calc_probs,
            birads_ordinal_level=predicted_birads_level,
            birads_confidence=birads_conf,
            saliency_regions=saliency_regions,
            saliency_path=saliency_path,
            cc_saliency_path=cc_saliency_path,
            mlo_saliency_path=mlo_saliency_path,
            clinical_header=clinical_header,
            checkpoint_info=checkpoint_summary,
            threshold_malignancy=LOCKED_PRIMARY_THRESHOLD,
            abnormality_type=abnormality_type
        )

        # Record explicit calibration and audit metadata
        card["metadata"]["ordinal_monotonicity"] = {
            "was_violated": was_violated,
            "projection_method": self.monotonicity_projection_method,
            "raw_cutoff_probs": [float(round(p, 4)) for p in mean_cutoff_probs],
            "projected_cutoff_probs": [float(round(p, 4)) for p in mean_cutoff_probs_mono]
        }
        card["metadata"]["confidence_calibration_status"] = confidence_status
        card["metadata"]["head_calibration_status"] = {
            "binary": self.binary_calib_status,
            "margin": margin_status,
            "calc": calc_status,
            "birads": birads_status
        }
        card["metadata"]["binary_calibration"] = {
            "status": self.binary_calib_status,
            "temperatures": self.temperatures,
            "threshold": LOCKED_PRIMARY_THRESHOLD
        }
        card["metadata"]["secondary_calibration"] = {
            "artifact_status": self.secondary_calib_status,
            "artifact_error": self.secondary_calib_error,
            "birads_category_distribution": [float(round(p, 4)) for p in birads_category_probs] if birads_category_probs else None,
            "temperatures": {
                "margin": t_margin,
                "calc": t_calc,
                "birads": t_birads
            }
        }

        # Enforce formal schema validation
        is_valid, err = validate_finding_card(card)
        if not is_valid:
            raise ValueError(f"Generated finding card failed schema validation: {err}")

        return card
