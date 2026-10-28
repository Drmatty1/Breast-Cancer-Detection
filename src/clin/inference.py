"""
Clinical Risk Agent Inference Module.

Governed by:
- docs/breast_cancer___ps1___methodology.pdf (§§ 3.2, 5.1)
- DECISIONS.md (DEC-OWNER-01, DEC-OWNER-02, DEC-OWNER-03)
- schemas/finding_card.schema.json
- reports/clin/CLIN_AGENT_C3_7_ORCHESTRATION_CONTRACT.md

Implements:
1. Strict input allowlisting for ['breast_density', 'breast_side'].
2. 5-fold XGBoost model loading and post-hoc temperature scaling.
3. Native TreeSHAP attribution extraction.
4. Schema-compliant Finding Card generation for research shadow logging.
5. Strict non-authoritative status: voting_eligibility=False, conf_a=0.0.
6. Clean abstention card generation on missing or invalid covariates.

SAFETY & COMPLIANCE NOTICE:
This module is a research prototype. It exhibits NO standalone discrimination
for breast cancer on CBIS-DDSM (ROC-AUC ~ 0.49). It executes strictly in
shadow/logging mode and must NOT be used for clinical diagnosis or consensus voting.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import os
import json
import time
import math
import numpy as np
import xgboost as xgb
import jsonschema


DEFAULT_MODELS_DIR = Path(r"D:\rp\reports\clin\stage_c3_training\models")
DEFAULT_SCHEMA_PATH = Path(r"D:\rp\schemas\finding_card.schema.json")
APPROVED_PREDICTORS = ("breast_density", "breast_side")

VALID_DENSITIES = {1, 2, 3, 4}
SIDE_MAP = {
    "LEFT": 0.0,
    "L": 0.0,
    "RIGHT": 1.0,
    "R": 1.0
}


def validate_schema(card: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validates finding card against JSON schema."""
    try:
        jsonschema.validate(instance=card, schema=schema)
        return True, None
    except jsonschema.ValidationError as e:
        return False, f"Schema validation error: {e.message} at path: {list(e.path)}"
    except Exception as e:
        return False, f"Unexpected validation error: {str(e)}"


def safe_str(val: Any) -> str:
    """Safely converts any object to string without raising an exception."""
    try:
        return str(val)
    except Exception:
        return f"<unprintable_{type(val).__name__}>"


def sanitize_for_json(
    val: Any,
    seen: Optional[set] = None,
    depth: int = 0,
    max_depth: int = 32
) -> Any:
    """
    Safely and recursively sanitizes arbitrary objects for strict RFC 8259 JSON compliance.

    Guarantees:
    1. Returns native Python scalars for numbers, booleans, strings, None.
    2. Converts non-finite floats (NaN, +/-Inf) to 'nan', 'inf', '-inf'.
    3. Recursively sanitizes nested dictionaries and lists/tuples without mutating inputs.
    4. Guards against cycles via object ID tracking and returns '<circular_ref>'.
    5. Guards against deep nesting via max_depth guard and returns '<max_depth_exceeded>'.
    6. Converts unsupported objects (sets, custom classes) to deterministic safe strings.
    7. Catches broken __str__ exceptions and returns '<unsupported_unprintable: ClassName>'.
    8. Never leaks unhandled exceptions.
    """
    try:
        if depth >= max_depth:
            return "<max_depth_exceeded>"

        if val is None:
            return None

        # Booleans must be checked before int (in Python, isinstance(True, int) is True)
        if isinstance(val, bool):
            return val

        # Handle NumPy scalars or 0-d arrays exposing .item()
        if hasattr(val, "item") and callable(getattr(val, "item")) and not isinstance(val, (list, tuple, dict, set)):
            try:
                native = val.item()
                if native is not val:
                    return sanitize_for_json(native, seen, depth + 1, max_depth)
            except Exception:
                pass

        # Numeric integer scalars
        if isinstance(val, (int, np.integer)):
            return int(val)

        # Floating-point scalars (Python and NumPy)
        if isinstance(val, (float, np.floating)):
            try:
                f_val = float(val)
                if math.isnan(f_val):
                    return "nan"
                if math.isinf(f_val):
                    return "-inf" if f_val < 0 else "inf"
                return f_val
            except Exception:
                return safe_str(val)

        # Standard strings
        if isinstance(val, str):
            return val

        # Tracking for circular references on container or unsupported objects
        val_id = id(val)
        if seen is None:
            seen = set()
        if val_id in seen:
            return "<circular_ref>"
        new_seen = seen | {val_id}

        # Dictionaries
        if isinstance(val, dict):
            sanitized_dict = {}
            for k, v in val.items():
                safe_k = sanitize_for_json(k, new_seen, depth + 1, max_depth)
                sanitized_dict[str(safe_k)] = sanitize_for_json(v, new_seen, depth + 1, max_depth)
            return sanitized_dict

        # Sequences: lists and tuples
        if isinstance(val, (list, tuple)):
            return [sanitize_for_json(item, new_seen, depth + 1, max_depth) for item in val]

        # Unsupported objects (sets, custom classes, functions, etc.)
        try:
            val_str = str(val)
            return f"<unsupported: {type(val).__name__}: {val_str}>"
        except Exception:
            return f"<unsupported_unprintable: {type(val).__name__}>"

    except Exception:
        # Ultimate fail-safe guard
        try:
            return f"<sanitization_failed: {type(val).__name__}>"
        except Exception:
            return "<sanitization_failed>"


class ClinicalRiskAgentInferencePipeline:
    """
    Ensemble inference pipeline for the Clinical Risk Agent.
    Evaluates 5-fold XGBoost models on allowlisted tabular features.
    """
    def __init__(
        self,
        models_dir: Optional[Union[str, Path]] = None,
        schema_path: Optional[Union[str, Path]] = None,
        lazy_load: bool = False
    ):
        self.models_dir = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
        self.schema_path = Path(schema_path) if schema_path is not None else DEFAULT_SCHEMA_PATH
        
        self.schema: Optional[Dict[str, Any]] = None
        self._load_schema()

        self.models: List[xgb.Booster] = []
        self.temperatures: List[float] = []
        self.is_initialized: bool = False
        self.init_error: Optional[str] = None

        if not lazy_load:
            self._load_models()

    def _load_schema(self) -> None:
        """Loads formal JSON schema for contract validation."""
        if self.schema_path.is_file():
            with open(self.schema_path, "r", encoding="utf-8") as f:
                self.schema = json.load(f)
        else:
            self.schema = None

    def _load_models(self) -> None:
        """Loads the 5 fold models and temperature calibration parameters."""
        try:
            if not self.models_dir.is_dir():
                raise FileNotFoundError(f"Clin models directory not found: {self.models_dir}")

            calib_file = self.models_dir / "calibration_parameters.json"
            if not calib_file.is_file():
                raise FileNotFoundError(f"Missing calibration parameters at: {calib_file}")

            with open(calib_file, "r", encoding="utf-8") as f:
                calib_data = json.load(f)

            self.models = []
            self.temperatures = []

            for k in range(5):
                model_file = self.models_dir / f"fold{k}_model.json"
                if not model_file.is_file():
                    raise FileNotFoundError(f"Missing model file: {model_file}")

                bst = xgb.Booster()
                bst.load_model(str(model_file))
                self.models.append(bst)

                fold_key = f"fold_{k}"
                if fold_key in calib_data and "temperature" in calib_data[fold_key]:
                    temp = float(calib_data[fold_key]["temperature"])
                else:
                    temp = 1.0
                self.temperatures.append(temp)

            self.is_initialized = True
            self.init_error = None
        except Exception as e:
            self.is_initialized = False
            self.init_error = str(e)
            self.models = []
            self.temperatures = []

    def validate_and_extract_features(
        self,
        clinical_covariates: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Validates covariates against the strict allowlist.
        Returns: (density_val, side_val, rejection_reason)
        """
        try:
            if clinical_covariates is None or not isinstance(clinical_covariates, dict):
                return None, None, "missing_or_non_dict_covariates"

            # Explicit allowlist extraction: only breast_density and breast_side are accessed
            raw_density = clinical_covariates.get("breast_density")
            raw_side = clinical_covariates.get("breast_side")

            if raw_density is None or raw_side is None:
                return None, None, "missing_required_covariates"

            # Explicitly reject boolean values (in Python, isinstance(True, int) is True)
            if isinstance(raw_density, bool):
                return None, None, f"invalid_breast_density_boolean_{raw_density}"

            # Validate density numeric value
            try:
                if isinstance(raw_density, str):
                    density_float = float(raw_density.strip())
                elif isinstance(raw_density, (int, float, np.integer, np.floating)):
                    density_float = float(raw_density)
                else:
                    return None, None, f"unsupported_breast_density_type_{type(raw_density).__name__}"

                # Reject NaN and +/- Infinity
                if math.isnan(density_float) or math.isinf(density_float):
                    return None, None, f"invalid_breast_density_non_finite_{safe_str(raw_density)}"

                # Strictly reject fractional values (e.g. 2.5)
                if not density_float.is_integer():
                    return None, None, f"fractional_breast_density_rejected_{safe_str(raw_density)}"

                density_int = int(density_float)
                if density_int not in VALID_DENSITIES:
                    return None, None, f"unsupported_breast_density_category_{density_int}"

                # Validated, normalized feature: float 1.0, 2.0, 3.0, or 4.0
                density_val = float(density_int)
            except (ValueError, TypeError, OverflowError):
                return None, None, f"malformed_breast_density_{safe_str(raw_density)}"

            # Validate side
            if not isinstance(raw_side, str):
                return None, None, f"malformed_breast_side_type_{type(raw_side)}"

            norm_side = raw_side.strip().upper()
            if norm_side not in SIDE_MAP:
                return None, None, f"unsupported_breast_side_{safe_str(raw_side)}"

            side_val = SIDE_MAP[norm_side]
            return density_val, side_val, None
        except Exception as e:
            return None, None, f"covariate_validation_error_{type(e).__name__}"

    def build_abstention_card(
        self,
        reason: str,
        clinical_covariates: Optional[Dict[str, Any]] = None,
        pair_identifier: str = "sample_pair"
    ) -> Dict[str, Any]:
        """
        Emits a schema-valid abstention finding card per DEC-OWNER-02 and DEC-OWNER-03.
        """
        raw_d = clinical_covariates.get("breast_density") if isinstance(clinical_covariates, dict) else None
        raw_s = clinical_covariates.get("breast_side") if isinstance(clinical_covariates, dict) else None

        # Recursively sanitize raw covariate values to guarantee strict JSON serialization compliance
        raw_d = sanitize_for_json(raw_d)
        raw_s = sanitize_for_json(raw_s)

        card = {
            "agent": "clin",
            "finding": {
                "lesion": None,
                "margin": None,
                "calcification": None,
                "risk_profile": "unavailable",
                "abstention_reason": reason
            },
            "BI_RADS_a": "unspecified",
            "conf_a": 0.0,
            "sal": {
                "top_regions": [],
                "path": None,
                "cc_path": None,
                "mlo_path": None
            },
            "sh": {
                "breast_density": raw_d,
                "breast_side": raw_s
            },
            "rationale": (
                f"Clinical Risk Agent abstained from prediction. Reason: {reason}. "
                "Agent operates strictly in non-authoritative shadow mode with zero consensus weight."
            ),
            "metadata": {
                "model_architecture": "xgboost_tabular_logistic_ensemble",
                "model_version": "clin_agent_v1.0",
                "pair_identifier": pair_identifier,
                "status": "abstained",
                "authority": "non_authoritative_research",
                "voting_eligibility": False,
                "is_clinically_validated": False,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "disclaimer": "Research prototype output only. Non-authoritative shadow mode.",
                "confidence_provenance": {
                    "conf_a_source": "non_authoritative_zero_weight_convention",
                    "is_conf_a_calibrated": False,
                    "malignancy_calibrated": False
                }
            }
        }
        if self.schema:
            is_valid, err = validate_schema(card, self.schema)
            if not is_valid:
                raise ValueError(f"Abstention card failed schema validation: {err}")
        return card

    def predict(
        self,
        clinical_covariates: Optional[Dict[str, Any]] = None,
        pair_identifier: str = "sample_pair"
    ) -> Dict[str, Any]:
        """
        Executes tabular Clin Agent inference on allowlisted features.

        Args:
            clinical_covariates: Dict containing 'breast_density' and 'breast_side'.
            pair_identifier: De-identified pair identifier string.

        Returns:
            Schema-valid finding card dictionary (status: success or abstained).
        """
        # 1. Validate and extract allowlisted features
        density_val, side_val, rejection_reason = self.validate_and_extract_features(clinical_covariates)
        if rejection_reason is not None:
            return self.build_abstention_card(rejection_reason, clinical_covariates, pair_identifier)

        # 2. Check model initialization
        if not self.is_initialized or len(self.models) == 0:
            return self.build_abstention_card(
                f"model_not_initialized_{self.init_error or 'unknown'}",
                clinical_covariates,
                pair_identifier
            )

        # 3. Construct input matrix: col 0 = density, col 1 = side (0.0 or 1.0)
        X = np.array([[density_val, side_val]], dtype=np.float32)
        dmat = xgb.DMatrix(X, feature_names=["breast_density", "breast_side"])

        # 4. Ensemble inference across all 5 folds
        uncal_probs = []
        cal_probs = []
        shap_contribs_list = []

        for k, bst in enumerate(self.models):
            # Raw logit margin
            raw_margin = float(bst.predict(dmat, output_margin=True)[0])
            p_uncal = 1.0 / (1.0 + np.exp(-raw_margin))
            uncal_probs.append(p_uncal)

            # Temperature calibrated probability
            temp = self.temperatures[k]
            p_cal = 1.0 / (1.0 + np.exp(-raw_margin / temp))
            cal_probs.append(p_cal)

            # Native TreeSHAP attributions: [contrib_density, contrib_side, bias]
            contribs = bst.predict(dmat, pred_contribs=True)[0]
            shap_contribs_list.append(contribs)

        mean_uncal = float(np.mean(uncal_probs))
        mean_cal = float(np.mean(cal_probs))
        mean_shap = np.mean(shap_contribs_list, axis=0)

        # TreeSHAP feature attributions dictionary
        tree_shap_dict = {
            "features": ["breast_density", "breast_side"],
            "contributions": [float(round(mean_shap[0], 6)), float(round(mean_shap[1], 6))],
            "bias": float(round(mean_shap[2], 6))
        }

        side_str = "LEFT" if side_val == 0.0 else "RIGHT"
        rationale = (
            f"Tabular clinical model ensemble estimated calibrated malignancy risk of {mean_cal:.4f} "
            f"(uncalibrated: {mean_uncal:.4f}) based on breast density {int(density_val)} and side {side_str}. "
            "Due to negative-result standalone discrimination (ROC-AUC ~ 0.49), agent is designated "
            "non-authoritative research with zero consensus voting weight."
        )

        card = {
            "agent": "clin",
            "finding": {
                "lesion": None,
                "margin": None,
                "calcification": None,
                "risk_profile": "malignancy_risk",
                "breast_density": int(density_val),
                "breast_side": side_str
            },
            "BI_RADS_a": "unspecified",
            "conf_a": 0.0,
            "sal": {
                "top_regions": [],
                "path": None,
                "cc_path": None,
                "mlo_path": None,
                "tree_shap": tree_shap_dict
            },
            "sh": {
                "breast_density": int(density_val),
                "breast_side": side_str
            },
            "rationale": rationale,
            "metadata": {
                "model_architecture": "xgboost_tabular_logistic_ensemble",
                "model_version": "clin_agent_v1.0",
                "pair_identifier": pair_identifier,
                "status": "success",
                "authority": "non_authoritative_research",
                "voting_eligibility": False,
                "is_clinically_validated": False,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "disclaimer": "Research prototype output only. Non-authoritative shadow mode.",
                "task_probabilities": {
                    "malignancy_calibrated": float(round(mean_cal, 6)),
                    "malignancy_uncalibrated": float(round(mean_uncal, 6))
                },
                "confidence_provenance": {
                    "conf_a_source": "non_authoritative_zero_weight_convention",
                    "is_conf_a_calibrated": False,
                    "malignancy_calibrated": True
                }
            }
        }

        # Validate against schema
        if self.schema:
            is_valid, err = validate_schema(card, self.schema)
            if not is_valid:
                raise ValueError(f"Clin finding card failed schema validation: {err}")

        return card
