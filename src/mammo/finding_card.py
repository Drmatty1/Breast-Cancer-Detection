"""
Finding Card Serialization and Schema Validation Module for Mammography Agent.

Governed by: docs/breast_cancer___ps1___methodology.pdf (§3.4, p. 4).
Serializes multi-task model predictions and saliency maps into schema-compliant finding cards.
Enforces schema validation via jsonschema.
"""
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import time
import jsonschema


DEFAULT_SCHEMA_PATH = Path(r"D:\rp\schemas\finding_card.schema.json")

# Authoritative class name lookups
MARGIN_CLASS_NAMES = {
    0: "circumscribed",
    1: "indistinct",
    2: "obscured",
    3: "spiculated"
}

CALC_CLASS_NAMES = {
    0: "amorphous",
    1: "coarse",
    2: "fine_linear_branching",
    3: "pleomorphic",
    4: "punctate"
}

BIRADS_LEVEL_NAMES = {
    0: "2",
    1: "3",
    2: "4",
    3: "5"
}


def build_finding_card(
    binary_prob_calibrated: float,
    margin_probs: Optional[List[float]] = None,
    calc_probs: Optional[List[float]] = None,
    birads_ordinal_level: Optional[int] = None,
    birads_confidence: Optional[float] = None,
    saliency_regions: Optional[List[List[float]]] = None,
    saliency_path: Optional[str] = None,
    cc_saliency_path: Optional[str] = None,
    mlo_saliency_path: Optional[str] = None,
    clinical_header: Optional[Dict[str, Any]] = None,
    model_version: str = "mammo_agent_v1.0",
    checkpoint_info: str = "mammo_agent_ensemble",
    threshold_malignancy: float = 0.367477005294,
    abnormality_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Serializes multi-task model outputs into the methodology-compliant finding card structure (§3.4).

    Args:
        binary_prob_calibrated: Calibrated malignancy probability in [0, 1].
        margin_probs: List of 4 class probabilities for mass margins (or None if unavailable).
        calc_probs: List of 5 class probabilities for calcification morphology (or None if unavailable).
        birads_ordinal_level: Predicted ordinal diagnostic level (0: Cat 2, 1: Cat 3, 2: Cat 4, 3: Cat 5).
        birads_confidence: Decision confidence score for the BI-RADS prediction (uncalibrated heuristic).
        saliency_regions: List of top bounding box coordinates [[ymin, xmin, ymax, xmax]].
        saliency_path: File path to combined or primary saliency overlay.
        cc_saliency_path: File path to CC view saliency overlay.
        mlo_saliency_path: File path to MLO view saliency overlay.
        clinical_header: Dict of clinical metadata (empty dict if perception-only).
        model_version: Model identifier string.
        checkpoint_info: Provenance checkpoint identifier.
        threshold_malignancy: Operating threshold for lesion malignancy.
        abnormality_type: Explicit clinical finding indicator ('mass', 'calcification', 'both'/'mixed',
                          or None/'unknown'). If None or unverified, secondary findings are represented as
                          unavailable (null) to prevent hallucinating unconfirmed lesion sub-types.

    Returns:
        Schema-valid Finding Card dictionary.
    """
    # 1. Lesion Finding
    is_malignant = binary_prob_calibrated >= threshold_malignancy
    lesion_status = "malignant" if is_malignant else "benign"

    # Determine supported abnormality types
    target_abnormality = abnormality_type
    if target_abnormality is None and clinical_header:
        target_abnormality = (
            clinical_header.get("abnormality_type")
            or clinical_header.get("abnormality")
            or clinical_header.get("finding_type")
        )

    support_mass = False
    support_calc = False

    if target_abnormality is not None:
        norm = str(target_abnormality).strip().lower()
        if norm in ("mass", "masses"):
            support_mass = True
        elif norm in ("calc", "calcification", "calcifications"):
            support_calc = True
        elif norm in ("both", "mixed", "mass_and_calc", "all"):
            support_mass = True
            support_calc = True
        elif norm in ("none", "unknown", "unspecified", "unavailable"):
            support_mass = False
            support_calc = False
        else:
            support_mass = False
            support_calc = False
    else:
        # If caller provided exactly one secondary task distribution, infer intent for that abnormality
        if margin_probs is not None and calc_probs is None:
            support_mass = True
        elif calc_probs is not None and margin_probs is None:
            support_calc = True
        else:
            # Both heads provided (or neither): without an explicit presence signal,
            # neither secondary finding is confirmed to exist.
            support_mass = False
            support_calc = False

    # 2. Margin Finding
    margin_finding = None
    if support_mass and margin_probs is not None and len(margin_probs) == 4:
        best_margin_idx = int(max(range(len(margin_probs)), key=lambda i: margin_probs[i]))
        margin_finding = MARGIN_CLASS_NAMES.get(best_margin_idx, "unspecified")

    # 3. Calcification Morphology Finding
    calc_finding = None
    if support_calc and calc_probs is not None and len(calc_probs) == 5:
        best_calc_idx = int(max(range(len(calc_probs)), key=lambda i: calc_probs[i]))
        calc_finding = CALC_CLASS_NAMES.get(best_calc_idx, "unspecified")

    # 4. BI-RADS Category
    if birads_ordinal_level is not None:
        birads_category_str = BIRADS_LEVEL_NAMES.get(birads_ordinal_level, str(birads_ordinal_level))
    else:
        birads_category_str = "unspecified"

    # 5. Decision Confidence Score
    # Per methodology §3.4, conf_a represents agent decision confidence score in [0, 1].
    # When birads_confidence is provided, it reflects the heuristic distance of the ensemble-averaged
    # ordinal cutoffs from the 0.5 decision boundary: 2 * max_j |q_j - 0.5|.
    # Note: Unlike the primary binary malignancy head (calibrated via locked temperature scaling),
    # secondary heads (including ordinal BI-RADS) are NOT post-hoc calibrated. Hence conf_a represents
    # an uncalibrated heuristic confidence score.
    if birads_confidence is not None:
        conf_a = float(round(birads_confidence, 4))
    else:
        conf_a = float(round(binary_prob_calibrated, 4))
    conf_a = max(0.0, min(1.0, conf_a))

    # 6. Saliency Artifacts
    top_regions = saliency_regions if saliency_regions is not None else []
    sal_dict = {
        "path": saliency_path,
        "top_regions": top_regions,
        "cc_path": cc_saliency_path,
        "mlo_path": mlo_saliency_path
    }

    # 7. Synthesize Natural Language Rationale
    findings_desc = []
    if margin_finding:
        findings_desc.append(f"{margin_finding} margin mass")
    if calc_finding:
        findings_desc.append(f"{calc_finding} calcifications")

    if findings_desc:
        findings_str = " with ".join(findings_desc)
    else:
        findings_str = f"{lesion_status} mammographic finding"

    rationale = (
        f"Two-view mammography assessment indicates {findings_str}. "
        f"Assigned BI-RADS category {birads_category_str} "
        f"with decision confidence {conf_a:.2f}."
    )

    card = {
        "agent": "mammo",
        "finding": {
            "lesion": lesion_status,
            "margin": margin_finding,
            "calcification": calc_finding
        },
        "BI_RADS_a": birads_category_str,
        "conf_a": conf_a,
        "sal": sal_dict,
        "sh": clinical_header if clinical_header is not None else {},
        "rationale": rationale,
        "metadata": {
            "model_architecture": "tf_efficientnet_lite0_multitask",
            "model_version": model_version,
            "checkpoint_source": checkpoint_info,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "disclaimer": "Research prototype output only. Not for clinical diagnostic use.",
            "task_probabilities": {
                "malignancy_calibrated": float(round(binary_prob_calibrated, 6)),
                "margin_distribution": [float(round(p, 4)) for p in margin_probs] if margin_probs else None,
                "calc_distribution": [float(round(p, 4)) for p in calc_probs] if calc_probs else None
            },
            "abnormality_support": {
                "evaluated_signal": target_abnormality,
                "mass_supported": support_mass,
                "calc_supported": support_calc
            },
            "confidence_provenance": {
                "conf_a_source": "heuristic_ordinal_cutoff_margin",
                "is_conf_a_calibrated": False,
                "malignancy_calibrated": True
            }
        }
    }

    return card


def validate_finding_card(card: Dict[str, Any], schema_path: Optional[Path] = None) -> Tuple[bool, Optional[str]]:
    """
    Validates finding card against the formal JSON schema.
    Returns (is_valid, error_message).
    """
    if schema_path is None:
        schema_path = DEFAULT_SCHEMA_PATH

    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        jsonschema.validate(instance=card, schema=schema)
        return True, None
    except jsonschema.ValidationError as e:
        return False, f"Schema validation error: {e.message} at path: {list(e.path)}"
    except Exception as e:
        return False, f"Unexpected error loading/validating schema: {str(e)}"
