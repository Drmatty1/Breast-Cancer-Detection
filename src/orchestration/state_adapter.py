"""
Supervisor State Adapter Module for Multi-Agent Mammography System.

Governed by:
- docs/breast_cancer___ps1___methodology.pdf (§§ 3.1, 3.4, 5.1, 5.3)
- DECISIONS.md (DEC-OWNER-01, DEC-OWNER-02, DEC-OWNER-03)
- reports/clin/CLIN_AGENT_C3_7_ORCHESTRATION_CONTRACT.md
- reports/rp2_experimental/agent_integration_audit.md

Key Architecture & Safety Guarantees:
1. Imports typed contract definitions from `src.orchestration.supervisor_contract`.
2. Maps dual-channel OrchestrationResponse from local MultiAgentOrchestrator into a
   typed, deterministic, JSON-serializable SupervisorState.
3. Enforces hard governance invariants:
   - Mammography Agent is the sole Authoritative Diagnostic Tier (voting_eligible=True).
   - Clinical Risk Agent runs strictly in an isolated Shadow Research Tier (voting_eligible=False).
     Any attempted voting_eligibility=True by Clin is forcefully rejected and overridden.
   - Pathology Agent is explicitly marked unavailable (status="unavailable", voting_eligible=False)
     with zero fabricated findings or clinical assumptions.
4. Deferral to Human Review:
   - Primary Mammo failure, schema invalidity, non-finite/out-of-range probabilities, or
     mandatory field omissions trigger defer_to_human=True with descriptive reasons.
   - Missing, failed, or abstaining Clin Agent NEVER triggers human deferral if Mammo is valid.
5. Clarification of Required vs Optional Fields in Mammography:
   - REQUIRED for successful authoritative output: BI_RADS_a, conf_a, malignancy_calibrated.
   - OPTIONAL (may legitimately be None / absent): margin_distribution (if non-mass),
     calc_distribution (if non-calc), ordinal_monotonicity (for RP1 baseline cards).
   - Missing optional distributions are NEVER silently treated as negative findings.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timezone
import copy
import math

from src.orchestration.supervisor_contract import (
    SupervisorState,
    AgentFindingPayload,
    ValidationIssue,
    StateValidationError,
    sanitize_for_json,
    VALID_BIRADS_CATEGORIES,
    VALID_AGENT_IDS,
    VALID_STATUS_VALUES,
    VALID_AUTHORITY_LEVELS,
    VALID_SEVERITIES,
    PRIMARY_AGENT_ID,
    SHADOW_AGENT_ID,
    PATH_AGENT_ID,
    AUTHORITY_AUTHORITATIVE,
    AUTHORITY_NON_AUTHORITATIVE
)
from src.orchestration.multi_agent_orchestrator import (
    OrchestrationResponse,
    OrchestrationAuditRecord
)


def _is_valid_prob(val: Any) -> bool:
    """Checks whether a value is a finite number in [0.0, 1.0]."""
    if val is None or isinstance(val, bool):
        return False
    if not isinstance(val, (int, float)):
        # Check numpy types if numpy is present
        try:
            import numpy as np
            if not isinstance(val, (np.integer, np.floating)):
                return False
        except ImportError:
            return False
    try:
        f = float(val)
        return not (math.isnan(f) or math.isinf(f)) and (0.0 <= f <= 1.0)
    except Exception:
        return False


def adapt_mammo_finding_card(
    card: Optional[Dict[str, Any]],
    audit_status: str = "SUCCESS"
) -> Tuple[AgentFindingPayload, List[ValidationIssue]]:
    """
    Validates and adapts a primary Mammography finding card into AgentFindingPayload.

    Required vs Optional Field Rules:
    - REQUIRED for success:
      - card != None, isinstance(card, dict)
      - agent == "mammo"
      - BI_RADS_a in {"2", "3", "4", "5", "unspecified"}
      - conf_a: finite float in [0.0, 1.0]
      - metadata.task_probabilities.malignancy_calibrated: finite float in [0.0, 1.0]
    - OPTIONAL (legitimately absent without error):
      - metadata.task_probabilities.margin_distribution (None if finding is non-mass)
      - metadata.task_probabilities.calc_distribution (None if finding is non-calc)
      - metadata.ordinal_monotonicity (absent in RP1 baseline finding cards)
      - finding.margin / finding.calcification (None if finding type does not apply)
    - If optional distributions are present, their values are strictly checked for validity.
    """
    issues: List[ValidationIssue] = []

    # 1. Missing card or orchestrator failure
    if card is None or not isinstance(card, dict):
        status = "schema_invalid" if audit_status == "SCHEMA_INVALID" else "failed"
        issues.append(ValidationIssue(
            severity="ERROR",
            code="MISSING_PRIMARY_CARD",
            message=f"Authoritative Mammography finding card is missing or invalid (audit status: {audit_status}).",
            field="primary_finding_card"
        ))
        payload = AgentFindingPayload(
            agent_id=PRIMARY_AGENT_ID,
            status=status,
            authority_level=AUTHORITY_AUTHORITATIVE,
            voting_eligible=False,
            bi_rads="unspecified",
            confidence=0.0,
            confidence_type="none",
            p_malignancy=None,
            findings_summary={},
            task_probabilities={},
            explainability={},
            rationale=None,
            raw_finding_card=None,
            metadata={"audit_status": audit_status}
        )
        return payload, issues

    # 2. REQUIRED: Verify agent ID
    card_agent = card.get("agent")
    if card_agent != PRIMARY_AGENT_ID:
        issues.append(ValidationIssue(
            severity="ERROR",
            code="AGENT_ID_MISMATCH",
            message=f"Expected agent '{PRIMARY_AGENT_ID}', got '{card_agent}'.",
            field="agent"
        ))

    # 3. REQUIRED: Verify BI-RADS category
    raw_birads = card.get("BI_RADS_a")
    birads_str = str(raw_birads) if raw_birads is not None else "unspecified"
    if birads_str not in VALID_BIRADS_CATEGORIES:
        issues.append(ValidationIssue(
            severity="ERROR",
            code="INVALID_BIRADS_CATEGORY",
            message=f"Invalid BI-RADS category '{birads_str}'. Expected one of {sorted(VALID_BIRADS_CATEGORIES)}.",
            field="BI_RADS_a"
        ))

    # 4. REQUIRED: Verify decision confidence score conf_a
    raw_conf = card.get("conf_a")
    conf_a = 0.0
    if raw_conf is None or isinstance(raw_conf, bool) or not isinstance(raw_conf, (int, float)):
        issues.append(ValidationIssue(
            severity="ERROR",
            code="INVALID_CONFIDENCE_TYPE",
            message=f"conf_a must be a valid float, got {type(raw_conf).__name__}.",
            field="conf_a"
        ))
    else:
        try:
            f_conf = float(raw_conf)
            if math.isnan(f_conf) or math.isinf(f_conf):
                issues.append(ValidationIssue(
                    severity="ERROR",
                    code="NON_FINITE_CONFIDENCE",
                    message="conf_a contains non-finite value (NaN or Inf).",
                    field="conf_a"
                ))
            elif not (0.0 <= f_conf <= 1.0):
                issues.append(ValidationIssue(
                    severity="ERROR",
                    code="OUT_OF_RANGE_CONFIDENCE",
                    message=f"conf_a value {f_conf} is outside [0.0, 1.0].",
                    field="conf_a"
                ))
            else:
                conf_a = f_conf
        except Exception as e:
            issues.append(ValidationIssue(
                severity="ERROR",
                code="CONFIDENCE_CONVERSION_ERROR",
                message=f"Failed to convert conf_a to float: {str(e)}",
                field="conf_a"
            ))

    # Extract metadata and task probabilities
    metadata = copy.deepcopy(card.get("metadata", {}))
    task_probs = metadata.get("task_probabilities", {})

    # 5. REQUIRED (for successful execution): Calibrated malignancy probability
    raw_p_mal = task_probs.get("malignancy_calibrated")
    p_malignancy = None
    if raw_p_mal is None:
        issues.append(ValidationIssue(
            severity="ERROR",
            code="MISSING_MALIGNANCY_PROBABILITY",
            message="Required probability 'malignancy_calibrated' is missing from metadata.task_probabilities.",
            field="metadata.task_probabilities.malignancy_calibrated"
        ))
    elif not _is_valid_prob(raw_p_mal):
        issues.append(ValidationIssue(
            severity="ERROR",
            code="INVALID_MALIGNANCY_PROBABILITY",
            message=f"Malignancy probability {raw_p_mal} is non-finite or outside [0.0, 1.0].",
            field="metadata.task_probabilities.malignancy_calibrated"
        ))
    else:
        p_malignancy = float(raw_p_mal)

    # 6. OPTIONAL: Mass margin distribution (None if finding is non-mass)
    raw_margin = task_probs.get("margin_distribution")
    if raw_margin is not None:
        if isinstance(raw_margin, list) and len(raw_margin) == 4 and all(_is_valid_prob(p) for p in raw_margin):
            s = sum(float(p) for p in raw_margin)
            if abs(s - 1.0) > 0.05:
                issues.append(ValidationIssue(
                    severity="WARNING",
                    code="MARGIN_DISTRIBUTION_SUM_WARNING",
                    message=f"Margin probabilities sum to {s:.4f}, expected ~1.0.",
                    field="metadata.task_probabilities.margin_distribution"
                ))
        else:
            issues.append(ValidationIssue(
                severity="ERROR",
                code="INVALID_MARGIN_DISTRIBUTION",
                message=f"Margin distribution {raw_margin} is not a valid list of 4 probabilities in [0.0, 1.0].",
                field="metadata.task_probabilities.margin_distribution"
            ))

    # 7. OPTIONAL: Calcification distribution (None if finding is non-calc)
    raw_calc = task_probs.get("calc_distribution")
    if raw_calc is not None:
        if isinstance(raw_calc, list) and len(raw_calc) == 5 and all(_is_valid_prob(p) for p in raw_calc):
            s = sum(float(p) for p in raw_calc)
            if abs(s - 1.0) > 0.05:
                issues.append(ValidationIssue(
                    severity="WARNING",
                    code="CALC_DISTRIBUTION_SUM_WARNING",
                    message=f"Calcification probabilities sum to {s:.4f}, expected ~1.0.",
                    field="metadata.task_probabilities.calc_distribution"
                ))
        else:
            issues.append(ValidationIssue(
                severity="ERROR",
                code="INVALID_CALC_DISTRIBUTION",
                message=f"Calcification distribution {raw_calc} is not a valid list of 5 probabilities in [0.0, 1.0].",
                field="metadata.task_probabilities.calc_distribution"
            ))

    # 8. OPTIONAL: Ordinal monotonicity metadata (validated when present, legitimate if absent)
    ord_mono = metadata.get("ordinal_monotonicity")
    if ord_mono is not None:
        if not isinstance(ord_mono, dict):
            issues.append(ValidationIssue(
                severity="ERROR",
                code="INVALID_ORDINAL_MONOTONICITY_STRUCTURE",
                message="metadata.ordinal_monotonicity must be a dict when present.",
                field="metadata.ordinal_monotonicity"
            ))
        else:
            proj_cutoffs = ord_mono.get("projected_cutoff_probs")
            if proj_cutoffs is not None:
                if not (isinstance(proj_cutoffs, list) and len(proj_cutoffs) == 3):
                    issues.append(ValidationIssue(
                        severity="ERROR",
                        code="INVALID_ORDINAL_CUTOFFS_LENGTH",
                        message=f"Expected 3 projected cutoff probabilities, got {proj_cutoffs}.",
                        field="metadata.ordinal_monotonicity.projected_cutoff_probs"
                    ))
                else:
                    q0, q1, q2 = proj_cutoffs
                    if not (_is_valid_prob(q0) and _is_valid_prob(q1) and _is_valid_prob(q2)):
                        issues.append(ValidationIssue(
                            severity="ERROR",
                            code="INVALID_ORDINAL_CUTOFF_PROBABILITY",
                            message=f"Ordinal cutoff probabilities contain invalid or non-finite values: {proj_cutoffs}.",
                            field="metadata.ordinal_monotonicity.projected_cutoff_probs"
                        ))
                    elif not (q0 >= q1 - 1e-5 and q1 >= q2 - 1e-5):
                        issues.append(ValidationIssue(
                            severity="ERROR",
                            code="MONOTONICITY_VIOLATION_UNRESOLVED",
                            message=f"Projected cutoffs violate monotonicity ordering: q0={q0}, q1={q1}, q2={q2}.",
                            field="metadata.ordinal_monotonicity.projected_cutoff_probs"
                        ))

    # 9. Confidence calibration status determination
    conf_prov = metadata.get("confidence_provenance", {})
    if conf_prov.get("is_conf_a_calibrated", False):
        conf_type = "calibrated_heuristic"
    elif conf_prov.get("malignancy_calibrated", False) and conf_prov.get("conf_a_source") == "calibrated_binary_malignancy":
        conf_type = "calibrated_binary"
    else:
        conf_type = "uncalibrated_heuristic"

    # Saliency / Explainability
    sal = card.get("sal", {})
    explainability = {
        "top_regions": copy.deepcopy(sal.get("top_regions", [])),
        "saliency_path": sal.get("path"),
        "cc_path": sal.get("cc_path"),
        "mlo_path": sal.get("mlo_path")
    }

    # Findings summary
    findings_summary = copy.deepcopy(card.get("finding", {}))

    # Status & voting eligibility
    has_errors = any(issue.severity == "ERROR" for issue in issues)
    if audit_status != "SUCCESS":
        status = "schema_invalid" if audit_status == "SCHEMA_INVALID" else "failed"
        voting_eligible = False
    elif has_errors:
        status = "failed"
        voting_eligible = False
    else:
        status = "success"
        voting_eligible = True

    payload = AgentFindingPayload(
        agent_id=PRIMARY_AGENT_ID,
        status=status,
        authority_level=AUTHORITY_AUTHORITATIVE,
        voting_eligible=voting_eligible,
        bi_rads=birads_str,
        confidence=conf_a,
        confidence_type=conf_type,
        p_malignancy=p_malignancy,
        findings_summary=findings_summary,
        task_probabilities=copy.deepcopy(task_probs),
        explainability=explainability,
        rationale=card.get("rationale"),
        raw_finding_card=copy.deepcopy(card),
        metadata=metadata
    )
    return payload, issues


def adapt_clin_finding_card(
    card: Optional[Dict[str, Any]],
    audit_status: str = "SUCCESS",
    error_code: Optional[str] = None
) -> Tuple[AgentFindingPayload, List[ValidationIssue]]:
    """
    Validates and adapts a shadow Clinical Risk finding card into AgentFindingPayload.
    Enforces non-authoritative tier governance (voting_eligible=False strictly enforced).
    Never triggers human review on missing, failed, or abstaining covariates.
    """
    issues: List[ValidationIssue] = []

    status_map = {
        "SUCCESS": "success",
        "ABSTAINED": "abstained",
        "FAILED": "failed",
        "TIMEOUT": "timeout",
        "SCHEMA_INVALID": "schema_invalid",
        "SKIPPED": "skipped"
    }
    state_status = status_map.get(audit_status.upper(), "failed")

    # If shadow card is missing or None
    if card is None or not isinstance(card, dict):
        if state_status == "success":
            state_status = "failed"
        payload = AgentFindingPayload(
            agent_id=SHADOW_AGENT_ID,
            status=state_status,
            authority_level=AUTHORITY_NON_AUTHORITATIVE,
            voting_eligible=False,
            bi_rads="unspecified",
            confidence=0.0,
            confidence_type="non_authoritative_zero_weight",
            p_malignancy=None,
            findings_summary={},
            task_probabilities={},
            explainability={},
            rationale=None,
            raw_finding_card=None,
            metadata={"audit_status": audit_status, "clin_error_code": error_code}
        )
        return payload, issues

    # If card is present, check agent ID
    card_agent = card.get("agent")
    if card_agent != SHADOW_AGENT_ID:
        issues.append(ValidationIssue(
            severity="WARNING",
            code="AGENT_ID_MISMATCH",
            message=f"Expected agent '{SHADOW_AGENT_ID}', got '{card_agent}'.",
            field="agent"
        ))

    # HARD INVARIANT: Clin Agent must never have voting eligibility
    metadata = copy.deepcopy(card.get("metadata", {}))
    card_voting_eligibility = metadata.get("voting_eligibility", False)
    if card_voting_eligibility is True:
        issues.append(ValidationIssue(
            severity="ERROR",
            code="UNAUTHORIZED_VOTING_ELIGIBILITY",
            message="Clin Agent attempted to claim voting_eligibility=True. Strictly overridden to False.",
            field="metadata.voting_eligibility"
        ))
    voting_eligible = False  # Strictly False under all circumstances

    # Extract task probabilities
    task_probs = metadata.get("task_probabilities", {})
    raw_p_mal = task_probs.get("malignancy_calibrated")
    p_malignancy = None
    if raw_p_mal is not None:
        if _is_valid_prob(raw_p_mal):
            p_malignancy = float(raw_p_mal)
        else:
            issues.append(ValidationIssue(
                severity="WARNING",
                code="INVALID_CLIN_PROBABILITY",
                message=f"Clin malignancy probability {raw_p_mal} is non-finite or outside [0.0, 1.0].",
                field="metadata.task_probabilities.malignancy_calibrated"
            ))

    # TreeSHAP explainability
    sal = card.get("sal", {})
    tree_shap = sal.get("tree_shap", {})
    explainability = {
        "tree_shap": copy.deepcopy(tree_shap)
    }

    # Findings summary
    finding = card.get("finding", {})
    sh = card.get("sh", {})
    findings_summary = {
        "risk_profile": finding.get("risk_profile"),
        "breast_density": sh.get("breast_density") or finding.get("breast_density"),
        "breast_side": sh.get("breast_side") or finding.get("breast_side"),
        "rejection_reason": metadata.get("rejection_reason")
    }

    if metadata.get("status") in ("abstained", "abstention") or findings_summary.get("rejection_reason") is not None:
        state_status = "abstained"

    payload = AgentFindingPayload(
        agent_id=SHADOW_AGENT_ID,
        status=state_status,
        authority_level=AUTHORITY_NON_AUTHORITATIVE,
        voting_eligible=voting_eligible,
        bi_rads="unspecified",
        confidence=0.0,
        confidence_type="non_authoritative_zero_weight",
        p_malignancy=p_malignancy,
        findings_summary=findings_summary,
        task_probabilities=copy.deepcopy(task_probs),
        explainability=explainability,
        rationale=card.get("rationale"),
        raw_finding_card=copy.deepcopy(card),
        metadata=metadata
    )
    return payload, issues


def create_unavailable_pathology_payload() -> AgentFindingPayload:
    """
    Creates an explicit unavailable payload for the deferred Pathology Agent.
    Strictly avoids fabricating any clinical covariates, biopsy results, or predictions.
    """
    return AgentFindingPayload(
        agent_id=PATH_AGENT_ID,
        status="unavailable",
        authority_level=AUTHORITY_NON_AUTHORITATIVE,
        voting_eligible=False,
        bi_rads="unspecified",
        confidence=0.0,
        confidence_type="unavailable",
        p_malignancy=None,
        findings_summary={
            "status": "unavailable",
            "reason": "Histology WSI not available in CBIS-DDSM perception pipeline"
        },
        task_probabilities={},
        explainability={},
        rationale="Pathology Agent is deferred in RP2. No biopsy or histology WSI inputs are available. Zero diagnostic weight assigned.",
        raw_finding_card=None,
        metadata={
            "deferred": True,
            "reason": "histology_data_unavailable",
            "model_version": None
        }
    )


def adapt_orchestration_response(
    response: OrchestrationResponse,
    study_id: Optional[str] = None
) -> SupervisorState:
    """
    Main adapter entry point: transforms an OrchestrationResponse into a SupervisorState.

    Args:
        response: OrchestrationResponse returned by MultiAgentOrchestrator.
        study_id: Optional custom study identifier; defaults to audit record's pair_identifier.

    Returns:
        Structured, validated, JSON-serializable SupervisorState.
    """
    audit = response.audit_record
    effective_study_id = study_id or audit.pair_identifier or "unknown_study"

    # UTC timestamp resolution
    ts_utc = audit.timestamp_utc
    if not ts_utc:
        ts_utc = datetime.now(timezone.utc).isoformat()

    all_issues: List[ValidationIssue] = []
    deferral_reasons: List[str] = []

    # 1. Primary Mammo Agent adaptation
    mammo_payload, mammo_issues = adapt_mammo_finding_card(
        response.primary_finding_card,
        audit.mammo_status
    )
    all_issues.extend(mammo_issues)

    # 2. Shadow Clin Agent adaptation
    clin_payload, clin_issues = adapt_clin_finding_card(
        response.shadow_finding_card,
        audit.clin_status,
        audit.clin_error_code
    )
    all_issues.extend(clin_issues)

    # 3. Pathology Agent payload (unavailable)
    path_payload = create_unavailable_pathology_payload()

    # 4. Determine Deferral to Human Review
    # Invariant: Human deferral is governed by the authoritative channel.
    # Missing, abstaining, or failed shadow agent NEVER triggers human deferral.
    defer_to_human = False

    if mammo_payload.status != "success":
        defer_to_human = True
        deferral_reasons.append(
            f"Authoritative Mammography Agent failed or emitted invalid schema (status: {mammo_payload.status})."
        )

    for issue in mammo_issues:
        if issue.severity == "ERROR":
            defer_to_human = True
            deferral_reasons.append(f"Authoritative channel validation error [{issue.code}]: {issue.message}")

    if not defer_to_human and not mammo_payload.voting_eligible:
        defer_to_human = True
        deferral_reasons.append("Authoritative Mammography Agent is not eligible for diagnostic voting.")

    agents_map = {
        PRIMARY_AGENT_ID: mammo_payload,
        SHADOW_AGENT_ID: clin_payload,
        PATH_AGENT_ID: path_payload
    }

    state = SupervisorState(
        study_id=effective_study_id,
        timestamp_utc=ts_utc,
        agents=agents_map,
        primary_agent_id=PRIMARY_AGENT_ID,
        defer_to_human=defer_to_human,
        deferral_reasons=deferral_reasons,
        validation_issues=all_issues,
        orchestration_audit=audit.to_dict()
    )

    return state


__all__ = [
    "SupervisorState",
    "AgentFindingPayload",
    "ValidationIssue",
    "StateValidationError",
    "adapt_orchestration_response",
    "adapt_mammo_finding_card",
    "adapt_clin_finding_card",
    "create_unavailable_pathology_payload",
    "sanitize_for_json",
    "VALID_BIRADS_CATEGORIES",
    "VALID_AGENT_IDS",
    "VALID_STATUS_VALUES",
    "VALID_AUTHORITY_LEVELS",
    "VALID_SEVERITIES",
    "PRIMARY_AGENT_ID",
    "SHADOW_AGENT_ID",
    "PATH_AGENT_ID",
    "AUTHORITY_AUTHORITATIVE",
    "AUTHORITY_NON_AUTHORITATIVE"
]
