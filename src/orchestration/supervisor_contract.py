"""
Supervisor State Contract and Serialization Module.

Governed by:
- docs/breast_cancer___ps1___methodology.pdf (§§ 3.1, 3.4, 5.1, 5.3)
- DECISIONS.md (DEC-OWNER-01, DEC-OWNER-02, DEC-OWNER-03)
- reports/clin/CLIN_AGENT_C3_7_ORCHESTRATION_CONTRACT.md
- reports/rp2_experimental/agent_integration_audit.md

Key Architecture & Boundary Rules:
1. Pure, lightweight contract module for supervisor state definitions, validation,
   and RFC 8259-compliant JSON serialization/deserialization.
2. ZERO local model runtime dependencies: does NOT require PyTorch, Torchvision,
   timm, XGBoost, or jsonschema. Can be imported in any standard Python 3.10+
   environment (such as a remote LLM / LangGraph supervisor node).
3. Hardened deserialization & semantic validation:
   - Replaces permissive type coercions: rejects string-to-bool coercions (e.g. "false" is rejected).
   - Rejects non-string types in ValidationIssue fields and deferral reasons.
   - Enforces mandatory, complete predictions on successful authoritative Mammo agent:
     p_malignancy in [0.0, 1.0] and task_probabilities.malignancy_calibrated are strictly required.
     Missing mandatory predictions are never treated as negative findings or valid assessments.
   - Strictly validates nested task_probabilities distributions:
     scalar bounds, distribution lengths (margin=4, calc=5, birads=4, cutoffs=3), and finite values.
   - Preserves legitimate None values for optional lesion-specific distributions.
   - Enforces clinical governance invariants: Clin and Path agents CANNOT vote.
   - Enforces consistency: primary agent failures or critical ERROR issues strictly require defer_to_human=True.
4. Transport Serialization vs Semantic Validation:
   - sanitize_for_json() is strictly a transport-level serialization safeguard (RFC 8259 compliance).
   - It converts non-finite floats (NaN, Inf) to strings ("nan", "inf") to prevent serializer crashes.
   - Sanitizing data for JSON transport does NOT validate or approve semantic correctness.
   - Deserialization strictly rejects "nan", "inf", and non-finite floats in all prediction fields.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import copy
import json
import math

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None
    _HAS_NUMPY = False

VALID_BIRADS_CATEGORIES = {"2", "3", "4", "5", "unspecified"}
VALID_AGENT_IDS = {"mammo", "clin", "path"}
VALID_STATUS_VALUES = {
    "success",
    "abstained",
    "failed",
    "timeout",
    "schema_invalid",
    "skipped",
    "unavailable"
}
VALID_AUTHORITY_LEVELS = {"authoritative", "non_authoritative_research"}
VALID_SEVERITIES = {"ERROR", "WARNING", "INFO"}

PRIMARY_AGENT_ID = "mammo"
SHADOW_AGENT_ID = "clin"
PATH_AGENT_ID = "path"

AUTHORITY_AUTHORITATIVE = "authoritative"
AUTHORITY_NON_AUTHORITATIVE = "non_authoritative_research"


class StateValidationError(ValueError):
    """
    Raised when supervisor state or finding payload violates schema,
    contains invalid types, or breaches governance invariants during deserialization.
    """
    pass


@dataclass(frozen=True)
class ValidationIssue:
    """
    Structured issue identified during state validation and adaptation.
    """
    severity: str  # "ERROR", "WARNING", "INFO"
    code: str
    message: str
    field: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.severity, str) or self.severity not in VALID_SEVERITIES:
            raise StateValidationError(
                f"Invalid ValidationIssue severity '{self.severity}'. Expected one of {sorted(VALID_SEVERITIES)}."
            )
        if not isinstance(self.code, str) or not self.code.strip():
            raise StateValidationError("ValidationIssue code must be a non-empty string.")
        if not isinstance(self.message, str):
            raise StateValidationError("ValidationIssue message must be a string.")
        if self.field is not None and not isinstance(self.field, str):
            raise StateValidationError("ValidationIssue field must be a string or None.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "field": self.field
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ValidationIssue":
        if not isinstance(data, dict):
            raise StateValidationError(f"Expected dict for ValidationIssue, got {type(data).__name__}.")
        for req in ("severity", "code", "message"):
            if req not in data:
                raise StateValidationError(f"Missing required field '{req}' in ValidationIssue.")

        severity = data["severity"]
        if not isinstance(severity, str):
            raise StateValidationError(f"ValidationIssue severity must be a str, got {type(severity).__name__}: {repr(severity)}.")

        code = data["code"]
        if not isinstance(code, str):
            raise StateValidationError(f"ValidationIssue code must be a str, got {type(code).__name__}: {repr(code)}.")

        message = data["message"]
        if not isinstance(message, str):
            raise StateValidationError(f"ValidationIssue message must be a str, got {type(message).__name__}: {repr(message)}.")

        field_val = data.get("field")
        if field_val is not None and not isinstance(field_val, str):
            raise StateValidationError(f"ValidationIssue field must be a str or None, got {type(field_val).__name__}: {repr(field_val)}.")

        return cls(
            severity=severity,
            code=code,
            message=message,
            field=field_val
        )


def _validate_finite_prob_scalar(val: Any, name: str, agent_id: str) -> float:
    """Validates that a scalar value is a finite number in [0.0, 1.0]."""
    if val is None or isinstance(val, bool) or not isinstance(val, (int, float)):
        raise StateValidationError(
            f"{name} for agent '{agent_id}' must be a finite float in [0.0, 1.0], got {type(val).__name__}: {repr(val)}."
        )
    try:
        f = float(val)
    except Exception:
        raise StateValidationError(f"Failed to convert {name} to float for agent '{agent_id}': {repr(val)}.")
    if math.isnan(f) or math.isinf(f):
        raise StateValidationError(f"{name} for agent '{agent_id}' must be finite, got {f}.")
    if not (0.0 <= f <= 1.0):
        raise StateValidationError(f"{name} for agent '{agent_id}' must be in [0.0, 1.0], got {f}.")
    return f


def _validate_prob_distribution(dist: Any, expected_len: int, name: str, agent_id: str) -> List[float]:
    """Validates that a probability distribution is a list/tuple of finite numbers in [0.0, 1.0]."""
    if not isinstance(dist, (list, tuple)):
        raise StateValidationError(
            f"{name} for agent '{agent_id}' must be a list/tuple of {expected_len} probabilities or None, got {type(dist).__name__}: {repr(dist)}."
        )
    if len(dist) != expected_len:
        raise StateValidationError(
            f"{name} for agent '{agent_id}' must have length {expected_len}, got length {len(dist)}: {repr(dist)}."
        )
    validated = []
    for idx, p in enumerate(dist):
        f = _validate_finite_prob_scalar(p, f"{name}[{idx}]", agent_id)
        validated.append(f)
    return validated


@dataclass
class AgentFindingPayload:
    """
    Typed payload representing an individual agent's findings within SupervisorState.
    """
    agent_id: str
    status: str
    authority_level: str
    voting_eligible: bool
    bi_rads: str
    confidence: float
    confidence_type: str
    p_malignancy: Optional[float]
    findings_summary: Dict[str, Any]
    task_probabilities: Dict[str, Any]
    explainability: Dict[str, Any]
    rationale: Optional[str]
    raw_finding_card: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "status": self.status,
            "authority_level": self.authority_level,
            "voting_eligible": self.voting_eligible,
            "bi_rads": self.bi_rads,
            "confidence": self.confidence,
            "confidence_type": self.confidence_type,
            "p_malignancy": self.p_malignancy,
            "findings_summary": copy.deepcopy(self.findings_summary),
            "task_probabilities": copy.deepcopy(self.task_probabilities),
            "explainability": copy.deepcopy(self.explainability),
            "rationale": self.rationale,
            "raw_finding_card": copy.deepcopy(self.raw_finding_card),
            "metadata": copy.deepcopy(self.metadata)
        }

    @classmethod
    def from_dict(cls, data: Any) -> "AgentFindingPayload":
        """
        Hardened deserialization for AgentFindingPayload.
        Validates required fields, strict types (no permissive string coercions),
        permitted enum values, numeric bounds, governance barriers, and nested distributions.
        """
        if not isinstance(data, dict):
            raise StateValidationError(f"Expected dict for AgentFindingPayload, got {type(data).__name__}.")

        required_fields = (
            "agent_id",
            "status",
            "authority_level",
            "voting_eligible",
            "bi_rads",
            "confidence",
            "confidence_type"
        )
        for req in required_fields:
            if req not in data:
                raise StateValidationError(f"Missing required field '{req}' in AgentFindingPayload.")

        # 1. Validate agent_id
        agent_id = data["agent_id"]
        if not isinstance(agent_id, str) or agent_id not in VALID_AGENT_IDS:
            raise StateValidationError(
                f"Invalid agent_id '{agent_id}'. Expected one of {sorted(VALID_AGENT_IDS)}."
            )

        # 2. Validate status
        status = data["status"]
        if not isinstance(status, str) or status not in VALID_STATUS_VALUES:
            raise StateValidationError(
                f"Invalid status '{status}' for agent '{agent_id}'. Expected one of {sorted(VALID_STATUS_VALUES)}."
            )

        # 3. Validate authority_level
        authority_level = data["authority_level"]
        if not isinstance(authority_level, str) or authority_level not in VALID_AUTHORITY_LEVELS:
            raise StateValidationError(
                f"Invalid authority_level '{authority_level}' for agent '{agent_id}'. Expected one of {sorted(VALID_AUTHORITY_LEVELS)}."
            )

        # 4. Strict boolean validation for voting_eligible (REJECT string "false" or "true")
        voting_eligible = data["voting_eligible"]
        if not isinstance(voting_eligible, bool):
            raise StateValidationError(
                f"voting_eligible must be a strict boolean for agent '{agent_id}', got {type(voting_eligible).__name__}: {repr(voting_eligible)}."
            )

        # 5. HARD CLINICAL GOVERNANCE INVARIANTS:
        # Non-authoritative agents (Clin, Path) MUST NOT be voting_eligible and MUST be non_authoritative_research
        if agent_id in (SHADOW_AGENT_ID, PATH_AGENT_ID):
            if voting_eligible is not False:
                raise StateValidationError(
                    f"Governance breach: Agent '{agent_id}' cannot be voting_eligible. Strict voting_eligible=False required."
                )
            if authority_level != AUTHORITY_NON_AUTHORITATIVE:
                raise StateValidationError(
                    f"Governance breach: Agent '{agent_id}' must have authority_level='{AUTHORITY_NON_AUTHORITATIVE}', got '{authority_level}'."
                )

        # Authoritative agent (Mammo) invariant: non-success status cannot vote
        if agent_id == PRIMARY_AGENT_ID:
            if authority_level != AUTHORITY_AUTHORITATIVE:
                raise StateValidationError(
                    f"Governance breach: Primary agent '{agent_id}' must have authority_level='{AUTHORITY_AUTHORITATIVE}', got '{authority_level}'."
                )
            if status != "success" and voting_eligible is not False:
                raise StateValidationError(
                    f"Invalid state: Primary agent '{agent_id}' with status '{status}' cannot be voting_eligible."
                )

        # 6. Validate bi_rads
        bi_rads = data["bi_rads"]
        if not isinstance(bi_rads, str) or bi_rads not in VALID_BIRADS_CATEGORIES:
            raise StateValidationError(
                f"Invalid bi_rads '{bi_rads}' for agent '{agent_id}'. Expected one of {sorted(VALID_BIRADS_CATEGORIES)}."
            )

        # 7. Validate confidence (finite float in [0.0, 1.0])
        raw_conf = data["confidence"]
        f_conf = _validate_finite_prob_scalar(raw_conf, "confidence", agent_id)

        # 8. Validate confidence_type
        conf_type = data["confidence_type"]
        if not isinstance(conf_type, str) or not conf_type.strip():
            raise StateValidationError(f"confidence_type for agent '{agent_id}' must be a non-empty string.")

        # 9. Validate p_malignancy (optional float in [0.0, 1.0] or None)
        p_mal = data.get("p_malignancy")
        if p_mal is not None:
            p_malignancy = _validate_finite_prob_scalar(p_mal, "p_malignancy", agent_id)
        else:
            p_malignancy = None

        # MANDATORY CHECK FOR AUTHORITATIVE SUCCESS:
        # A successful, voting-eligible Mammography assessment MUST provide a valid p_malignancy.
        # A missing prediction cannot be interpreted as a negative finding or as a valid assessment.
        if agent_id == PRIMARY_AGENT_ID and status == "success" and voting_eligible:
            if p_malignancy is None:
                raise StateValidationError(
                    f"Primary authoritative agent '{agent_id}' with status='success' and voting_eligible=True "
                    "requires valid 'p_malignancy' in [0.0, 1.0]. Missing prediction cannot be inferred as negative."
                )

        # 10. Validate nested containers
        findings_summary = data.get("findings_summary")
        if findings_summary is not None and not isinstance(findings_summary, dict):
            raise StateValidationError(f"findings_summary for agent '{agent_id}' must be a dict.")
        findings_summary = dict(findings_summary) if findings_summary else {}

        task_probabilities = data.get("task_probabilities")
        if task_probabilities is not None and not isinstance(task_probabilities, dict):
            raise StateValidationError(f"task_probabilities for agent '{agent_id}' must be a dict.")
        task_probabilities = dict(task_probabilities) if task_probabilities else {}

        # 11. Validate recognized task_probabilities fields
        # A. malignancy_calibrated (scalar probability)
        if "malignancy_calibrated" in task_probabilities:
            m_cal = task_probabilities["malignancy_calibrated"]
            if m_cal is not None:
                f_mcal = _validate_finite_prob_scalar(m_cal, "task_probabilities.malignancy_calibrated", agent_id)
                # Consistency check with top-level p_malignancy
                if p_malignancy is not None and abs(p_malignancy - f_mcal) > 1e-4:
                    raise StateValidationError(
                        f"Inconsistent malignancy probability for agent '{agent_id}': "
                        f"p_malignancy ({p_malignancy}) != task_probabilities.malignancy_calibrated ({f_mcal})."
                    )
            elif agent_id == PRIMARY_AGENT_ID and status == "success" and voting_eligible:
                raise StateValidationError(
                    f"Primary authoritative agent '{agent_id}' with status='success' and voting_eligible=True "
                    "requires non-None 'malignancy_calibrated' in task_probabilities."
                )
        elif agent_id == PRIMARY_AGENT_ID and status == "success" and voting_eligible:
            raise StateValidationError(
                f"Primary authoritative agent '{agent_id}' with status='success' and voting_eligible=True "
                "requires 'malignancy_calibrated' in task_probabilities."
            )

        # B. malignancy_uncalibrated (scalar probability, Clin)
        if "malignancy_uncalibrated" in task_probabilities:
            m_uncal = task_probabilities["malignancy_uncalibrated"]
            if m_uncal is not None:
                _validate_finite_prob_scalar(m_uncal, "task_probabilities.malignancy_uncalibrated", agent_id)

        # C. margin_distribution (optional 4-class distribution for mass margins)
        if "margin_distribution" in task_probabilities:
            m_dist = task_probabilities["margin_distribution"]
            if m_dist is not None:
                _validate_prob_distribution(m_dist, 4, "task_probabilities.margin_distribution", agent_id)

        # D. calc_distribution (optional 5-class distribution for calcification morphology)
        if "calc_distribution" in task_probabilities:
            c_dist = task_probabilities["calc_distribution"]
            if c_dist is not None:
                _validate_prob_distribution(c_dist, 5, "task_probabilities.calc_distribution", agent_id)

        # E. birads_category_distribution (optional 4-class distribution for ordinal categories 2, 3, 4, 5)
        if "birads_category_distribution" in task_probabilities:
            b_dist = task_probabilities["birads_category_distribution"]
            if b_dist is not None:
                _validate_prob_distribution(b_dist, 4, "task_probabilities.birads_category_distribution", agent_id)

        # F. cutoff_probabilities (optional 3-element list of cutoff probs)
        if "cutoff_probabilities" in task_probabilities:
            cp = task_probabilities["cutoff_probabilities"]
            if cp is not None:
                _validate_prob_distribution(cp, 3, "task_probabilities.cutoff_probabilities", agent_id)

        explainability = data.get("explainability")
        if explainability is not None and not isinstance(explainability, dict):
            raise StateValidationError(f"explainability for agent '{agent_id}' must be a dict.")
        explainability = dict(explainability) if explainability else {}

        metadata = data.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise StateValidationError(f"metadata for agent '{agent_id}' must be a dict.")
        metadata = dict(metadata) if metadata else {}

        # Validate ordinal_monotonicity when present in metadata
        if "ordinal_monotonicity" in metadata:
            ord_mono = metadata["ordinal_monotonicity"]
            if ord_mono is not None:
                if not isinstance(ord_mono, dict):
                    raise StateValidationError(f"metadata.ordinal_monotonicity for agent '{agent_id}' must be a dict.")
                proj_cutoffs = ord_mono.get("projected_cutoff_probs")
                if proj_cutoffs is not None:
                    _validate_prob_distribution(
                        proj_cutoffs, 3, "metadata.ordinal_monotonicity.projected_cutoff_probs", agent_id
                    )
                    q0, q1, q2 = [float(x) for x in proj_cutoffs]
                    if not (q0 >= q1 - 1e-5 and q1 >= q2 - 1e-5):
                        raise StateValidationError(
                            f"metadata.ordinal_monotonicity.projected_cutoff_probs violates ordering: q0={q0}, q1={q1}, q2={q2}."
                        )

        raw_card = data.get("raw_finding_card")
        if raw_card is not None and not isinstance(raw_card, dict):
            raise StateValidationError(f"raw_finding_card for agent '{agent_id}' must be a dict or None.")

        rationale = data.get("rationale")
        if rationale is not None and not isinstance(rationale, str):
            raise StateValidationError(f"rationale for agent '{agent_id}' must be a string or None.")

        return cls(
            agent_id=agent_id,
            status=status,
            authority_level=authority_level,
            voting_eligible=voting_eligible,
            bi_rads=bi_rads,
            confidence=f_conf,
            confidence_type=conf_type,
            p_malignancy=p_malignancy,
            findings_summary=findings_summary,
            task_probabilities=task_probabilities,
            explainability=explainability,
            rationale=rationale,
            raw_finding_card=copy.deepcopy(raw_card),
            metadata=metadata
        )


@dataclass
class SupervisorState:
    """
    Deterministic, typed state container passed to the LangGraph Supervisor.
    """
    study_id: str
    timestamp_utc: str
    agents: Dict[str, AgentFindingPayload]
    primary_agent_id: str = PRIMARY_AGENT_ID
    defer_to_human: bool = False
    deferral_reasons: List[str] = field(default_factory=list)
    validation_issues: List[ValidationIssue] = field(default_factory=list)
    orchestration_audit: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts SupervisorState into a dictionary with sanitized JSON-safe values."""
        raw = {
            "study_id": self.study_id,
            "timestamp_utc": self.timestamp_utc,
            "primary_agent_id": self.primary_agent_id,
            "defer_to_human": self.defer_to_human,
            "deferral_reasons": list(self.deferral_reasons),
            "validation_issues": [issue.to_dict() for issue in self.validation_issues],
            "agents": {k: v.to_dict() for k, v in self.agents.items()},
            "orchestration_audit": copy.deepcopy(self.orchestration_audit)
        }
        return sanitize_for_json(raw)

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serializes SupervisorState to a strictly RFC 8259-compliant JSON string."""
        d = self.to_dict()
        return json.dumps(d, indent=indent, allow_nan=False)

    @classmethod
    def from_dict(cls, data: Any) -> "SupervisorState":
        """
        Hardened deserialization for SupervisorState.
        Validates required fields, strict types, consistency between primary-agent
        status and deferral, and complete payload integrity.
        """
        if not isinstance(data, dict):
            raise StateValidationError(f"Expected dict for SupervisorState, got {type(data).__name__}.")

        required_fields = ("study_id", "timestamp_utc", "primary_agent_id", "defer_to_human", "agents")
        for req in required_fields:
            if req not in data:
                raise StateValidationError(f"Missing required field '{req}' in SupervisorState.")

        # 1. Validate study_id and timestamp_utc
        study_id = data["study_id"]
        if not isinstance(study_id, str) or not study_id.strip():
            raise StateValidationError(f"study_id must be a non-empty string, got {type(study_id).__name__}.")

        timestamp_utc = data["timestamp_utc"]
        if not isinstance(timestamp_utc, str) or not timestamp_utc.strip():
            raise StateValidationError(f"timestamp_utc must be a non-empty string, got {type(timestamp_utc).__name__}.")

        # 2. Validate primary_agent_id
        primary_agent_id = data["primary_agent_id"]
        if not isinstance(primary_agent_id, str) or primary_agent_id != PRIMARY_AGENT_ID:
            raise StateValidationError(
                f"Invalid primary_agent_id '{primary_agent_id}'. Must be '{PRIMARY_AGENT_ID}'."
            )

        # 3. Strict boolean validation for defer_to_human (no string-to-bool coercion)
        defer_to_human = data["defer_to_human"]
        if not isinstance(defer_to_human, bool):
            raise StateValidationError(
                f"defer_to_human must be a strict boolean, got {type(defer_to_human).__name__}: {repr(defer_to_human)}."
            )

        # 4. Validate agents dict
        raw_agents = data["agents"]
        if not isinstance(raw_agents, dict):
            raise StateValidationError(f"agents must be a dict, got {type(raw_agents).__name__}.")
        if PRIMARY_AGENT_ID not in raw_agents:
            raise StateValidationError(f"agents dict must contain primary agent '{PRIMARY_AGENT_ID}'.")

        agents: Dict[str, AgentFindingPayload] = {}
        for k, v in raw_agents.items():
            if not isinstance(k, str) or k not in VALID_AGENT_IDS:
                raise StateValidationError(f"Unrecognized agent key '{k}' in agents dictionary.")
            payload = AgentFindingPayload.from_dict(v)
            if payload.agent_id != k:
                raise StateValidationError(f"Key mismatch: dictionary key '{k}' != payload agent_id '{payload.agent_id}'.")
            agents[k] = payload

        # 5. Validate validation_issues
        raw_issues = data.get("validation_issues", [])
        if not isinstance(raw_issues, list):
            raise StateValidationError(f"validation_issues must be a list, got {type(raw_issues).__name__}.")
        validation_issues = [ValidationIssue.from_dict(item) for item in raw_issues]

        # 6. Validate deferral_reasons (strictly reject non-string items)
        raw_reasons = data.get("deferral_reasons", [])
        if not isinstance(raw_reasons, list):
            raise StateValidationError(f"deferral_reasons must be a list, got {type(raw_reasons).__name__}.")
        deferral_reasons = []
        for i, r in enumerate(raw_reasons):
            if not isinstance(r, str):
                raise StateValidationError(
                    f"deferral_reasons[{i}] must be a str, got {type(r).__name__}: {repr(r)}."
                )
            deferral_reasons.append(r)

        # 7. HARD SAFETY CONSISTENCY INVARIANT:
        # If the primary authoritative agent failed, is schema_invalid, or is not voting eligible,
        # or if there are ERROR validation issues, defer_to_human MUST be True.
        primary_payload = agents[PRIMARY_AGENT_ID]
        has_primary_error = (primary_payload.status != "success") or (not primary_payload.voting_eligible)
        has_critical_issue = any(issue.severity == "ERROR" for issue in validation_issues)

        if (has_primary_error or has_critical_issue) and not defer_to_human:
            raise StateValidationError(
                "Safety invariant violation: Primary authoritative agent has failed or contains ERROR issues, "
                "but defer_to_human is False. An invalid state cannot be processed as valid."
            )

        if defer_to_human and len(deferral_reasons) == 0:
            # Documented policy: ensure at least one reason is documented if defer_to_human is True
            deferral_reasons.append("Human review required due to authoritative channel validation or status flag.")

        # 8. Validate orchestration_audit
        audit = data.get("orchestration_audit", {})
        if not isinstance(audit, dict):
            raise StateValidationError(f"orchestration_audit must be a dict, got {type(audit).__name__}.")

        return cls(
            study_id=study_id,
            timestamp_utc=timestamp_utc,
            agents=agents,
            primary_agent_id=primary_agent_id,
            defer_to_human=defer_to_human,
            deferral_reasons=deferral_reasons,
            validation_issues=validation_issues,
            orchestration_audit=dict(audit)
        )


def sanitize_for_json(
    val: Any,
    seen: Optional[set] = None,
    depth: int = 0,
    max_depth: int = 32
) -> Any:
    """
    Safely and recursively sanitizes arbitrary objects for strict RFC 8259 JSON compliance.

    TRANSPORT SAFEGUARD VS SEMANTIC VALIDATION:
    1. This function is strictly a transport-level serialization safeguard conforming to
       RFC 8259 JSON specifications (which forbids bare NaN and Infinity tokens).
    2. It converts non-finite floating point numbers (math.isnan -> "nan", math.isinf -> "inf"/"-inf")
       to deterministic string representations to allow error dumps or raw traces to be
       serialized without crashing JSON parsers.
    3. IMPORTANT: Sanitizing data for JSON does NOT validate or approve semantic correctness.
       Converting float("nan") to "nan" does not make it a validated prediction.
       When deserialized, AgentFindingPayload.from_dict() and SupervisorState.from_dict() strictly
       reject "nan", "inf", and non-finite numbers in prediction fields as StateValidationError.
    4. Booleans are strictly preserved as booleans; strings are preserved as strings.
    """
    try:
        if depth >= max_depth:
            return "<max_depth_exceeded>"

        if val is None:
            return None

        # Booleans MUST be checked before int (in Python isinstance(True, int) is True)
        if isinstance(val, bool):
            return val
        if _HAS_NUMPY and isinstance(val, np.bool_):
            return bool(val)

        # Handle NumPy scalars exposing .item()
        if _HAS_NUMPY and hasattr(val, "item") and callable(getattr(val, "item")) and not isinstance(val, (list, tuple, dict, set)):
            try:
                native = val.item()
                if native is not val:
                    return sanitize_for_json(native, seen, depth + 1, max_depth)
            except Exception:
                pass

        # Numeric integer scalars
        if isinstance(val, int):
            return val
        if _HAS_NUMPY and isinstance(val, np.integer):
            return int(val)

        # Floating-point scalars (Python and NumPy)
        is_float = isinstance(val, float) or (_HAS_NUMPY and isinstance(val, np.floating))
        if is_float:
            try:
                f_val = float(val)
                if math.isnan(f_val):
                    return "nan"
                if math.isinf(f_val):
                    return "-inf" if f_val < 0 else "inf"
                return f_val
            except Exception:
                return str(val)

        # Standard strings
        if isinstance(val, str):
            return val

        # Circular reference check
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

        # Sequences: lists, tuples, numpy ndarrays
        is_seq = isinstance(val, (list, tuple)) or (_HAS_NUMPY and isinstance(val, np.ndarray))
        if is_seq:
            return [sanitize_for_json(item, new_seen, depth + 1, max_depth) for item in val]

        # Dataclasses
        if hasattr(val, "__dataclass_fields__"):
            return sanitize_for_json(asdict(val), new_seen, depth + 1, max_depth)

        # Fallback to string representation
        try:
            return str(val)
        except Exception:
            return f"<unprintable_{type(val).__name__}>"

    except Exception:
        try:
            return f"<sanitization_failed_{type(val).__name__}>"
        except Exception:
            return "<sanitization_failed>"
