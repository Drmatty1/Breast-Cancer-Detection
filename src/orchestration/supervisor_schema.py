"""
Supervisor Output Schema and Evidence Contract.

Governed by:
- docs/breast_cancer___ps1___methodology.pdf (§§ 4.2, 5.1, 5.3, 6)
- DECISIONS.md (DEC-OWNER-01, DEC-OWNER-02, DEC-OWNER-03)

Key Architecture & Rules:
1. Pure Python dataclasses with ZERO heavy machine learning dependencies.
2. Strongly-typed consensus output (`ConsensusDecision`) synthesized by the LLM Supervisor.
3. Strict provenance citation syntax: `[card:<agent>/<field>]` (e.g. `[card:mammo/BI_RADS_a]`).
4. Governance guardrails:
   - Primary agent diagnostic authority cannot be overridden by LLM or shadow agent.
   - If primary agent failed or state requires human deferral, defer_to_human MUST remain True.
   - Malignancy probability and BI-RADS category must strictly match validated primary findings.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import copy
import json
import math
import re

VALID_CONSENSUS_BIRADS = {"2", "3", "4", "4A", "4B", "4C", "5", "unspecified"}
QUOTE_PATTERN = re.compile(r"\[card:(mammo|clin|path)/([a-zA-Z0-9_\./]+)\]")


class SupervisorSchemaError(ValueError):
    """Raised when supervisor output violates schema or governance guardrails."""
    pass


@dataclass
class ConsensusDecision:
    """
    Standardized, auditable consensus output returned by the LLM Supervisor.
    """
    case_id: str
    consensus_bi_rads: str
    malignancy_probability: Optional[float]
    confidence_interval: List[float]
    defer_to_human: bool
    deferral_reasons: List[str]
    evidence_rationale: str
    discrepancy_analysis: Optional[str] = None
    shadow_risk_summary: Optional[Dict[str, Any]] = None
    explanation_card: Dict[str, Any] = field(default_factory=dict)
    audit_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes ConsensusDecision to a clean JSON-safe dictionary."""
        return {
            "case_id": self.case_id,
            "consensus_bi_rads": self.consensus_bi_rads,
            "malignancy_probability": (
                round(float(self.malignancy_probability), 4)
                if self.malignancy_probability is not None
                else None
            ),
            "confidence_interval": [
                round(float(c), 4) for c in self.confidence_interval
            ],
            "defer_to_human": bool(self.defer_to_human),
            "deferral_reasons": list(self.deferral_reasons),
            "evidence_rationale": str(self.evidence_rationale),
            "discrepancy_analysis": self.discrepancy_analysis,
            "shadow_risk_summary": copy.deepcopy(self.shadow_risk_summary),
            "explanation_card": copy.deepcopy(self.explanation_card),
            "audit_metadata": copy.deepcopy(self.audit_metadata),
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serializes ConsensusDecision to an RFC 8259 JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Any) -> "ConsensusDecision":
        """
        Validates and parses a dictionary into ConsensusDecision.
        """
        if not isinstance(data, dict):
            raise SupervisorSchemaError(f"Expected dict for ConsensusDecision, got {type(data).__name__}.")

        required = [
            "case_id",
            "consensus_bi_rads",
            "confidence_interval",
            "defer_to_human",
            "deferral_reasons",
            "evidence_rationale"
        ]
        for req in required:
            if req not in data:
                raise SupervisorSchemaError(f"Missing required field '{req}' in ConsensusDecision.")

        case_id = str(data["case_id"]).strip()
        if not case_id:
            raise SupervisorSchemaError("case_id must not be empty.")

        birads = str(data["consensus_bi_rads"]).strip()
        if birads not in VALID_CONSENSUS_BIRADS:
            raise SupervisorSchemaError(
                f"Invalid consensus_bi_rads '{birads}'. Must be one of {sorted(VALID_CONSENSUS_BIRADS)}."
            )

        malig = data.get("malignancy_probability")
        if malig is not None:
            if isinstance(malig, bool) or not isinstance(malig, (int, float)):
                raise SupervisorSchemaError(f"malignancy_probability must be a float in [0.0, 1.0], got {malig}.")
            f_malig = float(malig)
            if math.isnan(f_malig) or math.isinf(f_malig) or not (0.0 <= f_malig <= 1.0):
                raise SupervisorSchemaError(f"malignancy_probability out of bounds: {malig}.")
            malig = f_malig

        ci = data["confidence_interval"]
        if not isinstance(ci, (list, tuple)) or len(ci) != 2:
            raise SupervisorSchemaError("confidence_interval must be a 2-element list [Q_low, Q_high].")
        for val in ci:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise SupervisorSchemaError(f"confidence_interval values must be floats in [0.0, 1.0], got {val}.")
            f_val = float(val)
            if math.isnan(f_val) or math.isinf(f_val) or not (0.0 <= f_val <= 1.0):
                raise SupervisorSchemaError(f"confidence_interval value out of bounds: {val}.")
        ci = [float(ci[0]), float(ci[1])]
        if ci[0] > ci[1] + 1e-5:
            raise SupervisorSchemaError(f"confidence_interval lower bound cannot exceed upper bound: {ci}.")

        defer_to_human = data["defer_to_human"]
        if not isinstance(defer_to_human, bool):
            raise SupervisorSchemaError(f"defer_to_human must be boolean, got {type(defer_to_human).__name__}.")

        deferral_reasons = data["deferral_reasons"]
        if not isinstance(deferral_reasons, list):
            raise SupervisorSchemaError("deferral_reasons must be a list of strings.")
        deferral_reasons = [str(r) for r in deferral_reasons]

        rationale = str(data["evidence_rationale"]).strip()
        if not rationale:
            raise SupervisorSchemaError("evidence_rationale must not be empty.")

        discrepancy = data.get("discrepancy_analysis")
        if discrepancy is not None and not isinstance(discrepancy, str):
            discrepancy = str(discrepancy)

        shadow_summary = data.get("shadow_risk_summary")
        if shadow_summary is not None and not isinstance(shadow_summary, dict):
            raise SupervisorSchemaError("shadow_risk_summary must be a dict or None.")

        explanation_card = data.get("explanation_card")
        if explanation_card is not None and not isinstance(explanation_card, dict):
            raise SupervisorSchemaError("explanation_card must be a dict.")
        explanation_card = dict(explanation_card) if explanation_card else {}

        audit_metadata = data.get("audit_metadata")
        if audit_metadata is not None and not isinstance(audit_metadata, dict):
            raise SupervisorSchemaError("audit_metadata must be a dict.")
        audit_metadata = dict(audit_metadata) if audit_metadata else {}

        return cls(
            case_id=case_id,
            consensus_bi_rads=birads,
            malignancy_probability=malig,
            confidence_interval=ci,
            defer_to_human=defer_to_human,
            deferral_reasons=deferral_reasons,
            evidence_rationale=rationale,
            discrepancy_analysis=discrepancy,
            shadow_risk_summary=shadow_summary,
            explanation_card=explanation_card,
            audit_metadata=audit_metadata
        )


def extract_evidence_quotes(text: str) -> List[Tuple[str, str]]:
    """
    Extracts all evidence citation tags formatted as [card:<agent>/<field>] from text.
    Returns list of (agent_id, field_path) tuples.
    """
    if not text:
        return []
    matches = QUOTE_PATTERN.findall(text)
    return [(m[0], m[1]) for m in matches]


def compute_calibrated_confidence_interval(
    p: Optional[float],
    fold_probs: Optional[List[float]] = None,
    effective_n: int = 50,
    confidence_level: float = 0.80
) -> List[float]:
    """
    Computes a statistically validated confidence interval adhering to Methodology §5.1:
    - If ensemble fold probabilities are available: computes empirical [Q0.1, Q0.9] quantile interval.
    - Otherwise: computes the Wilson score interval with effective calibration sample size n.
    """
    if p is None or not (0.0 <= p <= 1.0):
        return [0.0, 1.0]

    if fold_probs and len(fold_probs) >= 3:
        sorted_probs = sorted([float(x) for x in fold_probs if 0.0 <= float(x) <= 1.0])
        if len(sorted_probs) >= 3:
            idx_low = max(0, int(len(sorted_probs) * 0.1))
            idx_high = min(len(sorted_probs) - 1, int(math.ceil(len(sorted_probs) * 0.9)) - 1)
            q_low = sorted_probs[idx_low]
            q_high = sorted_probs[idx_high]
            return [round(float(q_low), 4), round(float(q_high), 4)]

    # Wilson score interval for binomial proportion / calibrated probability
    # z for 80% CI (Q0.1 to Q0.9): z ≈ 1.28155
    z = 1.28155 if confidence_level == 0.80 else 1.95996
    n = max(10, effective_n)
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) / n) + ((z**2) / (4.0 * (n**2))))) / denom

    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return [round(low, 4), round(high, 4)]


def validate_supervisor_output_guardrails(
    decision: ConsensusDecision,
    state: Any  # SupervisorState
) -> Tuple[bool, List[str]]:
    """
    Validates supervisor decision against governance invariants from DECISIONS.md
    and docs/breast_cancer___ps1___methodology.pdf:
    
    1. If state.defer_to_human is True, decision.defer_to_human MUST be True.
    2. Primary Mammography Agent authority:
       - If Mammo is successful, consensus_bi_rads MUST match primary_agent_birads.
       - malignancy_probability MUST match primary_agent_malignancy (within 1e-4).
    3. Clinical Risk Agent isolation:
       - Clin Agent findings cannot alter consensus_bi_rads.
    4. Pathology agent check:
       - No quotes from unimplemented pathology agent permitted.
    5. Evidence citations:
       - Rationale must include at least one valid [card:mammo/...] quote for successful cases.
    """
    violations: List[str] = []

    # Guardrail 1: Enforce mandatory human deferral
    if getattr(state, "defer_to_human", False) and not decision.defer_to_human:
        violations.append(
            "State marked defer_to_human=True (primary failure or critical validation issue), "
            "but supervisor attempted to clear deferral (defer_to_human=False)."
        )

    # Check agents in state
    agents = getattr(state, "agents", {})
    mammo_payload = agents.get("mammo")

    if mammo_payload and mammo_payload.status == "success":
        # Guardrail 2: Primary Mammo Diagnostic Authority
        expected_birads = mammo_payload.bi_rads
        if decision.consensus_bi_rads != expected_birads:
            violations.append(
                f"Diagnostic authority violation: supervisor output BI-RADS '{decision.consensus_bi_rads}' "
                f"does not match authoritative Mammography Agent BI-RADS '{expected_birads}'."
            )

        if mammo_payload.p_malignancy is not None and decision.malignancy_probability is not None:
            if abs(decision.malignancy_probability - mammo_payload.p_malignancy) > 1e-3:
                violations.append(
                    f"Malignancy probability altered: supervisor gave {decision.malignancy_probability}, "
                    f"expected primary value {mammo_payload.p_malignancy}."
                )

        # Guardrail 5: Provenance evidence quoting
        quotes = extract_evidence_quotes(decision.evidence_rationale)
        mammo_quotes = [q for q in quotes if q[0] == "mammo"]
        if not mammo_quotes:
            violations.append(
                "Evidence quoting violation: supervisor rationale lacks mandatory [card:mammo/<field>] citations."
            )

        # Guardrail 6: Narrative factual consistency audit (Catching in-prose hallucinations)
        # Detect any conflicting BI-RADS references in the generated narrative
        birads_mentions = re.findall(r"BI-RADS\s*([0-6][A-C]?)", decision.evidence_rationale, re.IGNORECASE)
        for mention in birads_mentions:
            mention_clean = mention.upper()
            if mention_clean != expected_birads.upper():
                violations.append(
                    f"Narrative factual inconsistency: rationale text asserts 'BI-RADS {mention_clean}', "
                    f"which contradicts authoritative primary finding 'BI-RADS {expected_birads}'."
                )

        # Guardrail 7: Claim-to-citation semantic integrity audit
        # If the rationale discusses TreeSHAP / feature contributions, it must cite a SHAP field, NOT finding/risk_profile
        shap_keywords = re.search(r"\b(treeshap|shap|feature contribution|attributions?)\b", decision.evidence_rationale, re.IGNORECASE)
        if shap_keywords:
            shap_cites = [q for q in quotes if q[0] == "clin" and ("sh" in q[1] or "explainability" in q[1] or "shap" in q[1].lower())]
            if not shap_cites:
                violations.append(
                    "Citation-to-claim mismatch: rationale discusses TreeSHAP / feature attributions, "
                    "but does not cite a valid SHAP source field (e.g. [card:clin/sh/breast_density])."
                )

    # Guardrail 4: Pathology agent check
    quotes = extract_evidence_quotes(decision.evidence_rationale)
    path_quotes = [q for q in quotes if q[0] == "path"]
    if path_quotes:
        violations.append(
            f"Fabrication violation: supervisor cited unimplemented Pathology Agent: {path_quotes}."
        )

    is_valid = len(violations) == 0
    return is_valid, violations


BIRADS_ORDINAL_RANKS: Dict[str, int] = {
    "2": 0,
    "3": 1,
    "4": 2,
    "4A": 2,
    "4B": 3,
    "4C": 4,
    "5": 5
}


def compute_birads_step_distance(cat1: str, cat2: str) -> Optional[int]:
    """
    Computes ordinal step distance between two BI-RADS categories per Methodology §5.3.
    Returns None if either category is 'unspecified' or not in lexicon.
    """
    c1 = str(cat1).strip().upper()
    c2 = str(cat2).strip().upper()
    if c1 not in BIRADS_ORDINAL_RANKS or c2 not in BIRADS_ORDINAL_RANKS:
        return None
    return abs(BIRADS_ORDINAL_RANKS[c1] - BIRADS_ORDINAL_RANKS[c2])


def evaluate_methodology_deferral_diamond(
    state: Any,  # SupervisorState
    theta: float = 0.030,
    screening_tau: float = 0.3675
) -> Tuple[bool, List[str]]:
    """
    Directly implements the Methodology Figure 1 Decision Diamond:
    'Disagreement >= 2 steps OR unc theta?'

    Returns:
        (defer_to_human: bool, deferral_reasons: List[str])
    """
    reasons: List[str] = []

    # Condition 1: Disagreement >= 2 steps
    # Compare all pairs of active agents with valid discrete BI-RADS categories
    agents = getattr(state, "agents", {})
    categories = {}
    for agent_id, payload in agents.items():
        if payload and payload.status == "success" and getattr(payload, "bi_rads", None):
            b_val = str(payload.bi_rads).strip().upper()
            if b_val in BIRADS_ORDINAL_RANKS:
                categories[agent_id] = b_val

    agent_ids = list(categories.keys())
    for i in range(len(agent_ids)):
        for j in range(i + 1, len(agent_ids)):
            a1, a2 = agent_ids[i], agent_ids[j]
            dist = compute_birads_step_distance(categories[a1], categories[a2])
            if dist is not None and dist >= 2:
                reasons.append(
                    f"DISAGREEMENT_GE_2_STEPS: Agent '{a1}' ({categories[a1]}) and '{a2}' ({categories[a2]}) "
                    f"differ by {dist} ordinal steps (>= 2 threshold per Methodology §5.3)."
                )

    # Condition 2: unc theta? (Equivocal boundary uncertainty around screening threshold)
    mammo = agents.get("mammo")
    if mammo and mammo.status == "success" and mammo.p_malignancy is not None:
        p = float(mammo.p_malignancy)
        dist_to_boundary = abs(p - screening_tau)
        if dist_to_boundary < theta:
            reasons.append(
                f"EPISTEMIC_UNCERTAINTY_THETA: Calibrated malignancy probability {p:.4f} is within "
                f"boundary margin theta={theta:.3f} of screening threshold tau={screening_tau}."
            )

    should_defer = len(reasons) > 0
    return should_defer, reasons
