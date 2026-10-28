"""
LLM Supervisor Reasoning Engine.

Governed by:
- docs/breast_cancer___ps1___methodology.pdf (§§ 4.1, 4.2, 5.1, 5.3, 6)
- DECISIONS.md (DEC-OWNER-01, DEC-OWNER-02, DEC-OWNER-03)

Implements the Reasoning Layer using Qwen2.5-3B-Instruct via local Ollama
(or a pluggable client interface for offline deterministic testing).
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import json
import time
import math
import re
import urllib.request
import urllib.error

from src.orchestration.supervisor_contract import (
    SupervisorState,
    AgentFindingPayload,
    ValidationIssue
)
from src.orchestration.supervisor_schema import (
    ConsensusDecision,
    SupervisorSchemaError,
    extract_evidence_quotes,
    validate_supervisor_output_guardrails,
    compute_calibrated_confidence_interval,
    evaluate_methodology_deferral_diamond
)


class LLMClientError(RuntimeError):
    """Raised when the LLM client encounters an API or connection error."""
    pass


class BaseLLMClient:
    """Abstract interface for LLM backends."""
    def generate(self, prompt: str, json_mode: bool = True, temperature: float = 0.0) -> str:
        raise NotImplementedError


class OllamaLLMClient(BaseLLMClient):
    """
    HTTP client for local Ollama server (e.g. Qwen2.5-3B-Instruct).
    Requires zero external third-party dependencies (pure standard library).
    """
    def __init__(
        self,
        model_name: str = "qwen2.5:3b",
        host: str = "127.0.0.1",
        port: int = 11434,
        timeout_seconds: float = 30.0
    ):
        self.model_name = model_name
        self.endpoint = f"http://{host}:{port}/api/generate"
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str, json_mode: bool = True, temperature: float = 0.0) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        if json_mode:
            payload["format"] = "json"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                result_bytes = resp.read()
                result_json = json.loads(result_bytes.decode("utf-8"))
                return result_json.get("response", "")
        except urllib.error.URLError as e:
            raise LLMClientError(f"Failed to connect to Ollama at {self.endpoint}: {e}")
        except Exception as e:
            raise LLMClientError(f"Error querying Ollama API: {e}")


class MockLLMClient(BaseLLMClient):
    """
    Deterministic mock client for offline unit testing and fallback verification.
    """
    def __init__(self, response_generator: Optional[Any] = None):
        self.response_generator = response_generator

    def generate(self, prompt: str, json_mode: bool = True, temperature: float = 0.0) -> str:
        if callable(self.response_generator):
            return self.response_generator(prompt)
        if isinstance(self.response_generator, str):
            return self.response_generator

        # Default sensible mock response that satisfies methodology constraints
        return json.dumps({
            "consensus_bi_rads": "4C",
            "malignancy_probability": 0.8123,
            "confidence_interval": [0.7623, 0.8623],
            "defer_to_human": False,
            "deferral_reasons": [],
            "evidence_rationale": (
                "Primary mammography assessment demonstrates a spiculated mass "
                "[card:mammo/finding/margin] with fine pleomorphic calcifications "
                "[card:mammo/finding/calcification], resulting in an authoritative BI-RADS 4C rating "
                "[card:mammo/BI_RADS_a]. Clinical risk is evaluated in shadow mode [card:clin/finding/risk_profile]."
            ),
            "discrepancy_analysis": "No significant discrepancy observed between modalities."
        })


def build_supervisor_prompt(state: SupervisorState) -> str:
    """
    Constructs the prompt for Qwen2.5-3B encoding methodology rules,
    agent findings, and evidence quoting protocols.
    """
    mammo = state.agents.get("mammo")
    clin = state.agents.get("clin")

    mammo_birads = mammo.bi_rads if mammo and mammo.status == "success" else "unspecified"
    p_malig = mammo.p_malignancy if mammo and mammo.status == "success" and mammo.p_malignancy is not None else 0.5
    calib_ci = compute_calibrated_confidence_interval(p_malig)

    prompt = f"""You are the Multi-Agent Mammography Diagnostic Supervisor.
Your role is to synthesize the specialist agent findings into an auditable consensus diagnostic decision.

=== METHODOLOGY GOVERNANCE RULES ===
1. Sole Diagnostic Authority: The Mammography Agent ('mammo') is the ONLY authoritative diagnostic tier.
   Its BI-RADS ('{mammo_birads}') and calibrated malignancy probability MUST form the consensus diagnostic tier.
   CRITICAL: In your rationale text, when stating the BI-RADS, you MUST state the exact value 'BI-RADS {mammo_birads}'. Never alter or subcategorize it (e.g. if the category is 4, do NOT write 4C).
2. Clinical Agent Shadow Mode: The Clinical Risk Agent ('clin') operates strictly in SHADOW MODE (voting_eligible=false).
   It provides contextual risk and TreeSHAP feature contributions. It CANNOT vote, alter BI-RADS, or override the Mammography Agent.
3. Mandatory Evidence Quoting: For EVERY factual statement in your rationale, you MUST cite the exact field source using the tag format [card:<agent>/<field>].
   - If citing lesions or margin, cite [card:mammo/finding/margin] or [card:mammo/finding/lesion].
   - If citing BI-RADS, cite [card:mammo/BI_RADS_a].
   - If citing clinical risk profile, cite [card:clin/finding/risk_profile].
   - If citing TreeSHAP feature attributions, you MUST cite [card:clin/sh/breast_density] (do NOT cite finding/risk_profile for SHAP values).
   Write 2 to 3 full, fluent clinical sentences explaining the findings and embedding the tags naturally.
   Example: "Two-view mammography demonstrates the findings [card:mammo/finding/margin] rated BI-RADS {mammo_birads} [card:mammo/BI_RADS_a]. Tabular clinical risk is evaluated in shadow mode with density assessment [card:clin/finding/risk_profile], with TreeSHAP contribution from breast density [card:clin/sh/breast_density]."
   Do NOT output raw tags alone; embed them into informative clinical prose.
4. Deferral Policy:
   - If Mammography Agent failed or state has defer_to_human=true, you MUST set defer_to_human=true.
   - If inter-agent discrepancy between BI-RADS categories is >= 2 steps, flag discrepancy and defer.
   - Do NOT invent findings or cite unimplemented agents (like pathology).

=== CASE DATA ===
Study ID: {state.study_id}
State Deferral Flag: {state.defer_to_human}
State Deferral Reasons: {state.deferral_reasons}

--- Primary Mammography Agent Findings ---
Status: {mammo.status if mammo else 'MISSING'}
BI-RADS: {mammo.bi_rads if mammo else 'N/A'}
Calibrated Malignancy Probability: {mammo.p_malignancy if mammo else 'N/A'}
Confidence: {mammo.confidence if mammo else 'N/A'}
Findings Summary: {json.dumps(mammo.findings_summary if mammo else {})}
Task Probabilities: {json.dumps(mammo.task_probabilities if mammo else {})}
Explainability: {json.dumps(mammo.explainability if mammo else {})}

--- Shadow Clinical Risk Agent Findings ---
Status: {clin.status if clin else 'UNAVAILABLE'}
Risk Probability: {clin.p_malignancy if clin else 'N/A'} [card:clin/p_malignancy]
Findings Summary: {json.dumps(clin.findings_summary if clin else {})} [card:clin/finding/risk_profile]
SHAP Attributions: {json.dumps(clin.explainability.get('tree_shap_attributions', {}) if clin else {})} [card:clin/sh/breast_density]

=== OUTPUT REQUIREMENTS ===
You must respond with ONLY a valid JSON object matching this exact schema:
{{
  "consensus_bi_rads": "{mammo_birads}",
  "malignancy_probability": {mammo.p_malignancy if mammo and mammo.status == 'success' and mammo.p_malignancy is not None else 'null'},
  "confidence_interval": {json.dumps(calib_ci)},
  "defer_to_human": { 'true' if state.defer_to_human else 'false' },
  "deferral_reasons": {json.dumps(state.deferral_reasons)},
  "evidence_rationale": "Comprehensive clinical rationale citing [card:mammo/...] and [card:clin/...] tags.",
  "discrepancy_analysis": "Analysis of agreement or discrepancy between modalities."
}}
"""
    return prompt


def clean_llm_json_response(raw_text: str) -> str:
    """Strips markdown code fences and cleans JSON output from LLM."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


class LLMSupervisor:
    """
    The central RP2 LLM Supervisor agent.
    Coordinates deliberation, communicates with local Qwen 2.5 3B,
    and enforces medical governance and hallucination barriers.
    """
    def __init__(
        self,
        client: Optional[BaseLLMClient] = None,
        model_name: str = "qwen2.5:3b",
        theta: float = 0.0,
        screening_tau: float = 0.3675
    ):
        self.client = client or OllamaLLMClient(model_name=model_name)
        self.model_name = getattr(self.client, "model_name", "mock_supervisor")
        self.theta = theta
        self.screening_tau = screening_tau

    def deliberate(
        self,
        state: SupervisorState,
        theta: Optional[float] = None
    ) -> ConsensusDecision:
        """
        Executes a supervised deliberation cycle over the validated SupervisorState.
        Guarantees safe, schema-compliant, and governance-validated ConsensusDecision
        directly adhering to Methodology Figure 1 Decision Diamond:
        'Disagreement >= 2 steps OR unc theta?'
        """
        start_time = time.time()
        mammo = state.agents.get("mammo")
        clin = state.agents.get("clin")

        active_theta = theta if theta is not None else self.theta

        # 1. Immediate Safety Check: Primary failure or explicit deferral in state
        force_defer = state.defer_to_human or (not mammo) or (mammo.status != "success")
        initial_reasons = list(state.deferral_reasons)
        if not mammo or mammo.status != "success":
            if "PRIMARY_AGENT_FAILED" not in initial_reasons:
                initial_reasons.append("PRIMARY_AGENT_FAILED")

        # 2. Evaluate Methodology Figure 1 Decision Diamond: "Disagreement >= 2 steps OR unc theta?"
        diamond_defer, diamond_reasons = evaluate_methodology_deferral_diamond(
            state, theta=active_theta, screening_tau=self.screening_tau
        )
        if diamond_defer:
            force_defer = True
            for r in diamond_reasons:
                if r not in initial_reasons:
                    initial_reasons.append(r)

        # Build Explanation Card components
        explanation_card = {
            "saliency_path": mammo.explainability.get("saliency_path") if mammo else None,
            "top_regions": mammo.explainability.get("top_regions", []) if mammo else [],
            "shap_summary": (
                clin.explainability.get("tree_shap_attributions")
                if clin and clin.status == "success"
                else {}
            ),
            "consensus_trail": [
                {
                    "stage": "input_state_validation",
                    "timestamp_utc": state.timestamp_utc,
                    "validation_issues_count": len(state.validation_issues),
                    "defer_to_human": state.defer_to_human
                },
                {
                    "stage": "methodology_decision_diamond",
                    "condition": "Disagreement >= 2 steps OR unc theta?",
                    "branch_taken": "Yes (Defer to Human Reader)" if force_defer else "No (Consensus + CI + Evidence Card)",
                    "reasons": initial_reasons
                }
            ]
        }

        # 2. Build Prompt & Query LLM
        prompt = build_supervisor_prompt(state)
        llm_success = False
        parsed_llm_json: Dict[str, Any] = {}
        llm_error_note: Optional[str] = None

        try:
            raw_response = self.client.generate(prompt, json_mode=True, temperature=0.0)
            cleaned_json = clean_llm_json_response(raw_response)
            parsed_llm_json = json.loads(cleaned_json)
            llm_success = True
        except Exception as e:
            llm_error_note = f"LLM generation/decoding failed: {e}"

        # 3. Assemble Consensus Decision Candidate
        stat_ci = compute_calibrated_confidence_interval(mammo.p_malignancy if mammo else None)
        ci = (
            parsed_llm_json.get("confidence_interval")
            if llm_success and isinstance(parsed_llm_json.get("confidence_interval"), list) and len(parsed_llm_json["confidence_interval"]) == 2
            else stat_ci
        )

        proposed_bi_rads = (
            str(parsed_llm_json.get("consensus_bi_rads")).strip()
            if (llm_success and parsed_llm_json.get("consensus_bi_rads"))
            else (mammo.bi_rads if mammo and mammo.status == "success" else "unspecified")
        )
        proposed_malig = (
            float(parsed_llm_json.get("malignancy_probability"))
            if (llm_success and parsed_llm_json.get("malignancy_probability") is not None and isinstance(parsed_llm_json.get("malignancy_probability"), (int, float)))
            else (mammo.p_malignancy if mammo and mammo.status == "success" else None)
        )

        defer_to_human = force_defer or bool(parsed_llm_json.get("defer_to_human", False))
        deferral_reasons = list(initial_reasons)
        if parsed_llm_json.get("deferral_reasons"):
            for r in parsed_llm_json["deferral_reasons"]:
                if r not in deferral_reasons:
                    deferral_reasons.append(str(r))

        if llm_error_note:
            defer_to_human = True
            if "LLM_DECODING_ERROR" not in deferral_reasons:
                deferral_reasons.append("LLM_DECODING_ERROR")
            evidence_rationale = (
                f"Automated fallback rationale: Mammography agent reported BI-RADS {mammo.bi_rads if mammo else 'unspecified'} "
                f"[card:mammo/BI_RADS_a]. LLM deliberation encountered an issue ({llm_error_note}). "
                f"Case deferred for human radiologist review."
            )
        else:
            evidence_rationale = str(parsed_llm_json.get("evidence_rationale", "")).strip()
            if not evidence_rationale:
                evidence_rationale = (
                    f"Mammography assessment indicates BI-RADS {mammo.bi_rads if mammo else 'unspecified'} [card:mammo/BI_RADS_a]. "
                    f"Malignancy probability: {mammo.p_malignancy if mammo else 'N/A'} [card:mammo/p_malignancy]."
                )

        shadow_summary = None
        if clin and clin.status == "success":
            shadow_summary = {
                "agent_id": "clin",
                "authority": "non_authoritative_research",
                "risk_probability": clin.p_malignancy,
                "findings": clin.findings_summary
            }

        decision_candidate = ConsensusDecision(
            case_id=state.study_id,
            consensus_bi_rads=proposed_bi_rads,
            malignancy_probability=proposed_malig,
            confidence_interval=ci,
            defer_to_human=defer_to_human,
            deferral_reasons=deferral_reasons,
            evidence_rationale=evidence_rationale,
            discrepancy_analysis=parsed_llm_json.get("discrepancy_analysis"),
            shadow_risk_summary=shadow_summary,
            explanation_card=explanation_card,
            audit_metadata={
                "supervisor_model": self.model_name,
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "deliberation_round": 1,
                "guardrail_status": "pending"
            }
        )

        # 4. Enforce Governance Guardrails Post-Validation
        is_valid, violations = validate_supervisor_output_guardrails(decision_candidate, state)
        if not is_valid:
            # Overwrite rationale and force safe human deferral
            decision_candidate.defer_to_human = True
            # Revert any altered diagnostic findings back to authoritative primary findings
            if mammo and mammo.status == "success":
                decision_candidate.consensus_bi_rads = mammo.bi_rads
                decision_candidate.malignancy_probability = mammo.p_malignancy
            for v in violations:
                decision_candidate.deferral_reasons.append(f"GUARDRAIL_VIOLATION: {v}")
            # Ensure proper mammo quote is present in fallback rationale
            decision_candidate.evidence_rationale = (
                f"Automated safety fallback: Mammography Agent reported BI-RADS {decision_candidate.consensus_bi_rads} [card:mammo/BI_RADS_a]. "
                f"Supervisory guardrail detected violations: {'; '.join(violations)}. "
                f"Case deferred to human reader."
            )
            decision_candidate.audit_metadata["guardrail_status"] = "failed_deferred"
        else:
            decision_candidate.audit_metadata["guardrail_status"] = "passed"

        explanation_card["consensus_trail"].append({
            "stage": "final_consensus_evaluation",
            "guardrail_status": decision_candidate.audit_metadata["guardrail_status"],
            "defer_to_human": decision_candidate.defer_to_human
        })

        return decision_candidate
