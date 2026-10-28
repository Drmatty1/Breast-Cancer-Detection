"""
Unit and Integration Tests for LLM Supervisor Agent.

Governed by:
- docs/breast_cancer___ps1___methodology.pdf (§§ 4.2, 5.1, 5.3, 6)
- DECISIONS.md (DEC-OWNER-01, DEC-OWNER-02, DEC-OWNER-03)
"""
import unittest
import json
import urllib.request

from src.orchestration.supervisor_contract import (
    SupervisorState,
    AgentFindingPayload,
    ValidationIssue
)
from src.orchestration.supervisor_schema import (
    ConsensusDecision,
    SupervisorSchemaError,
    extract_evidence_quotes,
    validate_supervisor_output_guardrails
)
from src.orchestration.llm_supervisor import (
    LLMSupervisor,
    MockLLMClient,
    OllamaLLMClient,
    build_supervisor_prompt,
    clean_llm_json_response
)


def _make_dummy_mammo_payload(
    status: str = "success",
    bi_rads: str = "4C",
    confidence: float = 0.85,
    p_malignancy: float = 0.8123
) -> AgentFindingPayload:
    return AgentFindingPayload(
        agent_id="mammo",
        status=status,
        authority_level="authoritative",
        voting_eligible=True,
        bi_rads=bi_rads,
        confidence=confidence,
        confidence_type="calibrated_temperature",
        p_malignancy=p_malignancy,
        findings_summary={
            "lesion": "present",
            "margin": "spiculated",
            "calcification": "fine_pleomorphic"
        },
        task_probabilities={
            "malignancy_calibrated": p_malignancy,
            "margin_distribution": [0.05, 0.1, 0.05, 0.8],
            "calc_distribution": [0.05, 0.05, 0.05, 0.8, 0.05],
            "birads_category_distribution": [0.02, 0.03, 0.15, 0.80]
        },
        explainability={
            "saliency_path": "reports/mammo/gradcam_4C.png",
            "top_regions": [[120, 200, 80, 80]]
        },
        rationale="Spiculated mass with fine pleomorphic calcifications.",
        raw_finding_card={},
        metadata={}
    )


def _make_dummy_clin_payload(
    status: str = "success",
    p_malignancy: float = 0.42
) -> AgentFindingPayload:
    return AgentFindingPayload(
        agent_id="clin",
        status=status,
        authority_level="non_authoritative_research",
        voting_eligible=False,
        bi_rads="unspecified",
        confidence=0.0,
        confidence_type="non_authoritative_zero_weight_convention",
        p_malignancy=p_malignancy,
        findings_summary={
            "lesion": None,
            "risk_profile": "elevated",
            "breast_density": 3
        },
        task_probabilities={
            "malignancy_calibrated": p_malignancy
        },
        explainability={
            "tree_shap_attributions": {"density": 0.14, "age": 0.08}
        },
        rationale="Elevated breast density contributes to higher lifetime risk.",
        raw_finding_card={},
        metadata={}
    )


class TestSupervisorSchema(unittest.TestCase):
    """Tests for ConsensusDecision schema and validation rules."""

    def test_valid_consensus_decision_serialization(self):
        decision = ConsensusDecision(
            case_id="TEST_001",
            consensus_bi_rads="4C",
            malignancy_probability=0.8123,
            confidence_interval=[0.76, 0.86],
            defer_to_human=False,
            deferral_reasons=[],
            evidence_rationale="Spiculated mass [card:mammo/finding/margin] rated BI-RADS 4C [card:mammo/BI_RADS_a]."
        )
        d_dict = decision.to_dict()
        self.assertEqual(d_dict["consensus_bi_rads"], "4C")
        self.assertEqual(d_dict["malignancy_probability"], 0.8123)
        self.assertFalse(d_dict["defer_to_human"])

        json_str = decision.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["case_id"], "TEST_001")

        restored = ConsensusDecision.from_dict(parsed)
        self.assertEqual(restored.case_id, "TEST_001")
        self.assertEqual(restored.consensus_bi_rads, "4C")

    def test_invalid_birads_rejected(self):
        with self.assertRaises(SupervisorSchemaError):
            ConsensusDecision.from_dict({
                "case_id": "TEST_002",
                "consensus_bi_rads": "INVALID_CATEGORY",
                "confidence_interval": [0.1, 0.2],
                "defer_to_human": False,
                "deferral_reasons": [],
                "evidence_rationale": "Valid rationale."
            })

    def test_invalid_confidence_interval_rejected(self):
        # Inverted interval (Q_low > Q_high)
        with self.assertRaises(SupervisorSchemaError):
            ConsensusDecision.from_dict({
                "case_id": "TEST_003",
                "consensus_bi_rads": "2",
                "confidence_interval": [0.8, 0.2],
                "defer_to_human": False,
                "deferral_reasons": [],
                "evidence_rationale": "Rationale."
            })

    def test_evidence_quote_extraction(self):
        text = (
            "Mass noted [card:mammo/finding/margin] with calcifications [card:mammo/finding/calcification]. "
            "Primary BI-RADS is 4C [card:mammo/BI_RADS_a]. Contextual risk is elevated [card:clin/finding/risk_profile]."
        )
        quotes = extract_evidence_quotes(text)
        self.assertEqual(len(quotes), 4)
        self.assertIn(("mammo", "BI_RADS_a"), quotes)
        self.assertIn(("clin", "finding/risk_profile"), quotes)


class TestLLMSupervisorEngine(unittest.TestCase):
    """Tests for LLMSupervisor reasoning and governance guardrails."""

    def test_concordant_synthesis(self):
        mammo = _make_dummy_mammo_payload()
        clin = _make_dummy_clin_payload()
        state = SupervisorState(
            study_id="STUDY_CONCORDANT",
            timestamp_utc="2026-10-27T00:00:00Z",
            agents={"mammo": mammo, "clin": clin},
            defer_to_human=False
        )

        mock_resp = json.dumps({
            "consensus_bi_rads": "4C",
            "malignancy_probability": 0.8123,
            "confidence_interval": [0.76, 0.86],
            "defer_to_human": False,
            "deferral_reasons": [],
            "evidence_rationale": (
                "Primary mammography reveals a spiculated mass [card:mammo/finding/margin] "
                "with fine pleomorphic calcifications [card:mammo/finding/calcification], "
                "concordant with BI-RADS 4C [card:mammo/BI_RADS_a]. "
                "Clinical risk shows elevated density [card:clin/explainability/density]."
            ),
            "discrepancy_analysis": "Concordant high suspicion findings."
        })

        client = MockLLMClient(response_generator=mock_resp)
        supervisor = LLMSupervisor(client=client)
        decision = supervisor.deliberate(state)

        self.assertEqual(decision.case_id, "STUDY_CONCORDANT")
        self.assertEqual(decision.consensus_bi_rads, "4C")
        self.assertFalse(decision.defer_to_human)
        self.assertEqual(decision.audit_metadata["guardrail_status"], "passed")

    def test_primary_failure_forces_human_deferral(self):
        mammo = _make_dummy_mammo_payload(status="failed", bi_rads="unspecified", p_malignancy=None)
        state = SupervisorState(
            study_id="STUDY_MAMMO_FAILED",
            timestamp_utc="2026-10-27T00:00:00Z",
            agents={"mammo": mammo},
            defer_to_human=True,
            deferral_reasons=["PRIMARY_AGENT_FAILED"]
        )

        # Mock LLM tries to clear deferral and claim normal diagnosis
        mock_resp = json.dumps({
            "consensus_bi_rads": "2",
            "malignancy_probability": 0.05,
            "confidence_interval": [0.01, 0.1],
            "defer_to_human": False,
            "deferral_reasons": [],
            "evidence_rationale": "Overriding failure: patient looks benign."
        })

        client = MockLLMClient(response_generator=mock_resp)
        supervisor = LLMSupervisor(client=client)
        decision = supervisor.deliberate(state)

        # Safety Guardrail: Deferral must remain TRUE
        self.assertTrue(decision.defer_to_human)
        self.assertIn("PRIMARY_AGENT_FAILED", decision.deferral_reasons)

    def test_hallucination_attempt_rejected(self):
        mammo = _make_dummy_mammo_payload(status="success", bi_rads="4C", p_malignancy=0.8123)
        state = SupervisorState(
            study_id="STUDY_HALLUCINATION_TEST",
            timestamp_utc="2026-10-27T00:00:00Z",
            agents={"mammo": mammo},
            defer_to_human=False
        )

        # Mock LLM attempts to fabricate a lower BI-RADS (e.g. 2 instead of 4C)
        mock_resp = json.dumps({
            "consensus_bi_rads": "2",
            "malignancy_probability": 0.05,
            "confidence_interval": [0.01, 0.1],
            "defer_to_human": False,
            "deferral_reasons": [],
            "evidence_rationale": "I think this is benign [card:mammo/BI_RADS_a]."
        })

        client = MockLLMClient(response_generator=mock_resp)
        supervisor = LLMSupervisor(client=client)
        decision = supervisor.deliberate(state)

        # Safety Guardrail: must enforce authoritative BI-RADS and force human deferral due to violation
        self.assertTrue(decision.defer_to_human)
        self.assertEqual(decision.consensus_bi_rads, "4C")
        self.assertTrue(any("Diagnostic authority violation" in r for r in decision.deferral_reasons))
        self.assertEqual(decision.audit_metadata["guardrail_status"], "failed_deferred")

    def test_pathology_fabrication_rejected(self):
        mammo = _make_dummy_mammo_payload()
        state = SupervisorState(
            study_id="STUDY_PATH_FABRICATION",
            timestamp_utc="2026-10-27T00:00:00Z",
            agents={"mammo": mammo},
            defer_to_human=False
        )

        # Mock LLM tries to cite unimplemented pathology agent
        mock_resp = json.dumps({
            "consensus_bi_rads": "4C",
            "malignancy_probability": 0.8123,
            "confidence_interval": [0.76, 0.86],
            "defer_to_human": False,
            "deferral_reasons": [],
            "evidence_rationale": (
                "Mass confirmed [card:mammo/BI_RADS_a]. "
                "Pathology confirms invasive ductal carcinoma [card:path/wsi_patch_malignancy]."
            )
        })

        client = MockLLMClient(response_generator=mock_resp)
        supervisor = LLMSupervisor(client=client)
        decision = supervisor.deliberate(state)

        # Safety Guardrail: Pathology citation is strictly prohibited
        self.assertTrue(decision.defer_to_human)
        self.assertTrue(any("Fabrication violation" in r for r in decision.deferral_reasons))

    def test_narrative_birads_contradiction_caught(self):
        # Case where structured BI-RADS is '4', but LLM text claims 'BI-RADS 4C'
        mammo = _make_dummy_mammo_payload(status="success", bi_rads="4", p_malignancy=0.4874)
        state = SupervisorState(
            study_id="STUDY_CONTRADICTION_TEST",
            timestamp_utc="2026-10-27T00:00:00Z",
            agents={"mammo": mammo},
            defer_to_human=False
        )

        mock_resp = json.dumps({
            "consensus_bi_rads": "4",
            "malignancy_probability": 0.4874,
            "confidence_interval": [0.4374, 0.5374],
            "defer_to_human": False,
            "deferral_reasons": [],
            "evidence_rationale": (
                "Two-view mammography demonstrates a spiculated mass [card:mammo/finding/margin] "
                "rated BI-RADS 4C [card:mammo/BI_RADS_a]."  # In-prose contradiction!
            )
        })

        client = MockLLMClient(response_generator=mock_resp)
        supervisor = LLMSupervisor(client=client)
        decision = supervisor.deliberate(state)

        # Safety Guardrail: must catch narrative contradiction and force human deferral
        self.assertTrue(decision.defer_to_human)
        self.assertEqual(decision.consensus_bi_rads, "4")
        self.assertTrue(any("Narrative factual inconsistency" in r for r in decision.deferral_reasons))
        self.assertEqual(decision.audit_metadata["guardrail_status"], "failed_deferred")

    def test_shap_citation_mismatch_caught(self):
        # Case where rationale discusses TreeSHAP, but cites [card:clin/finding/risk_profile] instead of [card:clin/sh/...]
        mammo = _make_dummy_mammo_payload(status="success", bi_rads="4", p_malignancy=0.4329)
        clin = _make_dummy_clin_payload(status="success")
        state = SupervisorState(
            study_id="STUDY_SHAP_MISMATCH_TEST",
            timestamp_utc="2026-10-27T00:00:00Z",
            agents={"mammo": mammo, "clin": clin},
            defer_to_human=False
        )

        mock_resp = json.dumps({
            "consensus_bi_rads": "4",
            "malignancy_probability": 0.4329,
            "confidence_interval": [0.35, 0.52],
            "defer_to_human": False,
            "deferral_reasons": [],
            "evidence_rationale": (
                "Mammography reveals lesion [card:mammo/finding/margin] rated BI-RADS 4 [card:mammo/BI_RADS_a]. "
                "TreeSHAP feature contributions indicate elevated risk due to density [card:clin/finding/risk_profile]."  # Mismatched citation!
            )
        })

        client = MockLLMClient(response_generator=mock_resp)
        supervisor = LLMSupervisor(client=client)
        decision = supervisor.deliberate(state)

        # Safety Guardrail: must catch the citation mismatch and force human deferral
        self.assertTrue(decision.defer_to_human)
        self.assertTrue(any("Citation-to-claim mismatch" in r for r in decision.deferral_reasons))
        self.assertEqual(decision.audit_metadata["guardrail_status"], "failed_deferred")

    def test_malformed_llm_json_fallback(self):
        mammo = _make_dummy_mammo_payload()
        state = SupervisorState(
            study_id="STUDY_MALFORMED_JSON",
            timestamp_utc="2026-10-27T00:00:00Z",
            agents={"mammo": mammo},
            defer_to_human=False
        )

        client = MockLLMClient(response_generator="<<< Not valid JSON output >>>")
        supervisor = LLMSupervisor(client=client)
        decision = supervisor.deliberate(state)

        self.assertTrue(decision.defer_to_human)
        self.assertIn("LLM_DECODING_ERROR", decision.deferral_reasons)
        self.assertEqual(decision.consensus_bi_rads, "4C")


class TestLiveOllamaQwenDeliberation(unittest.TestCase):
    """Integration test with live local Qwen 2.5 3B via Ollama."""

    def test_live_qwen25_deliberation(self):
        # Verify Ollama is reachable
        try:
            req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status != 200:
                    self.skipTest("Ollama server returned non-200.")
        except Exception:
            self.skipTest("Ollama server not reachable at 127.0.0.1:11434.")

        mammo = _make_dummy_mammo_payload(
            status="success",
            bi_rads="4C",
            confidence=0.88,
            p_malignancy=0.8412
        )
        clin = _make_dummy_clin_payload(
            status="success",
            p_malignancy=0.35
        )
        state = SupervisorState(
            study_id="LIVE_STUDY_P00004",
            timestamp_utc="2026-10-27T12:00:00Z",
            agents={"mammo": mammo, "clin": clin},
            defer_to_human=False
        )

        supervisor = LLMSupervisor(client=OllamaLLMClient(model_name="qwen2.5:3b"))
        decision = supervisor.deliberate(state)

        self.assertEqual(decision.case_id, "LIVE_STUDY_P00004")
        self.assertEqual(decision.consensus_bi_rads, "4C")
        self.assertIsNotNone(decision.evidence_rationale)
        self.assertIn("[card:mammo/", decision.evidence_rationale)
        self.assertEqual(decision.audit_metadata["supervisor_model"], "qwen2.5:3b")


if __name__ == "__main__":
    unittest.main()
