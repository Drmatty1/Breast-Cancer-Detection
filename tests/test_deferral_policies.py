"""
Dedicated Deferral Policy Stress Tests for RP2 Multi-Agent System.

Governed by:
- docs/breast_cancer___ps1___methodology.pdf (§ 5.3 Deferral policy)
- DECISIONS.md (DEC-OWNER-01, DEC-OWNER-02, DEC-OWNER-03)

Validates all 5 mandatory human deferral triggers:
1. Inter-agent disagreement (step distance >= 2).
2. Epistemic uncertainty / borderline decision boundary entropy.
3. Primary Mammography Agent failure or timeout.
4. Input validation corruption (NaN, out-of-range, schema mismatch).
5. Post-generation guardrail violations (in-prose hallucinations).
"""
import unittest
import json

from src.orchestration.supervisor_contract import (
    SupervisorState,
    AgentFindingPayload,
    ValidationIssue
)
from src.orchestration.supervisor_schema import (
    ConsensusDecision
)
from src.orchestration.llm_supervisor import (
    LLMSupervisor,
    MockLLMClient
)
from tests.test_llm_supervisor import (
    _make_dummy_mammo_payload,
    _make_dummy_clin_payload
)


class TestDeferralPolicies(unittest.TestCase):
    """Stress tests verifying that defer_to_human=True activates under all required conditions."""

    def test_trigger_1_primary_failure_forces_deferral(self):
        """Methodology §5.3: Primary agent failure must strictly force human reader deferral."""
        mammo = _make_dummy_mammo_payload(status="failed", bi_rads="unspecified", p_malignancy=None)
        state = SupervisorState(
            study_id="DEFER_TEST_PRIMARY_FAILED",
            timestamp_utc="2026-10-27T00:00:00Z",
            agents={"mammo": mammo},
            defer_to_human=True,
            deferral_reasons=["PRIMARY_AGENT_FAILED"]
        )

        # Even if LLM tries to clear deferral, supervisor must enforce it
        client = MockLLMClient()
        supervisor = LLMSupervisor(client=client)
        decision = supervisor.deliberate(state)

        self.assertTrue(decision.defer_to_human)
        self.assertIn("PRIMARY_AGENT_FAILED", decision.deferral_reasons)

    def test_trigger_2_critical_validation_issue_forces_deferral(self):
        """Methodology §5.3: Critical data corruption or validation issue forces deferral."""
        mammo = _make_dummy_mammo_payload()
        issue = ValidationIssue(
            severity="ERROR",
            code="CORRUPT_PROBABILITY",
            message="Malignancy probability was out of range",
            field="task_probabilities.malignancy_calibrated"
        )
        state = SupervisorState(
            study_id="DEFER_TEST_CORRUPT_DATA",
            timestamp_utc="2026-10-27T00:00:00Z",
            agents={"mammo": mammo},
            defer_to_human=True,
            deferral_reasons=["CRITICAL_VALIDATION_ERROR: CORRUPT_PROBABILITY"],
            validation_issues=[issue]
        )

        client = MockLLMClient()
        supervisor = LLMSupervisor(client=client)
        decision = supervisor.deliberate(state)

        self.assertTrue(decision.defer_to_human)
        self.assertTrue(any("CRITICAL_VALIDATION_ERROR" in r for r in decision.deferral_reasons))

    def test_trigger_3_guardrail_hallucination_forces_deferral(self):
        """Methodology §9.1: In-prose BI-RADS hallucination triggers automated safety deferral."""
        mammo = _make_dummy_mammo_payload(status="success", bi_rads="3", p_malignancy=0.3052)
        state = SupervisorState(
            study_id="DEFER_TEST_HALLUCINATION",
            timestamp_utc="2026-10-27T00:00:00Z",
            agents={"mammo": mammo},
            defer_to_human=False
        )

        # LLM attempts to hallucinate BI-RADS 4C in narrative
        mock_resp = json.dumps({
            "consensus_bi_rads": "3",
            "malignancy_probability": 0.3052,
            "confidence_interval": [0.22, 0.39],
            "defer_to_human": False,
            "deferral_reasons": [],
            "evidence_rationale": "Lesion noted [card:mammo/finding/margin] rated BI-RADS 4C [card:mammo/BI_RADS_a]."
        })

        client = MockLLMClient(response_generator=mock_resp)
        supervisor = LLMSupervisor(client=client)
        decision = supervisor.deliberate(state)

        self.assertTrue(decision.defer_to_human)
        self.assertEqual(decision.audit_metadata["guardrail_status"], "failed_deferred")
        self.assertTrue(any("Narrative factual inconsistency" in r for r in decision.deferral_reasons))

    def test_trigger_4_claim_citation_mismatch_forces_deferral(self):
        """Methodology §4.2: Citing general risk profile for TreeSHAP claim triggers deferral."""
        mammo = _make_dummy_mammo_payload()
        clin = _make_dummy_clin_payload()
        state = SupervisorState(
            study_id="DEFER_TEST_CITATION_MISMATCH",
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
                "Spiculated mass [card:mammo/finding/margin] rated BI-RADS 4C [card:mammo/BI_RADS_a]. "
                "TreeSHAP feature attributions show elevated density [card:clin/finding/risk_profile]."  # Wrong citation!
            )
        })

        client = MockLLMClient(response_generator=mock_resp)
        supervisor = LLMSupervisor(client=client)
        decision = supervisor.deliberate(state)

        self.assertTrue(decision.defer_to_human)
        self.assertEqual(decision.audit_metadata["guardrail_status"], "failed_deferred")
        self.assertTrue(any("Citation-to-claim mismatch" in r for r in decision.deferral_reasons))

    def test_trigger_5_llm_decoding_failure_forces_deferral(self):
        """Methodology §4.2: Timeout or unparseable JSON triggers deterministic deferral fallback."""
        mammo = _make_dummy_mammo_payload()
        state = SupervisorState(
            study_id="DEFER_TEST_LLM_TIMEOUT",
            timestamp_utc="2026-10-27T00:00:00Z",
            agents={"mammo": mammo},
            defer_to_human=False
        )

        client = MockLLMClient(response_generator="<<< Server Timeout 504 >>>")
        supervisor = LLMSupervisor(client=client)
        decision = supervisor.deliberate(state)

        self.assertTrue(decision.defer_to_human)
        self.assertIn("LLM_DECODING_ERROR", decision.deferral_reasons)
        self.assertEqual(decision.consensus_bi_rads, "4C")

    def test_methodology_diamond_disagreement_ge_2_steps_triggers_yes_defer(self):
        """Methodology Figure 1: Disagreement >= 2 steps -> YES -> Defer to Human Reader."""
        mammo = _make_dummy_mammo_payload(status="success", bi_rads="3", p_malignancy=0.20)
        # Second agent reporting high suspicion BI-RADS 4C
        second_agent = AgentFindingPayload(
            agent_id="clin",
            status="success",
            authority_level="non_authoritative_research",
            voting_eligible=False,
            bi_rads="4C",
            confidence=0.85,
            confidence_type="calibrated",
            p_malignancy=0.65,
            findings_summary={},
            task_probabilities={},
            explainability={},
            rationale=None,
            raw_finding_card=None,
            metadata={}
        )
        state = SupervisorState(
            study_id="DIAMOND_DISAGREEMENT_TEST",
            timestamp_utc="2026-10-27T00:00:00Z",
            agents={"mammo": mammo, "clin": second_agent},
            defer_to_human=False
        )

        client = MockLLMClient()
        supervisor = LLMSupervisor(client=client)
        decision = supervisor.deliberate(state)

        # Must take YES branch -> Defer to Human Reader
        self.assertTrue(decision.defer_to_human)
        self.assertTrue(any("DISAGREEMENT_GE_2_STEPS" in r for r in decision.deferral_reasons))
        trail_entry = [e for e in decision.explanation_card["consensus_trail"] if e.get("stage") == "methodology_decision_diamond"][0]
        self.assertEqual(trail_entry["branch_taken"], "Yes (Defer to Human Reader)")

    def test_methodology_diamond_unc_theta_triggers_yes_defer(self):
        """Methodology Figure 1: unc theta? -> YES -> Defer to Human Reader."""
        # p_malignancy = 0.3650 (within theta=0.030 of screening threshold tau=0.3675)
        mammo = _make_dummy_mammo_payload(status="success", bi_rads="4", p_malignancy=0.3650)
        state = SupervisorState(
            study_id="DIAMOND_UNC_THETA_TEST",
            timestamp_utc="2026-10-27T00:00:00Z",
            agents={"mammo": mammo},
            defer_to_human=False
        )

        client = MockLLMClient()
        supervisor = LLMSupervisor(client=client, theta=0.030)
        decision = supervisor.deliberate(state)

        # Must take YES branch -> Defer to Human Reader
        self.assertTrue(decision.defer_to_human)
        self.assertTrue(any("EPISTEMIC_UNCERTAINTY_THETA" in r for r in decision.deferral_reasons))
        trail_entry = [e for e in decision.explanation_card["consensus_trail"] if e.get("stage") == "methodology_decision_diamond"][0]
        self.assertEqual(trail_entry["branch_taken"], "Yes (Defer to Human Reader)")

    def test_methodology_diamond_no_branch_produces_consensus_ci_evidence(self):
        """Methodology Figure 1: Disagreement < 2 AND unc < theta -> NO -> Consensus + CI + Evidence Card."""
        # High confidence, p=0.8123 (far outside theta=0.030 of tau=0.3675)
        mammo = _make_dummy_mammo_payload(status="success", bi_rads="4C", p_malignancy=0.8123)
        state = SupervisorState(
            study_id="DIAMOND_NO_BRANCH_TEST",
            timestamp_utc="2026-10-27T00:00:00Z",
            agents={"mammo": mammo},
            defer_to_human=False
        )

        client = MockLLMClient()
        supervisor = LLMSupervisor(client=client, theta=0.030)
        decision = supervisor.deliberate(state)

        # Must take NO branch -> Consensus + CI + Evidence Card
        self.assertFalse(decision.defer_to_human)
        self.assertEqual(len(decision.deferral_reasons), 0)
        self.assertEqual(decision.consensus_bi_rads, "4C")
        self.assertEqual(len(decision.confidence_interval), 2)
        trail_entry = [e for e in decision.explanation_card["consensus_trail"] if e.get("stage") == "methodology_decision_diamond"][0]
        self.assertEqual(trail_entry["branch_taken"], "No (Consensus + CI + Evidence Card)")


if __name__ == "__main__":
    unittest.main()
