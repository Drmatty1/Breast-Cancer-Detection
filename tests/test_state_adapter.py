"""
Unit tests for the Supervisor State Adapter module (Milestone 1).

Governed by:
- docs/breast_cancer___ps1___methodology.pdf (§§ 3.1, 3.4, 5.1, 5.3)
- DECISIONS.md (DEC-OWNER-01, DEC-OWNER-02, DEC-OWNER-03)
- reports/clin/CLIN_AGENT_C3_7_ORCHESTRATION_CONTRACT.md
- reports/rp2_experimental/agent_integration_audit.md
"""
import unittest
import json
import math
import copy
import numpy as np

from src.orchestration.state_adapter import (
    SupervisorState,
    AgentFindingPayload,
    ValidationIssue,
    StateValidationError,
    adapt_orchestration_response,
    adapt_mammo_finding_card,
    adapt_clin_finding_card,
    create_unavailable_pathology_payload,
    sanitize_for_json,
    PRIMARY_AGENT_ID,
    SHADOW_AGENT_ID,
    PATH_AGENT_ID
)
from src.orchestration.multi_agent_orchestrator import (
    OrchestrationResponse,
    OrchestrationAuditRecord
)


def _create_sample_mammo_card(
    birads: str = "4",
    conf_a: float = 0.85,
    p_mal: float = 0.72,
    with_secondary: bool = True,
    with_monotonicity: bool = True
) -> dict:
    """Helper to construct a valid Mammography finding card."""
    card = {
        "agent": "mammo",
        "finding": {
            "lesion": "malignant",
            "margin": "spiculated" if with_secondary else None,
            "calcification": "pleomorphic" if with_secondary else None
        },
        "BI_RADS_a": birads,
        "conf_a": conf_a,
        "sal": {
            "top_regions": [[100.0, 150.0, 300.0, 350.0]],
            "path": "/reports/mammo/saliency_overlay.png",
            "cc_path": "/reports/mammo/cc_overlay.png",
            "mlo_path": "/reports/mammo/mlo_overlay.png"
        },
        "sh": {
            "breast_density": 3,
            "breast_side": "LEFT"
        },
        "rationale": "Two-view mammography assessment indicates spiculated margin mass. Assigned BI-RADS category 4 with decision confidence 0.85.",
        "metadata": {
            "model_architecture": "tf_efficientnet_lite0_multitask",
            "model_version": "mammo_agent_v1.0",
            "checkpoint_source": "mammo_agent_ensemble",
            "timestamp": "2026-10-25T12:00:00Z",
            "disclaimer": "Research prototype output only. Not for clinical diagnostic use.",
            "task_probabilities": {
                "malignancy_calibrated": p_mal,
                "margin_distribution": [0.10, 0.15, 0.20, 0.55] if with_secondary else None,
                "calc_distribution": [0.05, 0.10, 0.15, 0.60, 0.10] if with_secondary else None
            },
            "confidence_provenance": {
                "conf_a_source": "heuristic_ordinal_cutoff_margin",
                "is_conf_a_calibrated": False,
                "malignancy_calibrated": True
            }
        }
    }
    if with_monotonicity:
        card["metadata"]["ordinal_monotonicity"] = {
            "was_violated": True,
            "projection_method": "pava",
            "raw_cutoff_probs": [0.65, 0.70, 0.40],
            "projected_cutoff_probs": [0.675, 0.675, 0.40],
            "category_probabilities": [0.325, 0.0, 0.275, 0.40]
        }
    return card


def _create_sample_clin_card(
    p_mal: float = 0.48,
    density: int = 3,
    side: str = "LEFT",
    voting_eligibility: bool = False
) -> dict:
    """Helper to construct a valid Clinical Risk finding card."""
    return {
        "agent": "clin",
        "finding": {
            "lesion": None,
            "margin": None,
            "calcification": None,
            "risk_profile": "malignancy_risk",
            "breast_density": density,
            "breast_side": side
        },
        "BI_RADS_a": "unspecified",
        "conf_a": 0.0,
        "sal": {
            "top_regions": [],
            "path": None,
            "cc_path": None,
            "mlo_path": None,
            "tree_shap": {
                "features": ["breast_density", "breast_side"],
                "contributions": [-0.0125, 0.0042],
                "bias": -0.0512
            }
        },
        "sh": {
            "breast_density": density,
            "breast_side": side
        },
        "rationale": "Tabular clinical model ensemble estimated calibrated malignancy risk of 0.48.",
        "metadata": {
            "model_architecture": "xgboost_tabular_logistic_ensemble",
            "model_version": "clin_agent_v1.0",
            "status": "success",
            "authority": "non_authoritative_research",
            "voting_eligibility": voting_eligibility,
            "task_probabilities": {
                "malignancy_calibrated": p_mal,
                "malignancy_uncalibrated": 0.49
            },
            "confidence_provenance": {
                "conf_a_source": "non_authoritative_zero_weight_convention",
                "is_conf_a_calibrated": False
            }
        }
    }


def _create_sample_audit(
    pair_id: str = "P_00008_LEFT",
    mammo_status: str = "SUCCESS",
    clin_status: str = "SUCCESS",
    clin_error_code: str = None
) -> OrchestrationAuditRecord:
    """Helper to construct an OrchestrationAuditRecord."""
    return OrchestrationAuditRecord(
        audit_id="audit-uuid-1234",
        pair_identifier=pair_id,
        timestamp_utc="2026-10-25T12:00:00Z",
        orchestrator_version="1.0.0",
        mammo_status=mammo_status,
        clin_status=clin_status,
        mammo_duration_ms=45.2,
        clin_duration_ms=8.1,
        schema_validation_passed=True,
        clin_error_code=clin_error_code,
        audit_persisted=True
    )


class TestSupervisorStateAdapter(unittest.TestCase):
    """Test suite for Milestone 1 Supervisor State Adapter."""

    def test_01_adapt_valid_orchestration_response(self):
        """Test full end-to-end adaptation with valid Mammo and Clin responses."""
        mammo_card = _create_sample_mammo_card()
        clin_card = _create_sample_clin_card()
        audit = _create_sample_audit()
        response = OrchestrationResponse(
            primary_finding_card=mammo_card,
            shadow_finding_card=clin_card,
            audit_record=audit
        )

        state = adapt_orchestration_response(response)

        self.assertIsInstance(state, SupervisorState)
        self.assertEqual(state.study_id, "P_00008_LEFT")
        self.assertEqual(state.timestamp_utc, "2026-10-25T12:00:00Z")
        self.assertEqual(state.primary_agent_id, PRIMARY_AGENT_ID)
        self.assertFalse(state.defer_to_human)
        self.assertEqual(len(state.deferral_reasons), 0)

        # Check Mammo payload
        mammo = state.agents[PRIMARY_AGENT_ID]
        self.assertEqual(mammo.agent_id, PRIMARY_AGENT_ID)
        self.assertEqual(mammo.status, "success")
        self.assertEqual(mammo.authority_level, "authoritative")
        self.assertTrue(mammo.voting_eligible)
        self.assertEqual(mammo.bi_rads, "4")
        self.assertAlmostEqual(mammo.confidence, 0.85)
        self.assertAlmostEqual(mammo.p_malignancy, 0.72)

        # Check Clin payload
        clin = state.agents[SHADOW_AGENT_ID]
        self.assertEqual(clin.agent_id, SHADOW_AGENT_ID)
        self.assertEqual(clin.status, "success")
        self.assertEqual(clin.authority_level, "non_authoritative_research")
        self.assertFalse(clin.voting_eligible)
        self.assertEqual(clin.bi_rads, "unspecified")
        self.assertEqual(clin.confidence, 0.0)
        self.assertAlmostEqual(clin.p_malignancy, 0.48)

        # Check Pathology payload
        path = state.agents[PATH_AGENT_ID]
        self.assertEqual(path.agent_id, PATH_AGENT_ID)
        self.assertEqual(path.status, "unavailable")
        self.assertEqual(path.authority_level, "non_authoritative_research")
        self.assertFalse(path.voting_eligible)
        self.assertIsNone(path.p_malignancy)

    def test_02_mammo_finding_card_preservation_and_immutability(self):
        """Test exact preservation and deepcopy immutability of raw finding cards."""
        mammo_card = _create_sample_mammo_card()
        clin_card = _create_sample_clin_card()
        audit = _create_sample_audit()
        response = OrchestrationResponse(
            primary_finding_card=mammo_card,
            shadow_finding_card=clin_card,
            audit_record=audit
        )

        state = adapt_orchestration_response(response)

        # Confirm exact content equality initially
        self.assertEqual(state.agents[PRIMARY_AGENT_ID].raw_finding_card, mammo_card)

        # Mutate the original finding card in response
        mammo_card["conf_a"] = 0.9999
        mammo_card["metadata"]["mutated_key"] = "leak"

        # Assert state copy is completely protected from mutation
        self.assertNotEqual(state.agents[PRIMARY_AGENT_ID].raw_finding_card["conf_a"], 0.9999)
        self.assertNotIn("mutated_key", state.agents[PRIMARY_AGENT_ID].raw_finding_card["metadata"])

        # Mutate the state's raw_finding_card
        state.agents[PRIMARY_AGENT_ID].raw_finding_card["BI_RADS_a"] = "5"
        self.assertNotEqual(mammo_card["BI_RADS_a"], "5")

    def test_03_binary_calibration_metadata_preservation(self):
        """Test preservation of binary calibration flags and confidence provenance."""
        mammo_card = _create_sample_mammo_card()
        mammo_card["metadata"]["confidence_provenance"] = {
            "conf_a_source": "calibrated_binary_malignancy",
            "is_conf_a_calibrated": True,
            "malignancy_calibrated": True
        }
        payload, issues = adapt_mammo_finding_card(mammo_card, "SUCCESS")
        self.assertEqual(payload.confidence_type, "calibrated_heuristic")
        self.assertTrue(payload.metadata["confidence_provenance"]["malignancy_calibrated"])

    def test_04_mass_margin_and_calc_distribution_preservation(self):
        """Test extraction and validation of multi-class secondary distributions."""
        mammo_card = _create_sample_mammo_card(with_secondary=True)
        payload, issues = adapt_mammo_finding_card(mammo_card, "SUCCESS")

        self.assertIn("margin_distribution", payload.task_probabilities)
        self.assertIn("calc_distribution", payload.task_probabilities)
        self.assertEqual(len(payload.task_probabilities["margin_distribution"]), 4)
        self.assertEqual(len(payload.task_probabilities["calc_distribution"]), 5)
        self.assertEqual(payload.findings_summary["margin"], "spiculated")
        self.assertEqual(payload.findings_summary["calcification"], "pleomorphic")

    def test_05_birads_monotonicity_pava_preservation(self):
        """Test preservation and validation of PAVA monotonic ordinal projection."""
        mammo_card = _create_sample_mammo_card(with_monotonicity=True)
        mammo_card["metadata"]["ordinal_monotonicity"] = {
            "was_violated": True,
            "projection_method": "pava",
            "raw_cutoff_probs": [0.70, 0.75, 0.40],
            "projected_cutoff_probs": [0.725, 0.725, 0.40],
            "category_probabilities": [0.275, 0.0, 0.325, 0.40]
        }
        payload, issues = adapt_mammo_finding_card(mammo_card, "SUCCESS")

        mono = payload.metadata["ordinal_monotonicity"]
        self.assertTrue(mono["was_violated"])
        self.assertEqual(mono["projection_method"], "pava")
        self.assertEqual(mono["projected_cutoff_probs"], [0.725, 0.725, 0.40])
        self.assertFalse(any(issue.severity == "ERROR" for issue in issues))

    def test_06_birads_monotonicity_sequential_upper_bound_preservation(self):
        """Test preservation and validation of sequential_upper_bound projection."""
        mammo_card = _create_sample_mammo_card(with_monotonicity=True)
        mammo_card["metadata"]["ordinal_monotonicity"] = {
            "was_violated": True,
            "projection_method": "sequential_upper_bound",
            "raw_cutoff_probs": [0.70, 0.75, 0.40],
            "projected_cutoff_probs": [0.70, 0.70, 0.40],
            "category_probabilities": [0.30, 0.0, 0.30, 0.40]
        }
        payload, issues = adapt_mammo_finding_card(mammo_card, "SUCCESS")

        mono = payload.metadata["ordinal_monotonicity"]
        self.assertEqual(mono["projection_method"], "sequential_upper_bound")
        self.assertEqual(mono["projected_cutoff_probs"], [0.70, 0.70, 0.40])
        self.assertFalse(any(issue.severity == "ERROR" for issue in issues))

    def test_07_clin_voting_eligibility_enforcement(self):
        """Test that Clin voting_eligibility=True is caught as an ERROR and forced to False."""
        clin_card = _create_sample_clin_card(voting_eligibility=True)
        payload, issues = adapt_clin_finding_card(clin_card, "SUCCESS")

        # Must strictly force voting_eligible to False
        self.assertFalse(payload.voting_eligible)

        # Must record an ERROR issue for governance breach attempt
        error_issues = [i for i in issues if i.code == "UNAUTHORIZED_VOTING_ELIGIBILITY"]
        self.assertEqual(len(error_issues), 1)
        self.assertEqual(error_issues[0].severity, "ERROR")

    def test_08_missing_clin_does_not_trigger_human_review(self):
        """Test that a missing, skipped, or failed Clin Agent never triggers human deferral."""
        mammo_card = _create_sample_mammo_card()
        audit = _create_sample_audit(clin_status="SKIPPED")
        response = OrchestrationResponse(
            primary_finding_card=mammo_card,
            shadow_finding_card=None,
            audit_record=audit
        )

        state = adapt_orchestration_response(response)

        self.assertFalse(state.defer_to_human)
        self.assertEqual(len(state.deferral_reasons), 0)
        self.assertEqual(state.agents[SHADOW_AGENT_ID].status, "skipped")
        self.assertFalse(state.agents[SHADOW_AGENT_ID].voting_eligible)

    def test_09_missing_mammo_triggers_human_review(self):
        """Test that missing or failed Mammo Agent triggers deferral to human review."""
        clin_card = _create_sample_clin_card()
        audit = _create_sample_audit(mammo_status="FAILED")
        response = OrchestrationResponse(
            primary_finding_card=None,
            shadow_finding_card=clin_card,
            audit_record=audit
        )

        state = adapt_orchestration_response(response)

        self.assertTrue(state.defer_to_human)
        self.assertTrue(any("Mammography" in r for r in state.deferral_reasons))
        self.assertEqual(state.agents[PRIMARY_AGENT_ID].status, "failed")
        self.assertFalse(state.agents[PRIMARY_AGENT_ID].voting_eligible)

    def test_10_non_finite_probability_handling(self):
        """Test that NaN and Inf probabilities trigger validation ERROR and deferral."""
        # Case A: NaN confidence
        mammo_card = _create_sample_mammo_card()
        mammo_card["conf_a"] = float("nan")
        payload, issues = adapt_mammo_finding_card(mammo_card, "SUCCESS")
        self.assertTrue(any(i.code == "NON_FINITE_CONFIDENCE" for i in issues))
        self.assertEqual(payload.status, "failed")
        self.assertFalse(payload.voting_eligible)

        # Case B: Inf malignancy probability
        mammo_card2 = _create_sample_mammo_card()
        mammo_card2["metadata"]["task_probabilities"]["malignancy_calibrated"] = float("inf")
        payload2, issues2 = adapt_mammo_finding_card(mammo_card2, "SUCCESS")
        self.assertTrue(any(i.code == "INVALID_MALIGNANCY_PROBABILITY" for i in issues2))
        self.assertEqual(payload2.status, "failed")

    def test_11_out_of_range_probability_handling(self):
        """Test that probabilities outside [0.0, 1.0] trigger validation ERROR and deferral."""
        mammo_card = _create_sample_mammo_card()
        mammo_card["conf_a"] = 1.25
        payload, issues = adapt_mammo_finding_card(mammo_card, "SUCCESS")
        self.assertTrue(any(i.code == "OUT_OF_RANGE_CONFIDENCE" for i in issues))
        self.assertEqual(payload.status, "failed")

        mammo_card2 = _create_sample_mammo_card()
        mammo_card2["metadata"]["task_probabilities"]["malignancy_calibrated"] = -0.05
        payload2, issues2 = adapt_mammo_finding_card(mammo_card2, "SUCCESS")
        self.assertTrue(any(i.code == "INVALID_MALIGNANCY_PROBABILITY" for i in issues2))
        self.assertEqual(payload2.status, "failed")

    def test_12_invalid_birads_category(self):
        """Test that unsupported BI-RADS categories trigger validation ERROR and deferral."""
        mammo_card = _create_sample_mammo_card(birads="7")
        payload, issues = adapt_mammo_finding_card(mammo_card, "SUCCESS")
        self.assertTrue(any(i.code == "INVALID_BIRADS_CATEGORY" for i in issues))
        self.assertEqual(payload.status, "failed")
        self.assertFalse(payload.voting_eligible)

    def test_13_unavailable_pathology_agent(self):
        """Test that Pathology Agent payload is explicitly unavailable with zero clinical assumptions."""
        payload = create_unavailable_pathology_payload()
        self.assertEqual(payload.agent_id, PATH_AGENT_ID)
        self.assertEqual(payload.status, "unavailable")
        self.assertEqual(payload.authority_level, "non_authoritative_research")
        self.assertFalse(payload.voting_eligible)
        self.assertEqual(payload.bi_rads, "unspecified")
        self.assertEqual(payload.confidence, 0.0)
        self.assertIsNone(payload.p_malignancy)
        self.assertIn("unavailable", payload.findings_summary["status"].lower())
        self.assertTrue(payload.metadata["deferred"])
        self.assertIsNone(payload.raw_finding_card)

    def test_14_agent_failures_handling(self):
        """Test distinction between primary agent failure and shadow agent failure."""
        # Primary failure
        audit_prim_fail = _create_sample_audit(mammo_status="FAILED", clin_status="SUCCESS")
        resp_prim = OrchestrationResponse(
            primary_finding_card=None,
            shadow_finding_card=_create_sample_clin_card(),
            audit_record=audit_prim_fail
        )
        state_prim = adapt_orchestration_response(resp_prim)
        self.assertTrue(state_prim.defer_to_human)

        # Shadow failure
        audit_shad_fail = _create_sample_audit(mammo_status="SUCCESS", clin_status="FAILED")
        resp_shad = OrchestrationResponse(
            primary_finding_card=_create_sample_mammo_card(),
            shadow_finding_card=None,
            audit_record=audit_shad_fail
        )
        state_shad = adapt_orchestration_response(resp_shad)
        self.assertFalse(state_shad.defer_to_human)
        self.assertEqual(state_shad.agents[SHADOW_AGENT_ID].status, "failed")

    def test_15_agent_abstentions_handling(self):
        """Test that Clin Agent abstention is properly tracked and does not trigger deferral."""
        clin_card = _create_sample_clin_card()
        clin_card["metadata"]["status"] = "abstained"
        clin_card["metadata"]["rejection_reason"] = "missing_density"
        audit = _create_sample_audit(clin_status="ABSTAINED")
        resp = OrchestrationResponse(
            primary_finding_card=_create_sample_mammo_card(),
            shadow_finding_card=clin_card,
            audit_record=audit
        )
        state = adapt_orchestration_response(resp)
        self.assertFalse(state.defer_to_human)
        self.assertEqual(state.agents[SHADOW_AGENT_ID].status, "abstained")
        self.assertEqual(state.agents[SHADOW_AGENT_ID].findings_summary["rejection_reason"], "missing_density")

    def test_16_audit_record_preservation(self):
        """Test that all audit record provenance fields are preserved in SupervisorState."""
        audit = _create_sample_audit(pair_id="TEST_PAIR_42", mammo_status="SUCCESS", clin_status="SUCCESS")
        resp = OrchestrationResponse(
            primary_finding_card=_create_sample_mammo_card(),
            shadow_finding_card=_create_sample_clin_card(),
            audit_record=audit
        )
        state = adapt_orchestration_response(resp)
        self.assertIn("orchestration_audit", state.to_dict())
        audit_dict = state.orchestration_audit
        self.assertEqual(audit_dict["audit_id"], "audit-uuid-1234")
        self.assertEqual(audit_dict["pair_identifier"], "TEST_PAIR_42")
        self.assertEqual(audit_dict["mammo_status"], "SUCCESS")
        self.assertEqual(audit_dict["clin_status"], "SUCCESS")
        self.assertAlmostEqual(audit_dict["mammo_duration_ms"], 45.2)

    def test_17_clean_json_serialization_and_roundtrip(self):
        """Test that SupervisorState serializes to valid RFC 8259 JSON and reconstructs via from_dict."""
        resp = OrchestrationResponse(
            primary_finding_card=_create_sample_mammo_card(),
            shadow_finding_card=_create_sample_clin_card(),
            audit_record=_create_sample_audit()
        )
        state = adapt_orchestration_response(resp)

        # Serialize to JSON
        json_str = state.to_json(indent=2)
        self.assertIsInstance(json_str, str)

        # Parse with standard library
        parsed_dict = json.loads(json_str)
        self.assertEqual(parsed_dict["study_id"], "P_00008_LEFT")
        self.assertEqual(parsed_dict["primary_agent_id"], PRIMARY_AGENT_ID)

        # Reconstruct state from dictionary
        reconstructed = SupervisorState.from_dict(parsed_dict)
        self.assertEqual(reconstructed.study_id, state.study_id)
        self.assertEqual(reconstructed.defer_to_human, state.defer_to_human)
        self.assertEqual(reconstructed.agents[PRIMARY_AGENT_ID].bi_rads, state.agents[PRIMARY_AGENT_ID].bi_rads)
        self.assertEqual(reconstructed.agents[SHADOW_AGENT_ID].status, state.agents[SHADOW_AGENT_ID].status)
        self.assertEqual(reconstructed.agents[PATH_AGENT_ID].status, "unavailable")

    def test_18_numpy_and_non_serializable_sanitization(self):
        """Test that NumPy scalars and arrays are safely sanitized without raising TypeError."""
        mammo_card = _create_sample_mammo_card()
        # Inject numpy scalars and arrays into metadata
        mammo_card["metadata"]["np_float"] = np.float32(0.854)
        mammo_card["metadata"]["np_int"] = np.int64(42)
        mammo_card["metadata"]["np_array"] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mammo_card["metadata"]["np_bool"] = np.bool_(True)

        resp = OrchestrationResponse(
            primary_finding_card=mammo_card,
            shadow_finding_card=_create_sample_clin_card(),
            audit_record=_create_sample_audit()
        )
        state = adapt_orchestration_response(resp)

        # to_json must succeed cleanly
        json_str = state.to_json()
        parsed = json.loads(json_str)
        self.assertAlmostEqual(parsed["agents"]["mammo"]["metadata"]["np_float"], 0.854, places=3)
        self.assertEqual(parsed["agents"]["mammo"]["metadata"]["np_int"], 42)
        self.assertEqual(parsed["agents"]["mammo"]["metadata"]["np_array"], [1.0, 2.0, 3.0])
        self.assertTrue(parsed["agents"]["mammo"]["metadata"]["np_bool"])

    def test_19_deserialization_missing_required_fields(self):
        """Test that missing required fields in state or payload raise StateValidationError."""
        # 1. Missing study_id in SupervisorState
        invalid_state_dict = {
            "timestamp_utc": "2026-10-25T12:00:00Z",
            "primary_agent_id": PRIMARY_AGENT_ID,
            "defer_to_human": False,
            "agents": {}
        }
        with self.assertRaises(StateValidationError) as ctx:
            SupervisorState.from_dict(invalid_state_dict)
        self.assertIn("Missing required field 'study_id'", str(ctx.exception))

        # 2. Missing primary agent in agents dict
        invalid_state_dict2 = {
            "study_id": "P_00008_LEFT",
            "timestamp_utc": "2026-10-25T12:00:00Z",
            "primary_agent_id": PRIMARY_AGENT_ID,
            "defer_to_human": False,
            "agents": {}
        }
        with self.assertRaises(StateValidationError) as ctx2:
            SupervisorState.from_dict(invalid_state_dict2)
        self.assertIn("must contain primary agent 'mammo'", str(ctx2.exception))

        # 3. Missing required field 'status' in AgentFindingPayload
        invalid_payload_dict = {
            "agent_id": "mammo",
            "authority_level": "authoritative",
            "voting_eligible": True,
            "bi_rads": "4",
            "confidence": 0.85,
            "confidence_type": "calibrated_heuristic"
        }
        with self.assertRaises(StateValidationError) as ctx3:
            AgentFindingPayload.from_dict(invalid_payload_dict)
        self.assertIn("Missing required field 'status'", str(ctx3.exception))

    def test_20_deserialization_incorrect_field_types(self):
        """Test that incorrect field types raise StateValidationError."""
        # Incorrect type for confidence (string instead of float)
        payload_dict = {
            "agent_id": "mammo",
            "status": "success",
            "authority_level": "authoritative",
            "voting_eligible": True,
            "bi_rads": "4",
            "confidence": "very_confident",
            "confidence_type": "heuristic"
        }
        with self.assertRaises(StateValidationError) as ctx:
            AgentFindingPayload.from_dict(payload_dict)
        self.assertIn("must be a finite float", str(ctx.exception))

        # Incorrect type for agents (list instead of dict)
        state_dict = {
            "study_id": "P_00008_LEFT",
            "timestamp_utc": "2026-10-25T12:00:00Z",
            "primary_agent_id": PRIMARY_AGENT_ID,
            "defer_to_human": False,
            "agents": []
        }
        with self.assertRaises(StateValidationError) as ctx2:
            SupervisorState.from_dict(state_dict)
        self.assertIn("agents must be a dict", str(ctx2.exception))

    def test_21_strict_boolean_validation_no_string_coercion(self):
        """Test that string values 'false' and 'true' are strictly rejected without coercion."""
        # String 'false' passed for voting_eligible
        payload_dict = {
            "agent_id": "clin",
            "status": "success",
            "authority_level": "non_authoritative_research",
            "voting_eligible": "false",
            "bi_rads": "unspecified",
            "confidence": 0.0,
            "confidence_type": "non_authoritative_zero_weight"
        }
        with self.assertRaises(StateValidationError) as ctx:
            AgentFindingPayload.from_dict(payload_dict)
        self.assertIn("voting_eligible must be a strict boolean", str(ctx.exception))

        # String 'false' passed for defer_to_human
        mammo_payload = {
            "agent_id": "mammo",
            "status": "success",
            "authority_level": "authoritative",
            "voting_eligible": True,
            "bi_rads": "4",
            "confidence": 0.85,
            "confidence_type": "calibrated_heuristic"
        }
        state_dict = {
            "study_id": "P_00008_LEFT",
            "timestamp_utc": "2026-10-25T12:00:00Z",
            "primary_agent_id": PRIMARY_AGENT_ID,
            "defer_to_human": "false",
            "agents": {"mammo": mammo_payload}
        }
        with self.assertRaises(StateValidationError) as ctx2:
            SupervisorState.from_dict(state_dict)
        self.assertIn("defer_to_human must be a strict boolean", str(ctx2.exception))

    def test_22_non_finite_and_out_of_range_deserialization(self):
        """Test that NaN, Inf, and out-of-range numbers raise StateValidationError on deserialization."""
        # NaN confidence
        payload_nan = {
            "agent_id": "mammo",
            "status": "success",
            "authority_level": "authoritative",
            "voting_eligible": True,
            "bi_rads": "4",
            "confidence": float("nan"),
            "confidence_type": "calibrated_heuristic"
        }
        with self.assertRaises(StateValidationError) as ctx:
            AgentFindingPayload.from_dict(payload_nan)
        self.assertIn("must be finite", str(ctx.exception))

        # Out-of-range confidence (>1.0)
        payload_oor = copy.deepcopy(payload_nan)
        payload_oor["confidence"] = 1.05
        with self.assertRaises(StateValidationError) as ctx2:
            AgentFindingPayload.from_dict(payload_oor)
        self.assertIn("must be in [0.0, 1.0]", str(ctx2.exception))

        # Out-of-range p_malignancy (<0.0)
        payload_pmal = copy.deepcopy(payload_nan)
        payload_pmal["confidence"] = 0.85
        payload_pmal["p_malignancy"] = -0.05
        with self.assertRaises(StateValidationError) as ctx3:
            AgentFindingPayload.from_dict(payload_pmal)
        self.assertIn("p_malignancy for agent 'mammo' must be in [0.0, 1.0]", str(ctx3.exception))

    def test_23_unauthorized_clin_voting_rejected_in_deserialization(self):
        """Test that Clin or Path payloads with voting_eligible=True raise StateValidationError."""
        clin_illegal = {
            "agent_id": "clin",
            "status": "success",
            "authority_level": "non_authoritative_research",
            "voting_eligible": True,
            "bi_rads": "unspecified",
            "confidence": 0.0,
            "confidence_type": "non_authoritative_zero_weight"
        }
        with self.assertRaises(StateValidationError) as ctx:
            AgentFindingPayload.from_dict(clin_illegal)
        self.assertIn("Governance breach: Agent 'clin' cannot be voting_eligible", str(ctx.exception))

        path_illegal = copy.deepcopy(clin_illegal)
        path_illegal["agent_id"] = "path"
        with self.assertRaises(StateValidationError) as ctx2:
            AgentFindingPayload.from_dict(path_illegal)
        self.assertIn("Governance breach: Agent 'path' cannot be voting_eligible", str(ctx2.exception))

    def test_24_inconsistent_deferral_state_rejected(self):
        """Test that inconsistent states (e.g. failed Mammo but defer_to_human=False) are rejected."""
        mammo_failed = {
            "agent_id": "mammo",
            "status": "failed",
            "authority_level": "authoritative",
            "voting_eligible": False,
            "bi_rads": "unspecified",
            "confidence": 0.0,
            "confidence_type": "none"
        }
        state_dict = {
            "study_id": "P_00008_LEFT",
            "timestamp_utc": "2026-10-25T12:00:00Z",
            "primary_agent_id": PRIMARY_AGENT_ID,
            "defer_to_human": False,  # Contradicts failed primary agent
            "agents": {"mammo": mammo_failed}
        }
        with self.assertRaises(StateValidationError) as ctx:
            SupervisorState.from_dict(state_dict)
        self.assertIn("Safety invariant violation", str(ctx.exception))

    def test_25_remote_contract_import_isolation(self):
        """Test that supervisor_contract can be imported and executed with zero ML runtime dependencies."""
        import sys
        # Temporarily mock absent ML dependencies
        orig_torch = sys.modules.get("torch")
        orig_xgboost = sys.modules.get("xgboost")
        try:
            sys.modules["torch"] = None
            sys.modules["xgboost"] = None

            from src.orchestration.supervisor_contract import (
                SupervisorState as RemoteSupervisorState,
                AgentFindingPayload as RemotePayload,
                ValidationIssue as RemoteIssue
            )

            # Construct valid payload purely with standard types
            p = RemotePayload(
                agent_id="mammo",
                status="success",
                authority_level="authoritative",
                voting_eligible=True,
                bi_rads="4",
                confidence=0.88,
                confidence_type="calibrated_heuristic",
                p_malignancy=0.75,
                findings_summary={"lesion": "malignant"},
                task_probabilities={"malignancy_calibrated": 0.75},
                explainability={},
                rationale="Two-view assessment.",
                raw_finding_card=None,
                metadata={}
            )
            s = RemoteSupervisorState(
                study_id="ISOLATED_TEST_001",
                timestamp_utc="2026-10-25T12:00:00Z",
                agents={"mammo": p},
                defer_to_human=False
            )
            json_str = s.to_json()
            self.assertIn("ISOLATED_TEST_001", json_str)
            reconstructed = RemoteSupervisorState.from_dict(json.loads(json_str))
            self.assertEqual(reconstructed.study_id, "ISOLATED_TEST_001")
            self.assertTrue(reconstructed.agents["mammo"].voting_eligible)
        finally:
            if orig_torch is not None:
                sys.modules["torch"] = orig_torch
            else:
                sys.modules.pop("torch", None)
            if orig_xgboost is not None:
                sys.modules["xgboost"] = orig_xgboost
            else:
                sys.modules.pop("xgboost", None)

    def test_26_illustrative_example_payload_handling(self):
        """Test that synthetic illustrative reference payloads are properly processed and labeled."""
        synthetic_payload = {
            "study_id": "SYNTHETIC_REFERENCE_EXAMPLE_001",
            "timestamp_utc": "2026-10-25T12:00:00Z",
            "primary_agent_id": "mammo",
            "defer_to_human": False,
            "deferral_reasons": [],
            "validation_issues": [],
            "agents": {
                "mammo": {
                    "agent_id": "mammo",
                    "status": "success",
                    "authority_level": "authoritative",
                    "voting_eligible": True,
                    "bi_rads": "4",
                    "confidence": 0.85,
                    "confidence_type": "calibrated_heuristic",
                    "p_malignancy": 0.72,
                    "findings_summary": {"lesion": "malignant", "margin": "spiculated"},
                    "task_probabilities": {"malignancy_calibrated": 0.72},
                    "explainability": {},
                    "rationale": "Illustrative synthetic rationale.",
                    "raw_finding_card": None,
                    "metadata": {"provenance": "synthetic_reference_example"}
                },
                "clin": {
                    "agent_id": "clin",
                    "status": "success",
                    "authority_level": "non_authoritative_research",
                    "voting_eligible": False,
                    "bi_rads": "unspecified",
                    "confidence": 0.0,
                    "confidence_type": "non_authoritative_zero_weight",
                    "p_malignancy": 0.48,
                    "findings_summary": {"breast_density": 3, "breast_side": "LEFT"},
                    "task_probabilities": {"malignancy_calibrated": 0.48},
                    "explainability": {},
                    "rationale": "Illustrative clinical shadow rationale.",
                    "raw_finding_card": None,
                    "metadata": {"provenance": "synthetic_reference_example"}
                },
                "path": {
                    "agent_id": "path",
                    "status": "unavailable",
                    "authority_level": "non_authoritative_research",
                    "voting_eligible": False,
                    "bi_rads": "unspecified",
                    "confidence": 0.0,
                    "confidence_type": "unavailable",
                    "p_malignancy": None,
                    "findings_summary": {"status": "unavailable"},
                    "task_probabilities": {},
                    "explainability": {},
                    "rationale": "Pathology Agent deferred.",
                    "raw_finding_card": None,
                    "metadata": {"deferred": True}
                }
            },
            "orchestration_audit": {
                "audit_id": "synthetic-audit-uuid",
                "pair_identifier": "SYNTHETIC_REFERENCE_EXAMPLE_001",
                "timestamp_utc": "2026-10-25T12:00:00Z",
                "orchestrator_version": "1.0.0",
                "mammo_status": "SUCCESS",
                "clin_status": "SUCCESS",
                "mammo_duration_ms": 40.0,
                "clin_duration_ms": 5.0,
                "schema_validation_passed": True
            }
        }
        state = SupervisorState.from_dict(synthetic_payload)
        self.assertEqual(state.study_id, "SYNTHETIC_REFERENCE_EXAMPLE_001")
        self.assertEqual(state.agents["mammo"].metadata.get("provenance"), "synthetic_reference_example")
        self.assertFalse(state.defer_to_human)

    def test_27_optional_fields_legitimately_absent(self):
        """Test that legitimately absent optional distributions do NOT trigger errors or deferral."""
        mammo_card = _create_sample_mammo_card(with_secondary=False, with_monotonicity=False)
        # Ensure optional heads are None
        self.assertIsNone(mammo_card["metadata"]["task_probabilities"]["margin_distribution"])
        self.assertIsNone(mammo_card["metadata"]["task_probabilities"]["calc_distribution"])
        self.assertNotIn("ordinal_monotonicity", mammo_card["metadata"])

        payload, issues = adapt_mammo_finding_card(mammo_card, "SUCCESS")

        self.assertEqual(payload.status, "success")
        self.assertTrue(payload.voting_eligible)
        # Verify no ERROR issues were recorded for missing optional fields
        error_issues = [i for i in issues if i.severity == "ERROR"]
        self.assertEqual(len(error_issues), 0)

    def test_28_missing_mandatory_malignancy_triggers_deferral(self):
        """Test that missing mandatory malignancy probability triggers validation ERROR and deferral."""
        mammo_card = _create_sample_mammo_card()
        # Delete required head
        del mammo_card["metadata"]["task_probabilities"]["malignancy_calibrated"]

        payload, issues = adapt_mammo_finding_card(mammo_card, "SUCCESS")

        self.assertEqual(payload.status, "failed")
        self.assertFalse(payload.voting_eligible)
        missing_issues = [i for i in issues if i.code == "MISSING_MALIGNANCY_PROBABILITY"]
        self.assertEqual(len(missing_issues), 1)
        self.assertEqual(missing_issues[0].severity, "ERROR")

    def test_29_successful_voting_mammo_missing_p_malignancy_rejected(self):
        """Test that a successful voting-eligible Mammo payload missing p_malignancy raises StateValidationError."""
        payload_dict = {
            "agent_id": "mammo",
            "status": "success",
            "authority_level": "authoritative",
            "voting_eligible": True,
            "bi_rads": "4",
            "confidence": 0.85,
            "confidence_type": "calibrated_heuristic",
            "p_malignancy": None,  # Missing mandatory prediction on success
            "task_probabilities": {"malignancy_calibrated": 0.72}
        }
        with self.assertRaises(StateValidationError) as ctx:
            AgentFindingPayload.from_dict(payload_dict)
        self.assertIn("requires valid 'p_malignancy'", str(ctx.exception))

    def test_30_invalid_nested_scalar_probabilities(self):
        """Test that invalid nested scalar probabilities raise StateValidationError."""
        # Case A: Out-of-bounds malignancy_calibrated (< 0.0)
        payload_dict = {
            "agent_id": "mammo",
            "status": "success",
            "authority_level": "authoritative",
            "voting_eligible": True,
            "bi_rads": "4",
            "confidence": 0.85,
            "confidence_type": "calibrated_heuristic",
            "p_malignancy": 0.72,
            "task_probabilities": {"malignancy_calibrated": -0.2}
        }
        with self.assertRaises(StateValidationError) as ctx:
            AgentFindingPayload.from_dict(payload_dict)
        self.assertIn("must be in [0.0, 1.0]", str(ctx.exception))

        # Case B: Inconsistent malignancy probabilities (p_malignancy != task_probabilities.malignancy_calibrated)
        payload_inconsistent = copy.deepcopy(payload_dict)
        payload_inconsistent["p_malignancy"] = 0.50
        payload_inconsistent["task_probabilities"]["malignancy_calibrated"] = 0.90
        with self.assertRaises(StateValidationError) as ctx2:
            AgentFindingPayload.from_dict(payload_inconsistent)
        self.assertIn("Inconsistent malignancy probability", str(ctx2.exception))

    def test_31_invalid_or_non_finite_elements_in_secondary_distributions(self):
        """Test that non-finite, out-of-range, or wrong-length secondary distributions raise StateValidationError."""
        base_payload = {
            "agent_id": "mammo",
            "status": "success",
            "authority_level": "authoritative",
            "voting_eligible": True,
            "bi_rads": "4",
            "confidence": 0.85,
            "confidence_type": "calibrated_heuristic",
            "p_malignancy": 0.72,
            "task_probabilities": {
                "malignancy_calibrated": 0.72
            }
        }

        # Case A: Non-finite float (NaN) in margin_distribution
        p_nan = copy.deepcopy(base_payload)
        p_nan["task_probabilities"]["margin_distribution"] = [0.1, float("nan"), 0.3, 0.4]
        with self.assertRaises(StateValidationError) as ctx:
            AgentFindingPayload.from_dict(p_nan)
        self.assertIn("must be finite", str(ctx.exception))

        # Case B: Wrong length for margin_distribution (3 instead of 4)
        p_len = copy.deepcopy(base_payload)
        p_len["task_probabilities"]["margin_distribution"] = [0.1, 0.2, 0.7]
        with self.assertRaises(StateValidationError) as ctx2:
            AgentFindingPayload.from_dict(p_len)
        self.assertIn("must have length 4", str(ctx2.exception))

        # Case C: Out-of-range probability in calc_distribution (> 1.0)
        p_oor = copy.deepcopy(base_payload)
        p_oor["task_probabilities"]["calc_distribution"] = [0.1, 0.2, 0.3, 1.25, 0.1]
        with self.assertRaises(StateValidationError) as ctx3:
            AgentFindingPayload.from_dict(p_oor)
        self.assertIn("must be in [0.0, 1.0]", str(ctx3.exception))

    def test_32_legitimately_absent_optional_distributions_preserved(self):
        """Test that legitimately absent or None optional distributions pass validation and roundtrip."""
        payload_dict = {
            "agent_id": "mammo",
            "status": "success",
            "authority_level": "authoritative",
            "voting_eligible": True,
            "bi_rads": "4",
            "confidence": 0.85,
            "confidence_type": "calibrated_heuristic",
            "p_malignancy": 0.72,
            "findings_summary": {"lesion": "malignant", "margin": None, "calcification": None},
            "task_probabilities": {
                "malignancy_calibrated": 0.72,
                "margin_distribution": None,
                "calc_distribution": None
            },
            "explainability": {},
            "rationale": "Assessment.",
            "raw_finding_card": None,
            "metadata": {}
        }
        payload = AgentFindingPayload.from_dict(payload_dict)
        self.assertIsNone(payload.task_probabilities["margin_distribution"])
        self.assertIsNone(payload.task_probabilities["calc_distribution"])
        self.assertTrue(payload.voting_eligible)

        # Confirm serialization and roundtrip
        d = payload.to_dict()
        self.assertIsNone(d["task_probabilities"]["margin_distribution"])
        reconstructed = AgentFindingPayload.from_dict(d)
        self.assertIsNone(reconstructed.task_probabilities["margin_distribution"])

    def test_33_invalid_types_for_validation_issue_and_deferral_reasons(self):
        """Test that non-string types in ValidationIssue and deferral_reasons raise StateValidationError."""
        # Non-string severity in ValidationIssue
        with self.assertRaises(StateValidationError) as ctx:
            ValidationIssue.from_dict({"severity": 1, "code": "ERR", "message": "msg"})
        self.assertIn("severity must be a str", str(ctx.exception))

        # Non-string code in ValidationIssue
        with self.assertRaises(StateValidationError) as ctx2:
            ValidationIssue.from_dict({"severity": "ERROR", "code": 999, "message": "msg"})
        self.assertIn("code must be a str", str(ctx2.exception))

        # Non-string field in ValidationIssue
        with self.assertRaises(StateValidationError) as ctx3:
            ValidationIssue.from_dict({"severity": "ERROR", "code": "ERR", "message": "msg", "field": 42})
        self.assertIn("field must be a str or None", str(ctx3.exception))

        # Non-string item in deferral_reasons
        mammo_payload = {
            "agent_id": "mammo",
            "status": "failed",
            "authority_level": "authoritative",
            "voting_eligible": False,
            "bi_rads": "unspecified",
            "confidence": 0.0,
            "confidence_type": "none"
        }
        state_dict = {
            "study_id": "P_00008_LEFT",
            "timestamp_utc": "2026-10-25T12:00:00Z",
            "primary_agent_id": PRIMARY_AGENT_ID,
            "defer_to_human": True,
            "deferral_reasons": ["Valid reason", 12345],  # 12345 is not a string
            "agents": {"mammo": mammo_payload}
        }
        with self.assertRaises(StateValidationError) as ctx4:
            SupervisorState.from_dict(state_dict)
        self.assertIn("deferral_reasons[1] must be a str", str(ctx4.exception))

    def test_34_nan_infinity_sanitization_vs_semantic_validation(self):
        """Test distinction: sanitize_for_json allows JSON export of NaN, but deserialization strictly rejects it."""
        corrupt_data = {
            "agent_id": "mammo",
            "status": "success",
            "authority_level": "authoritative",
            "voting_eligible": True,
            "bi_rads": "4",
            "confidence": float("nan"),
            "confidence_type": "calibrated_heuristic",
            "p_malignancy": float("nan"),
            "task_probabilities": {"malignancy_calibrated": float("nan")}
        }
        # 1. Transport sanitization safeguard: converts NaN to "nan" so json.dumps doesn't crash
        sanitized = sanitize_for_json(corrupt_data)
        self.assertEqual(sanitized["confidence"], "nan")
        self.assertEqual(sanitized["p_malignancy"], "nan")
        json_str = json.dumps(sanitized, allow_nan=False)
        self.assertIn('"confidence": "nan"', json_str)

        # 2. Semantic validation safeguard: deserialization MUST reject "nan" as an invalid prediction
        with self.assertRaises(StateValidationError) as ctx:
            AgentFindingPayload.from_dict(json.loads(json_str))
        self.assertIn("confidence for agent 'mammo' must be a finite float", str(ctx.exception))

    def test_35_clinical_risk_and_pathology_voting_attempts_rejected(self):
        """Test that Clin or Path attempting to claim voting_eligible=True is rejected in state deserialization."""
        valid_mammo = {
            "agent_id": "mammo",
            "status": "success",
            "authority_level": "authoritative",
            "voting_eligible": True,
            "bi_rads": "4",
            "confidence": 0.85,
            "confidence_type": "calibrated_heuristic",
            "p_malignancy": 0.72,
            "task_probabilities": {"malignancy_calibrated": 0.72}
        }
        illegal_clin = {
            "agent_id": "clin",
            "status": "success",
            "authority_level": "non_authoritative_research",
            "voting_eligible": True,  # Illegal
            "bi_rads": "unspecified",
            "confidence": 0.0,
            "confidence_type": "non_authoritative_zero_weight"
        }
        state_dict = {
            "study_id": "P_00008_LEFT",
            "timestamp_utc": "2026-10-25T12:00:00Z",
            "primary_agent_id": PRIMARY_AGENT_ID,
            "defer_to_human": False,
            "agents": {
                "mammo": valid_mammo,
                "clin": illegal_clin
            }
        }
        with self.assertRaises(StateValidationError) as ctx:
            SupervisorState.from_dict(state_dict)
        self.assertIn("Governance breach: Agent 'clin' cannot be voting_eligible", str(ctx.exception))

    def test_36_failed_or_invalid_primary_with_defer_false_rejected(self):
        """Test that failed or invalid primary results with defer_to_human=False are strictly rejected."""
        mammo_schema_invalid = {
            "agent_id": "mammo",
            "status": "schema_invalid",
            "authority_level": "authoritative",
            "voting_eligible": False,
            "bi_rads": "unspecified",
            "confidence": 0.0,
            "confidence_type": "none"
        }
        state_dict = {
            "study_id": "P_00008_LEFT",
            "timestamp_utc": "2026-10-25T12:00:00Z",
            "primary_agent_id": PRIMARY_AGENT_ID,
            "defer_to_human": False,  # Contradicts schema_invalid primary
            "agents": {"mammo": mammo_schema_invalid}
        }
        with self.assertRaises(StateValidationError) as ctx:
            SupervisorState.from_dict(state_dict)
        self.assertIn("Safety invariant violation", str(ctx.exception))

    def test_37_representative_states_serialization_roundtrip(self):
        """Test complete serialization and deserialization roundtrip on representative state."""
        resp = OrchestrationResponse(
            primary_finding_card=_create_sample_mammo_card(),
            shadow_finding_card=_create_sample_clin_card(),
            audit_record=_create_sample_audit()
        )
        state = adapt_orchestration_response(resp)

        # Export to JSON
        json_str = state.to_json(indent=2)
        parsed_dict = json.loads(json_str)

        # Ingest and reconstruct
        reconstructed = SupervisorState.from_dict(parsed_dict)
        self.assertEqual(reconstructed.study_id, state.study_id)
        self.assertEqual(reconstructed.defer_to_human, state.defer_to_human)
        self.assertEqual(reconstructed.agents["mammo"].bi_rads, "4")
        self.assertAlmostEqual(reconstructed.agents["mammo"].p_malignancy, 0.72)
        self.assertEqual(reconstructed.agents["clin"].status, "success")
        self.assertFalse(reconstructed.agents["clin"].voting_eligible)
        self.assertEqual(reconstructed.agents["path"].status, "unavailable")


if __name__ == "__main__":
    unittest.main()
