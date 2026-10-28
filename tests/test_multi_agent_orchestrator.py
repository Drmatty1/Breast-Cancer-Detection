"""
Focused Unit and Integration Tests for Multi-Agent Orchestrator.

Governed by:
- docs/breast_cancer___ps1___methodology.pdf (§§ 3.1, 3.4, 5.1, 5.3)
- DECISIONS.md (DEC-OWNER-01, DEC-OWNER-02, DEC-OWNER-03)
- reports/clin/CLIN_AGENT_C3_7_ORCHESTRATION_CONTRACT.md

Tests:
1. test_mammo_output_unchanged_when_clin_succeeds
2. test_mammo_output_unchanged_when_clin_abstains
3. test_mammo_output_unchanged_when_clin_raises
4. test_mammo_output_unchanged_when_clin_times_out
5. test_repeated_clin_timeouts_bound_latency_and_do_not_leak_workers
6. test_invalid_clin_card_is_rejected_and_mammo_preserved
7. test_strict_breast_density_validation_and_normalization
8. test_clin_never_enters_voting_or_consensus_channel
9. test_unspecified_birads_does_not_trigger_rank_comparison
10. test_mammo_failure_never_falls_back_to_clin
11. test_logging_failure_is_reported_without_corrupting_mammo_result
12. test_audit_record_distinguishes_success_abstention_error_and_timeout
13. test_allowlisted_features_only_reach_clin
"""
import copy
import json
import math
import time
import threading
import unittest
from typing import Dict, Any, Optional

import numpy as np

from src.mammo.finding_card import build_finding_card
from src.clin.inference import ClinicalRiskAgentInferencePipeline, sanitize_for_json, safe_str
from src.orchestration.multi_agent_orchestrator import (
    MultiAgentOrchestrator,
    OrchestrationRequest,
    OrchestrationResponse,
    MammoInferenceError,
    MammoSchemaValidationError
)


class MockMammoPipeline:
    """Mock Mammography pipeline returning a canonical finding card."""
    def __init__(self, should_fail: bool = False, fail_with_invalid_card: bool = False):
        self.should_fail = should_fail
        self.fail_with_invalid_card = fail_with_invalid_card
        self.calls = []
        self.last_returned_card = None
        self.last_snapshot = None

    def predict(
        self,
        cc_input: Any,
        mlo_input: Any,
        generate_saliency: bool = True,
        saliency_output_dir: Optional[str] = None,
        pair_identifier: str = "sample_pair",
        clinical_header: Optional[Dict[str, Any]] = None,
        abnormality_type: Optional[str] = None
    ) -> Dict[str, Any]:
        self.calls.append({
            "pair_identifier": pair_identifier,
            "clinical_header": clinical_header,
            "abnormality_type": abnormality_type
        })
        if self.should_fail:
            raise RuntimeError("Simulated Mammography Agent forward pass exception")

        card = build_finding_card(
            binary_prob_calibrated=0.8841,
            margin_probs=[0.05, 0.10, 0.05, 0.80],
            calc_probs=None,
            birads_ordinal_level=3,
            birads_confidence=0.91,
            saliency_regions=[[100.0, 150.0, 250.0, 300.0]],
            abnormality_type="mass"
        )
        if self.fail_with_invalid_card:
            card["conf_a"] = 999.0  # Schema violation (> 1.0)
        self.last_snapshot = copy.deepcopy(card)
        self.last_returned_card = card
        return card


class MockClinPipeline:
    """Mock Clin Agent pipeline with configurable behavior."""
    def __init__(
        self,
        behavior: str = "success",
        sleep_time: float = 0.0
    ):
        self.behavior = behavior
        self.sleep_time = sleep_time
        self.received_covariates = []

    def predict(
        self,
        clinical_covariates: Optional[Dict[str, Any]] = None,
        pair_identifier: str = "sample_pair"
    ) -> Dict[str, Any]:
        self.received_covariates.append(clinical_covariates)
        if self.sleep_time > 0:
            time.sleep(self.sleep_time)

        if self.behavior == "raise":
            raise RuntimeError("Simulated Clin XGBoost CUDA error")
        elif self.behavior == "invalid_schema":
            return {
                "agent": "clin",
                "finding": {"lesion": 12345},  # Invalid type (integer instead of string/null)
                "BI_RADS_a": "unspecified",
                "conf_a": 0.0,
                "sal": {"top_regions": []},
                "sh": {},
                "rationale": "Invalid card",
                "metadata": {}
            }
        elif self.behavior == "abstain":
            return {
                "agent": "clin",
                "finding": {
                    "lesion": None,
                    "margin": None,
                    "calcification": None,
                    "risk_profile": "unavailable",
                    "abstention_reason": "missing_required_covariates"
                },
                "BI_RADS_a": "unspecified",
                "conf_a": 0.0,
                "sal": {"top_regions": []},
                "sh": {},
                "rationale": "Abstained due to missing covariates.",
                "metadata": {
                    "model_architecture": "xgboost_tabular_logistic_ensemble",
                    "model_version": "clin_agent_v1.0",
                    "status": "abstained",
                    "authority": "non_authoritative_research",
                    "voting_eligibility": False,
                    "is_clinically_validated": False
                }
            }
        else:
            return {
                "agent": "clin",
                "finding": {
                    "lesion": None,
                    "margin": None,
                    "calcification": None,
                    "risk_profile": "malignancy_risk",
                    "breast_density": 2,
                    "breast_side": "LEFT"
                },
                "BI_RADS_a": "unspecified",
                "conf_a": 0.0,
                "sal": {
                    "top_regions": [],
                    "tree_shap": {"features": ["breast_density", "breast_side"], "contributions": [0.01, -0.02], "bias": -0.22}
                },
                "sh": {"breast_density": 2, "breast_side": "LEFT"},
                "rationale": "Calibrated risk 0.443. Non-authoritative shadow mode.",
                "metadata": {
                    "model_architecture": "xgboost_tabular_logistic_ensemble",
                    "model_version": "clin_agent_v1.0",
                    "status": "success",
                    "authority": "non_authoritative_research",
                    "voting_eligibility": False,
                    "is_clinically_validated": False,
                    "task_probabilities": {"malignancy_calibrated": 0.4431, "malignancy_uncalibrated": 0.4452}
                }
            }


class TestMultiAgentOrchestrator(unittest.TestCase):
    def setUp(self):
        self.mock_mammo = MockMammoPipeline()
        self.mock_clin = MockClinPipeline(behavior="success")
        self.orchestrator = MultiAgentOrchestrator(
            mammo_pipeline=self.mock_mammo,
            clin_pipeline=self.mock_clin,
            timeout_seconds=0.5,
            max_workers=2
        )

    def tearDown(self):
        if hasattr(self, "orchestrator") and self.orchestrator is not None:
            self.orchestrator.close(wait=False)

    def test_mammo_output_unchanged_when_clin_succeeds(self):
        """1. Assert Mammo card is completely unchanged when Clin succeeds."""
        request = OrchestrationRequest(
            pair_identifier="P_00008_LEFT",
            cc_input="dummy_cc",
            mlo_input="dummy_mlo",
            clinical_covariates={"breast_density": 2, "breast_side": "LEFT"}
        )
        response = self.orchestrator.run(request)

        # Exact object identity: Mammo pipeline return reference is preserved directly
        self.assertIs(response.primary_finding_card, self.mock_mammo.last_returned_card)
        # Deep equality: content is completely unmutated compared to pre-invocation snapshot
        self.assertEqual(response.primary_finding_card, self.mock_mammo.last_snapshot)
        self.assertEqual(response.mammo_status, "SUCCESS")
        self.assertEqual(response.clin_status, "SUCCESS")
        self.assertIsNotNone(response.shadow_finding_card)
        self.assertEqual(response.shadow_finding_card["agent"], "clin")

        # Invariance check: no Clin artifacts in primary card
        self.assertNotIn("tree_shap", response.primary_finding_card["sal"])
        self.assertNotIn("xgboost", json.dumps(response.primary_finding_card).lower())

    def test_mammo_output_unchanged_when_clin_abstains(self):
        """2. Assert Mammo card is completely unchanged when Clin abstains."""
        clin_abstain = MockClinPipeline(behavior="abstain")
        orch = MultiAgentOrchestrator(mammo_pipeline=self.mock_mammo, clin_pipeline=clin_abstain)
        try:
            request = OrchestrationRequest(
                pair_identifier="P_00008_LEFT",
                cc_input="dummy_cc",
                mlo_input="dummy_mlo",
                clinical_covariates={}
            )
            response = orch.run(request)

            self.assertIs(response.primary_finding_card, self.mock_mammo.last_returned_card)
            self.assertEqual(response.primary_finding_card, self.mock_mammo.last_snapshot)
            self.assertEqual(response.mammo_status, "SUCCESS")
            self.assertEqual(response.clin_status, "ABSTAINED")
            self.assertEqual(response.shadow_finding_card["metadata"]["status"], "abstained")
            self.assertEqual(response.shadow_finding_card["conf_a"], 0.0)
            self.assertIsNone(response.shadow_finding_card["finding"]["lesion"])
        finally:
            orch.close(wait=False)

    def test_mammo_output_unchanged_when_clin_raises(self):
        """3. Assert Mammo card is completely unchanged when Clin raises an exception."""
        clin_raise = MockClinPipeline(behavior="raise")
        orch = MultiAgentOrchestrator(mammo_pipeline=self.mock_mammo, clin_pipeline=clin_raise)
        try:
            request = OrchestrationRequest(
                pair_identifier="P_00008_LEFT",
                cc_input="dummy_cc",
                mlo_input="dummy_mlo",
                clinical_covariates={"breast_density": 2, "breast_side": "LEFT"}
            )
            response = orch.run(request)

            self.assertIs(response.primary_finding_card, self.mock_mammo.last_returned_card)
            self.assertEqual(response.primary_finding_card, self.mock_mammo.last_snapshot)
            self.assertEqual(response.mammo_status, "SUCCESS")
            self.assertEqual(response.clin_status, "FAILED")
            self.assertIsNone(response.shadow_finding_card)
            self.assertIn("RuntimeError", response.audit_record.clin_error_code)
        finally:
            orch.close(wait=False)

    def test_mammo_output_unchanged_when_clin_times_out(self):
        """4. Assert Mammo card is returned within bounded timeout without waiting for slow Clin."""
        # Clin mock sleeps for 0.40 seconds, timeout configured to 0.05 seconds
        clin_slow = MockClinPipeline(behavior="success", sleep_time=0.40)
        orch = MultiAgentOrchestrator(
            mammo_pipeline=self.mock_mammo,
            clin_pipeline=clin_slow,
            timeout_seconds=0.05,
            max_workers=2
        )
        try:
            request = OrchestrationRequest(
                pair_identifier="P_00008_LEFT",
                cc_input="dummy_cc",
                mlo_input="dummy_mlo",
                clinical_covariates={"breast_density": 2, "breast_side": "LEFT"}
            )
            t0 = time.perf_counter()
            response = orch.run(request)
            elapsed = time.perf_counter() - t0

            # Bounded return check: elapsed time must be close to timeout (0.05s) and well below 0.40s
            # Tolerance margin: < 0.20s accounts for scheduler jitter on CI/Windows
            self.assertLess(elapsed, 0.20, f"Orchestrator waited for slow Clin! Elapsed: {elapsed:.4f}s")
            self.assertGreaterEqual(elapsed, 0.04, f"Returned faster than expected timeout: {elapsed:.4f}s")

            self.assertIs(response.primary_finding_card, self.mock_mammo.last_returned_card)
            self.assertEqual(response.primary_finding_card, self.mock_mammo.last_snapshot)
            self.assertEqual(response.mammo_status, "SUCCESS")
            self.assertEqual(response.clin_status, "TIMEOUT")
            self.assertIsNone(response.shadow_finding_card)
            self.assertIn("clin_timeout_exceeded", response.audit_record.clin_error_code)
        finally:
            orch.close(wait=False)

    def test_repeated_clin_timeouts_bound_latency_and_do_not_leak_workers(self):
        """5. Assert repeated timeouts consistently bound latency and do not leak threads."""
        clin_slow = MockClinPipeline(behavior="success", sleep_time=0.08)
        orch = MultiAgentOrchestrator(
            mammo_pipeline=self.mock_mammo,
            clin_pipeline=clin_slow,
            timeout_seconds=0.03,
            max_workers=2
        )
        try:
            request = OrchestrationRequest(
                pair_identifier="P_00008_LEFT",
                cc_input="dummy_cc",
                mlo_input="dummy_mlo",
                clinical_covariates={"breast_density": 2, "breast_side": "LEFT"}
            )
            # Run 3 repeated timeouts where each task finishes before next invocation
            for i in range(3):
                t0 = time.perf_counter()
                resp = orch.run(request)
                elapsed = time.perf_counter() - t0
                self.assertLess(elapsed, 0.15, f"Call {i} took too long: {elapsed:.4f}s")
                self.assertEqual(resp.clin_status, "TIMEOUT")
                self.assertIs(resp.primary_finding_card, self.mock_mammo.last_returned_card)
                self.assertEqual(resp.primary_finding_card, self.mock_mammo.last_snapshot)
                time.sleep(0.06)  # allow background worker to finish task and release capacity

            # Check that worker threads in the pool are bounded by max_workers (2)
            shadow_threads = [
                t.name for t in threading.enumerate()
                if orch._thread_prefix in t.name
            ]
            self.assertLessEqual(
                len(shadow_threads), 2,
                f"Worker thread accumulation detected! Active threads: {shadow_threads}"
            )
        finally:
            orch.close(wait=False)

    def test_invalid_clin_card_is_rejected_and_mammo_preserved(self):
        """6. Assert malformed Clin card is rejected without affecting Mammo output."""
        clin_invalid = MockClinPipeline(behavior="invalid_schema")
        orch = MultiAgentOrchestrator(mammo_pipeline=self.mock_mammo, clin_pipeline=clin_invalid)
        try:
            request = OrchestrationRequest(
                pair_identifier="P_00008_LEFT",
                cc_input="dummy_cc",
                mlo_input="dummy_mlo",
                clinical_covariates={"breast_density": 2, "breast_side": "LEFT"}
            )
            response = orch.run(request)

            self.assertIs(response.primary_finding_card, self.mock_mammo.last_returned_card)
            self.assertEqual(response.primary_finding_card, self.mock_mammo.last_snapshot)
            self.assertEqual(response.clin_status, "SCHEMA_INVALID")
            self.assertIsNone(response.shadow_finding_card)
            self.assertIn("schema_validation_failed", response.audit_record.clin_error_code)
        finally:
            orch.close(wait=False)

    def test_strict_breast_density_validation_and_normalization(self):
        """7. Assert strict density validation: rejects booleans, floats, non-finite, out-of-range."""
        pipeline = ClinicalRiskAgentInferencePipeline(lazy_load=True)

        # (input_val, should_accept, expected_normalized_value)
        test_cases = [
            (1, True, 1.0),
            (2, True, 2.0),
            (3, True, 3.0),
            (4, True, 4.0),
            (2.0, True, 2.0),
            ("2", True, 2.0),
            ("2.0", True, 2.0),
            (2.5, False, None),
            ("2.5", False, None),
            (0, False, None),
            (5, False, None),
            (True, False, None),
            (False, False, None),
            (float("nan"), False, None),
            (float("inf"), False, None),
            (float("-inf"), False, None),
            (None, False, None),
            ("abc", False, None),
            ("2.5.0", False, None)
        ]

        for val, should_accept, expected_norm in test_cases:
            covs = {"breast_density": val, "breast_side": "LEFT"}
            d_val, s_val, reason = pipeline.validate_and_extract_features(covs)
            is_accepted = (reason is None)
            self.assertEqual(
                is_accepted, should_accept,
                f"Validation mismatch for {val!r}: got accepted={is_accepted}, expected={should_accept}, reason={reason}"
            )
            if should_accept:
                self.assertEqual(d_val, expected_norm)

            # Test through predict() ensuring clean schema-valid abstention on invalid cases
            card = pipeline.predict(covs)
            self.assertIsNotNone(card)
            if should_accept:
                # Lazy pipeline without weights abstains with model_not_initialized, which is valid
                self.assertIn(card["metadata"]["status"], ("success", "abstained"))
            else:
                self.assertEqual(card["metadata"]["status"], "abstained")
                self.assertEqual(card["conf_a"], 0.0)
                self.assertIsNone(card["finding"]["lesion"])
                self.assertEqual(card["finding"]["risk_profile"], "unavailable")

    def test_clin_never_enters_voting_or_consensus_channel(self):
        """8. Assert Clin Agent has zero channel into primary voting results."""
        request = OrchestrationRequest(
            pair_identifier="P_00008_LEFT",
            cc_input="dummy_cc",
            mlo_input="dummy_mlo",
            clinical_covariates={"breast_density": 2, "breast_side": "LEFT"}
        )
        response = self.orchestrator.run(request)

        # Primary card is strictly Mammo
        self.assertEqual(response.primary_finding_card["agent"], "mammo")
        # Ensure no Clin field contaminated Mammo card
        self.assertNotIn("tree_shap", response.primary_finding_card["sal"])
        self.assertNotIn("xgboost", json.dumps(response.primary_finding_card).lower())

        # Shadow card is strictly marked non-voting
        self.assertFalse(response.shadow_finding_card["metadata"]["voting_eligibility"])
        self.assertEqual(response.shadow_finding_card["metadata"]["authority"], "non_authoritative_research")

    def test_unspecified_birads_does_not_trigger_rank_comparison(self):
        """9. Assert Clin BI-RADS is 'unspecified' and immune to rank difference calculations."""
        real_clin = ClinicalRiskAgentInferencePipeline(lazy_load=False)
        card = real_clin.predict({"breast_density": 3, "breast_side": "RIGHT"})

        self.assertEqual(card["BI_RADS_a"], "unspecified")

        # Deferral logic rule check: ordinal rank extraction must reject 'unspecified'
        valid_birads_ranks = {"2": 2, "3": 3, "4": 4, "4A": 4, "4B": 4, "4C": 4, "5": 5}
        clin_rank = valid_birads_ranks.get(card["BI_RADS_a"])
        self.assertIsNone(clin_rank)  # Confirms no numeric rank can be computed

    def test_mammo_failure_never_falls_back_to_clin(self):
        """10. Assert Mammo failure raises explicit exception and never substitutes Clin."""
        failing_mammo = MockMammoPipeline(should_fail=True)
        orch = MultiAgentOrchestrator(mammo_pipeline=failing_mammo, clin_pipeline=self.mock_clin)
        try:
            request = OrchestrationRequest(
                pair_identifier="P_00008_LEFT",
                cc_input="dummy_cc",
                mlo_input="dummy_mlo",
                clinical_covariates={"breast_density": 2, "breast_side": "LEFT"}
            )
            with self.assertRaises(MammoInferenceError):
                orch.run(request)
        finally:
            orch.close(wait=False)

    def test_logging_failure_is_reported_without_corrupting_mammo_result(self):
        """11. Assert audit logging failure reports warning but returns intact Mammo card."""
        def failing_logger(rec):
            raise IOError("Simulated disk write failure")

        orch = MultiAgentOrchestrator(
            mammo_pipeline=self.mock_mammo,
            clin_pipeline=self.mock_clin,
            audit_logger=failing_logger
        )
        try:
            request = OrchestrationRequest(
                pair_identifier="P_00008_LEFT",
                cc_input="dummy_cc",
                mlo_input="dummy_mlo",
                clinical_covariates={"breast_density": 2, "breast_side": "LEFT"}
            )
            response = orch.run(request)

            # Mammo card is preserved 100%
            self.assertIs(response.primary_finding_card, self.mock_mammo.last_returned_card)
            self.assertEqual(response.primary_finding_card, self.mock_mammo.last_snapshot)
            # Audit logging failure is explicitly captured
            self.assertFalse(response.audit_record.audit_persisted)
            self.assertTrue(
                "IOError" in response.audit_record.logging_warning or "OSError" in response.audit_record.logging_warning
            )
        finally:
            orch.close(wait=False)

    def test_audit_record_distinguishes_success_abstention_error_and_timeout(self):
        """12. Assert audit record accurately distinguishes all execution states."""
        req = OrchestrationRequest(
            pair_identifier="P_00008_LEFT",
            cc_input="dummy_cc",
            mlo_input="dummy_mlo",
            clinical_covariates={"breast_density": 2, "breast_side": "LEFT"}
        )

        orch1 = MultiAgentOrchestrator(self.mock_mammo, MockClinPipeline("success"))
        orch2 = MultiAgentOrchestrator(self.mock_mammo, MockClinPipeline("abstain"))
        orch3 = MultiAgentOrchestrator(self.mock_mammo, MockClinPipeline("raise"))
        orch4 = MultiAgentOrchestrator(self.mock_mammo, MockClinPipeline("success", sleep_time=0.3), timeout_seconds=0.05)
        try:
            # 1. Success
            res_ok = orch1.run(req)
            self.assertEqual(res_ok.audit_record.clin_status, "SUCCESS")

            # 2. Abstention
            res_abs = orch2.run(req)
            self.assertEqual(res_abs.audit_record.clin_status, "ABSTAINED")

            # 3. Error
            res_err = orch3.run(req)
            self.assertEqual(res_err.audit_record.clin_status, "FAILED")

            # 4. Timeout
            res_to = orch4.run(req)
            self.assertEqual(res_to.audit_record.clin_status, "TIMEOUT")
        finally:
            orch1.close(wait=False)
            orch2.close(wait=False)
            orch3.close(wait=False)
            orch4.close(wait=False)

    def test_allowlisted_features_only_reach_clin(self):
        """13. Assert only allowlisted features reach Clin pipeline, stripping forbidden metadata."""
        mock_clin = MockClinPipeline("success")
        orch = MultiAgentOrchestrator(self.mock_mammo, mock_clin)
        try:
            # Inject contaminated clinical metadata dictionary
            contaminated_covariates = {
                "breast_density": 3,
                "breast_side": "RIGHT",
                "patient_id": "P_99999",
                "pathology": "MALIGNANT",
                "target_binary": 1,
                "patient_name": "ANONYMOUS",
                "notes": "Biopsy positive"
            }
            request = OrchestrationRequest(
                pair_identifier="P_00008_LEFT",
                cc_input="dummy_cc",
                mlo_input="dummy_mlo",
                clinical_covariates=contaminated_covariates
            )
            orch.run(request)

            # Verify what mock_clin received
            self.assertEqual(len(mock_clin.received_covariates), 1)
            received = mock_clin.received_covariates[0]

            # Must only contain breast_density and breast_side
            self.assertEqual(set(received.keys()), {"breast_density", "breast_side"})
            self.assertNotIn("patient_id", received)
            self.assertNotIn("pathology", received)
            self.assertNotIn("target_binary", received)
        finally:
            orch.close(wait=False)

    def test_executor_lifecycle_worker_saturation_fast_path(self):
        """14. Scenario 2: When both workers are busy, subsequent request returns immediately without queuing."""
        # Mock pipeline sleeps for 0.40s
        clin_slow = MockClinPipeline(behavior="success", sleep_time=0.40)
        orch = MultiAgentOrchestrator(
            mammo_pipeline=self.mock_mammo,
            clin_pipeline=clin_slow,
            timeout_seconds=0.04,
            max_workers=2
        )
        try:
            req = OrchestrationRequest(
                pair_identifier="P_00008_LEFT",
                cc_input="dummy_cc",
                mlo_input="dummy_mlo",
                clinical_covariates={"breast_density": 2, "breast_side": "LEFT"}
            )
            # Dispatch call 1: occupies worker 1, caller times out after 0.04s
            r1 = orch.run(req)
            self.assertEqual(r1.clin_status, "TIMEOUT")

            # Dispatch call 2: occupies worker 2, caller times out after 0.04s
            r2 = orch.run(req)
            self.assertEqual(r2.clin_status, "TIMEOUT")

            # Dispatch call 3 while workers 1 & 2 are still busy executing
            t0 = time.perf_counter()
            r3 = orch.run(req)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Must return immediately without queuing (well under 20ms)
            self.assertLess(elapsed_ms, 40.0, f"Saturated request was queued! Took: {elapsed_ms:.2f}ms")
            self.assertEqual(r3.clin_status, "SKIPPED")
            self.assertEqual(r3.audit_record.clin_error_code, "clin_worker_capacity_saturated")
            self.assertIs(r3.primary_finding_card, self.mock_mammo.last_returned_card)
            self.assertEqual(r3.primary_finding_card, self.mock_mammo.last_snapshot)
        finally:
            orch.close(wait=False)

    def test_executor_lifecycle_recovery_after_slow_tasks_finish(self):
        """15. Scenario 8: Slow tasks completing after timeout frees capacity for subsequent requests."""
        clin_slow = MockClinPipeline(behavior="success", sleep_time=0.10)
        orch = MultiAgentOrchestrator(
            mammo_pipeline=self.mock_mammo,
            clin_pipeline=clin_slow,
            timeout_seconds=0.03,
            max_workers=1
        )
        try:
            req = OrchestrationRequest(
                pair_identifier="P_00008_LEFT",
                cc_input="dummy_cc",
                mlo_input="dummy_mlo",
                clinical_covariates={"breast_density": 2, "breast_side": "LEFT"}
            )
            # Call 1: times out
            r1 = orch.run(req)
            self.assertEqual(r1.clin_status, "TIMEOUT")

            # Immediate call 2: saturated
            r2 = orch.run(req)
            self.assertEqual(r2.clin_status, "SKIPPED")
            self.assertEqual(r2.audit_record.clin_error_code, "clin_worker_capacity_saturated")

            # Wait for background task on worker to finish (0.10s - 0.03s + margin = 0.12s)
            time.sleep(0.12)

            # Switch mock pipeline to instant response
            orch.clin_pipeline = MockClinPipeline(behavior="success", sleep_time=0.0)

            # Call 3: capacity has been restored, succeeds immediately
            r3 = orch.run(req)
            self.assertEqual(r3.clin_status, "SUCCESS")
            self.assertIsNotNone(r3.shadow_finding_card)
        finally:
            orch.close(wait=False)

    def test_executor_lifecycle_close_and_rejection_of_subsequent_requests(self):
        """16. Scenario 3 & 4: close(wait=False) terminates executor and subsequent run() raises RuntimeError."""
        clin_slow = MockClinPipeline(behavior="success", sleep_time=0.30)
        orch = MultiAgentOrchestrator(
            mammo_pipeline=self.mock_mammo,
            clin_pipeline=clin_slow,
            timeout_seconds=0.05,
            max_workers=2
        )
        req = OrchestrationRequest(
            pair_identifier="P_00008_LEFT",
            cc_input="dummy_cc",
            mlo_input="dummy_mlo",
            clinical_covariates={"breast_density": 2, "breast_side": "LEFT"}
        )
        # Run one call that times out (task running in background)
        r = orch.run(req)
        self.assertEqual(r.clin_status, "TIMEOUT")

        # Close with wait=False: must return immediately without blocking on running task
        t0 = time.perf_counter()
        orch.close(wait=False)
        close_elapsed = time.perf_counter() - t0
        self.assertLess(close_elapsed, 0.05, f"close(wait=False) blocked! Elapsed: {close_elapsed:.4f}s")
        self.assertTrue(orch._is_closed)

        # Subsequent run() must raise explicit RuntimeError
        with self.assertRaises(RuntimeError) as ctx:
            orch.run(req)
        self.assertIn("closed and cannot accept new requests", str(ctx.exception))

    def test_executor_lifecycle_repeated_close_is_safe_noop(self):
        """17. Scenario 5: Repeated close calls are idempotent safe no-ops."""
        orch = MultiAgentOrchestrator(mammo_pipeline=self.mock_mammo, clin_pipeline=self.mock_clin)
        # Calling close multiple times sequentially
        orch.close(wait=False)
        orch.close(wait=False)
        orch.close(wait=True)
        self.assertTrue(orch._is_closed)
        self.assertIsNone(orch._executor)

    def test_executor_lifecycle_context_manager_exit_while_tasks_running(self):
        """18. Scenario 6: Context-manager exit while tasks are running closes cleanly and non-blockingly."""
        clin_slow = MockClinPipeline(behavior="success", sleep_time=0.30)
        req = OrchestrationRequest(
            pair_identifier="P_00008_LEFT",
            cc_input="dummy_cc",
            mlo_input="dummy_mlo",
            clinical_covariates={"breast_density": 2, "breast_side": "LEFT"}
        )
        t0 = time.perf_counter()
        with MultiAgentOrchestrator(
            mammo_pipeline=self.mock_mammo,
            clin_pipeline=clin_slow,
            timeout_seconds=0.04,
            max_workers=2
        ) as orch:
            r = orch.run(req)
            self.assertEqual(r.clin_status, "TIMEOUT")

        # Context manager exit must not block on 0.30s background task
        total_time = time.perf_counter() - t0
        self.assertLess(total_time, 0.15, f"Context exit blocked on background task! Time: {total_time:.4f}s")
        self.assertTrue(orch._is_closed)

        # Post-exit run raises RuntimeError
        with self.assertRaises(RuntimeError):
            orch.run(req)

    def test_executor_lifecycle_destruction_and_del(self):
        """19. Scenario 7: Object destruction (__del__) cleanly executes non-blocking close."""
        orch = MultiAgentOrchestrator(
            mammo_pipeline=self.mock_mammo,
            clin_pipeline=self.mock_clin,
            max_workers=2
        )
        # Initialize executor
        _ = orch._get_executor()
        self.assertIsNotNone(orch._executor)
        # Call __del__
        orch.__del__()
        self.assertTrue(orch._is_closed)
        self.assertIsNone(orch._executor)

    def test_numpy_non_finite_scalars_abstention_and_json_serialization(self):
        """20. Assert NumPy non-finite scalars trigger valid abstention and serialize with allow_nan=False."""
        clin_pipeline = ClinicalRiskAgentInferencePipeline(lazy_load=True)
        orch = MultiAgentOrchestrator(mammo_pipeline=self.mock_mammo, clin_pipeline=clin_pipeline)

        non_finite_inputs = [
            ("np_float32_nan", np.float32("nan")),
            ("np_float64_nan", np.float64("nan")),
            ("np_float32_inf", np.float32("inf")),
            ("np_float64_neg_inf", np.float64("-inf")),
            ("py_float_nan", float("nan")),
            ("py_float_inf", float("inf")),
            ("py_float_neg_inf", float("-inf")),
        ]

        try:
            for name, val in non_finite_inputs:
                with self.subTest(non_finite_case=name):
                    req = OrchestrationRequest(
                        pair_identifier=f"P_TEST_{name}",
                        cc_input="dummy_cc",
                        mlo_input="dummy_mlo",
                        clinical_covariates={"breast_density": val, "breast_side": "L"}
                    )
                    resp = orch.run(req)

                    # Mammo card preserved
                    self.assertIs(resp.primary_finding_card, self.mock_mammo.last_returned_card)
                    self.assertEqual(resp.primary_finding_card, self.mock_mammo.last_snapshot)
                    self.assertEqual(resp.mammo_status, "SUCCESS")
                    self.assertEqual(resp.clin_status, "ABSTAINED")

                    # Clin card exists and is abstained
                    card = resp.shadow_finding_card
                    self.assertIsNotNone(card)
                    self.assertEqual(card["metadata"]["status"], "abstained")
                    self.assertIn("non_finite", card["finding"]["abstention_reason"])

                    # Density field in sh is string sanitized representation
                    density_sh = card["sh"]["breast_density"]
                    self.assertIsInstance(density_sh, str)
                    self.assertIn(density_sh, ("nan", "inf", "-inf"))

                    # Strict JSON compliance: must serialize with allow_nan=False without error
                    json_str = json.dumps(card, allow_nan=False)
                    self.assertIsInstance(json_str, str)

                    # Assert no literal NaN or Infinity tokens in raw JSON string
                    self.assertNotIn(": NaN", json_str)
                    self.assertNotIn(": Infinity", json_str)
                    self.assertNotIn(": -Infinity", json_str)
        finally:
            orch.close(wait=False)

    def test_numpy_finite_scalars_prediction_and_json_serialization(self):
        """21. Assert NumPy finite scalar values produce valid predictions and serialize with allow_nan=False."""
        clin_pipeline = ClinicalRiskAgentInferencePipeline(lazy_load=False)
        orch = MultiAgentOrchestrator(mammo_pipeline=self.mock_mammo, clin_pipeline=clin_pipeline)

        finite_inputs = [
            ("np_int32_2", np.int32(2)),
            ("np_int64_3", np.int64(3)),
            ("np_float32_2", np.float32(2.0)),
            ("np_float64_4", np.float64(4.0)),
        ]

        try:
            for name, val in finite_inputs:
                with self.subTest(finite_case=name):
                    req = OrchestrationRequest(
                        pair_identifier=f"P_TEST_{name}",
                        cc_input="dummy_cc",
                        mlo_input="dummy_mlo",
                        clinical_covariates={"breast_density": val, "breast_side": "R"}
                    )
                    resp = orch.run(req)

                    # Primary card preserved
                    self.assertIs(resp.primary_finding_card, self.mock_mammo.last_returned_card)
                    self.assertEqual(resp.primary_finding_card, self.mock_mammo.last_snapshot)
                    self.assertEqual(resp.mammo_status, "SUCCESS")
                    self.assertEqual(resp.clin_status, "SUCCESS")

                    card = resp.shadow_finding_card
                    self.assertIsNotNone(card)
                    self.assertEqual(card["metadata"]["status"], "success")

                    # Density field in sh must be valid normalized category integer
                    self.assertIn(card["sh"]["breast_density"], (1, 2, 3, 4))
                    self.assertIsInstance(card["sh"]["breast_density"], int)

                    # Strict JSON compliance: must serialize with allow_nan=False
                    json_str = json.dumps(card, allow_nan=False)
                    self.assertIsInstance(json_str, str)
        finally:
            orch.close(wait=False)

    def test_recursive_sanitizer_module_level_unit_tests(self):
        """22. Direct unit tests for module-level sanitize_for_json."""
        class CustomObj:
            def __init__(self, val):
                self.val = val
            def __str__(self):
                return f"Custom({self.val})"

        class BrokenStr:
            def __str__(self):
                raise RuntimeError("Intentional __str__ failure")

        # 1. Primitives
        self.assertIsNone(sanitize_for_json(None))
        self.assertTrue(sanitize_for_json(True))
        self.assertFalse(sanitize_for_json(False))
        self.assertEqual(sanitize_for_json(123), 123)
        self.assertEqual(sanitize_for_json(3.14), 3.14)
        self.assertEqual(sanitize_for_json("text"), "text")

        # 2. NumPy scalars
        self.assertEqual(sanitize_for_json(np.int32(42)), 42)
        self.assertEqual(sanitize_for_json(np.float64(2.5)), 2.5)
        self.assertEqual(sanitize_for_json(np.float32("nan")), "nan")
        self.assertEqual(sanitize_for_json(np.float64("inf")), "inf")
        self.assertEqual(sanitize_for_json(float("-inf")), "-inf")

        # 3. Nested dictionary and list
        raw_struct = {
            "a": [np.float32("nan"), np.int64(7)],
            "b": {"nested": float("inf"), "finite": np.float32(1.5)}
        }
        sanitized = sanitize_for_json(raw_struct)
        self.assertEqual(sanitized, {
            "a": ["nan", 7],
            "b": {"nested": "inf", "finite": 1.5}
        })
        # Assert raw_struct was NOT mutated
        self.assertTrue(np.isnan(raw_struct["a"][0]))
        self.assertIsInstance(raw_struct["a"][0], np.float32)

        # 4. Sets and custom objects
        self.assertIn("<unsupported: set:", sanitize_for_json({1, 2}))
        self.assertIn("<unsupported: CustomObj: Custom(99)>", sanitize_for_json(CustomObj(99)))
        self.assertEqual(sanitize_for_json(BrokenStr()), "<unsupported_unprintable: BrokenStr>")

        # 5. Cycles and deep recursion
        circ_list = []
        circ_list.append(circ_list)
        self.assertEqual(sanitize_for_json(circ_list), ["<circular_ref>"])

        circ_dict = {}
        circ_dict["self"] = circ_dict
        self.assertEqual(sanitize_for_json(circ_dict), {"self": "<circular_ref>"})

        deep_list = 0
        for _ in range(40):
            deep_list = [deep_list]
        sanitized_deep = sanitize_for_json(deep_list)
        # Verify max depth was reached safely without RecursionError
        json_deep = json.dumps(sanitized_deep, allow_nan=False)
        self.assertIn("<max_depth_exceeded>", json_deep)

    def test_nested_non_finite_values_abstention_and_json_compliance(self):
        """23. Section 4.A: Nested non-finite values in covariates trigger clean abstention and strict JSON compliance."""
        clin_pipeline = ClinicalRiskAgentInferencePipeline(lazy_load=True)
        orch = MultiAgentOrchestrator(mammo_pipeline=self.mock_mammo, clin_pipeline=clin_pipeline)

        test_cases = [
            ("side_nested_dict_nan", {"breast_density": 2, "breast_side": {"raw": np.float32("nan")}}),
            ("side_nested_list_inf", {"breast_density": 2, "breast_side": [np.float64("inf"), float("-inf")]}),
            ("density_nested_dict_nan", {"breast_density": {"val": np.float32("nan")}, "breast_side": "L"}),
            ("density_nested_list_inf", {"breast_density": [float("inf")], "breast_side": "R"}),
            ("deep_nested_mixed", {
                "breast_density": {"outer": [{"inner": np.float32("nan"), "count": np.int32(1)}]},
                "breast_side": {"dir": [float("-inf"), np.float64("inf")]}
            })
        ]

        try:
            for name, covs in test_cases:
                with self.subTest(case=name):
                    req = OrchestrationRequest(
                        pair_identifier=f"P_{name}",
                        cc_input="dummy_cc",
                        mlo_input="dummy_mlo",
                        clinical_covariates=covs
                    )
                    resp = orch.run(req)

                    # 1. Primary Mammo card is preserved (identity and deep content)
                    self.assertIs(resp.primary_finding_card, self.mock_mammo.last_returned_card)
                    self.assertEqual(resp.primary_finding_card, self.mock_mammo.last_snapshot)
                    self.assertEqual(resp.mammo_status, "SUCCESS")

                    # 2. Clin abstains
                    self.assertEqual(resp.clin_status, "ABSTAINED")
                    card = resp.shadow_finding_card
                    self.assertIsNotNone(card)
                    self.assertEqual(card["agent"], "clin")
                    self.assertEqual(card["metadata"]["status"], "abstained")
                    self.assertEqual(card["metadata"]["voting_eligibility"], False)

                    # 3. Schema validation passes
                    is_valid, err = clin_pipeline.schema_validate(card) if hasattr(clin_pipeline, "schema_validate") else (True, None)
                    # Use validate_finding_card
                    from src.mammo.finding_card import validate_finding_card
                    is_valid, err = validate_finding_card(card)
                    self.assertTrue(is_valid, f"Card schema validation failed: {err}")

                    # 4. Strict JSON serialization passes
                    json_str = json.dumps(card, allow_nan=False)
                    self.assertIsInstance(json_str, str)
                    self.assertNotIn(": NaN", json_str)
                    self.assertNotIn(": Infinity", json_str)
                    self.assertNotIn(": -Infinity", json_str)
        finally:
            orch.close(wait=False)

    def test_unsupported_values_abstention_and_serialization(self):
        """24. Section 4.B: Unsupported values (sets, custom objects, broken __str__) abstain and serialize cleanly."""
        clin_pipeline = ClinicalRiskAgentInferencePipeline(lazy_load=True)
        orch = MultiAgentOrchestrator(mammo_pipeline=self.mock_mammo, clin_pipeline=clin_pipeline)

        class CustomClass:
            def __init__(self, x):
                self.x = x
            def __str__(self):
                return f"CustomClass(x={self.x})"

        class BrokenStrClass:
            def __str__(self):
                raise ValueError("Deliberate __str__ exception")

        raw_set = {10, 20, 30}
        custom_obj = CustomClass("patient_meta")
        broken_obj = BrokenStrClass()

        test_cases = [
            ("set_side", {"breast_density": 2, "breast_side": raw_set}),
            ("custom_side", {"breast_density": 2, "breast_side": custom_obj}),
            ("broken_str_density", {"breast_density": broken_obj, "breast_side": "L"}),
            ("nested_unsupported", {"breast_density": {"meta": {"obj": custom_obj, "set": raw_set}}, "breast_side": "R"}),
        ]

        try:
            for name, covs in test_cases:
                with self.subTest(case=name):
                    req = OrchestrationRequest(
                        pair_identifier=f"P_{name}",
                        cc_input="dummy_cc",
                        mlo_input="dummy_mlo",
                        clinical_covariates=covs
                    )
                    resp = orch.run(req)

                    # Primary Mammo card preserved
                    self.assertIs(resp.primary_finding_card, self.mock_mammo.last_returned_card)
                    self.assertEqual(resp.primary_finding_card, self.mock_mammo.last_snapshot)
                    self.assertEqual(resp.mammo_status, "SUCCESS")

                    # Clin abstains
                    self.assertEqual(resp.clin_status, "ABSTAINED")
                    card = resp.shadow_finding_card
                    self.assertIsNotNone(card)
                    self.assertEqual(card["metadata"]["status"], "abstained")

                    # Schema validation
                    from src.mammo.finding_card import validate_finding_card
                    is_valid, err = validate_finding_card(card)
                    self.assertTrue(is_valid, f"Card schema validation failed: {err}")

                    # Strict JSON serialization
                    json_str = json.dumps(card, allow_nan=False)
                    self.assertIsInstance(json_str, str)

            # Confirm original objects were NOT mutated
            self.assertEqual(raw_set, {10, 20, 30})
            self.assertEqual(custom_obj.x, "patient_meta")
        finally:
            orch.close(wait=False)

    def test_circular_references_abstention_and_serialization(self):
        """25. Section 4.C: Circular references in covariates do not crash and serialize strictly."""
        clin_pipeline = ClinicalRiskAgentInferencePipeline(lazy_load=True)
        orch = MultiAgentOrchestrator(mammo_pipeline=self.mock_mammo, clin_pipeline=clin_pipeline)

        # Build self-referential containers
        circ_list = [1, 2]
        circ_list.append(circ_list)

        circ_dict = {"key": "val"}
        circ_dict["self"] = circ_dict

        test_cases = [
            ("circ_list_side", {"breast_density": 2, "breast_side": circ_list}),
            ("circ_dict_side", {"breast_density": 2, "breast_side": circ_dict}),
            ("circ_dict_density", {"breast_density": circ_dict, "breast_side": "L"}),
            ("mutual_circ", {"breast_density": circ_dict, "breast_side": circ_list}),
        ]

        try:
            for name, covs in test_cases:
                with self.subTest(case=name):
                    req = OrchestrationRequest(
                        pair_identifier=f"P_{name}",
                        cc_input="dummy_cc",
                        mlo_input="dummy_mlo",
                        clinical_covariates=covs
                    )
                    resp = orch.run(req)

                    # Primary Mammo card preserved
                    self.assertIs(resp.primary_finding_card, self.mock_mammo.last_returned_card)
                    self.assertEqual(resp.primary_finding_card, self.mock_mammo.last_snapshot)
                    self.assertEqual(resp.mammo_status, "SUCCESS")

                    # Clin abstains without infinite recursion
                    self.assertEqual(resp.clin_status, "ABSTAINED")
                    card = resp.shadow_finding_card
                    self.assertIsNotNone(card)

                    # Schema validation
                    from src.mammo.finding_card import validate_finding_card
                    is_valid, err = validate_finding_card(card)
                    self.assertTrue(is_valid, f"Card schema validation failed: {err}")

                    # Strict JSON serialization
                    json_str = json.dumps(card, allow_nan=False)
                    self.assertIsInstance(json_str, str)
                    self.assertIn("<circular_ref>", json_str)

            # Assert original inputs are still intact and cyclic
            self.assertIs(circ_list[2], circ_list)
            self.assertIs(circ_dict["self"], circ_dict)
        finally:
            orch.close(wait=False)

    def test_normal_behavior_regression_and_isolation(self):
        """26. Section 4.D & 4.E: Normal validation rules, isolation, and immutability remain strictly enforced."""
        clin_pipeline = ClinicalRiskAgentInferencePipeline(lazy_load=False)
        orch = MultiAgentOrchestrator(mammo_pipeline=self.mock_mammo, clin_pipeline=clin_pipeline)

        try:
            # 1. Missing covariates -> abstains
            r_missing = orch.run(OrchestrationRequest("P_MISSING", "dummy_cc", "dummy_mlo", clinical_covariates={}))
            self.assertEqual(r_missing.clin_status, "ABSTAINED")
            self.assertIs(r_missing.primary_finding_card, self.mock_mammo.last_returned_card)
            self.assertEqual(r_missing.primary_finding_card, self.mock_mammo.last_snapshot)

            # 2. Boolean density -> abstains
            r_bool = orch.run(OrchestrationRequest("P_BOOL", "dummy_cc", "dummy_mlo", clinical_covariates={"breast_density": True, "breast_side": "L"}))
            self.assertEqual(r_bool.clin_status, "ABSTAINED")
            self.assertIs(r_bool.primary_finding_card, self.mock_mammo.last_returned_card)
            self.assertEqual(r_bool.primary_finding_card, self.mock_mammo.last_snapshot)

            # 3. Fractional density -> abstains
            r_frac = orch.run(OrchestrationRequest("P_FRAC", "dummy_cc", "dummy_mlo", clinical_covariates={"breast_density": 2.5, "breast_side": "L"}))
            self.assertEqual(r_frac.clin_status, "ABSTAINED")
            self.assertIs(r_frac.primary_finding_card, self.mock_mammo.last_returned_card)
            self.assertEqual(r_frac.primary_finding_card, self.mock_mammo.last_snapshot)

            # 4. Out of range category -> abstains
            r_oor = orch.run(OrchestrationRequest("P_OOR", "dummy_cc", "dummy_mlo", clinical_covariates={"breast_density": 5, "breast_side": "L"}))
            self.assertEqual(r_oor.clin_status, "ABSTAINED")
            self.assertIs(r_oor.primary_finding_card, self.mock_mammo.last_returned_card)
            self.assertEqual(r_oor.primary_finding_card, self.mock_mammo.last_snapshot)

            # 5. Non-finite density -> abstains
            r_nan = orch.run(OrchestrationRequest("P_NAN", "dummy_cc", "dummy_mlo", clinical_covariates={"breast_density": float("nan"), "breast_side": "L"}))
            self.assertEqual(r_nan.clin_status, "ABSTAINED")
            self.assertIs(r_nan.primary_finding_card, self.mock_mammo.last_returned_card)
            self.assertEqual(r_nan.primary_finding_card, self.mock_mammo.last_snapshot)

            # 6. Valid integer density & side -> succeeds
            r_valid = orch.run(OrchestrationRequest("P_VALID", "dummy_cc", "dummy_mlo", clinical_covariates={"breast_density": 3, "breast_side": "R"}))
            self.assertEqual(r_valid.clin_status, "SUCCESS")
            self.assertEqual(r_valid.shadow_finding_card["metadata"]["status"], "success")
            self.assertEqual(r_valid.shadow_finding_card["metadata"]["voting_eligibility"], False)
            self.assertIs(r_valid.primary_finding_card, self.mock_mammo.last_returned_card)
            self.assertEqual(r_valid.primary_finding_card, self.mock_mammo.last_snapshot)

            # 7. Valid NumPy integer scalar density -> succeeds
            r_np = orch.run(OrchestrationRequest("P_NP", "dummy_cc", "dummy_mlo", clinical_covariates={"breast_density": np.int32(2), "breast_side": "LEFT"}))
            self.assertEqual(r_np.clin_status, "SUCCESS")
            self.assertIs(r_np.primary_finding_card, self.mock_mammo.last_returned_card)
            self.assertEqual(r_np.primary_finding_card, self.mock_mammo.last_snapshot)

            # Verify in all cases primary Mammo card was unmutated and isolated
            for resp in (r_missing, r_bool, r_frac, r_oor, r_nan, r_valid, r_np):
                self.assertNotIn("tree_shap", resp.primary_finding_card["sal"])
                self.assertNotIn("xgboost", json.dumps(resp.primary_finding_card).lower())
        finally:
            orch.close(wait=False)


if __name__ == "__main__":
    unittest.main()
