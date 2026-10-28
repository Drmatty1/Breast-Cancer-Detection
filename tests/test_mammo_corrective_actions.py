"""
Isolated Synthetic Unit and Regression Tests for Mammo Agent Corrective Actions.

Governed by: reports/MAMMO_AGENT_FINAL_COMPLIANCE_AUDIT.md and methodology requirements.

Tests:
1. Fix 1: Abnormality-aware finding card generation for mass-only, calcification-only, mixed, and unknown scenarios.
2. Fix 1: Prevention of hallucinated findings when abnormality presence is unknown or unsupported.
3. Fix 1: Natural language rationale isolation (no combined assertions unless independently supported).
4. Fix 2: Verification that the word 'calibrated' does NOT describe the heuristic BI-RADS confidence score in rationale.
5. Fix 2: Auditing of confidence provenance and preservation of primary malignancy temperature calibration.
6. Fix 3: Inference-only ordinal monotonicity enforcement on synthetic non-monotonic cutoff probabilities.
7. Fix 3: Monotonicity preservation on valid ordinal cutoffs.
8. Fix 3: Checkpoint loadability check confirming existing trained weights remain 100% compatible.

Operates purely on synthetic tensors. Zero inference on evaluation images or quarantined cohorts.
"""
import unittest
from pathlib import Path
import numpy as np
import torch

from src.mammo.finding_card import build_finding_card, validate_finding_card
from src.mammo.inference import enforce_ordinal_monotonicity, MammographyAgentInferencePipeline
from src.models.mammo_agent import MammographyAgentModel


class TestMammoAgentCorrectiveActions(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.ckpt_fold0 = Path(r"D:\rp\reports\mammo_agent_training\checkpoints\mammo_agent_fold0_best.pt")
        self.assertTrue(self.ckpt_fold0.is_file(), f"Missing checkpoint: {self.ckpt_fold0}")

    # =========================================================================
    # FIX 1: Abnormality-Aware Finding Cards
    # =========================================================================

    def test_fix1_mass_only_scenario(self):
        """Verify mass-only abnormality asserts margin, suppresses calcification, and formats rationale."""
        card = build_finding_card(
            binary_prob_calibrated=0.82,
            margin_probs=[0.05, 0.10, 0.15, 0.70],  # argmax: 3 -> spiculated
            calc_probs=[0.10, 0.05, 0.05, 0.75, 0.05],  # argmax: 3 -> pleomorphic (should be ignored)
            birads_ordinal_level=2,
            birads_confidence=0.65,
            abnormality_type="mass"
        )
        is_valid, err = validate_finding_card(card)
        self.assertTrue(is_valid, f"Schema validation error: {err}")

        # Assertions
        self.assertEqual(card["finding"]["margin"], "spiculated")
        self.assertIsNone(card["finding"]["calcification"], "Calcification must be None for mass-only abnormality.")
        self.assertIn("spiculated margin mass", card["rationale"])
        self.assertNotIn("calcification", card["rationale"].lower())

    def test_fix1_calcification_only_scenario(self):
        """Verify calcification-only abnormality asserts morphology, suppresses margin, and formats rationale."""
        card = build_finding_card(
            binary_prob_calibrated=0.45,
            margin_probs=[0.05, 0.10, 0.15, 0.70],  # should be ignored
            calc_probs=[0.05, 0.05, 0.80, 0.05, 0.05],  # argmax: 2 -> fine_linear_branching
            birads_ordinal_level=2,
            birads_confidence=0.58,
            abnormality_type="calcification"
        )
        is_valid, err = validate_finding_card(card)
        self.assertTrue(is_valid, f"Schema validation error: {err}")

        # Assertions
        self.assertIsNone(card["finding"]["margin"], "Margin must be None for calc-only abnormality.")
        self.assertEqual(card["finding"]["calcification"], "fine_linear_branching")
        self.assertIn("fine_linear_branching calcifications", card["rationale"])
        self.assertNotIn("margin mass", card["rationale"].lower())

    def test_fix1_mixed_abnormality_scenario(self):
        """Verify mixed abnormality asserts both margin and morphology when independently supported."""
        card = build_finding_card(
            binary_prob_calibrated=0.91,
            margin_probs=[0.05, 0.85, 0.05, 0.05],  # indistinct
            calc_probs=[0.05, 0.05, 0.05, 0.80, 0.05],  # pleomorphic
            birads_ordinal_level=3,
            birads_confidence=0.75,
            abnormality_type="both"
        )
        is_valid, err = validate_finding_card(card)
        self.assertTrue(is_valid, f"Schema validation error: {err}")

        # Assertions
        self.assertEqual(card["finding"]["margin"], "indistinct")
        self.assertEqual(card["finding"]["calcification"], "pleomorphic")
        self.assertIn("indistinct margin mass with pleomorphic calcifications", card["rationale"])

    def test_fix1_unknown_abnormality_scenario(self):
        """Verify unknown abnormality suppresses both secondary findings and does not hallucinate either."""
        # Both distributions provided from forward pass, but abnormality type is unknown / None
        card = build_finding_card(
            binary_prob_calibrated=0.74,
            margin_probs=[0.10, 0.70, 0.10, 0.10],
            calc_probs=[0.10, 0.10, 0.60, 0.10, 0.10],
            birads_ordinal_level=2,
            birads_confidence=0.62,
            abnormality_type=None  # Unknown / unspecified
        )
        is_valid, err = validate_finding_card(card)
        self.assertTrue(is_valid, f"Schema validation error: {err}")

        # Assertions
        self.assertIsNone(card["finding"]["margin"], "Margin must be None when abnormality presence is unknown.")
        self.assertIsNone(card["finding"]["calcification"], "Calc must be None when abnormality presence is unknown.")
        self.assertEqual(card["finding"]["lesion"], "malignant")
        self.assertIn("malignant mammographic finding", card["rationale"])
        self.assertNotIn("margin mass", card["rationale"].lower())
        self.assertNotIn("calcification", card["rationale"].lower())

    def test_fix1_pipeline_abnormality_gating(self):
        """Verify end-to-end inference pipeline correctly respects abnormality_type parameter."""
        pipeline = MammographyAgentInferencePipeline(
            checkpoint_paths=[self.ckpt_fold0],
            device=self.device
        )
        synth_cc = np.random.uniform(0.0, 1.0, (512, 512)).astype(np.float32)
        synth_mlo = np.random.uniform(0.0, 1.0, (512, 512)).astype(np.float32)

        # 1. Inference with abnormality_type="mass"
        card_mass = pipeline.predict(
            cc_input=synth_cc,
            mlo_input=synth_mlo,
            generate_saliency=False,
            abnormality_type="mass"
        )
        self.assertIsNotNone(card_mass["finding"]["margin"])
        self.assertIsNone(card_mass["finding"]["calcification"])

        # 2. Inference without abnormality_type (unknown)
        card_unknown = pipeline.predict(
            cc_input=synth_cc,
            mlo_input=synth_mlo,
            generate_saliency=False
        )
        self.assertIsNone(card_unknown["finding"]["margin"])
        self.assertIsNone(card_unknown["finding"]["calcification"])

    # =========================================================================
    # FIX 2: Correct Confidence Terminology
    # =========================================================================

    def test_fix2_rationale_confidence_wording(self):
        """Verify rationale text does NOT describe heuristic BI-RADS score as calibrated confidence."""
        card = build_finding_card(
            binary_prob_calibrated=0.88,
            margin_probs=[0.1, 0.2, 0.3, 0.4],
            calc_probs=None,
            birads_ordinal_level=2,
            birads_confidence=0.7401,
            abnormality_type="mass"
        )

        rationale = card["rationale"]
        # Confirm unsupported phrase is removed
        self.assertNotIn("calibrated decision confidence", rationale)
        self.assertNotIn("calibrated confidence", rationale.lower())
        # Confirm accurate description is present
        self.assertIn("decision confidence 0.74", rationale)

    def test_fix2_confidence_provenance_metadata(self):
        """Verify confidence score provenance is accurately recorded in metadata."""
        card = build_finding_card(
            binary_prob_calibrated=0.765432,
            birads_ordinal_level=1,
            birads_confidence=0.6123
        )
        self.assertIn("confidence_provenance", card["metadata"])
        prov = card["metadata"]["confidence_provenance"]
        self.assertEqual(prov["conf_a_source"], "heuristic_ordinal_cutoff_margin")
        self.assertFalse(prov["is_conf_a_calibrated"])
        self.assertTrue(prov["malignancy_calibrated"])
        self.assertAlmostEqual(card["metadata"]["task_probabilities"]["malignancy_calibrated"], 0.765432, places=5)

    # =========================================================================
    # FIX 3: BI-RADS Ordinal Monotonicity Safeguard
    # =========================================================================

    def test_fix3_monotonicity_safeguard_on_violations(self):
        """Verify enforce_ordinal_monotonicity corrects non-monotonic cutoff probabilities."""
        # Synthetic non-monotonic cutoff probabilities: q_0 < q_1
        raw_cutoffs = np.array([0.45, 0.72, 0.20])
        mono_cutoffs, was_violated = enforce_ordinal_monotonicity(raw_cutoffs)

        self.assertTrue(was_violated, "Expected violation flag to be True for non-monotonic cutoffs.")
        # Sequential upper-bounding: q_1* = min(0.45, 0.72) = 0.45
        self.assertAlmostEqual(mono_cutoffs[0], 0.45)
        self.assertAlmostEqual(mono_cutoffs[1], 0.45)
        self.assertAlmostEqual(mono_cutoffs[2], 0.20)
        # Check strict monotonic non-increasing property: q_0 >= q_1 >= q_2
        self.assertTrue(mono_cutoffs[0] >= mono_cutoffs[1] >= mono_cutoffs[2])

    def test_fix3_monotonicity_safeguard_on_valid_inputs(self):
        """Verify enforce_ordinal_monotonicity preserves already monotonic cutoff probabilities."""
        # Valid monotonic sequence: q_0 >= q_1 >= q_2
        valid_cutoffs = np.array([0.80, 0.55, 0.15])
        mono_cutoffs, was_violated = enforce_ordinal_monotonicity(valid_cutoffs)

        self.assertFalse(was_violated, "Expected violation flag to be False for valid cutoffs.")
        np.testing.assert_allclose(mono_cutoffs, valid_cutoffs)

    def test_fix3_checkpoint_architecture_compatibility(self):
        """Verify checkpoint loads with complete state dict compatibility without parameter modification."""
        model = MammographyAgentModel(pretrained_backbone=False)
        ckpt = torch.load(self.ckpt_fold0, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        self.assertEqual(len(missing_keys), 0, f"Missing keys when loading checkpoint: {missing_keys}")
        self.assertEqual(len(unexpected_keys), 0, f"Unexpected keys when loading checkpoint: {unexpected_keys}")


if __name__ == "__main__":
    unittest.main()
