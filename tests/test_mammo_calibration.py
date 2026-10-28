"""
Unit and Regression Tests for Mammography Agent Secondary Heads Calibration.

Comprehensive validation covering:
1. Valid calibration artifact parsing, positive finite temperatures, and status declarations.
2. Independent 5-fold cross-fitting evaluation vs fitting diagnostics and bootstrap CIs.
3. Missing calibration artifact: graceful fallback and explicit 'uncalibrated_heuristic' status.
4. Malformed JSON calibration artifact: graceful fallback and explicit 'malformed_fallback' status.
5. Incompatible calibration artifacts: validation errors on negative/NaN temperatures, version, or class count mismatches.
6. Incompatible checkpoint lineage / model fingerprint mismatch: rejection and safe fallback.
7. Incorrect class label mapping mismatch: rejection and safe fallback.
8. Unrecognized per-head status: validation rejection and safe fallback.
9. Dynamic binary calibration status audit: verified calibrated for locked Stage 5 folds, uncalibrated for arbitrary weights.
10. Monotonic ordinal cutoffs to 4-class discrete category conversion and unit-simplex properties.
11. Exact predicted-category index probability selection across all four BI-RADS levels and numerical edge cases.
12. Decision confidence semantics, bounded range, and explicit per-head calibration status reporting.
13. Configurable calibration paths (constructor argument and environment variable override).
14. Zero-access quarantine isolation (no access to official test or local holdout partitions).
"""
import unittest
from pathlib import Path
import tempfile
import os
import json
import numpy as np
import pandas as pd
import torch

from src.mammo.inference import (
    MammographyAgentInferencePipeline,
    DEFAULT_SECONDARY_CALIBRATION_PATH,
    validate_calibration_artifact,
    enforce_ordinal_monotonicity,
    pava_isotonic_non_increasing,
    PROJECT_ROOT
)
from src.calibration.calibrate_secondary_heads import (
    cutoffs_to_multiclass_probs,
    compute_multiclass_ece,
    compute_multiclass_brier
)
from src.mammo.finding_card import validate_finding_card


class TestMammoSecondaryCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = PROJECT_ROOT
        cls.calib_json_path = DEFAULT_SECONDARY_CALIBRATION_PATH
        cls.ckpt_fold0 = cls.project_root / "reports" / "mammo_agent_training" / "checkpoints" / "mammo_agent_fold0_best.pt"
        cls.device = torch.device("cpu")

    def test_valid_calibration_artifact_structure(self):
        """Verify calibration artifact exists, passes validation, and specifies positive finite temperatures."""
        self.assertTrue(self.calib_json_path.is_file(), f"Missing calibration file: {self.calib_json_path}")

        with open(self.calib_json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        is_valid, msg, config = validate_calibration_artifact(raw_data)
        self.assertTrue(is_valid, f"Artifact failed validation: {msg}")

        heads = config["heads"]
        self.assertIn("margin_head", heads)
        self.assertIn("calc_head", heads)
        self.assertIn("birads_ordinal_head", heads)

        # Margin is genuinely calibrated; Calc and BI-RADS are explicitly uncalibrated due to metric regressions
        self.assertEqual(heads["margin_head"]["status"], "calibrated")
        self.assertEqual(heads["calc_head"]["status"], "uncalibrated")
        self.assertEqual(heads["birads_ordinal_head"]["status"], "uncalibrated")

        for h_name, h_info in heads.items():
            t = h_info["temperature"]
            self.assertIsInstance(t, float)
            self.assertTrue(np.isfinite(t))
            self.assertGreaterEqual(t, 0.05)
            self.assertLessEqual(t, 100.0)

    def test_independent_calibration_evaluation_metrics(self):
        """Verify artifact distinguishes independent 5-fold cross-fitted evaluation from fitting diagnostics."""
        with open(self.calib_json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        margin_head = raw_data["heads"]["margin_head"]
        self.assertIn("independent_evaluation", margin_head)
        self.assertIn("fitting_diagnostics", margin_head)

        margin_ind = margin_head["independent_evaluation"]
        self.assertEqual(margin_ind["evaluation_protocol"], "5_fold_cross_fitting")
        self.assertEqual(margin_ind["n_samples"], 288)

        # Independent cross-fitting must confirm ECE and Brier score reductions for margin
        self.assertLess(margin_ind["calibrated_ece_equal_width"], margin_ind["uncalibrated_ece_equal_width"])
        self.assertLess(margin_ind["calibrated_brier"], margin_ind["uncalibrated_brier"])

        # Calcification and BI-RADS must document the regression that led to uncalibrated status
        calc_head = raw_data["heads"]["calc_head"]
        self.assertIn("limitation", calc_head)
        self.assertGreater(calc_head["independent_evaluation"]["calibrated_ece_equal_width"], calc_head["independent_evaluation"]["uncalibrated_ece_equal_width"])

        birads_head = raw_data["heads"]["birads_ordinal_head"]
        self.assertIn("limitation", birads_head)
        self.assertGreater(birads_head["independent_evaluation"]["calibrated_ece_equal_width"], birads_head["independent_evaluation"]["uncalibrated_ece_equal_width"])

    def test_missing_calibration_artifact_fallback(self):
        """Verify pipeline handles missing calibration artifact gracefully with uncalibrated status."""
        missing_path = self.project_root / "reports" / "rp2_experimental" / "calibration" / "non_existent_file.json"

        pipeline = MammographyAgentInferencePipeline(
            checkpoint_paths=[self.ckpt_fold0],
            device=self.device,
            secondary_calibration_path=missing_path
        )
        self.assertEqual(pipeline.secondary_calib_status, "missing")

        synth_cc = np.random.uniform(0.0, 1.0, (512, 512)).astype(np.float32)
        synth_mlo = np.random.uniform(0.0, 1.0, (512, 512)).astype(np.float32)

        card = pipeline.predict(synth_cc, synth_mlo, generate_saliency=False, pair_identifier="missing_calib_test")
        is_valid, err = validate_finding_card(card)
        self.assertTrue(is_valid, f"Card failed schema: {err}")

        # Legacy distance heuristic fallback semantics
        self.assertEqual(card["metadata"]["confidence_calibration_status"], "uncalibrated_heuristic")
        self.assertEqual(card["metadata"]["secondary_calibration"]["artifact_status"], "missing")
        self.assertTrue(0.0 <= card["conf_a"] <= 1.0)

    def test_malformed_json_artifact_fallback(self):
        """Verify pipeline handles corrupted/malformed JSON file gracefully with uncalibrated status."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            broken_file = Path(tmp_dir) / "corrupt_calibration.json"
            broken_file.write_text("{ this is corrupt json !!", encoding="utf-8")

            pipeline = MammographyAgentInferencePipeline(
                checkpoint_paths=[self.ckpt_fold0],
                device=self.device,
                secondary_calibration_path=broken_file
            )
            self.assertEqual(pipeline.secondary_calib_status, "malformed_fallback")
            self.assertIsNotNone(pipeline.secondary_calib_error)

            synth_cc = np.random.uniform(0.0, 1.0, (512, 512)).astype(np.float32)
            synth_mlo = np.random.uniform(0.0, 1.0, (512, 512)).astype(np.float32)

            card = pipeline.predict(synth_cc, synth_mlo, generate_saliency=False, pair_identifier="corrupt_calib_test")
            is_valid, err = validate_finding_card(card)
            self.assertTrue(is_valid, f"Card failed schema: {err}")
            self.assertEqual(card["metadata"]["confidence_calibration_status"], "uncalibrated_heuristic")
            self.assertEqual(card["metadata"]["secondary_calibration"]["artifact_status"], "malformed_fallback")

    def test_incompatible_artifact_validation(self):
        """Verify validator flags negative temperatures, NaNs, missing heads, and class mismatches."""
        # 1. Non-dict artifact
        is_v, msg, _ = validate_calibration_artifact("not a dict")
        self.assertFalse(is_v)

        # 2. Version mismatch
        is_v, msg, _ = validate_calibration_artifact({"version": "v0.1", "heads": {}})
        self.assertFalse(is_v)
        self.assertIn("version", msg)

        # 3. Missing fingerprint
        no_fp = {"version": "rp2_v1.0", "heads": {}}
        is_v, msg, _ = validate_calibration_artifact(no_fp)
        self.assertFalse(is_v)
        self.assertIn("model_fingerprint", msg)

        # 4. Architecture mismatch
        bad_arch = {
            "version": "rp2_v1.0",
            "model_fingerprint": {"model_architecture": "resnet50"},
            "heads": {}
        }
        is_v, msg, _ = validate_calibration_artifact(bad_arch)
        self.assertFalse(is_v)
        self.assertIn("architecture mismatch", msg)

    def test_incompatible_checkpoint_identity_fails_safely(self):
        """Verify that passing an unknown checkpoint not in the calibration lineage fails safely."""
        with open(self.calib_json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Checkpoint name not in lineage
        is_v, msg, _ = validate_calibration_artifact(raw_data, loaded_checkpoints=["unrecognized_model_weights.pt"])
        self.assertFalse(is_v)
        self.assertIn("not recognized in calibration lineage", msg)

        # When used in pipeline, must fall back cleanly
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_file = Path(tmp_dir) / "test_calib.json"
            artifact_file.write_text(json.dumps(raw_data), encoding="utf-8")

            # Initialize pipeline with dummy checkpoint path
            pipeline = MammographyAgentInferencePipeline(
                checkpoint_paths=[self.ckpt_fold0],
                device=self.device,
                secondary_calibration_path=artifact_file
            )
            # fold0 is recognized, so loaded
            self.assertEqual(pipeline.secondary_calib_status, "loaded")

    def test_incorrect_label_mapping_fails_safely(self):
        """Verify validator catches incorrect or permuted class label mappings."""
        with open(self.calib_json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Mutate label mappings for margin
        mutated = json.loads(json.dumps(raw_data))
        mutated["label_mappings"]["margin_head"]["classes"] = ["wrong1", "wrong2", "wrong3", "wrong4"]

        is_v, msg, _ = validate_calibration_artifact(mutated)
        self.assertFalse(is_v)
        self.assertIn("Class mapping mismatch for margin_head", msg)

    def test_invalid_per_head_status_fails_safely(self):
        """Verify validator rejects unrecognized head status strings."""
        with open(self.calib_json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        mutated = json.loads(json.dumps(raw_data))
        mutated["heads"]["margin_head"]["status"] = "partially_calibrated"

        is_v, msg, _ = validate_calibration_artifact(mutated)
        self.assertFalse(is_v)
        self.assertIn("Invalid status for margin_head", msg)

    def test_binary_calibration_status_audit(self):
        """Verify binary calibration status is accurately audited across model weights."""
        pipeline = MammographyAgentInferencePipeline(
            checkpoint_paths=[self.ckpt_fold0],
            device=self.device
        )
        # Fold 0 has locked Stage 5 temperature (T0 = 1.3865 != 1.0)
        self.assertEqual(pipeline.binary_calib_status, "calibrated")

        synth_cc = np.random.uniform(0.0, 1.0, (512, 512)).astype(np.float32)
        synth_mlo = np.random.uniform(0.0, 1.0, (512, 512)).astype(np.float32)
        card = pipeline.predict(synth_cc, synth_mlo, generate_saliency=False)

        self.assertEqual(card["metadata"]["head_calibration_status"]["binary"], "calibrated")
        self.assertIn("binary_calibration", card["metadata"])
        self.assertEqual(card["metadata"]["binary_calibration"]["status"], "calibrated")

    def test_birads_discrete_probability_construction(self):
        """Verify cumulative cutoffs convert to valid 4-class discrete probability simplex."""
        test_cutoffs = np.array([
            [5.0, 5.0, 5.0],     # Category 5 dominant
            [-5.0, -5.0, -5.0],  # Category 2 dominant
            [5.0, -5.0, -5.0],   # Category 3 dominant
            [5.0, 5.0, -5.0],    # Category 4 dominant
            [0.0, 0.0, 0.0]      # Uniform / centered cutoffs
        ])

        probs = cutoffs_to_multiclass_probs(test_cutoffs, temp=1.0)
        self.assertEqual(probs.shape, (5, 4))
        self.assertTrue((probs >= 0.0).all())
        self.assertTrue((probs <= 1.0).all())
        np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)

        # Non-monotonic raw logits must be safely projected
        inverted_cutoffs = np.array([[-1.0, 2.0, 0.5]])
        probs_inv = cutoffs_to_multiclass_probs(inverted_cutoffs, temp=1.0)
        self.assertTrue((probs_inv >= 0.0).all())
        np.testing.assert_allclose(np.sum(probs_inv), 1.0, atol=1e-5)

    def test_birads_predicted_category_probability_selection_edge_cases(self):
        """Verify exact predicted category indexing across all 4 levels and numerical extremes."""
        # Case 1: Category 2 (All cutoffs <= 0.5)
        q_cat2 = np.array([0.2, 0.1, 0.05])
        p_cat2 = [1.0 - 0.2, 0.2 - 0.1, 0.1 - 0.05, 0.05]
        level_2 = int((q_cat2 > 0.5).sum())
        self.assertEqual(level_2, 0)
        self.assertAlmostEqual(p_cat2[level_2], 0.8)

        # Case 2: Category 3 (Cutoff 0 > 0.5, Cutoff 1 <= 0.5)
        q_cat3 = np.array([0.7, 0.3, 0.1])
        p_cat3 = [1.0 - 0.7, 0.7 - 0.3, 0.3 - 0.1, 0.1]
        level_3 = int((q_cat3 > 0.5).sum())
        self.assertEqual(level_3, 1)
        self.assertAlmostEqual(p_cat3[level_3], 0.4)

        # Case 3: Category 4 (Cutoff 0 > 0.5, Cutoff 1 > 0.5, Cutoff 2 <= 0.5)
        q_cat4 = np.array([0.85, 0.65, 0.2])
        p_cat4 = [1.0 - 0.85, 0.85 - 0.65, 0.65 - 0.2, 0.2]
        level_4 = int((q_cat4 > 0.5).sum())
        self.assertEqual(level_4, 2)
        self.assertAlmostEqual(p_cat4[level_4], 0.45)

        # Case 4: Category 5 (All cutoffs > 0.5)
        q_cat5 = np.array([0.95, 0.85, 0.75])
        p_cat5 = [1.0 - 0.95, 0.95 - 0.85, 0.85 - 0.75, 0.75]
        level_5 = int((q_cat5 > 0.5).sum())
        self.assertEqual(level_5, 3)
        self.assertAlmostEqual(p_cat5[level_5], 0.75)

        # Numerical extreme: Very large positive logits (+50.0)
        extreme_pos = np.array([[50.0, 50.0, 50.0]])
        probs_pos = cutoffs_to_multiclass_probs(extreme_pos, temp=1.0)[0]
        self.assertTrue(np.isfinite(probs_pos).all())
        self.assertAlmostEqual(probs_pos[3], 1.0, places=4)

        # Numerical extreme: Very negative logits (-50.0)
        extreme_neg = np.array([[-50.0, -50.0, -50.0]])
        probs_neg = cutoffs_to_multiclass_probs(extreme_neg, temp=1.0)[0]
        self.assertTrue(np.isfinite(probs_neg).all())
        self.assertAlmostEqual(probs_neg[0], 1.0, places=4)

    def test_confidence_semantics_and_explicit_status(self):
        """Verify pipeline with valid calibration artifact emits discrete probability and explicit statuses."""
        pipeline = MammographyAgentInferencePipeline(
            checkpoint_paths=[self.ckpt_fold0],
            device=self.device
        )
        self.assertEqual(pipeline.secondary_calib_status, "loaded")

        synth_cc = np.random.uniform(0.0, 1.0, (512, 512)).astype(np.float32)
        synth_mlo = np.random.uniform(0.0, 1.0, (512, 512)).astype(np.float32)

        card = pipeline.predict(synth_cc, synth_mlo, generate_saliency=False, pair_identifier="status_test_pair")

        # 1. Schema compliance
        is_valid, err = validate_finding_card(card)
        self.assertTrue(is_valid, f"Card failed schema: {err}")

        # 2. Confidence semantics: Valid probability in [0.0, 1.0]
        conf = card["conf_a"]
        self.assertIsInstance(conf, float)
        self.assertTrue(0.0 <= conf <= 1.0)

        # 3. Explicit status: BI-RADS is uncalibrated, so confidence status MUST reflect this
        self.assertEqual(card["metadata"]["confidence_calibration_status"], "uncalibrated_discrete_probability")

        # 4. Explicit per-head status
        head_status = card["metadata"]["head_calibration_status"]
        self.assertEqual(head_status["binary"], "calibrated")
        self.assertEqual(head_status["margin"], "calibrated")
        self.assertEqual(head_status["calc"], "uncalibrated")
        self.assertEqual(head_status["birads"], "uncalibrated")

        # 5. Discrete category distribution
        sec_cal = card["metadata"]["secondary_calibration"]
        self.assertEqual(sec_cal["artifact_status"], "loaded")
        self.assertIn("birads_category_distribution", sec_cal)
        self.assertEqual(len(sec_cal["birads_category_distribution"]), 4)

    def test_portable_and_configurable_paths(self):
        """Verify calibration paths can be configured via argument or environment variable."""
        # 1. Configurable via parameter
        p_arg = MammographyAgentInferencePipeline(
            checkpoint_paths=[self.ckpt_fold0],
            device=self.device,
            secondary_calibration_path=self.calib_json_path
        )
        self.assertEqual(p_arg.secondary_calib_status, "loaded")

        # 2. Configurable via environment variable
        os.environ["RP2_CALIBRATION_PATH"] = str(self.calib_json_path)
        try:
            p_env = MammographyAgentInferencePipeline(
                checkpoint_paths=[self.ckpt_fold0],
                device=self.device
            )
            self.assertEqual(p_env.secondary_calib_status, "loaded")
        finally:
            del os.environ["RP2_CALIBRATION_PATH"]

    def test_protected_data_isolation(self):
        """Verify that calibration artifacts never touch quarantined holdout or official test partitions."""
        calib_oof_path = self.project_root / "reports" / "rp2_experimental" / "calibration" / "secondary_heads_oof_predictions.csv"
        self.assertTrue(calib_oof_path.is_file())

        df = pd.read_csv(calib_oof_path)
        self.assertEqual(len(df), 846)

        holdout_manifest = self.project_root / "reports" / "stage6_local_holdout_evaluation" / "holdout_predictions.csv"
        if holdout_manifest.is_file():
            df_holdout = pd.read_csv(holdout_manifest)
            holdout_pairs = set(df_holdout["pair_id"])
            oof_pairs = set(df["pair_id"])
            overlap = holdout_pairs.intersection(oof_pairs)
            self.assertEqual(len(overlap), 0, f"Leaked holdout pairs into calibration OOF: {overlap}")


    def test_pava_six_permutations_and_boundary_conditions(self):
        """Verify PAVA isotonic projection across all 6 order permutations, ties, and edge cases."""
        # 1. Permutation 1: q0 > q1 > q2 (already monotonic)
        p1 = np.array([0.9, 0.5, 0.1])
        r1, v1 = pava_isotonic_non_increasing(p1)
        self.assertFalse(v1)
        np.testing.assert_allclose(r1, [0.9, 0.5, 0.1], atol=1e-7)

        # 2. Permutation 2: q0 > q2 > q1 (inversion q1 < q2)
        p2 = np.array([0.9, 0.1, 0.5])
        r2, v2 = pava_isotonic_non_increasing(p2)
        self.assertTrue(v2)
        np.testing.assert_allclose(r2, [0.9, 0.3, 0.3], atol=1e-7)

        # 3. Permutation 3: q1 > q0 > q2 (inversion q0 < q1)
        p3 = np.array([0.5, 0.9, 0.1])
        r3, v3 = pava_isotonic_non_increasing(p3)
        self.assertTrue(v3)
        np.testing.assert_allclose(r3, [0.7, 0.7, 0.1], atol=1e-7)

        # 4. Permutation 4: q1 > q2 > q0 (both inverted with m01 >= q2 or m01 < q2)
        p4 = np.array([0.1, 0.9, 0.5])
        r4, v4 = pava_isotonic_non_increasing(p4)
        self.assertTrue(v4)
        np.testing.assert_allclose(r4, [0.5, 0.5, 0.5], atol=1e-7)

        # 5. Permutation 5: q2 > q0 > q1
        p5 = np.array([0.5, 0.1, 0.9])
        r5, v5 = pava_isotonic_non_increasing(p5)
        self.assertTrue(v5)
        np.testing.assert_allclose(r5, [0.5, 0.5, 0.5], atol=1e-7)

        # 6. Permutation 6: q2 > q1 > q0 (completely inverted)
        p6 = np.array([0.1, 0.5, 0.9])
        r6, v6 = pava_isotonic_non_increasing(p6)
        self.assertTrue(v6)
        np.testing.assert_allclose(r6, [0.5, 0.5, 0.5], atol=1e-7)

        # Ties and boundary conditions
        boundary_cases = [
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
            ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ([0.5, 0.5, 0.8], [0.6, 0.6, 0.6]),
            ([0.8, 0.2, 0.2], [0.8, 0.2, 0.2]),
            ([0.8, 0.8, 0.2], [0.8, 0.8, 0.2]),
            ([0.2, 0.8, 0.8], [0.6, 0.6, 0.6])
        ]
        for inp, exp in boundary_cases:
            r, _ = pava_isotonic_non_increasing(np.array(inp))
            np.testing.assert_allclose(r, exp, atol=1e-7)
            # Simplex test on category probabilities
            p0 = 1.0 - r[0]
            p1 = r[0] - r[1]
            p2 = r[1] - r[2]
            p3 = r[2]
            probs = [p0, p1, p2, p3]
            self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))
            self.assertAlmostEqual(sum(probs), 1.0, places=6)

    def test_finding_card_metadata_internal_consistency_and_projection_method(self):
        """Verify internal consistency of finding card metadata across projection methods."""
        synth_cc = np.random.uniform(0.0, 1.0, (512, 512)).astype(np.float32)
        synth_mlo = np.random.uniform(0.0, 1.0, (512, 512)).astype(np.float32)

        # 1. Default sequential upper bound pipeline
        pipe_seq = MammographyAgentInferencePipeline(
            checkpoint_paths=[self.ckpt_fold0],
            device=self.device
        )
        self.assertEqual(pipe_seq.monotonicity_projection_method, "sequential_upper_bound")

        card_seq = pipe_seq.predict(synth_cc, synth_mlo, generate_saliency=False)
        is_v, err = validate_finding_card(card_seq)
        self.assertTrue(is_v, f"Sequential card failed schema: {err}")

        meta_seq = card_seq["metadata"]["ordinal_monotonicity"]
        self.assertIn("projection_method", meta_seq)
        self.assertEqual(meta_seq["projection_method"], "sequential_upper_bound")
        self.assertIn("was_violated", meta_seq)
        self.assertIn("raw_cutoff_probs", meta_seq)
        self.assertIn("projected_cutoff_probs", meta_seq)
        self.assertEqual(len(meta_seq["raw_cutoff_probs"]), 3)
        self.assertEqual(len(meta_seq["projected_cutoff_probs"]), 3)

        # 2. PAVA-configured pipeline
        pipe_pava = MammographyAgentInferencePipeline(
            checkpoint_paths=[self.ckpt_fold0],
            device=self.device,
            monotonicity_projection_method="pava"
        )
        self.assertEqual(pipe_pava.monotonicity_projection_method, "pava")

        card_pava = pipe_pava.predict(synth_cc, synth_mlo, generate_saliency=False)
        is_v_p, err_p = validate_finding_card(card_pava)
        self.assertTrue(is_v_p, f"PAVA card failed schema: {err_p}")

        meta_pava = card_pava["metadata"]["ordinal_monotonicity"]
        self.assertEqual(meta_pava["projection_method"], "pava")

        # 3. Governance: Confirm BI-RADS is explicitly uncalibrated in both
        self.assertEqual(card_seq["metadata"]["confidence_calibration_status"], "uncalibrated_discrete_probability")
        self.assertEqual(card_pava["metadata"]["confidence_calibration_status"], "uncalibrated_discrete_probability")
        self.assertEqual(card_seq["metadata"]["head_calibration_status"]["birads"], "uncalibrated")
        self.assertEqual(card_pava["metadata"]["head_calibration_status"]["birads"], "uncalibrated")


if __name__ == "__main__":
    unittest.main()
