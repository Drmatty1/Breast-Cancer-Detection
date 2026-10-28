"""
Unit tests for the RP1 demo harness (scripts/demo_mammo_clin.py).

Verifies:
1. Canonical pair/path binding and manifest resolution.
2. Frozen Variant A preprocessing contract enforcement ((512, 512), float32, finite, [0.0, 1.0]).
3. Fail-closed quarantine verification against missing, malformed, duplicate, or conflicting manifests.
4. Artifact collision protection without altering existing project files.
5. Real try/finally orchestrator shutdown across the harness execution path.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from scripts.demo_mammo_clin import (
    verify_development_membership,
    resolve_pair_image_paths,
    validate_image_array_from_disk,
    prepare_saliency_output_dir,
    run_demo,
    PROJECT_ROOT,
    DEFAULT_PAIR_ID,
    DEFAULT_CC_PATH,
    DEFAULT_MLO_PATH,
)
from src.orchestration.multi_agent_orchestrator import MultiAgentOrchestrator


class TestDemoHarness(unittest.TestCase):
    """Test suite for demo harness safety, verification, contract, and lifecycle cleanup."""

    # -------------------------------------------------------------------------
    # 1. Split and Quarantine Verification Tests
    # -------------------------------------------------------------------------

    def test_verify_development_membership_success(self):
        """1. Verified development sample P_00008_LEFT passes all quarantine and split checks."""
        verif = verify_development_membership(DEFAULT_PAIR_ID, PROJECT_ROOT)
        self.assertEqual(verif["pair_id"], "P_00008_LEFT")
        self.assertEqual(verif["patient_id"], "P_00008")
        self.assertEqual(verif["breast_side"], "LEFT")
        self.assertEqual(verif["partition"], "train")
        self.assertEqual(verif["outer_fold"], 1)
        self.assertTrue(verif["is_development"])
        self.assertFalse(verif["is_local_holdout"])
        self.assertFalse(verif["is_quarantined_test"])

        # Confirm bound paths
        self.assertEqual(verif["cc_path"], DEFAULT_CC_PATH)
        self.assertEqual(verif["mlo_path"], DEFAULT_MLO_PATH)
        self.assertTrue(verif["cc_path"].is_file())
        self.assertTrue(verif["mlo_path"].is_file())

    def test_verify_development_membership_rejects_nonexistent_pair(self):
        """2. Non-existent pair raises ValueError (fail-closed)."""
        with self.assertRaises(ValueError) as ctx:
            verify_development_membership("NONEXISTENT_PAIR_ID_LEFT", PROJECT_ROOT)
        self.assertIn("not found in canonical pair manifest", str(ctx.exception))

    def test_verify_development_membership_rejects_missing_manifest(self):
        """3. Missing canonical manifest or test metadata file raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            empty_root = Path(tmp_dir)
            with self.assertRaises(FileNotFoundError) as ctx:
                verify_development_membership(DEFAULT_PAIR_ID, empty_root)
            self.assertIn("Required verification manifest missing", str(ctx.exception))

    def test_verify_development_membership_rejects_malformed_test_manifest(self):
        """4. Test metadata missing patient_id column raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_root = Path(tmp_dir)
            # Create manifest hierarchy
            m_dir = fake_root / "data" / "manifests" / "stage2"
            s_dir = fake_root / "reports" / "stage5_cross_validation_plan"
            t_dir = fake_root / "data" / "source_metadata"
            m_dir.mkdir(parents=True)
            s_dir.mkdir(parents=True)
            t_dir.mkdir(parents=True)

            # Copy or write valid primary manifests
            pd.DataFrame([{"pair_id": "P_99999_LEFT", "patient_id": "P_99999", "breast_side": "LEFT", "partition": "train"}]).to_csv(m_dir / "cohort_pairs.csv", index=False)
            pd.DataFrame([{"patient_id": "P_99999", "partition": "train"}]).to_csv(m_dir / "patient_splits.csv", index=False)
            pd.DataFrame([{"pair_id": "P_99999_LEFT", "outer_fold": 0}]).to_csv(s_dir / "cv_splits_manifest.csv", index=False)
            pd.DataFrame([{"patient_id": "P_00001"}]).to_csv(t_dir / "calc_case_description_test_set.csv", index=False)
            # Malformed mass test manifest without patient_id
            pd.DataFrame([{"wrong_col": "P_00002"}]).to_csv(t_dir / "mass_case_description_test_set.csv", index=False)

            with self.assertRaises(ValueError) as ctx:
                verify_development_membership("P_99999_LEFT", fake_root)
            self.assertIn("missing 'patient_id' column", str(ctx.exception))

    def test_verify_development_membership_rejects_duplicate_rows(self):
        """5. Duplicate records for a pair in canonical manifests raise ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_root = Path(tmp_dir)
            m_dir = fake_root / "data" / "manifests" / "stage2"
            s_dir = fake_root / "reports" / "stage5_cross_validation_plan"
            t_dir = fake_root / "data" / "source_metadata"
            m_dir.mkdir(parents=True)
            s_dir.mkdir(parents=True)
            t_dir.mkdir(parents=True)

            # Write duplicate rows for pair
            dup_cohort = pd.DataFrame([
                {"pair_id": "P_99999_LEFT", "patient_id": "P_99999", "breast_side": "LEFT", "partition": "train"},
                {"pair_id": "P_99999_LEFT", "patient_id": "P_99999", "breast_side": "LEFT", "partition": "train"},
            ])
            dup_cohort.to_csv(m_dir / "cohort_pairs.csv", index=False)
            pd.DataFrame([{"patient_id": "P_99999", "partition": "train"}]).to_csv(m_dir / "patient_splits.csv", index=False)
            pd.DataFrame([{"pair_id": "P_99999_LEFT", "outer_fold": 0}]).to_csv(s_dir / "cv_splits_manifest.csv", index=False)
            pd.DataFrame([{"patient_id": "P_00001"}]).to_csv(t_dir / "calc_case_description_test_set.csv", index=False)
            pd.DataFrame([{"patient_id": "P_00002"}]).to_csv(t_dir / "mass_case_description_test_set.csv", index=False)

            with self.assertRaises(ValueError) as ctx:
                verify_development_membership("P_99999_LEFT", fake_root)
            self.assertIn("duplicate records", str(ctx.exception))

    def test_verify_development_membership_rejects_conflicting_partitions(self):
        """6. Conflicting partition assignments between cohort and patient manifests raise ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_root = Path(tmp_dir)
            m_dir = fake_root / "data" / "manifests" / "stage2"
            s_dir = fake_root / "reports" / "stage5_cross_validation_plan"
            t_dir = fake_root / "data" / "source_metadata"
            m_dir.mkdir(parents=True)
            s_dir.mkdir(parents=True)
            t_dir.mkdir(parents=True)

            pd.DataFrame([{"pair_id": "P_99999_LEFT", "patient_id": "P_99999", "breast_side": "LEFT", "partition": "train"}]).to_csv(m_dir / "cohort_pairs.csv", index=False)
            # Conflicting partition in patient_splits
            pd.DataFrame([{"patient_id": "P_99999", "partition": "dev"}]).to_csv(m_dir / "patient_splits.csv", index=False)
            pd.DataFrame([{"pair_id": "P_99999_LEFT", "outer_fold": 0}]).to_csv(s_dir / "cv_splits_manifest.csv", index=False)
            pd.DataFrame([{"patient_id": "P_00001"}]).to_csv(t_dir / "calc_case_description_test_set.csv", index=False)
            pd.DataFrame([{"patient_id": "P_00002"}]).to_csv(t_dir / "mass_case_description_test_set.csv", index=False)

            with self.assertRaises(ValueError) as ctx:
                verify_development_membership("P_99999_LEFT", fake_root)
            self.assertIn("Conflicting partition classification", str(ctx.exception))

    def test_verify_development_membership_rejects_holdout_or_test_patient(self):
        """7. Patients belonging to local_holdout or official test set raise PermissionError."""
        # Holdout patient P_00001
        with self.assertRaises(PermissionError) as ctx:
            verify_development_membership("P_00001_LEFT", PROJECT_ROOT)
        self.assertIn("SECURITY VIOLATION", str(ctx.exception))

        # Test set patient P_00038 disguised in a fake manifest directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_root = Path(tmp_dir)
            m_dir = fake_root / "data" / "manifests" / "stage2"
            s_dir = fake_root / "reports" / "stage5_cross_validation_plan"
            t_dir = fake_root / "data" / "source_metadata"
            m_dir.mkdir(parents=True)
            s_dir.mkdir(parents=True)
            t_dir.mkdir(parents=True)

            pd.DataFrame([{"pair_id": "P_00038_LEFT", "patient_id": "P_00038", "breast_side": "LEFT", "partition": "train"}]).to_csv(m_dir / "cohort_pairs.csv", index=False)
            pd.DataFrame([{"patient_id": "P_00038", "partition": "train"}]).to_csv(m_dir / "patient_splits.csv", index=False)
            pd.DataFrame([{"pair_id": "P_00038_LEFT", "outer_fold": 0}]).to_csv(s_dir / "cv_splits_manifest.csv", index=False)
            pd.DataFrame([{"patient_id": "P_00038"}]).to_csv(t_dir / "calc_case_description_test_set.csv", index=False)
            pd.DataFrame([{"patient_id": "P_00002"}]).to_csv(t_dir / "mass_case_description_test_set.csv", index=False)

            with self.assertRaises(PermissionError) as ctx:
                verify_development_membership("P_00038_LEFT", fake_root)
            self.assertIn("quarantined official test metadata", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 2. Pair and Image Path Binding Tests
    # -------------------------------------------------------------------------

    def test_pair_path_binding_rejects_mismatched_paths(self):
        """8. Providing an image path belonging to a different pair/patient raises ValueError."""
        # Attempting to run demo for P_00008_LEFT using a fake different path
        mismatched_cc = PROJECT_ROOT / "data" / "processed" / "stage3b_comparison" / "A" / "P_00021" / "LEFT" / "CC.npy"
        with self.assertRaises(ValueError) as ctx:
            run_demo(
                pair_id=DEFAULT_PAIR_ID,
                cc_path=mismatched_cc,
                generate_saliency=False,
                test_failure_simulation=False,
            )
        self.assertIn("Path mismatch: provided cc_path", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 3. Preprocessing Contract Validation Tests
    # -------------------------------------------------------------------------

    def test_validate_image_array_from_disk_success(self):
        """9. Verified CC and MLO arrays from disk pass all dimension, dtype, range, and finiteness checks."""
        cc = validate_image_array_from_disk(DEFAULT_CC_PATH)
        self.assertEqual(cc.shape, (512, 512))
        self.assertEqual(cc.dtype, np.float32)
        self.assertTrue(np.isfinite(cc).all())
        self.assertGreaterEqual(float(cc.min()), 0.0)
        self.assertLessEqual(float(cc.max()), 1.0)

        mlo = validate_image_array_from_disk(DEFAULT_MLO_PATH)
        self.assertEqual(mlo.shape, (512, 512))
        self.assertEqual(mlo.dtype, np.float32)
        self.assertTrue(np.isfinite(mlo).all())
        self.assertGreaterEqual(float(mlo.min()), 0.0)
        self.assertLessEqual(float(mlo.max()), 1.0)

    def test_validate_image_array_from_disk_rejects_wrong_shape(self):
        """10. Array with incorrect dimensions raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            tmp_path = Path(f.name)
            np.save(tmp_path, np.zeros((256, 256), dtype=np.float32))

        try:
            with self.assertRaises(ValueError) as ctx:
                validate_image_array_from_disk(tmp_path)
            self.assertIn("shape mismatch", str(ctx.exception))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_validate_image_array_from_disk_rejects_wrong_dtype(self):
        """11. Array with float64 or non-float32 dtype raises TypeError."""
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            tmp_path = Path(f.name)
            np.save(tmp_path, np.zeros((512, 512), dtype=np.float64))

        try:
            with self.assertRaises(TypeError) as ctx:
                validate_image_array_from_disk(tmp_path)
            self.assertIn("must be float32 dtype", str(ctx.exception))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_validate_image_array_from_disk_rejects_below_range(self):
        """12. Array with values below 0.0 raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            tmp_path = Path(f.name)
            arr = np.zeros((512, 512), dtype=np.float32)
            arr[0, 0] = -0.01
            np.save(tmp_path, arr)

        try:
            with self.assertRaises(ValueError) as ctx:
                validate_image_array_from_disk(tmp_path)
            self.assertIn("values out of bounds [0.0, 1.0]", str(ctx.exception))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_validate_image_array_from_disk_rejects_above_range(self):
        """13. Array with values above 1.0 raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            tmp_path = Path(f.name)
            arr = np.ones((512, 512), dtype=np.float32)
            arr[0, 0] = 1.05
            np.save(tmp_path, arr)

        try:
            with self.assertRaises(ValueError) as ctx:
                validate_image_array_from_disk(tmp_path)
            self.assertIn("values out of bounds [0.0, 1.0]", str(ctx.exception))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_validate_image_array_from_disk_rejects_non_finite_nan(self):
        """14. Array with NaN raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            tmp_path = Path(f.name)
            arr = np.full((512, 512), 0.5, dtype=np.float32)
            arr[10, 10] = np.nan
            np.save(tmp_path, arr)

        try:
            with self.assertRaises(ValueError) as ctx:
                validate_image_array_from_disk(tmp_path)
            self.assertIn("contains non-finite values", str(ctx.exception))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_validate_image_array_from_disk_rejects_non_finite_inf(self):
        """15. Array with Inf raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            tmp_path = Path(f.name)
            arr = np.full((512, 512), 0.5, dtype=np.float32)
            arr[10, 10] = np.inf
            np.save(tmp_path, arr)

        try:
            with self.assertRaises(ValueError) as ctx:
                validate_image_array_from_disk(tmp_path)
            self.assertIn("contains non-finite values", str(ctx.exception))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_validate_image_array_from_disk_rejects_non_npy(self):
        """16. Non-npy extension raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            tmp_path = Path(f.name)
            tmp_path.write_text("invalid")

        try:
            with self.assertRaises(ValueError) as ctx:
                validate_image_array_from_disk(tmp_path)
            self.assertIn("Expected .npy array file", str(ctx.exception))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    # -------------------------------------------------------------------------
    # 4. Artifact Protection and Collision Prevention Tests
    # -------------------------------------------------------------------------

    def test_artifact_collision_fails_clearly(self):
        """17. When intended overlay file already exists in output_dir, FileExistsError is raised."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir)
            # Create a colliding mock file
            collision_file = out_path / f"{DEFAULT_PAIR_ID}_gradcam_CC.png"
            collision_file.write_bytes(b"existing_overlay")

            with self.assertRaises(FileExistsError) as ctx:
                prepare_saliency_output_dir(out_path, DEFAULT_PAIR_ID, generate_saliency=True)
            self.assertIn("Collision detected", str(ctx.exception))
            # Verify existing file was NOT modified
            self.assertEqual(collision_file.read_bytes(), b"existing_overlay")

    def test_artifact_unique_directory_generation(self):
        """18. When output_dir is None, a unique non-colliding directory is created on disk."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_root = Path(tmp_dir)
            unique_dir = prepare_saliency_output_dir(None, DEFAULT_PAIR_ID, generate_saliency=True, project_root=fake_root)
            self.assertTrue(unique_dir.is_dir())
            self.assertIn(f"run_{DEFAULT_PAIR_ID}_", unique_dir.name)

    # -------------------------------------------------------------------------
    # 5. Real Try/Finally Lifecycle Shutdown Tests
    # -------------------------------------------------------------------------

    @patch("scripts.demo_mammo_clin.MultiAgentOrchestrator")
    def test_harness_try_finally_cleanup_on_exception(self, mock_orch_cls):
        """19. Exercises the harness's actual run_demo try/finally cleanup path on execution error."""
        # Instantiate a real MultiAgentOrchestrator to monitor its shutdown state
        real_orch = MultiAgentOrchestrator(
            mammo_pipeline=MagicMock(),
            clin_pipeline=MagicMock(),
            timeout_seconds=5.0,
            max_workers=2,
        )
        self.assertFalse(real_orch._is_closed)
        mock_orch_cls.return_value = real_orch

        # Force real_orch.run to fail as if an unexpected error occurred during inference
        with patch.object(real_orch, "run", side_effect=RuntimeError("Simulated inference crash inside try block")):
            with self.assertRaises(RuntimeError) as ctx:
                run_demo(
                    pair_id=DEFAULT_PAIR_ID,
                    generate_saliency=False,
                    test_failure_simulation=False,
                )
            self.assertIn("Simulated inference crash inside try block", str(ctx.exception))

        # Assert that the harness's own finally block cleanly closed the orchestrator
        self.assertTrue(real_orch._is_closed)
        self.assertIsNone(real_orch._executor)


if __name__ == "__main__":
    unittest.main()
