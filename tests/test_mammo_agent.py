"""
Isolated Unit and Integration Tests for Mammo Agent Phase A & B.

Tests:
1. Model output dimensions and finite logits with synthetic tensors.
2. Missing-target masking in MaskedMultiTaskLoss.
3. Loss computation and gradient flow when a batch contains zero valid labels for a secondary task.
4. Checkpoint key and tensor-shape compatibility loading Stage 5 Fold 0 checkpoint.
5. Patient-level split integrity and quarantine validation of mammo_multitask_targets.csv.

Operates purely on synthetic tensors and manifest tables. Zero inference on evaluation images.
"""
import unittest
from pathlib import Path
import json
import torch
import pandas as pd
import numpy as np

from src.models.mammo_agent import MammographyAgentModel, MaskedMultiTaskLoss


class TestMammoAgentArchitecture(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = MammographyAgentModel(pretrained_backbone=False)
        self.model.eval()

    def test_model_output_dimensions_and_finite_logits(self):
        """Verify model output dimensions and finite logits on synthetic inputs."""
        batch_size = 4
        cc = torch.randn(batch_size, 1, 512, 512, dtype=torch.float32)
        mlo = torch.randn(batch_size, 1, 512, 512, dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model(cc, mlo)

        # Check required keys
        expected_keys = {
            "binary_logit", "margin_logits", "calc_logits",
            "birads_ordinal_logits", "attention_weights", "fused_features"
        }
        self.assertTrue(expected_keys.issubset(set(outputs.keys())))

        # 1. Binary Malignancy Logit: [B]
        self.assertEqual(outputs["binary_logit"].shape, torch.Size([batch_size]))
        self.assertTrue(torch.isfinite(outputs["binary_logit"]).all())

        # 2. Margin Logits: [B, 4]
        self.assertEqual(outputs["margin_logits"].shape, torch.Size([batch_size, 4]))
        self.assertTrue(torch.isfinite(outputs["margin_logits"]).all())

        # 3. Calcification Morphology Logits: [B, 5]
        self.assertEqual(outputs["calc_logits"].shape, torch.Size([batch_size, 5]))
        self.assertTrue(torch.isfinite(outputs["calc_logits"]).all())

        # 4. Ordinal BI-RADS Cutoff Logits: [B, 3]
        self.assertEqual(outputs["birads_ordinal_logits"].shape, torch.Size([batch_size, 3]))
        self.assertTrue(torch.isfinite(outputs["birads_ordinal_logits"]).all())

        # 5. Attention weights: [B, 2] summing to 1.0
        self.assertEqual(outputs["attention_weights"].shape, torch.Size([batch_size, 2]))
        sum_alpha = outputs["attention_weights"].sum(dim=1)
        self.assertTrue(torch.allclose(sum_alpha, torch.ones_like(sum_alpha), atol=1e-5))

        # 6. Fused features: [B, 2560]
        self.assertEqual(outputs["fused_features"].shape, torch.Size([batch_size, 2560]))

    def test_missing_target_masking(self):
        """Verify that masked multi-task loss correctly isolates masked targets."""
        loss_fn = MaskedMultiTaskLoss()
        batch_size = 4

        # Synthetic predictions
        preds = {
            "binary_logit": torch.tensor([0.5, -0.2, 1.2, -0.8], requires_grad=True),
            "margin_logits": torch.randn(batch_size, 4, requires_grad=True),
            "calc_logits": torch.randn(batch_size, 5, requires_grad=True),
            "birads_ordinal_logits": torch.randn(batch_size, 3, requires_grad=True)
        }

        # Pair 0 & 1: Mass only (calc masked). Pair 2 & 3: Calc only (margin masked)
        targets = {
            "binary_target": torch.tensor([1, 0, 1, 0], dtype=torch.float32),
            "binary_mask": torch.tensor([1, 1, 1, 1], dtype=torch.int64),
            "margin_label": torch.tensor([3, 0, -1, -1], dtype=torch.int64),
            "margin_mask": torch.tensor([1, 1, 0, 0], dtype=torch.int64),
            "calc_label": torch.tensor([-1, -1, 3, 0], dtype=torch.int64),
            "calc_mask": torch.tensor([0, 0, 1, 1], dtype=torch.int64),
            "birads_label": torch.tensor([2, 0, 3, 1], dtype=torch.int64),
            "birads_mask": torch.tensor([1, 1, 1, 1], dtype=torch.int64)
        }

        loss, breakdown = loss_fn(preds, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(breakdown["n_valid_margin"], 2)
        self.assertEqual(breakdown["n_valid_calc"], 2)
        self.assertEqual(breakdown["n_valid_birads"], 4)

        loss.backward()

        # Check gradients exist and are finite
        self.assertIsNotNone(preds["binary_logit"].grad)
        self.assertTrue(torch.isfinite(preds["binary_logit"].grad).all())
        self.assertIsNotNone(preds["margin_logits"].grad)
        # Margin gradients for unmasked rows 0,1 should be non-zero; for masked rows 2,3 must be exactly zero!
        self.assertTrue((preds["margin_logits"].grad[2:] == 0.0).all())
        # Calc gradients for unmasked rows 2,3 should be non-zero; for masked rows 0,1 must be exactly zero!
        self.assertTrue((preds["calc_logits"].grad[:2] == 0.0).all())

    def test_loss_computation_zero_valid_labels(self):
        """Verify loss computation when a batch contains zero valid labels for secondary tasks."""
        loss_fn = MaskedMultiTaskLoss()
        batch_size = 2

        preds = {
            "binary_logit": torch.tensor([0.1, -0.4], requires_grad=True),
            "margin_logits": torch.randn(batch_size, 4, requires_grad=True),
            "calc_logits": torch.randn(batch_size, 5, requires_grad=True),
            "birads_ordinal_logits": torch.randn(batch_size, 3, requires_grad=True)
        }

        # Entire batch is pure calcification, with invalid BI-RADS
        targets = {
            "binary_target": torch.tensor([1, 0], dtype=torch.float32),
            "binary_mask": torch.tensor([1, 1], dtype=torch.int64),
            "margin_label": torch.tensor([-1, -1], dtype=torch.int64),
            "margin_mask": torch.tensor([0, 0], dtype=torch.int64),  # Zero valid margins!
            "calc_label": torch.tensor([3, 0], dtype=torch.int64),
            "calc_mask": torch.tensor([1, 1], dtype=torch.int64),
            "birads_label": torch.tensor([-1, -1], dtype=torch.int64),
            "birads_mask": torch.tensor([0, 0], dtype=torch.int64)   # Zero valid birads!
        }

        loss, breakdown = loss_fn(preds, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(breakdown["margin_loss"], 0.0)
        self.assertEqual(breakdown["birads_loss"], 0.0)
        self.assertEqual(breakdown["n_valid_margin"], 0)
        self.assertEqual(breakdown["n_valid_birads"], 0)

        # Backward should not crash or produce NaNs. Unused heads have None or zero grad.
        loss.backward()
        self.assertIsNotNone(preds["binary_logit"].grad)
        self.assertTrue(torch.isfinite(preds["binary_logit"].grad).all())
        self.assertIsNotNone(preds["calc_logits"].grad)
        self.assertTrue(torch.isfinite(preds["calc_logits"].grad).all())

        if preds["margin_logits"].grad is not None:
            self.assertTrue(torch.all(preds["margin_logits"].grad == 0.0))
        else:
            self.assertIsNone(preds["margin_logits"].grad)

        if preds["birads_ordinal_logits"].grad is not None:
            self.assertTrue(torch.all(preds["birads_ordinal_logits"].grad == 0.0))
        else:
            self.assertIsNone(preds["birads_ordinal_logits"].grad)

    def test_checkpoint_compatibility(self):
        """Verify checkpoint loading compatibility from Stage 5 Fold 0 best checkpoint."""
        ckpt_path = Path(r"D:\rp\reports\stage5_cross_validation_gpu\checkpoints\model_fold0_best.pt")
        self.assertTrue(ckpt_path.is_file(), f"Stage 5 checkpoint not found: {ckpt_path}")

        load_summary = self.model.load_stage5_checkpoint(str(ckpt_path))

        # Expected 305 tensors in Stage 5 checkpoint: 294 backbone + 3 fusion + 8 binary head
        self.assertEqual(load_summary["loaded_keys_count"], 305)

        # Expected new head keys initialized: 8 in MarginHead + 8 in CalcificationHead + 8 in OrdinalBiRadsHead = 24
        self.assertEqual(load_summary["initialized_keys_count"], 24)

        # Verify no shape mismatches in skipped_keys
        shape_mismatches = [s for s in load_summary["skipped_keys"] if "mismatch" in s[1].lower()]
        self.assertEqual(len(shape_mismatches), 0, f"Encountered shape mismatches: {shape_mismatches}")


class TestMultitaskTargetsSplitIntegrity(unittest.TestCase):
    def test_patient_level_split_integrity(self):
        """Verify patient-level disjointness and integrity of mammo_multitask_targets.csv."""
        targets_path = Path(r"D:\rp\data\manifests\stage2\mammo_multitask_targets.csv")
        self.assertTrue(targets_path.is_file(), f"Targets CSV not found: {targets_path}")

        df = pd.read_csv(targets_path)

        # 1. Exact counts
        self.assertEqual(len(df), 846, f"Expected 846 rows, got {len(df)}")
        self.assertEqual(df["pair_id"].nunique(), 846, "Duplicate pair_id found!")
        self.assertEqual(df["patient_id"].nunique(), 786, "Expected 786 unique patients!")

        # 2. Folds
        self.assertEqual(set(df["outer_fold"].unique()), {0, 1, 2, 3, 4})

        # 3. Patient disjointness across folds
        for f1 in range(5):
            p1 = set(df[df["outer_fold"] == f1]["patient_id"])
            for f2 in range(f1 + 1, 5):
                p2 = set(df[df["outer_fold"] == f2]["patient_id"])
                overlap = p1.intersection(p2)
                self.assertEqual(len(overlap), 0, f"Patient overlap between fold {f1} and {f2}: {overlap}")

        # 4. Local Holdout Quarantine Verification
        splits_path = Path(r"D:\rp\data\manifests\stage2\patient_splits.csv")
        self.assertTrue(splits_path.is_file())
        splits_df = pd.read_csv(splits_path)
        holdout_pats = set(splits_df[splits_df["partition"] == "local_holdout"]["patient_id"])
        dev_pats = set(df["patient_id"])
        holdout_overlap = dev_pats.intersection(holdout_pats)
        self.assertEqual(len(holdout_overlap), 0, f"FATAL: Development targets overlap with Local Holdout patients! {holdout_overlap}")


if __name__ == "__main__":
    unittest.main()
