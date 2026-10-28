"""
Isolated Integration Tests for Mammo Agent Phase 4.

Tests:
1. Finding card JSON schema validation (complete, partially unavailable, and malformed).
2. Pipeline initialization and checkpoint loading compatibility.
3. Two-view input ordering and forward prediction on synthetic inputs.
4. Finite and normalized Grad-CAM saliency heatmaps for CC and MLO.
5. Error handling for missing views, wrong tensor shapes, and non-existent file paths.

Operates purely on synthetic tensors. Zero inference on evaluation images.
"""
import unittest
from pathlib import Path
import json
import numpy as np
import torch

from src.mammo.finding_card import build_finding_card, validate_finding_card
from src.models.mammo_agent import MammographyAgentModel
from src.mammo.saliency import GradCAM, extract_top_regions, overlay_heatmap_on_image
from src.mammo.inference import MammographyAgentInferencePipeline


class TestMammoAgentIntegration(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.ckpt_fold0 = Path(r"D:\rp\reports\mammo_agent_training\checkpoints\mammo_agent_fold0_best.pt")
        self.assertTrue(self.ckpt_fold0.is_file(), f"Missing checkpoint: {self.ckpt_fold0}")

    def test_schema_validation_complete_and_partial(self):
        """Verify schema validation for complete, partial, and malformed cards."""
        # 1. Complete card (both mass and calcification independently supported)
        card_complete = build_finding_card(
            binary_prob_calibrated=0.78,
            margin_probs=[0.05, 0.10, 0.05, 0.80],
            calc_probs=[0.10, 0.05, 0.05, 0.75, 0.05],
            birads_ordinal_level=2,
            saliency_regions=[[312.0, 410.0, 78.0, 130.0]],
            saliency_path="gradcam_4C.png",
            abnormality_type="both"
        )
        is_valid, err = validate_finding_card(card_complete)
        self.assertTrue(is_valid, f"Complete card failed validation: {err}")
        self.assertEqual(card_complete["finding"]["margin"], "spiculated")
        self.assertEqual(card_complete["finding"]["calcification"], "pleomorphic")
        self.assertEqual(card_complete["BI_RADS_a"], "4")

        # 2. Partial card (pure mass, no calcification)
        card_mass_only = build_finding_card(
            binary_prob_calibrated=0.85,
            margin_probs=[0.05, 0.80, 0.05, 0.10],
            calc_probs=None,
            birads_ordinal_level=2,
            saliency_regions=[[100.0, 100.0, 200.0, 200.0]]
        )
        is_valid, err = validate_finding_card(card_mass_only)
        self.assertTrue(is_valid, f"Mass-only card failed validation: {err}")
        self.assertIsNone(card_mass_only["finding"]["calcification"])

        # 3. Partial card (pure calc, no mass margin)
        card_calc_only = build_finding_card(
            binary_prob_calibrated=0.42,
            margin_probs=None,
            calc_probs=[0.60, 0.10, 0.10, 0.10, 0.10],
            birads_ordinal_level=1,
            saliency_regions=[]
        )
        is_valid, err = validate_finding_card(card_calc_only)
        self.assertTrue(is_valid, f"Calc-only card failed validation: {err}")
        self.assertIsNone(card_calc_only["finding"]["margin"])

        # 4. Malformed card (missing required field 'agent')
        malformed_card = card_complete.copy()
        del malformed_card["agent"]
        is_valid, err = validate_finding_card(malformed_card)
        self.assertFalse(is_valid, "Malformed card unexpectedly passed validation!")

    def test_pipeline_checkpoint_loading_and_compatibility(self):
        """Verify pipeline initialization and checkpoint compatibility."""
        pipeline = MammographyAgentInferencePipeline(
            checkpoint_paths=[self.ckpt_fold0],
            device=self.device
        )
        self.assertEqual(len(pipeline.models), 1)
        self.assertEqual(pipeline.checkpoint_names[0], "mammo_agent_fold0_best.pt")
        # Fold 0 locked temperature: 1.386468...
        self.assertAlmostEqual(pipeline.temperatures[0], 1.3864682912826538, places=5)

    def test_two_view_inference_and_output_shapes(self):
        """Verify end-to-end inference on synthetic paired arrays produces valid finding cards."""
        pipeline = MammographyAgentInferencePipeline(
            checkpoint_paths=[self.ckpt_fold0],
            device=self.device
        )

        # Generate synthetic 512x512 mammogram arrays in [0.0, 1.0]
        synth_cc = np.random.uniform(0.0, 1.0, (512, 512)).astype(np.float32)
        synth_mlo = np.random.uniform(0.0, 1.0, (512, 512)).astype(np.float32)

        card = pipeline.predict(
            cc_input=synth_cc,
            mlo_input=synth_mlo,
            generate_saliency=True,
            pair_identifier="unit_test_synthetic"
        )

        # Validate card against formal schema
        is_valid, err = validate_finding_card(card)
        self.assertTrue(is_valid, f"Pipeline output failed schema validation: {err}")

        # Check required fields
        self.assertEqual(card["agent"], "mammo")
        self.assertIn(card["finding"]["lesion"], ["malignant", "benign"])
        self.assertTrue(0.0 <= card["conf_a"] <= 1.0)
        self.assertIsInstance(card["sal"]["top_regions"], list)
        self.assertIsInstance(card["rationale"], str)

    def test_saliency_cc_and_mlo_finite_and_normalized(self):
        """Verify Grad-CAM produces separate finite, normalized heatmaps for CC and MLO."""
        model = MammographyAgentModel(pretrained_backbone=False).to(self.device)
        cam_engine = GradCAM(model)

        cc = torch.randn(1, 1, 512, 512, dtype=torch.float32, device=self.device)
        mlo = torch.randn(1, 1, 512, 512, dtype=torch.float32, device=self.device)

        results = cam_engine.generate_cam(cc, mlo, task="binary")
        cam_cc = results["cam_cc"][0]
        cam_mlo = results["cam_mlo"][0]
        cam_engine.remove_hooks()

        # Check shapes
        self.assertEqual(cam_cc.shape, (512, 512))
        self.assertEqual(cam_mlo.shape, (512, 512))

        # Check finite and bounds in [0.0, 1.0]
        self.assertTrue(np.isfinite(cam_cc).all())
        self.assertTrue(np.isfinite(cam_mlo).all())
        self.assertTrue(cam_cc.min() >= 0.0 and cam_cc.max() <= 1.0)
        self.assertTrue(cam_mlo.min() >= 0.0 and cam_mlo.max() <= 1.0)

    def test_safe_error_handling_invalid_inputs(self):
        """Verify pipeline raises informative errors on invalid or missing inputs."""
        pipeline = MammographyAgentInferencePipeline(
            checkpoint_paths=[self.ckpt_fold0],
            device=self.device
        )
        valid_img = np.random.uniform(0.0, 1.0, (512, 512)).astype(np.float32)

        # 1. Missing view
        with self.assertRaises(ValueError):
            pipeline.predict(cc_input=valid_img, mlo_input=None)

        with self.assertRaises(ValueError):
            pipeline.predict(cc_input=None, mlo_input=valid_img)

        # 2. Wrong dimensions (e.g. 256x256 instead of 512x512)
        wrong_shape_img = np.random.uniform(0.0, 1.0, (256, 256)).astype(np.float32)
        with self.assertRaises(ValueError):
            pipeline.predict(cc_input=wrong_shape_img, mlo_input=valid_img)

        # 3. Non-existent file path
        with self.assertRaises(FileNotFoundError):
            pipeline.predict(cc_input="non_existent_file.npy", mlo_input=valid_img)


if __name__ == "__main__":
    unittest.main()
