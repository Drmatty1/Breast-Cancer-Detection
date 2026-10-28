import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\rp")
sys.path.insert(0, str(ROOT / "src"))
from preprocessing.mammography import PreprocessConfig, preprocess_pixels  # noqa: E402
from preprocessing.compare_stage3b import cfg_dict  # noqa: E402
from test_mammography import metadata_for, synthetic_breast  # noqa: E402


class Stage3BComparisonTests(unittest.TestCase):
    def test_variants_preserve_baseline_settings_except_declared_axis(self):
        a = PreprocessConfig()
        b = PreprocessConfig(clahe_clip_limit=1.0)
        c = PreprocessConfig(crop_margin_fraction=0.005)
        da, db, dc = cfg_dict(a), cfg_dict(b), cfg_dict(c)
        self.assertEqual(da["clahe_clip_limit"], 2.0)
        self.assertEqual(db["clahe_clip_limit"], 1.0)
        self.assertEqual(dc["clahe_clip_limit"], da["clahe_clip_limit"])
        self.assertEqual({k:v for k,v in da.items() if k != "clahe_clip_limit"},
                         {k:v for k,v in db.items() if k != "clahe_clip_limit"})
        self.assertEqual({k:v for k,v in da.items() if k != "crop_margin_fraction"},
                         {k:v for k,v in dc.items() if k != "crop_margin_fraction"})

    def test_comparison_outputs_remain_valid_and_changes_are_deterministic(self):
        pixels = synthetic_breast()
        metadata = metadata_for(pixels)
        cfgs = (PreprocessConfig(), PreprocessConfig(clahe_clip_limit=1.0),
                PreprocessConfig(crop_margin_fraction=0.005))
        first = [preprocess_pixels(pixels, metadata, cfg) for cfg in cfgs]
        second = [preprocess_pixels(pixels, metadata, cfg)[0] for cfg in cfgs]
        for (image, metrics, qa), repeat in zip(first, second):
            self.assertEqual(image.shape, (512, 512))
            self.assertEqual(image.dtype, np.float32)
            self.assertTrue(np.isfinite(image).all())
            self.assertTrue(np.array_equal(image, repeat))
            self.assertTrue(qa["output_mask"].any())
            self.assertEqual(metrics["horizontal_flip"], False)
        # A parameter change need not alter a quantized output on every input;
        # exact parameter isolation is checked above and real-pilot differences
        # are recorded in the comparison metrics/report.
        self.assertEqual(len({r[1]["mask_coverage_source"] for r in first}), 1)

    def test_tight_crop_does_not_remove_detected_mask_pixels(self):
        pixels = synthetic_breast()
        image, metrics, qa = preprocess_pixels(pixels, metadata_for(pixels), PreprocessConfig(crop_margin_fraction=0.005))
        self.assertEqual(image.shape, (512, 512))
        self.assertGreater(metrics["mask_coverage_output"], 0.0)
        self.assertTrue(qa["output_mask"].any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
