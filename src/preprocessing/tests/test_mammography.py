import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\rp")
sys.path.insert(0, str(ROOT / "src"))
from preprocessing.mammography import (  # noqa: E402
    DecodeError, MalformedMetadataError, PreprocessConfig, ReviewRequired,
    clahe_u8, decode_dataset, decode_dicom, largest_component,
    normalize_polarity, preprocess_pixels, valid_pixel_mask,
    validate_breast_mask, validate_image_provenance, validate_pair_record,
)


def synthetic_breast(rows=128, cols=96):
    yy, xx = np.mgrid[0:rows, 0:cols]
    ellipse = ((xx - cols * 0.48) / (cols * 0.37)) ** 2 + ((yy - rows * 0.50) / (rows * 0.47)) ** 2 < 1.0
    values = (12000 + 26000 * (xx / max(cols - 1, 1)) + 6000 * (yy / max(rows - 1, 1))).astype(np.uint16)
    values[~ellipse] = 0
    values[ellipse & ((xx + yy) % 17 == 0)] = 65535
    return values


def metadata_for(array, **updates):
    meta = {
        "Rows": array.shape[0], "Columns": array.shape[1], "SamplesPerPixel": 1,
        "PhotometricInterpretation": "MONOCHROME2", "BitsAllocated": 16,
        "BitsStored": 16, "HighBit": 15, "PixelRepresentation": 0,
        "PixelPaddingValue": None, "PixelPaddingRangeLimit": None,
        "RescaleSlope": None, "RescaleIntercept": None,
        "ModalityLUTSequence": None, "PresentationLUTShape": None,
    }
    meta.update(updates)
    return meta


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class BrokenPixelDataset:
    @property
    def pixel_array(self):
        raise ValueError("malformed pixel data")


class MammographyTests(unittest.TestCase):
    def test_dicom_decoding_failures_and_malformed_metadata(self):
        with self.assertRaises(DecodeError):
            decode_dataset(BrokenPixelDataset())
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "invalid.dcm"
            bad.write_bytes(b"not a DICOM")
            with self.assertRaises(DecodeError):
                decode_dicom(bad)
        arr = synthetic_breast()
        malformed = metadata_for(arr)
        del malformed["Rows"]
        with self.assertRaises(MalformedMetadataError):
            preprocess_pixels(arr, malformed)
        non_identity = metadata_for(arr, RescaleSlope=2.0)
        with self.assertRaises(ReviewRequired):
            preprocess_pixels(arr, non_identity)

    def test_monochrome_polarity(self):
        arr = np.array([[0, 100, 200], [300, 400, 500]], dtype=np.uint16)
        valid = np.ones_like(arr, dtype=bool)
        mono2 = normalize_polarity(arr, "MONOCHROME2", valid)
        mono1 = normalize_polarity(arr, "MONOCHROME1", valid)
        np.testing.assert_array_equal(mono2, arr.astype(np.float32))
        np.testing.assert_array_equal(mono1, (500-arr).astype(np.float32))
        with self.assertRaises(MalformedMetadataError):
            normalize_polarity(arr, "RGB", valid)
        base = synthetic_breast()
        mono1 = np.uint16(65535) - base
        mono1_meta = metadata_for(mono1, PhotometricInterpretation="MONOCHROME1")
        out, _, _ = preprocess_pixels(mono1, mono1_meta)
        self.assertEqual(out.shape, (512, 512))
        self.assertGreater(float(out.max()), 0.0)

    def test_constant_and_near_constant_images_flag_for_review(self):
        for value in (12000, 12000 + np.arange(64, dtype=np.uint16).reshape(8, 8) % 2):
            arr = np.full((96, 96), value, dtype=np.uint16) if np.isscalar(value) else np.full((96,96), 12000, dtype=np.uint16) + np.tile(value, (12,12))
            with self.assertRaises(ReviewRequired):
                preprocess_pixels(arr, metadata_for(arr))

    def test_pixel_padding_range_and_nonfinite_values(self):
        arr = np.array([[0.0, 1.0, 2.0], [3.0, np.nan, 65535.0]], dtype=np.float32)
        valid = valid_pixel_mask(arr, padding_value=1, padding_range_limit=3)
        expected = np.array([[True, False, False], [False, False, True]])
        np.testing.assert_array_equal(valid, expected)
        arr2 = synthetic_breast()
        arr2[20, 20] = 65535
        valid2 = valid_pixel_mask(arr2, padding_value=65535)
        self.assertFalse(valid2[20, 20])
        arr3 = synthetic_breast().astype(np.float32)
        arr3[48, 45] = np.nan
        arr3[49, 46] = -999.0
        meta3 = metadata_for(arr3, PixelPaddingValue=-999)
        out, metrics, _ = preprocess_pixels(arr3, meta3)
        self.assertTrue(np.isfinite(out).all())
        self.assertGreater(metrics["invalid_pixel_fraction"], 0.0)

    def test_empty_and_implausible_masks_are_rejected(self):
        with self.assertRaises(ReviewRequired):
            largest_component(np.zeros((32, 32), dtype=bool))
        cfg = PreprocessConfig()
        with self.assertRaises(ReviewRequired):
            validate_breast_mask(np.ones((32, 32), dtype=bool), 1000, 1000, cfg)
        with self.assertRaises(ReviewRequired):
            validate_breast_mask(np.eye(32, dtype=bool), 32, 32, cfg)

    def test_output_shape_dtype_and_finite_values(self):
        arr = synthetic_breast()
        out, metrics, qa = preprocess_pixels(arr, metadata_for(arr))
        self.assertEqual(out.shape, (512, 512))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.isfinite(out).all())
        self.assertGreater(float(out.max()), 0.0)
        self.assertIn("mask_coverage_output", metrics)
        self.assertEqual(qa["mask_source"].shape, arr.shape)

    def test_pair_completeness_and_cc_mlo_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            cc = Path(tmp) / "cc.dcm"; mlo = Path(tmp) / "mlo.dcm"
            cc.touch(); mlo.touch()
            row = {"pair_id":"P_00001_LEFT","patient_id":"P_00001","breast_side":"LEFT","partition":"train",
                   "cc_local_path":str(cc),"mlo_local_path":str(mlo),"cc_study_instance_uid":"1.1","cc_series_instance_uid":"1.2",
                   "mlo_study_instance_uid":"1.3","mlo_series_instance_uid":"1.4"}
            validate_pair_record(row)
            broken = dict(row); broken["mlo_series_instance_uid"] = ""
            with self.assertRaises(MalformedMetadataError): validate_pair_record(broken)
            same = dict(row); same["mlo_study_instance_uid"] = "1.1"; same["mlo_series_instance_uid"] = "1.2"
            with self.assertRaises(MalformedMetadataError): validate_pair_record(same)
            val = dict(row); val["partition"] = "validation"
            with self.assertRaises(ReviewRequired): validate_pair_record(val)

    def test_dicom_uid_patient_side_and_view_provenance(self):
        pair = {"patient_id":"P_00001","breast_side":"LEFT","cc_study_instance_uid":"1.1","cc_series_instance_uid":"1.2"}
        ds = {"PatientID":"Mass-Training_P_00001_LEFT_CC","Laterality":"L","PatientOrientation":"CC",
              "StudyInstanceUID":"1.1","SeriesInstanceUID":"1.2"}
        identity = validate_image_provenance(ds, pair, "CC")
        self.assertEqual(identity["patient_id"], "P_00001")
        bad = dict(ds); bad["SeriesInstanceUID"] = "9.9"
        with self.assertRaises(ReviewRequired): validate_image_provenance(bad, pair, "CC")

    def test_deterministic_output_for_fixed_config(self):
        arr = synthetic_breast()
        cfg = PreprocessConfig()
        a, ma, _ = preprocess_pixels(arr, metadata_for(arr), cfg)
        b, mb, _ = preprocess_pixels(arr, metadata_for(arr), cfg)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(ma["threshold_u8"], mb["threshold_u8"])
        self.assertEqual(ma["mask_coverage_output"], mb["mask_coverage_output"])

    def test_no_writes_to_source_dicom_or_stage2_manifests(self):
        cohort = ROOT / "data/manifests/stage2/cohort_pairs.csv"
        splits = ROOT / "data/manifests/stage2/patient_splits.csv"
        with cohort.open(encoding="utf-8-sig", newline="") as stream:
            import csv
            cohort_rows = list(csv.DictReader(stream))
        first = next(row for row in cohort_rows if row["partition"] == "train")
        source = Path(first["cc_local_path"])
        before = {str(p): sha256(p) for p in (source, cohort, splits)}
        decoded = decode_dicom(source)
        self.assertEqual(decoded.pixels.ndim, 2)
        after = {str(p): sha256(p) for p in (source, cohort, splits)}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main(verbosity=2)

