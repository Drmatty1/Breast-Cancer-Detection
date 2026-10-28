"""
Unit test suite for Clin Agent feature allowlist enforcement and leakage prevention.

Governing contracts:
- docs/MASTER_PROMPT.md (Operating Contract & Hard Rule 8)
- reports/clin/CLIN_AGENT_C1_AUDIT.md
- reports/clin/CLIN_AGENT_C2_DECISION_PROPOSAL.md (DEC-CLIN-01, DEC-CLIN-02)
- reports/clin/CLIN_AGENT_C2_1_LEAKAGE_AUDIT.md

This test strictly verifies feature matrix extraction without executing model training,
hyperparameter tuning, calibrator fitting, or inference.
"""

import unittest
import pandas as pd
from typing import Tuple, List, Set


# Canonical specification constants
APPROVED_CLIN_PREDICTORS: List[str] = ["breast_density", "breast_side"]
TARGET_COLUMN: str = "target_binary"

# Explicit set of columns that must NEVER enter the predictor matrix X
FORBIDDEN_COLUMNS: Set[str] = {
    # Target columns
    "target_binary",
    "target_binary_name",
    "target_masked",
    "patient_stratum",  # Target-derived patient-level stratification label
    "source_pathologies_json",  # Raw ground-truth pathology labels
    "pathology",
    "pathology_original",
    "assessment",  # Current lesion BI-RADS score (fatal target leakage)
    # Identifiers
    "pair_id",
    "patient_id",
    # Provenance metadata
    "source_record_ids_json",
    "source_record_count",
    "abnormality_type_audit",  # Imaging-derived audit attribute (DEC-CLIN-02)
    # Split metadata
    "outer_fold",
    "fold0_role",
    "fold1_role",
    "fold2_role",
    "fold3_role",
    "fold4_role",
    # Mammo secondary task heads / perception descriptors
    "subtlety",
    "mass_shape",
    "mass_margins",
    "calc_type",
    "calc_distribution",
    # Mammo model outputs / latents
    "mammo_pred",
    "mammo_logits",
    "mammo_latent"
}


def select_clin_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts strictly allowlisted clinical predictors X and binary target y.
    
    Guarantees:
    1. X contains strictly ['breast_density', 'breast_side'].
    2. y contains strictly 'target_binary'.
    3. Identifiers, targets, split metadata, and provenance are strictly excluded from X.
    4. Raises an error if any forbidden column is present in X.
    """
    for col in APPROVED_CLIN_PREDICTORS:
        if col not in df.columns:
            raise KeyError(f"Required predictor '{col}' missing from dataframe.")
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' missing from dataframe.")

    # Explicit allowlist projection
    X = df[list(APPROVED_CLIN_PREDICTORS)].copy()
    y = df[TARGET_COLUMN].copy()

    # Leakage assertion guard
    leakage = set(X.columns).intersection(FORBIDDEN_COLUMNS)
    if leakage:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED: Forbidden columns present in predictor matrix X: {leakage}")

    return X, y


class TestClinFeatureAllowlist(unittest.TestCase):
    """Synthetic unit tests for Clin Agent feature extraction and leakage containment."""

    def test_allowlist_returns_strictly_two_predictors(self):
        """Verify that selector extracts exactly the two allowlisted predictors and binary target."""
        synthetic_df = pd.DataFrame({
            "pair_id": ["P_00001_LEFT", "P_00002_RIGHT"],
            "patient_id": ["P_00001", "P_00002"],
            "breast_side": ["LEFT", "RIGHT"],
            "breast_density": [2, 3],
            "abnormality_type_audit": ["mass", "calcification"],
            "patient_stratum": [0, 1],
            "source_pathologies_json": ['["BENIGN"]', '["MALIGNANT"]'],
            "target_binary": [0, 1],
            "target_binary_name": ["benign", "malignant"],
            "outer_fold": [0, 1],
            "fold0_role": ["inner_train", "outer_test"]
        })

        X, y = select_clin_features(synthetic_df)

        self.assertEqual(list(X.columns), APPROVED_CLIN_PREDICTORS)
        self.assertEqual(X.shape, (2, 2))
        self.assertEqual(y.name, TARGET_COLUMN)
        self.assertEqual(list(y.values), [0, 1])

    def test_target_derived_columns_are_strictly_excluded(self):
        """Verify that target-derived columns never enter the predictor matrix X."""
        synthetic_df = pd.DataFrame({
            "breast_side": ["LEFT", "RIGHT"],
            "breast_density": [1, 4],
            "target_binary": [0, 1],
            "target_binary_name": ["benign", "malignant"],
            "patient_stratum": [0, 1],
            "source_pathologies_json": ['["BENIGN"]', '["MALIGNANT"]'],
            "pathology": ["BENIGN", "MALIGNANT"],
            "assessment": [2, 5],
            "subtlety": [3, 4],
            "mass_margins": ["circumscribed", "spiculated"],
            "mammo_logits": [-1.2, 2.4]
        })

        X, y = select_clin_features(synthetic_df)

        for col in FORBIDDEN_COLUMNS:
            self.assertNotIn(col, X.columns, f"Forbidden column '{col}' leaked into X!")

    def test_missing_approved_predictor_raises_key_error(self):
        """Verify that missing breast_density raises KeyError rather than guessing or imputing."""
        incomplete_df = pd.DataFrame({
            "breast_side": ["LEFT", "RIGHT"],
            "target_binary": [0, 1]
        })

        with self.assertRaises(KeyError):
            select_clin_features(incomplete_df)

    def test_selection_on_actual_canonical_feature_table(self):
        """Verify that select_clin_features runs cleanly on the actual 846-pair manifest."""
        manifest_path = "data/manifests/clin_feature_table.csv"
        df = pd.read_csv(manifest_path)

        X, y = select_clin_features(df)

        self.assertEqual(X.shape, (846, 2))
        self.assertEqual(list(X.columns), ["breast_density", "breast_side"])
        self.assertEqual(y.shape, (846,))
        self.assertEqual(y.name, "target_binary")
        self.assertFalse(X.isna().any().any(), "X must not contain NaNs")
        self.assertFalse(y.isna().any(), "y must not contain NaNs")


if __name__ == "__main__":
    unittest.main()
