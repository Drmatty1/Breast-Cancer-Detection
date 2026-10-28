"""
Script to build the development-only multi-task targets manifest:
D:\\rp\\data\\manifests\\stage2\\mammo_multitask_targets.csv

Operates strictly on the 846 development pairs from cv_splits_manifest.csv (786 patients).
Preserves missing/inapplicable/conflicting labels as masked targets (mask=0, label=-1).
Never accesses the quarantined local holdout or official test partitions.
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np


def build_mammo_multitask_targets() -> Path:
    project_root = Path(r"D:\rp")
    cv_manifest_path = project_root / "reports" / "stage5_cross_validation_gpu" / "cv_splits_manifest.csv"
    disp_path = project_root / "data" / "manifests" / "stage2" / "source_record_disposition.csv"
    mass_meta_path = project_root / "data" / "source_metadata" / "mass_case_description_train_set.csv"
    out_targets_path = project_root / "data" / "manifests" / "stage2" / "mammo_multitask_targets.csv"

    assert cv_manifest_path.is_file(), f"Missing CV manifest: {cv_manifest_path}"
    assert disp_path.is_file(), f"Missing source record disposition: {disp_path}"
    assert mass_meta_path.is_file(), f"Missing mass metadata: {mass_meta_path}"

    cv_df = pd.read_csv(cv_manifest_path)
    disp_df = pd.read_csv(disp_path)
    mass_df = pd.read_csv(mass_meta_path)

    # Filter strictly for development pairs
    dev_pairs = set(cv_df["pair_id"])
    assert len(dev_pairs) == 846, f"Expected 846 dev pairs, found {len(dev_pairs)}"
    assert cv_df["patient_id"].nunique() == 786, f"Expected 786 unique patients, found {cv_df['patient_id'].nunique()}"

    dev_disp = disp_df[disp_df["pair_id"].isin(dev_pairs)].copy()

    # Join mass margins using exact source_row_number (1-indexed)
    mass_rows = dev_disp[dev_disp["source_csv"] == "mass_case_description_train_set.csv"].copy()
    mass_indices = mass_rows["source_row_number"] - 1
    dev_disp.loc[mass_rows.index, "mass_margins"] = mass_df.iloc[mass_indices]["mass margins"].values
    dev_disp.loc[mass_rows.index, "mass_shape"] = mass_df.iloc[mass_indices]["mass shape"].values

    # Authoritative Class Vocabularies derived strictly from Methodology §3.1:
    # Margin: circumscribed, indistinct (ill-defined in CBIS), obscured, spiculated
    margin_vocab = {
        "CIRCUMSCRIBED": 0,
        "ILL_DEFINED": 1,
        "OBSCURED": 2,
        "SPICULATED": 3
    }

    # Calcification morphology: amorphous, coarse (heterogeneous), fine-linear branching, pleomorphic, punctate
    calc_vocab = {
        "AMORPHOUS": 0,
        "COARSE": 1,
        "FINE_LINEAR_BRANCHING": 2,
        "PLEOMORPHIC": 3,
        "PUNCTATE": 4
    }

    # BI-RADS diagnostic assessment: 2 (benign), 3 (probably benign), 4 (suspicious), 5 (highly suggestive)
    # Note: Category 0 represents incomplete imaging; categories are ordered 2 < 3 < 4 < 5
    birads_diag_vocab = {
        2: 0,
        3: 1,
        4: 2,
        5: 3
    }

    rows = []
    for _, r_cv in cv_df.iterrows():
        pid = r_cv["pair_id"]
        p_disp = dev_disp[dev_disp["pair_id"] == pid]
        assert len(p_disp) > 0, f"No disposition records for pair {pid}"

        # 1. Binary target (Lesion malignancy) - 100% complete
        bin_target = int(r_cv["binary_target"])
        bin_mask = 1

        # 2. Margin target (Mass cases only; non-conflicting exact methodology vocabulary)
        margins = sorted(list(set(p_disp["mass_margins"].dropna())))
        if len(margins) == 1 and margins[0] in margin_vocab:
            margin_label = margin_vocab[margins[0]]
            margin_mask = 1
        else:
            # Masked: non-mass, missing, microlobulated, compound, or view conflict
            margin_label = -1
            margin_mask = 0

        # 3. Calcification morphology target (Calc cases only; non-conflicting exact methodology vocabulary)
        calcs = sorted(list(set(p_disp["calc_type"].dropna())))
        if len(calcs) == 1 and calcs[0] in calc_vocab:
            calc_label = calc_vocab[calcs[0]]
            calc_mask = 1
        else:
            # Masked: non-calc, missing, other vocabulary, compound, or view conflict
            calc_label = -1
            calc_mask = 0

        # 4. BI-RADS assessment target (Diagnostic categories 2-5 only; non-conflicting)
        assessments = sorted(list(set(p_disp["assessment"].dropna())))
        if len(assessments) == 1 and assessments[0] in birads_diag_vocab:
            birads_label = birads_diag_vocab[assessments[0]]
            birads_mask = 1
        else:
            # Masked: category 0 (incomplete), category 1, missing, or view conflict
            birads_label = -1
            birads_mask = 0

        source_ids = sorted(list(p_disp["source_record_id"].unique()))

        rows.append({
            "pair_id": pid,
            "patient_id": r_cv["patient_id"],
            "outer_fold": int(r_cv["outer_fold"]),
            "binary_target": bin_target,
            "binary_mask": bin_mask,
            "margin_label": margin_label,
            "margin_mask": margin_mask,
            "calc_label": calc_label,
            "calc_mask": calc_mask,
            "birads_label": birads_label,
            "birads_mask": birads_mask,
            "raw_margins_json": json.dumps(margins),
            "raw_calcs_json": json.dumps(calcs),
            "raw_assessments_json": json.dumps([int(a) for a in assessments]),
            "source_record_ids_json": json.dumps(source_ids)
        })

    df_out = pd.DataFrame(rows)

    # Strict validation assertions before writing
    assert len(df_out) == 846, f"Expected 846 rows, got {len(df_out)}"
    assert df_out["pair_id"].nunique() == 846, "Duplicate pair_id found in targets!"
    assert df_out["patient_id"].nunique() == 786, "Patient count mismatch!"
    assert set(df_out["outer_fold"].unique()) == {0, 1, 2, 3, 4}, "Folds mismatch!"

    # Verify patient disjointness across folds
    for f1 in range(5):
        p1 = set(df_out[df_out["outer_fold"] == f1]["patient_id"])
        for f2 in range(f1 + 1, 5):
            p2 = set(df_out[df_out["outer_fold"] == f2]["patient_id"])
            assert len(p1.intersection(p2)) == 0, f"Patient overlap between fold {f1} and {f2}!"

    out_targets_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_targets_path, index=False)
    print(f"Successfully generated {out_targets_path} with {len(df_out)} rows.")
    return out_targets_path


if __name__ == "__main__":
    build_mammo_multitask_targets()
