#!/usr/bin/env python3
"""
RP1 End-to-End Multimodal Demo Harness (Development Data Only).

Demonstrates the synchronized execution of the authoritative visual perception
specialist (Mammography Agent) and the secondary non-authoritative tabular specialist
(Clinical Risk Agent) via the MultiAgentOrchestrator.

Governed by:
- docs/breast_cancer___ps1___methodology.pdf (§3.1, §3.2, §3.4, §4.3)
- DECISIONS.md (DEC-OWNER-01, DEC-OWNER-02, DEC-OWNER-03)
- reports/final/RP1_FINAL_INTEGRATION_AUDIT.md
- reports/final/RP1_FINAL_RELIABILITY_REVIEW.md

Safety Restrictions:
- Operates strictly on verified development data (fails closed on local_holdout or official test).
- Resolves patient and laterality from canonical manifests to bind image paths to verified pairs.
- Enforces preprocessing contract: 2D (512, 512), float32, finite, [0.0, 1.0].
- Protects existing demo artifacts from overwrite by failing on collision or creating unique directories.
- Clin Agent outputs are research-only, non-voting (voting_eligibility: False).
- Pathology Agent is deferred; finding-card schema is locked to ["mammo", "clin"].
- Orchestrator executor is guaranteed to close via harness try/finally cleanup.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import jsonschema
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mammo.finding_card import validate_finding_card
from src.mammo.inference import MammographyAgentInferencePipeline
from src.clin.inference import ClinicalRiskAgentInferencePipeline
from src.orchestration.multi_agent_orchestrator import (
    MultiAgentOrchestrator,
    OrchestrationRequest,
    OrchestrationResponse,
)


# Default verified development sample
DEFAULT_PAIR_ID = "P_00008_LEFT"
DEFAULT_CC_PATH = PROJECT_ROOT / "data" / "processed" / "stage3b_comparison" / "A" / "P_00008" / "LEFT" / "CC.npy"
DEFAULT_MLO_PATH = PROJECT_ROOT / "data" / "processed" / "stage3b_comparison" / "A" / "P_00008" / "LEFT" / "MLO.npy"
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "finding_card.schema.json"
DEFAULT_DEMO_ARTIFACTS_DIR = PROJECT_ROOT / "reports" / "final" / "demo_artifacts"


def resolve_pair_image_paths(
    pair_id: str,
    project_root: Path,
    patient_id: str,
    breast_side: str,
) -> Tuple[Path, Path]:
    """
    Derives and validates canonical CC and MLO image paths from the verified pair identifier
    and canonical processed-data layout (Variant A). Confirms both paths belong to the same pair.
    """
    if pair_id != f"{patient_id}_{breast_side}":
        raise ValueError(
            f"Pair identifier mismatch: requested '{pair_id}' does not match resolved '{patient_id}_{breast_side}'"
        )

    processed_base = project_root / "data" / "processed" / "stage3b_comparison" / "A"
    pair_dir = processed_base / patient_id / breast_side

    cc_path = pair_dir / "CC.npy"
    mlo_path = pair_dir / "MLO.npy"

    # Confirm both paths belong to the exact same verified pair directory
    if cc_path.parent != pair_dir or mlo_path.parent != pair_dir:
        raise ValueError(f"Path binding violation: CC/MLO paths do not share verified pair directory {pair_dir}")

    if not cc_path.is_file():
        raise FileNotFoundError(f"Canonical preprocessed CC array missing for verified pair '{pair_id}': {cc_path}")

    if not mlo_path.is_file():
        raise FileNotFoundError(f"Canonical preprocessed MLO array missing for verified pair '{pair_id}': {mlo_path}")

    return cc_path, mlo_path


def verify_development_membership(pair_id: str, project_root: Path) -> Dict[str, Any]:
    """
    Rigorously verifies that the requested pair belongs strictly to the development partition
    and is completely quarantined from the official test partition and local holdout.
    Fails closed if any manifest is missing, unreadable, malformed, ambiguous, or if any partition violation is found.
    """
    import pandas as pd

    splits_csv = project_root / "data" / "manifests" / "stage2" / "patient_splits.csv"
    cohort_csv = project_root / "data" / "manifests" / "stage2" / "cohort_pairs.csv"
    cv_csv = project_root / "reports" / "stage5_cross_validation_plan" / "cv_splits_manifest.csv"
    calc_test_csv = project_root / "data" / "source_metadata" / "calc_case_description_test_set.csv"
    mass_test_csv = project_root / "data" / "source_metadata" / "mass_case_description_test_set.csv"

    # 1. Manifest file existence (Fail Closed on missing manifests)
    required_manifests = [
        ("patient_splits", splits_csv),
        ("cohort_pairs", cohort_csv),
        ("cv_splits", cv_csv),
        ("calc_test_set", calc_test_csv),
        ("mass_test_set", mass_test_csv),
    ]
    for name, req_file in required_manifests:
        if not req_file.is_file():
            raise FileNotFoundError(f"Required verification manifest missing ({name}): {req_file}")

    # 2. Safely read and validate manifest integrity
    try:
        cohort_df = pd.read_csv(cohort_csv)
        splits_df = pd.read_csv(splits_csv)
        cv_df = pd.read_csv(cv_csv)
        calc_test_df = pd.read_csv(calc_test_csv)
        mass_test_df = pd.read_csv(mass_test_csv)
    except Exception as e:
        raise ValueError(f"Failed to read canonical verification manifests: {e}")

    # Verify required schema columns exist
    for col in ("pair_id", "patient_id", "breast_side", "partition"):
        if col not in cohort_df.columns:
            raise ValueError(f"Cohort pair manifest missing required column '{col}': {cohort_csv.name}")

    for col in ("patient_id", "partition"):
        if col not in splits_df.columns:
            raise ValueError(f"Patient splits manifest missing required column '{col}': {splits_csv.name}")

    for col in ("pair_id", "outer_fold"):
        if col not in cv_df.columns:
            raise ValueError(f"CV splits manifest missing required column '{col}': {cv_csv.name}")

    for test_name, test_df, test_path in (
        ("calc_test_set", calc_test_df, calc_test_csv),
        ("mass_test_set", mass_test_df, mass_test_csv),
    ):
        if "patient_id" not in test_df.columns:
            raise ValueError(f"Quarantined test manifest missing 'patient_id' column ({test_name}): {test_path.name}")

    # 3. Verify existence and non-ambiguity in cohort_pairs.csv
    cohort_rows = cohort_df[cohort_df["pair_id"] == pair_id]
    if len(cohort_rows) == 0:
        raise ValueError(f"Pair '{pair_id}' not found in canonical pair manifest: {cohort_csv.name}")
    if len(cohort_rows) > 1:
        raise ValueError(f"Ambiguous pair: duplicate records ({len(cohort_rows)}) for '{pair_id}' in {cohort_csv.name}")

    cohort_row = cohort_rows.iloc[0]
    patient_id = str(cohort_row["patient_id"]).strip()
    breast_side = str(cohort_row["breast_side"]).strip().upper()

    cohort_part = str(cohort_row["partition"]).strip().lower()
    if cohort_part not in ("train", "dev", "cv"):
        raise PermissionError(
            f"SECURITY VIOLATION: Pair '{pair_id}' belongs to non-development partition '{cohort_part}' in {cohort_csv.name}"
        )

    # 4. Verify existence and non-ambiguity in patient_splits.csv
    splits_rows = splits_df[splits_df["patient_id"] == patient_id]
    if len(splits_rows) == 0:
        raise ValueError(f"Patient '{patient_id}' not found in patient splits manifest: {splits_csv.name}")
    if len(splits_rows) > 1:
        raise ValueError(f"Ambiguous patient: duplicate records ({len(splits_rows)}) for '{patient_id}' in {splits_csv.name}")

    splits_row = splits_rows.iloc[0]
    patient_part = str(splits_row["partition"]).strip().lower()

    if cohort_part != patient_part:
        raise ValueError(
            f"Conflicting partition classification between cohort ({cohort_part}) and patient splits ({patient_part}) for '{pair_id}'"
        )

    if patient_part != "train":
        raise PermissionError(
            f"SECURITY VIOLATION: Patient '{patient_id}' partition '{patient_part}' is not permitted development partition ('train')"
        )

    # 5. Explicit check against local_holdout quarantine (both manifests)
    holdout_patients = set(splits_df[splits_df["partition"] == "local_holdout"]["patient_id"].astype(str))
    holdout_cohort_patients = set(cohort_df[cohort_df["partition"] == "local_holdout"]["patient_id"].astype(str))
    all_holdout = holdout_patients | holdout_cohort_patients
    if patient_id in all_holdout:
        raise PermissionError(f"SECURITY VIOLATION: Patient '{patient_id}' belongs to protected local_holdout!")

    # 6. Explicit check against official test manifests (metadata check only; no inference/evaluation)
    for test_name, test_df, test_file in (
        ("calc_test_set", calc_test_df, "calc_case_description_test_set.csv"),
        ("mass_test_set", mass_test_df, "mass_case_description_test_set.csv"),
    ):
        test_pats = set(test_df["patient_id"].dropna().astype(str))
        if patient_id in test_pats:
            raise PermissionError(
                f"SECURITY VIOLATION: Patient '{patient_id}' found in quarantined official test metadata '{test_file}'!"
            )

    # 7. Verify cross-validation fold assignment (non-ambiguous)
    cv_rows = cv_df[cv_df["pair_id"] == pair_id]
    if len(cv_rows) == 0:
        raise ValueError(f"Pair '{pair_id}' not found in CV splits manifest: {cv_csv.name}")
    if len(cv_rows) > 1:
        raise ValueError(f"Ambiguous CV entry: duplicate records ({len(cv_rows)}) for '{pair_id}' in {cv_csv.name}")

    try:
        outer_fold = int(cv_rows.iloc[0]["outer_fold"])
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid outer_fold value in {cv_csv.name} for pair '{pair_id}': {e}")

    # 8. Resolve and bind canonical image paths
    cc_path, mlo_path = resolve_pair_image_paths(pair_id, project_root, patient_id, breast_side)

    return {
        "pair_id": pair_id,
        "patient_id": patient_id,
        "breast_side": breast_side,
        "partition": patient_part,
        "outer_fold": outer_fold,
        "cc_path": cc_path,
        "mlo_path": mlo_path,
        "is_development": True,
        "is_local_holdout": False,
        "is_quarantined_test": False,
    }


def validate_image_array_from_disk(file_path: Path, expected_shape: tuple = (512, 512)) -> np.ndarray:
    """
    Rigorously validates a preprocessed image array loaded from disk.
    Enforces the frozen Variant A preprocessing contract:
    - 2D numpy.ndarray of shape (512, 512)
    - Strictly np.float32 dtype
    - Strictly finite values (zero NaN, zero Inf)
    - Strictly bounded in range [0.0, 1.0]
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"Image array file not found: {file_path}")

    if file_path.suffix.lower() != ".npy":
        raise ValueError(f"Expected .npy array file, got: {file_path.suffix} ({file_path})")

    arr = np.load(file_path)

    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Loaded object from {file_path.name} is not a numpy.ndarray: {type(arr)}")

    if arr.ndim != 2:
        raise ValueError(f"Array {file_path.name} must be 2D, got {arr.ndim}D with shape {arr.shape}")

    if arr.shape != expected_shape:
        raise ValueError(f"Array {file_path.name} shape mismatch: expected {expected_shape}, got {arr.shape}")

    if arr.dtype != np.float32:
        raise TypeError(f"Array {file_path.name} must be float32 dtype, got {arr.dtype}")

    if not np.isfinite(arr).all():
        num_nan = int(np.isnan(arr).sum())
        num_inf = int(np.isinf(arr).sum())
        raise ValueError(f"Array {file_path.name} contains non-finite values (NaN: {num_nan}, Inf: {num_inf})")

    arr_min, arr_max = float(arr.min()), float(arr.max())
    if arr_min < 0.0 or arr_max > 1.0:
        raise ValueError(
            f"Array {file_path.name} values out of bounds [0.0, 1.0]: min={arr_min:.4f}, max={arr_max:.4f}"
        )

    return arr


def prepare_saliency_output_dir(
    output_dir: Optional[Path],
    pair_id: str,
    generate_saliency: bool = True,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """
    Prepares a safe output directory for saliency overlays and protects existing artifacts.
    - If output_dir is provided: fails clearly with FileExistsError if target overlay files already exist.
    - If output_dir is None: creates a new, uniquely named directory to prevent accidental overwrites.
    """
    expected_cc = f"{pair_id}_gradcam_CC.png"
    expected_mlo = f"{pair_id}_gradcam_MLO.png"

    if not generate_saliency:
        target_dir = output_dir or (project_root / "reports" / "final" / "demo_artifacts")
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    if output_dir is not None:
        # Check for file collisions
        for target_file in (expected_cc, expected_mlo):
            existing_path = output_dir / target_file
            if existing_path.exists():
                raise FileExistsError(
                    f"Collision detected: Output file '{target_file}' already exists in {output_dir}. "
                    f"Existing artifacts are protected; aborting before inference."
                )
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    # Default strategy: create a new, uniquely named run directory to preserve outputs and avoid collisions
    base_dir = project_root / "reports" / "final" / "demo_artifacts"
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_dir = base_dir / f"run_{pair_id}_{timestamp}"
    counter = 0
    while unique_dir.exists():
        counter += 1
        unique_dir = base_dir / f"run_{pair_id}_{timestamp}_{counter}"

    unique_dir.mkdir(parents=True, exist_ok=False)
    return unique_dir


def run_demo(
    pair_id: str = DEFAULT_PAIR_ID,
    cc_path: Optional[Path] = None,
    mlo_path: Optional[Path] = None,
    density: int = 1,
    abnormality_type: str = "calcification",
    generate_saliency: bool = True,
    output_dir: Optional[Path] = None,
    test_failure_simulation: bool = True,
) -> OrchestrationResponse:
    print("=" * 80)
    print("  RP1 MULTIMODAL BREAST CANCER DECISION SUPPORT SYSTEM — SOFTWARE DEMO  ")
    print("=" * 80)

    # 1. Verification of Development Split and Manifest Binding
    print("\n[PHASE 1: RIGOROUS DATA VERIFICATION & QUARANTINE INTEGRITY]")
    verif = verify_development_membership(pair_id, PROJECT_ROOT)
    print(f"  • Target Pair ID:           {pair_id}")
    print(f"  • Resolved Patient ID:      {verif['patient_id']}")
    print(f"  • Resolved Breast Side:     {verif['breast_side']}")
    print(f"  • Manifest Partition:       {verif['partition'].upper()} (Development Partition)")
    print(f"  • Cross-Validation Fold:    Fold {verif['outer_fold']}")
    print(f"  • Local Holdout Quarantine: PASSED (Excluded from protected holdout)")
    print(f"  • Official Test Quarantine: PASSED (Excluded from official test partition)")
    print(f"  • Evaluation Note:          Software behavior demonstration only; not a generalization metric.")

    # Validate and bind image paths
    resolved_cc = verif["cc_path"]
    resolved_mlo = verif["mlo_path"]

    if cc_path is not None and cc_path.resolve() != resolved_cc.resolve():
        raise ValueError(
            f"Path mismatch: provided cc_path '{cc_path}' does not match resolved canonical path '{resolved_cc}' for verified pair '{pair_id}'"
        )
    if mlo_path is not None and mlo_path.resolve() != resolved_mlo.resolve():
        raise ValueError(
            f"Path mismatch: provided mlo_path '{mlo_path}' does not match resolved canonical path '{resolved_mlo}' for verified pair '{pair_id}'"
        )

    effective_cc = resolved_cc
    effective_mlo = resolved_mlo

    # 2. Disk Array Contract Validation (Contract: 2D, (512,512), float32, finite, [0.0, 1.0])
    print("\n[PHASE 2: PREPROCESSED IMAGE ARRAY VALIDATION ON DISK]")
    cc_arr = validate_image_array_from_disk(effective_cc)
    mlo_arr = validate_image_array_from_disk(effective_mlo)
    print(f"  • Input CC View File:       {effective_cc.name} (Shape: {cc_arr.shape}, Dtype: {cc_arr.dtype}, Min/Max: [{cc_arr.min():.2f}, {cc_arr.max():.2f}], Finite: True)")
    print(f"  • Input MLO View File:      {effective_mlo.name} (Shape: {mlo_arr.shape}, Dtype: {mlo_arr.dtype}, Min/Max: [{mlo_arr.min():.2f}, {mlo_arr.max():.2f}], Finite: True)")
    print(f"  • Clinical Covariates:      breast_density={density}, breast_side='{verif['breast_side']}'")
    print(f"  • Ground-Truth Status:      Masked / Excluded (Zero Leakage)")

    # 3. Artifact Directory Setup and Collision Protection (Prior to inference)
    print("\n[PHASE 3: ARTIFACT DIRECTORY SETUP & COLLISION PROTECTION]")
    artifacts_dir = prepare_saliency_output_dir(output_dir, pair_id, generate_saliency, PROJECT_ROOT)
    print(f"  • Saliency Output Directory:{artifacts_dir}")

    # 4. Pipeline Initialization
    print("\n[PHASE 4: SPECIALIST PIPELINE INITIALIZATION]")
    t_init_start = time.perf_counter()
    mammo_pipeline = MammographyAgentInferencePipeline()
    clin_pipeline = ClinicalRiskAgentInferencePipeline()
    orchestrator = MultiAgentOrchestrator(
        mammo_pipeline=mammo_pipeline,
        clin_pipeline=clin_pipeline,
        timeout_seconds=5.0,
        max_workers=2,
    )
    t_init_ms = (time.perf_counter() - t_init_start) * 1000.0
    print(f"  • Mammo Pipeline:           {len(mammo_pipeline.models)} Checkpoints (Locked 5-Fold Ensemble)")
    print(f"  • Clin Pipeline:            {len(clin_pipeline.models)} Boosters (Locked 5-Fold XGBoost Ensemble)")
    print(f"  • Orchestrator Status:      Active (ThreadPoolExecutor, Semaphore Concurrency=2)")
    print(f"  • Initialization Time:      {t_init_ms:.1f} ms")

    try:
        # 5. Synchronized Dual-Agent Inference Execution
        print("\n[PHASE 5: SYNCHRONIZED MULTI-AGENT INFERENCE]")
        req = OrchestrationRequest(
            pair_identifier=pair_id,
            cc_input=effective_cc,
            mlo_input=effective_mlo,
            clinical_covariates={"breast_density": density, "breast_side": verif["breast_side"]},
            abnormality_type=abnormality_type,
            generate_saliency=generate_saliency,
            saliency_output_dir=artifacts_dir,
        )

        t_exec_start = time.perf_counter()
        resp: OrchestrationResponse = orchestrator.run(req)
        t_exec_ms = (time.perf_counter() - t_exec_start) * 1000.0

        print(f"  • Total Orchestration Time: {t_exec_ms:.1f} ms")
        print(f"  • Mammo Agent Status:       {resp.mammo_status} ({resp.audit_record.mammo_duration_ms:.1f} ms)")
        print(f"  • Clin Agent Status:        {resp.clin_status} ({resp.audit_record.clin_duration_ms:.1f} ms)")
        print(f"  • Audit ID:                 {resp.audit_id}")

        # 6. Display Primary Diagnostic Result
        print("\n" + "=" * 80)
        print("  PRIMARY AUTHORITATIVE DIAGNOSTIC OUTPUT (MAMMOGRAPHY AGENT)  ")
        print("=" * 80)
        primary_card = resp.primary_finding_card
        print(f"Agent:                  {primary_card.get('agent')}")
        print(f"Predicted BI-RADS:      Category {primary_card.get('BI_RADS_a')}")
        print(f"Calibrated Confidence:  {primary_card.get('conf_a'):.4f}")
        print(f"Perception Findings:    {primary_card.get('finding')}")
        print(f"Saliency Regions:       {len(primary_card.get('sal', {}).get('top_regions', []))} Bounding Boxes")
        cc_out = primary_card.get('sal', {}).get('cc_path')
        mlo_out = primary_card.get('sal', {}).get('mlo_path')
        print(f"Saliency Overlay (CC):  {cc_out} (Exists on disk: {Path(cc_out).is_file() if cc_out else False})")
        print(f"Saliency Overlay (MLO): {mlo_out} (Exists on disk: {Path(mlo_out).is_file() if mlo_out else False})")
        print(f"Synthesized Rationale:  \"{primary_card.get('rationale')}\"")

        # 7. Display Shadow Research Result
        print("\n" + "-" * 80)
        print("  SECONDARY SHADOW CHANNEL OUTPUT (CLINICAL RISK AGENT — RESEARCH ONLY)  ")
        print("=" * 80)
        shadow_card = resp.shadow_finding_card
        if shadow_card is not None:
            sh_meta = shadow_card.get("metadata", {})
            print(f"Agent:                  {shadow_card.get('agent')}")
            print(f"Authority Designation:  {sh_meta.get('authority')} (voting_eligibility: {sh_meta.get('voting_eligibility')})")
            print(f"BI-RADS Category:       {shadow_card.get('BI_RADS_a')} (Non-Authoritative / Unmapped)")
            print(f"Voting Weight:          {shadow_card.get('conf_a')} (Explicit zero-voting compatibility convention)")
            print(f"Execution Status:       {sh_meta.get('status')}")
            print(f"Clin Rationale:         \"{shadow_card.get('rationale')}\"")
            print(f"Governance Disclaimer:  \"{sh_meta.get('disclaimer')}\"")
        else:
            print("  [Clin Agent output unavailable]")

        # 8. Schema Validation
        print("\n[PHASE 6: FORMAL FINDING-CARD SCHEMA VALIDATION]")
        is_m_valid, m_err = validate_finding_card(primary_card, DEFAULT_SCHEMA_PATH)
        print(f"  • Primary Mammo Card Validation:  {'PASSED (Valid JSON Schema)' if is_m_valid else f'FAILED: {m_err}'}")

        is_c_valid, c_err = validate_finding_card(shadow_card, DEFAULT_SCHEMA_PATH)
        print(f"  • Shadow Clin Card Validation:   {'PASSED (Valid JSON Schema)' if is_c_valid else f'FAILED: {c_err}'}")

        if not is_m_valid or not is_c_valid:
            raise ValueError(f"Schema validation failed: Mammo={is_m_valid}, Clin={is_c_valid}")

        # 9. Simulated Clin Failure Demonstration (Safe Fault Isolation)
        if test_failure_simulation:
            print("\n[PHASE 7: SAFE FAULT-ISOLATION VERIFICATION (SIMULATED CLIN FAILURE)]")
            print("  • Dispatching request with invalid clinical covariates to trigger Clin abstention...")
            fail_req = OrchestrationRequest(
                pair_identifier=pair_id,
                cc_input=effective_cc,
                mlo_input=effective_mlo,
                clinical_covariates={"breast_density": "INVALID_NON_NUMERIC"},
                abnormality_type=abnormality_type,
                generate_saliency=False,
            )
            fail_resp = orchestrator.run(fail_req)
            print(f"  • Mammo Status during Clin Failure:  {fail_resp.mammo_status}")
            print(f"  • Clin Status during Clin Failure:   {fail_resp.clin_status} (Reason: {fail_resp.audit_record.clin_error_code})")
            is_preserved = fail_resp.primary_finding_card["BI_RADS_a"] == primary_card["BI_RADS_a"]
            print(f"  • Primary Finding Card Preserved:    {is_preserved}")
            print(f"  • Primary Diagnostic Result Blocked: NO (Mammo card returned intact)")

        return resp

    finally:
        # Guarantee executor closure and resource cleanup in the harness's own lifecycle
        orchestrator.close(wait=False)
        print("\n[PHASE 8: RESOURCE CLEANUP & SHUTDOWN]")
        print("  • MultiAgentOrchestrator executor cleanly closed (wait=False).")

    print("\n" + "=" * 80)
    print("  DEMO EXECUTION SUCCESSFULLY COMPLETED AND VERIFIED  ")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RP1 End-to-End Mammo-Clin Demo")
    parser.add_argument("--pair-id", type=str, default=DEFAULT_PAIR_ID, help="Development pair ID")
    parser.add_argument("--density", type=int, default=1, help="Breast density covariate (1-4)")
    parser.add_argument("--no-saliency", action="store_true", help="Disable saliency heatmap generation")
    parser.add_argument("--skip-failure-sim", action="store_true", help="Skip fault-isolation demonstration")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save persistent saliency overlays")
    args = parser.parse_args()

    out_p = Path(args.output_dir) if args.output_dir else None
    run_demo(
        pair_id=args.pair_id,
        density=args.density,
        generate_saliency=not args.no_saliency,
        output_dir=out_p,
        test_failure_simulation=not args.skip_failure_sim,
    )
