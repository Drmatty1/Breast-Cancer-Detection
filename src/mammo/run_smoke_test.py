"""
Real-Data Smoke Test of the Integrated Mammography Agent.

Safeguards:
- Uses strictly one development pair: P_00004_LEFT (Patient P_00004, development partition).
- Zero access to quarantined Local Holdout or Official Test partitions.
- Saves deliverables under reports/mammo_agent_smoke_test/.
- Strictly read-only on all existing checkpoints, manifests, and reports.
"""
from pathlib import Path
import sys
import os

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import time
import json
import numpy as np
import torch
import cv2
import jsonschema

from src.mammo.inference import MammographyAgentInferencePipeline
from src.mammo.finding_card import validate_finding_card


def run_smoke_test():
    print("=" * 80)
    print("MAMMO AGENT REAL-DATA SMOKE TEST")
    print("=" * 80)

    project_root = Path(r"D:\rp")
    out_dir = project_root / "reports" / "mammo_agent_smoke_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = project_root / "data" / "cached_preprocessed_512x512"
    pair_id = "P_00004_LEFT"
    patient_id = "P_00004"

    # 1. Preflight Partition Verification
    splits_path = project_root / "data" / "manifests" / "stage2" / "patient_splits.csv"
    p_splits = pd_read_splits = open(splits_path, "r", encoding="utf-8").readlines()
    pat_lines = [l for l in p_splits if patient_id in l]
    assert len(pat_lines) > 0, f"Patient {patient_id} not found in patient_splits.csv"
    assert "local_holdout" not in pat_lines[0], f"FATAL: Patient {patient_id} is in local_holdout!"
    print(f"Preflight: Pair {pair_id} verified as development patient ({pat_lines[0].strip()})")

    cc_path = cache_dir / f"{pair_id}_CC.npy"
    mlo_path = cache_dir / f"{pair_id}_MLO.npy"
    assert cc_path.is_file(), f"Missing CC file: {cc_path}"
    assert mlo_path.is_file(), f"Missing MLO file: {mlo_path}"

    cc_arr = np.load(cc_path)
    mlo_arr = np.load(mlo_path)
    assert cc_arr.shape == (512, 512), f"CC shape mismatch: {cc_arr.shape}"
    assert mlo_arr.shape == (512, 512), f"MLO shape mismatch: {mlo_arr.shape}"
    print(f"Loaded {pair_id} CC and MLO views: (512, 512), float32, range [{cc_arr.min():.2f}, {cc_arr.max():.2f}]")

    # 2. Pipeline Initialization
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing inference pipeline on {device}...")
    pipeline = MammographyAgentInferencePipeline(device=device)
    init_time = time.time() - t0
    print(f"Pipeline initialized in {init_time:.2f}s with {len(pipeline.models)} checkpoints.")

    # 3. Single-Pass Paired Inference + Saliency
    t1 = time.time()
    card = pipeline.predict(
        cc_input=cc_arr,
        mlo_input=mlo_arr,
        generate_saliency=True,
        saliency_output_dir=out_dir,
        pair_identifier=pair_id
    )
    inference_time = time.time() - t1
    print(f"Inference and Grad-CAM completed in {inference_time:.3f}s.")

    # 4. Standardize Output File Paths as Required by Spec
    cc_overlay_final = out_dir / "cc_gradcam_overlay.png"
    mlo_overlay_final = out_dir / "mlo_gradcam_overlay.png"
    finding_card_final = out_dir / "finding_card.json"

    # Rename / copy overlays to exact filenames
    src_cc_overlay = out_dir / f"{pair_id}_gradcam_CC.png"
    src_mlo_overlay = out_dir / f"{pair_id}_gradcam_MLO.png"

    if src_cc_overlay.is_file():
        src_cc_overlay.replace(cc_overlay_final)
    if src_mlo_overlay.is_file():
        src_mlo_overlay.replace(mlo_overlay_final)

    # Update paths in finding card
    card["sal"]["cc_path"] = str(cc_overlay_final)
    card["sal"]["mlo_path"] = str(mlo_overlay_final)
    card["sal"]["path"] = str(cc_overlay_final)

    # 5. Schema Validation
    schema_path = project_root / "schemas" / "finding_card.schema.json"
    is_valid, err = validate_finding_card(card, schema_path)
    assert is_valid, f"FATAL: Finding card failed schema validation: {err}"
    print(f"Finding card successfully validated against {schema_path.name}.")

    # Write serialized finding card
    with open(finding_card_final, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    print(f"Saved Finding Card: {finding_card_final}")
    print(f"Saved CC Overlay:   {cc_overlay_final}")
    print(f"Saved MLO Overlay:  {mlo_overlay_final}")

    # 6. Detailed Inspection of Model Outputs
    print("\n" + "=" * 80)
    print("MAMMO AGENT OUTPUT SUMMARY")
    print("=" * 80)
    print(f"Agent:                 {card['agent']}")
    print(f"Finding - Lesion:      {card['finding']['lesion']}")
    print(f"Finding - Margin:      {card['finding']['margin']}")
    print(f"Finding - Calc:        {card['finding']['calcification']}")
    print(f"BI-RADS Category:      {card['BI_RADS_a']}")
    print(f"Calibrated Confidence: {card['conf_a']:.4f}")
    print(f"Saliency Regions:      {len(card['sal']['top_regions'])} detected: {card['sal']['top_regions']}")
    print(f"Rationale:             {card['rationale']}")
    print(f"Task Probabilities:    {card['metadata']['task_probabilities']}")
    print(f"Disclaimer:            {card['metadata']['disclaimer']}")

    # Return summary dict for report generation
    return {
        "pair_id": pair_id,
        "patient_id": patient_id,
        "init_time": init_time,
        "inference_time": inference_time,
        "device": str(device),
        "finding_card": card,
        "finding_card_path": str(finding_card_final),
        "cc_overlay_path": str(cc_overlay_final),
        "mlo_overlay_path": str(mlo_overlay_final)
    }


if __name__ == "__main__":
    run_smoke_test()
