"""
Stage 4B Caching Script: Preprocess and cache eligible train and validation pairs.
Applies frozen Variant A preprocessing, enforces 0.80 component retention threshold,
and logs/excludes failures under approved reason codes (Option D/C).
"""
import os
import sys
import json
import csv
import time
from pathlib import Path
import numpy as np

# Import frozen preprocessing
sys.path.insert(0, r'D:\rp\src')
from preprocessing.mammography import (
    decode_dicom,
    validate_image_provenance,
    metadata_from_dataset,
    validate_pixel_metadata,
    preprocess_pixels,
    PreprocessConfig,
    ReviewRequired,
    DecodeError,
    MalformedMetadataError,
    PreprocessingError
)

def main():
    print("[1/5] Initializing caching environment and configurations...")
    cache_dir = Path(r"D:\rp\data\cached_preprocessed_512x512")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = Path(r"D:\rp\configs\preprocessing\stage3_pilot.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg = PreprocessConfig(**{k: v for k, v in cfg_dict.items() if hasattr(PreprocessConfig, k)})
    print(f"Loaded PreprocessConfig: output_size={cfg.output_size}, retention_threshold={cfg.minimum_component_retention}")

    manifest_path = Path(r"D:\rp\data\manifests\stage2\cohort_pairs.csv")
    with open(manifest_path, "r", encoding="utf-8-sig") as f:
        all_rows = list(csv.DictReader(f))
    
    # Filter strictly to train and validation. Never holdout!
    dev_pairs = [r for r in all_rows if r["partition"] in ("train", "validation")]
    print(f"Total development pairs in manifest: {len(dev_pairs)} (Train: {sum(1 for r in dev_pairs if r['partition'] == 'train')}, Val: {sum(1 for r in dev_pairs if r['partition'] == 'validation')})")

    exclusions = []
    retained_pairs = []
    
    total_views = len(dev_pairs) * 2
    processed_views = 0
    start_time = time.time()
    
    print("\n[2/5] Beginning caching of CC and MLO views...")
    for idx, pair in enumerate(dev_pairs, 1):
        pair_id = pair["pair_id"]
        patient_id = pair["patient_id"]
        partition = pair["partition"]
        side = pair["breast_side"]
        
        pair_failed = False
        failure_view = None
        failure_retention = 0.0
        failure_reason = None
        failure_code = None
        
        pair_images = {}
        
        # Process CC then MLO
        for view in ["CC", "MLO"]:
            processed_views += 1
            role = view.lower()
            dcm_path = Path(pair[f"{role}_local_path"])
            
            try:
                decoded = decode_dicom(dcm_path)
                prov = validate_image_provenance(decoded.dataset, pair, view)
                meta = metadata_from_dataset(decoded.dataset)
                validate_pixel_metadata(decoded.pixels, meta)
                img, metrics, _ = preprocess_pixels(decoded.pixels, meta, cfg)
                
                ret = metrics.get("component_retention", 0.0)
                if ret < cfg.minimum_component_retention:
                    pair_failed = True
                    failure_view = view
                    failure_retention = ret
                    failure_code = "QC_FAIL_LOW_COMPONENT_RETENTION"
                    failure_reason = f"Largest component retains {ret:.4f} < {cfg.minimum_component_retention:.2f}"
                    break
                
                # Check output range
                if img.min() < 0.0 or img.max() > 1.0:
                    raise PreprocessingError(f"Image array out of range [0, 1]: min={img.min()}, max={img.max()}")
                
                pair_images[view] = (img, ret)
                
            except ReviewRequired as e:
                pair_failed = True
                failure_view = view
                failure_code = "QC_FAIL_LOW_COMPONENT_RETENTION" if "retains only" in str(e) else "QC_FAIL_REVIEW_REQUIRED"
                failure_reason = str(e)
                break
            except Exception as e:
                pair_failed = True
                failure_view = view
                failure_code = "QC_FAIL_PREPROCESSING_ERROR"
                failure_reason = str(e)
                break
        
        if pair_failed:
            partner_view = "MLO" if failure_view == "CC" else "CC"
            print(f"  [EXCLUSION] Pair {pair_id} ({partition}): {failure_view} failed -> {failure_code}: {failure_reason}")
            exclusions.append({
                "pair_id": pair_id,
                "patient_id": patient_id,
                "partition": partition,
                "failed_view": failure_view,
                "partner_view": partner_view,
                "measured_retention": failure_retention,
                "failure_code": failure_code,
                "partner_code": "QC_EXCLUDED_PARTNER_VIEW_FAILED",
                "reason_details": failure_reason
            })
        else:
            # Save both views as compressed numpy arrays (.npy)
            for view, (img, ret) in pair_images.items():
                out_fname = f"{patient_id}_{side}_{view}.npy"
                out_path = cache_dir / out_fname
                np.save(str(out_path), img.astype(np.float32))
            
            retained_pairs.append({
                "pair_id": pair_id,
                "patient_id": patient_id,
                "breast_side": side,
                "partition": partition,
                "binary_target": int(pair["binary_target"]),
                "binary_target_name": pair["binary_target_name"],
                "cc_cache_file": f"{patient_id}_{side}_CC.npy",
                "mlo_cache_file": f"{patient_id}_{side}_MLO.npy",
                "cc_retention": pair_images["CC"][1],
                "mlo_retention": pair_images["MLO"][1]
            })
            
        if idx % 50 == 0 or idx == len(dev_pairs):
            elapsed = time.time() - start_time
            rate = processed_views / elapsed
            remaining = (total_views - processed_views) / rate if rate > 0 else 0
            print(f"  Progress: {idx}/{len(dev_pairs)} pairs ({processed_views}/{total_views} views) | Rate: {rate:.2f} views/s | Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m")

    print("\n[3/5] Caching complete. Writing logs and manifests...")
    report_dir = Path(r"D:\rp\reports\stage4b_pilot")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Save exclusions log
    exclusions_path = report_dir / "qc_exclusions.csv"
    with open(exclusions_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["pair_id", "patient_id", "partition", "failed_view", "partner_view", "measured_retention", "failure_code", "partner_code", "reason_details"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(exclusions)
    print(f"Wrote {len(exclusions)} exclusions to {exclusions_path}")
    
    # Save retained cache manifest
    manifest_out_path = report_dir / "cached_cohort_manifest.csv"
    with open(manifest_out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["pair_id", "patient_id", "breast_side", "partition", "binary_target", "binary_target_name", "cc_cache_file", "mlo_cache_file", "cc_retention", "mlo_retention"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(retained_pairs)
    print(f"Wrote {len(retained_pairs)} retained pairs to {manifest_out_path}")

    # Cohort accounting
    train_retained = [p for p in retained_pairs if p["partition"] == "train"]
    val_retained = [p for p in retained_pairs if p["partition"] == "validation"]
    
    train_benign = sum(1 for p in train_retained if p["binary_target"] == 0)
    train_malignant = sum(1 for p in train_retained if p["binary_target"] == 1)
    
    val_benign = sum(1 for p in val_retained if p["binary_target"] == 0)
    val_malignant = sum(1 for p in val_retained if p["binary_target"] == 1)
    
    train_pts = len(set(p["patient_id"] for p in train_retained))
    val_pts = len(set(p["patient_id"] for p in val_retained))
    
    # Patient disjointness check
    overlap = set(p["patient_id"] for p in train_retained).intersection(set(p["patient_id"] for p in val_retained))
    assert len(overlap) == 0, f"PATIENT OVERLAP DETECTED: {overlap}"
    
    pos_weight = train_benign / train_malignant if train_malignant > 0 else 1.0
    
    print("\n[4/5] ================= COHORT ACCOUNTING =================")
    print(f"TRAIN:      {len(train_retained)} pairs across {train_pts} patients")
    print(f"            Benign (0): {train_benign} ({train_benign/len(train_retained)*100:.2f}%)")
    print(f"            Malignant (1): {train_malignant} ({train_malignant/len(train_retained)*100:.2f}%)")
    print(f"            Computed pos_weight (N_benign / N_malignant): {pos_weight:.6f}")
    print(f"VALIDATION: {len(val_retained)} pairs across {val_pts} patients")
    print(f"            Benign (0): {val_benign} ({val_benign/len(val_retained)*100:.2f}%)")
    print(f"            Malignant (1): {val_malignant} ({val_malignant/len(val_retained)*100:.2f}%)")
    print(f"EXCLUSIONS: {len(exclusions)} total pairs excluded across train and val")
    print(f"DISJOINT:   Zero patient overlap verified between train and validation.")
    print("=========================================================\n")

if __name__ == "__main__":
    main()
