"""
Stage 5 — Patient-Level 5-Fold Cross-Validation for CBIS-DDSM Binary Malignancy (GPU Accelerated).
NVIDIA RTX 3050 (6 GB VRAM) Mixed-Precision Training (FP16 AMP).
Features:
- Deterministic, patient-disjoint cyclic block reservation (Outer Test, Inner Val, Inner Train).
- Mixed precision via torch.amp.autocast('cuda') + torch.amp.GradScaler('cuda').
- Full state checkpointing: weights, optimizer, scheduler, scaler, RNGs, epoch, metrics, and patience.
- Resumption capability: support for resuming mid-fold or skipping completed folds via --resume.
- Zero touch of local holdout and official test partitions.
"""
import os
import sys
import json
import time
import math
import random
import csv
import hashlib
import argparse
from pathlib import Path

# Enforce offline operation
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, r"D:\rp")
from src.models.paired_model import PairedDualViewModel
from src.models.dataset import PairedMammogramDataset, collate_paired_batch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_roc_auc_numpy(targets: np.ndarray, probs: np.ndarray) -> float:
    """Exact ROC-AUC via Mann-Whitney U statistic with average tie-ranking."""
    n_pos = int(np.sum(targets == 1))
    n_neg = int(np.sum(targets == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
        
    order = np.argsort(probs)
    sorted_probs = probs[order]
    sorted_targets = targets[order]
    
    ranks = np.zeros(len(probs), dtype=float)
    i = 0
    while i < len(probs):
        j = i
        while j < len(probs) and sorted_probs[j] == sorted_probs[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[i:j] = avg_rank
        i = j
        
    rank_sum_pos = np.sum(ranks[sorted_targets == 1])
    u = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))


def compute_pr_auc_numpy(targets: np.ndarray, probs: np.ndarray) -> tuple[float, list[float], list[float]]:
    """Average Precision (PR-AUC) via step integration."""
    order = np.argsort(probs)[::-1]
    sorted_probs = probs[order]
    sorted_targets = targets[order]
    
    n_pos = int(np.sum(targets == 1))
    if n_pos == 0:
        return 0.0, [0.0], [0.0]
        
    tp_cumsum = np.cumsum(sorted_targets == 1)
    fp_cumsum = np.cumsum(sorted_targets == 0)
    
    recalls = tp_cumsum / n_pos
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    recalls = np.concatenate([[0.0], recalls])
    precisions = np.concatenate([[precisions[0]], precisions])
    
    ap = float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]))
    return ap, recalls.tolist(), precisions.tolist()


def compute_roc_curve_points(targets: np.ndarray, probs: np.ndarray):
    """Computes FPR, TPR points and optimal thresholds."""
    thresholds = np.sort(np.unique(probs))[::-1]
    thresholds = np.concatenate([[thresholds[0] + 1e-6], thresholds, [0.0]])
    
    tprs = []
    fprs = []
    n_pos = int(np.sum(targets == 1))
    n_neg = int(np.sum(targets == 0))
    
    best_j = -1.0
    best_thresh = 0.5
    sens_at_spec90 = 0.0
    thresh_at_spec90 = 0.5
    
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        tp = np.sum((preds == 1) & (targets == 1))
        fp = np.sum((preds == 1) & (targets == 0))
        
        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        spec = 1.0 - fpr
        
        tprs.append(tpr)
        fprs.append(fpr)
        
        j = tpr + spec - 1.0
        if j > best_j:
            best_j = j
            best_thresh = float(thresh)
            
        if spec >= 0.90:
            if tpr > sens_at_spec90:
                sens_at_spec90 = tpr
                thresh_at_spec90 = float(thresh)
                
    return np.array(fprs), np.array(tprs), thresholds, best_thresh, float(sens_at_spec90), float(thresh_at_spec90)


def compute_metrics(targets: np.ndarray, probs: np.ndarray, threshold: float = 0.50):
    """Computes discrimination and threshold classification metrics."""
    roc_auc = compute_roc_auc_numpy(targets, probs)
    pr_auc, rec_list, prec_list = compute_pr_auc_numpy(targets, probs)
    brier = float(np.mean((probs - targets) ** 2))
    
    preds = (probs >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (targets == 1)))
    fp = int(np.sum((preds == 1) & (targets == 0)))
    tn = int(np.sum((preds == 0) & (targets == 0)))
    fn = int(np.sum((preds == 0) & (targets == 1)))
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
    acc = (tp + tn) / len(targets)
    
    fprs, tprs, threshs, opt_thresh, sens_at_spec90, thresh_spec90 = compute_roc_curve_points(targets, probs)
    
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "sensitivity_at_spec90": float(sens_at_spec90),
        "threshold_at_spec90": float(thresh_spec90),
        "optimal_threshold_youden": float(opt_thresh),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }


def compute_calibration_metrics(targets: np.ndarray, probs: np.ndarray, n_bins: int = 10):
    """Computes Expected Calibration Error (ECE), MCE, and reliability diagram bins."""
    brier = float(np.mean((probs - targets) ** 2))
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    
    ece = 0.0
    mce = 0.0
    bins_data = []
    
    total_samples = len(targets)
    for i in range(n_bins):
        b_low = bin_boundaries[i]
        b_high = bin_boundaries[i+1]
        
        if i == n_bins - 1:
            in_bin = (probs >= b_low) & (probs <= b_high)
        else:
            in_bin = (probs >= b_low) & (probs < b_high)
            
        count = int(np.sum(in_bin))
        prop = count / total_samples
        
        if count > 0:
            avg_conf = float(np.mean(probs[in_bin]))
            avg_acc = float(np.mean(targets[in_bin]))
            abs_err = abs(avg_conf - avg_acc)
            ece += abs_err * prop
            mce = max(mce, abs_err)
            acc_se = float(np.sqrt(avg_acc * (1.0 - avg_acc) / count)) if count > 1 else 0.0
            
            bins_data.append({
                "bin_index": i,
                "bin_range": [float(b_low), float(b_high)],
                "count": count,
                "confidence": avg_conf,
                "accuracy": avg_acc,
                "accuracy_se": acc_se,
                "abs_error": abs_err
            })
        else:
            bins_data.append({
                "bin_index": i,
                "bin_range": [float(b_low), float(b_high)],
                "count": 0,
                "confidence": float((b_low + b_high) / 2.0),
                "accuracy": 0.0,
                "accuracy_se": 0.0,
                "abs_error": 0.0
            })
            
    return {
        "brier_score": brier,
        "ece": float(ece),
        "mce": float(mce),
        "bins": bins_data
    }


def fit_temperature_scaling(logits: np.ndarray, targets: np.ndarray) -> float:
    """Fits scalar temperature T > 0 by minimizing NLL on inner validation set."""
    t_logits = torch.from_numpy(logits).float()
    t_targets = torch.from_numpy(targets).float()
    
    log_temp = nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=50)
    criterion = nn.BCEWithLogitsLoss()
    
    def eval_fn():
        optimizer.zero_grad()
        temp = torch.exp(log_temp)
        scaled_logits = t_logits / temp
        loss = criterion(scaled_logits, t_targets)
        loss.backward()
        return loss
        
    optimizer.step(eval_fn)
    T = float(torch.exp(log_temp).item())
    return max(0.05, min(T, 10.0))


def bootstrap_metrics_patient_clustered(targets: np.ndarray, probs: np.ndarray, patient_ids: list[str], n_bootstraps: int = 2000, seed: int = 42):
    """Computes 95% bootstrap confidence intervals with patient-level clustering."""
    rng = np.random.RandomState(seed)
    unique_patients = np.unique(patient_ids)
    n_pats = len(unique_patients)
    
    pat_to_indices = {p: [] for p in unique_patients}
    for idx, p in enumerate(patient_ids):
        pat_to_indices[p].append(idx)
        
    boot_roc = []
    boot_pr = []
    boot_brier = []
    boot_sens_spec90 = []
    
    for _ in range(n_bootstraps):
        sampled_pats = rng.choice(unique_patients, size=n_pats, replace=True)
        sampled_indices = []
        for p in sampled_pats:
            sampled_indices.extend(pat_to_indices[p])
            
        y_true = targets[sampled_indices]
        y_prob = probs[sampled_indices]
        
        if len(np.unique(y_true)) < 2:
            continue
            
        m = compute_metrics(y_true, y_prob)
        boot_roc.append(m["roc_auc"])
        boot_pr.append(m["pr_auc"])
        boot_brier.append(m["brier_score"])
        boot_sens_spec90.append(m["sensitivity_at_spec90"])
        
    return {
        "roc_auc_ci_95": [float(np.percentile(boot_roc, 2.5)), float(np.percentile(boot_roc, 97.5))],
        "pr_auc_ci_95": [float(np.percentile(boot_pr, 2.5)), float(np.percentile(boot_pr, 97.5))],
        "brier_score_ci_95": [float(np.percentile(boot_brier, 2.5)), float(np.percentile(boot_brier, 97.5))],
        "sens_at_spec90_ci_95": [float(np.percentile(boot_sens_spec90, 2.5)), float(np.percentile(boot_sens_spec90, 97.5))]
    }


def evaluate_gpu(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    """Runs a complete forward evaluation pass with AMP on GPU."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_probs = []
    all_logits = []
    all_pair_ids = []
    all_patient_ids = []
    
    with torch.no_grad():
        for batch in loader:
            cc = batch["X_cc"].to(device, non_blocking=True)
            mlo = batch["X_mlo"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
                logits, alpha = model(cc, mlo)
                loss = criterion(logits, targets)
                probs = torch.sigmoid(logits)
            
            bs = targets.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            
            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_logits.extend(logits.cpu().numpy().tolist())
            all_pair_ids.extend(batch["pair_ids"])
            all_patient_ids.extend(batch["patient_ids"])
            
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    targets_np = np.array(all_targets)
    probs_np = np.array(all_probs)
    
    metrics = compute_metrics(targets_np, probs_np)
    metrics["val_loss"] = float(avg_loss)
    
    return metrics, targets_np, probs_np, np.array(all_logits), all_pair_ids, all_patient_ids


def generate_svg_plot(filename: Path, title: str, xlabel: str, ylabel: str, series: list[dict]):
    """Generates a standalone clean SVG vector graphic."""
    width, height = 700, 450
    margin = {"top": 60, "right": 40, "bottom": 60, "left": 70}
    pw = width - margin["left"] - margin["right"]
    ph = height - margin["top"] - margin["bottom"]
    
    all_x = [pt[0] for s in series for pt in s["points"]]
    all_y = [pt[1] for s in series for pt in s["points"]]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    if max_x == min_x: max_x += 1.0
    if max_y == min_y: max_y += 1.0
    
    def tx(x): return margin["left"] + (x - min_x) / (max_x - min_x) * pw
    def ty(y): return margin["top"] + ph - (y - min_y) / (max_y - min_y) * ph
    
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="background:#ffffff; font-family:sans-serif;">']
    svg.append(f'<text x="{width/2}" y="32" text-anchor="middle" font-size="16" font-weight="bold" fill="#222">{title}</text>')
    svg.append(f'<rect x="{margin["left"]}" y="{margin["top"]}" width="{pw}" height="{ph}" fill="#fcfcfc" stroke="#ccc" stroke-width="1"/>')
    
    for i in range(5):
        y_val = min_y + i * (max_y - min_y) / 4
        y_pos = ty(y_val)
        svg.append(f'<line x1="{margin["left"]}" y1="{y_pos}" x2="{margin["left"]+pw}" y2="{y_pos}" stroke="#eee" stroke-width="1"/>')
        svg.append(f'<text x="{margin["left"]-8}" y="{y_pos+4}" text-anchor="end" font-size="11" fill="#666">{y_val:.2f}</text>')
        
    for i in range(5):
        x_val = min_x + i * (max_x - min_x) / 4
        x_pos = tx(x_val)
        svg.append(f'<line x1="{x_pos}" y1="{margin["top"]}" x2="{x_pos}" y2="{margin["top"]+ph}" stroke="#eee" stroke-width="1"/>')
        svg.append(f'<text x="{x_pos}" y="{margin["top"]+ph+18}" text-anchor="middle" font-size="11" fill="#666">{x_val:.1f}</text>')
        
    svg.append(f'<text x="{margin["left"]+pw/2}" y="{height-15}" text-anchor="middle" font-size="13" font-weight="500" fill="#333">{xlabel}</text>')
    svg.append(f'<text x="20" y="{margin["top"]+ph/2}" text-anchor="middle" font-size="13" font-weight="500" fill="#333" transform="rotate(-90 20 {margin["top"]+ph/2})">{ylabel}</text>')
    
    for s in series:
        color = s.get("color", "#2563eb")
        width_line = s.get("width", 2)
        dash = s.get("dash", "")
        pts = s["points"]
        if len(pts) < 2: continue
        d_str = " ".join([f'{"M" if i == 0 else "L"} {tx(p[0]):.1f} {ty(p[1]):.1f}' for i, p in enumerate(pts)])
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(f'<path d="{d_str}" fill="none" stroke="{color}" stroke-width="{width_line}"{dash_attr}/>')
        
    leg_x = margin["left"] + 20
    leg_y = margin["top"] + 25
    for idx, s in enumerate(series):
        name = s.get("name", "")
        color = s.get("color", "#2563eb")
        svg.append(f'<rect x="{leg_x}" y="{leg_y + idx*20 - 10}" width="14" height="14" fill="{color}"/>')
        svg.append(f'<text x="{leg_x + 20}" y="{leg_y + idx*20 + 2}" font-size="12" fill="#333">{name}</text>')
        
    svg.append('</svg>')
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def run_stage5_cross_validation_gpu(resume: bool = False):
    print("=" * 80)
    print("STAGE 5 — PATIENT-LEVEL 5-FOLD CROSS-VALIDATION (GPU ACCELERATED)")
    print("NVIDIA RTX 3050 (6 GB VRAM) Mixed-Precision Baseline")
    print("=" * 80)
    
    start_total_time = time.time()
    
    # Verify CUDA device
    assert torch.cuda.is_available(), "FATAL: CUDA not available for GPU run!"
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Device: {gpu_name} ({gpu_mem:.2f} GB VRAM)")
    
    # 1. Output directory setup (Isolated GPU directory)
    plan_dir = Path(r"D:\rp\reports\stage5_cross_validation_plan")
    cv_dir = Path(r"D:\rp\reports\stage5_cross_validation_gpu")
    cv_dir.mkdir(parents=True, exist_ok=True)
    (cv_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (cv_dir / "figures").mkdir(parents=True, exist_ok=True)
    
    manifest_src = plan_dir / "cv_splits_manifest.csv"
    manifest_dst = cv_dir / "cv_splits_manifest.csv"
    
    if not manifest_src.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_src}")
        
    with open(manifest_src, "rb") as f:
        manifest_bytes = f.read()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    expected_hash = "855596b5c77b3aa9970a1e07b2c68353abf6be7d909029c2fb42c5e46e1ec25e"
    print(f"Split Manifest SHA-256: {manifest_hash}")
    assert manifest_hash == expected_hash, f"FATAL: Manifest hash mismatch! Expected {expected_hash}, got {manifest_hash}"
    
    with open(manifest_dst, "wb") as f:
        f.write(manifest_bytes)
        
    with open(manifest_dst, "r", encoding="utf-8") as f:
        all_records = list(csv.DictReader(f))
        
    print(f"Loaded {len(all_records)} development pairs across {len(set(r['patient_id'] for r in all_records))} unique patients.")
    
    # Global Pre-Flight Assertions
    cache_dir = Path(r"D:\rp\data\cached_preprocessed_512x512")
    for r in all_records:
        assert (cache_dir / r["cc_cache_file"]).is_file(), f"Missing CC file: {r['cc_cache_file']}"
        assert (cache_dir / r["mlo_cache_file"]).is_file(), f"Missing MLO file: {r['mlo_cache_file']}"
    print("Pre-flight: All 1,692 cached numpy tensors verified readable on disk.")
    
    canonical_splits = r"D:\rp\data\manifests\stage2\patient_splits.csv"
    with open(canonical_splits, "r", encoding="utf-8") as f:
        r_splits = list(csv.DictReader(f))
    holdout_pats = set(r["\ufeffpatient_id"] for r in r_splits if r["partition"] == "local_holdout")
    dev_pats = set(r["patient_id"] for r in all_records)
    assert dev_pats.isdisjoint(holdout_pats), "FATAL: Development pool overlaps with Local Holdout!"
    print("Pre-flight: Local Holdout firewall verified (0 overlapping patients).")
    
    # Containers
    all_oof_predictions = []
    all_fold_histories = []
    calibration_models = {}
    fold_summaries = {}
    
    log_path = cv_dir / "cv_execution_log.txt"
    log_fp = open(log_path, "a" if resume else "w", encoding="utf-8")
    
    def log_print(msg: str):
        print(msg, flush=True)
        log_fp.write(msg + "\n")
        log_fp.flush()
        
    log_print(f"Stage 5 GPU 5-Fold Cross-Validation launched at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"Manifest Hash: {manifest_hash}")
    log_print(f"Hardware: {gpu_name} ({gpu_mem:.2f} GB VRAM) | AMP FP16 Enabled\n")
    
    batch_size = 8
    num_accumulation_steps = 2  # Effective batch size = 16
    
    # Check for existing partial OOF predictions if resuming
    oof_partial_path = cv_dir / "oof_predictions_partial.json"
    if resume and oof_partial_path.exists():
        with open(oof_partial_path, "r", encoding="utf-8") as f:
            all_oof_predictions = json.load(f)
        log_print(f"Resumption: Loaded {len(all_oof_predictions)} existing out-of-fold predictions.")
        
    for fold_k in range(5):
        fold_start_time = time.time()
        ckpt_best_path = cv_dir / "checkpoints" / f"model_fold{fold_k}_best.pt"
        ckpt_latest_path = cv_dir / "checkpoints" / f"model_fold{fold_k}_latest.pt"
        
        # Check if fold already completed in resume mode
        fold_k_completed = False
        if resume and ckpt_best_path.exists():
            fold_pats = set(r["patient_id"] for r in all_records if r[f"fold{fold_k}_role"] == "outer_test")
            existing_oof_pats = set(p["patient_id"] for p in all_oof_predictions if p.get("outer_fold") == fold_k)
            if fold_pats == existing_oof_pats and len(existing_oof_pats) > 0:
                log_print(f"Fold {fold_k} already completed. Skipping to next fold.")
                fold_k_completed = True
                
        if fold_k_completed:
            continue
            
        log_print("=" * 80)
        log_print(f"STARTING FOLD {fold_k} (Outer Test = Fold {fold_k}, Inner Val = Fold {(fold_k+1)%5})")
        log_print("=" * 80)
        
        role_col = f"fold{fold_k}_role"
        inner_train_records = [r for r in all_records if r[role_col] == "inner_train"]
        inner_val_records = [r for r in all_records if r[role_col] == "inner_val"]
        outer_test_records = [r for r in all_records if r[role_col] == "outer_test"]
        
        pats_train = set(r["patient_id"] for r in inner_train_records)
        pats_val = set(r["patient_id"] for r in inner_val_records)
        pats_test = set(r["patient_id"] for r in outer_test_records)
        
        assert pats_train.isdisjoint(pats_val)
        assert pats_train.isdisjoint(pats_test)
        assert pats_val.isdisjoint(pats_test)
        assert len(pats_train | pats_val | pats_test) == 786
        
        log_print(f"Fold {fold_k} Partitions:")
        log_print(f"  Inner Train: {len(pats_train)} patients, {len(inner_train_records)} pairs")
        log_print(f"  Inner Val:   {len(pats_val)} patients, {len(inner_val_records)} pairs")
        log_print(f"  Outer Test:  {len(pats_test)} patients, {len(outer_test_records)} pairs")
        
        train_targets = [int(r["binary_target"]) for r in inner_train_records]
        n_benign = sum(1 for t in train_targets if t == 0)
        n_malignant = sum(1 for t in train_targets if t == 1)
        pos_weight_value = n_benign / n_malignant
        pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        log_print(f"  Train Balance: Benign={n_benign}, Malignant={n_malignant} -> pos_weight={pos_weight_value:.6f}")
        
        fold_seed = 42 + fold_k
        set_seed(fold_seed)
        
        train_dataset = PairedMammogramDataset(inner_train_records, is_training=True, seed=fold_seed)
        val_dataset = PairedMammogramDataset(inner_val_records, is_training=False, seed=fold_seed)
        test_dataset = PairedMammogramDataset(outer_test_records, is_training=False, seed=fold_seed)
        
        # Dataloaders with pin_memory on GPU
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_paired_batch, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_paired_batch, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_paired_batch, num_workers=2, pin_memory=True)
        
        model = PairedDualViewModel(pretrained_backbone=True, dropout=0.3).to(device)
        scaler = torch.amp.GradScaler("cuda")
        
        # Checkpoint state
        best_val_loss = float("inf")
        best_epoch = -1
        patience_counter = 0
        patience_limit = 5
        start_epoch = 1
        
        # Resumption mid-fold
        if resume and ckpt_latest_path.exists():
            ckpt_latest = torch.load(ckpt_latest_path, map_location=device)
            if ckpt_latest.get("fold") == fold_k and ckpt_latest.get("epoch") < 20:
                start_epoch = ckpt_latest["epoch"] + 1
                model.load_state_dict(ckpt_latest["model_state_dict"])
                best_val_loss = ckpt_latest.get("best_val_loss", best_val_loss)
                best_epoch = ckpt_latest.get("best_epoch", best_epoch)
                patience_counter = ckpt_latest.get("patience_counter", 0)
                if "scaler_state_dict" in ckpt_latest and ckpt_latest["scaler_state_dict"] is not None:
                    scaler.load_state_dict(ckpt_latest["scaler_state_dict"])
                if "rng_state_torch" in ckpt_latest:
                    torch.set_rng_state(ckpt_latest["rng_state_torch"])
                if "rng_state_cuda" in ckpt_latest and ckpt_latest["rng_state_cuda"] is not None:
                    torch.cuda.set_rng_state(ckpt_latest["rng_state_cuda"])
                log_print(f"Resumption: Resuming Fold {fold_k} from Epoch {start_epoch} (Best Val Loss so far: {best_val_loss:.4f} at Epoch {best_epoch}).")
                
        # Phase 1 Warmup (Epochs 1-3)
        if start_epoch <= 3:
            model.set_backbone_trainable(False)
            optimizer_p1 = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=1e-3,
                weight_decay=1e-4
            )
            
            for ep in range(start_epoch, 4):
                ep_start = time.time()
                model.train()
                train_loss_acc = 0.0
                optimizer_p1.zero_grad()
                
                for step_idx, batch in enumerate(train_loader):
                    cc = batch["X_cc"].to(device, non_blocking=True)
                    mlo = batch["X_mlo"].to(device, non_blocking=True)
                    targets = batch["targets"].to(device, non_blocking=True)
                    
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits, alpha = model(cc, mlo)
                        loss = criterion(logits, targets) / num_accumulation_steps
                        
                    scaler.scale(loss).backward()
                    
                    if (step_idx + 1) % num_accumulation_steps == 0 or (step_idx + 1) == len(train_loader):
                        scaler.step(optimizer_p1)
                        scaler.update()
                        optimizer_p1.zero_grad()
                        
                    train_loss_acc += loss.item() * num_accumulation_steps * targets.size(0)
                    
                ep_duration = time.time() - ep_start
                tr_loss = train_loss_acc / len(inner_train_records)
                val_m, _, _, _, _, _ = evaluate_gpu(model, val_loader, criterion, device)
                peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
                
                log_print(f"Epoch {ep} Complete (Warmup GPU): Duration={ep_duration:.1f}s | Peak VRAM={peak_vram:.1f}MB | Train Loss={tr_loss:.4f} | InnerVal Loss={val_m['val_loss']:.4f} | InnerVal ROC-AUC={val_m['roc_auc']:.4f}")
                
                if val_m["val_loss"] < best_val_loss:
                    best_val_loss = val_m["val_loss"]
                    best_epoch = ep
                    torch.save({
                        "fold": fold_k,
                        "epoch": ep,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer_p1.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "val_loss": best_val_loss,
                        "val_roc_auc": val_m["roc_auc"],
                        "val_metrics": val_m,
                        "rng_state_torch": torch.get_rng_state(),
                        "rng_state_cuda": torch.cuda.get_rng_state(),
                        "best_val_loss": best_val_loss,
                        "best_epoch": best_epoch,
                        "patience_counter": patience_counter
                    }, ckpt_best_path)
                    log_print(f"  -> Checkpoint updated (Lowest InnerVal Loss: {best_val_loss:.4f})")
                    
                all_fold_histories.append({
                    "fold": fold_k,
                    "epoch": ep,
                    "phase": 1,
                    "duration_sec": ep_duration,
                    "peak_vram_mb": peak_vram,
                    "train_loss": tr_loss,
                    "val_loss": val_m["val_loss"],
                    "val_roc_auc": val_m["roc_auc"],
                    "val_pr_auc": val_m["pr_auc"],
                    "val_brier": val_m["brier_score"],
                    "val_sens_spec90": val_m["sensitivity_at_spec90"],
                    "lr_backbone": 0.0,
                    "lr_head": 1e-3
                })
            start_epoch = 4
            
        # Phase 2 Fine-Tuning (Epochs 4-20)
        log_print(f"\n--- Fold {fold_k}: Phase 2 Fine-Tuning (Backbone Unfrozen, GPU) ---")
        model.set_backbone_trainable(True)
        optimizer_p2 = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": 1e-5, "weight_decay": 1e-4},
            {"params": list(model.fusion.parameters()) + list(model.head.parameters()), "lr": 1e-4, "weight_decay": 1e-4}
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=17, eta_min=1e-6)
        
        # Advance scheduler if resuming mid-phase 2
        if start_epoch > 4:
            for _ in range(start_epoch - 4):
                scheduler.step()
                
        for ep in range(start_epoch, 21):
            ep_start = time.time()
            model.train()
            train_loss_acc = 0.0
            optimizer_p2.zero_grad()
            
            for step_idx, batch in enumerate(train_loader):
                cc = batch["X_cc"].to(device, non_blocking=True)
                mlo = batch["X_mlo"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)
                
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits, alpha = model(cc, mlo)
                    loss = criterion(logits, targets) / num_accumulation_steps
                    
                scaler.scale(loss).backward()
                
                if (step_idx + 1) % num_accumulation_steps == 0 or (step_idx + 1) == len(train_loader):
                    scaler.step(optimizer_p2)
                    scaler.update()
                    optimizer_p2.zero_grad()
                    
                train_loss_acc += loss.item() * num_accumulation_steps * targets.size(0)
                
            scheduler.step()
            ep_duration = time.time() - ep_start
            tr_loss = train_loss_acc / len(inner_train_records)
            val_m, _, _, _, _, _ = evaluate_gpu(model, val_loader, criterion, device)
            peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
            
            curr_lr_bb = optimizer_p2.param_groups[0]["lr"]
            curr_lr_head = optimizer_p2.param_groups[1]["lr"]
            log_print(f"Epoch {ep:2d} (Fine-Tune GPU): Duration={ep_duration:.1f}s | Peak VRAM={peak_vram:.1f}MB | Train Loss={tr_loss:.4f} | InnerVal Loss={val_m['val_loss']:.4f} | InnerVal ROC-AUC={val_m['roc_auc']:.4f}")
            
            # Save latest checkpoint with full state
            torch.save({
                "fold": fold_k,
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_p2.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_loss": val_m["val_loss"],
                "val_roc_auc": val_m["roc_auc"],
                "val_metrics": val_m,
                "rng_state_torch": torch.get_rng_state(),
                "rng_state_cuda": torch.cuda.get_rng_state(),
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "patience_counter": patience_counter
            }, ckpt_latest_path)
            
            if val_m["val_loss"] < best_val_loss:
                best_val_loss = val_m["val_loss"]
                best_epoch = ep
                patience_counter = 0
                torch.save({
                    "fold": fold_k,
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer_p2.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "val_loss": best_val_loss,
                    "val_roc_auc": val_m["roc_auc"],
                    "val_metrics": val_m,
                    "rng_state_torch": torch.get_rng_state(),
                    "rng_state_cuda": torch.cuda.get_rng_state(),
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "patience_counter": patience_counter
                }, ckpt_best_path)
                log_print(f"  -> Checkpoint updated (Lowest InnerVal Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                log_print(f"  -> No improvement in InnerVal loss ({patience_counter}/{patience_limit})")
                if patience_counter >= patience_limit:
                    log_print(f"  -> Early stopping triggered at Epoch {ep}. Reverting to Epoch {best_epoch}.")
                    break
                    
            all_fold_histories.append({
                "fold": fold_k,
                "epoch": ep,
                "phase": 2,
                "duration_sec": ep_duration,
                "peak_vram_mb": peak_vram,
                "train_loss": tr_loss,
                "val_loss": val_m["val_loss"],
                "val_roc_auc": val_m["roc_auc"],
                "val_pr_auc": val_m["pr_auc"],
                "val_brier": val_m["brier_score"],
                "val_sens_spec90": val_m["sensitivity_at_spec90"],
                "lr_backbone": curr_lr_bb,
                "lr_head": curr_lr_head
            })
            
        # 4. Out-of-Fold Evaluation & Calibration for Fold k
        log_print(f"\n--- Fold {fold_k}: Post-Training Calibration & Outer Test Evaluation ---")
        best_ckpt = torch.load(ckpt_best_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        model.eval()
        
        # Fit temperature on inner val
        val_m, val_targets_np, val_probs_np, val_logits_np, _, _ = evaluate_gpu(model, val_loader, criterion, device)
        T_k = fit_temperature_scaling(val_logits_np, val_targets_np)
        log_print(f"Fold {fold_k}: Selected Checkpoint Epoch={best_epoch} | InnerVal Loss={val_m['val_loss']:.4f} | InnerVal ROC-AUC={val_m['roc_auc']:.4f}")
        log_print(f"Fold {fold_k}: Fitted InnerVal Temperature T = {T_k:.4f}")
        calibration_models[f"fold_{fold_k}"] = {
            "selected_epoch": best_epoch,
            "inner_val_loss": val_m["val_loss"],
            "inner_val_roc_auc": val_m["roc_auc"],
            "temperature": T_k
        }
        
        # Evaluate ONCE on outer test
        test_m, test_targets_np, test_probs_raw, test_logits_np, test_pair_ids, test_patient_ids = evaluate_gpu(model, test_loader, criterion, device)
        test_probs_cal = 1.0 / (1.0 + np.exp(-test_logits_np / T_k))
        
        test_cal_m = compute_calibration_metrics(test_targets_np, test_probs_cal)
        test_raw_cal_m = compute_calibration_metrics(test_targets_np, test_probs_raw)
        
        log_print(f"Fold {fold_k} Outer Test Results (UNBIASED OUT-OF-SAMPLE):")
        log_print(f"  ROC-AUC:                {test_m['roc_auc']:.4f}")
        log_print(f"  PR-AUC:                 {test_m['pr_auc']:.4f}")
        log_print(f"  Brier (Raw / Cal):      {test_raw_cal_m['brier_score']:.4f} / {test_cal_m['brier_score']:.4f}")
        log_print(f"  ECE (Raw / Cal):        {test_raw_cal_m['ece']:.4f} / {test_cal_m['ece']:.4f}")
        log_print(f"  Sens @ Spec 90%:        {test_m['sensitivity_at_spec90']:.4f}")
        
        for i in range(len(test_targets_np)):
            all_oof_predictions.append({
                "pair_id": test_pair_ids[i],
                "patient_id": test_patient_ids[i],
                "outer_fold": fold_k,
                "binary_target": int(test_targets_np[i]),
                "logit": float(test_logits_np[i]),
                "prob_raw": float(test_probs_raw[i]),
                "prob_calibrated": float(test_probs_cal[i])
            })
            
        fold_summaries[f"fold_{fold_k}"] = {
            "best_epoch": best_epoch,
            "temperature": T_k,
            "outer_test_metrics": test_m,
            "outer_test_calibration_raw": test_raw_cal_m,
            "outer_test_calibration_cal": test_cal_m,
            "fold_duration_sec": time.time() - fold_start_time
        }
        
        # Save partial predictions checkpoint for resumption
        with open(oof_partial_path, "w", encoding="utf-8") as f:
            json.dump(all_oof_predictions, f)
            
        log_print(f"Fold {fold_k} completed in {(time.time() - fold_start_time)/60:.1f} minutes.\n")
        
    # 5. Global Out-of-Fold Aggregation
    log_print("=" * 80)
    log_print("AGGREGATING 5-FOLD OUT-OF-FOLD (OOF) PREDICTIONS")
    log_print("=" * 80)
    
    assert len(all_oof_predictions) == 846, f"Expected 846 OOF predictions, got {len(all_oof_predictions)}"
    
    oof_csv_path = cv_dir / "oof_predictions.csv"
    with open(oof_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_oof_predictions[0].keys()))
        writer.writeheader()
        writer.writerows(all_oof_predictions)
    log_print(f"Wrote {oof_csv_path} with 846 predictions.")
    
    hist_csv_path = cv_dir / "cv_training_history.csv"
    with open(hist_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_fold_histories[0].keys()))
        writer.writeheader()
        writer.writerows(all_fold_histories)
    log_print(f"Wrote {hist_csv_path} with {len(all_fold_histories)} epoch records.")
    
    with open(cv_dir / "calibration_models.json", "w", encoding="utf-8") as f:
        json.dump(calibration_models, f, indent=2)
        
    oof_targets = np.array([p["binary_target"] for p in all_oof_predictions])
    oof_probs_raw = np.array([p["prob_raw"] for p in all_oof_predictions])
    oof_probs_cal = np.array([p["prob_calibrated"] for p in all_oof_predictions])
    oof_patient_ids = [p["patient_id"] for p in all_oof_predictions]
    
    oof_raw_metrics = compute_metrics(oof_targets, oof_probs_raw)
    oof_cal_metrics = compute_metrics(oof_targets, oof_probs_cal)
    
    cal_analysis_raw = compute_calibration_metrics(oof_targets, oof_probs_raw)
    cal_analysis_cal = compute_calibration_metrics(oof_targets, oof_probs_cal)
    
    boot_ci = bootstrap_metrics_patient_clustered(oof_targets, oof_probs_cal, oof_patient_ids, n_bootstraps=2000, seed=42)
    
    log_print("\nPOOLED OUT-OF-FOLD (OOF) PERFORMANCE EVALUATION (846 Pairs / 786 Patients):")
    log_print(f"  ROC-AUC:                       {oof_cal_metrics['roc_auc']:.4f} (95% CI: [{boot_ci['roc_auc_ci_95'][0]:.4f}, {boot_ci['roc_auc_ci_95'][1]:.4f}])")
    log_print(f"  PR-AUC (Average Precision):    {oof_cal_metrics['pr_auc']:.4f} (95% CI: [{boot_ci['pr_auc_ci_95'][0]:.4f}, {boot_ci['pr_auc_ci_95'][1]:.4f}])")
    log_print(f"  Brier Score (Raw / Cal):       {cal_analysis_raw['brier_score']:.4f} / {cal_analysis_cal['brier_score']:.4f} (95% CI: [{boot_ci['brier_score_ci_95'][0]:.4f}, {boot_ci['brier_score_ci_95'][1]:.4f}])")
    log_print(f"  Expected Calibration Error:    {cal_analysis_raw['ece']:.4f} (Raw) -> {cal_analysis_cal['ece']:.4f} (Calibrated)")
    log_print(f"  Sensitivity @ 90% Specificity: {oof_cal_metrics['sensitivity_at_spec90']:.4f} (at threshold {oof_cal_metrics['threshold_at_spec90']:.4f})")
    log_print(f"  Optimal Youden Threshold:      {oof_cal_metrics['optimal_threshold_youden']:.4f}")
    
    lock_data = {
        "locking_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "Stage 5 GPU 5-Fold Cross-Validation Locking Seal",
        "cohort": "846 Retained Development Pairs (786 Unique Patients)",
        "locked_operating_thresholds": {
            "tau_screening_spec90": oof_cal_metrics["threshold_at_spec90"],
            "tau_balanced_youden": oof_cal_metrics["optimal_threshold_youden"],
            "tau_default_050": 0.50
        },
        "locked_calibration_temperatures": {f"fold_{k}": calibration_models[f"fold_{k}"]["temperature"] for k in range(5)},
        "baseline_oof_metrics": {
            "roc_auc": oof_cal_metrics["roc_auc"],
            "pr_auc": oof_cal_metrics["pr_auc"],
            "brier_score": cal_analysis_cal["brier_score"],
            "ece": cal_analysis_cal["ece"],
            "sensitivity_at_spec90": oof_cal_metrics["sensitivity_at_spec90"],
            "optimal_threshold_youden": oof_cal_metrics["optimal_threshold_youden"]
        }
    }
    with open(cv_dir / "evaluation_lock.json", "w", encoding="utf-8") as f:
        json.dump(lock_data, f, indent=2)
    log_print(f"Wrote {cv_dir / 'evaluation_lock.json'}")
    
    eval_summary = {
        "experiment_name": "Stage 5 GPU 5-Fold Cross-Validation",
        "task": "Binary Malignancy Classification",
        "hardware": f"{gpu_name} ({gpu_mem:.2f} GB VRAM)",
        "backbone": "tf_efficientnet_lite0 (ImageNet-1k offline)",
        "precision": "Mixed Precision FP16 (AMP)",
        "total_development_pairs": 846,
        "total_development_patients": 786,
        "fold_summaries": fold_summaries,
        "pooled_oof_metrics_calibrated": oof_cal_metrics,
        "pooled_oof_metrics_raw": oof_raw_metrics,
        "calibration_analysis_raw": cal_analysis_raw,
        "calibration_analysis_calibrated": cal_analysis_cal,
        "bootstrap_95_ci": boot_ci,
        "total_wall_clock_seconds": time.time() - start_total_time
    }
    with open(cv_dir / "oof_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)
    log_print(f"Wrote {cv_dir / 'oof_evaluation_summary.json'}")
    
    # SVG Visualizations
    log_print("\nGenerating vector SVG visualizations...")
    loss_series = []
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#d97706"]
    for k in range(5):
        f_hist = [h for h in all_fold_histories if h["fold"] == k]
        pts_tr = [(h["epoch"], h["train_loss"]) for h in f_hist]
        pts_val = [(h["epoch"], h["val_loss"]) for h in f_hist]
        loss_series.append({"name": f"Fold {k} InnerTrain", "points": pts_tr, "color": colors[k], "width": 1, "dash": "3,3"})
        loss_series.append({"name": f"Fold {k} InnerVal", "points": pts_val, "color": colors[k], "width": 2})
    generate_svg_plot(cv_dir / "figures" / "fold_loss_trajectories.svg", "Stage 5 GPU 5-Fold Cross-Validation: Loss Trajectories", "Epoch", "BCE Loss", loss_series)
    
    fprs_raw, tprs_raw, _, _, _, _ = compute_roc_curve_points(oof_targets, oof_probs_cal)
    roc_pts = [(float(fprs_raw[i]), float(tprs_raw[i])) for i in range(len(fprs_raw))]
    roc_series = [
        {"name": f"Pooled OOF ROC (AUC = {oof_cal_metrics['roc_auc']:.4f})", "points": roc_pts, "color": "#2563eb", "width": 3},
        {"name": "Random Chance", "points": [(0.0, 0.0), (1.0, 1.0)], "color": "#9ca3af", "width": 1, "dash": "4,4"}
    ]
    generate_svg_plot(cv_dir / "figures" / "oof_roc_curve.svg", "Stage 5 GPU Pooled Out-of-Fold ROC Curve (N=846)", "False Positive Rate (1 - Specificity)", "True Positive Rate (Sensitivity)", roc_series)
    
    _, rec_list, prec_list = compute_pr_auc_numpy(oof_targets, oof_probs_cal)
    pr_pts = [(float(rec_list[i]), float(prec_list[i])) for i in range(len(rec_list))]
    base_pos_rate = float(np.mean(oof_targets))
    pr_series = [
        {"name": f"Pooled OOF PR (AP = {oof_cal_metrics['pr_auc']:.4f})", "points": pr_pts, "color": "#059669", "width": 3},
        {"name": f"Prevalence Baseline ({base_pos_rate:.2%})", "points": [(0.0, base_pos_rate), (1.0, base_pos_rate)], "color": "#9ca3af", "width": 1, "dash": "4,4"}
    ]
    generate_svg_plot(cv_dir / "figures" / "oof_pr_curve.svg", "Stage 5 GPU Pooled Out-of-Fold Precision-Recall Curve (N=846)", "Recall (Sensitivity)", "Precision (PPV)", pr_series)
    
    cal_pts = [(b["confidence"], b["accuracy"]) for b in cal_analysis_cal["bins"] if b["count"] > 0]
    raw_pts = [(b["confidence"], b["accuracy"]) for b in cal_analysis_raw["bins"] if b["count"] > 0]
    cal_series = [
        {"name": f"Calibrated (ECE = {cal_analysis_cal['ece']:.4f})", "points": cal_pts, "color": "#2563eb", "width": 2},
        {"name": f"Raw Sigmoid (ECE = {cal_analysis_raw['ece']:.4f})", "points": raw_pts, "color": "#dc2626", "width": 2, "dash": "3,3"},
        {"name": "Perfect Calibration", "points": [(0.0, 0.0), (1.0, 1.0)], "color": "#9ca3af", "width": 1, "dash": "4,4"}
    ]
    generate_svg_plot(cv_dir / "figures" / "oof_calibration_curve.svg", "Stage 5 GPU Pooled Out-of-Fold Calibration Curve", "Mean Predicted Probability", "Empirical Fraction of Positives", cal_series)
    
    # Checkpoint reproducibility verification
    log_print("\nVerifying checkpoint reload reproducibility for all 5 folds...")
    for k in range(5):
        ckpt_path = cv_dir / "checkpoints" / f"model_fold{k}_best.pt"
        test_model = PairedDualViewModel(pretrained_backbone=False, dropout=0.3).to(device)
        ckpt_obj = torch.load(ckpt_path, map_location=device)
        test_model.load_state_dict(ckpt_obj["model_state_dict"])
        test_model.eval()
        
        test_records_k = [r for r in all_records if r[f"fold{k}_role"] == "outer_test"][:2]
        test_ds = PairedMammogramDataset(test_records_k, is_training=False, seed=42)
        test_ld = DataLoader(test_ds, batch_size=2, shuffle=False, collate_fn=collate_paired_batch)
        b = next(iter(test_ld))
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                l1, _ = test_model(b["X_cc"].to(device), b["X_mlo"].to(device))
                l2, _ = test_model(b["X_cc"].to(device), b["X_mlo"].to(device))
            assert torch.allclose(l1, l2), f"Fold {k}: Non-deterministic forward pass!"
    log_print("All 5 fold checkpoints verified bit-reproducible.")
    
    total_duration_sec = time.time() - start_total_time
    log_print("=" * 80)
    log_print(f"STAGE 5 GPU 5-FOLD CROSS-VALIDATION COMPLETED SUCCESSFULLY in {total_duration_sec/3600:.2f} hours ({total_duration_sec:.1f}s)")
    log_print("=" * 80)
    log_fp.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5 5-Fold Cross-Validation on GPU")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoints and completed folds")
    args = parser.parse_args()
    run_stage5_cross_validation_gpu(resume=args.resume)
