"""
Stage 4B Pilot — Binary Malignancy Training and Evaluation Script (Arm A: ImageNet Initialization).
Fully self-contained: relies ONLY on Python standard library + numpy + torch + timm (no pandas/sklearn/matplotlib required).
"""
import os
import sys
import json
import time
import math
import random
import csv
from pathlib import Path

# Ensure offline operation
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure local source imports
sys.path.insert(0, r"D:\rp")
from src.models.paired_model import PairedDualViewModel
from src.models.dataset import PairedMammogramDataset, collate_paired_batch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_roc_auc_numpy(targets: np.ndarray, probs: np.ndarray) -> float:
    """Exact ROC-AUC via Mann-Whitney U statistic (handles ties via average ranks)."""
    n_pos = int(np.sum(targets == 1))
    n_neg = int(np.sum(targets == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
        
    order = np.argsort(probs)
    sorted_probs = probs[order]
    sorted_targets = targets[order]
    
    # Compute ranks with tie-averaging
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


def compute_roc_curve_points(targets: np.ndarray, probs: np.ndarray):
    """Computes FPR, TPR points and optimal Youden threshold."""
    thresholds = np.sort(np.unique(probs))[::-1]
    thresholds = np.concatenate([[thresholds[0] + 1e-6], thresholds, [0.0]])
    
    tprs = []
    fprs = []
    n_pos = int(np.sum(targets == 1))
    n_neg = int(np.sum(targets == 0))
    
    best_j = -1.0
    best_thresh = 0.5
    sens_at_spec90 = 0.0
    
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
                
    return np.array(fprs), np.array(tprs), thresholds, best_thresh, float(sens_at_spec90)


def compute_pr_auc_numpy(targets: np.ndarray, probs: np.ndarray) -> tuple[float, list[float], list[float]]:
    """Computes Average Precision (PR-AUC) using trapezoidal / step integration."""
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
    
    # Prepend recall=0, precision=first precision
    recalls = np.concatenate([[0.0], recalls])
    precisions = np.concatenate([[precisions[0]], precisions])
    
    # Average precision
    ap = float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]))
    return ap, recalls.tolist(), precisions.tolist()


def compute_metrics(targets: np.ndarray, probs: np.ndarray, threshold: float = 0.50):
    """Computes comprehensive metrics using pure numpy."""
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
    
    fprs, tprs, threshs, opt_thresh, sens_at_spec90 = compute_roc_curve_points(targets, probs)
    
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "accuracy_0.5": float(acc),
        "sensitivity_0.5": float(sens),
        "specificity_0.5": float(spec),
        "precision_0.5": float(prec),
        "f1_0.5": float(f1),
        "sensitivity_at_spec90": float(sens_at_spec90),
        "optimal_threshold_youden": opt_thresh,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "fprs": fprs.tolist(), "tprs": tprs.tolist(),
        "recalls": rec_list, "precisions": prec_list
    }


def patient_cluster_bootstrap_ci(
    patient_ids: list[str],
    targets: np.ndarray,
    probs: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42
):
    """Patient-level cluster bootstrap 95% CIs."""
    rng = np.random.RandomState(seed)
    
    # Map patient_id to list of indices
    pat_to_indices = {}
    for idx, pid in enumerate(patient_ids):
        pat_to_indices.setdefault(pid, []).append(idx)
        
    unique_patients = list(pat_to_indices.keys())
    n_pat = len(unique_patients)
    
    boot_roc = []
    boot_pr = []
    boot_brier = []
    boot_sens_spec90 = []
    
    for _ in range(n_resamples):
        sampled_pats = rng.choice(unique_patients, size=n_pat, replace=True)
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


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    """Runs a complete validation pass."""
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
            cc = batch["X_cc"].to(device)
            mlo = batch["X_mlo"].to(device)
            targets = batch["targets"].to(device)
            
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
    """Generates a standalone, beautiful SVG vector graphic plot without external packages."""
    width, height = 700, 450
    margin = {"top": 60, "right": 40, "bottom": 60, "left": 70}
    pw = width - margin["left"] - margin["right"]
    ph = height - margin["top"] - margin["bottom"]
    
    # Determine bounds
    all_x = [pt[0] for s in series for pt in s["points"]]
    all_y = [pt[1] for s in series for pt in s["points"]]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    if max_x == min_x: max_x += 1.0
    if max_y == min_y: max_y += 1.0
    
    def tx(x): return margin["left"] + (x - min_x) / (max_x - min_x) * pw
    def ty(y): return margin["top"] + ph - (y - min_y) / (max_y - min_y) * ph
    
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="background:#ffffff; font-family:sans-serif;">']
    
    # Title
    svg.append(f'<text x="{width/2}" y="32" text-anchor="middle" font-size="16" font-weight="bold" fill="#222">{title}</text>')
    
    # Grid and axes
    svg.append(f'<rect x="{margin["left"]}" y="{margin["top"]}" width="{pw}" height="{ph}" fill="#fcfcfc" stroke="#ccc" stroke-width="1"/>')
    
    # Ticks Y
    for i in range(5):
        y_val = min_y + i * (max_y - min_y) / 4
        y_pos = ty(y_val)
        svg.append(f'<line x1="{margin["left"]}" y1="{y_pos}" x2="{margin["left"]+pw}" y2="{y_pos}" stroke="#eee" stroke-width="1"/>')
        svg.append(f'<text x="{margin["left"]-8}" y="{y_pos+4}" text-anchor="end" font-size="11" fill="#666">{y_val:.2f}</text>')
        
    # Ticks X
    for i in range(5):
        x_val = min_x + i * (max_x - min_x) / 4
        x_pos = tx(x_val)
        svg.append(f'<line x1="{x_pos}" y1="{margin["top"]}" x2="{x_pos}" y2="{margin["top"]+ph}" stroke="#eee" stroke-width="1"/>')
        svg.append(f'<text x="{x_pos}" y="{margin["top"]+ph+18}" text-anchor="middle" font-size="11" fill="#666">{x_val:.1f}</text>')
        
    # Axis labels
    svg.append(f'<text x="{margin["left"]+pw/2}" y="{height-15}" text-anchor="middle" font-size="13" font-weight="500" fill="#333">{xlabel}</text>')
    svg.append(f'<text x="20" y="{margin["top"]+ph/2}" text-anchor="middle" font-size="13" font-weight="500" fill="#333" transform="rotate(-90 20 {margin["top"]+ph/2})">{ylabel}</text>')
    
    # Series
    legend_items = []
    for s in series:
        pts = s["points"]
        color = s.get("color", "#1f77b4")
        width_line = s.get("width", 2)
        dash = s.get("dash", "")
        dash_attr = f'stroke-dasharray="{dash}"' if dash else ""
        
        path_data = " ".join([f"{'M' if idx==0 else 'L'} {tx(p[0]):.1f} {ty(p[1]):.1f}" for idx, p in enumerate(pts)])
        svg.append(f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="{width_line}" {dash_attr}/>')
        
        # Dots if requested
        if s.get("dots", False):
            for p in pts:
                svg.append(f'<circle cx="{tx(p[0]):.1f}" cy="{ty(p[1]):.1f}" r="3.5" fill="{color}"/>')
                
        legend_items.append((s["name"], color))
        
    # Legend
    leg_x = margin["left"] + 15
    leg_y = margin["top"] + 20
    for idx, (name, col) in enumerate(legend_items):
        item_y = leg_y + idx * 18
        svg.append(f'<line x1="{leg_x}" y1="{item_y}" x2="{leg_x+20}" y2="{item_y}" stroke="{col}" stroke-width="2.5"/>')
        svg.append(f'<text x="{leg_x+26}" y="{item_y+4}" font-size="11" fill="#333">{name}</text>')
        
    svg.append('</svg>')
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def train_stage4b(seed: int = 42):
    set_seed(seed)
    device = torch.device("cpu")
    print(f"=== STAGE 4B PILOT TRAINING LAUNCH (Arm A: ImageNet Initialization | Seed: {seed}) ===")
    print(f"PyTorch Device: {device}")
    
    # Paths
    report_dir = Path(r"D:\rp\reports\stage4b_pilot")
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = report_dir / "cached_cohort_manifest.csv"
    checkpoints_dir = report_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Cached manifest not found: {manifest_path}")
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_records = list(reader)
        
    print(f"Loaded cached manifest with {len(all_records)} pairs.")
    
    train_records = [r for r in all_records if r["partition"] == "train"]
    val_records = [r for r in all_records if r["partition"] in ["val", "validation"]]
    
    print(f"\nCohort Accounting:")
    print(f"  Train pairs: {len(train_records)}")
    print(f"  Val pairs:   {len(val_records)}")
    
    # Assert patient disjointness
    train_patients = set(r["patient_id"] for r in train_records)
    val_patients = set(r["patient_id"] for r in val_records)
    overlap = train_patients.intersection(val_patients)
    print(f"  Unique train patients: {len(train_patients)}")
    print(f"  Unique val patients:   {len(val_patients)}")
    print(f"  Patient overlap count: {len(overlap)}")
    assert len(overlap) == 0, f"FATAL: Patient leakage detected! Overlapping patients: {overlap}"
    
    # Recompute class distribution & pos_weight on retained training set
    train_targets = [int(r["binary_target"]) for r in train_records]
    n_train_benign = sum(1 for t in train_targets if t == 0)
    n_train_malignant = sum(1 for t in train_targets if t == 1)
    pos_weight_value = n_train_benign / n_train_malignant
    
    val_targets = [int(r["binary_target"]) for r in val_records]
    n_val_benign = sum(1 for t in val_targets if t == 0)
    n_val_malignant = sum(1 for t in val_targets if t == 1)
    
    print(f"\nClass Distribution & Loss Weighting:")
    print(f"  Train: Benign={n_train_benign} ({n_train_benign/len(train_records):.2%}), Malignant={n_train_malignant} ({n_train_malignant/len(train_records):.2%})")
    print(f"  Val:   Benign={n_val_benign} ({n_val_benign/len(val_records):.2%}), Malignant={n_val_malignant} ({n_val_malignant/len(val_records):.2%})")
    print(f"  Calculated pos_weight = {n_train_benign} / {n_train_malignant} = {pos_weight_value:.6f}")
    
    # Dataloaders
    batch_size = 8
    train_dataset = PairedMammogramDataset(train_records, is_training=True, seed=seed)
    val_dataset = PairedMammogramDataset(val_records, is_training=False, seed=seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_paired_batch,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_paired_batch,
        num_workers=0
    )
    
    # Loss criterion
    pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    # Model instantiation
    print(f"\nInitializing PairedDualViewModel with offline ImageNet weights...")
    model = PairedDualViewModel(pretrained_backbone=True, dropout=0.3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    
    # Phase 1 setup (warmup epochs 1-3)
    model.set_backbone_trainable(False)
    optimizer_p1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=1e-4
    )
    
    # Benchmark Epoch 1 to determine epoch budget
    print(f"\n--- BENCHMARKING EPOCH 1 ---")
    start_ep1 = time.time()
    
    model.train()
    ep1_loss = 0.0
    ep1_samples = 0
    for batch in train_loader:
        cc = batch["X_cc"].to(device)
        mlo = batch["X_mlo"].to(device)
        targets = batch["targets"].to(device)
        
        optimizer_p1.zero_grad()
        logits, _ = model(cc, mlo)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer_p1.step()
        
        ep1_loss += loss.item() * targets.size(0)
        ep1_samples += targets.size(0)
        
    ep1_duration = time.time() - start_ep1
    ep1_train_loss = ep1_loss / ep1_samples
    print(f"Epoch 1 completed in {ep1_duration:.2f} seconds ({ep1_duration/60:.2f} minutes). Train Loss: {ep1_train_loss:.4f}")
    
    # Evaluate Epoch 1
    val_m1, _, _, _, _, _ = evaluate(model, val_loader, criterion, device)
    print(f"Epoch 1 Validation: Loss={val_m1['val_loss']:.4f}, ROC-AUC={val_m1['roc_auc']:.4f}, PR-AUC={val_m1['pr_auc']:.4f}")
    
    # 15-minute epoch budget rule
    if ep1_duration > 15 * 60:
        total_epochs = 8
        print(f"\n[RULE] Epoch 1 duration ({ep1_duration/60:.2f}m) > 15m. Using fallback budget of {total_epochs} epochs.")
    else:
        total_epochs = 20
        print(f"\n[RULE] Epoch 1 duration ({ep1_duration/60:.2f}m) <= 15m. Using standard budget of {total_epochs} epochs.")
        
    history = [{
        "epoch": 1,
        "phase": 1,
        "backbone_frozen": True,
        "epoch_duration_sec": ep1_duration,
        "train_loss": ep1_train_loss,
        "val_loss": val_m1["val_loss"],
        "roc_auc": val_m1["roc_auc"],
        "pr_auc": val_m1["pr_auc"],
        "brier_score": val_m1["brier_score"],
        "sensitivity_0.5": val_m1["sensitivity_0.5"],
        "specificity_0.5": val_m1["specificity_0.5"],
        "f1_0.5": val_m1["f1_0.5"],
        "accuracy_0.5": val_m1["accuracy_0.5"]
    }]
    
    best_val_roc = val_m1["roc_auc"]
    best_epoch = 1
    
    # Save checkpoint 1 as latest and best
    ckpt_best_path = checkpoints_dir / f"checkpoint_best_seed{seed}.pt"
    ckpt_latest_path = checkpoints_dir / f"checkpoint_latest_seed{seed}.pt"
    
    torch.save({
        "epoch": 1,
        "seed": seed,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer_p1.state_dict(),
        "metrics": val_m1,
        "pos_weight": pos_weight_value
    }, ckpt_best_path)
    torch.save({
        "epoch": 1,
        "seed": seed,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer_p1.state_dict(),
        "metrics": val_m1,
        "pos_weight": pos_weight_value
    }, ckpt_latest_path)
    
    # Optimizer and scheduler for remaining epochs
    current_optimizer = optimizer_p1
    current_scheduler = None
    
    for epoch in range(2, total_epochs + 1):
        ep_start = time.time()
        
        if epoch <= 3:
            # Phase 1: Warmup with frozen backbone
            phase = 1
            backbone_frozen = True
            model.set_backbone_trainable(False)
        else:
            # Phase 2: Unfreeze backbone with differential learning rates
            phase = 2
            backbone_frozen = False
            if epoch == 4:
                print(f"\n--- ENTERING PHASE 2 (Fine-tuning full model) ---")
                model.set_backbone_trainable(True)
                current_optimizer = torch.optim.AdamW([
                    {"params": model.backbone.parameters(), "lr": 1e-5, "weight_decay": 1e-4},
                    {"params": list(model.fusion.parameters()) + list(model.head.parameters()), "lr": 1e-4, "weight_decay": 1e-4}
                ])
                t_max = total_epochs - 3
                current_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    current_optimizer, T_max=t_max, eta_min=1e-6
                )
                
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for batch in train_loader:
            cc = batch["X_cc"].to(device)
            mlo = batch["X_mlo"].to(device)
            targets = batch["targets"].to(device)
            
            current_optimizer.zero_grad()
            logits, _ = model(cc, mlo)
            loss = criterion(logits, targets)
            loss.backward()
            current_optimizer.step()
            
            train_loss += loss.item() * targets.size(0)
            train_samples += targets.size(0)
            
        if current_scheduler is not None:
            current_scheduler.step()
            
        ep_train_loss = train_loss / train_samples
        ep_duration = time.time() - ep_start
        
        # Validation pass
        val_metrics, _, _, _, _, _ = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch:02d}/{total_epochs:02d} (P{phase}) [{ep_duration:.1f}s]: "
              f"Train Loss={ep_train_loss:.4f} | Val Loss={val_metrics['val_loss']:.4f} | "
              f"Val ROC-AUC={val_metrics['roc_auc']:.4f} | Val PR-AUC={val_metrics['pr_auc']:.4f}")
        
        epoch_record = {
            "epoch": epoch,
            "phase": phase,
            "backbone_frozen": backbone_frozen,
            "epoch_duration_sec": ep_duration,
            "train_loss": ep_train_loss,
            "val_loss": val_metrics["val_loss"],
            "roc_auc": val_metrics["roc_auc"],
            "pr_auc": val_metrics["pr_auc"],
            "brier_score": val_metrics["brier_score"],
            "sensitivity_0.5": val_metrics["sensitivity_0.5"],
            "specificity_0.5": val_metrics["specificity_0.5"],
            "f1_0.5": val_metrics["f1_0.5"],
            "accuracy_0.5": val_metrics["accuracy_0.5"]
        }
        history.append(epoch_record)
        
        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": current_optimizer.state_dict(),
            "metrics": val_metrics,
            "pos_weight": pos_weight_value
        }, ckpt_latest_path)
        
        # Save best checkpoint
        if val_metrics["roc_auc"] > best_val_roc:
            best_val_roc = val_metrics["roc_auc"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": current_optimizer.state_dict(),
                "metrics": val_metrics,
                "pos_weight": pos_weight_value
            }, ckpt_best_path)
            print(f"  --> [NEW BEST] Checkpoint saved! Best Val ROC-AUC: {best_val_roc:.4f}")
            
    print(f"\nTraining completed. Best Validation ROC-AUC: {best_val_roc:.4f} at Epoch {best_epoch}")
    
    # --- Checkpoint Save/Resume Verification ---
    print(f"\n--- VERIFYING CHECKPOINT SAVE/RESUME ---")
    resumed_model = PairedDualViewModel(pretrained_backbone=False).to(device)
    ckpt = torch.load(ckpt_best_path, map_location=device)
    resumed_model.load_state_dict(ckpt["model_state_dict"])
    resumed_model.eval()
    
    resumed_metrics, targets_val, probs_val, logits_val, pair_ids_val, pat_ids_val = evaluate(
        resumed_model, val_loader, criterion, device
    )
    print(f"Original best ROC-AUC: {best_val_roc:.6f}")
    print(f"Resumed model ROC-AUC: {resumed_metrics['roc_auc']:.6f}")
    assert abs(resumed_metrics["roc_auc"] - best_val_roc) < 1e-5, "Resume verification failed!"
    print("Checkpoint save/resume validation PASSED.")
    
    # Save training history CSV
    suffix = f"_seed{seed}"
    history_path = report_dir / f"training_history{suffix}.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["epoch", "phase", "backbone_frozen", "epoch_duration_sec", "train_loss", "val_loss", "roc_auc", "pr_auc", "brier_score", "sensitivity_0.5", "specificity_0.5", "f1_0.5", "accuracy_0.5"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved training history to {history_path}")
    
    # Save validation predictions CSV
    preds_path = report_dir / f"predictions_validation{suffix}.csv"
    with open(preds_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["pair_id", "patient_id", "target", "predicted_probability", "predicted_logit"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pid, pat, y, p, logit in zip(pair_ids_val, pat_ids_val, targets_val, probs_val, logits_val):
            writer.writerow({
                "pair_id": pid,
                "patient_id": pat,
                "target": int(y),
                "predicted_probability": f"{float(p):.6f}",
                "predicted_logit": f"{float(logit):.6f}"
            })
    print(f"Saved validation predictions to {preds_path}")
    
    # Compute patient-level cluster bootstrap 95% CIs
    print(f"\nComputing patient-level cluster bootstrap 95% CIs (1,000 resamples)...")
    boot_cis = patient_cluster_bootstrap_ci(pat_ids_val, targets_val, probs_val, n_resamples=1000, seed=seed)
    print(f"Bootstrap CIs: {boot_cis}")
    
    # Generate SVG Plots
    print(f"\nGenerating vector SVG plots...")
    # 1. Loss curves
    loss_pts_train = [(h["epoch"], h["train_loss"]) for h in history]
    loss_pts_val = [(h["epoch"], h["val_loss"]) for h in history]
    generate_svg_plot(
        report_dir / f"loss_curves{suffix}.svg",
        f"Stage 4B Pilot: Loss Progression (Arm A: ImageNet | Seed {seed})",
        "Epoch", "Weighted BCE Loss",
        [
            {"name": "Train Loss", "points": loss_pts_train, "color": "#1f77b4", "width": 2.5, "dots": True},
            {"name": "Val Loss", "points": loss_pts_val, "color": "#ff7f0e", "width": 2.5, "dots": True}
        ]
    )
    
    # 2. ROC curve
    roc_pts = list(zip(resumed_metrics["fprs"], resumed_metrics["tprs"]))
    chance_pts = [(0.0, 0.0), (1.0, 1.0)]
    generate_svg_plot(
        report_dir / f"roc_curve{suffix}.svg",
        f"Stage 4B Pilot: Validation ROC Curve (AUC = {resumed_metrics['roc_auc']:.4f} | Seed {seed})",
        "False Positive Rate (1 - Specificity)", "True Positive Rate (Sensitivity)",
        [
            {"name": f"Paired Model (AUC={resumed_metrics['roc_auc']:.4f})", "points": roc_pts, "color": "#2ca02c", "width": 2.5},
            {"name": "Chance (AUC=0.5000)", "points": chance_pts, "color": "#888888", "width": 1.5, "dash": "4,4"}
        ]
    )
    
    # 3. PR curve
    pr_pts = list(zip(resumed_metrics["recalls"], resumed_metrics["precisions"]))
    prevalence = float(np.mean(targets_val))
    baseline_pts = [(0.0, prevalence), (1.0, prevalence)]
    generate_svg_plot(
        report_dir / f"pr_curve{suffix}.svg",
        f"Stage 4B Pilot: Validation PR Curve (AUC = {resumed_metrics['pr_auc']:.4f} | Seed {seed})",
        "Recall (Sensitivity)", "Precision (PPV)",
        [
            {"name": f"Paired Model (PR-AUC={resumed_metrics['pr_auc']:.4f})", "points": pr_pts, "color": "#d62728", "width": 2.5},
            {"name": f"Prevalence Baseline ({prevalence:.2%})", "points": baseline_pts, "color": "#888888", "width": 1.5, "dash": "4,4"}
        ]
    )
    
    # Clean up non-serializable elements from resumed_metrics for JSON
    resumed_metrics_clean = {k: v for k, v in resumed_metrics.items() if not isinstance(v, list)}
    
    # Save evaluation summary JSON
    summary = {
        "pilot_name": "Stage 4B Binary Malignancy Pilot",
        "arm": "Arm A (ImageNet Initialization)",
        "random_seed": seed,
        "backbone": "tf_efficientnet_lite0",
        "pretrained_weights": "ImageNet-1k (offline)",
        "rsna_package_b_status": "PENDING_GATE_06_CLEARANCE (Skipped per instruction)",
        "cohort": {
            "total_retained_pairs": len(all_records),
            "train_pairs": len(train_records),
            "train_patients": len(train_patients),
            "train_benign": n_train_benign,
            "train_malignant": n_train_malignant,
            "train_pos_weight": pos_weight_value,
            "val_pairs": len(val_records),
            "val_patients": len(val_patients),
            "val_benign": n_val_benign,
            "val_malignant": n_val_malignant,
            "patient_overlap_count": len(overlap)
        },
        "training": {
            "total_epochs": total_epochs,
            "epoch_1_duration_sec": ep1_duration,
            "fallback_8_epoch_triggered": bool(ep1_duration > 15 * 60),
            "best_epoch": best_epoch,
            "best_val_roc_auc": float(best_val_roc)
        },
        "best_validation_metrics": resumed_metrics_clean,
        "bootstrap_confidence_intervals_95": boot_cis
    }
    
    summary_path = report_dir / f"evaluation_summary{suffix}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation summary to {summary_path}")
    
    print(f"\n=== STAGE 4B PILOT RUN (SEED {seed}) FINISHED SUCCESSFULLY ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    train_stage4b(seed=args.seed)
