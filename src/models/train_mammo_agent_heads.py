"""
Stage: Multi-Task Mammography Agent Secondary Heads Training (Development-Only).

Protocol & Constraints:
- Trains the three newly initialized secondary task heads:
  1. Mass Margin Classification (4 classes)
  2. Calcification Morphology Classification (5 classes)
  3. Ordinal BI-RADS Category (3 cumulative cutoffs)
- Freezes the tf_efficientnet_lite0 backbone, TwoViewAttentionFusion, and BinaryMalignancyHead.
- Reuses the five Stage 5 Best Checkpoints (model_fold{k}_best.pt) as the starting point for each fold.
- For each outer fold k in {0, 1, 2, 3, 4}:
  - Outer test fold: Fold k (held out strictly for final out-of-sample evaluation).
  - Inner validation fold: Fold (k+1)%5 (used for checkpoint selection and early stopping).
  - Inner training folds: Remaining 3 folds (used for parameter updates).
- Patient separation strictly preserved (0 patient overlap across folds).
- Output directory: reports/mammo_agent_training/
- Quarantined local holdout and official test are NEVER accessed.
"""
from typing import Dict, List, Tuple, Any
from pathlib import Path
import sys
import os

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import time
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, accuracy_score,
    mean_absolute_error, cohen_kappa_score, roc_auc_score
)

from src.models.mammo_agent import MammographyAgentModel, MaskedMultiTaskLoss


class MammoMultiTaskDataset(Dataset):
    """
    Dataset for ipsilateral dual-view mammography multi-task learning.
    Reads pre-cached float32 arrays from disk and returns 4-task targets and masks.
    """
    def __init__(self, records: List[Dict[str, Any]], cache_dir: Path, is_training: bool = False, seed: int = 42):
        self.records = records
        self.cache_dir = cache_dir
        self.is_training = is_training
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records[idx]
        pid = row["pair_id"]

        cc_path = self.cache_dir / f"{pid}_CC.npy"
        mlo_path = self.cache_dir / f"{pid}_MLO.npy"

        cc_img = np.load(cc_path).astype(np.float32)
        mlo_img = np.load(mlo_path).astype(np.float32)

        # Coupled horizontal flip for training
        if self.is_training and self.rng.random() < 0.5:
            cc_img = np.fliplr(cc_img).copy()
            mlo_img = np.fliplr(mlo_img).copy()

        # Map [0.0, 1.0] -> [-1.0, 1.0]
        cc_norm = cc_img * 2.0 - 1.0
        mlo_norm = mlo_img * 2.0 - 1.0

        cc_tensor = torch.from_numpy(cc_norm).unsqueeze(0)    # [1, 512, 512]
        mlo_tensor = torch.from_numpy(mlo_norm).unsqueeze(0)  # [1, 512, 512]

        return {
            "X_cc": cc_tensor,
            "X_mlo": mlo_tensor,
            "binary_target": torch.tensor(row["binary_target"], dtype=torch.float32),
            "binary_mask": torch.tensor(row["binary_mask"], dtype=torch.int64),
            "margin_label": torch.tensor(row["margin_label"], dtype=torch.int64),
            "margin_mask": torch.tensor(row["margin_mask"], dtype=torch.int64),
            "calc_label": torch.tensor(row["calc_label"], dtype=torch.int64),
            "calc_mask": torch.tensor(row["calc_mask"], dtype=torch.int64),
            "birads_label": torch.tensor(row["birads_label"], dtype=torch.int64),
            "birads_mask": torch.tensor(row["birads_mask"], dtype=torch.int64),
            "pair_id": pid,
            "patient_id": row["patient_id"]
        }


def collate_multitask_batch(batch: List[Dict[str, Any]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], List[str], List[str]]:
    inputs = {
        "cc": torch.stack([b["X_cc"] for b in batch], dim=0),
        "mlo": torch.stack([b["X_mlo"] for b in batch], dim=0)
    }
    targets = {
        "binary_target": torch.stack([b["binary_target"] for b in batch], dim=0),
        "binary_mask": torch.stack([b["binary_mask"] for b in batch], dim=0),
        "margin_label": torch.stack([b["margin_label"] for b in batch], dim=0),
        "margin_mask": torch.stack([b["margin_mask"] for b in batch], dim=0),
        "calc_label": torch.stack([b["calc_label"] for b in batch], dim=0),
        "calc_mask": torch.stack([b["calc_mask"] for b in batch], dim=0),
        "birads_label": torch.stack([b["birads_label"] for b in batch], dim=0),
        "birads_mask": torch.stack([b["birads_mask"] for b in batch], dim=0),
    }
    pair_ids = [b["pair_id"] for b in batch]
    patient_ids = [b["patient_id"] for b in batch]
    return inputs, targets, pair_ids, patient_ids


def evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    criterion: MaskedMultiTaskLoss,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluates multi-task model on a dataset loader and computes all per-task metrics.
    """
    model.eval()
    total_loss_acc = 0.0
    sec_loss_acc = 0.0
    n_batches = 0

    all_binary_preds = []
    all_binary_targets = []

    all_margin_preds = []
    all_margin_targets = []

    all_calc_preds = []
    all_calc_targets = []

    all_birads_preds = []
    all_birads_targets = []

    with torch.no_grad():
        for inputs, targets, _, _ in loader:
            cc = inputs["cc"].to(device, non_blocking=True)
            mlo = inputs["mlo"].to(device, non_blocking=True)
            targets_gpu = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
                preds = model(cc, mlo)
                loss, breakdown = criterion(preds, targets_gpu)

            total_loss_acc += breakdown["total_loss"]
            sec_loss = breakdown["margin_loss"] + breakdown["calc_loss"] + breakdown["birads_loss"]
            sec_loss_acc += sec_loss
            n_batches += 1

            # 1. Binary Malignancy
            b_probs = torch.sigmoid(preds["binary_logit"]).cpu().numpy()
            b_tgts = targets["binary_target"].cpu().numpy()
            all_binary_preds.extend(b_probs)
            all_binary_targets.extend(b_tgts)

            # 2. Margin (mask == 1 only)
            m_mask = (targets["margin_mask"] == 1).cpu().numpy()
            if m_mask.sum() > 0:
                m_classes = torch.argmax(preds["margin_logits"], dim=-1).cpu().numpy()
                m_tgts = targets["margin_label"].cpu().numpy()
                all_margin_preds.extend(m_classes[m_mask])
                all_margin_targets.extend(m_tgts[m_mask])

            # 3. Calcification (mask == 1 only)
            c_mask = (targets["calc_mask"] == 1).cpu().numpy()
            if c_mask.sum() > 0:
                c_classes = torch.argmax(preds["calc_logits"], dim=-1).cpu().numpy()
                c_tgts = targets["calc_label"].cpu().numpy()
                all_calc_preds.extend(c_classes[c_mask])
                all_calc_targets.extend(c_tgts[c_mask])

            # 4. Ordinal BI-RADS (mask == 1 only)
            bi_mask = (targets["birads_mask"] == 1).cpu().numpy()
            if bi_mask.sum() > 0:
                # Cumulative cutoff probabilities > 0.5 (or logits > 0)
                # Ordinal level y in {0, 1, 2, 3} is sum of cutoffs > 0
                bi_logits = preds["birads_ordinal_logits"].cpu().numpy()
                bi_pred_levels = (bi_logits > 0.0).sum(axis=-1)
                bi_tgts = targets["birads_label"].cpu().numpy()
                all_birads_preds.extend(bi_pred_levels[bi_mask])
                all_birads_targets.extend(bi_tgts[bi_mask])

    avg_total_loss = total_loss_acc / max(1, n_batches)
    avg_sec_loss = sec_loss_acc / max(1, n_batches)

    # Compute Task Metrics
    # Binary
    bin_targets_np = np.array(all_binary_targets)
    bin_preds_np = np.array(all_binary_preds)
    try:
        bin_auc = float(roc_auc_score(bin_targets_np, bin_preds_np))
    except Exception:
        bin_auc = 0.5

    # Margin
    if len(all_margin_targets) > 0:
        m_t = np.array(all_margin_targets)
        m_p = np.array(all_margin_preds)
        margin_f1 = float(f1_score(m_t, m_p, average="macro", zero_division=0))
        margin_bal_acc = float(balanced_accuracy_score(m_t, m_p))
        margin_acc = float(accuracy_score(m_t, m_p))
        margin_count = len(m_t)
    else:
        margin_f1, margin_bal_acc, margin_acc, margin_count = 0.0, 0.0, 0.0, 0

    # Calc
    if len(all_calc_targets) > 0:
        c_t = np.array(all_calc_targets)
        c_p = np.array(all_calc_preds)
        calc_f1 = float(f1_score(c_t, c_p, average="macro", zero_division=0))
        calc_bal_acc = float(balanced_accuracy_score(c_t, c_p))
        calc_acc = float(accuracy_score(c_t, c_p))
        calc_count = len(c_t)
    else:
        calc_f1, calc_bal_acc, calc_acc, calc_count = 0.0, 0.0, 0.0, 0

    # Ordinal BI-RADS
    if len(all_birads_targets) > 0:
        b_t = np.array(all_birads_targets)
        b_p = np.array(all_birads_preds)
        birads_mae = float(mean_absolute_error(b_t, b_p))
        birads_acc = float(accuracy_score(b_t, b_p))
        birads_acc_off1 = float(np.mean(np.abs(b_t - b_p) <= 1))
        try:
            birads_qwk = float(cohen_kappa_score(b_t, b_p, weights="quadratic"))
        except Exception:
            birads_qwk = 0.0
        birads_count = len(b_t)
    else:
        birads_mae, birads_acc, birads_acc_off1, birads_qwk, birads_count = 0.0, 0.0, 0.0, 0.0, 0

    return {
        "val_total_loss": avg_total_loss,
        "val_secondary_loss": avg_sec_loss,
        "binary_auc": bin_auc,
        "margin_macro_f1": margin_f1,
        "margin_bal_acc": margin_bal_acc,
        "margin_acc": margin_acc,
        "margin_valid_count": margin_count,
        "calc_macro_f1": calc_f1,
        "calc_bal_acc": calc_bal_acc,
        "calc_acc": calc_acc,
        "calc_valid_count": calc_count,
        "birads_mae": birads_mae,
        "birads_exact_acc": birads_acc,
        "birads_off1_acc": birads_acc_off1,
        "birads_qwk": birads_qwk,
        "birads_valid_count": birads_count
    }


def train_secondary_heads():
    print("=" * 80)
    print("MAMMO AGENT SECONDARY HEADS TRAINING — 5-FOLD PATIENT-DISJOINT CV")
    print("=" * 80)

    project_root = Path(r"D:\rp")
    targets_csv_path = project_root / "data" / "manifests" / "stage2" / "mammo_multitask_targets.csv"
    cache_dir = project_root / "data" / "cached_preprocessed_512x512"
    stage5_ckpts_dir = project_root / "reports" / "stage5_cross_validation_gpu" / "checkpoints"

    output_dir = project_root / "reports" / "mammo_agent_training"
    ckpt_out_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_out_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "training_execution_log.txt"
    log_fp = open(log_path, "w", encoding="utf-8")

    def log_print(msg: str):
        print(msg, flush=True)
        log_fp.write(msg + "\n")
        log_fp.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    log_print(f"Device: {device} ({gpu_name})")
    log_print(f"Targets Manifest: {targets_csv_path}")

    df_targets = pd.read_csv(targets_csv_path)
    records = df_targets.to_dict(orient="records")
    assert len(records) == 846, f"Expected 846 development pairs, found {len(records)}"

    batch_size = 16
    max_epochs = 20
    patience = 5
    lr = 1e-3
    weight_decay = 1e-4

    all_fold_results = {}

    for fold_k in range(5):
        log_print("\n" + "=" * 80)
        log_print(f"STARTING SECONDARY HEADS TRAINING FOR FOLD {fold_k}")
        log_print(f"  Outer Test Fold:      Fold {fold_k} (HELD OUT FOR UNBIASED EVALUATION)")
        val_fold = (fold_k + 1) % 5
        train_folds = [f for f in range(5) if f not in [fold_k, val_fold]]
        log_print(f"  Inner Validation Fold: Fold {val_fold} (CHECKPOINT SELECTION & EARLY STOPPING)")
        log_print(f"  Inner Training Folds:   Folds {train_folds} (GRADIENT UPDATES)")
        log_print("=" * 80)

        # 1. Partition records
        train_records = [r for r in records if r["outer_fold"] in train_folds]
        val_records = [r for r in records if r["outer_fold"] == val_fold]
        test_records = [r for r in records if r["outer_fold"] == fold_k]

        pats_train = set(r["patient_id"] for r in train_records)
        pats_val = set(r["patient_id"] for r in val_records)
        pats_test = set(r["patient_id"] for r in test_records)

        assert pats_train.isdisjoint(pats_val), "FATAL: Train overlaps with Validation!"
        assert pats_train.isdisjoint(pats_test), "FATAL: Train overlaps with Outer Test!"
        assert pats_val.isdisjoint(pats_test), "FATAL: Validation overlaps with Outer Test!"

        log_print(f"Partition Sample Counts:")
        log_print(f"  Train: {len(train_records)} pairs across {len(pats_train)} patients")
        log_print(f"         (Margins: {sum(r['margin_mask'] for r in train_records)}, Calcs: {sum(r['calc_mask'] for r in train_records)}, BIRADS: {sum(r['birads_mask'] for r in train_records)})")
        log_print(f"  Val:   {len(val_records)} pairs across {len(pats_val)} patients")
        log_print(f"         (Margins: {sum(r['margin_mask'] for r in val_records)}, Calcs: {sum(r['calc_mask'] for r in val_records)}, BIRADS: {sum(r['birads_mask'] for r in val_records)})")
        log_print(f"  Test:  {len(test_records)} pairs across {len(pats_test)} patients")

        # 2. Datasets and Loaders
        fold_seed = 100 + fold_k
        train_ds = MammoMultiTaskDataset(train_records, cache_dir, is_training=True, seed=fold_seed)
        val_ds = MammoMultiTaskDataset(val_records, cache_dir, is_training=False, seed=fold_seed)
        test_ds = MammoMultiTaskDataset(test_records, cache_dir, is_training=False, seed=fold_seed)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_multitask_batch, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_multitask_batch, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_multitask_batch, num_workers=2, pin_memory=True)

        # 3. Model setup and parameter freezing
        model = MammographyAgentModel(pretrained_backbone=False).to(device)
        base_ckpt_path = stage5_ckpts_dir / f"model_fold{fold_k}_best.pt"
        load_summary = model.load_stage5_checkpoint(str(base_ckpt_path), device=device)
        log_print(f"Loaded base checkpoint: {base_ckpt_path.name} ({load_summary['loaded_keys_count']} tensors transferred)")

        # Freeze backbone, fusion, and binary head
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.fusion.parameters():
            param.requires_grad = False
        for param in model.head_binary.parameters():
            param.requires_grad = False

        # Enable gradients exclusively for secondary heads
        trainable_params = []
        for head in [model.head_margin, model.head_calc, model.head_birads]:
            for param in head.parameters():
                param.requires_grad = True
                trainable_params.append(param)

        n_trainable = sum(p.numel() for p in trainable_params)
        n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        log_print(f"Model parameters: {n_trainable:,} trainable (secondary heads) | {n_frozen:,} frozen")

        # 4. Optimizer, Scheduler, Loss
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
        scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

        # In secondary heads training, lambda_binary=0.0 so binary head is untouched
        criterion = MaskedMultiTaskLoss(lambda_binary=0.0, lambda_margin=1.0, lambda_calc=1.0, lambda_birads=1.0)

        best_val_loss = float("inf")
        best_epoch = -1
        patience_counter = 0
        best_val_metrics = {}

        best_ckpt_path = ckpt_out_dir / f"mammo_agent_fold{fold_k}_best.pt"

        # 5. Epoch Loop
        for epoch in range(1, max_epochs + 1):
            epoch_start = time.time()
            model.train()
            # Ensure frozen modules stay in eval mode
            model.backbone.eval()
            model.fusion.eval()
            model.head_binary.eval()

            train_loss_acc = 0.0
            n_train_batches = 0

            for inputs, targets, _, _ in train_loader:
                cc = inputs["cc"].to(device, non_blocking=True)
                mlo = inputs["mlo"].to(device, non_blocking=True)
                targets_gpu = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
                    preds = model(cc, mlo)
                    loss, breakdown = criterion(preds, targets_gpu)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss_acc += breakdown["total_loss"]
                n_train_batches += 1

            scheduler.step()
            train_sec_loss = train_loss_acc / max(1, n_train_batches)
            ep_time = time.time() - epoch_start

            # Evaluate on inner validation fold
            val_m = evaluate_dataset(model, val_loader, criterion, device)

            log_print(
                f"Epoch {epoch:2d}/{max_epochs} ({ep_time:.1f}s) | "
                f"Train Sec Loss: {train_sec_loss:.4f} | "
                f"Val Sec Loss: {val_m['val_secondary_loss']:.4f} | "
                f"Margin F1: {val_m['margin_macro_f1']:.4f} | "
                f"Calc F1: {val_m['calc_macro_f1']:.4f} | "
                f"BIRADS MAE: {val_m['birads_mae']:.4f} (QWK: {val_m['birads_qwk']:.4f})"
            )

            # Checkpoint selection on Inner Validation Loss
            if val_m["val_secondary_loss"] < best_val_loss:
                best_val_loss = val_m["val_secondary_loss"]
                best_epoch = epoch
                best_val_metrics = val_m
                patience_counter = 0

                torch.save({
                    "fold": fold_k,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "val_metrics": best_val_metrics,
                    "trainable_parameters": n_trainable,
                    "frozen_parameters": n_frozen
                }, best_ckpt_path)
                log_print(f"  -> Checkpoint updated: {best_ckpt_path.name} (Val Sec Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    log_print(f"  -> Early stopping triggered after {patience} epochs without improvement (reverting to Epoch {best_epoch}).")
                    break

        log_print(f"\nFold {fold_k} Training Complete. Best Epoch: {best_epoch} (Val Sec Loss: {best_val_loss:.4f})")

        # 6. Unbiased Out-of-Sample Evaluation on Outer Test Fold
        log_print(f"Evaluating Best Model on Outer Test Fold {fold_k} (UNBIASED OUT-OF-SAMPLE)...")
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        test_m = evaluate_dataset(model, test_loader, criterion, device)

        log_print(f"Fold {fold_k} Outer Test Results:")
        log_print(f"  Margin Classification:     Macro-F1={test_m['margin_macro_f1']:.4f} | BalAcc={test_m['margin_bal_acc']:.4f} | Acc={test_m['margin_acc']:.4f} (N={test_m['margin_valid_count']})")
        log_print(f"  Calcification Morphology:  Macro-F1={test_m['calc_macro_f1']:.4f} | BalAcc={test_m['calc_bal_acc']:.4f} | Acc={test_m['calc_acc']:.4f} (N={test_m['calc_valid_count']})")
        log_print(f"  BI-RADS Ordinal Category:  MAE={test_m['birads_mae']:.4f} | Off-1 Acc={test_m['birads_off1_acc']:.4f} | Exact Acc={test_m['birads_exact_acc']:.4f} | QWK={test_m['birads_qwk']:.4f} (N={test_m['birads_valid_count']})")
        log_print(f"  Binary Malignancy Check:   ROC-AUC={test_m['binary_auc']:.4f} (Frozen Head Integrity)")

        all_fold_results[f"fold_{fold_k}"] = {
            "best_epoch": best_epoch,
            "best_val_metrics": best_val_metrics,
            "outer_test_metrics": test_m
        }

    # Save summary json
    summary_path = output_dir / "mammo_agent_cv_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_fold_results, f, indent=2)

    log_print("\n" + "=" * 80)
    log_print(f"MAMMO AGENT SECONDARY HEADS CV TRAINING COMPLETE.")
    log_print(f"Summary metrics saved to: {summary_path}")
    log_print("=" * 80)
    log_fp.close()


if __name__ == "__main__":
    train_secondary_heads()
