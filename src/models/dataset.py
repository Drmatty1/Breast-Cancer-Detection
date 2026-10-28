"""
Paired Mammography Dataset for Stage 4B.
Loads cached float32 arrays in [0.0, 1.0], maps to [-1.0, 1.0], applies coupled augmentation.
"""
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class PairedMammogramDataset(Dataset):
    """
    Dataset for ipsilateral dual-view mammography (CC and MLO).
    Reads pre-cached float32 arrays from disk.
    """
    def __init__(self, records: list[dict], is_training: bool = False, seed: int = 42):
        self.records = records
        self.is_training = is_training
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        
        # Resolve path
        cache_dir = Path(r"D:\rp\data\cached_preprocessed_512x512")
        cc_path = Path(row["cc_cache_file"])
        if not cc_path.is_absolute() or not cc_path.exists():
            cc_path = cache_dir / cc_path.name
            
        mlo_path = Path(row["mlo_cache_file"])
        if not mlo_path.is_absolute() or not mlo_path.exists():
            mlo_path = cache_dir / mlo_path.name
        
        cc_img = np.load(cc_path).astype(np.float32)
        mlo_img = np.load(mlo_path).astype(np.float32)
        
        # Coupled clinical horizontal flip: if flipped, both ipsilateral views flip together
        if self.is_training and self.rng.random() < 0.5:
            cc_img = np.fliplr(cc_img).copy()
            mlo_img = np.fliplr(mlo_img).copy()
            
        # Map [0.0, 1.0] -> [-1.0, 1.0]
        cc_norm = cc_img * 2.0 - 1.0
        mlo_norm = mlo_img * 2.0 - 1.0
        
        # Add channel dimension [1, 512, 512]
        cc_tensor = torch.from_numpy(cc_norm).unsqueeze(0)
        mlo_tensor = torch.from_numpy(mlo_norm).unsqueeze(0)
        
        target = float(row["binary_target"])
        target_tensor = torch.tensor(target, dtype=torch.float32)
        
        return {
            "X_cc": cc_tensor,
            "X_mlo": mlo_tensor,
            "target": target_tensor,
            "pair_id": row["pair_id"],
            "patient_id": row["patient_id"],
            "breast_side": row.get("breast_side", "")
        }


def collate_paired_batch(batch: list[dict]) -> dict:
    """Collate function for PairedMammogramDataset."""
    X_cc = torch.stack([b["X_cc"] for b in batch], dim=0)
    X_mlo = torch.stack([b["X_mlo"] for b in batch], dim=0)
    targets = torch.stack([b["target"] for b in batch], dim=0)
    pair_ids = [b["pair_id"] for b in batch]
    patient_ids = [b["patient_id"] for b in batch]
    sides = [b["breast_side"] for b in batch]
    
    return {
        "X_cc": X_cc,
        "X_mlo": X_mlo,
        "targets": targets,
        "pair_ids": pair_ids,
        "patient_ids": patient_ids,
        "breast_sides": sides
    }
