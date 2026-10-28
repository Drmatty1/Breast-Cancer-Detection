"""
Mammography Multi-Task Agent Architecture.

Extends the paired dual-view backbone with the four methodology-mandated task heads:
1. Binary Malignancy / Lesion Presence (Scalar Logit, R)
2. Mass Margin Classification (4 Logits, R^4)
3. Calcification Morphology Classification (5 Logits, R^5)
4. Ordinal BI-RADS Assessment (3 Cumulative Cutoff Logits, R^3)

Governed by: docs/breast_cancer___ps1___methodology.pdf (§3.1, pp. 3-4; §8, p. 7).
Preserves existing modules from src/models/paired_model.py without modification.
"""
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from src.models.paired_model import TwoViewAttentionFusion, BinaryMalignancyHead


class MarginHead(nn.Module):
    """
    Multi-class head for mass margin classification over the four methodology classes:
    0: circumscribed, 1: indistinct (ill-defined), 2: obscured, 3: spiculated.
    Maps fused representation [B, 2560] -> [B, 4].
    """
    def __init__(self, in_features: int = 2560, num_classes: int = 4, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CalcificationHead(nn.Module):
    """
    Multi-class head for calcification morphology classification over the five methodology classes:
    0: amorphous, 1: coarse, 2: fine_linear_branching, 3: pleomorphic, 4: punctate.
    Maps fused representation [B, 2560] -> [B, 5].
    """
    def __init__(self, in_features: int = 2560, num_classes: int = 5, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OrdinalBiRadsHead(nn.Module):
    """
    Ordinal head for BI-RADS diagnostic category estimation over levels 2, 3, 4, 5.
    Uses cumulative link / ranking formulation with K-1 = 3 cutoff indicators:
      Cutoff 0: P(Category > 2)
      Cutoff 1: P(Category > 3)
      Cutoff 2: P(Category > 4) (i.e. Category 5)
    Maps fused representation [B, 2560] -> [B, 3].
    """
    def __init__(self, in_features: int = 2560, num_cutoffs: int = 3, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_cutoffs, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MammographyAgentModel(nn.Module):
    """
    Four-Head Mammography Agent Model.
    - Shared Backbone: tf_efficientnet_lite0 (in_chans=1, global_pool='avg') -> R^1280
    - Two-View Attention Fusion: TwoViewAttentionFusion -> R^2560
    - Head 1: BinaryMalignancyHead (Lesion presence / malignancy) -> R^1
    - Head 2: MarginHead (Mass margins: 4 classes) -> R^4
    - Head 3: CalcificationHead (Calc morphology: 5 classes) -> R^5
    - Head 4: OrdinalBiRadsHead (Ordinal BI-RADS: 3 cutoffs) -> R^3
    """
    def __init__(
        self,
        pretrained_backbone: bool = False,
        dropout_backbone: float = 0.3,
        dropout_heads: float = 0.2
    ):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_lite0",
            pretrained=pretrained_backbone,
            in_chans=1,
            num_classes=0,
            global_pool="avg"
        )
        self.fusion = TwoViewAttentionFusion(in_features=1280, att_dim=128)
        self.head_binary = BinaryMalignancyHead(in_features=2560, hidden_dim=256, dropout=dropout_backbone)
        self.head_margin = MarginHead(in_features=2560, num_classes=4, hidden_dim=128, dropout=dropout_heads)
        self.head_calc = CalcificationHead(in_features=2560, num_classes=5, hidden_dim=128, dropout=dropout_heads)
        self.head_birads = OrdinalBiRadsHead(in_features=2560, num_cutoffs=3, hidden_dim=128, dropout=dropout_heads)

    def forward(self, cc: torch.Tensor, mlo: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for ipsilateral paired mammographic views.
        Args:
            cc: Float tensor of shape [B, 1, 512, 512]
            mlo: Float tensor of shape [B, 1, 512, 512]
        Returns:
            Dict containing logits for all four task heads and attention weights.
        """
        batch_size = cc.size(0)
        combined = torch.cat([cc, mlo], dim=0)   # [2*B, 1, H, W]
        features = self.backbone(combined)        # [2*B, 1280]
        f_cc = features[:batch_size]              # [B, 1280]
        f_mlo = features[batch_size:]             # [B, 1280]

        f_fused, alpha = self.fusion(f_cc, f_mlo)  # [B, 2560], [B, 2]

        binary_logit = self.head_binary(f_fused).squeeze(-1)    # [B]
        margin_logits = self.head_margin(f_fused)               # [B, 4]
        calc_logits = self.head_calc(f_fused)                   # [B, 5]
        birads_ordinal_logits = self.head_birads(f_fused)       # [B, 3]

        return {
            "binary_logit": binary_logit,
            "margin_logits": margin_logits,
            "calc_logits": calc_logits,
            "birads_ordinal_logits": birads_ordinal_logits,
            "attention_weights": alpha,
            "fused_features": f_fused
        }

    def load_stage5_checkpoint(
        self,
        checkpoint_path: str,
        load_binary_head: bool = True,
        device: torch.device = torch.device("cpu")
    ) -> Dict[str, Any]:
        """
        Loads backbone, attention fusion, and optionally binary head weights from a Stage 5 CV checkpoint.
        New task heads (margin, calc, birads) remain cleanly initialized.
        """
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)

        loaded_keys = []
        skipped_keys = []

        model_dict = self.state_dict()

        for k, v in state_dict.items():
            # Map backbone and fusion directly
            target_k = None
            if k.startswith("backbone.") or k.startswith("fusion."):
                target_k = k
            elif load_binary_head and k.startswith("head."):
                # Map head.net.* -> head_binary.net.*
                target_k = k.replace("head.", "head_binary.")

            if target_k is not None and target_k in model_dict:
                if model_dict[target_k].shape == v.shape:
                    model_dict[target_k].copy_(v)
                    loaded_keys.append(target_k)
                else:
                    skipped_keys.append((target_k, f"Shape mismatch: {model_dict[target_k].shape} vs {v.shape}"))
            else:
                skipped_keys.append((k, "Not in target model or excluded"))

        initialized_keys = [k for k in model_dict.keys() if k not in loaded_keys]

        return {
            "checkpoint_path": str(checkpoint_path),
            "loaded_keys_count": len(loaded_keys),
            "initialized_keys_count": len(initialized_keys),
            "loaded_keys_sample": loaded_keys[:5],
            "initialized_heads": ["head_margin", "head_calc", "head_birads"],
            "skipped_keys": skipped_keys
        }


class MaskedMultiTaskLoss(nn.Module):
    """
    Masked Multi-Task Loss for the Mammography Agent.
    Computes weighted sum of:
    1. Binary Cross Entropy (Malignancy, weight=lambda_binary)
    2. Masked Cross Entropy (Margin, weight=lambda_margin)
    3. Masked Cross Entropy (Calcification, weight=lambda_calc)
    4. Masked Ordinal BCE Loss (BI-RADS, weight=lambda_birads)

    Gracefully handles batches where a secondary task has zero valid labels (mask.sum() == 0).
    """
    def __init__(
        self,
        pos_weight_binary: Optional[torch.Tensor] = None,
        lambda_binary: float = 1.0,
        lambda_margin: float = 0.5,
        lambda_calc: float = 0.5,
        lambda_birads: float = 0.5
    ):
        super().__init__()
        self.bce_binary = nn.BCEWithLogitsLoss(pos_weight=pos_weight_binary)
        self.ce_margin = nn.CrossEntropyLoss(reduction="none")
        self.ce_calc = nn.CrossEntropyLoss(reduction="none")
        self.bce_ordinal = nn.BCEWithLogitsLoss(reduction="none")

        self.lambda_binary = lambda_binary
        self.lambda_margin = lambda_margin
        self.lambda_calc = lambda_calc
        self.lambda_birads = lambda_birads

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = preds["binary_logit"].device

        # 1. Binary Malignancy Loss
        bin_mask = targets["binary_mask"].float()
        bin_loss = self.bce_binary(preds["binary_logit"], targets["binary_target"].float())

        # 2. Masked Margin Loss
        margin_mask = targets["margin_mask"].float()
        n_valid_margin = margin_mask.sum()
        if n_valid_margin > 0:
            # Mask out invalid target values (-1) before loss calculation to prevent index errors
            safe_margin_targets = torch.where(targets["margin_mask"] == 1, targets["margin_label"], torch.zeros_like(targets["margin_label"]))
            unreduced_margin_loss = self.ce_margin(preds["margin_logits"], safe_margin_targets)
            margin_loss = (unreduced_margin_loss * margin_mask).sum() / n_valid_margin
        else:
            margin_loss = torch.tensor(0.0, device=device)

        # 3. Masked Calcification Loss
        calc_mask = targets["calc_mask"].float()
        n_valid_calc = calc_mask.sum()
        if n_valid_calc > 0:
            safe_calc_targets = torch.where(targets["calc_mask"] == 1, targets["calc_label"], torch.zeros_like(targets["calc_label"]))
            unreduced_calc_loss = self.ce_calc(preds["calc_logits"], safe_calc_targets)
            calc_loss = (unreduced_calc_loss * calc_mask).sum() / n_valid_calc
        else:
            calc_loss = torch.tensor(0.0, device=device)

        # 4. Masked Ordinal BI-RADS Loss
        birads_mask = targets["birads_mask"].float()
        n_valid_birads = birads_mask.sum()
        if n_valid_birads > 0:
            # Ordinal cumulative targets: label y in {0, 1, 2, 3} -> indicators I(y > k) for k in {0, 1, 2}
            safe_birads_targets = torch.where(targets["birads_mask"] == 1, targets["birads_label"], torch.zeros_like(targets["birads_label"]))
            # Shape [B, 3]: col 0 is I(y > 0), col 1 is I(y > 1), col 2 is I(y > 2)
            cutoffs = torch.tensor([0, 1, 2], device=device).unsqueeze(0)  # [1, 3]
            ordinal_target_matrix = (safe_birads_targets.unsqueeze(1) > cutoffs).float()  # [B, 3]
            unreduced_ordinal_loss = self.bce_ordinal(preds["birads_ordinal_logits"], ordinal_target_matrix).mean(dim=1)  # [B]
            birads_loss = (unreduced_ordinal_loss * birads_mask).sum() / n_valid_birads
        else:
            birads_loss = torch.tensor(0.0, device=device)

        total_loss = (
            self.lambda_binary * bin_loss +
            self.lambda_margin * margin_loss +
            self.lambda_calc * calc_loss +
            self.lambda_birads * birads_loss
        )

        loss_breakdown = {
            "total_loss": float(total_loss.item()),
            "binary_loss": float(bin_loss.item()),
            "margin_loss": float(margin_loss.item()),
            "calc_loss": float(calc_loss.item()),
            "birads_loss": float(birads_loss.item()),
            "n_valid_margin": int(n_valid_margin.item()),
            "n_valid_calc": int(n_valid_calc.item()),
            "n_valid_birads": int(n_valid_birads.item())
        }

        return total_loss, loss_breakdown
