"""
Paired Dual-View Mammography Model Architecture for Stage 4B.

Shared EfficientNet-Lite0 backbone + Two-View Attention Fusion + Decoupled Binary Head.
"""
import torch
import torch.nn as nn
import timm


class TwoViewAttentionFusion(nn.Module):
    """
    Learned two-view attention fusion mechanism for ipsilateral CC and MLO mammographic views.
    Projects feature vectors to a hidden salience space (d_att=128), computes softmax attention,
    and returns concatenated attention-weighted features in R^2560.
    """
    def __init__(self, in_features: int = 1280, att_dim: int = 128):
        super().__init__()
        self.fc_att = nn.Linear(in_features, att_dim, bias=True)
        self.v_att = nn.Linear(att_dim, 1, bias=False)

    def forward(self, f_cc: torch.Tensor, f_mlo: torch.Tensor):
        # f_cc, f_mlo: [B, D]
        # Stack into [B, 2, D]
        f_stack = torch.stack([f_cc, f_mlo], dim=1)
        # Hidden salience representation: [B, 2, att_dim]
        h = torch.tanh(self.fc_att(f_stack))
        # Salience scores: [B, 2, 1]
        scores = self.v_att(h)
        # Attention weights: [B, 2, 1]
        alpha = torch.softmax(scores, dim=1)
        
        alpha_cc = alpha[:, 0, :]   # [B, 1]
        alpha_mlo = alpha[:, 1, :]  # [B, 1]
        
        f_cc_weighted = alpha_cc * f_cc
        f_mlo_weighted = alpha_mlo * f_mlo
        
        f_fused = torch.cat([f_cc_weighted, f_mlo_weighted], dim=1)  # [B, 2*D]
        return f_fused, alpha.squeeze(-1)  # f_fused: [B, 2560], alpha: [B, 2]


class BinaryMalignancyHead(nn.Module):
    """
    Decoupled binary malignancy classification head.
    Maps fused representation [B, 2560] -> scalar logit [B, 1].
    """
    def __init__(self, in_features: int = 2560, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PairedDualViewModel(nn.Module):
    """
    Full Paired Dual-View Mammography Network.
    - Shared Backbone: tf_efficientnet_lite0 (in_chans=1, num_classes=0, global_pool='avg')
    - Two-View Attention Fusion: 1280 -> 2560
    - Classification Head: 2560 -> 1
    """
    def __init__(self, pretrained_backbone: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_lite0",
            pretrained=pretrained_backbone,
            in_chans=1,
            num_classes=0,
            global_pool="avg"
        )
        self.fusion = TwoViewAttentionFusion(in_features=1280, att_dim=128)
        self.head = BinaryMalignancyHead(in_features=2560, hidden_dim=256, dropout=dropout)

    def forward(self, cc: torch.Tensor, mlo: torch.Tensor):
        # cc, mlo: [B, 1, H, W]
        batch_size = cc.size(0)
        # Concatenate along batch dimension for a single shared backbone forward pass
        combined = torch.cat([cc, mlo], dim=0)  # [2*B, 1, H, W]
        features = self.backbone(combined)       # [2*B, 1280]
        f_cc = features[:batch_size]             # [B, 1280]
        f_mlo = features[batch_size:]            # [B, 1280]
        
        f_fused, alpha = self.fusion(f_cc, f_mlo)  # [B, 2560], [B, 2]
        logits = self.head(f_fused).squeeze(-1)    # [B]
        return logits, alpha

    def set_backbone_trainable(self, trainable: bool):
        """Freeze or unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = trainable
