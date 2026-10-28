"""
Grad-CAM Saliency Attribution Module for Mammography Agent.

Governed by: docs/breast_cancer___ps1___methodology.pdf (§3.1, p. 4; §6, p. 6).
Generates dual-view (CC and MLO) gradient-weighted class activation maps (Grad-CAM)
attached to the final convolutional block of tf_efficientnet_lite0 (backbone.conv_head).

IMPORTANT NOTICE:
Grad-CAM heatmaps visualize model feature attribution and gradient sensitivity.
They do NOT represent verified histological lesion boundaries, segmentations, or causal diagnoses.
"""
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


class GradCAM:
    """
    Lightweight PyTorch hook-based Grad-CAM implementation for Paired Dual-View models.
    Targets model.backbone.conv_head to generate separate CC and MLO heatmaps.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.backbone.conv_head

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._forward_hook = self.target_layer.register_forward_hook(self._save_activations)
        self._backward_hook = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module: nn.Module, input: Any, output: torch.Tensor):
        self.activations = output

    def _save_gradients(self, module: nn.Module, grad_input: Any, grad_output: Tuple[torch.Tensor, ...]):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        """Cleanly removes forward and backward hooks."""
        self._forward_hook.remove()
        self._backward_hook.remove()

    def generate_cam(
        self,
        cc: torch.Tensor,
        mlo: torch.Tensor,
        task: str = "binary",
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Computes separate normalized Grad-CAM heatmaps for CC and MLO views.

        Args:
            cc: Tensor [B, 1, H, W]
            mlo: Tensor [B, 1, H, W]
            task: Task to explain: 'binary', 'margin', 'calc', or 'birads'.
            target_class: Class index to backpropagate (if applicable).

        Returns:
            Dict containing:
                'cam_cc': numpy array [B, H, W] in [0.0, 1.0]
                'cam_mlo': numpy array [B, H, W] in [0.0, 1.0]
                'predictions': model output dict
        """
        self.model.eval()
        self.activations = None
        self.gradients = None

        batch_size, _, orig_h, orig_w = cc.shape
        self.model.zero_grad()

        # Forward pass
        preds = self.model(cc, mlo)

        # Select target scalar for backward pass
        if task == "binary":
            score = preds["binary_logit"].sum()
        elif task == "margin":
            logits = preds["margin_logits"]
            idx = target_class if target_class is not None else torch.argmax(logits, dim=-1)
            score = logits.gather(1, idx.view(-1, 1) if isinstance(idx, torch.Tensor) else torch.tensor([[idx]]*batch_size, device=logits.device)).sum()
        elif task == "calc":
            logits = preds["calc_logits"]
            idx = target_class if target_class is not None else torch.argmax(logits, dim=-1)
            score = logits.gather(1, idx.view(-1, 1) if isinstance(idx, torch.Tensor) else torch.tensor([[idx]]*batch_size, device=logits.device)).sum()
        elif task == "birads":
            score = preds["birads_ordinal_logits"].sum()
        else:
            raise ValueError(f"Unknown task: {task}")

        # Backward pass to capture gradients at target layer
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            # Fallback if hooks did not trigger
            empty_cc = np.zeros((batch_size, orig_h, orig_w), dtype=np.float32)
            empty_mlo = np.zeros((batch_size, orig_h, orig_w), dtype=np.float32)
            return {"cam_cc": empty_cc, "cam_mlo": empty_mlo, "predictions": preds}

        # Activations and gradients have shape [2*B, 1280, H', W']
        # Compute channel weights via global average pooling of gradients: [2*B, C, 1, 1]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Weighted combination followed by ReLU
        cam = torch.relu(torch.sum(weights * self.activations, dim=1, keepdim=True))  # [2*B, 1, H', W']

        # Split into CC and MLO views
        cam_cc = cam[:batch_size]   # [B, 1, H', W']
        cam_mlo = cam[batch_size:]  # [B, 1, H', W']

        # Upsample to original input resolution [B, 1, orig_h, orig_w]
        cam_cc_up = F.interpolate(cam_cc, size=(orig_h, orig_w), mode="bilinear", align_corners=False).squeeze(1)
        cam_mlo_up = F.interpolate(cam_mlo, size=(orig_h, orig_w), mode="bilinear", align_corners=False).squeeze(1)

        # Min-max normalize each map to [0.0, 1.0] safely
        cam_cc_np = cam_cc_up.detach().cpu().numpy()
        cam_mlo_np = cam_mlo_up.detach().cpu().numpy()

        cam_cc_norm = np.zeros_like(cam_cc_np)
        cam_mlo_norm = np.zeros_like(cam_mlo_np)

        for b in range(batch_size):
            # CC normalization
            c_min, c_max = cam_cc_np[b].min(), cam_cc_np[b].max()
            if c_max > c_min + 1e-8:
                cam_cc_norm[b] = (cam_cc_np[b] - c_min) / (c_max - c_min)
            else:
                cam_cc_norm[b] = np.zeros_like(cam_cc_np[b])

            # MLO normalization
            m_min, m_max = cam_mlo_np[b].min(), cam_mlo_np[b].max()
            if m_max > m_min + 1e-8:
                cam_mlo_norm[b] = (cam_mlo_np[b] - m_min) / (m_max - m_min)
            else:
                cam_mlo_norm[b] = np.zeros_like(cam_mlo_np[b])

        return {
            "cam_cc": cam_cc_norm,
            "cam_mlo": cam_mlo_norm,
            "predictions": preds
        }


def extract_top_regions(heatmap: np.ndarray, threshold: float = 0.5, max_regions: int = 3) -> List[List[float]]:
    """
    Extracts bounding boxes of high-activation regions from a 2D heatmap [H, W] in [0, 1].
    Returns list of bounding boxes: [[ymin, xmin, ymax, xmax]].
    """
    if heatmap.ndim != 2:
        return []

    binary_mask = (heatmap >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    # Sort contours by area descending
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in sorted_contours[:max_regions]:
        x, y, w, h = cv2.boundingRect(c)
        if w > 4 and h > 4:  # Filter out trivial noise specks
            # Format as [ymin, xmin, ymax, xmax]
            regions.append([float(y), float(x), float(y + h), float(x + w)])

    return regions


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Blends a 2D heatmap on a single-channel grayscale image without mutating the source.
    Args:
        image: 2D array [H, W] in [-1, 1] or [0, 1]
        heatmap: 2D array [H, W] in [0, 1]
        alpha: Blend weight for the heatmap overlay (0.0 = image only, 1.0 = heatmap only)
    Returns:
        RGB image array [H, W, 3] uint8 in [0, 255]
    """
    # Normalize image to uint8 [0, 255]
    img = image.copy()
    if img.min() < 0:
        img = (img + 1.0) / 2.0  # Map [-1, 1] -> [0, 1]
    img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

    # Colorize heatmap
    heat_uint8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, colormap)

    # Blend
    blended = cv2.addWeighted(heat_color, alpha, img_bgr, 1.0 - alpha, 0)
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    return blended_rgb
