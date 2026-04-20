from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


IGNORE_LABEL = 255


def update_confusion_matrix(
    confmat: torch.Tensor,
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
    num_classes: int,
    ignore_label: int = IGNORE_LABEL,
) -> torch.Tensor:
    """
    Update confusion matrix using one batch of predictions and targets.

    pred_mask: [B,H,W] or [H,W]
    true_mask: [B,H,W] or [H,W]
    """
    if pred_mask.ndim == 2:
        pred_mask = pred_mask.unsqueeze(0)
    if true_mask.ndim == 2:
        true_mask = true_mask.unsqueeze(0)

    if pred_mask.shape != true_mask.shape:
        raise ValueError(f"Shape mismatch: pred {tuple(pred_mask.shape)} vs true {tuple(true_mask.shape)}")

    valid = (true_mask != ignore_label)
    if valid.sum().item() == 0:
        return confmat

    pred = pred_mask[valid].view(-1).to(torch.int64)
    true = true_mask[valid].view(-1).to(torch.int64)

    idx = true * num_classes + pred
    binc = torch.bincount(idx, minlength=num_classes * num_classes)
    confmat += binc.reshape(num_classes, num_classes).to(confmat.device)
    return confmat


@dataclass
class SegmentationMetricResult:
    pixel_accuracy: float
    mean_iou: float
    mean_class_accuracy: float
    per_class_iou: np.ndarray
    per_class_accuracy: np.ndarray
    confusion_matrix: np.ndarray


def compute_segmentation_metrics(confmat: torch.Tensor) -> SegmentationMetricResult:
    """
    Compute pixel accuracy, mean IoU, and per-class scores from confusion matrix.
    """
    cm = confmat.detach().cpu().numpy().astype(np.float64)

    true_pos = np.diag(cm)
    gt_pixels = cm.sum(axis=1)
    pred_pixels = cm.sum(axis=0)
    union = gt_pixels + pred_pixels - true_pos

    pixel_accuracy = true_pos.sum() / max(cm.sum(), 1.0)

    per_class_iou = np.full((cm.shape[0],), np.nan, dtype=np.float64)
    valid_union = union > 0
    per_class_iou[valid_union] = true_pos[valid_union] / union[valid_union]

    per_class_accuracy = np.full((cm.shape[0],), np.nan, dtype=np.float64)
    valid_gt = gt_pixels > 0
    per_class_accuracy[valid_gt] = true_pos[valid_gt] / gt_pixels[valid_gt]

    mean_iou = float(np.nanmean(per_class_iou)) if np.any(valid_union) else 0.0
    mean_class_accuracy = float(np.nanmean(per_class_accuracy)) if np.any(valid_gt) else 0.0

    return SegmentationMetricResult(
        pixel_accuracy=float(pixel_accuracy),
        mean_iou=float(mean_iou),
        mean_class_accuracy=float(mean_class_accuracy),
        per_class_iou=per_class_iou,
        per_class_accuracy=per_class_accuracy,
        confusion_matrix=cm,
    )


class StreamingSegmentationMetrics:
    def __init__(self, num_classes: int, ignore_label: int = IGNORE_LABEL, device: str = "cpu"):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    def update_from_logits(self, image_logits: torch.Tensor, true_mask: torch.Tensor):
        pred_mask = torch.argmax(image_logits, dim=1)
        update_confusion_matrix(
            confmat=self.confmat,
            pred_mask=pred_mask,
            true_mask=true_mask,
            num_classes=self.num_classes,
            ignore_label=self.ignore_label,
        )

    def compute(self) -> SegmentationMetricResult:
        return compute_segmentation_metrics(self.confmat)