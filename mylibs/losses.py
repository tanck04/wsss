from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


UNLABELED_SEED = -1
IGNORE_LABEL = 255


def seed_cross_entropy_loss(
    image_logits: torch.Tensor,
    seed_labels: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    unlabeled_value: int = UNLABELED_SEED,
) -> torch.Tensor:
    """
    Cross-entropy loss computed only on labeled seed pixels.

    image_logits: [B, K, H, W]
    seed_labels:  [B, H, W], with unlabeled pixels marked as -1
    """
    if image_logits.ndim != 4:
        raise ValueError(f"Expected image_logits [B,K,H,W], got {tuple(image_logits.shape)}")
    if seed_labels.ndim != 3:
        raise ValueError(f"Expected seed_labels [B,H,W], got {tuple(seed_labels.shape)}")

    valid = (seed_labels != unlabeled_value)
    if valid.sum().item() == 0:
        return image_logits.sum() * 0.0

    logits_valid = image_logits.permute(0, 2, 3, 1)[valid]  # [N, K]
    labels_valid = seed_labels[valid]                        # [N]

    return F.cross_entropy(
        logits_valid,
        labels_valid,
        weight=class_weights,
    )


def full_mask_cross_entropy_loss(
    image_logits: torch.Tensor,
    full_mask: torch.Tensor,
    ignore_label: int = IGNORE_LABEL,
) -> torch.Tensor:
    """
    Standard cross-entropy on a full segmentation mask, used only for evaluation/monitoring.
    """
    if image_logits.ndim != 4:
        raise ValueError(f"Expected image_logits [B,K,H,W], got {tuple(image_logits.shape)}")
    if full_mask.ndim != 3:
        raise ValueError(f"Expected full_mask [B,H,W], got {tuple(full_mask.shape)}")

    valid = (full_mask != ignore_label)
    if valid.sum().item() == 0:
        return image_logits.sum() * 0.0

    return F.cross_entropy(
        image_logits,
        full_mask,
        ignore_index=ignore_label,
    )


def pooled_fg_class_logits(
    image_logits: torch.Tensor,
    pooling: str = "logsumexp",
) -> torch.Tensor:
    """
    Convert dense logits [B,K,H,W] into image-level foreground class logits [B,K-1].

    Background class 0 is excluded from the pooled foreground scores.
    """
    if image_logits.ndim != 4:
        raise ValueError(f"Expected image_logits [B,K,H,W], got {tuple(image_logits.shape)}")

    B, K, H, W = image_logits.shape
    if K < 2:
        raise ValueError("Need at least background + one foreground class.")

    fg_logits = image_logits[:, 1:, :, :].reshape(B, K - 1, H * W)

    if pooling == "logsumexp":
        pooled = torch.logsumexp(fg_logits, dim=-1) - math.log(H * W)
    elif pooling == "max":
        pooled = fg_logits.max(dim=-1).values
    elif pooling == "mean":
        pooled = fg_logits.mean(dim=-1)
    else:
        raise ValueError(f"Unknown pooling mode: {pooling}")

    return pooled


def image_level_tag_bce_loss(
    image_logits: torch.Tensor,
    tag_targets_fg: torch.Tensor,
    pooling: str = "logsumexp",
) -> torch.Tensor:
    """
    Multi-label BCE loss for foreground class tags.

    image_logits:   [B, K, H, W]
    tag_targets_fg: [B, K-1] with entries in {0,1}
    """
    pooled_logits = pooled_fg_class_logits(
        image_logits=image_logits,
        pooling=pooling,
    )

    if pooled_logits.shape != tag_targets_fg.shape:
        raise ValueError(
            f"Shape mismatch: pooled_logits {tuple(pooled_logits.shape)} vs "
            f"tag_targets_fg {tuple(tag_targets_fg.shape)}"
        )

    return F.binary_cross_entropy_with_logits(
        pooled_logits,
        tag_targets_fg,
    )


def compute_seed_accuracy(
    image_logits: torch.Tensor,
    seed_labels: torch.Tensor,
    unlabeled_value: int = UNLABELED_SEED,
) -> Tuple[int, int]:
    """
    Returns:
        correct_seed_pixels, total_labeled_seed_pixels
    """
    preds = torch.argmax(image_logits, dim=1)
    valid = (seed_labels != unlabeled_value)

    total = int(valid.sum().item())
    if total == 0:
        return 0, 0

    correct = int(((preds == seed_labels) & valid).sum().item())
    return correct, total


def combined_wsss_loss(
    image_logits: torch.Tensor,
    seed_labels: torch.Tensor,
    tag_targets_fg: Optional[torch.Tensor] = None,
    lambda_tag: float = 0.0,
    class_weights: Optional[torch.Tensor] = None,
    tag_pooling: str = "logsumexp",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Combined weak-supervision loss:
      total = seed_ce + lambda_tag * tag_bce
    """
    seed_ce = seed_cross_entropy_loss(
        image_logits=image_logits,
        seed_labels=seed_labels,
        class_weights=class_weights,
    )

    total = seed_ce
    tag_bce = image_logits.sum() * 0.0

    if lambda_tag > 0.0 and tag_targets_fg is not None:
        tag_bce = image_level_tag_bce_loss(
            image_logits=image_logits,
            tag_targets_fg=tag_targets_fg,
            pooling=tag_pooling,
        )
        total = total + lambda_tag * tag_bce

    loss_dict = {
        "total_loss": total.detach(),
        "seed_ce": seed_ce.detach(),
        "tag_bce": tag_bce.detach(),
    }
    return total, loss_dict

def feature_aware_patch_smoothness_loss(
    patch_logits: torch.Tensor,
    patch_features: torch.Tensor,
    sigma: float = 0.5,
    detach_features: bool = True,
) -> torch.Tensor:
    """
    Encourage nearby patch predictions to be similar, especially when the
    frozen DINO features are similar.

    patch_logits:   [B, K, H_p, W_p]
    patch_features: [B, C, H_p, W_p]

    Loss:
      weighted squared difference between neighboring class probabilities,
      where the weight is larger if neighboring features are similar.
    """
    if patch_logits.ndim != 4:
        raise ValueError(f"Expected patch_logits [B,K,H,W], got {tuple(patch_logits.shape)}")
    if patch_features.ndim != 4:
        raise ValueError(f"Expected patch_features [B,C,H,W], got {tuple(patch_features.shape)}")

    feats = patch_features.detach() if detach_features else patch_features
    probs = torch.softmax(patch_logits, dim=1)

    # horizontal neighbors
    prob_diff_h = (probs[:, :, :, 1:] - probs[:, :, :, :-1]).pow(2).sum(dim=1)   # [B,H,W-1]
    feat_diff_h = (feats[:, :, :, 1:] - feats[:, :, :, :-1]).pow(2).mean(dim=1)  # [B,H,W-1]
    weight_h = torch.exp(-feat_diff_h / max(sigma, 1e-6))

    # vertical neighbors
    prob_diff_v = (probs[:, :, 1:, :] - probs[:, :, :-1, :]).pow(2).sum(dim=1)   # [B,H-1,W]
    feat_diff_v = (feats[:, :, 1:, :] - feats[:, :, :-1, :]).pow(2).mean(dim=1)  # [B,H-1,W]
    weight_v = torch.exp(-feat_diff_v / max(sigma, 1e-6))

    loss_h = (weight_h * prob_diff_h).mean()
    loss_v = (weight_v * prob_diff_v).mean()

    return 0.5 * (loss_h + loss_v)


def combined_wsss_loss_with_smoothness(
    image_logits: torch.Tensor,
    seed_labels: torch.Tensor,
    patch_logits: torch.Tensor,
    patch_features: torch.Tensor,
    tag_targets_fg: Optional[torch.Tensor] = None,
    lambda_tag: float = 0.0,
    lambda_smooth: float = 0.0,
    class_weights: Optional[torch.Tensor] = None,
    tag_pooling: str = "logsumexp",
    smoothness_sigma: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    total = seed_ce + lambda_tag * tag_bce + lambda_smooth * smooth_loss
    """
    seed_ce = seed_cross_entropy_loss(
        image_logits=image_logits,
        seed_labels=seed_labels,
        class_weights=class_weights,
    )

    tag_bce = image_logits.sum() * 0.0
    if lambda_tag > 0.0 and tag_targets_fg is not None:
        tag_bce = image_level_tag_bce_loss(
            image_logits=image_logits,
            tag_targets_fg=tag_targets_fg,
            pooling=tag_pooling,
        )

    smooth_loss = image_logits.sum() * 0.0
    if lambda_smooth > 0.0:
        smooth_loss = feature_aware_patch_smoothness_loss(
            patch_logits=patch_logits,
            patch_features=patch_features,
            sigma=smoothness_sigma,
            detach_features=True,
        )

    total = seed_ce + lambda_tag * tag_bce + lambda_smooth * smooth_loss

    loss_dict = {
        "total_loss": total.detach(),
        "seed_ce": seed_ce.detach(),
        "tag_bce": tag_bce.detach(),
        "smooth_loss": smooth_loss.detach(),
    }
    return total, loss_dict