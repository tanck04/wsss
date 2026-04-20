from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from mylibs.dinov2_features import extract_patch_features, IMAGENET_MEAN, IMAGENET_STD
from mylibs.weak_labels import decode_voc_mask, render_seed_overlay, overlay_mask_on_image


IGNORE_LABEL = 255
NUM_CLASSES = 21


def denormalize_image_tensor(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert normalized tensor [3,H,W] back to uint8 RGB image [H,W,3].
    """
    if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
        raise ValueError(f"Expected image_tensor [3,H,W], got {tuple(image_tensor.shape)}")

    x = image_tensor.detach().cpu().float().clone()
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype).view(3, 1, 1)

    x = x * std + mean
    x = torch.clamp(x, 0.0, 1.0)
    x = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return x


def tensor_mask_to_numpy(mask_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert mask tensor [H,W] to numpy int64 [H,W].
    """
    if mask_tensor.ndim != 2:
        raise ValueError(f"Expected mask tensor [H,W], got {tuple(mask_tensor.shape)}")
    return mask_tensor.detach().cpu().numpy().astype(np.int64)


@torch.no_grad()
def predict_single_sample(
    backbone: torch.nn.Module,
    model: torch.nn.Module,
    sample: Dict,
    device: str,
    patch_size: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Run inference on one dataset sample.

    Returns dict containing:
    - pred_mask: [H,W]
    - gt_mask: [H,W]
    - seed_labels: [H,W]
    - seed_mask: [H,W]
    - image_rgb: [H,W,3]
    - image_logits: [K,H,W]
    """
    model.eval()
    backbone.eval()

    image_tensor = sample["image"].unsqueeze(0).to(device, non_blocking=True)  # [1,3,H,W]
    gt_mask = tensor_mask_to_numpy(sample["mask"])
    seed_labels = tensor_mask_to_numpy(sample["seed_labels"])
    seed_mask = sample["seed_mask"].detach().cpu().numpy().astype(bool)
    image_rgb = denormalize_image_tensor(sample["image"])

    H, W = gt_mask.shape

    feat_dict = extract_patch_features(
        model=backbone,
        image_tensor=image_tensor,
        patch_size=patch_size,
    )
    patch_map = feat_dict["patch_map"]

    out = model(
        patch_map=patch_map,
        output_size=(H, W),
    )
    image_logits = out["image_logits"][0].detach().cpu()  # [K,H,W]
    pred_mask = torch.argmax(image_logits, dim=0).numpy().astype(np.int64)

    return {
        "pred_mask": pred_mask,
        "gt_mask": gt_mask,
        "seed_labels": seed_labels,
        "seed_mask": seed_mask,
        "image_rgb": image_rgb,
        "image_logits": image_logits.numpy(),
    }


def compute_sample_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    num_classes: int = NUM_CLASSES,
    ignore_label: int = IGNORE_LABEL,
) -> Dict[str, float]:
    """
    Compute simple per-sample metrics:
    - pixel accuracy
    - sample mIoU across classes that appear in GT or prediction
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Shape mismatch: pred {pred_mask.shape} vs gt {gt_mask.shape}")

    valid = (gt_mask != ignore_label)
    if valid.sum() == 0:
        return {"pixel_acc": 0.0, "sample_mIoU": 0.0}

    pred = pred_mask[valid]
    gt = gt_mask[valid]

    pixel_acc = float((pred == gt).mean())

    ious = []
    for cls_id in range(num_classes):
        pred_c = (pred == cls_id)
        gt_c = (gt == cls_id)
        union = np.logical_or(pred_c, gt_c).sum()
        if union == 0:
            continue
        inter = np.logical_and(pred_c, gt_c).sum()
        ious.append(inter / union)

    sample_miou = float(np.mean(ious)) if len(ious) > 0 else 0.0
    return {
        "pixel_acc": pixel_acc,
        "sample_mIoU": sample_miou,
    }


def make_error_map(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    ignore_label: int = IGNORE_LABEL,
) -> np.ndarray:
    """
    Create RGB error map:
    - black   = correct
    - red     = incorrect
    - white   = ignore/void
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Shape mismatch: pred {pred_mask.shape} vs gt {gt_mask.shape}")

    H, W = gt_mask.shape
    err = np.zeros((H, W, 3), dtype=np.uint8)

    ignore = (gt_mask == ignore_label)
    wrong = (pred_mask != gt_mask) & (~ignore)
    correct = (pred_mask == gt_mask) & (~ignore)

    err[correct] = np.array([0, 0, 0], dtype=np.uint8)
    err[wrong] = np.array([220, 20, 60], dtype=np.uint8)
    err[ignore] = np.array([255, 255, 255], dtype=np.uint8)
    return err


def visualize_single_prediction(
    image_rgb: np.ndarray,
    seed_labels: np.ndarray,
    seed_mask: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    title: str,
    save_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Create a 1x5 qualitative figure:
    - input image
    - weak labels
    - predicted overlay
    - GT overlay
    - error map
    """
    weak_overlay = render_seed_overlay(
        image=image_rgb,
        seed_labels=seed_labels,
        seed_mask=seed_mask,
        point_radius=3,
    )

    pred_rgb = decode_voc_mask(pred_mask.astype(np.uint8))
    gt_rgb = decode_voc_mask(gt_mask.astype(np.uint8))
    pred_overlay = overlay_mask_on_image(image_rgb, pred_rgb, alpha=0.45)
    gt_overlay = overlay_mask_on_image(image_rgb, gt_rgb, alpha=0.45)
    error_rgb = make_error_map(pred_mask, gt_mask)

    metrics = compute_sample_metrics(pred_mask, gt_mask)
    pred_title = (
        f"Prediction\n"
        f"pix acc={metrics['pixel_acc']:.3f}, "
        f"sample mIoU={metrics['sample_mIoU']:.3f}"
    )

    fig, axes = plt.subplots(1, 5, figsize=(24, 5))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    axes[1].imshow(weak_overlay)
    axes[1].set_title("Weak labels (seeds)")
    axes[1].axis("off")

    axes[2].imshow(pred_overlay)
    axes[2].set_title(pred_title)
    axes[2].axis("off")

    axes[3].imshow(gt_overlay)
    axes[3].set_title("Ground-truth mask")
    axes[3].axis("off")

    axes[4].imshow(error_rgb)
    axes[4].set_title("Error map")
    axes[4].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_model_on_dataset_indices(
    backbone: torch.nn.Module,
    model: torch.nn.Module,
    dataset,
    indices: Iterable[int],
    device: str,
    patch_size: int = 14,
    save_dir: Optional[Path] = None,
    model_name: str = "model",
    show: bool = True,
):
    """
    Visualize one model on several dataset examples.
    """
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        sample = dataset[idx]
        pred = predict_single_sample(
            backbone=backbone,
            model=model,
            sample=sample,
            device=device,
            patch_size=patch_size,
        )

        image_id = sample.get("image_id", f"sample_{idx:05d}")
        title = f"{model_name} — {image_id}"

        save_path = None
        if save_dir is not None:
            save_path = save_dir / f"{model_name}_{image_id}.png"

        visualize_single_prediction(
            image_rgb=pred["image_rgb"],
            seed_labels=pred["seed_labels"],
            seed_mask=pred["seed_mask"],
            pred_mask=pred["pred_mask"],
            gt_mask=pred["gt_mask"],
            title=title,
            save_path=save_path,
            show=show,
        )


def compare_models_on_single_sample(
    backbone: torch.nn.Module,
    models: Dict[str, torch.nn.Module],
    dataset,
    idx: int,
    device: str,
    patch_size: int = 14,
    save_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Compare multiple trained models on the same sample.

    Layout:
    Row 1: input, weak labels, GT
    Row 2+: model predictions
    """
    sample = dataset[idx]
    image_id = sample.get("image_id", f"sample_{idx:05d}")

    base = {}
    first_model_name = list(models.keys())[0]
    first_pred = predict_single_sample(
        backbone=backbone,
        model=models[first_model_name],
        sample=sample,
        device=device,
        patch_size=patch_size,
    )

    image_rgb = first_pred["image_rgb"]
    gt_mask = first_pred["gt_mask"]
    seed_labels = first_pred["seed_labels"]
    seed_mask = first_pred["seed_mask"]

    weak_overlay = render_seed_overlay(
        image=image_rgb,
        seed_labels=seed_labels,
        seed_mask=seed_mask,
        point_radius=3,
    )
    gt_rgb = decode_voc_mask(gt_mask.astype(np.uint8))
    gt_overlay = overlay_mask_on_image(image_rgb, gt_rgb, alpha=0.45)

    n_models = len(models)
    fig, axes = plt.subplots(n_models + 1, 3, figsize=(14, 4 * (n_models + 1)))

    if n_models == 1:
        axes = np.expand_dims(axes, axis=0)

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title(f"Input image\n{image_id}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(weak_overlay)
    axes[0, 1].set_title("Weak labels (seeds)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(gt_overlay)
    axes[0, 2].set_title("Ground-truth mask")
    axes[0, 2].axis("off")

    for row_idx, (model_name, model) in enumerate(models.items(), start=1):
        pred = predict_single_sample(
            backbone=backbone,
            model=model,
            sample=sample,
            device=device,
            patch_size=patch_size,
        )

        pred_mask = pred["pred_mask"]
        pred_rgb = decode_voc_mask(pred_mask.astype(np.uint8))
        pred_overlay = overlay_mask_on_image(image_rgb, pred_rgb, alpha=0.45)
        error_rgb = make_error_map(pred_mask, gt_mask)
        metrics = compute_sample_metrics(pred_mask, gt_mask)

        axes[row_idx, 0].imshow(pred_overlay)
        axes[row_idx, 0].set_title(
            f"{model_name}\n"
            f"pix acc={metrics['pixel_acc']:.3f}, "
            f"sample mIoU={metrics['sample_mIoU']:.3f}"
        )
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(error_rgb)
        axes[row_idx, 1].set_title("Error map")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].axis("off")
        axes[row_idx, 2].text(
            0.02,
            0.98,
            f"Model: {model_name}\n"
            f"Image ID: {image_id}\n"
            f"Pixel accuracy: {metrics['pixel_acc']:.4f}\n"
            f"Sample mIoU: {metrics['sample_mIoU']:.4f}",
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
        )

    fig.suptitle(f"Qualitative comparison on {image_id}", fontsize=15)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

def visualize_single_prediction_fullsup(
    image_rgb: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    title: str,
    save_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Full-supervision qualitative figure:
    - input image
    - predicted overlay
    - GT overlay
    - error map
    """
    pred_rgb = decode_voc_mask(pred_mask.astype(np.uint8))
    gt_rgb = decode_voc_mask(gt_mask.astype(np.uint8))
    pred_overlay = overlay_mask_on_image(image_rgb, pred_rgb, alpha=0.45)
    gt_overlay = overlay_mask_on_image(image_rgb, gt_rgb, alpha=0.45)
    error_rgb = make_error_map(pred_mask, gt_mask)

    metrics = compute_sample_metrics(pred_mask, gt_mask)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    axes[1].imshow(pred_overlay)
    axes[1].set_title(
        f"Prediction\npix acc={metrics['pixel_acc']:.3f}, sample mIoU={metrics['sample_mIoU']:.3f}"
    )
    axes[1].axis("off")

    axes[2].imshow(gt_overlay)
    axes[2].set_title("Ground-truth mask")
    axes[2].axis("off")

    axes[3].imshow(error_rgb)
    axes[3].set_title("Error map")
    axes[3].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_model_on_dataset_indices_fullsup(
    backbone: torch.nn.Module,
    model: torch.nn.Module,
    dataset,
    indices: Iterable[int],
    device: str,
    patch_size: int = 14,
    save_dir: Optional[Path] = None,
    model_name: str = "model",
    show: bool = True,
):
    """
    Visualize a fully-supervised model on selected dataset indices.
    """
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        sample = dataset[idx]
        pred = predict_single_sample(
            backbone=backbone,
            model=model,
            sample=sample,
            device=device,
            patch_size=patch_size,
        )

        image_id = sample.get("image_id", f"sample_{idx:05d}")
        title = f"{model_name} — {image_id}"

        save_path = None
        if save_dir is not None:
            save_path = save_dir / f"{model_name}_{image_id}.png"

        visualize_single_prediction_fullsup(
            image_rgb=pred["image_rgb"],
            pred_mask=pred["pred_mask"],
            gt_mask=pred["gt_mask"],
            title=title,
            save_path=save_path,
            show=show,
        )