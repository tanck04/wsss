from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from mylibs.dinov2_features import extract_patch_features
from mylibs.losses import (
    combined_wsss_loss,
    compute_seed_accuracy,
    full_mask_cross_entropy_loss,
)
from mylibs.metrics import StreamingSegmentationMetrics


def move_batch_to_device(batch: Dict, device: str) -> Dict:
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v
    return moved


def train_one_epoch(
    backbone: torch.nn.Module,
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    patch_size: int = 14,
    lambda_tag: float = 0.0,
    class_weights: Optional[torch.Tensor] = None,
    grad_clip: Optional[float] = None,
    tag_pooling: str = "logsumexp",
) -> Dict[str, float]:
    backbone.eval()
    model.train()

    running_total = 0.0
    running_seed_ce = 0.0
    running_tag_bce = 0.0
    num_images = 0

    seed_correct = 0
    seed_total = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        images = batch["image"]             # [B,3,H,W]
        seed_labels = batch["seed_labels"]  # [B,H,W]
        tags_fg = batch["tags_fg"]          # [B,20]

        B, _, H, W = images.shape

        with torch.no_grad():
            feat_dict = extract_patch_features(
                model=backbone,
                image_tensor=images,
                patch_size=patch_size,
            )
            patch_map = feat_dict["patch_map"]

        out = model(
            patch_map=patch_map,
            output_size=(H, W),
        )
        image_logits = out["image_logits"]

        total_loss, loss_dict = combined_wsss_loss(
            image_logits=image_logits,
            seed_labels=seed_labels,
            tag_targets_fg=tags_fg,
            lambda_tag=lambda_tag,
            class_weights=class_weights,
            tag_pooling=tag_pooling,
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        correct, total = compute_seed_accuracy(
            image_logits=image_logits,
            seed_labels=seed_labels,
        )

        batch_size = B
        num_images += batch_size
        running_total += float(loss_dict["total_loss"].item()) * batch_size
        running_seed_ce += float(loss_dict["seed_ce"].item()) * batch_size
        running_tag_bce += float(loss_dict["tag_bce"].item()) * batch_size
        seed_correct += correct
        seed_total += total

    return {
        "train_total_loss": running_total / max(num_images, 1),
        "train_seed_ce": running_seed_ce / max(num_images, 1),
        "train_tag_bce": running_tag_bce / max(num_images, 1),
        "train_seed_acc": seed_correct / max(seed_total, 1),
    }


@torch.no_grad()
def evaluate_model(
    backbone: torch.nn.Module,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    num_classes: int = 21,
    patch_size: int = 14,
    ignore_label: int = 255,
) -> Dict[str, float]:
    backbone.eval()
    model.eval()

    metrics = StreamingSegmentationMetrics(
        num_classes=num_classes,
        ignore_label=ignore_label,
        device="cpu",
    )

    running_full_ce = 0.0
    num_images = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        images = batch["image"]   # [B,3,H,W]
        full_mask = batch["mask"] # [B,H,W]

        B, _, H, W = images.shape

        feat_dict = extract_patch_features(
            model=backbone,
            image_tensor=images,
            patch_size=patch_size,
        )
        patch_map = feat_dict["patch_map"]

        out = model(
            patch_map=patch_map,
            output_size=(H, W),
        )
        image_logits = out["image_logits"]

        full_ce = full_mask_cross_entropy_loss(
            image_logits=image_logits,
            full_mask=full_mask,
            ignore_label=ignore_label,
        )

        metrics.update_from_logits(
            image_logits=image_logits.detach().cpu(),
            true_mask=full_mask.detach().cpu(),
        )

        batch_size = B
        num_images += batch_size
        running_full_ce += float(full_ce.item()) * batch_size

    result = metrics.compute()

    return {
        "val_full_ce": running_full_ce / max(num_images, 1),
        "val_pixel_acc": result.pixel_accuracy,
        "val_mean_acc": result.mean_class_accuracy,
        "val_mIoU": result.mean_iou,
        "val_per_class_iou": result.per_class_iou,
        "val_per_class_acc": result.per_class_accuracy,
    }


def fit_segmentation_model(
    backbone: torch.nn.Module,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_epochs: int = 10,
    num_classes: int = 21,
    patch_size: int = 14,
    lambda_tag: float = 0.0,
    class_weights: Optional[torch.Tensor] = None,
    grad_clip: Optional[float] = None,
    tag_pooling: str = "logsumexp",
    save_best_path: Optional[str] = None,
    restore_best: bool = True,
) -> Tuple[Dict[str, list], Dict]:
    history = {
        "train_total_loss": [],
        "train_seed_ce": [],
        "train_tag_bce": [],
        "train_seed_acc": [],
        "val_full_ce": [],
        "val_pixel_acc": [],
        "val_mean_acc": [],
        "val_mIoU": [],
    }

    best_miou = -1.0
    best_state = None
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        train_stats = train_one_epoch(
            backbone=backbone,
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            patch_size=patch_size,
            lambda_tag=lambda_tag,
            class_weights=class_weights,
            grad_clip=grad_clip,
            tag_pooling=tag_pooling,
        )

        val_stats = evaluate_model(
            backbone=backbone,
            model=model,
            dataloader=val_loader,
            device=device,
            num_classes=num_classes,
            patch_size=patch_size,
        )

        for k in history:
            if k in train_stats:
                history[k].append(train_stats[k])
            elif k in val_stats:
                history[k].append(val_stats[k])

        print(
            f"[Epoch {epoch:02d}/{num_epochs:02d}] "
            f"train_total={train_stats['train_total_loss']:.4f} | "
            f"seed_ce={train_stats['train_seed_ce']:.4f} | "
            f"tag_bce={train_stats['train_tag_bce']:.4f} | "
            f"seed_acc={train_stats['train_seed_acc']:.4f} | "
            f"val_ce={val_stats['val_full_ce']:.4f} | "
            f"val_mIoU={val_stats['val_mIoU']:.4f} | "
            f"val_pixacc={val_stats['val_pixel_acc']:.4f}"
        )

        if val_stats["val_mIoU"] > best_miou:
            best_miou = val_stats["val_mIoU"]
            best_epoch = epoch
            best_state = {
                "epoch": epoch,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "best_val_mIoU": best_miou,
                "history": copy.deepcopy(history),
                "val_stats": val_stats,
            }

            if save_best_path is not None:
                Path(save_best_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, save_best_path)

    if restore_best and best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    print(f"Best epoch: {best_epoch}, best val mIoU: {best_miou:.4f}")
    return history, best_state


def plot_training_history(history: Dict[str, list], title_prefix: str = ""):
    epochs = range(1, len(history["train_total_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    axes[0].plot(epochs, history["train_total_loss"], label="train total")
    axes[0].plot(epochs, history["train_seed_ce"], label="train seed CE")
    axes[0].plot(epochs, history["val_full_ce"], label="val full-mask CE")
    axes[0].set_title(f"{title_prefix}Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_seed_acc"], label="train seed acc")
    axes[1].plot(epochs, history["val_pixel_acc"], label="val pixel acc")
    axes[1].set_title(f"{title_prefix}Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    axes[2].plot(epochs, history["val_mIoU"], label="val mIoU")
    axes[2].plot(epochs, history["val_mean_acc"], label="val mean class acc")
    axes[2].set_title(f"{title_prefix}Segmentation metrics")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

def train_one_epoch_fullsup(
    backbone: torch.nn.Module,
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    patch_size: int = 14,
    grad_clip: Optional[float] = None,
    ignore_label: int = 255,
) -> Dict[str, float]:
    """
    One epoch of fully-supervised training using full segmentation masks.

    Backbone remains frozen.
    Only the decoder is trained.
    """
    backbone.eval()
    model.train()

    running_full_ce = 0.0
    num_images = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        images = batch["image"]   # [B,3,H,W]
        full_mask = batch["mask"] # [B,H,W]

        B, _, H, W = images.shape

        with torch.no_grad():
            feat_dict = extract_patch_features(
                model=backbone,
                image_tensor=images,
                patch_size=patch_size,
            )
            patch_map = feat_dict["patch_map"]

        out = model(
            patch_map=patch_map,
            output_size=(H, W),
        )
        image_logits = out["image_logits"]

        loss = full_mask_cross_entropy_loss(
            image_logits=image_logits,
            full_mask=full_mask,
            ignore_label=ignore_label,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_size = B
        num_images += batch_size
        running_full_ce += float(loss.item()) * batch_size

    return {
        "train_full_ce": running_full_ce / max(num_images, 1),
    }


def fit_segmentation_model_fullsup(
    backbone: torch.nn.Module,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_epochs: int = 10,
    num_classes: int = 21,
    patch_size: int = 14,
    grad_clip: Optional[float] = None,
    save_best_path: Optional[str] = None,
    restore_best: bool = True,
) -> Tuple[Dict[str, list], Dict]:
    """
    Train a segmentation decoder using full masks as an upper bound.

    Best checkpoint is selected by validation mIoU.
    """
    history = {
        "train_full_ce": [],
        "val_full_ce": [],
        "val_pixel_acc": [],
        "val_mean_acc": [],
        "val_mIoU": [],
    }

    best_miou = -1.0
    best_state = None
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        train_stats = train_one_epoch_fullsup(
            backbone=backbone,
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            patch_size=patch_size,
            grad_clip=grad_clip,
        )

        val_stats = evaluate_model(
            backbone=backbone,
            model=model,
            dataloader=val_loader,
            device=device,
            num_classes=num_classes,
            patch_size=patch_size,
        )

        history["train_full_ce"].append(train_stats["train_full_ce"])
        history["val_full_ce"].append(val_stats["val_full_ce"])
        history["val_pixel_acc"].append(val_stats["val_pixel_acc"])
        history["val_mean_acc"].append(val_stats["val_mean_acc"])
        history["val_mIoU"].append(val_stats["val_mIoU"])

        print(
            f"[Epoch {epoch:02d}/{num_epochs:02d}] "
            f"train_full_ce={train_stats['train_full_ce']:.4f} | "
            f"val_full_ce={val_stats['val_full_ce']:.4f} | "
            f"val_mIoU={val_stats['val_mIoU']:.4f} | "
            f"val_pixel_acc={val_stats['val_pixel_acc']:.4f}"
        )

        if val_stats["val_mIoU"] > best_miou:
            best_miou = val_stats["val_mIoU"]
            best_epoch = epoch
            best_state = {
                "epoch": epoch,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "best_val_mIoU": best_miou,
                "history": copy.deepcopy(history),
                "val_stats": val_stats,
            }

            if save_best_path is not None:
                Path(save_best_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, save_best_path)

    if restore_best and best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    print(f"Best epoch: {best_epoch}, best val mIoU: {best_miou:.4f}")
    return history, best_state


def plot_training_history_fullsup(history: Dict[str, list], title_prefix: str = ""):
    """
    Plot curves for the full-mask upper-bound experiment.
    """
    epochs = range(1, len(history["train_full_ce"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_full_ce"], label="train full-mask CE")
    axes[0].plot(epochs, history["val_full_ce"], label="val full-mask CE")
    axes[0].set_title(f"{title_prefix}Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend()

    axes[1].plot(epochs, history["val_mIoU"], label="val mIoU")
    axes[1].plot(epochs, history["val_pixel_acc"], label="val pixel acc")
    axes[1].plot(epochs, history["val_mean_acc"], label="val mean class acc")
    axes[1].set_title(f"{title_prefix}Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

from mylibs.losses import combined_wsss_loss_with_smoothness


def train_one_epoch_wsss_smooth(
    backbone: torch.nn.Module,
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    patch_size: int = 14,
    lambda_tag: float = 0.0,
    lambda_smooth: float = 0.0,
    smoothness_sigma: float = 0.5,
    class_weights: Optional[torch.Tensor] = None,
    grad_clip: Optional[float] = None,
    tag_pooling: str = "logsumexp",
) -> Dict[str, float]:
    backbone.eval()
    model.train()

    running_total = 0.0
    running_seed_ce = 0.0
    running_tag_bce = 0.0
    running_smooth = 0.0
    num_images = 0

    seed_correct = 0
    seed_total = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        images = batch["image"]
        seed_labels = batch["seed_labels"]
        tags_fg = batch["tags_fg"]

        B, _, H, W = images.shape

        with torch.no_grad():
            feat_dict = extract_patch_features(
                model=backbone,
                image_tensor=images,
                patch_size=patch_size,
            )
            patch_map = feat_dict["patch_map"]

        out = model(
            patch_map=patch_map,
            output_size=(H, W),
        )
        patch_logits = out["patch_logits"]
        image_logits = out["image_logits"]

        total_loss, loss_dict = combined_wsss_loss_with_smoothness(
            image_logits=image_logits,
            seed_labels=seed_labels,
            patch_logits=patch_logits,
            patch_features=patch_map,
            tag_targets_fg=tags_fg,
            lambda_tag=lambda_tag,
            lambda_smooth=lambda_smooth,
            class_weights=class_weights,
            tag_pooling=tag_pooling,
            smoothness_sigma=smoothness_sigma,
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        correct, total = compute_seed_accuracy(
            image_logits=image_logits,
            seed_labels=seed_labels,
        )

        num_images += B
        running_total += float(loss_dict["total_loss"].item()) * B
        running_seed_ce += float(loss_dict["seed_ce"].item()) * B
        running_tag_bce += float(loss_dict["tag_bce"].item()) * B
        running_smooth += float(loss_dict["smooth_loss"].item()) * B
        seed_correct += correct
        seed_total += total

    return {
        "train_total_loss": running_total / max(num_images, 1),
        "train_seed_ce": running_seed_ce / max(num_images, 1),
        "train_tag_bce": running_tag_bce / max(num_images, 1),
        "train_smooth_loss": running_smooth / max(num_images, 1),
        "train_seed_acc": seed_correct / max(seed_total, 1),
    }


def fit_segmentation_model_wsss_smooth(
    backbone: torch.nn.Module,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_epochs: int = 10,
    num_classes: int = 21,
    patch_size: int = 14,
    lambda_tag: float = 0.0,
    lambda_smooth: float = 0.0,
    smoothness_sigma: float = 0.5,
    class_weights: Optional[torch.Tensor] = None,
    grad_clip: Optional[float] = None,
    tag_pooling: str = "logsumexp",
    save_best_path: Optional[str] = None,
    restore_best: bool = True,
):
    history = {
        "train_total_loss": [],
        "train_seed_ce": [],
        "train_tag_bce": [],
        "train_smooth_loss": [],
        "train_seed_acc": [],
        "val_full_ce": [],
        "val_pixel_acc": [],
        "val_mean_acc": [],
        "val_mIoU": [],
    }

    best_miou = -1.0
    best_state = None
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        train_stats = train_one_epoch_wsss_smooth(
            backbone=backbone,
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            patch_size=patch_size,
            lambda_tag=lambda_tag,
            lambda_smooth=lambda_smooth,
            smoothness_sigma=smoothness_sigma,
            class_weights=class_weights,
            grad_clip=grad_clip,
            tag_pooling=tag_pooling,
        )

        val_stats = evaluate_model(
            backbone=backbone,
            model=model,
            dataloader=val_loader,
            device=device,
            num_classes=num_classes,
            patch_size=patch_size,
        )

        for k in history:
            if k in train_stats:
                history[k].append(train_stats[k])
            elif k in val_stats:
                history[k].append(val_stats[k])

        print(
            f"[Epoch {epoch:02d}/{num_epochs:02d}] "
            f"train_total={train_stats['train_total_loss']:.4f} | "
            f"seed_ce={train_stats['train_seed_ce']:.4f} | "
            f"tag_bce={train_stats['train_tag_bce']:.4f} | "
            f"smooth={train_stats['train_smooth_loss']:.4f} | "
            f"seed_acc={train_stats['train_seed_acc']:.4f} | "
            f"val_mIoU={val_stats['val_mIoU']:.4f}"
        )

        if val_stats["val_mIoU"] > best_miou:
            best_miou = val_stats["val_mIoU"]
            best_epoch = epoch
            best_state = {
                "epoch": epoch,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "best_val_mIoU": best_miou,
                "history": copy.deepcopy(history),
                "val_stats": val_stats,
            }

            if save_best_path is not None:
                Path(save_best_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, save_best_path)

    if restore_best and best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    print(f"Best epoch: {best_epoch}, best val mIoU: {best_miou:.4f}")
    return history, best_state