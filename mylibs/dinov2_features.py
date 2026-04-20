from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_dinov2_transform(input_size: int = 448) -> transforms.Compose:
    """
    Build the image transform used before passing an RGB image to DINOv2.

    Notes:
    - input_size should be divisible by patch_size (14 for dinov2_*14 models).
    - We use bicubic resize + ImageNet normalization.
    """
    return transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_dinov2_backbone(
    model_name: str = "dinov2_vits14",
    device: str = "cuda",
) -> torch.nn.Module:
    """
    Load a pretrained DINOv2 backbone from the official torch.hub entry.

    Supported official names include:
    - dinov2_vits14
    - dinov2_vitb14
    - dinov2_vitl14
    - dinov2_vitg14
    - and _reg variants

    The backbone is frozen and set to eval mode.
    """
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval()
    model.to(device)

    for p in model.parameters():
        p.requires_grad = False

    return model


def pil_to_rgb_numpy(img_pil: Image.Image) -> np.ndarray:
    """Convert PIL image to uint8 RGB numpy array."""
    return np.array(img_pil.convert("RGB"), dtype=np.uint8)


def prepare_single_image(
    img_pil: Image.Image,
    transform: transforms.Compose,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Convert a PIL RGB image into a batched tensor [1, 3, H, W].
    """
    x = transform(img_pil.convert("RGB"))
    x = x.unsqueeze(0).to(device, non_blocking=True)
    return x


def _tokens_to_patch_map(
    patch_tokens: torch.Tensor,
    image_hw: Tuple[int, int],
    patch_size: int = 14,
) -> torch.Tensor:
    """
    Convert patch tokens [B, N, C] into patch map [B, C, H_p, W_p].
    """
    if patch_tokens.ndim != 3:
        raise ValueError(f"Expected patch_tokens with shape [B, N, C], got {tuple(patch_tokens.shape)}")

    B, N, C = patch_tokens.shape
    H, W = image_hw
    H_p = H // patch_size
    W_p = W // patch_size

    if N != H_p * W_p:
        raise ValueError(
            f"Token count mismatch: got N={N}, but expected H_p*W_p={H_p}*{W_p}={H_p * W_p}. "
            f"Make sure input_size is divisible by patch_size={patch_size}."
        )

    patch_map = patch_tokens.transpose(1, 2).reshape(B, C, H_p, W_p).contiguous()
    return patch_map


@torch.no_grad()
def extract_patch_features(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    patch_size: int = 14,
) -> Dict[str, torch.Tensor]:
    """
    Extract patch-level DINOv2 features robustly.

    Returns dict with:
    - patch_tokens: [B, N, C]
    - patch_map:    [B, C, H_p, W_p]

    This function is deliberately defensive because different hub/backbone variants
    can expose features slightly differently.
    """
    if image_tensor.ndim != 4:
        raise ValueError(f"Expected image_tensor [B,3,H,W], got {tuple(image_tensor.shape)}")

    B, _, H, W = image_tensor.shape

    # Preferred path for official DINOv2 backbones
    if hasattr(model, "forward_features"):
        feats = model.forward_features(image_tensor)

        if isinstance(feats, dict):
            if "x_norm_patchtokens" in feats:
                patch_tokens = feats["x_norm_patchtokens"]  # [B, N, C]
            elif "x_prenorm" in feats:
                x = feats["x_prenorm"]
                if x.ndim != 3:
                    raise ValueError(f"Unexpected x_prenorm shape: {tuple(x.shape)}")

                # Try dropping CLS token if present
                if x.shape[1] == (H // patch_size) * (W // patch_size) + 1:
                    patch_tokens = x[:, 1:, :]
                elif x.shape[1] == (H // patch_size) * (W // patch_size):
                    patch_tokens = x
                else:
                    raise ValueError(
                        f"Could not infer patch tokens from x_prenorm shape {tuple(x.shape)}"
                    )
            else:
                raise KeyError(
                    "forward_features() returned a dict, but no usable patch-token key was found. "
                    f"Available keys: {list(feats.keys())}"
                )

        elif torch.is_tensor(feats):
            # Some variants may directly return tokens
            x = feats
            if x.ndim != 3:
                raise ValueError(f"Unexpected tensor output shape: {tuple(x.shape)}")

            expected = (H // patch_size) * (W // patch_size)
            if x.shape[1] == expected + 1:
                patch_tokens = x[:, 1:, :]
            elif x.shape[1] == expected:
                patch_tokens = x
            else:
                raise ValueError(
                    f"Unexpected token count in output tensor: {tuple(x.shape)}"
                )
        else:
            raise TypeError(f"Unsupported output type from forward_features(): {type(feats)}")

    elif hasattr(model, "get_intermediate_layers"):
        out = model.get_intermediate_layers(
            image_tensor,
            n=1,
            reshape=False,
            return_class_token=False,
        )

        if isinstance(out, (list, tuple)):
            patch_tokens = out[0]
        else:
            patch_tokens = out

        if patch_tokens.ndim == 4:
            # Some APIs may already return [B, C, H_p, W_p]
            patch_map = patch_tokens.contiguous()
            patch_tokens = patch_map.flatten(2).transpose(1, 2).contiguous()
            return {
                "patch_tokens": patch_tokens,
                "patch_map": patch_map,
            }

        if patch_tokens.ndim != 3:
            raise ValueError(f"Unexpected get_intermediate_layers output shape: {tuple(patch_tokens.shape)}")

    else:
        raise AttributeError(
            "This DINOv2 model exposes neither forward_features() nor get_intermediate_layers()."
        )

    patch_map = _tokens_to_patch_map(
        patch_tokens=patch_tokens,
        image_hw=(H, W),
        patch_size=patch_size,
    )

    return {
        "patch_tokens": patch_tokens.contiguous(),
        "patch_map": patch_map.contiguous(),
    }


@torch.no_grad()
def upsample_patch_logits(
    patch_logits: torch.Tensor,
    output_hw: Tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Upsample patch-grid logits/features back to image resolution.
    Useful later for decoder output.
    """
    return F.interpolate(
        patch_logits,
        size=output_hw,
        mode=mode,
        align_corners=False if mode in ("bilinear", "bicubic") else None,
    )


def feature_norm_map(patch_map: torch.Tensor) -> np.ndarray:
    """
    Compute a simple feature-norm heatmap from patch_map [1, C, H_p, W_p].
    Returns a normalized numpy array [H_p, W_p] in [0, 1].
    """
    if patch_map.ndim != 4 or patch_map.shape[0] != 1:
        raise ValueError(f"Expected patch_map [1, C, H_p, W_p], got {tuple(patch_map.shape)}")

    norm_map = torch.norm(patch_map[0], dim=0)  # [H_p, W_p]
    norm_map = norm_map.detach().cpu().numpy()

    norm_min = norm_map.min()
    norm_max = norm_map.max()
    if norm_max > norm_min:
        norm_map = (norm_map - norm_min) / (norm_max - norm_min)
    else:
        norm_map = np.zeros_like(norm_map, dtype=np.float32)

    return norm_map.astype(np.float32)


def visualize_feature_extraction(
    image_rgb: np.ndarray,
    norm_map: np.ndarray,
    sample_id: str,
    save_dir: Path,
    show: bool = True,
) -> Path:
    """
    Save a simple qualitative visualization:
    - original image
    - DINO patch-feature norm heatmap
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    im = axes[1].imshow(norm_map, cmap="viridis")
    axes[1].set_title("DINO patch-feature norm")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(f"Frozen DINOv2 feature extraction — {sample_id}", fontsize=13)
    fig.tight_layout()

    out_path = save_dir / f"{sample_id}_dinov2_features.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path