from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DecoderType = Literal["linear", "two_layer"]


class LinearDecoder(nn.Module):
    """
    Simple linear segmentation decoder.

    Input:
        patch_map: [B, C, H_p, W_p]
    Output:
        logits:    [B, K, H_p, W_p]

    This is just a 1x1 convolution, i.e. a per-patch linear classifier.
    """
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=1,
            bias=True,
        )

    def forward(self, patch_map: torch.Tensor) -> torch.Tensor:
        if patch_map.ndim != 4:
            raise ValueError(f"Expected patch_map [B, C, H_p, W_p], got {tuple(patch_map.shape)}")
        return self.proj(patch_map)


class TwoLayerDecoder(nn.Module):
    """
    Small 2-layer segmentation decoder.

    Input:
        patch_map: [B, C, H_p, W_p]
    Output:
        logits:    [B, K, H_p, W_p]

    Architecture:
        1x1 conv -> norm (optional) -> GELU -> dropout -> 1x1 conv
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_batchnorm: bool = False,
    ):
        super().__init__()

        norm_layer = nn.BatchNorm2d(hidden_dim) if use_batchnorm else nn.Identity()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=True),
            norm_layer,
            nn.GELU(),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1, bias=True),
        )

    def forward(self, patch_map: torch.Tensor) -> torch.Tensor:
        if patch_map.ndim != 4:
            raise ValueError(f"Expected patch_map [B, C, H_p, W_p], got {tuple(patch_map.shape)}")
        return self.net(patch_map)


def upsample_patch_logits(
    patch_logits: torch.Tensor,
    output_size: Tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Upsample patch-grid logits to full image resolution.

    Args:
        patch_logits: [B, K, H_p, W_p]
        output_size:  (H, W)
        mode:         interpolation mode

    Returns:
        image_logits: [B, K, H, W]
    """
    if patch_logits.ndim != 4:
        raise ValueError(f"Expected patch_logits [B, K, H_p, W_p], got {tuple(patch_logits.shape)}")

    if mode in ("bilinear", "bicubic"):
        return F.interpolate(
            patch_logits,
            size=output_size,
            mode=mode,
            align_corners=False,
        )
    else:
        return F.interpolate(
            patch_logits,
            size=output_size,
            mode=mode,
        )


class FrozenFeatureSegmentationModel(nn.Module):
    """
    Thin wrapper around a decoder operating on frozen patch features.

    Input:
        patch_map:   [B, C, H_p, W_p]
        output_size: (H, W)

    Output:
        patch_logits: [B, K, H_p, W_p]
        image_logits: [B, K, H, W]
    """
    def __init__(self, decoder: nn.Module, upsample_mode: str = "bilinear"):
        super().__init__()
        self.decoder = decoder
        self.upsample_mode = upsample_mode

    def forward(
        self,
        patch_map: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
    ):
        patch_logits = self.decoder(patch_map)

        if output_size is None:
            return {
                "patch_logits": patch_logits,
                "image_logits": patch_logits,
            }

        image_logits = upsample_patch_logits(
            patch_logits,
            output_size=output_size,
            mode=self.upsample_mode,
        )
        return {
            "patch_logits": patch_logits,
            "image_logits": image_logits,
        }


def build_decoder(
    decoder_type: DecoderType,
    in_channels: int,
    num_classes: int,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    use_batchnorm: bool = False,
) -> nn.Module:
    """
    Factory for segmentation decoders.
    """
    if decoder_type == "linear":
        return LinearDecoder(
            in_channels=in_channels,
            num_classes=num_classes,
        )

    if decoder_type == "two_layer":
        return TwoLayerDecoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )

    raise ValueError(f"Unknown decoder_type: {decoder_type}")


def build_segmentation_model(
    decoder_type: DecoderType,
    in_channels: int,
    num_classes: int,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    use_batchnorm: bool = False,
    upsample_mode: str = "bilinear",
) -> FrozenFeatureSegmentationModel:
    """
    Build the full segmentation model used on top of frozen DINO patch features.
    """
    decoder = build_decoder(
        decoder_type=decoder_type,
        in_channels=in_channels,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        use_batchnorm=use_batchnorm,
    )
    return FrozenFeatureSegmentationModel(
        decoder=decoder,
        upsample_mode=upsample_mode,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class ModelSummary:
    decoder_type: str
    in_channels: int
    num_classes: int
    total_params: int
    trainable_params: int


def summarize_model(
    model: nn.Module,
    decoder_type: str,
    in_channels: int,
    num_classes: int,
) -> ModelSummary:
    return ModelSummary(
        decoder_type=decoder_type,
        in_channels=in_channels,
        num_classes=num_classes,
        total_params=count_parameters(model),
        trainable_params=count_trainable_parameters(model),
    )