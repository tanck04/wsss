from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCSegmentation

from mylibs.dinov2_features import build_dinov2_transform
from mylibs.weak_labels import generate_sparse_seed_labels


IGNORE_LABEL = 255
UNLABELED_SEED = -1
NUM_CLASSES = 21


def resize_label_map(label_map: np.ndarray, output_size: int) -> np.ndarray:
    """
    Resize a 2D integer label map using nearest-neighbor interpolation.
    Safe for masks containing values like 255 or -1.
    """
    resized = cv2.resize(
        label_map.astype(np.int32),
        (output_size, output_size),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(np.int64)


def build_fg_tag_vector(tag_ids: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """
    Build a foreground-only multi-hot vector of shape [num_classes - 1].
    Class 0 (background) is excluded from the tag vector.
    """
    vec = np.zeros((num_classes - 1,), dtype=np.float32)
    for cls_id in tag_ids:
        cls_id = int(cls_id)
        if 1 <= cls_id < num_classes:
            vec[cls_id - 1] = 1.0
    return vec


class VOCWeakSegmentationDataset(Dataset):
    """
    Pascal VOC 2012 dataset wrapper for weakly-supervised semantic segmentation.

    Returns:
        {
            "image":       FloatTensor [3, S, S], DINO-normalized
            "mask":        LongTensor  [S, S], full resized mask for evaluation
            "seed_labels": LongTensor  [S, S], -1 for unlabeled seed positions
            "seed_mask":   BoolTensor  [S, S]
            "tags_fg":     FloatTensor [20], foreground class presence vector
            "index":       LongTensor scalar
            "image_id":    str
        }
    """
    def __init__(
        self,
        root: str,
        image_set: str,
        input_size: int = 448,
        fg_fraction: float = 0.05,
        bg_fraction: float = 0.005,
        erosion_kernel: int = 5,
        erosion_iter: int = 1,
        min_points_per_class: int = 5,
        min_bg_points: int = 20,
        year: str = "2012",
        download: bool = False,
    ):
        super().__init__()

        self.root = root
        self.image_set = image_set
        self.input_size = input_size
        self.fg_fraction = fg_fraction
        self.bg_fraction = bg_fraction
        self.erosion_kernel = erosion_kernel
        self.erosion_iter = erosion_iter
        self.min_points_per_class = min_points_per_class
        self.min_bg_points = min_bg_points

        self.raw_dataset = VOCSegmentation(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
        )

        self.image_transform = build_dinov2_transform(input_size=input_size)

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_pil, mask_pil = self.raw_dataset[idx]

        image_tensor = self.image_transform(img_pil.convert("RGB"))  # [3, S, S]

        mask_np = np.array(mask_pil, dtype=np.uint8)

        weak = generate_sparse_seed_labels(
            mask=mask_np,
            fg_fraction=self.fg_fraction,
            bg_fraction=self.bg_fraction,
            erosion_kernel=self.erosion_kernel,
            erosion_iter=self.erosion_iter,
            min_points_per_class=self.min_points_per_class,
            min_bg_points=self.min_bg_points,
        )

        full_mask_resized = resize_label_map(mask_np, self.input_size)
        seed_labels_resized = resize_label_map(weak["seed_labels"], self.input_size)
        seed_mask_resized = (seed_labels_resized != UNLABELED_SEED)

        tags_fg = build_fg_tag_vector(weak["tags"], num_classes=NUM_CLASSES)

        # Try to expose a stable image id for reporting/visualization
        if hasattr(self.raw_dataset, "images"):
            image_id = Path(self.raw_dataset.images[idx]).stem
        else:
            image_id = f"{self.image_set}_{idx:05d}"

        sample = {
            "image": torch.from_numpy(image_tensor.numpy()).float(),
            "mask": torch.from_numpy(full_mask_resized).long(),
            "seed_labels": torch.from_numpy(seed_labels_resized).long(),
            "seed_mask": torch.from_numpy(seed_mask_resized).bool(),
            "tags_fg": torch.from_numpy(tags_fg).float(),
            "index": torch.tensor(idx, dtype=torch.long),
            "image_id": image_id,
        }
        return sample

class VOCFullSupervisionDataset(Dataset):
    """
    Pascal VOC 2012 dataset wrapper for fully-supervised upper-bound training.

    Returns:
        {
            "image":       FloatTensor [3, S, S], DINO-normalized
            "mask":        LongTensor  [S, S], full resized mask
            "seed_labels": LongTensor  [S, S], dummy = -1 everywhere
            "seed_mask":   BoolTensor  [S, S], dummy = False everywhere
            "tags_fg":     FloatTensor [20], optional foreground class tags
            "index":       LongTensor scalar
            "image_id":    str
        }

    Why include dummy seed fields?
    - So the same visualization utilities can still work later if needed.
    """
    def __init__(
        self,
        root: str,
        image_set: str,
        input_size: int = 448,
        year: str = "2012",
        download: bool = False,
    ):
        super().__init__()

        self.root = root
        self.image_set = image_set
        self.input_size = input_size

        self.raw_dataset = VOCSegmentation(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
        )

        self.image_transform = build_dinov2_transform(input_size=input_size)

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_pil, mask_pil = self.raw_dataset[idx]

        # Image tensor for DINOv2
        image_tensor = self.image_transform(img_pil.convert("RGB")).float()

        # Full segmentation mask
        mask_np = np.array(mask_pil, dtype=np.uint8)
        full_mask_resized = resize_label_map(mask_np, self.input_size)

        # Optional foreground tag vector for reporting/consistency
        present_classes = [int(c) for c in np.unique(mask_np) if c not in (0, IGNORE_LABEL)]
        tags_fg = build_fg_tag_vector(np.array(present_classes, dtype=np.int64), num_classes=NUM_CLASSES)

        # Dummy seed placeholders so visualization code remains compatible
        seed_labels = np.full((self.input_size, self.input_size), fill_value=UNLABELED_SEED, dtype=np.int64)
        seed_mask = np.zeros((self.input_size, self.input_size), dtype=bool)

        if hasattr(self.raw_dataset, "images"):
            image_id = Path(self.raw_dataset.images[idx]).stem
        else:
            image_id = f"{self.image_set}_{idx:05d}"

        sample = {
            "image": image_tensor,
            "mask": torch.from_numpy(full_mask_resized).long(),
            "seed_labels": torch.from_numpy(seed_labels).long(),
            "seed_mask": torch.from_numpy(seed_mask).bool(),
            "tags_fg": torch.from_numpy(tags_fg).float(),
            "index": torch.tensor(idx, dtype=torch.long),
            "image_id": image_id,
        }
        return sample