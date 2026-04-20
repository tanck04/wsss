from __future__ import annotations

from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from mylibs.dinov2_features import extract_patch_features


UNLABELED_SEED = -1


@torch.no_grad()
def generate_pseudo_labels(
    backbone: torch.nn.Module,
    teacher_model: torch.nn.Module,
    dataset,
    device: str,
    patch_size: int = 14,
    threshold: float = 0.80,
    keep_background: bool = False,
    batch_size: int = 1,
    num_workers: int = 0,
) -> Tuple[Dict[int, np.ndarray], Dict[str, float]]:
    """
    Generate pseudo-label maps from a trained teacher model.

    Returns:
      pseudo_dict[idx] = [H,W] int64 map with UNLABELED_SEED for rejected pixels
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    backbone.eval()
    teacher_model.eval()

    pseudo_dict = {}
    total_kept = 0
    total_pixels = 0

    running_idx = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        B, _, H, W = images.shape

        feat_dict = extract_patch_features(
            model=backbone,
            image_tensor=images,
            patch_size=patch_size,
        )
        patch_map = feat_dict["patch_map"]

        out = teacher_model(
            patch_map=patch_map,
            output_size=(H, W),
        )
        logits = out["image_logits"]                     # [B,K,H,W]
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)            # [B,H,W]

        for b in range(B):
            conf_b = conf[b]
            pred_b = pred[b]

            keep = (conf_b >= threshold)
            if not keep_background:
                keep = keep & (pred_b != 0)

            pseudo = torch.full_like(pred_b, fill_value=UNLABELED_SEED, dtype=torch.long)
            pseudo[keep] = pred_b[keep]

            pseudo_np = pseudo.detach().cpu().numpy().astype(np.int64)
            pseudo_dict[running_idx] = pseudo_np

            total_kept += int((pseudo_np != UNLABELED_SEED).sum())
            total_pixels += int(pseudo_np.size)
            running_idx += 1

    stats = {
        "threshold": threshold,
        "coverage": total_kept / max(total_pixels, 1),
        "num_images": len(dataset),
    }
    return pseudo_dict, stats


class PseudoLabelOverlayDataset(Dataset):
    """
    Overlay pseudo-labels on top of an existing weak dataset.
    Pseudo-labels fill only previously unlabeled seed positions.
    """
    def __init__(self, base_dataset, pseudo_dict: Dict[int, np.ndarray]):
        self.base_dataset = base_dataset
        self.pseudo_dict = pseudo_dict

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        sample = deepcopy(self.base_dataset[idx])

        base_seed = sample["seed_labels"].clone()   # [H,W]
        pseudo = torch.from_numpy(self.pseudo_dict[idx]).long()

        unlabeled = (base_seed == UNLABELED_SEED)
        use_pseudo = (pseudo != UNLABELED_SEED) & unlabeled

        merged = base_seed.clone()
        merged[use_pseudo] = pseudo[use_pseudo]

        sample["seed_labels"] = merged
        sample["seed_mask"] = (merged != UNLABELED_SEED)
        return sample