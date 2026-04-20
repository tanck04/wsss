from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

from mylibs.vis import denormalize_image_tensor


UNLABELED_SEED = -1


def _component_boxes(mask: np.ndarray, min_area: int = 64, max_components: int = 8):
    """
    Extract bounding boxes from connected components of a binary mask.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    boxes = []
    for comp_id in range(1, num_labels):
        x, y, w, h, area = stats[comp_id]
        if area < min_area:
            continue
        boxes.append((x, y, x + w - 1, y + h - 1))
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0]+1)*(b[3]-b[1]+1), reverse=True)
    return boxes[:max_components]


def refine_pseudolabels_with_sam(
    dataset,
    pseudo_dict: Dict[int, np.ndarray],
    sam_checkpoint: str,
    model_type: str,
    device: str,
    min_area: int = 64,
    min_iou_keep: float = 0.30,
    max_components_per_class: int = 8,
) -> Tuple[Dict[int, np.ndarray], Dict[str, float]]:
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    refined_dict = {}
    total_kept = 0
    total_pixels = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        image_rgb = denormalize_image_tensor(sample["image"])
        pseudo = pseudo_dict[idx]
        refined = np.full_like(pseudo, fill_value=UNLABELED_SEED, dtype=np.int64)

        predictor.set_image(image_rgb)

        classes = [c for c in np.unique(pseudo) if c > 0]
        for cls_id in classes:
            cls_mask = (pseudo == cls_id).astype(np.uint8)
            boxes = _component_boxes(
                cls_mask,
                min_area=min_area,
                max_components=max_components_per_class,
            )

            for box in boxes:
                box_np = np.array(box, dtype=np.float32)
                masks, scores, _ = predictor.predict(
                    box=box_np,
                    multimask_output=True,
                )

                # pick mask best aligned with original pseudo component
                best_mask = None
                best_iou = -1.0
                for m in masks:
                    inter = np.logical_and(m, cls_mask > 0).sum()
                    union = np.logical_or(m, cls_mask > 0).sum()
                    iou = inter / max(union, 1)
                    if iou > best_iou:
                        best_iou = iou
                        best_mask = m

                if best_mask is not None and best_iou >= min_iou_keep:
                    refined[best_mask] = cls_id

        refined_dict[idx] = refined
        total_kept += int((refined != UNLABELED_SEED).sum())
        total_pixels += int(refined.size)

    stats = {
        "coverage": total_kept / max(total_pixels, 1),
        "num_images": len(dataset),
        "min_iou_keep": min_iou_keep,
        "min_area": min_area,
    }
    return refined_dict, stats