from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt

IGNORE_LABEL = 255

VOC_CLASSES = [
    "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

VOC_COLORMAP = np.array([
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
], dtype=np.uint8)


def decode_voc_mask(mask: np.ndarray, colormap: np.ndarray = VOC_COLORMAP) -> np.ndarray:
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    valid = mask != IGNORE_LABEL
    color_mask[valid] = colormap[mask[valid]]
    color_mask[~valid] = np.array([255, 255, 255], dtype=np.uint8)
    return color_mask


def overlay_mask_on_image(image: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image_f = image.astype(np.float32)
    mask_f = mask_rgb.astype(np.float32)
    blended = (1 - alpha) * image_f + alpha * mask_f
    return np.clip(blended, 0, 255).astype(np.uint8)


def extract_image_tags(mask: np.ndarray) -> List[int]:
    cls_ids = np.unique(mask)
    return sorted([int(c) for c in cls_ids if c not in (0, IGNORE_LABEL)])


def tags_to_names(tag_ids: List[int]) -> List[str]:
    return [VOC_CLASSES[c] for c in tag_ids]


def compute_class_boxes(mask: np.ndarray) -> Dict[int, Tuple[int, int, int, int]]:
    boxes = {}
    for cls_id in extract_image_tags(mask):
        ys, xs = np.where(mask == cls_id)
        if len(xs) == 0 or len(ys) == 0:
            continue
        boxes[cls_id] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return boxes


def erode_binary_mask(binary_mask: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    binary_mask = binary_mask.astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    eroded = cv2.erode(binary_mask, kernel, iterations=iterations)
    return eroded.astype(bool)


def safe_sample_points(coords: np.ndarray, n_samples: int) -> np.ndarray:
    if len(coords) == 0:
        return np.empty((0, 2), dtype=np.int64)
    if n_samples >= len(coords):
        return coords
    idx = np.random.choice(len(coords), size=n_samples, replace=False)
    return coords[idx]


def generate_sparse_seed_labels(
    mask: np.ndarray,
    fg_fraction: float = 0.05,
    bg_fraction: float = 0.01,
    erosion_kernel: int = 5,
    erosion_iter: int = 1,
    min_points_per_class: int = 5,
    min_bg_points: int = 20,
) -> Dict[str, np.ndarray]:
    H, W = mask.shape
    seed_mask = np.zeros((H, W), dtype=bool)
    seed_labels = np.full((H, W), fill_value=-1, dtype=np.int64)
    fg_seed_mask = np.zeros((H, W), dtype=bool)
    bg_seed_mask = np.zeros((H, W), dtype=bool)

    present_classes = extract_image_tags(mask)

    for cls_id in present_classes:
        cls_region = (mask == cls_id)
        cls_region_eroded = erode_binary_mask(cls_region, kernel_size=erosion_kernel, iterations=erosion_iter)
        usable_region = cls_region_eroded if cls_region_eroded.sum() > 0 else cls_region

        coords = np.argwhere(usable_region)
        if len(coords) == 0:
            continue

        n_samples = max(min_points_per_class, int(np.ceil(fg_fraction * len(coords))))
        sampled = safe_sample_points(coords, n_samples)

        for y, x in sampled:
            seed_mask[y, x] = True
            fg_seed_mask[y, x] = True
            seed_labels[y, x] = cls_id

    bg_region = (mask == 0)
    bg_region_eroded = erode_binary_mask(bg_region, kernel_size=erosion_kernel, iterations=erosion_iter)
    usable_bg = bg_region_eroded if bg_region_eroded.sum() > 0 else bg_region

    bg_coords = np.argwhere(usable_bg)
    if len(bg_coords) > 0:
        n_bg_samples = max(min_bg_points, int(np.ceil(bg_fraction * len(bg_coords))))
        sampled_bg = safe_sample_points(bg_coords, n_bg_samples)

        for y, x in sampled_bg:
            seed_mask[y, x] = True
            bg_seed_mask[y, x] = True
            seed_labels[y, x] = 0

    return {
        "seed_mask": seed_mask,
        "seed_labels": seed_labels,
        "fg_seed_mask": fg_seed_mask,
        "bg_seed_mask": bg_seed_mask,
        "tags": np.array(present_classes, dtype=np.int64),
    }


def render_seed_overlay(image: np.ndarray, seed_labels: np.ndarray, seed_mask: np.ndarray, point_radius: int = 3) -> np.ndarray:
    canvas = image.copy()
    ys, xs = np.where(seed_mask)
    for y, x in zip(ys, xs):
        cls_id = int(seed_labels[y, x])
        if cls_id == -1:
            continue
        color = tuple(int(c) for c in VOC_COLORMAP[cls_id].tolist())
        cv2.circle(canvas, (int(x), int(y)), point_radius, color, thickness=-1)
        cv2.circle(canvas, (int(x), int(y)), point_radius + 1, (255, 255, 255), thickness=1)
    return canvas


def draw_boxes_on_image(image: np.ndarray, boxes: Dict[int, Tuple[int, int, int, int]], thickness: int = 2) -> np.ndarray:
    canvas = image.copy()
    for cls_id, (x1, y1, x2, y2) in boxes.items():
        color = tuple(int(c) for c in VOC_COLORMAP[cls_id].tolist())
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=thickness)
        cv2.putText(
            canvas,
            VOC_CLASSES[cls_id],
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA
        )
    return canvas


def summarize_sample(mask: np.ndarray, weak: Dict[str, np.ndarray]) -> Dict[str, object]:
    tags = weak["tags"].tolist()
    return {
        "present_class_ids": tags,
        "present_class_names": tags_to_names(tags),
        "num_fg_seeds": int(weak["fg_seed_mask"].sum()),
        "num_bg_seeds": int(weak["bg_seed_mask"].sum()),
        "num_total_seeds": int(weak["seed_mask"].sum()),
        "mask_shape": tuple(mask.shape),
    }


def visualize_problem_setup(
    image: np.ndarray,
    mask: np.ndarray,
    weak: Dict[str, np.ndarray],
    sample_id: str,
    save_dir: Path,
    show: bool = True,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    tags = weak["tags"].tolist()
    boxes = compute_class_boxes(mask)

    full_mask_rgb = decode_voc_mask(mask)
    full_mask_overlay = overlay_mask_on_image(image, full_mask_rgb, alpha=0.45)
    seed_overlay = render_seed_overlay(image, weak["seed_labels"], weak["seed_mask"], point_radius=3)
    boxed_image = draw_boxes_on_image(image, boxes)

    tag_text = "Present tags:\n" + "\n".join([f"{cls_id:>2}: {VOC_CLASSES[cls_id]}" for cls_id in tags])
    if len(tags) == 0:
        tag_text = "Present tags:\n(no foreground classes)"

    fig, axes = plt.subplots(1, 5, figsize=(24, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(full_mask_overlay)
    axes[1].set_title("Full mask overlay")
    axes[1].axis("off")

    axes[2].imshow(seed_overlay)
    axes[2].set_title("Sparse seeds")
    axes[2].axis("off")

    axes[3].imshow(boxed_image)
    axes[3].set_title("Optional class boxes")
    axes[3].axis("off")

    axes[4].axis("off")
    axes[4].text(0.0, 1.0, tag_text, fontsize=11, va="top", ha="left", family="monospace")
    axes[4].set_title("Image-level tags")

    fig.suptitle(f"Weak supervision design — {sample_id}", fontsize=14)
    fig.tight_layout()

    out_path = save_dir / f"{sample_id}_problem_setup.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path