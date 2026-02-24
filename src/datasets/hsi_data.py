#!/usr/bin/env python3
# hsi_data.py

"""
Data helpers: labeled pixel extraction and patch extraction.
You can reuse these across datasets.
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split


def extract_labeled_pixels(cube: np.ndarray,
                           gt: np.ndarray,
                           ignore_label: int = 0):
    """
    Flatten (H, W, B) to (N, B) for all labeled pixels (gt != ignore_label).

    Returns:
        X: (N, B)
        y: (N,)  -- 0..C-1 (background removed)
    """
    assert cube.shape[:2] == gt.shape, "Cube and GT spatial shapes mismatch."

    mask = gt != ignore_label
    X = cube[mask, :]                 # (N, B)

    # gt labels are usually 0..C with 0 = background.
    # Shift to 0..C-1 for non-background pixels.
    y = gt[mask].astype(np.int64)     # e.g. 1..C
    y = y - 1                         # -> 0..C-1
    return X, y


def extract_center_patches(cube: np.ndarray,
                           gt: np.ndarray,
                           patch_size: int = 5,
                           ignore_label: int = 0):
    """
    Extract patches centered on labeled pixels.

    cube: (H, W, B)
    gt:   (H, W)
    patch_size: odd integer, e.g., 5 or 7

    Returns:
        patches: (N, patch_size, patch_size, B)
        labels:  (N,)  -- 0..C-1
    Only extracts patches where the full patch fits inside the image.
    """
    H, W, B = cube.shape
    assert gt.shape == (H, W)

    assert patch_size % 2 == 1, "patch_size should be odd."
    r = patch_size // 2

    ys, xs = np.where(gt != ignore_label)

    patches = []
    labels = []

    for y, x in zip(ys, xs):
        if y - r < 0 or y + r >= H or x - r < 0 or x + r >= W:
            continue  # skip edges where full patch doesn't fit

        patch = cube[y - r:y + r + 1, x - r:x + r + 1, :]  # (ps, ps, B)
        patches.append(patch)

        # shift label from 1..C to 0..C-1
        lab = int(gt[y, x])
        labels.append(lab - 1)

    patches = np.stack(patches, axis=0)                 # (N, ps, ps, B)
    labels = np.array(labels, dtype=np.int64)           # (N,)

    return patches, labels




def load_hsi_dataset(dataset_name: str, data_dir: str = os.path.join("data", "processed")):
    """
    Loads HSI cube and GT label map from data/processed.

    Expected filenames:
      - {dataset}_clean.npy
      - {dataset}_gt.npy

    Example:
      dataset_name="paviaU" -> data/processed/paviaU_clean.npy and data/processed/paviaU_gt.npy
    """
    ds = dataset_name  # keep original casing (Windows paths may be case-sensitive by naming)
    cube_path = os.path.join(data_dir, f"{ds}_clean.npy")
    gt_path   = os.path.join(data_dir, f"{ds}_gt.npy")

    if not os.path.exists(cube_path):
        raise FileNotFoundError(f"Cube file not found: {cube_path}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    cube = np.load(cube_path, mmap_mode="r")  # (H, W, B)
    gt = np.load(gt_path)                     # (H, W)
    return cube, gt


def make_pixel_dataset(
    cube: np.ndarray,
    gt: np.ndarray,
    test_size: float = 0.2,
    seed: int = 0,
    drop_background: bool = True,
):
    """
    Convert (H,W,B) cube + (H,W) GT into pixel-level dataset.

    Returns:
      X_train, y_train, X_test, y_test

    Notes:
      - Background is assumed to be label 0.
      - If drop_background=True, it removes 0 labels.
      - Labels are converted to 0..C-1 (common for sklearn).
    """
    if drop_background:
        rows, cols = np.where(gt > 0)
        y = gt[rows, cols].astype(np.int64) - 1
    else:
        rows, cols = np.where(np.ones_like(gt, dtype=bool))
        y = gt[rows, cols].astype(np.int64)

    X = cube[rows, cols, :]
    X = np.asarray(X, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, y_train, X_test, y_test