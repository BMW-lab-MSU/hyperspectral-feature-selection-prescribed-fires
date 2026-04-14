#!/usr/bin/env python3
# src/hsi_data.py
"""
Data helpers for hyperspectral image loading.

Functions
---------
extract_labeled_pixels  : flatten labeled pixels to (N, B) for RF/SVM/KNN
extract_center_patches  : extract spatial patches for 3D-CNN
"""

import numpy as np


def extract_labeled_pixels(cube: np.ndarray,
                           gt: np.ndarray,
                           ignore_label: int = 0):
    """
    Extract labeled pixels from a hyperspectral cube.

    Parameters
    ----------
    cube         : (H, W, B) float32 hyperspectral cube
    gt           : (H, W) int ground truth map (0 = background)
    ignore_label : label value to treat as background

    Returns
    -------
    X : (N, B) float32 — spectral feature matrix
    y : (N,)   int64  — class labels in [0, C-1]
    """
    assert cube.shape[:2] == gt.shape, \
        f"Cube {cube.shape[:2]} and GT {gt.shape} spatial dims mismatch."

    mask = gt != ignore_label
    X    = np.asarray(cube[mask, :], dtype=np.float32)   # (N, B)
    y    = gt[mask].astype(np.int64) - 1                 # 1..C → 0..C-1
    return X, y


def extract_center_patches(cube: np.ndarray,
                           gt: np.ndarray,
                           patch_size: int = 5,
                           ignore_label: int = 0):
    """
    Extract fixed-size spatial patches centred on labeled pixels.

    Only pixels where the full patch fits inside the image boundary
    are included (no padding).

    Parameters
    ----------
    cube       : (H, W, B) float32 hyperspectral cube
    gt         : (H, W) int ground truth map
    patch_size : odd integer patch width/height (default 5 → 5×5)
    ignore_label : background label value

    Returns
    -------
    patches : (N, patch_size, patch_size, B) float32
    labels  : (N,) int64 in [0, C-1]
    """
    H, W, B = cube.shape
    assert gt.shape == (H, W), \
        f"Cube spatial dims {(H, W)} and GT {gt.shape} mismatch."
    assert patch_size % 2 == 1, "patch_size must be an odd integer."

    r = patch_size // 2

    ys, xs = np.where(gt != ignore_label)

    patches_list = []
    labels_list  = []

    for y, x in zip(ys, xs):
        # Skip boundary pixels where the full patch does not fit
        if y - r < 0 or y + r >= H or x - r < 0 or x + r >= W:
            continue

        patch = cube[y - r : y + r + 1,
                     x - r : x + r + 1, :]     # (ps, ps, B)
        patches_list.append(patch)
        labels_list.append(int(gt[y, x]) - 1)  # 1..C → 0..C-1

    patches = np.stack(patches_list, axis=0).astype(np.float32)  # (N, ps, ps, B)
    labels  = np.array(labels_list, dtype=np.int64)              # (N,)

    return patches, labels
