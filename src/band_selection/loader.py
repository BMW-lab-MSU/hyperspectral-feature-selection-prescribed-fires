#!/usr/bin/env python3
# band_selection.py

"""
Utility to load band-order files (PCA, SSEP, SRPA, DRL) and
slice cubes to Top-K bands.
"""

import os
import numpy as np


def _make_order_path(dataset_name: str, method: str) -> str:
    m = method.lower()
    ds = dataset_name

    if m == "pca":
        # e.g., montana_pca_bandselection_results/montana_pca_order.npy
        return f"{ds}_pca_bandselection_results/{ds}_pca_order.npy"
    elif m == "ssep":
        # e.g., montana_ssep_band_selection_results/montana_ssep_order.npy
        return f"{ds}_ssep_band_selection_results/{ds}_ssep_order.npy"
    elif m == "srpa":
        # e.g., montana_srpa_band_selection_results/montana_srpa_order.npy
        return f"{ds}_srpa_band_selection_results/{ds}_srpa_order.npy"
    elif m == "drl":
        # e.g., montana_drl_bandselection_results_n/montana_drl_order.npy
        return f"{ds}_drl_bandselection_results/{ds}_drl_order.npy"
    else:
        raise ValueError(f"Unknown band selection method: {method}")


def load_band_order(dataset_name: str, method: str) -> np.ndarray:
    """
    Return a 1D numpy array of band indices (0-based) sorted from best to worst.
    """
    path = _make_order_path(dataset_name, method)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Band order file not found: {path}")
    order = np.load(path)
    return order


def select_top_k_bands(cube: np.ndarray, band_order: np.ndarray, k: int):
    """
    cube: (H, W, B), band_order: (B,), k: int
    returns:
        cube_k: (H, W, k)
        selected_indices: 1D array of length k
    """
    idx = band_order[:k]
    cube_k = cube[:, :, idx]
    return cube_k, idx
