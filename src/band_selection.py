#!/usr/bin/env python3
# src/band_selection.py
"""
Utilities for loading pre-computed band orders and slicing cubes.

Each band selection method (PCA, SSEP, SRPA, DRL, KMCBS) produces:
  {dataset}_{method}_band_selection_results/
      {dataset}_{method}_order.npy   ← band indices sorted best-first

IMPORTANT: paviaU uses capital U in folder and file names throughout.
           Paths are constructed using the original dataset_name case.
"""

import os
import numpy as np


# ── Path helpers ───────────────────────────────────────────────────────────────

def _make_order_path(dataset_name: str, method: str) -> str:
    """
    Build path to the band order .npy file for a given dataset + method.

    Uses original case of dataset_name to preserve 'paviaU' capital U.
    """
    m  = method.lower()
    ds = dataset_name   # preserve original case (paviaU, not paviau)

    path_map = {
        "pca":   f"{ds}_pca_bandselection_results/{ds}_pca_order.npy",
        "ssep":  f"{ds}_ssep_band_selection_results/{ds}_ssep_order.npy",
        "srpa":  f"{ds}_srpa_band_selection_results/{ds}_srpa_order.npy",
        "drl":   f"{ds}_drl_bandselection_results/{ds}_drl_order.npy",
        "kmcbs": f"{ds}_kmcbs_band_selection_results/{ds}_kmcbs_order.npy",
    }

    if m not in path_map:
        raise ValueError(
            f"Unknown band selection method: '{method}'. "
            f"Choose from: {list(path_map.keys())}"
        )
    return path_map[m]


def load_band_order(dataset_name: str, method: str) -> np.ndarray:
    """
    Load pre-computed band order array.

    Parameters
    ----------
    dataset_name : e.g. 'indian_pines', 'paviaU', 'montana'
    method       : 'PCA', 'SSEP', 'SRPA', 'DRL', or 'KMCBS' (case-insensitive)

    Returns
    -------
    order : (B,) int array of band indices sorted best-first (rank 0 = most important)

    Raises
    ------
    FileNotFoundError if the .npy file does not exist.
    """
    path = _make_order_path(dataset_name, method)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Band order file not found: {path}\n"
            f"Run the corresponding band selection script in src/band_selection/ first."
        )
    return np.load(path)


def select_top_k_bands(cube: np.ndarray,
                       band_order: np.ndarray,
                       k: int):
    """
    Slice a hyperspectral cube to the top-k ranked bands.

    Parameters
    ----------
    cube       : (H, W, B) hyperspectral cube
    band_order : (B,) band indices sorted best-first
    k          : number of bands to select

    Returns
    -------
    cube_k           : (H, W, k) sliced cube
    selected_indices : (k,) selected band indices
    """
    if k > len(band_order):
        raise ValueError(
            f"Requested k={k} but band_order has only {len(band_order)} entries."
        )
    idx    = band_order[:k]
    cube_k = cube[:, :, idx]
    return cube_k, idx


# ── SRPA component loaders ─────────────────────────────────────────────────────

def load_srpa_attention(dataset_name: str) -> np.ndarray:
    """
    Load SRPA attention scores for a dataset.
    Uses original case to handle paviaU correctly.
    """
    ds   = dataset_name
    path = os.path.join(
        f"{ds}_srpa_band_selection_results",
        f"{ds}_srpa_attention.npy"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"SRPA attention file not found: {path}")
    return np.load(path)


def load_srpa_redundancy(dataset_name: str) -> np.ndarray:
    """
    Load SRPA redundancy scores for a dataset.
    Uses original case to handle paviaU correctly.
    """
    ds   = dataset_name
    path = os.path.join(
        f"{ds}_srpa_band_selection_results",
        f"{ds}_srpa_redundancy.npy"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"SRPA redundancy file not found: {path}")
    return np.load(path)


def srpa_order_with_lambda(attn: np.ndarray,
                           redundancy: np.ndarray,
                           lam: float) -> np.ndarray:
    """
    Compute SRPA band ranking for a given lambda value.

    scores = attention - lambda * redundancy
    order  = argsort(scores) descending

    Parameters
    ----------
    attn       : (B,) attention scores from SRPA model
    redundancy : (B,) inter-band redundancy scores
    lam        : redundancy penalty coefficient (0 = attention only)

    Returns
    -------
    order : (B,) band indices sorted best-first
    """
    scores = attn - lam * redundancy
    return np.argsort(scores)[::-1]
