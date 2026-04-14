#!/usr/bin/env python3
# generic_ssep_band_selection.py
#
# SSEP band selection for any HSI dataset:
# - Computes GT edge map with Sobel
# - For each band, computes edge map and Dice/SSIM/overlap vs GT
# - Scores bands and saves ranking.

import os
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from skimage.filters import sobel
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# ==========================
# CONFIG – EDIT THIS PART
# ==========================
DATASET_NAME = "montana"  # e.g. "montana", "ksc", "botswana", "salinas", "indian_pines", "paviaU"

CUBE_PATH = "vnir_uav_noEDA_clean.npy"  # e.g. "KSC_clean.npy"
GT_PATH = "vnir_uav_noEDA_gt.npy"  # e.g. "KSC_gt.npy"


OUT_DIR = f"{DATASET_NAME}_ssep_band_selection_no_eda_results"
os.makedirs(OUT_DIR, exist_ok=True)

LOG_PATH = os.path.join(OUT_DIR, f"{DATASET_NAME}_ssep_band_selection_log.txt")


# ------------------------------------------------
# Tee logger (console + file)
# ------------------------------------------------
class TeeLogger:
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = TeeLogger(LOG_PATH)


def compute_mask_edge(mask):
    """
    Compute a binary edge map from labeled mask using Sobel.

    mask: (H, W) integer labels (0 = background, >0 = classes)
    Returns:
        mask_edge: (H, W) uint8, 1 on edges, 0 elsewhere
    """
    mask_float = mask.astype(float)
    edge = sobel(mask_float)
    return (edge > 0).astype(np.uint8)


def compute_band_edge_score(band_img, mask_edge, method='dice'):
    """
    Compare band edge map vs mask edge map with Dice / SSIM / overlap.

    band_img: (H, W) spectral band (float)
    mask_edge: (H, W) binary edge mask derived from GT
    """
    band_smoothed = gaussian_filter(band_img, sigma=1.0)

    # Edge map of the band
    band_edge = sobel(band_smoothed)

    thr = np.percentile(band_edge, 95)
    band_edge_bin = (band_edge > thr).astype(np.uint8)

    if method == 'ssim':
        score, _ = ssim(band_edge_bin, mask_edge, full=True)
        return float(score)

    elif method == 'dice':
        intersection = np.sum(band_edge_bin * mask_edge)
        total = np.sum(band_edge_bin) + np.sum(mask_edge)
        if total == 0:
            return 0.0
        return float((2.0 * intersection) / total)

    elif method == 'overlap':
        intersection = np.sum(band_edge_bin * mask_edge)
        union = np.sum(band_edge_bin) + np.sum(mask_edge) - intersection
        if union == 0:
            return 0.0
        return float(intersection / union)

    else:
        raise ValueError(f"Unsupported method: {method}")


def ssep_selection(cube, mask, method='dice'):
    """
    SSEP band scoring.

    cube: (H, W, B) - can be a memmap
    mask: (H, W) integer GT labels
    returns:
        band_scores: (B,) score per band
    """
    H, W, B = cube.shape
    print(f"Running SSEP on cube with shape {cube.shape}")
    print(f"Method: {method}")

    mask_edge = compute_mask_edge(mask)
    band_scores = np.zeros(B, dtype=np.float32)

    for i in tqdm(range(B), desc=f"[{DATASET_NAME}] Scoring bands (SSEP)"):
        band_img = cube[:, :, i]
        score = compute_band_edge_score(band_img, mask_edge, method=method)
        band_scores[i] = score

    return band_scores


if __name__ == "__main__":
    print("==========================================")
    print(f" {DATASET_NAME.upper()} SSEP band selection")
    print("Started:", datetime.datetime.now())
    print("==========================================\n")

    print(f"Loading cube from {CUBE_PATH} with mmap_mode='r' ...")
    cube = np.load(CUBE_PATH, mmap_mode="r")
    gt = np.load(GT_PATH)

    H, W, B = cube.shape
    print("HSI shape:", cube.shape)
    print("GT shape :", gt.shape)
    print("GT min/max:", gt.min(), gt.max())
    print()

    ssep_scores = ssep_selection(cube, gt, method="dice")
    order_ssep = np.argsort(ssep_scores)[::-1]

    print("\n=======================================")
    print("SSEP — Full Ranking (Top 10 bands, 0-based):", order_ssep[:10])
    print("Top-30 SSEP bands:", order_ssep[:30])
    print("Top-50 SSEP bands:", order_ssep[:50])
    print("=======================================\n")

    order_path = os.path.join(OUT_DIR, f"{DATASET_NAME}_ssep_order.npy")
    scores_path = os.path.join(OUT_DIR, f"{DATASET_NAME}_ssep_scores.npy")
    np.save(order_path, order_ssep)
    np.save(scores_path, ssep_scores)

    df = pd.DataFrame({
        "rank": np.arange(1, B + 1),
        "band_index": order_ssep,
        "ssep_score_sorted": ssep_scores[order_ssep]
    })
    csv_path = os.path.join(OUT_DIR, f"{DATASET_NAME}_ssep_band_scores.csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(B), ssep_scores, marker="o", linestyle="-")
    plt.xlabel("Band index (0-based)")
    plt.ylabel("SSEP score (Dice vs GT edges)")
    plt.title(f"{DATASET_NAME.upper()} — SSEP band scores vs band index")
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(OUT_DIR, f"{DATASET_NAME}_ssep_scores_vs_band.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print("Saved:")
    print(" -", order_path)
    print(" -", scores_path)
    print(" -", csv_path)
    print(" -", plot_path)
    print("\nDone:", datetime.datetime.now())
