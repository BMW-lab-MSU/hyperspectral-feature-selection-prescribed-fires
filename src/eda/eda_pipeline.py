#!/usr/bin/env python3
# hsi_clean_from_mat.py
#
# Tiny EDA + cleaning template for classic HSI datasets
# (KSC, Botswana, Salinas, etc.) from .mat → *_clean.npy.

import os
import sys
import datetime
import numpy as np
import pandas as pd
from scipy.io import loadmat

# ============================
# CONFIG – EDIT THIS SECTION
# ============================

DATASET_NAME = "indian_pines"  # "ksc", "botswana", or "salinas", "indian_pines", "PaviaU'

DATASET_CONFIG = {
    "ksc": {
        "mat_path": "KSC_corrected.mat",
        "gt_mat_path": "KSC_gt.mat",
        "cube_key": "KSC",
        "gt_key": "KSC_gt",
        "noisy_bands": [0, 1, 2, 3, 4, 5, 6, 7, 8],  # you can fill later after inspection
    },
    "botswana": {
        "mat_path": "Botswana.mat",
        "gt_mat_path": "Botswana_gt.mat",
        "cube_key": "Botswana",
        "gt_key": "Botswana_gt",
        "noisy_bands": [111, 138, 139, 140, 141, 142, 143, 144], # you can fill later after inspection
    },
    "salinas": {
        "mat_path": "Salinas_corrected.mat",  # or "Salinas.mat"
        "gt_mat_path": "Salinas_gt.mat",
        "cube_key": "salinas_corrected",
        "gt_key": "salinas_gt",
        "noisy_bands": [106, 107, 108, 109, 146, 147, 148, 198, 201, 202, 203], # you can fill later after inspection
    },
    "indian_pines": {
        "mat_path": "Indian_pines_corrected.mat",  # or "indian_pines.mat"
        "gt_mat_path": "Indian_pines_gt.mat",
        "cube_key": "indian_pines_corrected",
        "gt_key": "indian_pines_gt",
        "noisy_bands": [102, 103, 104, 143, 144, 145, 195, 197, 198, 199], # you can fill later after inspection
    },
    "paviaU": {
        "mat_path": "PaviaU.mat",  # or "PaviaU.mat"
        "gt_mat_path": "PaviaU_gt.mat",
        "cube_key": "paviaU",
        "gt_key": "paviaU_gt",
        "noisy_bands": [0, 1, 2, 3, 4, 5],# you can fill later after inspection
    },
}

# Normalization mode: "minmax" or "zscore"
NORM_MODE = "minmax"

OUT_DIR = f"{DATASET_NAME}_clean_results" # v1---> for first run with the noise bands and v2 --> for the second bands removing noise bands
os.makedirs(OUT_DIR, exist_ok=True)


# ============================
# Tiny helper: logger
# ============================

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


LOG_PATH = os.path.join(OUT_DIR, f"{DATASET_NAME}_clean_log.txt")
sys.stdout = TeeLogger(LOG_PATH)


# ============================
# Helper: suggest noisy bands
# ============================

def suggest_noisy_bands(band_mins, band_maxs, band_means, band_stds, cube_2d,
                        zero_frac_threshold=0.8,
                        low_std_percentile=5,
                        tiny_range_threshold_ratio=1e-3):
    """
    Heuristic suggestions for "suspicious" bands.

    Parameters
    ----------
    band_mins, band_maxs, band_means, band_stds : (B,)
        Per-band statistics on the *raw* cube.
    cube_2d : (N, B)
        Flattened cube (H*W, B) to measure zeros, etc.
    zero_frac_threshold : float
        Bands with >= this fraction of zeros are flagged.
    low_std_percentile : int
        Bands in the lowest X% of std are flagged.
    tiny_range_threshold_ratio : float
        Bands with (max - min) < ratio * global_range are flagged.
    """
    N, B = cube_2d.shape
    print("----- Heuristic noisy-band suggestions (no automatic removal) -----")

    # 1) Bands with many zeros
    zero_mask = (cube_2d == 0)
    frac_zero = zero_mask.sum(axis=0) / float(N)
    many_zeros = np.where(frac_zero >= zero_frac_threshold)[0]

    # 2) Very low std bands (near-constant)
    std_thresh = np.percentile(band_stds, low_std_percentile)
    low_std = np.where(band_stds <= std_thresh)[0]

    # 3) Very tiny dynamic range compared to global range
    global_range = band_maxs.max() - band_mins.min()
    if global_range <= 0:
        tiny_range = np.array([], dtype=int)
    else:
        band_ranges = band_maxs - band_mins
        tiny_threshold = global_range * tiny_range_threshold_ratio
        tiny_range = np.where(band_ranges <= tiny_threshold)[0]

    # Print them neatly (as suggestions)
    if len(many_zeros) > 0:
        print(f"  Bands with >={zero_frac_threshold * 100:.0f}% zeros:", many_zeros.tolist())
    else:
        print(f"  No bands with >= {zero_frac_threshold * 100:.0f}% zeros.")

    print(f"  Bands in lowest {low_std_percentile}% of std:", low_std.tolist())

    if len(tiny_range) > 0:
        print(f"  Bands with very tiny dynamic range (≤ {tiny_range_threshold_ratio:.1e} * global range):",
              tiny_range.tolist())
    else:
        print("  No bands with extremely tiny dynamic range.")

    # Union of all criteria (for convenience)
    union_candidates = sorted(set(many_zeros.tolist()) |
                              set(low_std.tolist()) |
                              set(tiny_range.tolist()))
    print("\n  >>> Combined candidate noisy bands (union of all criteria):",
          union_candidates)
    print("  (You can choose a subset of these to put into `noisy_bands` in DATASET_CONFIG.)")
    print("---------------------------------------------------------------------\n")

    return {
        "many_zeros": many_zeros,
        "low_std": low_std,
        "tiny_range": tiny_range,
        "union": np.array(union_candidates, dtype=int),
        "frac_zero": frac_zero,
    }


# ============================
# 1. Load .mat → cube + GT
# ============================

cfg = DATASET_CONFIG[DATASET_NAME]

print("========================================")
print(f" {DATASET_NAME.upper()} – EDA + Cleaning from .mat")
print("Started:", datetime.datetime.now())
print("========================================\n")

print("Config:")
for k, v in cfg.items():
    if "path" in k or "key" in k:
        print(f"  {k}: {v}")
print(f"  NORM_MODE: {NORM_MODE}")
print()

print(f"Loading cube from {cfg['mat_path']} ...")
mat = loadmat(cfg["mat_path"])
if cfg["cube_key"] not in mat:
    raise KeyError(f"Key '{cfg['cube_key']}' not found in {cfg['mat_path']}. "
                   f"Available keys: {list(mat.keys())}")

cube = mat[cfg["cube_key"]]  # (H, W, B)
cube = np.asarray(cube, dtype=np.float32)

print(f"Loading GT from {cfg['gt_mat_path']} ...")
gt_mat = loadmat(cfg["gt_mat_path"])
if cfg["gt_key"] not in gt_mat:
    raise KeyError(f"Key '{cfg['gt_key']}' not found in {cfg['gt_mat_path']}. "
                   f"Available keys: {list(gt_mat.keys())}")
gt = gt_mat[cfg["gt_key"]]
gt = np.asarray(gt, dtype=np.int32)

# reshape GT if needed
if gt.ndim == 2 and gt.shape[0] * gt.shape[1] == cube.shape[0] * cube.shape[1] \
        and gt.shape != cube.shape[:2]:
    gt = gt.reshape(cube.shape[0], cube.shape[1])

print("\nRaw shapes:")
print(f"  cube: {cube.shape}  (H, W, B)")
print(f"  gt  : {gt.shape}")
print()

H, W, B = cube.shape

# ============================
# 2. Tiny EDA
# ============================

cube_2d_raw = cube.reshape(-1, B)  # (H*W, B)

print("Intensity stats (over entire cube, per band):")
band_means = cube_2d_raw.mean(axis=0)
band_stds = cube_2d_raw.std(axis=0)
band_mins = cube_2d_raw.min(axis=0)
band_maxs = cube_2d_raw.max(axis=0)

print(f"  Global min intensity : {band_mins.min():.4f}")
print(f"  Global max intensity : {band_maxs.max():.4f}")
print(f"  Mean of band means   : {band_means.mean():.4f}")
print(f"  Mean of band stds    : {band_stds.mean():.4f}")
print()

print("GT stats:")
print(f"  GT min: {gt.min()}, GT max: {gt.max()}")
unique, counts = np.unique(gt, return_counts=True)
print("  Label distribution (label -> count):")
for u, c in zip(unique, counts):
    print(f"    {int(u):2d} -> {int(c)}")

print("\nNOTE: label 0 is usually background; others are classes.\n")

# ---- NEW: automatically suggest candidate noisy bands ----
_ = suggest_noisy_bands(
    band_mins=band_mins,
    band_maxs=band_maxs,
    band_means=band_means,
    band_stds=band_stds,
    cube_2d=cube_2d_raw,
    zero_frac_threshold=0.8,  # tweak if needed
    low_std_percentile=5,  # lowest 5% std
    tiny_range_threshold_ratio=1e-3  # bands with extremely small dynamic range
)

# ============================
# 3. Optional noisy-band removal (manual, via config)
# ============================

noisy_bands = cfg.get("noisy_bands", [])
if len(noisy_bands) > 0:
    noisy_bands = sorted(list(set(noisy_bands)))
    print(f"Removing {len(noisy_bands)} noisy bands (0-based indices): {noisy_bands}")
    mask = np.ones(B, dtype=bool)
    mask[noisy_bands] = False
    cube = cube[:, :, mask]
    B_clean = cube.shape[2]
    print(f"After noisy-band removal: cube shape = {cube.shape}")
else:
    print("No noisy bands specified; keeping all bands.")
    B_clean = B

print()

# ============================
# 4. Normalization
# ============================

cube_2d = cube.reshape(-1, B_clean)

if NORM_MODE.lower() == "minmax":
    print("Applying per-band MIN-MAX normalization to [0,1] ...")
    mins = cube_2d.min(axis=0)
    maxs = cube_2d.max(axis=0)
    denom = (maxs - mins)
    denom[denom == 0] = 1.0
    cube_norm = (cube_2d - mins) / denom

elif NORM_MODE.lower() == "zscore":
    print("Applying per-band Z-SCORE normalization ...")
    means = cube_2d.mean(axis=0)
    stds = cube_2d.std(axis=0)
    stds[stds == 0] = 1.0
    cube_norm = (cube_2d - means) / stds

else:
    raise ValueError(f"Unsupported NORM_MODE: {NORM_MODE}")

cube_norm = cube_norm.astype(np.float32).reshape(H, W, B_clean)
print("Normalization done.")
print(f"Normalized cube shape: {cube_norm.shape}\n")

# ============================
# 5. Save cleaned outputs
# ============================

clean_cube_path = os.path.join(OUT_DIR, f"{DATASET_NAME}_clean.npy")
gt_path_out = os.path.join(OUT_DIR, f"{DATASET_NAME}_gt.npy")
stats_csv_path = os.path.join(OUT_DIR, f"{DATASET_NAME}_clean_stats.csv")

np.save(clean_cube_path, cube_norm)
np.save(gt_path_out, gt.astype(np.int16))

print("Saved:")
print(f"  Clean cube -> {clean_cube_path}")
print(f"  GT         -> {gt_path_out}")

cube_norm_2d = cube_norm.reshape(-1, B_clean)
norm_means = cube_norm_2d.mean(axis=0)
norm_stds = cube_norm_2d.std(axis=0)
norm_mins = cube_norm_2d.min(axis=0)
norm_maxs = cube_norm_2d.max(axis=0)

df_stats = pd.DataFrame({
    "band_index": np.arange(B_clean),
    "orig_min": band_mins[:B_clean],
    "orig_max": band_maxs[:B_clean],
    "orig_mean": band_means[:B_clean],
    "orig_std": band_stds[:B_clean],
    "norm_min": norm_mins,
    "norm_max": norm_maxs,
    "norm_mean": norm_means,
    "norm_std": norm_stds,
})

df_stats.to_csv(stats_csv_path, index=False)
print(f"  Per-band stats -> {stats_csv_path}")

print("\nDone:", datetime.datetime.now())
