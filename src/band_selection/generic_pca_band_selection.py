#!/usr/bin/env python3
# generic_pca_band_selection.py
#
# PCA-based band scoring for any HSI dataset.
# Just edit the CONFIG block to point to your cube + GT.

import os
import sys
import datetime
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==========================
# CONFIG – EDIT THIS PART
# ==========================
DATASET_NAME = "montana"  # e.g. "montana", "ksc", "botswana", "salinas", "indian_pines", "paviaU"

CUBE_PATH = "vnir_uav_noEDA_clean.npy"  # e.g. "ksc_clean.npy"
GT_PATH = "paviaU_noEDA_gt.npy"  # e.g. "KSC_gt.npy"

OUT_ROOT = f"{DATASET_NAME}_pca_bandselection_no_eda_results"
os.makedirs(OUT_ROOT, exist_ok=True)

LOG_PATH = os.path.join(OUT_ROOT, f"{DATASET_NAME}_pca_band_selection_log.txt")
ORDER_PATH = os.path.join(OUT_ROOT, f"{DATASET_NAME}_pca_order.npy")
SCORES_PATH = os.path.join(OUT_ROOT, f"{DATASET_NAME}_pca_scores.npy")


# ---------------------------
# Tee logger
# ---------------------------
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


# ---------------------------
# Reproducibility
# ---------------------------
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
sys.stdout = TeeLogger(LOG_PATH)

print("========================================")
print(f" {DATASET_NAME.upper()} PCA Band Selection")
print("Started:", datetime.datetime.now())
print("========================================")

# ---------------------------
# Load data (memory-friendly)
# ---------------------------
print(f"Loading cube from {CUBE_PATH} (mmap_mode='r') ...")
cube = np.load(CUBE_PATH, mmap_mode="r")  # (H, W, B)
gt = np.load(GT_PATH)  # (H, W)

H, W, B = cube.shape
num_classes = int(gt.max())

print(f"Cube shape : {cube.shape}")
print(f"GT shape   : {gt.shape}")
print(f"Num bands  : {B}")
print(f"Num classes (excluding background): {num_classes}")
print()

# ---------------------------
# Collect labeled pixels only
# ---------------------------
rows, cols = np.where(gt > 0)  # ignore background (0)
y_labeled = gt[rows, cols] - 1  # 1..C -> 0..C-1

print(f"Total labeled pixels (gt>0): {rows.shape[0]}")

unique, counts = np.unique(gt[gt > 0], return_counts=True)
print("Class distribution (1..C):")
for u, c in zip(unique, counts):
    print(f"  Class {int(u)}: {int(c)}")

print("Extracting spectra for labeled pixels ...")
X_labeled = cube[rows, cols, :]  # (N_labeled, B)
X_labeled = np.asarray(X_labeled, dtype=np.float32)

print("X_labeled shape:", X_labeled.shape)
print()

# ---------------------------
# Standardize features
# ---------------------------
print("Standardizing bands (mean 0, var 1) ...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_labeled)

# ---------------------------
# PCA
# ---------------------------
n_components = B
print(f"Running PCA with n_components = {n_components} ...")
pca = PCA(n_components=n_components, svd_solver="full", random_state=SEED)
pca.fit(X_scaled)

expl_var_ratio = pca.explained_variance_ratio_  # (B,)
components = pca.components_  # (B, B)

print("Top-10 explained variance ratios:")
for i in range(min(10, B)):
    print(f"  PC{i + 1}: {expl_var_ratio[i] * 100:.2f}%")

cum_var_10 = expl_var_ratio[:10].sum() * 100
cum_var_20 = expl_var_ratio[:20].sum() * 100
cum_var_30 = expl_var_ratio[:30].sum() * 100
print(f"Cumulative variance PC1–10 : {cum_var_10:.2f}%")
print(f"Cumulative variance PC1–20 : {cum_var_20:.2f}%")
print(f"Cumulative variance PC1–30 : {cum_var_30:.2f}%")
print()

# ---------------------------
# Band importance scores
# ---------------------------
M = min(30, B)  # how many PCs to use for band scoring
print(f"Computing band scores using top {M} PCs ...")

weights = expl_var_ratio[:M]  # (M,)
comp_top = components[:M, :]  # (M, B)

# score[b] = sum_i (weights[i] * components[i, b]^2)
scores = np.sum((comp_top ** 2) * weights[:, None], axis=0)  # (B,)

# Sort bands by descending importance
order = np.argsort(scores)[::-1]

print("Top-20 bands by PCA score (0-based indices):")
for rank, b in enumerate(order[:20], start=1):
    print(f"  Rank {rank:2d}: band {b} (score={scores[b]:.6e})")

# ---------------------------
# Save outputs
# ---------------------------
np.save(ORDER_PATH, order)
np.save(SCORES_PATH, scores)

print(f"\nSaved band order to:  {ORDER_PATH}")
print(f"Saved band scores to: {SCORES_PATH}")

band_indices = np.arange(B)
ranks = np.empty_like(order)
ranks[order] = np.arange(1, B + 1)

df_scores = pd.DataFrame({
    "band_index": band_indices,
    "score": scores,
    "rank_1_is_best": ranks
}).sort_values("rank_1_is_best")

csv_path = os.path.join(OUT_ROOT, f"{DATASET_NAME}_pca_band_scores.csv")
df_scores.to_csv(csv_path, index=False)
print(f"Saved detailed band scores to: {csv_path}")

print("\nDone PCA band selection.")
print("Finished:", datetime.datetime.now())
