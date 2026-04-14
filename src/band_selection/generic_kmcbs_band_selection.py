#!/usr/bin/env python3
# generic_kmcbs_band_selection.py
#
# K-Means Clustering Band Selection (KMCBS) for any HSI dataset.
# Just edit the CONFIG block to point to your cube + GT.

import os
import sys
import datetime
import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================
# CONFIG – EDIT THIS PART
# ==========================
DATASET_NAME = "ksc"  # e.g. "montana", "ksc", "botswana", "salinas", "indian_pines", "paviaU"

CUBE_PATH = "ksc_noEDA_clean.npy"  # e.g. "ksc_clean.npy"
GT_PATH   = "ksc_noEDA_gt.npy"       # e.g. "KSC_gt.npy"

# Number of K-Means clusters.
# Rule of thumb: 50, or B // 3 for small cubes.
# The script auto-caps this at B if it is too large.
N_CLUSTERS = 50

OUT_ROOT   = f"{DATASET_NAME}_kmcbs_band_selection_results"
os.makedirs(OUT_ROOT, exist_ok=True)

LOG_PATH    = os.path.join(OUT_ROOT, f"{DATASET_NAME}_kmcbs_band_selection_log.txt")
ORDER_PATH  = os.path.join(OUT_ROOT, f"{DATASET_NAME}_kmcbs_order.npy")
SCORES_PATH = os.path.join(OUT_ROOT, f"{DATASET_NAME}_kmcbs_scores.npy")


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
print(f" {DATASET_NAME.upper()} KMCBS Band Selection")
print("Started:", datetime.datetime.now())
print("========================================")

# ---------------------------
# Load data (memory-friendly)
# ---------------------------
print(f"Loading cube from {CUBE_PATH} (mmap_mode='r') ...")
cube = np.load(CUBE_PATH, mmap_mode="r")   # (H, W, B)
gt   = np.load(GT_PATH)                    # (H, W)

H, W, B = cube.shape
num_classes = int(gt.max())

print(f"Cube shape : {cube.shape}")
print(f"GT shape   : {gt.shape}")
print(f"Num bands  : {B}")
print(f"Num classes (excluding background): {num_classes}")
print()

# Class distribution (for reference / sanity check)
unique_cls, counts_cls = np.unique(gt[gt > 0], return_counts=True)
print(f"Total labeled pixels (gt>0): {counts_cls.sum()}")
print("Class distribution (1..C):")
for u, c in zip(unique_cls, counts_cls):
    print(f"  Class {int(u)}: {int(c)}")
print()

# ---------------------------
# Build band feature matrix
#
# KMCBS clusters BANDS, not pixels.
# We build X_bands of shape (B, N_pix) where each ROW is one band's
# pixel values across the full image.  We subsample pixels if the
# image is very large to keep memory and compute manageable.
# ---------------------------
MAX_PIXELS = 50_000
print(f"Building band feature matrix (max_pixels={MAX_PIXELS}) ...")

X_all = cube.reshape(-1, B).astype(np.float32)   # (N_pix, B)
N_pix = X_all.shape[0]

if N_pix > MAX_PIXELS:
    rng = np.random.default_rng(seed=SEED)
    idx = rng.choice(N_pix, size=MAX_PIXELS, replace=False)
    X_all = X_all[idx]
    print(f"  Subsampled to {MAX_PIXELS} pixels for band clustering.")

# (B, N_pix) — one ROW per band, one COLUMN per pixel sample
X_bands = X_all.T
print(f"Band feature matrix shape: {X_bands.shape}  (B={B} rows, N_pix cols)")
print()

# ---------------------------
# Standardize band vectors
#
# StandardScaler scales each FEATURE (column) to zero mean / unit var.
# Our samples are bands (rows) and features are pixel values (columns),
# so we pass X_bands directly — NOT transposed — so that each pixel
# position (column) is normalised across all bands.
# This removes global brightness differences between bands before
# K-Means computes Euclidean distances.
# ---------------------------
print("Standardizing band vectors ...")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_bands)          # (B, N_pix), scales each pixel-column
print(f"  Scaled matrix shape: {X_scaled.shape}")
print()

# ---------------------------
# K-Means clustering on bands
#
# Each band (row in X_scaled) is one data point in N_pix-dimensional
# feature space.  K-Means groups spectrally similar bands together.
# ---------------------------
n_clusters = min(N_CLUSTERS, B)
print(f"Running K-Means (k={n_clusters}, n_init=10, random_state={SEED}) ...")
km = KMeans(
    n_clusters=n_clusters,
    n_init=10,
    random_state=SEED,
    max_iter=300,
)
km.fit(X_scaled)
labels = km.labels_    # (B,) — cluster ID for each band, 0-based

actual_k = len(np.unique(labels))
print(f"Actual clusters obtained: {actual_k}")
print()

# Cluster size summary
unique_labels, cluster_counts = np.unique(labels, return_counts=True)
print("Cluster size statistics (bands per cluster):")
print(f"  Min  : {cluster_counts.min()}")
print(f"  Max  : {cluster_counts.max()}")
print(f"  Mean : {cluster_counts.mean():.1f}")
print()

# ---------------------------
# Correlation-distance matrix
#
# dist(i, j) = 1 - |corr(band_i, band_j)|  in [0, 1]
# Used for:
#   (a) selecting the most central (representative) band per cluster
#   (b) ordering non-representative bands within each cluster
#
# Computed on the ORIGINAL (un-scaled) X_bands so that the correlation
# reflects the true spectral similarity between bands.
# ---------------------------
print("Computing band-band correlation-distance matrix ...")
C = np.corrcoef(X_bands)                  # (B, B), values in [-1, 1]
C = np.clip(C, -1.0, 1.0)
D = (1.0 - np.abs(C)).astype(np.float32)  # distance in [0, 1]
D = (D + D.T) / 2.0                       # enforce exact symmetry
np.fill_diagonal(D, 0.0)
print(f"  Correlation-distance matrix shape: {D.shape}")
print()

# ---------------------------
# Pick one representative band per cluster
#
# Representative = the band whose pixel-value vector is most correlated
# with the mean (centroid) of all bands in that cluster.
# This is the most "typical" band in the cluster.
# ---------------------------
print("Selecting representative band per cluster ...")
representatives = []    # one band index per cluster, in unique_labels order

for c in unique_labels:
    band_indices = np.where(labels == c)[0]

    if len(band_indices) == 1:
        # Only one band in the cluster — it is trivially the representative
        representatives.append(int(band_indices[0]))
        continue

    # Centroid = mean of all band vectors in this cluster (original scale)
    centroid = X_bands[band_indices].mean(axis=0)   # (N_pix,)

    dists = []
    for bi in band_indices:
        v = X_bands[bi]
        # Safe Pearson correlation
        if v.std() < 1e-9 or centroid.std() < 1e-9:
            dists.append(1.0)   # constant band → treat as maximally distant
        else:
            corr = float(np.corrcoef(v, centroid)[0, 1])
            dists.append(1.0 - abs(corr))

    best = band_indices[int(np.argmin(dists))]
    representatives.append(int(best))

representatives = np.array(representatives, dtype=np.int64)
print(f"  Representatives (cluster order, first 20): {representatives[:20].tolist()}")
print()

# Build lookup: cluster ID → its representative band index
clust_to_rep = {int(c): int(representatives[i])
                for i, c in enumerate(unique_labels)}

# ---------------------------
# Build band importance scores
#
# Representative band score  = cluster_size / max_cluster_size  (in [0,1])
#   Larger cluster → covers more of the spectrum → higher priority.
#
# Non-representative score   = norm_size * (1 - dist_to_rep)
#   Bands very close to the representative still carry some useful
#   information; bands far from it are mostly noise/redundant.
# ---------------------------
print("Computing band importance scores ...")
scores     = np.zeros(B, dtype=np.float32)
max_size   = float(cluster_counts.max())

for i, c in enumerate(unique_labels):
    rep_band  = clust_to_rep[int(c)]
    band_inds = np.where(labels == c)[0]
    norm_size = cluster_counts[i] / max_size          # [0, 1]

    scores[rep_band] = norm_size

    non_members = band_inds[band_inds != rep_band]
    for bi in non_members:
        dist_to_rep  = float(D[bi, rep_band])
        scores[bi]   = norm_size * max(0.0, 1.0 - dist_to_rep)

# ---------------------------
# Build full band ordering
#
# Priority order:
#   1. Representative bands, sorted by descending cluster size
#      (most spectrally important cluster first).
#   2. Non-representative bands, sorted within each cluster by ascending
#      correlation distance to their rep (closest = still somewhat useful).
#
# This guarantees that for Top-K <= n_clusters, select_top_k_bands()
# returns exactly the K most diverse representative bands.
# ---------------------------
print("Building full band ordering ...")

cluster_sizes = cluster_counts.astype(np.float32)
rep_priority  = np.argsort(-cluster_sizes)   # cluster positions, large-first
ordered_reps  = representatives[rep_priority]  # corresponding band indices

non_rep_list = []
for rank_idx in rep_priority:
    c        = int(unique_labels[rank_idx])
    rep_band = clust_to_rep[c]
    members  = np.where(labels == c)[0]
    non_mems = members[members != rep_band]

    if len(non_mems) == 0:
        continue

    dists_to_rep = D[non_mems, rep_band]
    sorted_non   = non_mems[np.argsort(dists_to_rep)]
    non_rep_list.extend(sorted_non.tolist())

order = np.concatenate([ordered_reps,
                        np.array(non_rep_list, dtype=np.int64)])

assert len(order) == B,            f"Order length mismatch: {len(order)} vs {B}"
assert len(np.unique(order)) == B, "Duplicate band indices detected!"

print("Top-20 bands by KMCBS (0-based indices):")
for rank, b in enumerate(order[:20], start=1):
    c_id   = int(labels[b])
    is_rep = "REP" if b == clust_to_rep[c_id] else "   "
    print(f"  Rank {rank:2d}: band {b:4d}  cluster={c_id:3d}  {is_rep}  "
          f"score={scores[b]:.6f}")

# ---------------------------
# Save outputs
# ---------------------------
np.save(ORDER_PATH,  order)
np.save(SCORES_PATH, scores)

print(f"\nSaved band order  to: {ORDER_PATH}")
print(f"Saved band scores to: {SCORES_PATH}")

band_indices_arr = np.arange(B, dtype=np.int64)
ranks_arr        = np.empty(B, dtype=np.int64)
ranks_arr[order] = np.arange(1, B + 1, dtype=np.int64)

df_scores = pd.DataFrame({
    "band_index"       : band_indices_arr,
    "cluster_id"       : labels.astype(np.int64),
    "is_representative": [b == clust_to_rep[int(labels[b])] for b in range(B)],
    "score"            : scores.astype(np.float64),
    "rank_1_is_best"   : ranks_arr,
}).sort_values("rank_1_is_best")

csv_path = os.path.join(OUT_ROOT, f"{DATASET_NAME}_kmcbs_band_scores.csv")
df_scores.to_csv(csv_path, index=False)
print(f"Saved detailed band scores to: {csv_path}")

print("\nDone KMCBS band selection.")
print("Finished:", datetime.datetime.now())