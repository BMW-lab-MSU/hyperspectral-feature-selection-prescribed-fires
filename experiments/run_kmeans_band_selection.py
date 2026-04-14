#!/usr/bin/env python3
# run_kmeans_band_selection.py
"""
K-Means clustering-based band selection.
Reviewer comment: include a clustering-based band selection method.

Method: K-Means Band Selection (KMCBS)
- For each Top-K value, cluster B bands into K groups
- Select the band closest to each cluster centroid
- One representative band per cluster = K selected bands

All 6 datasets, EDA condition, Top-K [5-50], all classifiers.
5 independent runs. Results saved to kmeans_band_selection_results.csv
"""

import os
import csv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from hsi_config import DATASETS, TOP_K_LIST
from hsi_data import extract_labeled_pixels, extract_center_patches
from train_models import (
    train_pixel_models,
    train_3dcnn_wrapper,
)

# ── Config ─────────────────────────────────────────────────────────────────────
NUM_RUNS    = 5
RESULTS_CSV = "kmeans_band_selection_results.csv"
METHOD_NAME = "KMCBS"

# ── CSV helpers ────────────────────────────────────────────────────────────────
def init_csv():
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset", "method", "band_mode", "top_k",
                "model", "run",
                "OA", "kappa",
                "precision_macro", "recall_macro", "f1_macro",
                "best_val_acc", "extra_info"
            ])

def append_row(dataset, method, band_mode, top_k,
               model, run, metrics, extra_info=""):
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            dataset, method, band_mode, top_k,
            model, run,
            metrics.get("OA"),
            metrics.get("kappa"),
            metrics.get("precision_macro"),
            metrics.get("recall_macro"),
            metrics.get("f1_macro"),
            metrics.get("best_val_acc", metrics.get("OA")),
            extra_info,
        ])

# ── K-Means Band Selection ─────────────────────────────────────────────────────
def kmeans_band_selection(cube, k, random_state=0):
    """
    Cluster B spectral bands into k groups using K-Means.
    Select the band closest to each cluster centroid.
    Returns selected band indices — exactly k bands.
    cube: (H, W, B)
    k: number of clusters = number of bands to select
    """
    H, W, B = cube.shape
    max_pixels = 50000
    rng = np.random.RandomState(random_state)

    # Sample pixel locations BEFORE reshaping — avoids OOM on large cubes
    total_pixels = H * W
    if total_pixels > max_pixels:
        pixel_idx = rng.choice(total_pixels, max_pixels, replace=False)
        rows  = pixel_idx // W
        cols  = pixel_idx % W
        X_sub = np.array(cube[rows, cols, :], dtype=np.float32)  # (max_pixels, B)
        X_sub = X_sub.T                                           # (B, max_pixels)
    else:
        X_sub = cube.reshape(-1, B).T.astype(np.float32)         # (B, N)

    # Normalize each band
    mu     = X_sub.mean(axis=1, keepdims=True)
    sigma  = X_sub.std(axis=1, keepdims=True) + 1e-8
    X_norm = (X_sub - mu) / sigma

    # Cluster B bands into k groups
    n_clusters = min(k, B)
    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300
    )
    km.fit(X_norm)

    # For each cluster select band closest to centroid
    selected = []
    for c in range(n_clusters):
        members  = np.where(km.labels_ == c)[0]
        if len(members) == 0:
            continue
        centroid = km.cluster_centers_[c]
        dists    = np.linalg.norm(X_norm[members] - centroid, axis=1)
        best     = members[np.argmin(dists)]
        selected.append(best)

    return np.array(selected)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    init_csv()

    for dataset_name, cfg in DATASETS.items():
        print(f"\n{'='*55}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*55}")

        if not os.path.exists(cfg["cube_path"]):
            print(f"  Cube not found: {cfg['cube_path']} — skipping")
            continue
        if not os.path.exists(cfg["gt_path"]):
            print(f"  GT not found: {cfg['gt_path']} — skipping")
            continue

        cube        = np.load(cfg["cube_path"], mmap_mode="r")
        gt          = np.load(cfg["gt_path"])
        num_classes = int(gt.max())
        B           = cube.shape[2]

        print(f"  Cube: {cube.shape}, classes: {num_classes}")

        for k in TOP_K_LIST:
            if k > B:
                continue

            print(f"\n  Top-{k} bands — K-Means with {k} clusters...")

            selected_bands = kmeans_band_selection(cube, k=k, random_state=0)

            if len(selected_bands) < k:
                print(f"  Only {len(selected_bands)} bands — skipping")
                continue

            print(f"  Selected: {selected_bands}")

            cube_k = cube[:, :, selected_bands]
            extra  = f"indices={selected_bands.tolist()}"

            X_all, y_all = extract_labeled_pixels(
                cube_k, gt, cfg["ignore_label"]
            )
            patches, patch_labels = extract_center_patches(
                cube_k, gt,
                patch_size=cfg["patch_size"],
                ignore_label=cfg["ignore_label"]
            )

            for run in range(NUM_RUNS):
                print(f"    Run {run+1}/{NUM_RUNS}")

                # Pixel models — RF, SVM, KNN
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_all, y_all,
                    test_size=0.3,
                    stratify=y_all,
                    random_state=run
                )
                for model_name, res in train_pixel_models(
                    X_tr, y_tr, X_val, y_val
                ).items():
                    append_row(
                        dataset_name, METHOD_NAME, "band_selection", k,
                        model_name, run, res, extra_info=extra
                    )

                # 3D-CNN
                if patches.shape[0] < 10:
                    print(f"    Too few patches — skipping 3D-CNN")
                    continue

                p_tr, p_val, yp_tr, yp_val = train_test_split(
                    patches, patch_labels,
                    test_size=0.3,
                    stratify=patch_labels,
                    random_state=run
                )

                append_row(
                    dataset_name, METHOD_NAME, "band_selection", k,
                    "3DCNN", run,
                    train_3dcnn_wrapper(p_tr, yp_tr, p_val, yp_val, num_classes),
                    extra_info=extra
                )

    print(f"\nAll done. Results saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()