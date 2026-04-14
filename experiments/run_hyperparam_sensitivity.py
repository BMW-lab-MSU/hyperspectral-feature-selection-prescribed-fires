#!/usr/bin/env python3
# run_hyperparam_sensitivity.py
"""
Hyperparameter sensitivity analysis for Comment 1.

SSEP: vary sigma in {0.5, 1.0, 1.5, 2.0}
      Re-runs band scoring for each sigma value.
      sigma=1.0 is your existing value.

SRPA: vary lambda in {0.1, 0.2, 0.3, 0.5, 0.7}
      Recomputes ranking from saved attention + redundancy arrays.
      lambda=0.3 is your existing value.

All 6 datasets, EDA condition only.
Best classifier per method per dataset from existing Tables 2-7.
5 runs each to match existing protocol.
Results saved to hyperparam_sensitivity_results.csv.
Does NOT touch existing pipeline or results.
"""

import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.filters import sobel
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from hsi_config import DATASETS
from hsi_data import extract_labeled_pixels, extract_center_patches
from band_selection import select_top_k_bands
from train_models import train_pixel_models, train_3dcnn_wrapper

# ── Best configurations from your existing Tables 2–7 ─────────────────────────
# Format: dataset -> {method -> (top_k, classifier)}
BEST_CONFIG = {
    "indian_pines": {
        "SSEP": (50, "RF"),
        "SRPA": (50, "RF"),
    },
    "paviaU": {
        "SSEP": (30, "3DCNN"),
        "SRPA": (30, "3DCNN"),
    },
    "salinas": {
        "SSEP": (50, "RF"),
        "SRPA": (45, "RF"),
    },
    "botswana": {
        "SSEP": (40, "SVM"),
        "SRPA": (15, "3DCNN"),
    },
    "ksc": {
        "SSEP": (45, "RF"),
        "SRPA": (10, "3DCNN"),
    },
    "montana": {
        "SSEP": (20, "3DCNN"),
        "SRPA": (30, "3DCNN"),
    },
}

# ── Hyperparameter values to test ─────────────────────────────────────────────
SSEP_SIGMAS  = [0.5, 1.0, 1.5, 2.0]   # 1.0 is your existing value
SRPA_LAMBDAS = [0.1, 0.2, 0.3, 0.5, 0.7]  # 0.3 is your existing value

NUM_RUNS = 5
RESULTS_CSV = "hyperparam_sensitivity_results.csv"

# ── CSV helpers ────────────────────────────────────────────────────────────────
def init_csv():
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset", "method", "param_name", "param_value",
                "top_k", "classifier", "run",
                "OA", "kappa", "f1_macro"
            ])

def append_row(dataset, method, param_name, param_value,
               top_k, classifier, run, OA, kappa, f1):
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            dataset, method, param_name, param_value,
            top_k, classifier, run, OA, kappa, f1
        ])

# ── Classifier runner ──────────────────────────────────────────────────────────
def run_classifier(cube_k, gt, cfg, classifier, num_classes, run_seed):
    if classifier in ("RF", "SVM", "KNN"):
        X_all, y_all = extract_labeled_pixels(
            cube_k, gt, cfg["ignore_label"]
        )
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_all, y_all,
            test_size=0.3,
            stratify=y_all,
            random_state=run_seed
        )
        results = train_pixel_models(X_tr, y_tr, X_val, y_val)
        return results[classifier]

    elif classifier == "3DCNN":
        patches, labels = extract_center_patches(
            cube_k, gt,
            patch_size=cfg["patch_size"],
            ignore_label=cfg["ignore_label"]
        )
        p_tr, p_val, y_tr, y_val = train_test_split(
            patches, labels,
            test_size=0.3,
            stratify=labels,
            random_state=run_seed
        )
        return train_3dcnn_wrapper(p_tr, y_tr, p_val, y_val, num_classes)

    else:
        raise ValueError(f"Unknown classifier: {classifier}")

# ── SSEP band scoring with variable sigma ─────────────────────────────────────
def compute_mask_edge(gt):
    """Compute binary edge map from GT using Sobel."""
    edge = sobel(gt.astype(float))
    return (edge > 0).astype(np.uint8)

def ssep_score_bands(cube, gt, sigma):
    """
    Re-run SSEP band scoring with a given sigma.
    Returns band_scores (B,) and order (B,) sorted best to worst.
    """
    H, W, B = cube.shape
    mask_edge = compute_mask_edge(gt)
    band_scores = np.zeros(B, dtype=np.float32)

    for i in tqdm(range(B), desc=f"  SSEP sigma={sigma}", leave=False):
        band_img    = cube[:, :, i].astype(np.float32)
        smoothed    = gaussian_filter(band_img, sigma=sigma)
        band_edge   = sobel(smoothed)
        thr         = np.percentile(band_edge, 95)
        band_edge_b = (band_edge > thr).astype(np.uint8)

        # Dice coefficient
        intersection = np.sum(band_edge_b * mask_edge)
        total        = np.sum(band_edge_b) + np.sum(mask_edge)
        band_scores[i] = (2.0 * intersection / total) if total > 0 else 0.0

    order = np.argsort(band_scores)[::-1]
    return band_scores, order

# ── SRPA ranking with variable lambda ─────────────────────────────────────────
def load_srpa_components(dataset_name):
    """Load saved attention and redundancy arrays."""
    ds     = dataset_name.lower()
    folder = f"{ds}_srpa_band_selection_results"
    attn   = np.load(os.path.join(folder, f"{ds}_srpa_attention.npy"))
    red    = np.load(os.path.join(folder, f"{ds}_srpa_redundancy.npy"))
    return attn, red

def srpa_order_with_lambda(attn, redundancy, lam):
    """Recompute SRPA ranking: scores = attn - lam * redundancy."""
    scores = attn - lam * redundancy
    return np.argsort(scores)[::-1]

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
        print(f"  Cube: {cube.shape}, classes: {num_classes}")

        best = BEST_CONFIG.get(dataset_name, {})

        # ── SSEP sigma sensitivity ─────────────────────────────────────────────
        if "SSEP" in best:
            top_k, classifier = best["SSEP"]
            print(f"\n  [SSEP] Top-{top_k}, {classifier}")

            for sigma in SSEP_SIGMAS:
                print(f"\n    sigma = {sigma}")

                # Re-run SSEP band scoring with this sigma
                _, order = ssep_score_bands(cube, gt, sigma=sigma)
                cube_k, _ = select_top_k_bands(cube, order, top_k)

                for run in range(NUM_RUNS):
                    res = run_classifier(
                        cube_k, gt, cfg, classifier,
                        num_classes, run_seed=run
                    )
                    print(f"      Run {run+1}: OA={res['OA']:.2f}%")
                    append_row(
                        dataset_name, "SSEP", "sigma", sigma,
                        top_k, classifier, run,
                        res["OA"], res["kappa"], res["f1_macro"]
                    )

        # ── SRPA lambda sensitivity ────────────────────────────────────────────
        if "SRPA" in best:
            top_k, classifier = best["SRPA"]
            print(f"\n  [SRPA] Top-{top_k}, {classifier}")

            try:
                attn, redundancy = load_srpa_components(dataset_name)
            except FileNotFoundError as e:
                print(f"    Skipping SRPA sensitivity: {e}")
                continue

            for lam in SRPA_LAMBDAS:
                print(f"\n    lambda = {lam}")

                order  = srpa_order_with_lambda(attn, redundancy, lam)
                cube_k, _ = select_top_k_bands(cube, order, top_k)

                for run in range(NUM_RUNS):
                    res = run_classifier(
                        cube_k, gt, cfg, classifier,
                        num_classes, run_seed=run
                    )
                    print(f"      Run {run+1}: OA={res['OA']:.2f}%")
                    append_row(
                        dataset_name, "SRPA", "lambda", lam,
                        top_k, classifier, run,
                        res["OA"], res["kappa"], res["f1_macro"]
                    )

    print(f"\nDone. Results saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()