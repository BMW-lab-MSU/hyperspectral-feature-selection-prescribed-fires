#!/usr/bin/env python3
# run_ablation.py
"""
Ablation study for Comment 5.
SRPA: full (lambda=0.3) vs no redundancy penalty (lambda=0)
DRL:  full sequential selection vs random band selection (same Top-K)
All 6 datasets, EDA condition, best classifier per method per dataset.
5 runs each. Results saved to ablation_results.csv.
Does NOT touch existing pipeline or results.
"""

import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split

from hsi_config import DATASETS
from hsi_data import extract_labeled_pixels, extract_center_patches
from band_selection import load_band_order, select_top_k_bands
from train_models import train_pixel_models, train_3dcnn_wrapper

# ── Best configurations from your existing Tables 2–7 ─────────────────────────
# Format: dataset -> {method -> (top_k, classifier)}
# Classifier is either "RF", "SVM", "KNN", or "3DCNN"
BEST_CONFIG = {
    "indian_pines": {
        "SRPA": (50, "RF"),
        "DRL":  (10, "3DCNN"),
    },
    "paviaU": {
        "SRPA": (30, "3DCNN"),
        "DRL":  (15, "3DCNN"),
    },
    "salinas": {
        "SRPA": (45, "RF"),
        "DRL":  (50, "RF"),
    },
    "botswana": {
        "SRPA": (15, "3DCNN"),
        "DRL":  (10, "3DCNN"),
    },
    "ksc": {
        "SRPA": (10, "3DCNN"),
        "DRL":  (10, "3DCNN"),
    },
    "montana": {
        "SRPA": (30, "3DCNN"),
        "DRL":  (15, "3DCNN"),
    },
}

NUM_RUNS   = 5
LAMBDA_REG = 0.3   # your existing SRPA lambda
ABLATION_CSV = "ablation_results.csv"

# ── CSV helpers ────────────────────────────────────────────────────────────────
def init_csv():
    if not os.path.exists(ABLATION_CSV):
        with open(ABLATION_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset", "method", "variant", "top_k",
                "classifier", "run", "OA", "kappa", "f1_macro"
            ])

def append_row(dataset, method, variant, top_k, classifier, run, OA, kappa, f1):
    with open(ABLATION_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            dataset, method, variant, top_k,
            classifier, run, OA, kappa, f1
        ])

# ── Classifier runner ──────────────────────────────────────────────────────────
def run_classifier(cube_k, gt, cfg, classifier, num_classes, run_seed):
    """
    Runs one classifier on cube_k and returns metrics dict.
    """
    if classifier in ("RF", "SVM", "KNN"):
        X_all, y_all = extract_labeled_pixels(cube_k, gt, cfg["ignore_label"])
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

# ── SRPA ablation helpers ──────────────────────────────────────────────────────
def load_srpa_components(dataset_name):
    """
    Load saved attention and redundancy arrays for SRPA.
    Returns (attn, redundancy) as numpy arrays.
    """
    ds = dataset_name.lower()
    folder = f"{ds}_srpa_band_selection_results"

    attn_path = os.path.join(folder, f"{ds}_srpa_attention.npy")
    red_path  = os.path.join(folder, f"{ds}_srpa_redundancy.npy")

    if not os.path.exists(attn_path):
        raise FileNotFoundError(f"Attention file not found: {attn_path}")
    if not os.path.exists(red_path):
        raise FileNotFoundError(f"Redundancy file not found: {red_path}")

    attn       = np.load(attn_path)
    redundancy = np.load(red_path)
    return attn, redundancy

def srpa_order_no_penalty(attn):
    """
    Recompute SRPA ranking with lambda=0 (attention only, no redundancy penalty).
    """
    return np.argsort(attn)[::-1]

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    init_csv()

    for dataset_name, cfg in DATASETS.items():
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*50}")

        # Load data
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

        best = BEST_CONFIG.get(dataset_name, {})

        # ── SRPA ablation ──────────────────────────────────────────────────────
        if "SRPA" in best:
            top_k, classifier = best["SRPA"]
            print(f"\n  [SRPA] Top-{top_k}, {classifier}")

            try:
                attn, redundancy = load_srpa_components(dataset_name)
            except FileNotFoundError as e:
                print(f"  Skipping SRPA ablation: {e}")
                attn = None

            if attn is not None:

                # Variant 1: Full SRPA (lambda=0.3) — load existing order
                try:
                    order_full = load_band_order(dataset_name, "SRPA")
                    cube_full, _ = select_top_k_bands(cube, order_full, top_k)

                    for run in range(NUM_RUNS):
                        print(f"    Full SRPA run {run+1}/{NUM_RUNS}")
                        res = run_classifier(
                            cube_full, gt, cfg, classifier,
                            num_classes, run_seed=run
                        )
                        append_row(
                            dataset_name, "SRPA", "full_lambda0.3",
                            top_k, classifier, run,
                            res["OA"], res["kappa"], res["f1_macro"]
                        )
                        print(f"      OA={res['OA']:.2f}%")
                except FileNotFoundError as e:
                    print(f"    Skipping full SRPA: {e}")

                # Variant 2: SRPA no redundancy penalty (lambda=0)
                order_no_penalty = srpa_order_no_penalty(attn)
                cube_no_pen, _   = select_top_k_bands(
                    cube, order_no_penalty, top_k
                )

                for run in range(NUM_RUNS):
                    print(f"    SRPA no-penalty run {run+1}/{NUM_RUNS}")
                    res = run_classifier(
                        cube_no_pen, gt, cfg, classifier,
                        num_classes, run_seed=run
                    )
                    append_row(
                        dataset_name, "SRPA", "no_redundancy_lambda0",
                        top_k, classifier, run,
                        res["OA"], res["kappa"], res["f1_macro"]
                    )
                    print(f"      OA={res['OA']:.2f}%")

        # ── DRL ablation ───────────────────────────────────────────────────────
        if "DRL" in best:
            top_k, classifier = best["DRL"]
            print(f"\n  [DRL] Top-{top_k}, {classifier}")

            # Variant 1: Full DRL — load existing order
            try:
                order_drl      = load_band_order(dataset_name, "DRL")
                cube_drl, _    = select_top_k_bands(cube, order_drl, top_k)

                for run in range(NUM_RUNS):
                    print(f"    Full DRL run {run+1}/{NUM_RUNS}")
                    res = run_classifier(
                        cube_drl, gt, cfg, classifier,
                        num_classes, run_seed=run
                    )
                    append_row(
                        dataset_name, "DRL", "full_sequential",
                        top_k, classifier, run,
                        res["OA"], res["kappa"], res["f1_macro"]
                    )
                    print(f"      OA={res['OA']:.2f}%")
            except FileNotFoundError as e:
                print(f"    Skipping full DRL: {e}")

            # Variant 2: Random band selection (same Top-K)
            rng = np.random.RandomState(42)
            for run in range(NUM_RUNS):
                print(f"    Random selection run {run+1}/{NUM_RUNS}")

                # Different random seed per run for variety
                random_order  = rng.permutation(B)
                cube_rand, _  = select_top_k_bands(cube, random_order, top_k)

                res = run_classifier(
                    cube_rand, gt, cfg, classifier,
                    num_classes, run_seed=run
                )
                append_row(
                    dataset_name, "DRL", "random_selection",
                    top_k, classifier, run,
                    res["OA"], res["kappa"], res["f1_macro"]
                )
                print(f"      OA={res['OA']:.2f}%")

    print(f"\nDone. Results saved to {ABLATION_CSV}")


if __name__ == "__main__":
    main()