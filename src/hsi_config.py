#!/usr/bin/env python3
# src/hsi_config.py
"""
Global configuration for all datasets and experiments.

USAGE
-----
Edit the cube_path and gt_path values to match your local file structure.
All paths are relative to the project root directory.

DATA FORMAT
-----------
Each dataset requires two .npy files:
  - cube_path : (H, W, B) float32 hyperspectral cube, normalised to [0,1]
  - gt_path   : (H, W) int16 ground truth map (0 = background, 1..C = classes)

These are produced by the EDA scripts in src/eda/.

NOTE: paviaU uses capital U in all file and folder names throughout this project.
"""

# ── Dataset configurations ─────────────────────────────────────────────────────

DATASETS = {
    "indian_pines": {
        "cube_path":   "data/processed/indian_pines_clean.npy",
        "gt_path":     "data/processed/indian_pines_gt.npy",
        "num_classes": 16,
        "ignore_label": 0,
        "patch_size":  5,
        "noisy_bands": [102, 103, 104, 143, 144, 145, 195, 197, 198, 199],
        "description": "AVIRIS sensor, 145x145, 200 bands (after noisy removal), 16 classes",
    },
    "paviaU": {
        "cube_path":   "data/processed/paviaU_clean.npy",
        "gt_path":     "data/processed/paviaU_gt.npy",
        "num_classes": 9,
        "ignore_label": 0,
        "patch_size":  5,
        "noisy_bands": [0, 1, 2, 3, 4, 5],
        "description": "ROSIS sensor, 610x340, 103 bands, 9 classes",
    },
    "salinas": {
        "cube_path":   "data/processed/salinas_clean.npy",
        "gt_path":     "data/processed/salinas_gt.npy",
        "num_classes": 16,
        "ignore_label": 0,
        "patch_size":  5,
        "noisy_bands": [106, 107, 108, 109, 146, 147, 148, 198, 201, 202, 203],
        "description": "AVIRIS sensor, 512x217, 204 bands (after noisy removal), 16 classes",
    },
    "botswana": {
        "cube_path":   "data/processed/botswana_clean.npy",
        "gt_path":     "data/processed/botswana_gt.npy",
        "num_classes": 14,
        "ignore_label": 0,
        "patch_size":  5,
        "noisy_bands": [111, 138, 139, 140, 141, 142, 143, 144],
        "description": "Hyperion sensor, 1476x256, 145 bands (after noisy removal), 14 classes",
    },
    "ksc": {
        "cube_path":   "data/processed/ksc_clean.npy",
        "gt_path":     "data/processed/ksc_gt.npy",
        "num_classes": 13,
        "ignore_label": 0,
        "patch_size":  5,
        "noisy_bands": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "description": "AVIRIS sensor, 512x614, 176 bands (after noisy removal), 13 classes",
    },
    "montana": {
        "cube_path":   "data/processed/montana_vnir_clean.npy",
        "gt_path":     "data/processed/montana_gt.npy",
        "num_classes": 5,
        "ignore_label": 0,
        "patch_size":  5,
        "noisy_bands": [],
        "description": (
            "UAV VNIR sensor, Lubrecht Experimental Forest MT, "
            "211 bands, 5 classes: Dead Biomass, Dead Ponderosa Pine, "
            "Douglas Fir, Ponderosa Pine, Western Larch. "
            "NSF SMART FIRES project — available upon reasonable request."
        ),
    },
}

# ── Experiment parameters ──────────────────────────────────────────────────────

# Top-K band subset sizes to evaluate
TOP_K_LIST = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# Band selection methods
BAND_METHODS = ["PCA", "SSEP", "SRPA", "DRL", "KMCBS"]

# Classifiers
CLASSIFIERS = ["RF", "SVM", "KNN", "3DCNN", "HybridCNN"]

# Patch sizes for sensitivity analysis
PATCH_SIZE_LIST = [3, 5, 7]

# Random seed for reproducibility
RANDOM_SEED = 0

# Train / validation split ratio (test_size)
VAL_SIZE = 0.3
