#!/usr/bin/env python3
# hsi_config.py

"""
Global config for all HSI datasets and experiments.
Edit paths to match your local structure.
"""

DATASETS = {
    "indian_pines": {
        "cube_path": "indian_pines_clean.npy",
        "gt_path":   "Indian_pines_gt.npy",
        "num_classes": 16,         # excluding background
        "ignore_label": 0,
        "patch_size": 5,
    },
    # "paviaU": {
    #     "cube_path": "paviaU_clean.npy",
    #     "gt_path":   "paviaU_gt.npy",
    #     "num_classes": 9,
    #     "ignore_label": 0,
    #     "patch_size": 5,
    # },
    # "salinas": {
    #     "cube_path": "salinas_clean.npy",
    #     "gt_path":   "salinas_gt.npy",
    #     "num_classes": 16,
    #     "ignore_label": 0,
    #     "patch_size": 5,
    # },
    # "botswana": {
    #     "cube_path": "botswana_clean.npy",
    #     "gt_path":   "botswana_gt.npy",
    #     "num_classes": 14,
    #     "ignore_label": 0,
    #     "patch_size": 5,
    # },
    # "ksc": {
    #     "cube_path": "ksc_clean.npy",
    #     "gt_path":   "ksc_gt.npy",
    #     "num_classes": 13,
    #     "ignore_label": 0,
    #     "patch_size": 5,
    # },
    "montana": {
        "cube_path": "montana_vnir_clean_n.npy",
        "gt_path":   "montana_gt.npy",
        "num_classes": 6,
        "ignore_label": 0,
        "patch_size": 5,
    },
}

# Top-K values you want to loop over for band-selection methods
TOP_K_LIST = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
#TOP_K_LIST = [5]

# List of band-selection methods (string names you’ll use)
BAND_METHODS = ["none", "PCA", "SSEP", "SRPA", "DRL"]

# Which models to run; you can comment some out while debugging
CLASSIFIERS = ["RF", "SVM", "KNN", "3DCNN", "HybridCNN", "GCN"]
