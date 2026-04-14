#!/usr/bin/env python3
# run_full_hsi_pipeline.py

"""
Master pipeline:
- Loop over datasets
- Baseline (no band selection): RF/SVM/KNN/3DCNN/Hybrid/GCN
- Band selection (PCA, SSEP, SRPA, DRL) with Top-K list
- Logs everything to hsi_all_results.csv

You can comment out some models while debugging to save time.
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split

from hsi_config import DATASETS, TOP_K_LIST, BAND_METHODS
from hsi_data import extract_labeled_pixels, extract_center_patches
from band_selection import load_band_order, select_top_k_bands
from train_models import (
    train_pixel_models,
    train_3dcnn_wrapper,
    train_hybrid_cnn_wrapper,
    train_gcn_wrapper,
)
from logger_utils import init_results_csv, append_result_row


def log_model_result(dataset_name, method, band_mode, top_k,
                     model_name, metrics, conf_mat_path="", extra_info=""):
    """
    Convenience helper: pull all metrics from the dict and call append_result_row.
    """
    OA = metrics.get("OA")
    kappa = metrics.get("kappa")
    precision_macro = metrics.get("precision_macro")
    recall_macro = metrics.get("recall_macro")
    f1_macro = metrics.get("f1_macro")
    best_val_acc = metrics.get("best_val_acc", OA)

    append_result_row(
        dataset=dataset_name,
        method=method,
        band_mode=band_mode,
        top_k=top_k,
        model=model_name,
        OA=OA,
        kappa=kappa,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        f1_macro=f1_macro,
        best_val_acc=best_val_acc,
        conf_mat_path=conf_mat_path,
        extra_info=extra_info,
    )


def main():
    init_results_csv()

    for dataset_name, cfg in DATASETS.items():
        print("\n====================================")
        print(f"Dataset: {dataset_name}")
        print("====================================")

        cube_path = cfg["cube_path"]
        gt_path = cfg["gt_path"]

        if not os.path.exists(cube_path):
            print(f"  WARNING: Cube path not found: {cube_path} (skipping dataset)")
            continue
        if not os.path.exists(gt_path):
            print(f"  WARNING: GT path not found: {gt_path} (skipping dataset)")
            continue

        cube = np.load(cube_path, mmap_mode="r")  # (H, W, B)
        gt = np.load(gt_path)                     # (H, W)
        num_classes = int(gt.max())
        H, W, B = cube.shape
        print(f"Cube: {cube.shape}, GT: {gt.shape}, num_classes(from GT)={num_classes}")

        # ----------------------------------------------------
        # 1) Baseline — ALL bands (no band selection)
        # ----------------------------------------------------
        method = "none"
        band_mode = "no_selection"
        top_k = B

        print(f"\n[Baseline] {dataset_name} using ALL {B} bands")

        # Pixel-based data
        X_all, y_all = extract_labeled_pixels(cube, gt, cfg["ignore_label"])
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_all, y_all, test_size=0.3, stratify=y_all, random_state=0
        )

        # Train RF/SVM/KNN
        pixel_results = train_pixel_models(X_tr, y_tr, X_val, y_val)
        for model_name, res in pixel_results.items():
            log_model_result(
                dataset_name, method, band_mode, top_k,
                model_name, res,
                conf_mat_path="", extra_info="baseline_all_bands",
            )

        # 3D CNN & Hybrid CNN use patches
        patches, labels = extract_center_patches(
            cube, gt, patch_size=cfg["patch_size"], ignore_label=cfg["ignore_label"]
        )
        p_tr, p_val, y_p_tr, y_p_val = train_test_split(
            patches, labels, test_size=0.3, stratify=labels, random_state=0
        )

        print(f"[DEBUG] y_p_tr min/max: {y_p_tr.min()} / {y_p_tr.max()}, num_classes={num_classes}")

        # 3D CNN
        cnn_res = train_3dcnn_wrapper(p_tr, y_p_tr, p_val, y_p_val, num_classes)
        log_model_result(
            dataset_name, method, band_mode, top_k,
            "3DCNN", cnn_res,
            conf_mat_path="", extra_info="baseline_all_bands",
        )

        # Hybrid CNN
        hres = train_hybrid_cnn_wrapper(p_tr, y_p_tr, p_val, y_p_val, num_classes)
        log_model_result(
            dataset_name, method, band_mode, top_k,
            "HybridCNN", hres,
            conf_mat_path="", extra_info="baseline_all_bands",
        )

        # # GCN on full labeled pixels
        # gcn_res = train_gcn_wrapper(X_all, y_all, num_classes)
        # log_model_result(
        #     dataset_name, method, band_mode, top_k,
        #     "GCN", gcn_res,
        #     conf_mat_path="", extra_info="baseline_all_bands",
        # )

        # ----------------------------------------------------
        # 2) Band Selection (PCA, SSEP, SRPA, DRL)
        # ----------------------------------------------------
        for method in BAND_METHODS:
            print(f"\n[{dataset_name}] Method = {method}")
            try:
                band_order = load_band_order(dataset_name, method)
            except FileNotFoundError as e:
                print("  ", e)
                continue

            for k in TOP_K_LIST:
                if k > len(band_order):
                    continue

                print(f"  Top-{k} bands ...")

                cube_k, selected_indices = select_top_k_bands(cube, band_order, k)

                # --- pixel models ---
                X_all_k, y_all_k = extract_labeled_pixels(
                    cube_k, gt, cfg["ignore_label"]
                )
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_all_k, y_all_k, test_size=0.3,
                    stratify=y_all_k, random_state=0
                )

                pixel_results = train_pixel_models(X_tr, y_tr, X_val, y_val)
                extra = f"indices={selected_indices.tolist()}"

                for model_name, res in pixel_results.items():
                    log_model_result(
                        dataset_name, method, "band_selection", k,
                        model_name, res,
                        conf_mat_path="", extra_info=extra,
                    )

                # --- patch-based deep models ---
                patches_k, labels_k = extract_center_patches(
                    cube_k, gt,
                    patch_size=cfg["patch_size"],
                    ignore_label=cfg["ignore_label"]
                )
                if patches_k.shape[0] < 10:
                    print("  Too few patches after band selection; skipping deep models.")
                    continue

                p_tr, p_val, y_p_tr, y_p_val = train_test_split(
                    patches_k, labels_k, test_size=0.3,
                    stratify=labels_k, random_state=0
                )

                # 3D CNN
                cnn_res = train_3dcnn_wrapper(p_tr, y_p_tr, p_val, y_p_val, num_classes)
                log_model_result(
                    dataset_name, method, "band_selection", k,
                    "3DCNN", cnn_res,
                    conf_mat_path="", extra_info=extra,
                )

                # Hybrid CNN
                hres = train_hybrid_cnn_wrapper(p_tr, y_p_tr, p_val, y_p_val, num_classes)
                log_model_result(
                    dataset_name, method, "band_selection", k,
                    "HybridCNN", hres,
                    conf_mat_path="", extra_info=extra,
                )

                # GCN on pixel features with selected bands
    #             gcn_res = train_gcn_wrapper(X_all_k, y_all_k, num_classes)
    #             log_model_result(
    #                 dataset_name, method, "band_selection", k,
    #                 "GCN", gcn_res,
    #                 conf_mat_path="", extra_info=extra,
    #             )

    print("\nAll runs finished. Results saved in hsi_all_results.csv")


if __name__ == "__main__":
    main()
