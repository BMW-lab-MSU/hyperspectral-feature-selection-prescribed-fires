#!/usr/bin/env python3
# src/utils/logger_utils.py
"""
Central CSV logger for all HSI pipeline experiments.

All experiment scripts write results to a single CSV file so that
results can be aggregated and analysed without re-running experiments.
"""

import csv
import os

RESULTS_CSV = "outputs/results/hsi_all_results.csv"


def init_results_csv(path: str = RESULTS_CSV):
    """
    Create the results CSV with header row if it does not already exist.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset",
                "method",           # "none", "PCA", "SSEP", "SRPA", "DRL", "KMCBS"
                "band_mode",        # "no_selection" or "band_selection"
                "top_k",            # integer (or full B for no_selection)
                "model",            # "RF", "SVM", "KNN", "3DCNN", "HybridCNN"
                "run",              # run index 0..4
                "OA",               # Overall Accuracy (%)
                "kappa",            # Cohen's Kappa
                "precision_macro",  # macro-averaged precision (%)
                "recall_macro",     # macro-averaged recall (%)
                "f1_macro",         # macro-averaged F1 (%)
                "best_val_acc",     # best validation accuracy during training
                "extra_info",       # e.g. selected band indices
            ])


def append_result_row(dataset, method, band_mode, top_k, model, run,
                      OA, kappa, precision_macro, recall_macro, f1_macro,
                      best_val_acc, extra_info="",
                      path: str = RESULTS_CSV):
    """
    Append a single result row to the CSV.
    """
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            dataset, method, band_mode,
            int(top_k) if top_k is not None else "",
            model, run,
            OA, kappa, precision_macro, recall_macro, f1_macro,
            best_val_acc, extra_info,
        ])
