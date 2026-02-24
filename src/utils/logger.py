#!/usr/bin/env python3
# logger_utils.py

"""
Central CSV logger for all HSI experiments.
"""

import csv
import os

# RESULTS_CSV = "hsi_all_results.csv"
RESULTS_CSV = os.path.join("outputs", "aggregated", "all_runs.csv")


def init_results_csv():
    """
    Creates the CSV file with header if it does not exist.
    """
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset",
                "method",           # "none", "PCA", "SSEP", "SRPA", "DRL"
                "band_mode",        # "no_selection" or "band_selection"
                "top_k",            # integer (or full B)
                "model",            # "RF", "SVM", "KNN", "3DCNN", ...
                "OA",               # Overall Accuracy (%)
                "kappa",            # Cohen's kappa
                "precision_macro",  # macro-averaged precision (%)
                "recall_macro",     # macro-averaged recall (%)
                "f1_macro",         # macro-averaged F1 (%)
                "best_val_acc",     # best validation accuracy during training (%)
                "conf_mat_path",    # optional: path to confusion matrix file
                "extra_info",       # text: e.g. indices=[...]
            ])


def append_result_row(dataset, method, band_mode, top_k,
                      model,
                      OA, kappa,
                      precision_macro, recall_macro, f1_macro,
                      best_val_acc,
                      conf_mat_path="", extra_info=""):
    """
    Appends a single row to the CSV.

    All metric arguments can be floats or None. We don't cast to float so that
    'None' can be written directly if something is missing.
    """
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            dataset,
            method,
            band_mode,
            int(top_k) if top_k is not None else "",
            model,
            OA,
            kappa,
            precision_macro,
            recall_macro,
            f1_macro,
            best_val_acc,
            conf_mat_path,
            extra_info,
        ])
