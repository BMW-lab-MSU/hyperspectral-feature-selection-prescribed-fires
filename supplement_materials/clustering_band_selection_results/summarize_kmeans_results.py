#!/usr/bin/env python3
# summarize_kmeans_results.py
"""
Reads kmeans_band_selection_results.csv and prints:
1. Best config per dataset per classifier
2. Full table matching existing Tables 2-7 format
"""

import csv
from collections import defaultdict
import numpy as np

RESULTS_CSV = "kmeans_band_selection_results.csv"

datasets   = ["indian_pines", "paviaU", "salinas", "botswana", "ksc", "montana"]
models     = ["RF", "SVM", "KNN", "3DCNN"]
top_k_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# ── Load and deduplicate ────────────────────────────────────────────────────────
data = defaultdict(lambda: {"OA": [], "kappa": [], "f1_macro": [],
                             "precision_macro": [], "recall_macro": []})
seen = defaultdict(int)

with open(RESULTS_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row["dataset"], row["model"], int(row["top_k"]))
        run = int(row["run"])
        if run < 5 and seen[key] < 5:
            data[key]["OA"].append(float(row["OA"]))
            data[key]["kappa"].append(float(row["kappa"]))
            data[key]["f1_macro"].append(float(row["f1_macro"]))
            data[key]["precision_macro"].append(float(row["precision_macro"]))
            data[key]["recall_macro"].append(float(row["recall_macro"]))
            seen[key] += 1

# ── Best config per dataset per classifier ──────────────────────────────────────
print("=" * 80)
print("KMCBS — BEST CONFIGURATION PER DATASET PER CLASSIFIER")
print("(Best Top-K selected based on highest mean OA)")
print("=" * 80)
print(f"\n{'Dataset':<18} {'Clf':<8} {'K':>5} {'OA mean':>10} {'OA std':>8} "
      f"{'Kappa':>8} {'F1':>8} {'Prec':>8} {'Rec':>8}")
print("-" * 90)

best_configs = {}
for ds in datasets:
    best_configs[ds] = {}
    first = True
    for model in models:
        best_oa  = -1
        best_key = None
        for k in top_k_list:
            key = (ds, model, k)
            if not data[key]["OA"]:
                continue
            oa = np.mean(data[key]["OA"])
            if oa > best_oa:
                best_oa  = oa
                best_key = key
        if best_key:
            oa_arr = np.array(data[best_key]["OA"])
            kappa  = np.mean(data[best_key]["kappa"])
            f1     = np.mean(data[best_key]["f1_macro"])
            prec   = np.mean(data[best_key]["precision_macro"])
            rec    = np.mean(data[best_key]["recall_macro"])
            best_configs[ds][model] = {
                "k": best_key[2], "OA": oa_arr.mean(), "std": oa_arr.std(),
                "kappa": kappa, "f1": f1, "prec": prec, "rec": rec
            }
            ds_label = ds if first else ""
            first = False
            print(f"  {ds_label:<18} {model:<8} {best_key[2]:>5} "
                  f"{oa_arr.mean():>10.2f} {oa_arr.std():>8.2f} "
                  f"{kappa:>8.3f} {f1:>8.2f} {prec:>8.2f} {rec:>8.2f}")
    print()

# ── Overall best per dataset ────────────────────────────────────────────────────
print("=" * 80)
print("KMCBS — OVERALL BEST PER DATASET")
print("=" * 80)
print(f"\n{'Dataset':<18} {'Best Clf':<8} {'K':>5} {'OA':>10} {'Std':>8} "
      f"{'Kappa':>8} {'F1':>8}")
print("-" * 70)
for ds in datasets:
    best_oa = -1
    best    = None
    for model in models:
        if model not in best_configs[ds]:
            continue
        if best_configs[ds][model]["OA"] > best_oa:
            best_oa = best_configs[ds][model]["OA"]
            best    = (model, best_configs[ds][model])
    if best:
        m, cfg = best
        print(f"  {ds:<18} {m:<8} {cfg['k']:>5} {cfg['OA']:>10.2f} "
              f"{cfg['std']:>8.2f} {cfg['kappa']:>8.3f} {cfg['f1']:>8.2f}")

# ── OA vs Top-K curve per dataset ───────────────────────────────────────────────
print("\n" + "=" * 80)
print("KMCBS — OA vs TOP-K (3D-CNN only)")
print("=" * 80)
print(f"\n{'Dataset':<18}", end="")
for k in top_k_list:
    print(f" {f'K={k}':>8}", end="")
print()
print("-" * 100)
for ds in datasets:
    print(f"  {ds:<18}", end="")
    for k in top_k_list:
        key = (ds, "3DCNN", k)
        if data[key]["OA"]:
            print(f" {np.mean(data[key]['OA']):>8.2f}", end="")
        else:
            print(f" {'N/A':>8}", end="")
    print()