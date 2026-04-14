#!/usr/bin/env python3
# run_patch_sensitivity.py

import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from hsi_config import DATASETS
from hsi_data import extract_center_patches
from band_selection import load_band_order, select_top_k_bands
from models_deep import Simple3DCNN_PatchSensitivity, compute_classification_metrics
from torch.utils.data import TensorDataset, DataLoader

# ── experiment scope ───────────────────────────────────────
TARGET_DATASETS = ["indian_pines", "paviaU", "montana"]

BEST_DRL_TOPK = {
    "indian_pines": 10,
    "paviaU":       15,
    "montana":      15,
}

PATCH_SIZES = [3, 5, 7]
NUM_RUNS    = 5
PATCH_CSV   = "patch_sensitivity_results.csv"

# ── CSV helpers ────────────────────────────────────────────
def init_patch_csv():
    if not os.path.exists(PATCH_CSV):
        with open(PATCH_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset", "patch_size", "top_k", "run",
                "OA", "kappa", "f1_macro"
            ])

def append_patch_row(dataset, patch_size, top_k, run, OA, kappa, f1):
    with open(PATCH_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([dataset, patch_size, top_k, run, OA, kappa, f1])

# ── model trainer ──────────────────────────────────────────
def train_3dcnn_patch_experiment(patches_train, y_train,
                                  patches_val, y_val,
                                  num_classes,
                                  num_epochs=20,
                                  batch_size=32,
                                  lr=1e-3,
                                  device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  3D-CNN device: {device}")

    def to_tensor(patches):
        t = torch.from_numpy(patches.astype(np.float32))
        t = t.permute(0, 3, 1, 2).unsqueeze(1)  # (N,1,B,ps,ps)
        return t

    Xtr  = to_tensor(patches_train)
    Xval = to_tensor(patches_val)
    ytr  = torch.from_numpy(y_train.astype(np.int64))
    yv   = torch.from_numpy(y_val.astype(np.int64))

    model     = Simple3DCNN_PatchSensitivity(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(Xtr, ytr),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xval, yv),
                              batch_size=batch_size, shuffle=False)

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for bx, by in val_loader:
                preds    = model(bx.to(device)).argmax(1)
                correct += (preds == by.to(device)).sum().item()
                total   += by.size(0)
        val_acc = 100.0 * correct / max(1, total)
        print(f"    Epoch {epoch:02d} | Val Acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = model.state_dict()

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for bx, by in val_loader:
            all_preds.append(model(bx.to(device)).argmax(1).cpu().numpy())
            all_true.append(by.numpy())

    metrics = compute_classification_metrics(
        np.concatenate(all_true),
        np.concatenate(all_preds)
    )
    metrics["best_val_acc"] = best_val_acc
    return metrics


# ── main loop ──────────────────────────────────────────────
def main():
    init_patch_csv()

    for dataset_name in TARGET_DATASETS:
        cfg   = DATASETS[dataset_name]
        top_k = BEST_DRL_TOPK[dataset_name]

        print(f"\n=== {dataset_name} | DRL Top-{top_k} ===")

        cube = np.load(cfg["cube_path"], mmap_mode="r")
        gt   = np.load(cfg["gt_path"])
        num_classes = int(gt.max())

        try:
            band_order = load_band_order(dataset_name, "DRL")
        except FileNotFoundError as e:
            print(f"  Skipping {dataset_name}: {e}")
            continue

        cube_k, _ = select_top_k_bands(cube, band_order, top_k)

        for patch_size in PATCH_SIZES:
            print(f"\n  Patch size: {patch_size}x{patch_size}")

            patches, labels = extract_center_patches(
                cube_k, gt,
                patch_size=patch_size,
                ignore_label=cfg["ignore_label"]
            )

            if patches.shape[0] < 20:
                print(f"  Too few patches, skipping.")
                continue

            for run in range(NUM_RUNS):
                print(f"  Run {run+1}/{NUM_RUNS}")

                p_tr, p_val, y_tr, y_val = train_test_split(
                    patches, labels,
                    test_size=0.3,
                    stratify=labels,
                    random_state=run
                )

                res = train_3dcnn_patch_experiment(
                    p_tr, y_tr, p_val, y_val,
                    num_classes=num_classes
                )

                print(f"    OA={res['OA']:.2f}%  Kappa={res['kappa']:.3f}  F1={res['f1_macro']:.2f}%")

                append_patch_row(
                    dataset_name, patch_size, top_k, run,
                    res["OA"], res["kappa"], res["f1_macro"]
                )

    print(f"\nDone. Results saved to {PATCH_CSV}")


if __name__ == "__main__":
    main()