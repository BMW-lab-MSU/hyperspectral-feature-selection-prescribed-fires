#!/usr/bin/env python3
# run_class_imbalance.py
"""
Class imbalance analysis for Comment 2.
Montana UAV VNIR dataset only.
DRL Top-15 bands, EDA condition.

Strategies compared:
1. Baseline          — no imbalance handling
2. Class weighting   — weighted loss (RF + 3D-CNN)
3. SMOTE             — synthetic oversampling (RF only)
4. Focal loss        — hard example mining (3D-CNN only)

Reports OA, Kappa, Macro-F1, and per-class F1 for all 5 classes.
5 independent runs each.
Results saved to class_imbalance_results.csv
"""

import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from imblearn.over_sampling import SMOTE

from hsi_config import DATASETS
from hsi_data import extract_labeled_pixels, extract_center_patches
from band_selection import select_top_k_bands
from models_deep import Simple3DCNN

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_NAME = "montana"
TOP_K        = 15
NUM_RUNS     = 5
NUM_EPOCHS   = 20
BATCH_SIZE   = 32
LR           = 1e-3
RESULTS_CSV  = "class_imbalance_results.csv"

CLASS_NAMES = [
    "Dead_Biomass",
    "Dead_Ponderosa_Pine",
    "Douglas_Fir",
    "Ponderosa_Pine",
    "Western_Larch",
     "Unknown_Class_6", 
]

# ── Focal Loss ─────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            reduction="none"
        )
        pt   = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean()

# ── CSV helpers ────────────────────────────────────────────────────────────────
def init_csv():
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "strategy", "classifier", "run",
                "OA", "kappa", "f1_macro",
            ] + [f"f1_{c}" for c in CLASS_NAMES])

def append_row(strategy, classifier, run, OA, kappa, f1_macro, per_class_f1):
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            strategy, classifier, run,
            OA, kappa, f1_macro,
        ] + list(per_class_f1))

# ── Metrics ────────────────────────────────────────────────────────────────────
def get_metrics(y_true, y_pred, num_classes):
    OA        = accuracy_score(y_true, y_pred) * 100.0
    kappa     = cohen_kappa_score(y_true, y_pred)
    f1_macro  = f1_score(y_true, y_pred, average="macro",
                         zero_division=0) * 100.0
    per_class = f1_score(y_true, y_pred, average=None,
                         zero_division=0,
                         labels=list(range(num_classes))) * 100.0
    return OA, kappa, f1_macro, per_class

# ── Class weights helper ───────────────────────────────────────────────────────
def compute_class_weights(y_train, num_classes):
    counts  = np.bincount(y_train, minlength=num_classes).astype(float)
    counts  = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return weights

# ── RF trainer ─────────────────────────────────────────────────────────────────
def train_rf(X_tr, y_tr, X_val, y_val, num_classes, class_weight=None):
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=0
    )
    clf.fit(X_tr, y_tr)
    return get_metrics(y_val, clf.predict(X_val), num_classes)

# ── 3D-CNN trainer ─────────────────────────────────────────────────────────────
def train_3dcnn(p_tr, y_tr, p_val, y_val, num_classes,
                loss_fn_name="ce", class_weights_np=None):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def to_tensor(patches):
        t = torch.from_numpy(patches.astype(np.float32))
        return t.permute(0, 3, 1, 2).unsqueeze(1)

    Xtr  = to_tensor(p_tr)
    Xval = to_tensor(p_val)
    ytr  = torch.from_numpy(y_tr.astype(np.int64))
    yv   = torch.from_numpy(y_val.astype(np.int64))

    model     = Simple3DCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    w = torch.tensor(class_weights_np, dtype=torch.float32).to(device) \
        if class_weights_np is not None else None

    criterion = FocalLoss(gamma=2.0, weight=w) \
        if loss_fn_name == "focal" else nn.CrossEntropyLoss(weight=w)

    train_loader = DataLoader(TensorDataset(Xtr, ytr),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xval, yv),
                              batch_size=BATCH_SIZE, shuffle=False)

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for bx, by in val_loader:
                preds    = model(bx.to(device)).argmax(1)
                correct += (preds == by.to(device)).sum().item()
                total   += by.size(0)
        val_acc = 100.0 * correct / max(1, total)
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

    return get_metrics(
        np.concatenate(all_true),
        np.concatenate(all_preds),
        num_classes
    )

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    init_csv()

    cfg         = DATASETS[DATASET_NAME]
    cube        = np.load(cfg["cube_path"], mmap_mode="r")
    gt          = np.load(cfg["gt_path"])
    num_classes = int(gt[gt > 0].max())

    print(f"Dataset: {DATASET_NAME}")
    print(f"Cube: {cube.shape}, Actual num_classes: {num_classes}")

    order     = np.load("montana_drl_bandselection_results/montana_drl_order.npy")
    cube_k, _ = select_top_k_bands(cube, order, TOP_K)
    print(f"Selected {TOP_K} bands using DRL")

    X_all, y_all = extract_labeled_pixels(cube_k, gt, cfg["ignore_label"])
    patches, labels = extract_center_patches(
        cube_k, gt,
        patch_size=cfg["patch_size"],
        ignore_label=cfg["ignore_label"]
    )

    # Fix num_classes from actual data
    num_classes = int(max(y_all.max(), labels.max())) + 1
    print(f"num_classes from data: {num_classes}")
    print(f"y_all range: {y_all.min()} to {y_all.max()}")
    print(f"labels range: {labels.min()} to {labels.max()}")

    # One-time debug split to check SMOTE viability
    X_tr_debug, _, y_tr_debug, _ = train_test_split(
        X_all, y_all, test_size=0.3, stratify=y_all, random_state=0
    )
    min_count   = np.bincount(y_tr_debug).min()
    k_neighbors = min(5, min_count - 1)
    print(f"\nDebug split — Training class counts: {np.bincount(y_tr_debug)}")
    print(f"Min count: {min_count}")
    print(f"k_neighbors for SMOTE: {k_neighbors}")
    print(f"SMOTE viable: {'YES' if k_neighbors >= 1 else 'NO — will skip'}\n")

    for run in range(NUM_RUNS):
        print(f"\n{'='*50}")
        print(f"Run {run+1}/{NUM_RUNS}")
        print(f"{'='*50}")

        # Pixel split for RF
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_all, y_all,
            test_size=0.3, stratify=y_all, random_state=run
        )

        # Patch split for 3D-CNN
        p_tr, p_val, yp_tr, yp_val = train_test_split(
            patches, labels,
            test_size=0.3, stratify=labels, random_state=run
        )

        cw_np = compute_class_weights(y_tr, num_classes)

        # ── 1. Baseline RF ─────────────────────────────────────────────────
        print("\n  [1] Baseline RF")
        OA, kappa, f1m, pcf1 = train_rf(
            X_tr, y_tr, X_val, y_val, num_classes, class_weight=None)
        print(f"    OA={OA:.2f}%  Kappa={kappa:.3f}  F1={f1m:.2f}%")
        append_row("baseline", "RF", run, OA, kappa, f1m, pcf1)

        # ── 2. Class-weighted RF ───────────────────────────────────────────
        print("\n  [2] Class-weighted RF")
        OA, kappa, f1m, pcf1 = train_rf(
            X_tr, y_tr, X_val, y_val, num_classes, class_weight="balanced")
        print(f"    OA={OA:.2f}%  Kappa={kappa:.3f}  F1={f1m:.2f}%")
        append_row("class_weighted", "RF", run, OA, kappa, f1m, pcf1)


        # Check all classes present in training set
        train_counts = np.bincount(y_tr, minlength=num_classes)
        print(f"Training counts per class: {train_counts}")

        # Only run SMOTE if ALL classes have at least 2 samples
        min_count = train_counts.min()
        k_neighbors = min(5, min_count - 1)

        # ── 3. SMOTE RF ────────────────────────────────────────────────────
        print("\n  [3] SMOTE RF")
        try:
            min_count   = np.bincount(y_tr).min()
            k_neighbors = min(5, min_count - 1)
            if k_neighbors < 1:
                print(f"    Skipping SMOTE — min class count={min_count}, "
                      f"need at least 2 samples per class")
            else:
                sm = SMOTE(k_neighbors=k_neighbors, random_state=run)
                X_res, y_res = sm.fit_resample(X_tr, y_tr)
                print(f"    After SMOTE: {X_res.shape[0]} samples "
                      f"(was {X_tr.shape[0]})")
                print(f"    Resampled class counts: {np.bincount(y_res)}")
                OA, kappa, f1m, pcf1 = train_rf(
                    X_res, y_res, X_val, y_val, num_classes, class_weight=None)
                print(f"    OA={OA:.2f}%  Kappa={kappa:.3f}  F1={f1m:.2f}%")
                append_row("smote", "RF", run, OA, kappa, f1m, pcf1)
        except Exception as e:
            print(f"    SMOTE failed: {e}")

        # ── 4. Baseline 3D-CNN ─────────────────────────────────────────────
        print("\n  [4] Baseline 3D-CNN")
        OA, kappa, f1m, pcf1 = train_3dcnn(
            p_tr, yp_tr, p_val, yp_val, num_classes,
            loss_fn_name="ce", class_weights_np=None)
        print(f"    OA={OA:.2f}%  Kappa={kappa:.3f}  F1={f1m:.2f}%")
        append_row("baseline", "3DCNN", run, OA, kappa, f1m, pcf1)

        # ── 5. Class-weighted 3D-CNN ───────────────────────────────────────
        print("\n  [5] Class-weighted 3D-CNN")
        OA, kappa, f1m, pcf1 = train_3dcnn(
            p_tr, yp_tr, p_val, yp_val, num_classes,
            loss_fn_name="ce", class_weights_np=cw_np)
        print(f"    OA={OA:.2f}%  Kappa={kappa:.3f}  F1={f1m:.2f}%")
        append_row("class_weighted", "3DCNN", run, OA, kappa, f1m, pcf1)

        # ── 6. Focal Loss 3D-CNN ───────────────────────────────────────────
        print("\n  [6] Focal Loss 3D-CNN")
        OA, kappa, f1m, pcf1 = train_3dcnn(
            p_tr, yp_tr, p_val, yp_val, num_classes,
            loss_fn_name="focal", class_weights_np=None)
        print(f"    OA={OA:.2f}%  Kappa={kappa:.3f}  F1={f1m:.2f}%")
        append_row("focal_loss", "3DCNN", run, OA, kappa, f1m, pcf1)

    print(f"\nDone. Results saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()