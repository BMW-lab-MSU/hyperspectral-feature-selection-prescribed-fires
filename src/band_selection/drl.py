#!/usr/bin/env python3
# generic_drl_band_selection.py
# -*- coding: utf-8 -*-
#
# DRL-based band selection for any HSI dataset.
# Adapted from your Montana DRL script; just edit CONFIG.

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ==========================
# CONFIG – EDIT THIS PART
# ==========================
DATASET_NAME = "paviaU"  # e.g. "montana", "ksc", "botswana", "salinas", "indian_pines" , "paviaU"

CUBE_PATH = "paviaU_clean.npy"  # e.g. "KSC_clean.npy"
GT_PATH = "paviaU_gt.npy"  # e.g. "KSC_gt.npy"
OUT_DIR = f"{DATASET_NAME}_drl_bandselection_results"
os.makedirs(OUT_DIR, exist_ok=True)

K_BANDS = 50
EPISODES = 150
GAMMA = 1.0
LR = 1e-3
RF_N_EST = 50
VAL_SIZE = 0.3

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 1. Load data and flatten labeled pixels
# -----------------------------
print(f"Loading {DATASET_NAME} cube and GT (memmap for cube)...")
cube = np.load(CUBE_PATH, mmap_mode="r")
gt = np.load(GT_PATH)

H, W, Bc = cube.shape
print(f"Cube shape: {cube.shape}")
print(f"GT shape  : {gt.shape}")

rows, cols = np.where(gt > 0)
labels = gt[rows, cols]
N = rows.shape[0]

print(f"Total labeled pixels: {N}")
print(f"Number of bands (Bc): {Bc}")

print("Extracting labeled spectra from memmapped cube into RAM...")
X = cube[rows, cols, :]
X = np.asarray(X, dtype=np.float32)
y = labels.astype(int) - 1  # 1..C -> 0..C-1

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=VAL_SIZE, random_state=42, stratify=y
)

print("Train samples:", X_train.shape[0])
print("Val samples  :", X_val.shape[0])

del cube


def evaluate_subset(band_indices):
    if len(band_indices) == 0:
        return 0.0

    X_tr = X_train[:, band_indices]
    X_vl = X_val[:, band_indices]

    clf = RandomForestClassifier(
        n_estimators=RF_N_EST,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_vl)
    acc = accuracy_score(y_val, y_pred)
    return acc


class PolicyNet(nn.Module):
    def __init__(self, num_bands):
        super().__init__()
        self.fc1 = nn.Linear(num_bands, 128)
        self.fc2 = nn.Linear(128, num_bands)

    def forward(self, state_mask):
        x = state_mask.float()
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


policy = PolicyNet(Bc).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

print("\n=== Starting DRL training for band selection ===")
print(f"Episodes: {EPISODES}, K (subset size): {K_BANDS}\n")

score_sum = np.zeros(Bc, dtype=float)
count = np.zeros(Bc, dtype=float)

for episode in range(1, EPISODES + 1):
    selected_mask = np.zeros(Bc, dtype=np.float32)
    selected_bands = []
    log_probs = []

    for t in range(K_BANDS):
        state_tensor = torch.from_numpy(selected_mask).unsqueeze(0).to(device)

        logits = policy(state_tensor).squeeze(0)

        mask_torch = torch.from_numpy(selected_mask == 1).to(device)
        masked_logits = logits.masked_fill(mask_torch, -1e9)

        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        a = int(action.item())

        if selected_mask[a] == 1:
            unselected = np.where(selected_mask == 0)[0]
            if len(unselected) == 0:
                break
            a = int(np.random.choice(unselected))
        else:
            selected_mask[a] = 1.0
            selected_bands.append(a)
            log_probs.append(log_prob)

    reward = evaluate_subset(selected_bands)
    R = reward

    for b in selected_bands:
        score_sum[b] += R
        count[b] += 1

    if len(log_probs) > 0:
        loss = torch.tensor(0.0, device=device)
        for lp in log_probs:
            loss = loss - R * lp

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if episode % 10 == 0 or episode == 1:
        if np.any(count > 0):
            avg_band_score = (score_sum[count > 0] / count[count > 0]).mean()
        else:
            avg_band_score = 0.0
        print(
            f"Episode {episode:03d} | "
            f"Reward: {R:.4f} | "
            f"Avg band score (over used bands): {avg_band_score:.4f}"
        )

print("\n=== DRL training finished ===")

drl_scores = np.zeros(Bc, dtype=float)
nonzero = count > 0
drl_scores[nonzero] = score_sum[nonzero] / count[nonzero]

order_drl = np.argsort(drl_scores)[::-1]

print(f"\n=== DRL Band Ranking Summary ({DATASET_NAME}) ===")
print("Top-10 band indices:", order_drl[:10])
print("Top-30 band indices:", order_drl[:30])
print("Top-50 band indices:", order_drl[:50])

np.save(os.path.join(OUT_DIR, f"{DATASET_NAME}_drl_order.npy"), order_drl)
np.save(os.path.join(OUT_DIR, f"{DATASET_NAME}_drl_scores.npy"), drl_scores)

ranks = np.arange(1, Bc + 1)
df = pd.DataFrame({
    "rank": ranks,
    "band_index": order_drl,
    "drl_score_sorted": drl_scores[order_drl],
})
csv_path = os.path.join(OUT_DIR, f"{DATASET_NAME}_drl_band_scores.csv")
df.to_csv(csv_path, index=False)

plt.figure(figsize=(8, 5))
plt.plot(np.arange(Bc), drl_scores, marker="o", linestyle="-")
plt.xlabel("Band index (0-based)")
plt.ylabel("DRL score (average reward)")
plt.title(f"{DATASET_NAME.upper()} — DRL band scores vs band index")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plot_path = os.path.join(OUT_DIR, f"{DATASET_NAME}_drl_scores_vs_band.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print("\nSaved:")
print(" -", os.path.join(OUT_DIR, f"{DATASET_NAME}_drl_order.npy"))
print(" -", os.path.join(OUT_DIR, f"{DATASET_NAME}_drl_scores.npy"))
print(" -", csv_path)
print(" -", plot_path)
print(f"\nDRL band selection for {DATASET_NAME} DONE.")
