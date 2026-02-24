#!/usr/bin/env python3
# generic_srpa_band_selection.py
#
# SRPA band selection for any HSI dataset (VNIR or others).
# Directly adapted from your Montana SRPA script – only CONFIG changed.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# ==========================
# CONFIG – EDIT THIS PART
# ==========================
DATASET_NAME = "paviaU"  # e.g. "montana", "ksc", "botswana", "salinas", "indian_pines", "paviaU"

OUT_DIR = f"{DATASET_NAME}_srpa_band_selection_results"
os.makedirs(OUT_DIR, exist_ok=True)

CLEAN_PATH = "paviaU_clean.npy"  # e.g. "KSC_clean.npy"
GT_PATH = "paviaU_gt.npy"  # e.g. "KSC_gt.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device for SRPA:", device)


class SRPADataset(Dataset):
    def __init__(self, patches, labels):
        self.patches = patches
        self.labels = labels

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.patches[idx]).float()  # (ps, ps, B)
        x = x.unsqueeze(0)  # (1, ps, ps, B)
        y = torch.tensor(self.labels[idx]).long()
        return x, y


class SEBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // 4)
        self.fc2 = nn.Linear(in_channels // 4, in_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class SRPA3DCNN(nn.Module):
    def __init__(self, num_bands, num_classes):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, num_bands))
        self.se = SEBlock(num_bands)
        self.fc = nn.Linear(16 * num_bands, num_classes)

    def forward(self, x):  # x: (B, 1, H, W, Bands)
        x = self.pool(F.relu(self.conv1(x)))  # (B, 8, H/2, W/2, B)
        x = F.relu(self.conv2(x))  # (B,16,H/2,W/2,B)

        pooled = self.global_pool(x)  # (B,16,1,1,B)
        pooled = pooled.squeeze(2).squeeze(2)  # (B,16,B)

        attn_input = pooled.mean(dim=1)  # (B,Bands)
        attn_weights = self.se(attn_input)  # (B,Bands)

        feat = pooled.permute(0, 2, 1)  # (B,Bands,16)
        feat_flat = feat.reshape(feat.size(0), -1)  # (B,Bands*16)

        logits = self.fc(feat_flat)
        return logits, attn_weights


def compute_redundancy_matrix(patches):
    """
    patches: (N, H, W, B)
    Returns redundancy score per band (B,)
    """
    N, H, W, B = patches.shape
    X = patches.reshape(-1, B)

    if X.shape[0] > 100_000:
        idx = np.random.choice(X.shape[0], 100_000, replace=False)
        X = X[idx]

    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    redundancy = (np.sum(np.abs(corr), axis=1) - 1) / (B - 1)
    return redundancy


def srpa_selection(patches, labels, num_classes, lambda_penalty=0.3, batch_size=16):
    N, H, W, B = patches.shape
    print(f"Running SRPA on patches: {patches.shape}, num_classes={num_classes}")

    dataset = SRPADataset(patches, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SRPA3DCNN(num_bands=B, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 2
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            out, _ = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        print(f"SRPA Epoch {epoch + 1}/{num_epochs}, Loss={total_loss / len(dataset):.4f}")

    model.eval()
    attn_scores = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _, attn = model(x)
            attn_scores.append(attn.mean(dim=0).cpu().numpy())

    attn_mean = np.mean(np.vstack(attn_scores), axis=0)
    redundancy = compute_redundancy_matrix(patches)
    srpa_scores = attn_mean - lambda_penalty * redundancy

    return srpa_scores, attn_mean, redundancy


def extract_patches_from_cube(cube, gt, patch_size=5, max_patches=4000):
    H, W, B = cube.shape
    pad = patch_size // 2
    coords = []
    labels = []

    for r in range(pad, H - pad):
        for c in range(pad, W - pad):
            lab = gt[r, c]
            if lab > 0:
                coords.append((r, c))
                labels.append(lab - 1)  # 1..C-1 -> 0..C-1

    coords = np.array(coords)
    labels = np.array(labels)

    if len(coords) > max_patches:
        idx = np.random.choice(len(coords), max_patches, replace=False)
        coords = coords[idx]
        labels = labels[idx]

    patches = []
    for (r, c) in coords:
        r1 = r - pad
        r2 = r + pad + 1
        c1 = c - pad
        c2 = c + pad + 1
        patch = cube[r1:r2, c1:c2, :]
        patches.append(patch)

    patches = np.stack(patches, axis=0)
    return patches, labels


if __name__ == "__main__":
    print(f"Loading {DATASET_NAME} cube and GT...")
    cube = np.load(CLEAN_PATH, mmap_mode="r")
    gt = np.load(GT_PATH)

    H, W, Bc = cube.shape
    num_classes = int(gt.max())
    print("HSI shape:", cube.shape)
    print("GT shape :", gt.shape)
    print("Num classes:", num_classes)

    patches, labels = extract_patches_from_cube(
        cube, gt, patch_size=5, max_patches=4000
    )
    print("SRPA patches:", patches.shape)
    print("SRPA labels :", labels.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        patches, labels, test_size=0.2, random_state=0, stratify=labels
    )

    srpa_scores, attn_mean, redundancy = srpa_selection(
        X_train, y_train, num_classes=num_classes
    )

    order_srpa = np.argsort(srpa_scores)[::-1]

    print("\n=======================================")
    print("SRPA — Full Ranking (Top 10):", order_srpa[:10])
    print("Top-30 SRPA bands:", order_srpa[:30])
    print("Top-50 SRPA bands:", order_srpa[:50])
    print("=======================================\n")

    np.save(os.path.join(OUT_DIR, f"{DATASET_NAME}_srpa_order.npy"), order_srpa)
    np.save(os.path.join(OUT_DIR, f"{DATASET_NAME}_srpa_scores.npy"), srpa_scores)
    np.save(os.path.join(OUT_DIR, f"{DATASET_NAME}_srpa_attention.npy"), attn_mean)
    np.save(os.path.join(OUT_DIR, f"{DATASET_NAME}_srpa_redundancy.npy"), redundancy)

    df = pd.DataFrame({
        "rank": np.arange(1, Bc + 1),
        "band_index": order_srpa,
        "srpa_score_sorted": srpa_scores[order_srpa],
    })
    df.to_csv(os.path.join(OUT_DIR, f"{DATASET_NAME}_srpa_band_scores.csv"),
              index=False)

    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(Bc), srpa_scores, marker="o", linestyle="-")
    plt.xlabel("Band index (local)")
    plt.ylabel("SRPA score")
    plt.title(f"{DATASET_NAME.upper()} — SRPA band scores vs band index")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{DATASET_NAME}_srpa_scores_vs_band.png"),
                dpi=300)
    plt.close()

    print("Saved SRPA outputs in:", OUT_DIR)
