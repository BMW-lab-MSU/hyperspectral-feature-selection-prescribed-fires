#!/usr/bin/env python3
# models_deep.py

"""
Deep models for HSI:
- Simple 3D CNN
- Simple Hybrid CNN (spectral 1D + spatial 2D)
- Simple GCN (graph over labeled pixels) implemented in pure PyTorch.

All trainers return a dict with OA, kappa, optionally confusion matrix.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


def compute_classification_metrics(y_true, y_pred):
    """
    Returns metrics as a dict.
    OA is in PERCENT (0–100).
    Precision/Recall/F1 are also in PERCENT to match OA.
    Kappa is in [0,1].
    """
    OA = accuracy_score(y_true, y_pred) * 100.0
    kappa = cohen_kappa_score(y_true, y_pred)

    precision_macro = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    ) * 100.0
    recall_macro = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    ) * 100.0
    f1_macro = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    ) * 100.0

    conf_mat = confusion_matrix(y_true, y_pred)

    return {
        "OA": OA,
        "kappa": kappa,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "conf_mat": conf_mat,
    }


# ==========================================================
# 3D CNN
# ==========================================================

class Simple3DCNN(nn.Module):
    """
    Input shape: (B, 1, D, H, W) where D is spectral dimension (bands),
    H=W=patch_size.

    We'll reorder patches accordingly in the trainer.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        # We treat spectral dimension as "depth", spatial as H/W
        # Input channels = 1

        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=16,
            kernel_size=(3, 3, 7),  # (D,H,W)
            padding=(1, 1, 3),
        )
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(kernel_size=2)  # halves all dims

        self.conv2 = nn.Conv3d(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 3, 5),
            padding=(1, 1, 2),
        )
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(kernel_size=2)

        # AdaptiveAvgPool to (1,1,1) so we don't worry about exact dims
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: (B, 1, D, H, W)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.avgpool(x)  # (B, 32, 1, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 32)
        x = self.fc(x)  # (B, num_classes)
        return x


def train_3dcnn(patches_train: np.ndarray, y_train: np.ndarray,
                patches_val: np.ndarray, y_val: np.ndarray,
                num_classes: int,
                num_epochs: int = 20,
                batch_size: int = 32,
                lr: float = 1e-3,
                device: str = None):
    """
    Train the 3D CNN on patch data.

    patches_*: (N, ps, ps, B)
    y_*: (N,)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"3D CNN device: {device}")

    # Reorder patches to (N, 1, D, H, W)
    # We'll choose D=B (bands), H=ps, W=ps
    N_tr, ps, _, B = patches_train.shape
    N_val = patches_val.shape[0]

    # move spectral to depth dimension
    # (N, ps, ps, B) -> (N, B, ps, ps) then -> (N, 1, B, ps, ps)
    def to_3d_tensor(patches: np.ndarray):
        t = torch.from_numpy(patches.astype(np.float32))  # (N, ps, ps, B)
        t = t.permute(0, 3, 1, 2)  # (N, B, ps, ps)
        t = t.unsqueeze(1)  # (N, 1, B, ps, ps)
        return t

    Xtr = to_3d_tensor(patches_train)
    Xval = to_3d_tensor(patches_val)

    ytr = torch.from_numpy(y_train.astype(np.int64))
    yv = torch.from_numpy(y_val.astype(np.int64))

    model = Simple3DCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Simple training loop
    from torch.utils.data import TensorDataset, DataLoader

    train_ds = TensorDataset(Xtr, ytr)
    val_ds = TensorDataset(Xval, yv)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        correct_tr = 0
        total_tr = 0

        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)

            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct_tr += (preds == by).sum().item()
            total_tr += by.size(0)

        train_acc = 100.0 * correct_tr / max(1, total_tr)

        # validation
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                by = by.to(device)
                logits = model(bx)
                preds = logits.argmax(dim=1)
                correct_val += (preds == by).sum().item()
                total_val += by.size(0)

        val_acc = 100.0 * correct_val / max(1, total_val)
        print(f"[3D CNN] Epoch {epoch:02d} | Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    print(f"3D CNN → best Val Acc={best_val_acc:.2f}%")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics on val
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device)
            logits = model(bx)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(by.numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)

    # OA = accuracy_score(y_true, y_pred) * 100
    # kappa = cohen_kappa_score(y_true, y_pred)
    # conf_mat = confusion_matrix(y_true, y_pred)
    #
    # return {
    #     "OA": OA,
    #     "kappa": kappa,
    #     "conf_mat": conf_mat,
    #     "best_val_acc": best_val_acc,
    # }

    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["best_val_acc"] = best_val_acc

    return metrics


# ==========================================================
# Hybrid CNN (very simple version)
# ==========================================================

class HybridCNN(nn.Module):
    """
    Toy hybrid CNN with:
    - Spectral 1D conv over bands
    - Spatial 2D conv on a subset of bands (e.g., first band)
    Then concatenates features and predicts.

    Input shape: (B, 1, D, H, W) same as 3D CNN.
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # Spectral branch: treat D as length of 1D sequence
        # We'll pool over H,W first
        self.spectral_conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.spectral_conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.spectral_pool = nn.AdaptiveAvgPool1d(1)  # -> (B, 32, 1)

        # Spatial branch: use first few bands (e.g., 3) as channels
        self.spatial_conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.spatial_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))  # -> (B, 32,1,1)

        self.fc = nn.Linear(32 + 32, num_classes)

    def forward(self, x):
        # x: (B, 1, D, H, W)
        B, _, D, H, W = x.shape

        # ----- spectral branch -----
        # Pool spatial dims first: mean over H,W
        spec = x.mean(dim=[3, 4])  # (B, 1, D)
        spec = self.spectral_conv1(spec)
        spec = F.relu(spec)
        spec = self.spectral_conv2(spec)
        spec = F.relu(spec)
        spec = self.spectral_pool(spec)  # (B, 32, 1)
        spec = spec.view(B, 32)

        # ----- spatial branch -----
        # Take first 3 bands as channels
        # x[:,0,:,:,:] -> (B, D, H, W) but we want (B, C, H, W), C=3
        x_bands = x[:, 0, :, :, :]  # (B, D, H, W)
        if D >= 3:
            spatial = x_bands[:, :3, :, :]  # (B, 3, H, W)
        else:
            # If <3 bands, pad channels
            pad = 3 - D
            spatial = F.pad(x_bands, (0, 0, 0, 0, 0, pad))  # (B, 3, H, W)

        spatial = self.spatial_conv1(spatial)
        spatial = F.relu(spatial)
        spatial = self.spatial_conv2(spatial)
        spatial = F.relu(spatial)
        spatial = self.spatial_pool(spatial)  # (B, 32, 1, 1)
        spatial = spatial.view(B, 32)

        # concat
        feat = torch.cat([spec, spatial], dim=1)  # (B, 64)
        out = self.fc(feat)
        return out


def train_hybrid_cnn(patches_train: np.ndarray, y_train: np.ndarray,
                     patches_val: np.ndarray, y_val: np.ndarray,
                     num_classes: int,
                     num_epochs: int = 20,
                     batch_size: int = 32,
                     lr: float = 1e-3,
                     device: str = None):
    """
    Similar to train_3dcnn but uses HybridCNN.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hybrid CNN device: {device}")

    def to_3d_tensor(patches: np.ndarray):
        t = torch.from_numpy(patches.astype(np.float32))  # (N, ps, ps, B)
        t = t.permute(0, 3, 1, 2)  # (N, B, ps, ps)
        t = t.unsqueeze(1)  # (N, 1, B, ps, ps)
        return t

    Xtr = to_3d_tensor(patches_train)
    Xval = to_3d_tensor(patches_val)

    ytr = torch.from_numpy(y_train.astype(np.int64))
    yv = torch.from_numpy(y_val.astype(np.int64))

    model = HybridCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    from torch.utils.data import TensorDataset, DataLoader

    train_ds = TensorDataset(Xtr, ytr)
    val_ds = TensorDataset(Xval, yv)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        correct_tr = 0
        total_tr = 0

        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)

            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct_tr += (preds == by).sum().item()
            total_tr += by.size(0)

        train_acc = 100.0 * correct_tr / max(1, total_tr)

        # validation
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                by = by.to(device)
                logits = model(bx)
                preds = logits.argmax(dim=1)
                correct_val += (preds == by).sum().item()
                total_val += by.size(0)

        val_acc = 100.0 * correct_val / max(1, total_val)
        print(f"[Hybrid CNN] Epoch {epoch:02d} | Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    print(f"Hybrid CNN → best Val Acc={best_val_acc:.2f}%")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device)
            logits = model(bx)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(by.numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)

    # OA = accuracy_score(y_true, y_pred) * 100
    # kappa = cohen_kappa_score(y_true, y_pred)
    # conf_mat = confusion_matrix(y_true, y_pred)
    #
    # return {
    #     "OA": OA,
    #     "kappa": kappa,
    #     "conf_mat": conf_mat,
    #     "best_val_acc": best_val_acc,
    # }

    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["best_val_acc"] = best_val_acc

    return metrics


# ==========================================================
# Attention-based Hybrid CNN (3D + Attention + 2D)
# ==========================================================

class ChannelAttention1D(nn.Module):
    """
    Simple channel (feature) attention for 1D conv outputs.
    Input: (B, C, L) -> (B, C, L)
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.shape
        # global pooling over L
        avg = x.mean(dim=2)  # (B, C)
        attn = self.fc2(F.relu(self.fc1(avg)))  # (B, C)
        attn = torch.sigmoid(attn)  # (B, C)
        attn = attn.unsqueeze(2)  # (B, C, 1)
        return x * attn  # (B, C, L)


class ChannelAttention2D(nn.Module):
    """
    Simple channel attention for 2D conv outputs.
    Input: (B, C, H, W) -> (B, C, H, W)
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        avg = x.mean(dim=(2, 3))  # (B, C)
        attn = self.fc2(F.relu(self.fc1(avg)))  # (B, C)
        attn = torch.sigmoid(attn)  # (B, C)
        attn = attn.unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1)
        return x * attn


class AttentionHybridCNN(nn.Module):
    """
    Attention-based Hybrid CNN:
    - Spectral branch: 1D conv over bands + channel attention
    - Spatial branch: 2D conv over first 3 bands + channel attention
    - Concatenates spectral + spatial features and classifies.

    Input: (B, 1, D, H, W)  [same as Simple3DCNN]
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # ----- spectral branch -----
        self.spec_conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.spec_conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.spec_attn = ChannelAttention1D(64, reduction=4)
        self.spec_pool = nn.AdaptiveAvgPool1d(1)  # -> (B, 64, 1)

        # ----- spatial branch -----
        self.spa_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.spa_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.spa_attn = ChannelAttention2D(64, reduction=4)
        self.spa_pool = nn.AdaptiveAvgPool2d((1, 1))  # -> (B, 64, 1, 1)

        # Final classifier
        self.fc = nn.Linear(64 + 64, num_classes)

    def forward(self, x):
        # x: (B, 1, D, H, W)
        B, _, D, H, W = x.shape

        # ===== spectral branch =====
        # avg over H,W -> (B, 1, D)
        spec = x.mean(dim=[3, 4])  # (B, 1, D)
        spec = F.relu(self.spec_conv1(spec))  # (B, 32, D)
        spec = F.relu(self.spec_conv2(spec))  # (B, 64, D)
        spec = self.spec_attn(spec)  # (B, 64, D) with channel attention
        spec = self.spec_pool(spec)  # (B, 64, 1)
        spec = spec.view(B, 64)  # (B, 64)

        # ===== spatial branch =====
        # take first 3 bands as "RGB" channels
        x_bands = x[:, 0, :, :, :]  # (B, D, H, W)
        if D >= 3:
            spatial = x_bands[:, :3, :, :]  # (B, 3, H, W)
        else:
            # pad if < 3 bands
            pad = 3 - D
            spatial = F.pad(x_bands, (0, 0, 0, 0, 0, pad))  # (B, 3, H, W)

        spatial = F.relu(self.spa_conv1(spatial))  # (B, 32, H, W)
        spatial = F.relu(self.spa_conv2(spatial))  # (B, 64, H, W)
        spatial = self.spa_attn(spatial)  # (B, 64, H, W)
        spatial = self.spa_pool(spatial)  # (B, 64, 1, 1)
        spatial = spatial.view(B, 64)  # (B, 64)

        # ===== fusion =====
        feat = torch.cat([spec, spatial], dim=1)  # (B, 128)
        out = self.fc(feat)
        return out


def train_attention_hybrid_cnn(patches_train: np.ndarray, y_train: np.ndarray,
                               patches_val: np.ndarray, y_val: np.ndarray,
                               num_classes: int,
                               num_epochs: int = 20,
                               batch_size: int = 32,
                               lr: float = 1e-3,
                               device: str = None):
    """
    Trainer for AttentionHybridCNN.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Attention Hybrid CNN device: {device}")

    def to_3d_tensor(patches: np.ndarray):
        t = torch.from_numpy(patches.astype(np.float32))  # (N, ps, ps, B)
        t = t.permute(0, 3, 1, 2)  # (N, B, ps, ps)
        t = t.unsqueeze(1)  # (N, 1, B, ps, ps)
        return t

    Xtr = to_3d_tensor(patches_train)
    Xval = to_3d_tensor(patches_val)

    ytr = torch.from_numpy(y_train.astype(np.int64))
    yv = torch.from_numpy(y_val.astype(np.int64))

    model = AttentionHybridCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    from torch.utils.data import TensorDataset, DataLoader
    train_ds = TensorDataset(Xtr, ytr)
    val_ds = TensorDataset(Xval, yv)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        correct_tr = 0
        total_tr = 0

        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)

            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct_tr += (preds == by).sum().item()
            total_tr += by.size(0)

        train_acc = 100.0 * correct_tr / max(1, total_tr)

        # validation
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                by = by.to(device)
                logits = model(bx)
                preds = logits.argmax(dim=1)
                correct_val += (preds == by).sum().item()
                total_val += by.size(0)

        val_acc = 100.0 * correct_val / max(1, total_val)
        print(f"[Attn Hybrid] Epoch {epoch:02d} | Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    print(f"Attention Hybrid CNN → best Val Acc={best_val_acc:.2f}%")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device)
            logits = model(bx)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(by.numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)

    OA = accuracy_score(y_true, y_pred) * 100
    kappa = cohen_kappa_score(y_true, y_pred)
    conf_mat = confusion_matrix(y_true, y_pred)

    return {
        "OA": OA,
        "kappa": kappa,
        "conf_mat": conf_mat,
        "best_val_acc": best_val_acc,
    }


# ==========================================================
# Simple GCN
# ==========================================================

class SimpleGCN(nn.Module):
    """
    Basic 2-layer GCN using precomputed normalized adjacency.
    A_hat is applied as matmul in the trainer.
    """

    def __init__(self, in_features: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=False)

    def forward(self, X, A_norm):
        # X: (N, F), A_norm: (N, N)
        h = A_norm @ self.fc1(X)
        h = F.relu(h)
        out = A_norm @ self.fc2(h)
        return out


def build_knn_graph(X: np.ndarray, k: int = 8):
    """
    X: (N, F) node features
    Returns normalized adjacency matrix A_norm: torch.FloatTensor (N, N)
    """
    N, F_dim = X.shape
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, N), algorithm="auto").fit(X)
    distances, indices = nbrs.kneighbors(X)

    # Build adjacency
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in indices[i]:
            if i == j:
                continue
            A[i, j] = 1.0
            A[j, i] = 1.0

    # Add self loops
    A = A + np.eye(N, dtype=np.float32)

    # Symmetric normalization: D^-1/2 A D^-1/2
    deg = A.sum(axis=1)  # (N,)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    D_inv_sqrt = np.diag(deg_inv_sqrt)

    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    A_norm = torch.from_numpy(A_norm.astype(np.float32))

    return A_norm


def train_gcn(X_train_all: np.ndarray, y_train_all: np.ndarray,
              num_classes: int,
              train_fraction: float = 0.7,
              hidden_dim: int = 64,
              num_epochs: int = 200,
              lr: float = 1e-2,
              device: str = None):
    """
    Graph-based classification over labeled pixels.

    X_train_all: (N, F) full labeled set
    y_train_all: (N,)
    We'll split nodes into train/val sets randomly.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GCN device: {device}")

    N, F_dim = X_train_all.shape
    X = torch.from_numpy(X_train_all.astype(np.float32))  # (N, F)
    y = torch.from_numpy(y_train_all.astype(np.int64))  # (N,)

    # Build graph
    A_norm = build_knn_graph(X_train_all, k=8)
    A_norm = A_norm.to(device)
    X = X.to(device)
    y = y.to(device)

    # Train/val split at node level
    idx = np.arange(N)
    np.random.seed(0)
    np.random.shuffle(idx)
    split = int(train_fraction * N)
    train_idx = idx[:split]
    val_idx = idx[split:]

    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    model = SimpleGCN(in_features=F_dim, hidden_dim=hidden_dim,
                      num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X, A_norm)  # (N, num_classes)

        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        # Train acc
        preds_train = logits[train_mask].argmax(dim=1)
        acc_train = (preds_train == y[train_mask]).float().mean().item() * 100.0

        # Val acc
        model.eval()
        with torch.no_grad():
            logits_val = model(X, A_norm)
        preds_val = logits_val[val_mask].argmax(dim=1)
        acc_val = (preds_val == y[val_mask]).float().mean().item() * 100.0

        if epoch % 20 == 0 or epoch == 1:
            print(f"[GCN] Epoch {epoch:03d} | Loss={loss.item():.4f} | "
                  f"Train Acc={acc_train:.2f}% | Val Acc={acc_val:.2f}%")

        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_state = model.state_dict()

    print(f"GCN → best Val Acc={best_val_acc:.2f}%")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics on val
    model.eval()
    with torch.no_grad():
        logits = model(X, A_norm)
    preds = logits[val_mask].argmax(dim=1).cpu().numpy()
    y_true = y[val_mask].cpu().numpy()

    # OA = accuracy_score(y_true, preds) * 100
    # kappa = cohen_kappa_score(y_true, preds)
    # conf_mat = confusion_matrix(y_true, preds)
    #
    # return {
    #     "OA": OA,
    #     "kappa": kappa,
    #     "conf_mat": conf_mat,
    #     "best_val_acc": best_val_acc,
    # }
    # Final metrics on val

    metrics = compute_classification_metrics(y_true, preds)
    metrics["best_val_acc"] = best_val_acc

    return metrics


# ==========================================================
# Graph Attention Network (GAT)
# ==========================================================

class GATLayer(nn.Module):
    """
    Single-head GAT layer with additive attention.
    X: (N, F_in)
    A: (N, N) adjacency (0/1 or weighted)
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0, alpha: float = 0.2):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, A):
        # X: (N, F_in)
        N = X.size(0)
        h = self.W(X)  # (N, F_out)

        # Prepare pairwise concatenation for attention
        # h_i || h_j for all edges (i,j)
        # For efficiency we compute using broadcasting and mask with A.
        h_i = h.unsqueeze(1).repeat(1, N, 1)  # (N, N, F_out)
        h_j = h.unsqueeze(0).repeat(N, 1, 1)  # (N, N, F_out)
        a_input = torch.cat([h_i, h_j], dim=2)  # (N, N, 2*F_out)

        e = self.leakyrelu(self.a(a_input).squeeze(2))  # (N, N)
        # Mask out non-neighbors with -inf
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(A > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        h_prime = attention @ h  # (N, F_out)
        return h_prime


class SimpleGAT(nn.Module):
    """
    2-layer GAT (single head per layer for simplicity).
    """

    def __init__(self, in_features: int, hidden_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.gat1 = GATLayer(in_features, hidden_dim, dropout=dropout)
        self.gat2 = GATLayer(hidden_dim, num_classes, dropout=dropout)

    def forward(self, X, A):
        h = F.elu(self.gat1(X, A))
        out = self.gat2(h, A)
        return out


def train_gat(X_all: np.ndarray, y_all: np.ndarray,
              num_classes: int,
              train_fraction: float = 0.7,
              hidden_dim: int = 64,
              num_epochs: int = 200,
              lr: float = 5e-3,
              dropout: float = 0.0,
              device: str = None):
    """
    X_all: (N, F), y_all: (N,)
    Uses same KNN graph as GCN, but with GAT layers.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GAT device: {device}")

    from sklearn.neighbors import NearestNeighbors

    X_np = X_all.astype(np.float32)
    N, F_dim = X_np.shape

    # Build KNN adjacency (un-normalized; attention will handle weighting)
    nbrs = NearestNeighbors(n_neighbors=min(8 + 1, N), algorithm="auto").fit(X_np)
    distances, indices = nbrs.kneighbors(X_np)

    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in indices[i]:
            if i == j:
                continue
            A[i, j] = 1.0
            A[j, i] = 1.0
    # self-loops
    A += np.eye(N, dtype=np.float32)

    X = torch.from_numpy(X_np).to(device)
    y = torch.from_numpy(y_all.astype(np.int64)).to(device)
    A_t = torch.from_numpy(A).to(device)

    # Train/val split at node level
    idx = np.arange(N)
    np.random.seed(0)
    np.random.shuffle(idx)
    split = int(train_fraction * N)
    train_idx = idx[:split]
    val_idx = idx[split:]

    train_mask = torch.zeros(N, dtype=torch.bool, device=device)
    val_mask = torch.zeros(N, dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    model = SimpleGAT(in_features=F_dim, hidden_dim=hidden_dim,
                      num_classes=num_classes, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X, A_t)
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        preds_train = logits[train_mask].argmax(dim=1)
        acc_train = (preds_train == y[train_mask]).float().mean().item() * 100.0

        model.eval()
        with torch.no_grad():
            logits_val = model(X, A_t)
        preds_val = logits_val[val_mask].argmax(dim=1)
        acc_val = (preds_val == y[val_mask]).float().mean().item() * 100.0

        if epoch % 20 == 0 or epoch == 1:
            print(f"[GAT] Epoch {epoch:03d} | Loss={loss.item():.4f} | "
                  f"Train Acc={acc_train:.2f}% | Val Acc={acc_val:.2f}%")

        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_state = model.state_dict()

    print(f"GAT → best Val Acc={best_val_acc:.2f}%")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics
    model.eval()
    with torch.no_grad():
        logits = model(X, A_t)
    preds = logits[val_mask].argmax(dim=1).cpu().numpy()
    y_true = y[val_mask].cpu().numpy()

    OA = accuracy_score(y_true, preds) * 100
    kappa = cohen_kappa_score(y_true, preds)
    conf_mat = confusion_matrix(y_true, preds)

    return {
        "OA": OA,
        "kappa": kappa,
        "conf_mat": conf_mat,
        "best_val_acc": best_val_acc,
    }
