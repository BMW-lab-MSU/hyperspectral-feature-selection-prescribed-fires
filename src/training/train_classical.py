#!/usr/bin/env python3
# train_models.py

"""
Wrapper functions for:
- Classical ML: RF, SVM, KNN
- Deep models: 3D CNN, Hybrid CNN, GCN (imported from models_deep.py)
"""

from typing import Dict, Any
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from .models_deep import (
    train_3dcnn,
    train_hybrid_cnn,
    train_gcn,
)


# ==========================================================
# Metric helper
# ==========================================================
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
# Classical pixel-based models
# ==========================================================
def train_pixel_models(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """
    Train RF, SVM, KNN on flattened spectral features.
    Returns dict with metrics for each model.
    Each entry includes:
        OA, kappa, precision_macro, recall_macro, f1_macro, conf_mat, best_val_acc
    For classical models, best_val_acc = OA (no epochs).
    """
    results: Dict[str, Dict[str, Any]] = {}

    # ---------------- RF ----------------
    rf = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=0
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)

    metrics_rf = compute_classification_metrics(y_val, y_pred)
    metrics_rf["best_val_acc"] = metrics_rf["OA"]
    results["RF"] = metrics_rf

    # ---------------- SVM ----------------
    svm = SVC(kernel="rbf", C=10, gamma="scale")
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_val)

    metrics_svm = compute_classification_metrics(y_val, y_pred)
    metrics_svm["best_val_acc"] = metrics_svm["OA"]
    results["SVM"] = metrics_svm

    # ---------------- KNN ----------------
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)

    metrics_knn = compute_classification_metrics(y_val, y_pred)
    metrics_knn["best_val_acc"] = metrics_knn["OA"]
    results["KNN"] = metrics_knn

    return results


# ==========================================================
# Deep wrappers
# ==========================================================
def train_3dcnn_wrapper(patches_train: np.ndarray, y_train: np.ndarray,
                        patches_val: np.ndarray, y_val: np.ndarray,
                        num_classes: int):
    """
    Thin wrapper around train_3dcnn from models_deep.
    Assumes train_3dcnn returns:
        {OA, kappa, precision_macro, recall_macro, f1_macro,
         conf_mat, best_val_acc}
    """
    return train_3dcnn(
        patches_train,
        y_train,
        patches_val,
        y_val,
        num_classes=num_classes,
        num_epochs=20,
        batch_size=32,
        lr=1e-3,
    )


def train_hybrid_cnn_wrapper(patches_train: np.ndarray, y_train: np.ndarray,
                             patches_val: np.ndarray, y_val: np.ndarray,
                             num_classes: int):
    """
    Thin wrapper around train_hybrid_cnn from models_deep.
    Same expected metric dict as train_3dcnn.
    """
    return train_hybrid_cnn(
        patches_train,
        y_train,
        patches_val,
        y_val,
        num_classes=num_classes,
        num_epochs=20,
        batch_size=32,
        lr=1e-3,
    )


def train_gcn_wrapper(X_all: np.ndarray, y_all: np.ndarray,
                      num_classes: int):
    """
    Thin wrapper around train_gcn from models_deep.
    Assumes train_gcn returns the same metric dict.
    """
    return train_gcn(
        X_train_all=X_all,
        y_train_all=y_all,
        num_classes=num_classes,
        train_fraction=0.7,
        hidden_dim=64,
        num_epochs=200,
        lr=1e-2,
    )
