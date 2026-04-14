#!/usr/bin/env python3
# src/train_models.py
"""
Training wrappers for all classifiers used in the paper.

Classical (pixel-based)
-----------------------
train_pixel_models      : RF, SVM, KNN in one call

Deep (patch-based)
------------------
train_3dcnn_wrapper     : Simple 3D-CNN
train_hybrid_cnn_wrapper: Hybrid CNN (spectral 1D + spatial 2D)
train_gcn_wrapper       : Graph Convolutional Network

All wrappers return a dict with keys:
    OA, kappa, precision_macro, recall_macro, f1_macro,
    conf_mat, best_val_acc
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

from src.models.models_deep import train_3dcnn, train_hybrid_cnn, train_gcn


# ── Metrics helper ─────────────────────────────────────────────────────────────

def compute_classification_metrics(y_true, y_pred) -> Dict[str, Any]:
    """
    Compute standard classification metrics.

    Returns
    -------
    dict with OA (%), kappa, precision_macro (%), recall_macro (%),
    f1_macro (%), conf_mat (numpy array)
    """
    OA              = accuracy_score(y_true, y_pred) * 100.0
    kappa           = cohen_kappa_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro",
                                      zero_division=0) * 100.0
    recall_macro    = recall_score(y_true, y_pred, average="macro",
                                   zero_division=0) * 100.0
    f1_macro        = f1_score(y_true, y_pred, average="macro",
                               zero_division=0) * 100.0
    conf_mat        = confusion_matrix(y_true, y_pred)

    return {
        "OA":              OA,
        "kappa":           kappa,
        "precision_macro": precision_macro,
        "recall_macro":    recall_macro,
        "f1_macro":        f1_macro,
        "conf_mat":        conf_mat,
    }


# ── Classical models ───────────────────────────────────────────────────────────

def train_pixel_models(X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_val:   np.ndarray,
                       y_val:   np.ndarray) -> Dict[str, Dict[str, Any]]:
    """
    Train RF, SVM (RBF), and KNN on flattened spectral features.

    Parameters
    ----------
    X_train, X_val : (N, B) feature matrices
    y_train, y_val : (N,)   integer class labels [0, C-1]

    Returns
    -------
    dict keyed by 'RF', 'SVM', 'KNN', each containing metric dict
    """
    results: Dict[str, Dict[str, Any]] = {}

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
    rf.fit(X_train, y_train)
    m = compute_classification_metrics(y_val, rf.predict(X_val))
    m["best_val_acc"] = m["OA"]
    results["RF"] = m

    # Support Vector Machine (RBF kernel)
    svm = SVC(kernel="rbf", C=10, gamma="scale")
    svm.fit(X_train, y_train)
    m = compute_classification_metrics(y_val, svm.predict(X_val))
    m["best_val_acc"] = m["OA"]
    results["SVM"] = m

    # K-Nearest Neighbours
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    m = compute_classification_metrics(y_val, knn.predict(X_val))
    m["best_val_acc"] = m["OA"]
    results["KNN"] = m

    return results


# ── Deep model wrappers ────────────────────────────────────────────────────────

def train_3dcnn_wrapper(patches_train: np.ndarray,
                        y_train:       np.ndarray,
                        patches_val:   np.ndarray,
                        y_val:         np.ndarray,
                        num_classes:   int) -> Dict[str, Any]:
    """
    Train Simple 3D-CNN on patch data.

    Parameters
    ----------
    patches_train, patches_val : (N, ps, ps, B)
    y_train, y_val             : (N,) int labels [0, C-1]
    num_classes                : number of output classes C

    Returns
    -------
    metric dict (see compute_classification_metrics + best_val_acc)
    """
    return train_3dcnn(
        patches_train, y_train,
        patches_val,   y_val,
        num_classes=num_classes,
        num_epochs=20,
        batch_size=32,
        lr=1e-3,
    )


def train_hybrid_cnn_wrapper(patches_train: np.ndarray,
                             y_train:       np.ndarray,
                             patches_val:   np.ndarray,
                             y_val:         np.ndarray,
                             num_classes:   int) -> Dict[str, Any]:
    """
    Train Hybrid CNN (spectral 1D + spatial 2D branches) on patch data.
    """
    return train_hybrid_cnn(
        patches_train, y_train,
        patches_val,   y_val,
        num_classes=num_classes,
        num_epochs=20,
        batch_size=32,
        lr=1e-3,
    )


def train_gcn_wrapper(X_all:       np.ndarray,
                      y_all:       np.ndarray,
                      num_classes: int) -> Dict[str, Any]:
    """
    Train Graph Convolutional Network on labeled pixel features.
    Constructs a KNN graph internally.
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
