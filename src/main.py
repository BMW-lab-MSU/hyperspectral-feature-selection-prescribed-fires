#!/usr/bin/env python3
import argparse
import random
import numpy as np

from datasets.hsi_data import load_hsi_dataset, make_pixel_dataset
from band_selection.loader import load_band_order, select_top_k_bands
from training.train_classical import train_pixel_models
from utils.logger import init_results_csv, append_result_row


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description="HSI Pipeline: (optional) band selection -> classical model training")

    parser.add_argument("--dataset", type=str, required=True, help="e.g., paviaU, indian_pines, salinas, montana")
    parser.add_argument("--eda", action="store_true", help="Flag reserved (run EDA cleaning separately for now)")
    parser.add_argument("--method", type=str, default="none", help="none | PCA | SSEP | SRPA | DRL")
    parser.add_argument("--topk", type=int, default=0, help="Top-k bands (required if method != none)")
    parser.add_argument("--model", type=str, default="RF", help="RF | SVM | KNN")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_size", type=float, default=0.2)

    args = parser.parse_args()
    set_seed(args.seed)

    # 1) Load dataset
    cube, gt = load_hsi_dataset(args.dataset)

    # 2) Optional band selection
    selected_idx = None
    if args.method.lower() != "none":
        if args.topk <= 0:
            raise ValueError("--topk must be > 0 when --method is not 'none'")

        order = load_band_order(args.dataset, args.method)
        cube, selected_idx = select_top_k_bands(cube, order, args.topk)

    # 3) Pixels -> train/test
    X_train, y_train, X_test, y_test = make_pixel_dataset(
        cube, gt, test_size=args.test_size, seed=args.seed, drop_background=True
    )

    # 4) Train
    # metrics = train_pixel_models(X_train, y_train, X_test, y_test, model_name=args.model)
    metrics = train_pixel_models(X_train, y_train, X_test, y_test)

    # 5) Log
    init_results_csv()
    append_result_row(
        dataset=args.dataset,
        method=args.method,
        band_mode=("band_selection" if args.method.lower() != "none" else "no_selection"),
        top_k=(args.topk if args.method.lower() != "none" else 0),
        model=args.model,
        OA=metrics.get("OA"),
        kappa=metrics.get("kappa"),
        precision_macro=metrics.get("precision_macro"),
        recall_macro=metrics.get("recall_macro"),
        f1_macro=metrics.get("f1_macro"),
        best_val_acc=metrics.get("best_val_acc"),
        conf_mat_path="",
        extra_info=f"seed={args.seed}",
)
    
    # for model_name, m in metrics.items():
    # append_result_row(
    #     dataset=args.dataset,
    #     method=args.method,
    #     band_mode=("band_selection" if args.method.lower() != "none" else "no_selection"),
    #     top_k=(args.topk if args.method.lower() != "none" else 0),
    #     model=model_name,
    #     OA=m.get("OA"),
    #     kappa=m.get("kappa"),
    #     precision_macro=m.get("precision_macro"),
    #     recall_macro=m.get("recall_macro"),
    #     f1_macro=m.get("f1_macro"),
    #     best_val_acc=m.get("best_val_acc"),
    #     conf_mat_path="",                # optional: save later
    #     extra_info=f"seed={args.seed}",
    # )

    print("✅ Done. Metrics:", metrics)
    if selected_idx is not None:
        print("Selected bands (first 10):", selected_idx[:10])


if __name__ == "__main__":
    main()