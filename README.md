# Hyperspectral Band Selection for Ground Fuel Classification for Prescribed Fires

**Paper:** *"Hyperspectral Band Selection for Ground Fuel Classification for Prescribed Fires"*  
**Journal:** Remote Sensing (MDPI) — under review  
**Institution:** Montana State University, BMW Lab

---

## Overview

This repository provides the complete, reproducible experimental framework for the paper. We evaluate five hyperspectral band selection strategies across six datasets including a novel UAV VNIR dataset collected over prescribed burn sites in Montana, USA.

**Band selection methods evaluated:**

| Method | Type | Description |
|--------|------|-------------|
| PCA | Variance-based | Ranks bands by contribution to principal components |
| SSEP | Edge-aware | Spatial-Spectral Edge Preservation via Dice similarity to GT edges |
| SRPA | Attention-based | Spectral-Redundancy Penalized Attention with SE block |
| DRL | Learning-based | Deep Reinforcement Learning sequential band selection |
| KMCBS | Clustering-based | K-Means Clustering-Based Band Selection (centroid representative) |

**Classifiers evaluated:** Random Forest (RF), SVM (RBF), K-Nearest Neighbours (KNN), 3D-CNN, Hybrid-CNN

---

## Repository Structure

```
hyperspectral-feature-selection-prescribed-fires/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── data/
│   ├── raw/                     ← Place original .mat / .tif files here
│   └── processed/               ← Cleaned .npy files (produced by EDA scripts)
│
├── src/
│   ├── hsi_config.py            ← Dataset paths and experiment parameters
│   ├── hsi_data.py              ← Data loading: pixel extraction, patch extraction
│   ├── band_selection.py        ← Band order loading and cube slicing utilities
│   ├── train_models.py          ← Classifier training wrappers (RF, SVM, KNN, CNN)
│   │
│   ├── eda/
│   │   ├── hsi_eda.py           ← EDA + cleaning for .mat benchmark datasets
│   │   └── hsi_clean_from_mat_or_tif.py  ← EDA for UAV VNIR .tif + CSV datasets
│   │
│   ├── band_selection/
│   │   ├── generic_pca_band_selection.py   ← PCA band scoring
│   │   ├── generic_ssep_band_selection.py  ← SSEP (Sobel + Dice) band scoring
│   │   ├── generic_srpa_band_selection.py  ← SRPA (SE attention + redundancy) scoring
│   │   ├── generic_drl_band_selection.py   ← DRL (policy gradient) band selection
│   │   └── generic_kmcbs_band_selection.py ← KMCBS (K-Means clustering) band selection
│   │
│   ├── models/
│   │   └── models_deep.py       ← 3D-CNN, Hybrid CNN, GCN, GAT architectures
│   │
│   └── utils/
│       └── logger_utils.py      ← CSV results logger
│
├── experiments/
│   ├── run_full_hsi_pipeline.py      ← Main experiment: all datasets × methods × classifiers
│   ├── run_kmeans_band_selection.py  ← KMCBS classification experiment
│   ├── run_patch_sensitivity.py      ← Patch size sensitivity (3×3, 5×5, 7×7)
│   ├── run_ablation.py               ← Ablation study (SRPA penalty, DRL vs random)
│   ├── run_hyperparam_sensitivity.py ← Hyperparameter sensitivity (SSEP σ, SRPA λ)
│   └── run_class_imbalance.py        ← Class imbalance analysis (Montana only)
│
└── outputs/
    ├── results/                 ← CSV result files (auto-created)
    ├── band_scores/             ← Band order .npy files (auto-created by band selection)
    ├── plots/                   ← Band score plots
    └── logs/                    ← Training logs
```

---

## Installation

```bash
git clone https://github.com/BMW-lab-MSU/hyperspectral-feature-selection-prescribed-fires.git
cd hyperspectral-feature-selection-prescribed-fires
pip install -r requirements.txt
```

**GPU support (recommended for 3D-CNN):** Install PyTorch with CUDA from https://pytorch.org/get-started

---

## Data Setup

### Step 1 — Download benchmark datasets

Place the original `.mat` files in `data/raw/`:

| Dataset | File | Source |
|---------|------|--------|
| Indian Pines | `Indian_pines_corrected.mat`, `Indian_pines_gt.mat` | [EHU](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |
| Pavia University | `PaviaU.mat`, `PaviaU_gt.mat` | [EHU](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |
| Salinas | `Salinas_corrected.mat`, `Salinas_gt.mat` | [EHU](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |
| Botswana | `Botswana.mat`, `Botswana_gt.mat` | [EHU](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |
| KSC | `KSC_corrected.mat`, `KSC_gt.mat` | [EHU](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |

**Montana UAV VNIR dataset:** 97 GB UAV-collected VNIR cube from the Lubrecht Experimental Forest prescribed burn site. Available upon reasonable request (NSF SMART FIRES project). Contact the authors.

### Step 2 — Run EDA and preprocessing

For benchmark `.mat` datasets:

```bash
# Edit DATASET_NAME in the script, then run:
python src/eda/hsi_eda.py
```

For Montana UAV VNIR `.tif` + CSV labels:

```bash
python src/eda/hsi_clean_from_mat_or_tif.py
```

Each script produces two files in `data/processed/`:
- `{dataset}_clean.npy` — normalised hyperspectral cube (H, W, B) float32
- `{dataset}_gt.npy` — ground truth map (H, W) int16

### Step 3 — Update paths in hsi_config.py

Edit `src/hsi_config.py` to point `cube_path` and `gt_path` to your processed files.

---

## Reproducing the Paper

### Step 1 — Run band selection (one script per method per dataset)

Edit the `CONFIG` block at the top of each script, then run:

```bash
# PCA
python src/band_selection/generic_pca_band_selection.py

# SSEP
python src/band_selection/generic_ssep_band_selection.py

# SRPA
python src/band_selection/generic_srpa_band_selection.py

# DRL
python src/band_selection/generic_drl_band_selection.py

# KMCBS
python src/band_selection/generic_kmcbs_band_selection.py
```

Each script saves `{dataset}_{method}_order.npy` — the ranked band indices.

### Step 2 — Run the main classification pipeline (Tables 2–8)

```bash
python experiments/run_full_hsi_pipeline.py
```

This loops over all datasets, all band selection methods, all Top-K values [5–50], and all classifiers. Results are written to `outputs/results/hsi_all_results.csv`.

**Expected runtime:** 12–24 hours on GPU for all datasets × methods × classifiers.

### Step 3 — Run reviewer response experiments

```bash
# KMCBS comparison (Table 10)
python experiments/run_kmeans_band_selection.py

# Patch size sensitivity (Table 11) — Comment 3
python experiments/run_patch_sensitivity.py

# Ablation study (Table 12) — Comment 5
python experiments/run_ablation.py

# Hyperparameter sensitivity (Tables 13-14) — Comment 1
python experiments/run_hyperparam_sensitivity.py

# Class imbalance analysis (Table 15) — Comment 2
python experiments/run_class_imbalance.py
```

---

## Experimental Protocol

- **Train/validation split:** 70/30 stratified by class
- **Runs per configuration:** 5 independent runs with seeds 0–4
- **Reported metric:** Mean ± std over 5 runs
- **Patch size:** 5×5 (default), varied in sensitivity analysis
- **EDA condition:** noisy band removal + min-max normalisation per band
- **No-EDA condition:** min-max normalisation only (no band removal)

---

## Key Results Summary

| Dataset | Best Method | Best Classifier | Best OA (%) |
|---------|-------------|-----------------|-------------|
| Indian Pines | DRL | 3D-CNN | 89.23 ± 2.29 |
| Pavia University | DRL | 3D-CNN | 98.75 ± 0.36 |
| Salinas | KMCBS | 3D-CNN | 95.55 ± 0.51 |
| Botswana | KMCBS | 3D-CNN | 95.90 ± 1.32 |
| KSC | KMCBS | 3D-CNN | 94.68 ± 0.71 |
| Montana UAV VNIR | DRL | 3D-CNN | 58.19 ± 4.97 |

---

## Known Issues and Notes

**paviaU capitalisation:** All files and folders for the Pavia University dataset use capital U (`paviaU`), not lowercase. The `band_selection.py` utility preserves the original case of dataset names to avoid file-not-found errors.

**Montana large cube:** The Montana UAV VNIR cube is 10,706 × 8,360 × 211 pixels (≈97 GB). Memory-safe loading via `np.load(..., mmap_mode='r')` and pixel subsampling (50,000 pixels) is used throughout. Minimum recommended RAM: 32 GB.

**SMOTE on Montana:** SMOTE synthetic oversampling could not be applied to Montana because the most minority class (Dead Ponderosa Pine, 30 pixels) produces a sixth background class label after stratified splitting, causing k-neighbors to be 0. Class weighting and focal loss are used instead.

**DRL reward function:** The DRL implementation uses pure classification accuracy as the reward (no compactness penalty λ). The reward equation in the paper has been corrected to reflect this.

---

## Citation

If you use this code or the Montana dataset, please cite:

```bibtex
@article{karankot2025hsi,
  title   = {Hyperspectral Band Selection for Ground Fuel Classification
             for Prescribed Fires},
  author  = {Karankot, Shamana and others},
  journal = {Remote Sensing},
  year    = {2025},
  note    = {Under review}
}
```

---

## Acknowledgements

This material is based upon work supported in part by the National Science Foundation EPSCoR Cooperative Agreement OIA-2242802 (NSF SMART FIRES project). Computational efforts were performed on the Tempest High Performance Computing System at Montana State University (RRID:SCR 026229).

---

## License

MIT License. See LICENSE file for details.
