# Hyperspectral Band Selection for Ground Fuel Classification for Prescribed Fires


**“Hyperspectral Band Selection for Ground Fuel Classification for Prescribed Fires”**  
Submitted to *Remote Sensing (MDPI)*

---

## Abstract

This repository provides the complete experimental framework for evaluating hyperspectral band selection strategies for ground fuel classification in prescribed fire environments. The study investigates how dimensionality reduction and band ranking techniques influence classification performance across benchmark hyperspectral datasets and UAV-based VNIR imagery collected from prescribed burn sites.

We systematically compare:

- PCA (unsupervised variance ranking)
- SSEP (Spatial-Spectral Edge Preservation)
- SRPA (Spectral-Redundancy Penalized Attention)
- DRL-based sequential band selection

under both:

- **With Exploratory Data Analysis (EDA)**
- **Without EDA (No-EDA baseline)**

Classification performance is evaluated using:

- Random Forest (RF)
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- 3D Convolutional Neural Network (3D-CNN)

---

## Research Objectives

This repository reproduces experiments addressing:

1. How does EDA influence hyperspectral classification performance?
2. Which band selection strategy provides the best accuracy–efficiency trade-off?
3. How stable are selected bands across datasets?
4. How does performance vary with top-k band subsets (k = 5–50)?

---

## Datasets

The framework supports:

- Indian Pines
- Pavia University
- Salinas
- Botswana
- Kennedy Space Center (KSC)
- Montana UAV VNIR dataset (prescribed burn imagery)

The Montana dataset includes UAV-collected VNIR hyperspectral cubes for ground fuel classification under prescribed fire conditions.

---

## Experimental Pipeline

The workflow implemented in this repository follows the unified architecture described in the manuscript.

### 1. Dataset Loading

Raw hyperspectral cubes are loaded and prepared for analysis.



### 2. Exploratory Data Analysis (Optional)

When enabled, the EDA pipeline performs:

- Noisy band detection  
- Zero-value fraction analysis  
- Percentile clipping (Montana UAV VNIR)  
- Min–max normalization  

When disabled, raw data proceeds directly to modeling (No-EDA baseline).

### 3. Band Selection

The following ranking strategies are implemented:

- **PCA** – Variance-based unsupervised ranking  
- **SSEP** – Edge-guided Dice-based ranking  
- **SRPA** – Attention-based ranking with redundancy penalty  
- **DRL** – Sequential band selection via reinforcement learning  

Each method produces:

- Ranked band list  
- Top-k subsets (k = 5–50)

### 4. Classification

Selected band subsets are evaluated using:

- Random Forest (pixel-based)  
- SVM (pixel-based)  
- KNN (pixel-based)  
- 3D-CNN (patch-based spatial-spectral modeling)

### 5. Evaluation Metrics

Performance metrics include:

- Overall Accuracy (OA)  
- Cohen’s Kappa  
- Macro-F1 score  
- Confusion matrices  
- Δ performance (With-EDA − No-EDA)

---

## Repository Structure
```
hyperspectral-band-selection-prescribed-fires/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── configs/
│   ├── default.yaml
│   ├── eda_config.yaml
│   └── model_config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│

├── src/
│   ├── eda/
│   │   ├── eda_pipeline.py
│   │   ├── noisy_band_detection.py
│   │   ├── normalization.py
│   │   └── percentile_clipping.py
│   │
│   ├── band_selection/
│   │   ├── pca.py
│   │   ├── ssep.py
│   │   ├── srpa.py
│   │   └── drl.py
│   │
│   ├── training/
│   │   ├── rf.py
│   │   ├── svm.py
│   │   ├── knn.py
│   │   └── cnn3d.py
│   │
|   ├── datasets/
|   |
|   ├── utils/
|   |
|   |
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── confusion_matrix.py
│   │   └── comparison.py
│   │
│   └── main.py
│
├── experiments/
│   ├── run_full_experiment.py
│   └── run_ablation.py
│
├── outputs/
│   ├── band_scores/
│   ├── metrics/
│   ├── plots/
│   └── logs/
│
└── notebooks/
    ├── exploratory_analysis.ipynb
    ├── band_ranking_visualization.ipynb
    └── results_summary.ipynb

```

---

## Installation

```bash
git clone https://github.com/BMW-lab-MSU/hyperspectral-feature-selection-prescribed-firess.git
cd hyperspectral-band-selection-prescribed-fires
pip install -r requirements.txt
```
---

## Running Experiments
### Run with EDA:
```
python src/main.py --dataset indian_pines --eda True --method SRPA --topk 20
```
---
## Data Setup

After downloading datasets, place them under:

data/processed/

Expected filenames:

- paviaU_clean.npy
- paviaU_gt.npy
- indian_pines_clean.npy
- indian_pines_gt.npy
- ...

Each dataset must contain:
- Clean hyperspectral cube: (H, W, B)
- Ground truth map: (H, W)

Band selection order files must exist in the project root:

{dataset}_{method}_band_selection_results/
    {dataset}_{method}_order.npy

  ---
  ## Reproducibility

- Fixed random seeds for all experiments
- Stratified train/test split
- Band order files generated once per dataset
- All results logged to:

outputs/aggregated/all_runs.csv

Each row corresponds to one dataset–method–model–topk configuration.
---
## Data Availability

Public benchmark datasets (Indian Pines, Pavia University, Salinas, Botswana, KSC)
are available from their respective repositories.

The Montana UAV VNIR dataset (97GB) is part of the NSF SMART FIRES project and
is available upon reasonable request.
---

