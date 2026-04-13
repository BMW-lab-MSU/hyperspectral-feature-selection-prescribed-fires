"""
generate_corrected_tables.py
Reproduces Tables 2-8 for the HSI Band Selection paper from the 10 run CSVs.

Strategy (Section 3.1 of paper):
  For each dataset + method:
    1. Group all (top_k, model) configs; compute mean OA across runs.
    2. Select the single best (top_k, model) by highest mean OA.
    3. Report mean +/- std of that winning config across the runs.

Montana EDA baseline note:
  Run 1 used 273 full bands; runs 2-5 used 211 EDA-cleaned bands.
  For EDA "No Band Selection" on Montana we exclude run 1's 273-band result
  so all reported runs use the same (211-band) setup.

Table 8 (Cross-Dataset):
  For each method, average the per-dataset best-config OA across all 6 datasets.

Usage:
  Put this script in the same folder as your 10 CSV files and run:
      python generate_corrected_tables.py
"""

import pandas as pd
import numpy as np
import os

EDA_FILES = [f"hsi_all_results_eda_run_{i}.csv" for i in range(1, 6)]
NOEDA_FILES = [f"hsi_all_results_no_eda_run_{i}.csv" for i in range(1, 6)]


METHOD_ORDER = ['none', 'PCA', 'SSEP', 'SRPA', 'DRL']
METHOD_LABELS = {
    'none': 'No Band Selection', 'PCA': 'PCA',
    'SSEP': 'SSEP', 'SRPA': 'SRPA', 'DRL': 'DRL'
}
DATASET_ORDER = ['indian_pines', 'paviaU', 'salinas', 'botswana', 'ksc', 'montana']
DATASET_LABELS = {
    'indian_pines': 'Indian Pines',
    'paviaU': 'Pavia University',
    'salinas': 'Salinas',
    'botswana': 'Botswana',
    'ksc': 'KSC',
    'montana': 'Montana (UAV)',
}
TABLE_NUMS = {ds: i for i, ds in enumerate(DATASET_ORDER, 2)}


def load_runs(file_list):
    dfs = []
    for i, f in enumerate(file_list, 1):
        full_path = os.path.abspath(f)   # gets full path

        print(f"Reading file: {f}")
        print(f"Full path: {full_path}")
        df = pd.read_csv(f)
        df['run'] = i
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def best_configs(df, condition_label):
    rows = []
    for ds in DATASET_ORDER:
        sub = df[df['dataset'] == ds].copy()
        # Montana EDA: run 1 used 273 full bands; exclude for consistency
        if ds == 'montana' and condition_label == 'EDA':
            sub = sub[~((sub['method'] == 'none') &
                        (sub['run'] == 1) &
                        (sub['top_k'] == 273))]
        for m in METHOD_ORDER:
            msub = sub[sub['method'] == m]
            if msub.empty:
                continue
            grp = msub.groupby(['top_k', 'model'])['OA'].mean().reset_index()
            best = grp.loc[grp['OA'].idxmax()]
            sel = msub[(msub['top_k'] == best['top_k']) &
                       (msub['model'] == best['model'])]
            n = len(sel)
            rows.append({
                'condition': condition_label,
                'dataset': ds,
                'method': m,
                'top_k': int(best['top_k']),
                'model': best['model'],
                'OA_mean': sel['OA'].mean(),
                'OA_std': sel['OA'].std(ddof=1) if n > 1 else 0.0,
                'kappa_mean': sel['kappa'].mean(),
                'kappa_std': sel['kappa'].std(ddof=1) if n > 1 else 0.0,
                'f1_mean': sel['f1_macro'].mean(),
                'f1_std': sel['f1_macro'].std(ddof=1) if n > 1 else 0.0,
                'n_runs': n,
            })
    return pd.DataFrame(rows)


def fmt(mean, std):
    return f"{mean:.5f} +/- {std:.5f}"


def cfg_str(method, top_k, model):
    if method == 'none':
        return f"All | {model}"
    return f"Top {int(top_k)} | {model}"


def print_dataset_table(ds, eb, nb):
    tnum = TABLE_NUMS[ds]
    print(f"\n{'=' * 112}")
    print(f"  Table {tnum}: {DATASET_LABELS[ds]}")
    print(f"  Results reported as mean +/- std over five independent runs.")
    print(f"{'=' * 112}")

    # Build dicts keyed by method (avoids set_index issues)
    e_rows = {row['method']: row for _, row in eb[eb['dataset'] == ds].iterrows()}
    n_rows = {row['method']: row for _, csv in nb[nb['dataset'] == ds].iterrows()}

    # WITH EDA
    print(f"\n  WITH EDA")
    print(f"  {'Technique':<22} {'Configuration':<22} {'OA (%)':<30} {'Kappa':<28} {'F1 (%)':<30} Runs")
    print(f"  {'-' * 22} {'-' * 22} {'-' * 30} {'-' * 28} {'-' * 30} {'-' * 4}")
    for m in METHOD_ORDER:
        if m not in e_rows:
            continue
        r = e_rows[m]
        print(f"  {METHOD_LABELS[m]:<22} "
              f"{cfg_str(m, r['top_k'], r['model']):<22} "
              f"{fmt(r['OA_mean'], r['OA_std']):<30} "
              f"{fmt(r['kappa_mean'], r['kappa_std']):<28} "
              f"{fmt(r['f1_mean'], r['f1_std']):<30} "
              f"{int(r['n_runs'])}")

    # WITHOUT EDA + delta
    print(f"\n  WITHOUT EDA  (Delta = EDA - No-EDA)")
    print(f"  {'Technique':<22} {'Configuration':<22} {'OA (%)':<30} {'Kappa':<28} {'F1 (%)':<30} dOA / dKappa / dF1")
    print(f"  {'-' * 22} {'-' * 22} {'-' * 30} {'-' * 28} {'-' * 30} {'-' * 38}")
    for m in METHOD_ORDER:
        if m not in n_rows:
            continue
        r = n_rows[m]
        if m in e_rows:
            e = e_rows[m]
            delta = (f"{e['OA_mean'] - r['OA_mean']:+.5f} / "
                     f"{e['kappa_mean'] - r['kappa_mean']:+.5f} / "
                     f"{e['f1_mean'] - r['f1_mean']:+.5f}")
        else:
            delta = "N/A"
        print(f"  {METHOD_LABELS[m]:<22} "
              f"{cfg_str(m, r['top_k'], r['model']):<22} "
              f"{fmt(r['OA_mean'], r['OA_std']):<30} "
              f"{fmt(r['kappa_mean'], r['kappa_std']):<28} "
              f"{fmt(r['f1_mean'], r['f1_std']):<30} "
              f"{delta}")


def print_table8(eb, nb):
    print(f"\n{'=' * 112}")
    print(f"  Table 8: Cross-Dataset Aggregated Summary")
    print(f"  Per-dataset best-config OA/Kappa/F1 averaged across all 6 datasets.")
    print(f"  mean +/- std computed over those 6 per-dataset values.")
    print(f"{'=' * 112}")

    for cond_label, bdf in [('WITH EDA', eb), ('WITHOUT EDA', nb)]:
        print(f"\n  {cond_label}")
        print(f"  {'Technique':<22} {'OA (%) across datasets':<34} {'Kappa':<32} {'F1 (%)'}")
        print(f"  {'-' * 22} {'-' * 34} {'-' * 32} {'-' * 32}")
        for m in METHOD_ORDER:
            sub = bdf[bdf['method'] == m]
            if sub.empty:
                continue
            oa = sub['OA_mean'].values
            k = sub['kappa_mean'].values
            f1 = sub['f1_mean'].values
            print(f"  {METHOD_LABELS[m]:<22} "
                  f"{np.mean(oa):.5f} +/- {np.std(oa, ddof=1):.5f}          "
                  f"{np.mean(k):.5f} +/- {np.std(k, ddof=1):.5f}        "
                  f"{np.mean(f1):.5f} +/- {np.std(f1, ddof=1):.5f}")


def main():
    print("Loading run files ...")
    eda = load_runs(EDA_FILES)
    noeda = load_runs(NOEDA_FILES)
    print(f"  EDA rows: {len(eda)}   NoEDA rows: {len(noeda)}")

    print("Selecting best configs ...")
    eb = best_configs(eda, 'EDA')
    nb = best_configs(noeda, 'NoEDA')

    print("\n\n" + "=" * 112)
    print("  CORRECTED TABLES 2 - 8")
    print("=" * 112)

    for ds in DATASET_ORDER:
        print_dataset_table(ds, eb, nb)

    print_table8(eb, nb)
    print("\n\nDone.")


if __name__ == '__main__':
    main()
