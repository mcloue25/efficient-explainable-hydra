# HYDRA Saliency Reproduction Guide

This README explains how to run the migrated saliency pipeline from a clean WSL environment. It starts with smoke tests, regenerates the main paper saliency tables from the included combined CSVs, then reruns the WindowSHAP comparison, HYDRA saliency ablation and qualitative figures.

The recommended setup is to run each dataset/model pair as an independent process. This is especially important for MrSQM because its explanation backend can retain native memory inside a long-running Python process. Running per dataset keeps the experiment reproducible and makes interrupted runs resumable.

---

## Environment Setup

This project uses **Python 3.13+**. You can set up the environment using standard Python `venv` or `uv` (recommended). The primary dependencies are `torch`, `scikit-learn`, `numpy`, and `aeon` (for UCR dataset loading).

### Option 1: Using standard Python `venv`
If you prefer standard Python tools, a `requirements.txt` file is provided. Set up your virtual environment and install the dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Option 2: Using `uv` (Recommended)
If you have [uv](https://github.com/astral-sh/uv) installed, the project includes a `pyproject.toml` file. Simply run:
```bash
uv sync
source .venv/bin/activate
```

---

## Interactive Example & Dynamic Batching

If you are looking for an interactive demonstration, the root directory contains an example notebook:
* `./hydra_coffee_example.ipynb`

This notebook illustrates the updated HYDRA model, saliency map generation on the Coffee dataset, and the sequence-length-aware **Dynamic Batching** strategy. 

## Hardware Optimisation Results

The empirical results of the target sequence length optimisation and latency model comparisons across **M1, M2, and M4 Mac** systems (evaluated over 5 seeds) can all be found in the following directory:
* `./results/`

---


## 1. Environment setup

Run from a Linux, macOS, or WSL terminal at the repository root:

```bash
# Clone or enter the repository
cd path/to/HYDRA

# Activate your Python environment
source path/to/your/venv/bin/activate

# Make local project imports available
export PYTHONPATH=$PWD
```

Check the active environment:

```bash
which python
python --version
```

Optional CUDA check:

```bash
python - <<'PY'
import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

---

## 2. Import and smoke tests

Run these before any long experiment.

### 2.1 Model import test

```bash
python scripts/test_model_imports.py
```

Expected ending:

```text
Import test complete.
All model imports passed.
```

### 2.2 Saliency smoke test

```bash
python scripts/test_saliency_smoke.py
```

Expected ending:

```text
Smoke test complete.
All saliency smoke tests passed.
```

The smoke test checks that LR, HYDRA, and MrSQM can fit, predict, produce decision scores, generate saliency maps, and run top / random / bottom masking.

### 2.3 Syntax check

```bash
python -m compileall main.py classes scripts utils
```

Expected: no errors.

---

## 3. Clustering prerequisite

The saliency analysis uses signal-morphology cluster assignments.

Generate them once with:

```bash
python main.py cluster
```

Expected key outputs:

```text
outputs/clustering/csv/ucr_dataset_clusters_k4.csv
outputs/clustering/csv/closest_20_datasets_per_cluster.csv
outputs/clustering/csv/cluster_feature_tests.csv
outputs/clustering/csv/ucr_dataset_clusters_final.csv
outputs/clustering/imgs/signal_morphology_clusters_k4.png
```

Check the files exist:

```bash
ls outputs/clustering/csv/ucr_dataset_clusters_k4.csv
ls outputs/clustering/csv/closest_20_datasets_per_cluster.csv
```

The `closest_20_datasets_per_cluster.csv` file is used later for the WindowSHAP and ablation subset. Selecting the first 10 closest datasets per cluster gives the 38-dataset report subset because cluster 0 contains only 8 datasets.

---

## 4. Reproduce main paper saliency results from included CSVs

The repository includes the combined original saliency outputs used for the main perturbation tables:

```text
outputs/saliency/masking_original_report/lr_samples.csv
outputs/saliency/masking_original_report/lr_summary.csv
outputs/saliency/masking_original_report/hydra_samples.csv
outputs/saliency/masking_original_report/hydra_summary.csv
outputs/saliency/masking_original_report/mrsqm_samples.csv
outputs/saliency/masking_original_report/mrsqm_summary.csv
```

These files allow the examiner to regenerate the main saliency analysis tables and figures without rerunning the full LR, HYDRA and MrSQM perturbation study. The full rerun commands are still provided later as an optional reproduction route.

### 4.1 Generate clustering files

The saliency analysis, WindowSHAP comparison and ablation study use the signal-morphology cluster assignments. Generate them once with:

```bash
python main.py cluster
```

Expected key outputs:

```text
outputs/clustering/csv/ucr_dataset_clusters_k4.csv
outputs/clustering/csv/closest_20_datasets_per_cluster.csv
outputs/clustering/csv/cluster_feature_tests.csv
outputs/clustering/csv/ucr_dataset_clusters_final.csv
outputs/clustering/imgs/signal_morphology_clusters_k4.png
```

### 4.2 Regenerate main saliency analysis tables

```bash
python main.py analyse-saliency \
  --saliency-output-dir outputs/saliency/masking_original_report \
  --analysis-dir outputs/saliency/analysis_original_report \
  --fraction 0.10
```

Expected key outputs:

```text
outputs/saliency/analysis_original_report/flip_rate_10.csv
outputs/saliency/analysis_original_report/bounded_relative_score_drop_10.csv
outputs/saliency/analysis_original_report/hydra_paired_tests_10.csv
outputs/saliency/analysis_original_report/dataset_consistency_10.csv
outputs/saliency/analysis_original_report/hydra_nonflip_score_effect_10.csv
outputs/saliency/analysis_original_report/hydra_cluster_gap_10.csv
outputs/saliency/analysis_original_report/hydra_within_cluster_tests_10.csv
outputs/saliency/analysis_original_report/hydra_kruskal_cluster_gap_tests_10.csv
```

The expected headline values at `fraction = 0.10` are:

```text
Flip rate (%)
HYDRA: 26.50 / 16.72 / 11.38
LR:    40.21 / 22.69 / 17.73
MrSQM: 20.62 / 12.20 / 5.95

Bounded relative score drop
HYDRA: 0.372 / 0.278 / 0.123
LR:    0.190 / 0.070 / 0.015
MrSQM: 0.365 / 0.228 / 0.115
```

### 4.3 Regenerate main saliency figures

```bash
python main.py plot-saliency \
  --analysis-dir outputs/saliency/analysis_original_report \
  --figure-dir outputs/saliency/imgs
```

Expected outputs include:

```text
outputs/saliency/imgs/updated_prediction_bounded_score_drop.png
outputs/saliency/imgs/updated_prediction_bounded_score_drop.pdf
outputs/saliency/imgs/mean_score_drop_by_cluster.png
outputs/saliency/imgs/mean_score_drop_by_cluster.pdf
outputs/saliency/imgs/hydra_cluster_score_drop_heatmap_expansive.png
outputs/saliency/imgs/hydra_cluster_score_drop_heatmap_expansive.pdf
```


## 5. Regenerate WindowSHAP comparison

The WindowSHAP comparison is a separate report experiment. It does not use `masking_original_report` directly; instead, it reruns HYDRA saliency and WindowSHAP on the stratified signal-morphology subset.

The paper-level WindowSHAP comparison uses:

```text
datasets_per_cluster = 10
n_segments = 100
shap_nsamples = 500
max_samples_per_dataset = 20
seed = 42
```

Run:

```bash
python scripts/run_windowshap_comparison.py \
  --closest-csv outputs/clustering/csv/closest_20_datasets_per_cluster.csv \
  --cluster-csv outputs/clustering/csv/ucr_dataset_clusters_k4.csv \
  --output-dir outputs/saliency/windowshap \
  --datasets-per-cluster 10 \
  --n-segments 100 \
  --shap-nsamples 500 \
  --max-samples-per-dataset 20 \
  --seed 42
```

Expected outputs:

```text
outputs/saliency/windowshap/hydra_windowshap_samples.csv
outputs/saliency/windowshap/hydra_windowshap_summary.csv
outputs/saliency/windowshap/hydra_windowshap_timing.csv
outputs/saliency/windowshap/hydra_windowshap_by_dataset.csv
outputs/saliency/windowshap/hydra_windowshap_paired_tests.csv
outputs/saliency/windowshap/hydra_windowshap_samples_with_clusters.csv
outputs/saliency/windowshap/hydra_windowshap_cluster_summary.csv
outputs/saliency/windowshap/hydra_windowshap_cluster_kruskal_tests.csv
outputs/saliency/windowshap/hydra_windowshap_within_cluster_tests.csv
```

The expected report-level trend is that WindowSHAP selects more perturbation-sensitive regions, while HYDRA projection saliency is substantially faster per sample.


## 6. Regenerate HYDRA saliency ablation

The HYDRA saliency construction ablation is also a separate report experiment. It evaluates different ways of projecting HYDRA max/min features back to the temporal domain.

The paper-level ablation uses:

```text
datasets_per_cluster = 10
max_samples_per_dataset = 20
fraction = 0.10
seed = 42
variants = max_only, min_only, min_activation_scaled, combined
```

Run:

```bash
python scripts/run_hydra_ablation.py \
  --closest-csv outputs/clustering/csv/closest_20_datasets_per_cluster.csv \
  --output-dir outputs/saliency/ablation \
  --datasets-per-cluster 10 \
  --fraction 0.10 \
  --max-samples-per-dataset 20 \
  --seed 42
```

Expected outputs:

```text
outputs/saliency/ablation/hydra_saliency_ablation_samples.csv
outputs/saliency/ablation/hydra_saliency_ablation_summary.csv
```

The expected report-level pattern is that the combined max-plus-min rule gives the strongest bounded score-drop result, while the max-only variant can produce the highest flip rate.


## 7. Regenerate qualitative GunPoint figure

The qualitative figure compares HYDRA saliency, MrSQM saliency and WindowSHAP on the GunPoint dataset.

```bash
python scripts/run_qualitative_plot.py \
  --dataset GunPoint \
  --output-dir outputs/saliency/imgs \
  --n-segments 100 \
  --shap-nsamples 500
```

Expected outputs include:

```text
outputs/saliency/imgs/GunPoint_hydra_mrsqm_windowshap.png
outputs/saliency/imgs/GunPoint_saliency_comparison_100_500.png
outputs/saliency/imgs/Gunpoint_saliency_comparison_100_500.png
```

The lowercase-`p` filename is included for compatibility with the paper figure path.


## 8. Export report-ready LaTeX table snippets

After running the saliency analysis, WindowSHAP comparison and ablation commands, export the LaTeX table snippets:

```bash
python scripts/export_report_tables.py \
  --analysis-dir outputs/saliency/analysis_original_report \
  --windowshap-dir outputs/saliency/windowshap \
  --ablation-dir outputs/saliency/ablation \
  --cluster-csv outputs/clustering/csv/ucr_dataset_clusters_k4.csv \
  --output-dir outputs/report_tables
```

Expected outputs include:

```text
outputs/report_tables/flip_rate_10_latex.tex
outputs/report_tables/bounded_relative_score_drop_10_latex.tex
outputs/report_tables/hydra_paired_tests_10_latex.tex
outputs/report_tables/dataset_consistency_10_latex.tex
outputs/report_tables/hydra_nonflip_score_effect_10_latex.tex
outputs/report_tables/hydra_cluster_gap_10_latex.tex
outputs/report_tables/hydra_windowshap_summary_latex.tex
outputs/report_tables/hydra_windowshap_timing_latex.tex
outputs/report_tables/hydra_saliency_ablation_summary_latex.tex
outputs/report_tables/ucr_dataset_cluster_assignments_latex.tex
```


## 9. Complete paper-level command sequence using included saliency CSVs

From a clean terminal session at the repository root:

```bash
source path/to/your/venv/bin/activate
export PYTHONPATH=$PWD

python scripts/test_model_imports.py
python scripts/test_saliency_smoke.py
python -m compileall main.py classes scripts utils

python main.py cluster

python main.py analyse-saliency \
  --saliency-output-dir outputs/saliency/masking_original_report \
  --analysis-dir outputs/saliency/analysis_original_report \
  --fraction 0.10

python main.py plot-saliency \
  --analysis-dir outputs/saliency/analysis_original_report \
  --figure-dir outputs/saliency/imgs

python scripts/run_windowshap_comparison.py \
  --closest-csv outputs/clustering/csv/closest_20_datasets_per_cluster.csv \
  --cluster-csv outputs/clustering/csv/ucr_dataset_clusters_k4.csv \
  --output-dir outputs/saliency/windowshap \
  --datasets-per-cluster 10 \
  --n-segments 100 \
  --shap-nsamples 500 \
  --max-samples-per-dataset 20 \
  --seed 42

python scripts/run_hydra_ablation.py \
  --closest-csv outputs/clustering/csv/closest_20_datasets_per_cluster.csv \
  --output-dir outputs/saliency/ablation \
  --datasets-per-cluster 10 \
  --fraction 0.10 \
  --max-samples-per-dataset 20 \
  --seed 42

python scripts/run_qualitative_plot.py \
  --dataset GunPoint \
  --output-dir outputs/saliency/imgs \
  --n-segments 100 \
  --shap-nsamples 500

python scripts/export_report_tables.py \
  --analysis-dir outputs/saliency/analysis_original_report \
  --windowshap-dir outputs/saliency/windowshap \
  --ablation-dir outputs/saliency/ablation \
  --cluster-csv outputs/clustering/csv/ucr_dataset_clusters_k4.csv \
  --output-dir outputs/report_tables
```


## 10. Optional: rerun the full perturbation saliency study

The included `masking_original_report` files are sufficient to regenerate the main saliency tables and figures. If a full rerun is desired, run the following commands. This can take a long time, especially for MrSQM.

### 10.1 Run LR and HYDRA

```bash
python scripts/run_saliency_all.py \
  --models lr hydra \
  --output-dir outputs/saliency/per_dataset \
  --fractions 0.05,0.10,0.20 \
  --random-repeats 5 \
  --max-samples 50 \
  --seed 42
```

### 10.2 Run MrSQM

```bash
python scripts/run_saliency_all.py \
  --models mrsqm \
  --output-dir outputs/saliency/per_dataset \
  --fractions 0.05,0.10,0.20 \
  --random-repeats 3 \
  --max-samples 50 \
  --seed 42
```

Completed datasets are skipped automatically. Rerun the same command to resume an interrupted run. Use `--overwrite` only if intentionally recomputing completed datasets.

### 10.3 Combine full rerun outputs

```bash
python scripts/combine_saliency_outputs.py \
  --input-dir outputs/saliency/per_dataset \
  --output-dir outputs/saliency/masking \
  --models lr hydra mrsqm
```

Then analyse the full rerun with:

```bash
python main.py analyse-saliency \
  --saliency-output-dir outputs/saliency/masking \
  --analysis-dir outputs/saliency/analysis \
  --fraction 0.10
```


## 11. Sanity checks after running

Check for empty output files:

```bash
find outputs/saliency -type f -size 0
```

Expected: no output.

Check key CSV shapes:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

paths = [
    "outputs/saliency/analysis_original_report/combined_saliency_summary.csv",
    "outputs/saliency/analysis_original_report/combined_saliency_samples.csv",
    "outputs/saliency/windowshap/hydra_windowshap_samples.csv",
    "outputs/saliency/ablation/hydra_saliency_ablation_samples.csv",
]

for p in paths:
    path = Path(p)
    if path.exists():
        df = pd.read_csv(path)
        print(f"{p}: {df.shape}")
    else:
        print(f"MISSING: {p}")
PY
```

Check saliency figures:

```bash
ls outputs/saliency/imgs
ls outputs/saliency/imgs | grep -i gun
```

Check output folder sizes before committing:

```bash
du -sh outputs/saliency/*
```

Usually, commit code, README files, and selected small summary tables/figures. Avoid committing very large per-sample CSVs unless intentionally archiving results.

---

## 12. Troubleshooting

### MrSQM memory growth

MrSQM can retain memory during explanation. The per-dataset runner mitigates this by launching each dataset in a fresh Python process. Resume with the same command if interrupted.

### Logs not updating

If logs do not update live, check that the runner launches child Python with unbuffered output. Follow logs with:

```bash
tail -f outputs/saliency/per_dataset/logs/<model>_<DATASET>.log
```

### Incomplete or corrupted dataset output

If a run stops mid-dataset, delete only the incomplete pair and resume:

```bash
rm -f outputs/saliency/per_dataset/hydra_samples_<DATASET>.csv
rm -f outputs/saliency/per_dataset/hydra_summary_<DATASET>.csv
```

Replace `hydra` and `<DATASET>` as needed.

### `--datasets` error in `main.py`

`main.py` uses singular `--dataset`. For multiple explicit datasets, use the underlying scripts if supported, or run one dataset at a time through `main.py`.

Correct:

```bash
python main.py windowshap --dataset GunPoint --output-dir outputs/saliency/windowshap_validation
```

Incorrect:

```bash
python main.py windowshap --datasets GunPoint Coffee
```

### Plotting duplicate-label errors

The plotting script should aggregate dataset-level rows before plotting. If you see:

```text
ValueError: cannot reindex on an axis with duplicate labels
```

make sure `scripts/plot_saliency_figures.py` groups by `fraction`, `mode`, and for cluster heatmaps also by `cluster`, before pivoting/reindexing.

---

## 13. Reproducibility notes

The default random seed is:

```text
42
```

Core masking fractions are:

```text
0.05, 0.10, 0.20
```

The principal reported saliency tables use:

```text
fraction = 0.10
```

For manuscript reporting, keep these result types separate:

```text
validation runs
partial MrSQM runs
ten-dataset integration tests
full 128-dataset perturbation runs
38-dataset WindowSHAP subset runs
38-dataset ablation subset runs
included original-report CSV analysis
```

Only full archive results or explicitly labelled included original-report CSV analysis should be used for headline paper tables.