# HYDRA Saliency Reproduction Pipeline

This repository contains a migrated, script-based reproduction pipeline for the HYDRA saliency experiments originally developed in notebook form. The pipeline supports reproducible execution of LR, HYDRA, and MrSQM saliency masking experiments across UCR time series classification datasets, followed by result aggregation, statistical analysis, WindowSHAP comparison, ablation studies, and qualitative saliency visualisation.

The recommended workflow is to execute each `(model, dataset)` pair as an independent process and then combine the resulting per-dataset CSV files. This makes the experiment resumable, reduces memory accumulation in long-running processes, and provides a cleaner structure for open research.

---

## 1. Repository Overview

Expected high-level structure:

```text
HYDRA/
├── main.py
├── classes/
│   ├── clustering.py
│   ├── cluster_analysis.py
│   ├── saliency_evaluator.py
│   ├── saliency_analysis.py
│   ├── windowshap.py
│   ├── saliency_ablation.py
│   ├── qualitative_plots.py
│   └── models/
│       ├── hydra_original/
│       ├── optimised_hydra.py
│       ├── hydra_explainable.py
│       ├── lr_explainable.py
│       └── mrsqm_explainable.py
├── utils/
│   ├── data_utils.py
│   └── explainability.py
├── scripts/
│   ├── test_model_imports.py
│   ├── test_saliency_smoke.py
│   ├── run_saliency_dataset.py
│   ├── run_saliency_all.py
│   ├── combine_saliency_outputs.py
│   ├── run_windowshap_comparison.py
│   ├── run_hydra_ablation.py
│   └── run_qualitative_plot.py
├── data/
│   ├── summary.csv
│   └── ucr_dataset_types.csv
└── outputs/
    ├── clustering/
    └── saliency/
```

---

## 2. Environment Setup

The experiments are intended to be run from WSL/Linux. MrSQM may be unstable or unavailable on native Windows depending on the local installation.

From a WSL terminal:

```bash
cd ~/Research/HYDRA
source ~/envs/ts_env/bin/activate
export PYTHONPATH=$PWD
```

Check the active Python environment:

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

Expected CUDA output, if a compatible NVIDIA GPU is available:

```text
CUDA available: True
GPU: NVIDIA GeForce RTX ...
```

---

## 3. Smoke Tests

Before running the full study, run the import and saliency smoke tests.

### 3.1 Model Import Test

```bash
python scripts/test_model_imports.py
```

Expected ending:

```text
Import test complete.
All model imports passed.
```

This checks that the migrated LR, HYDRA, optimised HYDRA, and MrSQM wrappers can all be imported successfully.

### 3.2 Saliency Smoke Test

```bash
python scripts/test_saliency_smoke.py
```

Expected ending:

```text
Smoke test complete.
All saliency smoke tests passed.
```

This confirms that LR, HYDRA, and MrSQM can:

```text
fit
predict
produce decision scores
generate saliency maps
run top/random/bottom masking
produce bounded relative score-drop outputs
```

---

## 4. Clustering Prerequisite

The saliency analysis uses the signal-morphology cluster assignments produced by the clustering pipeline.

Check that the clustering output exists:

```bash
ls outputs/clustering/csv/ucr_dataset_clusters_k4.csv
```

The WindowSHAP and ablation scripts also use the closest-dataset list:

```bash
ls outputs/clustering/csv/closest_20_datasets_per_cluster.csv
```

If either file is missing, run clustering first. In `main.py`, use:

```python
def main():
    run_clustering_analysis()
```

Then run:

```bash
python main.py
```

Expected clustering outputs include:

```text
outputs/clustering/csv/ucr_dataset_feature_table.csv
outputs/clustering/csv/ucr_dataset_clusters_k4.csv
outputs/clustering/csv/ucr_dataset_clusters_final.csv
outputs/clustering/csv/closest_20_datasets_per_cluster.csv
outputs/clustering/imgs/signal_morphology_clusters_k4.png
```

After these files exist, clustering does not need to be rerun unless the clustering methodology changes.

---

## 5. Per-Dataset Saliency Execution Design

The main saliency experiments should be run using:

```text
scripts/run_saliency_all.py
```

This script launches:

```text
scripts/run_saliency_dataset.py
```

once for each `(model, dataset)` pair.

This is preferred over one monolithic Python process because:

```text
1. completed datasets are checkpointed as individual CSV files
2. interrupted runs can be resumed safely
3. native memory retained by model backends is released when each process exits
4. large UCR experiments are easier to audit and debug
```

Each completed dataset produces:

```text
outputs/saliency/per_dataset/<model>_samples_<DATASET>.csv
outputs/saliency/per_dataset/<model>_summary_<DATASET>.csv
```

Logs are written to:

```text
outputs/saliency/per_dataset/logs/
```

---

## 6. Script Argument Reference

### 6.1 `scripts/run_saliency_all.py`

General form:

```bash
python scripts/run_saliency_all.py \
  --models <MODEL_1> <MODEL_2> ... \
  --output-dir <OUTPUT_DIR> \
  --random-repeats <N> \
  --max-samples <N>
```

Important arguments:

| Argument | Purpose |
|---|---|
| `--models` | One or more of `lr`, `hydra`, `mrsqm` |
| `--datasets` | Optional explicit dataset list. If omitted, all datasets from `data/summary.csv` are used |
| `--summary-csv` | Dataset summary CSV. Default: `data/summary.csv` |
| `--output-dir` | Directory where per-dataset outputs are written |
| `--fractions` | Comma-separated masking fractions. Default: `0.05,0.10,0.20` |
| `--random-repeats` | Number of random masking repeats |
| `--max-samples` | Maximum number of correctly classified test samples per dataset. If omitted, all correct samples are used |
| `--seed` | Random seed. Default: `42` |
| `--overwrite` | Recompute completed datasets instead of skipping |

### 6.2 Resume Behaviour

By default, completed datasets are skipped if both files exist:

```text
<model>_samples_<DATASET>.csv
<model>_summary_<DATASET>.csv
```

To resume an interrupted run, rerun the same command without `--overwrite`.

To force recomputation, add:

```bash
--overwrite
```

---

## 7. Ten-Dataset End-to-End Test

Before running the full archive, run a smaller end-to-end test across ten representative datasets.

### 7.1 LR and HYDRA Test Run

```bash
python scripts/run_saliency_all.py \
  --models lr hydra \
  --datasets GunPoint Coffee CBF ECG200 ItalyPowerDemand ArrowHead OSULeaf Adiac Plane Chinatown \
  --output-dir outputs/saliency/per_dataset_test \
  --random-repeats 5 \
  --max-samples 50
```

### 7.2 MrSQM Test Run

```bash
python scripts/run_saliency_all.py \
  --models mrsqm \
  --datasets GunPoint Coffee CBF ECG200 ItalyPowerDemand ArrowHead OSULeaf Adiac Plane Chinatown \
  --output-dir outputs/saliency/per_dataset_test \
  --random-repeats 3 \
  --max-samples 50
```

### 7.3 Combine Test Outputs

```bash
python scripts/combine_saliency_outputs.py \
  --input-dir outputs/saliency/per_dataset_test \
  --output-dir outputs/saliency/masking_test_full_settings \
  --models lr hydra mrsqm
```

Expected combined files:

```text
outputs/saliency/masking_test_full_settings/lr_samples.csv
outputs/saliency/masking_test_full_settings/lr_summary.csv
outputs/saliency/masking_test_full_settings/hydra_samples.csv
outputs/saliency/masking_test_full_settings/hydra_summary.csv
outputs/saliency/masking_test_full_settings/mrsqm_samples.csv
outputs/saliency/masking_test_full_settings/mrsqm_summary.csv
```

### 7.4 Analyse Test Outputs

In `main.py`, use:

```python
def run_saliency_results_analysis_test():
    analysis = SaliencyResultsAnalysis(
        saliency_output_dir=OUTPUT_DIR / "saliency" / "masking_test_full_settings",
        cluster_csv_path=OUTPUT_DIR / "clustering" / "csv" / "ucr_dataset_clusters_k4.csv",
        output_dir=OUTPUT_DIR / "saliency" / "analysis_test_full_settings",
    )

    results = analysis.run(fraction=0.10)

    print("\nFlip-rate table:")
    print(results["flip_table"].round(2))

    print("\nBounded score-drop table:")
    print(results["bounded_table"].round(3))

    print("\nHYDRA paired tests:")
    print(results["paired_tests"].to_string(index=False, float_format="{:.3e}".format))

    print("\nHYDRA cluster gap table:")
    print(results["cluster_gap"].round(3))

    return results


def main():
    run_saliency_results_analysis_test()
```

Then run:

```bash
python main.py
```

Expected analysis outputs:

```text
outputs/saliency/analysis_test_full_settings/combined_saliency_samples.csv
outputs/saliency/analysis_test_full_settings/combined_saliency_summary.csv
outputs/saliency/analysis_test_full_settings/combined_saliency_summary_with_clusters.csv
outputs/saliency/analysis_test_full_settings/flip_rate_10.csv
outputs/saliency/analysis_test_full_settings/bounded_relative_score_drop_10.csv
outputs/saliency/analysis_test_full_settings/hydra_paired_tests_10.csv
outputs/saliency/analysis_test_full_settings/dataset_consistency_10.csv
outputs/saliency/analysis_test_full_settings/hydra_nonflip_score_effect_10.csv
outputs/saliency/analysis_test_full_settings/hydra_cluster_gap_10.csv
outputs/saliency/analysis_test_full_settings/hydra_within_cluster_tests_10.csv
outputs/saliency/analysis_test_full_settings/hydra_kruskal_cluster_gap_tests_10.csv
```

The ten-dataset test is a functional check only. It is not expected to match final reported results.

---

## 8. Full Core Saliency Study

The current stable reproduction configuration evaluates up to 50 correctly classified samples per dataset for all three models. LR and HYDRA use five random repeats, while MrSQM uses three random repeats due to the cost and memory behaviour of the MrSQM explanation backend.

### 8.1 Run LR and HYDRA

```bash
python scripts/run_saliency_all.py \
  --models lr hydra \
  --output-dir outputs/saliency/per_dataset \
  --random-repeats 5 \
  --max-samples 50
```

### 8.2 Run MrSQM

```bash
python scripts/run_saliency_all.py \
  --models mrsqm \
  --output-dir outputs/saliency/per_dataset \
  --random-repeats 3 \
  --max-samples 50
```

The MrSQM command can be rerun safely if interrupted. Completed datasets are skipped automatically.

### 8.3 Monitor a Running Dataset

List recent logs:

```bash
ls -lt outputs/saliency/per_dataset/logs | head -20
```

Tail a specific dataset log:

```bash
tail -f outputs/saliency/per_dataset/logs/mrsqm_NonInvasiveFetalECGThorax1.log
```

Monitor memory:

```bash
watch -n 1 free -h
```

Monitor GPU usage:

```bash
nvidia-smi
```

### 8.4 Check Completion Counts

```bash
ls outputs/saliency/per_dataset/lr_summary_*.csv | wc -l
ls outputs/saliency/per_dataset/hydra_summary_*.csv | wc -l
ls outputs/saliency/per_dataset/mrsqm_summary_*.csv | wc -l
```

Expected final counts:

```text
128
128
128
```

---

## 9. Partial Analysis During Long Runs

If one model is still running, partial outputs can be combined for sanity checking. For example, if MrSQM has completed 83 datasets:

```bash
python scripts/combine_saliency_outputs.py \
  --input-dir outputs/saliency/per_dataset \
  --output-dir outputs/saliency/masking_partial_mrsqm83 \
  --models lr hydra mrsqm
```

Partial outputs are useful for validating that the pipeline works, but they should not be used as final reported results.

---

## 10. Combine Full Saliency Outputs

Once all model/dataset pairs have completed, combine the per-dataset outputs:

```bash
python scripts/combine_saliency_outputs.py \
  --input-dir outputs/saliency/per_dataset \
  --output-dir outputs/saliency/masking \
  --models lr hydra mrsqm
```

Expected files:

```text
outputs/saliency/masking/lr_samples.csv
outputs/saliency/masking/lr_summary.csv
outputs/saliency/masking/hydra_samples.csv
outputs/saliency/masking/hydra_summary.csv
outputs/saliency/masking/mrsqm_samples.csv
outputs/saliency/masking/mrsqm_summary.csv
```

---

## 11. Final Core Saliency Analysis

In `main.py`, use:

```python
def run_saliency_results_analysis():
    analysis = SaliencyResultsAnalysis(
        saliency_output_dir=OUTPUT_DIR / "saliency" / "masking",
        cluster_csv_path=OUTPUT_DIR / "clustering" / "csv" / "ucr_dataset_clusters_k4.csv",
        output_dir=OUTPUT_DIR / "saliency" / "analysis",
    )

    results = analysis.run(fraction=0.10)

    print("\nFlip-rate table:")
    print(results["flip_table"].round(2))

    print("\nBounded score-drop table:")
    print(results["bounded_table"].round(3))

    print("\nHYDRA paired tests:")
    print(results["paired_tests"].to_string(index=False, float_format="{:.3e}".format))

    print("\nHYDRA cluster gap table:")
    print(results["cluster_gap"].round(3))

    return results


def main():
    run_saliency_results_analysis()
```

Then run:

```bash
python main.py
```

Expected final analysis files:

```text
outputs/saliency/analysis/combined_saliency_samples.csv
outputs/saliency/analysis/combined_saliency_summary.csv
outputs/saliency/analysis/combined_saliency_summary_with_clusters.csv
outputs/saliency/analysis/flip_rate_10.csv
outputs/saliency/analysis/bounded_relative_score_drop_10.csv
outputs/saliency/analysis/hydra_paired_tests_10.csv
outputs/saliency/analysis/dataset_consistency_10.csv
outputs/saliency/analysis/hydra_nonflip_score_effect_10.csv
outputs/saliency/analysis/hydra_cluster_gap_10.csv
outputs/saliency/analysis/hydra_within_cluster_tests_10.csv
outputs/saliency/analysis/hydra_kruskal_cluster_gap_tests_10.csv
```

Main files to inspect:

```text
outputs/saliency/analysis/flip_rate_10.csv
outputs/saliency/analysis/bounded_relative_score_drop_10.csv
outputs/saliency/analysis/hydra_paired_tests_10.csv
```

Expected qualitative pattern:

```text
top masking should generally be more disruptive than random masking,
and random masking should generally be more disruptive than bottom masking.
```

---

## 12. WindowSHAP Comparison

The WindowSHAP comparison uses the closest datasets from each signal-morphology cluster by default:

```text
outputs/clustering/csv/closest_20_datasets_per_cluster.csv
```

### 12.1 WindowSHAP Smoke Test

```bash
python scripts/run_windowshap_comparison.py \
  --datasets GunPoint \
  --output-dir outputs/saliency/windowshap_debug \
  --n-segments 25 \
  --shap-nsamples 100 \
  --max-samples-per-dataset 2
```

Expected outputs:

```text
outputs/saliency/windowshap_debug/hydra_windowshap_GunPoint.csv
outputs/saliency/windowshap_debug/hydra_windowshap_samples.csv
outputs/saliency/windowshap_debug/hydra_windowshap_summary.csv
outputs/saliency/windowshap_debug/hydra_windowshap_timing.csv
outputs/saliency/windowshap_debug/hydra_windowshap_by_dataset.csv
outputs/saliency/windowshap_debug/hydra_windowshap_paired_tests.csv
```

### 12.2 Paper-Style WindowSHAP Run

```bash
python scripts/run_windowshap_comparison.py \
  --output-dir outputs/saliency/windowshap \
  --n-segments 100 \
  --shap-nsamples 500 \
  --max-samples-per-dataset 20
```

Expected outputs:

```text
outputs/saliency/windowshap/hydra_windowshap_samples.csv
outputs/saliency/windowshap/hydra_windowshap_summary.csv
outputs/saliency/windowshap/hydra_windowshap_timing.csv
outputs/saliency/windowshap/hydra_windowshap_by_dataset.csv
outputs/saliency/windowshap/hydra_windowshap_paired_tests.csv
```

---

## 13. HYDRA Saliency Ablation Study

The ablation study compares HYDRA saliency construction variants:

```text
max_only
min_only
min_activation_scaled
combined
```

### 13.1 Ablation Smoke Test

```bash
python scripts/run_hydra_ablation.py \
  --datasets GunPoint \
  --output-dir outputs/saliency/ablation_debug \
  --fraction 0.10 \
  --max-samples-per-dataset 2
```

Expected outputs:

```text
outputs/saliency/ablation_debug/hydra_saliency_ablation_GunPoint.csv
outputs/saliency/ablation_debug/hydra_saliency_ablation_samples.csv
outputs/saliency/ablation_debug/hydra_saliency_ablation_summary.csv
```

### 13.2 Paper-Style Ablation Run

```bash
python scripts/run_hydra_ablation.py \
  --output-dir outputs/saliency/ablation \
  --fraction 0.10 \
  --max-samples-per-dataset 20
```

Expected outputs:

```text
outputs/saliency/ablation/hydra_saliency_ablation_samples.csv
outputs/saliency/ablation/hydra_saliency_ablation_summary.csv
```

---

## 14. Qualitative Saliency Plot

The qualitative plot compares HYDRA, MrSQM, and WindowSHAP saliency on a selected dataset, typically `GunPoint`.

### 14.1 Generate GunPoint Qualitative Figure

```bash
python scripts/run_qualitative_plot.py \
  --dataset GunPoint \
  --output-dir outputs/saliency/imgs \
  --n-segments 100 \
  --shap-nsamples 500
```

Expected output:

```text
outputs/saliency/imgs/GunPoint_hydra_mrsqm_windowshap.png
```

---

## 15. Recommended Full Command Sequence

From a clean WSL session:

```bash
cd ~/Research/HYDRA
source ~/envs/ts_env/bin/activate
export PYTHONPATH=$PWD

python scripts/test_model_imports.py
python scripts/test_saliency_smoke.py

ls outputs/clustering/csv/ucr_dataset_clusters_k4.csv
ls outputs/clustering/csv/closest_20_datasets_per_cluster.csv

python scripts/run_saliency_all.py \
  --models lr hydra \
  --output-dir outputs/saliency/per_dataset \
  --random-repeats 5 \
  --max-samples 50

python scripts/run_saliency_all.py \
  --models mrsqm \
  --output-dir outputs/saliency/per_dataset \
  --random-repeats 3 \
  --max-samples 50

python scripts/combine_saliency_outputs.py \
  --input-dir outputs/saliency/per_dataset \
  --output-dir outputs/saliency/masking \
  --models lr hydra mrsqm

python main.py

python scripts/run_windowshap_comparison.py \
  --output-dir outputs/saliency/windowshap \
  --n-segments 100 \
  --shap-nsamples 500 \
  --max-samples-per-dataset 20

python scripts/run_hydra_ablation.py \
  --output-dir outputs/saliency/ablation \
  --fraction 0.10 \
  --max-samples-per-dataset 20

python scripts/run_qualitative_plot.py \
  --dataset GunPoint \
  --output-dir outputs/saliency/imgs \
  --n-segments 100 \
  --shap-nsamples 500
```

Before running `python main.py`, ensure that `main()` calls:

```python
def main():
    run_saliency_results_analysis()
```

---

## 16. Troubleshooting

### 16.1 MrSQM Memory Growth

If memory usage grows while MrSQM runs, this is expected behaviour for the backend in long experiments. The per-dataset runner mitigates this by launching each dataset in a fresh Python process.

Monitor memory with:

```bash
watch -n 1 free -h
```

Resume the run with the same command if interrupted. Do not add `--overwrite` unless intentionally recomputing all datasets.

### 16.2 Logs Not Updating

If log files do not update live, ensure `scripts/run_saliency_all.py` launches child processes with unbuffered output:

```python
cmd = [
    sys.executable,
    "-u",
    str(script),
    ...
]
```

Then follow logs with:

```bash
tail -f outputs/saliency/per_dataset/logs/<model>_<DATASET>.log
```

### 16.3 CUDA Not Visible

Check:

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
PY
```

and:

```bash
nvidia-smi
```

If CUDA is unavailable, HYDRA will run on CPU unless configured otherwise.

### 16.4 Partial or Corrupted Dataset Outputs

If a run is interrupted during a dataset, delete only that dataset’s incomplete pair before resuming:

```bash
rm -f outputs/saliency/per_dataset/hydra_samples_<DATASET>.csv
rm -f outputs/saliency/per_dataset/hydra_summary_<DATASET>.csv
```

Do not delete completed datasets unless intentionally recomputing.

---

## 17. Reproducibility Notes

The default random seed is:

```text
42
```

Core masking fractions are:

```text
0.05, 0.10, 0.20
```

The principal reported tables use the 10% masking fraction:

```text
fraction = 0.10
```

The final aggregate results should be produced from combined per-dataset raw outputs rather than from intermediate printed console output.

For manuscript reporting, distinguish clearly between:

```text
partial validation runs
ten-dataset smoke or integration tests
full 128-dataset archive runs
WindowSHAP subset runs
ablation subset runs
```

Only full archive results should be used for final headline tables unless otherwise stated.
