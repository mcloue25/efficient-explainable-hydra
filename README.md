# HYDRA Saliency Reproduction Guide

This README explains how to run the migrated saliency pipeline from a clean WSL environment. It starts with smoke tests, then runs the full LR, HYDRA and MrSQM saliency experiments, combines the per-dataset outputs, and runs the final analysis tables.

The recommended setup is to run each dataset/model pair as an independent process. This is especially important for MrSQM because its explanation backend can retain native memory inside a long-running Python process. Running per dataset keeps the experiment reproducible and makes interrupted runs resumable.

---

## 1. Enter the project in WSL

From WSL:

```bash
cd ~/Research/HYDRA
source ~/envs/ts_env/bin/activate
export PYTHONPATH=$PWD
```

Check that Python is using the expected virtual environment:

```bash
which python
python --version
```

Optional: check CUDA is visible to PyTorch:

```bash
python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PY
```

---

## 2. Run import smoke tests

This checks that all migrated model files import correctly:

```bash
python scripts/test_model_imports.py
```

Expected ending:

```text
Import test complete.
All model imports passed.
```

Then run the saliency smoke test:

```bash
python scripts/test_saliency_smoke.py
```

Expected ending:

```text
Smoke test complete.
All saliency smoke tests passed.
```

This confirms that LR, HYDRA and MrSQM can fit, predict, explain, and run top/random/bottom masking on a small dataset.

---

## 3. Make sure clustering outputs exist

The final saliency analysis uses the clustering assignments here:

```text
outputs/clustering/csv/ucr_dataset_clusters_k4.csv
```

Check that it exists:

```bash
ls outputs/clustering/csv/ucr_dataset_clusters_k4.csv
```

If it does not exist, run the clustering pipeline first. In `main.py`, enable:

```python
def main():
    run_clustering_analysis()
```

Then run:

```bash
python main.py
```

Expected important output:

```text
outputs/clustering/csv/ucr_dataset_clusters_k4.csv
outputs/clustering/csv/closest_20_datasets_per_cluster.csv
```

After clustering has run once, you do not need to rerun it for every saliency experiment.

---

## 4. Optional clean start for saliency outputs

Only run this if you want to delete previous saliency outputs and start fresh:

```bash
rm -rf outputs/saliency/per_dataset
rm -rf outputs/saliency/masking
rm -rf outputs/saliency/analysis
```

Do not delete these folders if you are resuming a partially completed run.

---

## 5. Run a 10-dataset end-to-end test

Before running the full UCR archive, run a smaller test across ten datasets.

### 5.1 Run MrSQM test

MrSQM should be run with the stable settings used in the original workaround:

```bash
python scripts/run_saliency_all.py \
  --models mrsqm \
  --datasets GunPoint Coffee CBF ECG200 ItalyPowerDemand ArrowHead OSULeaf Adiac Plane Chinatown \
  --output-dir outputs/saliency/per_dataset_test \
  --random-repeats 3 \
  --max-samples 50
```

### 5.2 Run LR and HYDRA test

```bash
python scripts/run_saliency_all.py \
  --models lr hydra \
  --datasets GunPoint Coffee CBF ECG200 ItalyPowerDemand ArrowHead OSULeaf Adiac Plane Chinatown \
  --output-dir outputs/saliency/per_dataset_test \
  --random-repeats 5
```

### 5.3 Combine the 10-dataset test outputs

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

### 5.4 Run the 10-dataset analysis

In `main.py`, point the saliency analysis at the test output folder:

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

Expected analysis output files:

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

The 10-dataset values are only a sanity check. They are not expected to exactly match the final paper tables.

---

## 6. Run the full LR and HYDRA saliency experiments

For the full archive, run LR and HYDRA with all test samples and five random repeats:

```bash
python scripts/run_saliency_all.py \
  --models lr hydra \
  --output-dir outputs/saliency/per_dataset \
  --random-repeats 5
```

This writes one pair of CSV files per model/dataset:

```text
outputs/saliency/per_dataset/lr_samples_<DATASET>.csv
outputs/saliency/per_dataset/lr_summary_<DATASET>.csv
outputs/saliency/per_dataset/hydra_samples_<DATASET>.csv
outputs/saliency/per_dataset/hydra_summary_<DATASET>.csv
```

The runner skips completed model/dataset pairs unless `--overwrite` is used.

---

## 7. Run the full MrSQM saliency experiment

Run MrSQM separately with the stable settings:

```bash
python scripts/run_saliency_all.py \
  --models mrsqm \
  --output-dir outputs/saliency/per_dataset \
  --random-repeats 3 \
  --max-samples 50
```

This writes:

```text
outputs/saliency/per_dataset/mrsqm_samples_<DATASET>.csv
outputs/saliency/per_dataset/mrsqm_summary_<DATASET>.csv
```

MrSQM is run per dataset in a fresh Python process to avoid native memory accumulation in long-running jobs.

To monitor memory while MrSQM runs:

```bash
watch -n 1 free -h
```

Failed runs, if any, are logged under:

```text
outputs/saliency/per_dataset/logs/
```

---

## 8. Combine the full saliency outputs

After LR, HYDRA and MrSQM have finished, combine the per-dataset CSVs into the final analysis folder:

```bash
python scripts/combine_saliency_outputs.py \
  --input-dir outputs/saliency/per_dataset \
  --output-dir outputs/saliency/masking \
  --models lr hydra mrsqm
```

Expected combined files:

```text
outputs/saliency/masking/lr_samples.csv
outputs/saliency/masking/lr_summary.csv
outputs/saliency/masking/hydra_samples.csv
outputs/saliency/masking/hydra_summary.csv
outputs/saliency/masking/mrsqm_samples.csv
outputs/saliency/masking/mrsqm_summary.csv
```

---

## 9. Run the final saliency analysis

In `main.py`, point the analysis class at the full combined saliency folder:

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

---

## 10. Check the main paper tables

The most important outputs are:

```text
outputs/saliency/analysis/flip_rate_10.csv
outputs/saliency/analysis/bounded_relative_score_drop_10.csv
outputs/saliency/analysis/hydra_paired_tests_10.csv
```

For the 10% masking setting, the expected pattern is:

```text
top > random > bottom
```

for most model/metric combinations, especially in the full archive results.

---

## 11. Resuming interrupted runs

The per-dataset runner skips completed runs automatically. If the process stops halfway through, rerun the same command:

```bash
python scripts/run_saliency_all.py \
  --models mrsqm \
  --output-dir outputs/saliency/per_dataset \
  --random-repeats 3 \
  --max-samples 50
```

Completed datasets are skipped, and incomplete ones continue.

To force recomputation, add:

```bash
--overwrite
```

Example:

```bash
python scripts/run_saliency_all.py \
  --models mrsqm \
  --output-dir outputs/saliency/per_dataset \
  --random-repeats 3 \
  --max-samples 50 \
  --overwrite
```

---

## 12. Recommended full command sequence

From a clean WSL session:

```bash
cd ~/Research/HYDRA
source ~/envs/ts_env/bin/activate
export PYTHONPATH=$PWD

python scripts/test_model_imports.py
python scripts/test_saliency_smoke.py

ls outputs/clustering/csv/ucr_dataset_clusters_k4.csv

python scripts/run_saliency_all.py \
  --models lr hydra \
  --output-dir outputs/saliency/per_dataset \
  --random-repeats 5

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
```

Before running the last `python main.py`, make sure `main()` is set to:

```python
def main():
    run_saliency_results_analysis()
```

