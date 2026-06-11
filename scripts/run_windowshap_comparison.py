from pathlib import Path
import argparse
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from classes.windowshap import WindowSHAPComparison


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--closest-csv", default="outputs/clustering/csv/closest_20_datasets_per_cluster.csv")
    parser.add_argument("--cluster-csv", default="outputs/clustering/csv/ucr_dataset_clusters_k4.csv")
    parser.add_argument("--output-dir", default="outputs/saliency/windowshap")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--n-segments", type=int, default=100)
    parser.add_argument("--shap-nsamples", type=int, default=500)
    parser.add_argument("--max-samples-per-dataset", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.datasets:
        datasets = args.datasets
    else:
        closest = pd.read_csv(PROJECT_ROOT / args.closest_csv)
        datasets = closest["dataset"].dropna().unique().tolist()

    comparison = WindowSHAPComparison(
        datasets=datasets,
        output_dir=PROJECT_ROOT / args.output_dir,
        n_segments=args.n_segments,
        shap_nsamples=args.shap_nsamples,
        max_samples_per_dataset=args.max_samples_per_dataset,
        seed=args.seed,
    )

    results = comparison.run()
    comparison.cluster_level_analysis(PROJECT_ROOT / args.cluster_csv)

    print("\nWindowSHAP summary:")
    print(results["summary"].round(3))

    print("\nTiming summary:")
    print(results["timing"].round(3))


if __name__ == "__main__":
    main()
