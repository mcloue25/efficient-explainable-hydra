from pathlib import Path

import pandas as pd

from classes.clustering import SignalMorphologyClusterer
from classes.cluster_analysis import ClusterAnalysis
from classes.saliency_evaluator import SaliencyEvaluator
from classes.saliency_analysis import SaliencyResultsAnalysis
# from classes.windowshap import WindowSHAPComparison
# from classes.saliency_ablation import HydraSaliencyAblation
# from classes.qualitative_plots import QualitativeSaliencyPlotter


SEED = 42
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")


def load_dataset_names():
    summary = pd.read_csv(DATA_DIR / "summary.csv")
    return sorted(summary["dataset"].dropna().unique())


def run_clustering_analysis():
    datasets = load_dataset_names()

    clusterer = SignalMorphologyClusterer(
        datasets=datasets,
        output_dir=OUTPUT_DIR / "clustering",
        n_clusters=4,
        seed=SEED,
    )

    clustered_df = clusterer.run()

    analysis = ClusterAnalysis(clusterer)
    cluster_results = analysis.run()

    ucr_types = pd.read_csv(DATA_DIR / "ucr_dataset_types.csv")
    type_metrics = analysis.compare_to_ucr_types(ucr_types)
    analysis.type_contingency(ucr_types)
    analysis.type_summary(ucr_types)

    return clustered_df, cluster_results, type_metrics


# def run_saliency_analysis():
#     datasets = load_dataset_names()

#     evaluator = SaliencyEvaluator(
#         datasets=datasets,
#         output_dir=OUTPUT_DIR / "saliency" / "masking",
#         seed=SEED,
#     )

#     evaluator.run(model_names=("lr", "hydra", "mrsqm"))

#     analysis = SaliencyResultsAnalysis(
#         saliency_output_dir=OUTPUT_DIR / "saliency" / "masking",
#         cluster_csv_path=OUTPUT_DIR / "clustering" / "csv" / "ucr_dataset_clusters_k4.csv",
#         output_dir=OUTPUT_DIR / "saliency" / "analysis",
#     )

#     return analysis.run()


def run_saliency_analysis():
    datasets = [
        "GunPoint",
        "Coffee",
        "CBF",
        "ECG200",
        "ItalyPowerDemand",
        "ArrowHead",
        "OSULeaf",
        "Adiac",
        "Plane",
        "Chinatown",
    ]

    evaluator = SaliencyEvaluator(
        datasets=datasets,
        output_dir=OUTPUT_DIR / "saliency" / "masking_test_full_settings",
        fractions=(0.05, 0.10, 0.20),
        random_repeats=3,
        max_samples=50,
        only_correct=True,
        seed=SEED,
    )

    return evaluator.run(model_names=("mrsqm",))


def run_mrsqm_analysis_test():
    analysis = SaliencyResultsAnalysis(
        saliency_output_dir=OUTPUT_DIR / "saliency" / "masking_test_full_settings",
        cluster_csv_path=OUTPUT_DIR / "clustering" / "csv" / "ucr_dataset_clusters_k4.csv",
        output_dir=OUTPUT_DIR / "saliency" / "analysis_mrsqm_test",
    )

    analysis.load_results()

    print("\nLoaded samples:")
    print(analysis.samples.shape)

    print("\nLoaded summary:")
    print(analysis.summary.shape)

    print("\nMrSQM flip-rate table:")
    print(
        analysis.table_at_fraction(
            metric="flip_rate",
            fraction=0.10,
            as_percent=True,
        ).round(2)
    )

    print("\nMrSQM bounded score-drop table:")
    print(
        analysis.table_at_fraction(
            metric="mean_bounded_relative_score_drop",
            fraction=0.10,
            as_percent=False,
        ).round(3)
    )



def run_saliency_results_analysis_test():
    analysis = SaliencyResultsAnalysis(
        saliency_output_dir=OUTPUT_DIR / "saliency" / "masking_test",
        cluster_csv_path=OUTPUT_DIR / "clustering" / "csv" / "ucr_dataset_clusters_k4.csv",
        output_dir=OUTPUT_DIR / "saliency" / "analysis_test",
    )

    results = analysis.run(fraction=0.10)

    print("\nFlip-rate table:")
    print(results["flip_table"].round(2))

    print("\nBounded score-drop table:")
    print(results["bounded_table"].round(3))

    print("\nHYDRA paired tests:")
    print(results["paired_tests"].to_string(index=False, float_format="{:.3e}".format))

    if "cluster_gap" in results:
        print("\nHYDRA cluster gap table:")
        print(results["cluster_gap"].round(3))

    return results




def run_windowshap_analysis():
    closest = pd.read_csv(
        OUTPUT_DIR / "clustering" / "csv" / "closest_20_datasets_per_cluster.csv"
    )

    datasets = closest["dataset"].unique()

    comparison = WindowSHAPComparison(
        datasets=datasets,
        output_dir=OUTPUT_DIR / "saliency" / "windowshap",
        n_segments=100,
        shap_nsamples=500,
        max_samples_per_dataset=20,
        seed=SEED,
    )

    return comparison.run()


def run_saliency_ablation():
    closest = pd.read_csv(
        OUTPUT_DIR / "clustering" / "csv" / "closest_20_datasets_per_cluster.csv"
    )

    datasets = closest["dataset"].unique()

    ablation = HydraSaliencyAblation(
        datasets=datasets,
        output_dir=OUTPUT_DIR / "saliency" / "ablation",
        fraction=0.10,
        max_samples_per_dataset=20,
        seed=SEED,
    )

    return ablation.run()


def run_qualitative_plots():
    plotter = QualitativeSaliencyPlotter(
        output_dir=OUTPUT_DIR / "saliency" / "imgs",
        n_segments=100,
        shap_nsamples=500,
        seed=SEED,
    )

    return plotter.plot_hydra_mrsqm_windowshap(dataset="GunPoint")


def main():
    # run_clustering_analysis()

    # Add these one at a time as each class is migrated and tested:
    # run_saliency_analysis()
    
    run_mrsqm_analysis_test()

    # run_saliency_results_analysis_test()
    # run_windowshap_analysis()
    # run_saliency_ablation()
    # run_qualitative_plots()


if __name__ == "__main__":
    main()
















# from pathlib import Path

# import pandas as pd

# from classes.clustering import SignalMorphologyClusterer
# from classes.cluster_analysis import ClusterAnalysis

# from classes.models.hydra_explainable import HydraModelExplainable
# from classes.models.lr_explainable import LRRawExplainableModel
# from classes.models.mrsqm_explainable import MrSQMExplainableModel


# SEED = 42
# N_CLUSTERS = 4

# DATA_DIR = Path("data")
# OUTPUT_DIR = Path("outputs")


# def load_dataset_names(path=DATA_DIR / "summary.csv"):
#     ''' 
#     '''
#     summary = pd.read_csv(path)
#     return sorted(summary["dataset"].dropna().unique())


# def run_clustering_analysis():
#     ''' 
#     '''
#     clustering_output_dir = OUTPUT_DIR / "clustering"

#     datasets = load_dataset_names()

#     clusterer = SignalMorphologyClusterer(
#         datasets=datasets,
#         output_dir=clustering_output_dir,
#         n_clusters=N_CLUSTERS,
#         seed=SEED,
#     )

#     clustered_df = clusterer.run()

#     analysis = ClusterAnalysis(clusterer)
#     results = analysis.run()

#     print("\nRows:", len(clustered_df))

#     print("\nCluster sizes:")
#     print(clustered_df["cluster"].value_counts().sort_index())

#     print("\nCluster feature profile:")
#     print(results["report_feature_profile"].round(2))

#     print("\nKruskal-Wallis tests:")
#     print(results["feature_tests"].to_string(index=False, float_format="{:.3e}".format))

#     print("\nCandidate k values with no singleton clusters and minimum cluster size >= 5:")
#     k_eval = results["k_eval"]
#     candidate_k = k_eval[(k_eval["n_singletons"] == 0) & (k_eval["min_cluster_size"] >= 5)]
#     print(candidate_k.round(3))

#     print("\nClosest datasets to each cluster centroid:")
#     print(results["closest_datasets"].head(20))

#     ucr_types = pd.read_csv(DATA_DIR / "ucr_dataset_types.csv")
#     type_metrics = analysis.compare_to_ucr_types(ucr_types)
#     type_tables = analysis.type_contingency(ucr_types)
#     type_summary = analysis.type_summary(ucr_types)

#     print("\nUCR Type comparison:")
#     print(type_metrics.round(3))

#     print("\nDominant UCR Type per cluster:")
#     print(type_summary[["cluster", "n_datasets", "dominant_type", "dominant_type_percent"]].round(1))

#     return {
#         "clustered_df": clustered_df,
#         "analysis_results": results,
#         "type_metrics": type_metrics,
#         "type_tables": type_tables,
#         "type_summary": type_summary,
#     }


# def main():
#     run_clustering_analysis()

#     # Later:
#     # run_saliency_analysis()


# if __name__ == "__main__":
#     main()