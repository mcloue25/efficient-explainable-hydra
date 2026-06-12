from pathlib import Path
import argparse

import pandas as pd


CLUSTER_NAMES = {
    0: "High-frequency / high-curvature",
    1: "Smooth / low-complexity",
    2: "Short / moderately rough",
    3: "Spiky / multi-class",
}


def _read_optional(path):
    path = Path(path)
    if not path.exists():
        print(f"[WARN] Missing: {path}")
        return None
    return pd.read_csv(path)


def _write_latex(df, path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df.to_latex(index=kwargs.pop("index", False), escape=False, **kwargs))
    print(f"[OK] {path}")


def export_core_saliency_tables(analysis_dir, output_dir):
    analysis_dir = Path(analysis_dir)
    output_dir = Path(output_dir)

    table_specs = [
        ("flip_rate_10.csv", "flip_rate_10_latex.tex"),
        ("bounded_relative_score_drop_10.csv", "bounded_relative_score_drop_10_latex.tex"),
        ("hydra_paired_tests_10.csv", "hydra_paired_tests_10_latex.tex"),
        ("dataset_consistency_10.csv", "dataset_consistency_10_latex.tex"),
        ("hydra_nonflip_score_effect_10.csv", "hydra_nonflip_score_effect_10_latex.tex"),
        ("hydra_cluster_gap_10.csv", "hydra_cluster_gap_10_latex.tex"),
        ("hydra_within_cluster_tests_10.csv", "hydra_within_cluster_tests_10_latex.tex"),
        ("hydra_kruskal_cluster_gap_tests_10.csv", "hydra_kruskal_cluster_gap_tests_10_latex.tex"),
    ]

    for csv_name, tex_name in table_specs:
        df = _read_optional(analysis_dir / csv_name)
        if df is not None:
            _write_latex(df, output_dir / tex_name)


def export_windowshap_tables(windowshap_dir, output_dir):
    windowshap_dir = Path(windowshap_dir)
    output_dir = Path(output_dir)

    table_specs = [
        ("hydra_windowshap_summary.csv", "hydra_windowshap_summary_latex.tex"),
        ("hydra_windowshap_timing.csv", "hydra_windowshap_timing_latex.tex"),
        ("hydra_windowshap_paired_tests.csv", "hydra_windowshap_paired_tests_latex.tex"),
        ("hydra_windowshap_cluster_summary.csv", "hydra_windowshap_cluster_summary_latex.tex"),
        ("hydra_windowshap_cluster_kruskal_tests.csv", "hydra_windowshap_cluster_kruskal_tests_latex.tex"),
        ("hydra_windowshap_within_cluster_tests.csv", "hydra_windowshap_within_cluster_tests_latex.tex"),
    ]

    for csv_name, tex_name in table_specs:
        df = _read_optional(windowshap_dir / csv_name)
        if df is not None:
            _write_latex(df, output_dir / tex_name)


def export_ablation_tables(ablation_dir, output_dir):
    ablation_dir = Path(ablation_dir)
    output_dir = Path(output_dir)

    df = _read_optional(ablation_dir / "hydra_saliency_ablation_summary.csv")
    if df is not None:
        _write_latex(df, output_dir / "hydra_saliency_ablation_summary_latex.tex")


def export_cluster_assignment_table(cluster_csv, output_dir):
    cluster_csv = Path(cluster_csv)
    output_dir = Path(output_dir)

    clusters = _read_optional(cluster_csv)
    if clusters is None:
        return

    clusters = clusters[["dataset", "cluster"]].drop_duplicates().sort_values(["cluster", "dataset"])
    clusters["cluster_name"] = clusters["cluster"].map(CLUSTER_NAMES)

    lines = []
    lines.append(r"\begin{longtable}{lcl}")
    lines.append(r"\caption{UCR dataset assignments to the four signal-morphology clusters used for stratified saliency analysis.}")
    lines.append(r"\label{tab:ucr_dataset_cluster_assignments}\\")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Dataset} & \textbf{Cluster} & \textbf{Cluster meaning} \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append("")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Dataset} & \textbf{Cluster} & \textbf{Cluster meaning} \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append("")

    for row in clusters.itertuples(index=False):
        lines.append(f"{row.dataset} & {int(row.cluster)} & {row.cluster_name} \\")

    lines.append("")
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")

    output_path = output_dir / "ucr_dataset_cluster_assignments_latex.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"[OK] {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="outputs/saliency/analysis")
    parser.add_argument("--windowshap-dir", default="outputs/saliency/windowshap")
    parser.add_argument("--ablation-dir", default="outputs/saliency/ablation")
    parser.add_argument("--cluster-csv", default="outputs/clustering/csv/ucr_dataset_clusters_k4.csv")
    parser.add_argument("--output-dir", default="outputs/report_tables")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_core_saliency_tables(args.analysis_dir, output_dir)
    export_windowshap_tables(args.windowshap_dir, output_dir)
    export_ablation_tables(args.ablation_dir, output_dir)
    export_cluster_assignment_table(args.cluster_csv, output_dir)


if __name__ == "__main__":
    main()
