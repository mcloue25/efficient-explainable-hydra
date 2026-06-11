from dataclasses import dataclass
from pathlib import Path
import gc
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import kruskal, wilcoxon

from classes.models.hydra_explainable import HydraModelExplainable
from utils.data_utils import load_dataset
from utils.explainability import apply_mask, select_contiguous_window


CLUSTER_NAMES = {
    0: "High-frequency / high-curvature",
    1: "Smooth / low-complexity",
    2: "Short / moderately rough",
    3: "Spiky / multi-class",
}


def pred_to_hydra_class_index(model, pred_label):
    if len(model.classifier.classes_) > 2:
        return int(np.where(model.classifier.classes_ == pred_label)[0][0])
    return 0


def get_predicted_class_score(model, X, pred_label):
    decision = np.asarray(model.decision_function(X))

    if decision.ndim == 0:
        return float(decision)

    if decision.ndim == 1:
        margin = float(decision[0])
        return margin if pred_label == 1 else -margin

    classes = getattr(model.classifier, "classes_", None)
    if classes is not None and pred_label in classes:
        class_index = int(np.where(classes == pred_label)[0][0])
    else:
        class_index = int(pred_label)

    return float(decision[0, class_index])


def mask_to_start(mask):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    return int(idx[0])


def window_to_mask(start, length, series_length):
    mask = np.zeros(series_length, dtype=bool)
    mask[start:start + length] = True
    return mask


def compare_masks(mask_a, mask_b):
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    iou = intersection / union if union > 0 else 0.0
    denom = min(mask_a.sum(), mask_b.sum())
    overlap_fraction = intersection / denom if denom > 0 else 0.0

    a_idx = np.where(mask_a)[0]
    b_idx = np.where(mask_b)[0]

    if len(a_idx) == 0 or len(b_idx) == 0:
        centre_distance = np.nan
    else:
        a_centre = 0.5 * (a_idx[0] + a_idx[-1])
        b_centre = 0.5 * (b_idx[0] + b_idx[-1])
        centre_distance = abs(a_centre - b_centre) / len(mask_a)

    return {
        "iou": float(iou),
        "overlap_fraction": float(overlap_fraction),
        "normalised_centre_distance": float(centre_distance),
    }


@dataclass
class WindowSHAPExplainer:
    n_segments: int = 100
    nsamples: int = 500

    def __post_init__(self):
        try:
            import shap
        except ImportError as exc:
            raise ImportError("Install SHAP first with: pip install shap") from exc

        self.shap = shap

    def segment_bounds(self, series_length):
        edges = np.linspace(0, series_length, self.n_segments + 1).round().astype(int)
        return [
            (int(edges[i]), int(edges[i + 1]))
            for i in range(self.n_segments)
            if edges[i + 1] > edges[i]
        ]

    def apply_segment_coalition(self, x, coalition, bounds, fill_value=None):
        x_masked = np.asarray(x, dtype=np.float32).copy()

        if fill_value is None:
            fill_value = float(np.mean(x_masked))

        for keep, (start, end) in zip(coalition, bounds):
            if keep < 0.5:
                x_masked[start:end] = fill_value

        return x_masked

    def expand_segment_values(self, segment_values, bounds, series_length):
        out = np.zeros(series_length, dtype=np.float32)

        for value, (start, end) in zip(segment_values, bounds):
            out[start:end] = value

        return out

    def explain(self, model, x, pred_label):
        x = np.asarray(x, dtype=np.float32)
        series_length = len(x)
        bounds = self.segment_bounds(series_length)
        n_segments_actual = len(bounds)

        background = np.zeros((1, n_segments_actual), dtype=np.float32)
        instance = np.ones((1, n_segments_actual), dtype=np.float32)
        fill_value = float(np.mean(x))

        def predict_from_coalitions(coalitions):
            coalitions = np.asarray(coalitions, dtype=np.float32)

            X_masked = np.stack([
                self.apply_segment_coalition(
                    x=x,
                    coalition=coalition,
                    bounds=bounds,
                    fill_value=fill_value,
                )
                for coalition in coalitions
            ])

            scores = []
            for i in range(len(X_masked)):
                scores.append(
                    get_predicted_class_score(
                        model=model,
                        X=X_masked[i:i + 1],
                        pred_label=pred_label,
                    )
                )

            return np.asarray(scores)

        explainer = self.shap.KernelExplainer(predict_from_coalitions, background)
        shap_values = explainer.shap_values(instance, nsamples=self.nsamples)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.asarray(shap_values).reshape(-1)
        shap_time = self.expand_segment_values(shap_values, bounds, series_length)

        return np.abs(shap_time)


@dataclass
class WindowSHAPComparison:
    datasets: list[str]
    output_dir: Path | str = Path("outputs/saliency/windowshap")
    fractions: tuple[float, ...] = (0.05, 0.10, 0.20)
    n_segments: int = 100
    shap_nsamples: int = 500
    max_samples_per_dataset: int | None = 20
    seed: int = 42
    device: str | torch.device | None = None

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif not isinstance(self.device, torch.device):
            self.device = torch.device(self.device)

        self.explainer = WindowSHAPExplainer(
            n_segments=self.n_segments,
            nsamples=self.shap_nsamples,
        )

    def hydra_importance(self, model, x, pred_label):
        class_index = pred_to_hydra_class_index(model, pred_label)
        x_t = model._to_tensor(x[None, :])

        saliency = np.asarray(
            model.transform.get_saliency_map_fast(
                x_t,
                model.classifier,
                model.scaler,
                class_index=class_index,
            ),
            dtype=np.float32,
        )

        return np.abs(saliency)

    def compare_sample(self, model, x, y_true):
        x = np.asarray(x, dtype=np.float32)

        pred_before = int(model.predict(x[None, :])[0])

        t0 = time.perf_counter()
        hydra_importance = self.hydra_importance(model, x, pred_before)
        hydra_explain_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        shap_importance = self.explainer.explain(
            model=model,
            x=x,
            pred_label=pred_before,
        )
        windowshap_explain_time = time.perf_counter() - t0

        score_before = get_predicted_class_score(model, x[None, :], pred_before)

        rows = []

        for fraction in self.fractions:
            hydra_mask = select_contiguous_window(
                hydra_importance,
                fraction=fraction,
                mode="top",
            )

            shap_mask = select_contiguous_window(
                shap_importance,
                fraction=fraction,
                mode="top",
            )

            overlap = compare_masks(hydra_mask, shap_mask)

            x_hydra_masked = apply_mask(x, hydra_mask)
            x_shap_masked = apply_mask(x, shap_mask)

            score_after_hydra = get_predicted_class_score(
                model,
                x_hydra_masked[None, :],
                pred_before,
            )

            score_after_shap = get_predicted_class_score(
                model,
                x_shap_masked[None, :],
                pred_before,
            )

            pred_after_hydra = int(model.predict(x_hydra_masked[None, :])[0])
            pred_after_shap = int(model.predict(x_shap_masked[None, :])[0])

            rows.append({
                "true_label": int(y_true),
                "pred_before": pred_before,
                "fraction": fraction,
                "hydra_start": mask_to_start(hydra_mask),
                "shap_start": mask_to_start(shap_mask),
                "window_length": int(hydra_mask.sum()),
                "iou": overlap["iou"],
                "overlap_fraction": overlap["overlap_fraction"],
                "normalised_centre_distance": overlap["normalised_centre_distance"],
                "score_before": score_before,
                "hydra_score_drop": score_before - score_after_hydra,
                "shap_score_drop": score_before - score_after_shap,
                "hydra_flipped": int(pred_after_hydra != pred_before),
                "shap_flipped": int(pred_after_shap != pred_before),
                "hydra_explain_time_s": hydra_explain_time,
                "windowshap_explain_time_s": windowshap_explain_time,
                "windowshap_over_hydra_time_ratio": (
                    windowshap_explain_time / max(hydra_explain_time, 1e-12)
                ),
                "hydra_over_windowshap_time_ratio": (
                    hydra_explain_time / max(windowshap_explain_time, 1e-12)
                ),
            })

        return rows

    def run_dataset(self, dataset):
        output_path = self.output_dir / f"hydra_windowshap_{dataset}.csv"

        if output_path.exists():
            print(f"[SKIP] WindowSHAP comparison already exists for {dataset}")
            return pd.read_csv(output_path)

        print(f"\n=== HYDRA vs WindowSHAP: {dataset} ===")

        X_train, y_train, X_test, y_test, _ = load_dataset(dataset)

        model = HydraModelExplainable(
            input_dim=X_train.shape[-1],
            seed=self.seed,
            device=self.device,
        )
        model.fit(X_train, y_train)

        if not hasattr(model.transform, "get_saliency_map_fast"):
            raise AttributeError("model.transform does not have get_saliency_map_fast")

        preds = model.predict(X_test)
        correct_indices = np.where(preds == y_test)[0]

        if self.max_samples_per_dataset is not None:
            correct_indices = correct_indices[:self.max_samples_per_dataset]

        rows = []

        for count, idx in enumerate(correct_indices, start=1):
            print(f"Sample {count}/{len(correct_indices)} | idx={idx}")

            sample_rows = self.compare_sample(
                model=model,
                x=X_test[idx],
                y_true=y_test[idx],
            )

            for row in sample_rows:
                row["dataset"] = dataset
                row["sample_idx"] = int(idx)

            rows.extend(sample_rows)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

        del model, X_train, y_train, X_test, y_test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return df

    def run(self):
        frames = []

        for i, dataset in enumerate(self.datasets, start=1):
            print(f"\n[{i}/{len(self.datasets)}] WindowSHAP on {dataset}")
            frames.append(self.run_dataset(dataset))

        samples = pd.concat(frames, ignore_index=True)
        samples_path = self.output_dir / "hydra_windowshap_samples.csv"
        samples.to_csv(samples_path, index=False)

        summary = self.overlap_summary(samples)
        timing = self.timing_summary(samples)
        dataset_summary = self.dataset_summary(samples)
        paired = self.paired_tests(samples)

        summary.to_csv(self.output_dir / "hydra_windowshap_summary.csv", index=False)
        timing.to_csv(self.output_dir / "hydra_windowshap_timing.csv", index=False)
        dataset_summary.to_csv(self.output_dir / "hydra_windowshap_by_dataset.csv", index=False)
        paired.to_csv(self.output_dir / "hydra_windowshap_paired_tests.csv", index=False)

        print(f"Saved samples: {samples_path}")

        return {
            "samples": samples,
            "summary": summary,
            "timing": timing,
            "dataset_summary": dataset_summary,
            "paired_tests": paired,
        }

    def overlap_summary(self, df):
        return (
            df.groupby("fraction", as_index=False)
            .agg(
                mean_iou=("iou", "mean"),
                median_iou=("iou", "median"),
                mean_overlap_fraction=("overlap_fraction", "mean"),
                median_overlap_fraction=("overlap_fraction", "median"),
                mean_centre_distance=("normalised_centre_distance", "mean"),
                median_centre_distance=("normalised_centre_distance", "median"),
                mean_hydra_score_drop=("hydra_score_drop", "mean"),
                mean_shap_score_drop=("shap_score_drop", "mean"),
                hydra_flip_rate=("hydra_flipped", "mean"),
                shap_flip_rate=("shap_flipped", "mean"),
                n_samples=("sample_idx", "count"),
                n_datasets=("dataset", "nunique"),
            )
        )

    def timing_summary(self, df):
        timing_per_sample = df.drop_duplicates(["dataset", "sample_idx"]).copy()

        return pd.DataFrame({
            "metric": [
                "HYDRA explain time (s)",
                "WindowSHAP explain time (s)",
                "WindowSHAP / HYDRA ratio",
                "HYDRA / WindowSHAP ratio",
            ],
            "mean": [
                timing_per_sample["hydra_explain_time_s"].mean(),
                timing_per_sample["windowshap_explain_time_s"].mean(),
                timing_per_sample["windowshap_over_hydra_time_ratio"].mean(),
                timing_per_sample["hydra_over_windowshap_time_ratio"].mean(),
            ],
            "median": [
                timing_per_sample["hydra_explain_time_s"].median(),
                timing_per_sample["windowshap_explain_time_s"].median(),
                timing_per_sample["windowshap_over_hydra_time_ratio"].median(),
                timing_per_sample["hydra_over_windowshap_time_ratio"].median(),
            ],
            "std": [
                timing_per_sample["hydra_explain_time_s"].std(),
                timing_per_sample["windowshap_explain_time_s"].std(),
                timing_per_sample["windowshap_over_hydra_time_ratio"].std(),
                timing_per_sample["hydra_over_windowshap_time_ratio"].std(),
            ],
        })

    def dataset_summary(self, df):
        dataset_level = (
            df.groupby(["dataset", "fraction"], as_index=False)
            .agg(
                mean_iou=("iou", "mean"),
                median_iou=("iou", "median"),
                mean_overlap_fraction=("overlap_fraction", "mean"),
                mean_centre_distance=("normalised_centre_distance", "mean"),
                hydra_score_drop=("hydra_score_drop", "mean"),
                shap_score_drop=("shap_score_drop", "mean"),
                hydra_flip_rate=("hydra_flipped", "mean"),
                shap_flip_rate=("shap_flipped", "mean"),
                n_samples=("sample_idx", "nunique"),
            )
        )

        dataset_level["shap_minus_hydra_score_drop"] = (
            dataset_level["shap_score_drop"] - dataset_level["hydra_score_drop"]
        )
        dataset_level["shap_minus_hydra_flip_rate"] = (
            dataset_level["shap_flip_rate"] - dataset_level["hydra_flip_rate"]
        )

        return dataset_level

    def paired_tests(self, df):
        rows = []

        for fraction, group in df.groupby("fraction"):
            score_diff = group["shap_score_drop"] - group["hydra_score_drop"]
            flip_diff = group["shap_flipped"] - group["hydra_flipped"]

            for name, diff in [
                ("shap_score_drop_minus_hydra", score_diff),
                ("shap_flip_minus_hydra", flip_diff),
            ]:
                nonzero = diff[diff != 0]
                if len(nonzero) > 0:
                    p_value = wilcoxon(nonzero, alternative="greater").pvalue
                else:
                    p_value = np.nan

                rows.append({
                    "fraction": fraction,
                    "metric": name,
                    "mean_diff": float(diff.mean()),
                    "median_diff": float(diff.median()),
                    "positive_count": int((diff > 0).sum()),
                    "n_non_tied": int(len(nonzero)),
                    "wilcoxon_p": float(p_value) if not np.isnan(p_value) else np.nan,
                })

        return pd.DataFrame(rows)

    def cluster_level_analysis(self, cluster_csv_path):
        samples_path = self.output_dir / "hydra_windowshap_samples.csv"
        if not samples_path.exists():
            raise FileNotFoundError(f"Missing WindowSHAP samples file: {samples_path}")

        df = pd.read_csv(samples_path)
        clusters = pd.read_csv(cluster_csv_path)

        cluster_cols = ["dataset", "cluster"]
        if "cluster_name" in clusters.columns:
            cluster_cols.append("cluster_name")

        out = df.merge(clusters[cluster_cols].drop_duplicates(), on="dataset", how="left")

        if "cluster_name" not in out.columns:
            out["cluster_name"] = out["cluster"].map(CLUSTER_NAMES)

        out["shap_minus_hydra_score_drop"] = out["shap_score_drop"] - out["hydra_score_drop"]
        out["shap_minus_hydra_flip"] = out["shap_flipped"] - out["hydra_flipped"]

        cluster_summary = (
            out.groupby(["cluster", "cluster_name", "fraction"], as_index=False)
            .agg(
                mean_iou=("iou", "mean"),
                median_iou=("iou", "median"),
                mean_overlap_fraction=("overlap_fraction", "mean"),
                median_overlap_fraction=("overlap_fraction", "median"),
                mean_centre_distance=("normalised_centre_distance", "mean"),
                median_centre_distance=("normalised_centre_distance", "median"),
                mean_hydra_score_drop=("hydra_score_drop", "mean"),
                mean_shap_score_drop=("shap_score_drop", "mean"),
                mean_shap_minus_hydra_score_drop=("shap_minus_hydra_score_drop", "mean"),
                hydra_flip_rate=("hydra_flipped", "mean"),
                shap_flip_rate=("shap_flipped", "mean"),
                mean_shap_minus_hydra_flip=("shap_minus_hydra_flip", "mean"),
                n_samples=("sample_idx", "count"),
                n_datasets=("dataset", "nunique"),
            )
        )

        cluster_tests = self._kruskal_cluster_tests(out)
        within_cluster_tests = self._within_cluster_tests(out)

        out.to_csv(self.output_dir / "hydra_windowshap_samples_with_clusters.csv", index=False)
        cluster_summary.to_csv(self.output_dir / "hydra_windowshap_cluster_summary.csv", index=False)
        cluster_tests.to_csv(self.output_dir / "hydra_windowshap_cluster_kruskal_tests.csv", index=False)
        within_cluster_tests.to_csv(self.output_dir / "hydra_windowshap_within_cluster_tests.csv", index=False)

        return {
            "samples_with_clusters": out,
            "cluster_summary": cluster_summary,
            "cluster_tests": cluster_tests,
            "within_cluster_tests": within_cluster_tests,
        }

    def _kruskal_cluster_tests(self, df):
        rows = []

        for value_col in [
            "shap_minus_hydra_score_drop",
            "shap_minus_hydra_flip",
            "iou",
            "overlap_fraction",
        ]:
            for fraction, frac_df in df.groupby("fraction"):
                groups = [
                    g[value_col].dropna().values
                    for _, g in frac_df.groupby("cluster")
                ]
                groups = [g for g in groups if len(g) > 0]

                if len(groups) >= 2:
                    H, p_value = kruskal(*groups)
                else:
                    H, p_value = np.nan, np.nan

                rows.append({
                    "fraction": fraction,
                    "metric": value_col,
                    "kruskal_H": H,
                    "p_value": p_value,
                    "n_clusters": len(groups),
                    "n_samples": len(frac_df),
                })

        return pd.DataFrame(rows)

    def _within_cluster_tests(self, df):
        rows = []

        for (cluster, cluster_name, fraction), group in df.groupby(["cluster", "cluster_name", "fraction"]):
            score_diff = group["shap_score_drop"] - group["hydra_score_drop"]
            flip_diff = group["shap_flipped"] - group["hydra_flipped"]

            score_nonzero = score_diff[score_diff != 0]
            flip_nonzero = flip_diff[flip_diff != 0]

            score_p = wilcoxon(score_nonzero, alternative="greater").pvalue if len(score_nonzero) > 0 else np.nan
            flip_p = wilcoxon(flip_nonzero, alternative="greater").pvalue if len(flip_nonzero) > 0 else np.nan

            rows.append({
                "cluster": cluster,
                "cluster_name": cluster_name,
                "fraction": fraction,
                "n_samples": len(group),
                "n_datasets": group["dataset"].nunique(),
                "mean_hydra_score_drop": group["hydra_score_drop"].mean(),
                "mean_shap_score_drop": group["shap_score_drop"].mean(),
                "mean_shap_minus_hydra_score_drop": score_diff.mean(),
                "median_shap_minus_hydra_score_drop": score_diff.median(),
                "score_wilcoxon_p": score_p,
                "hydra_flip_rate": group["hydra_flipped"].mean(),
                "shap_flip_rate": group["shap_flipped"].mean(),
                "mean_shap_minus_hydra_flip": flip_diff.mean(),
                "median_shap_minus_hydra_flip": flip_diff.median(),
                "flip_wilcoxon_p": flip_p,
                "mean_iou": group["iou"].mean(),
                "mean_overlap_fraction": group["overlap_fraction"].mean(),
            })

        return pd.DataFrame(rows)
