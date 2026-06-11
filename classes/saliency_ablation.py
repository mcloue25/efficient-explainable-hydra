from dataclasses import dataclass
from pathlib import Path
import gc

import numpy as np
import pandas as pd
import torch

from classes.models.hydra_explainable import HydraModelExplainable
from classes.windowshap import get_predicted_class_score, pred_to_hydra_class_index
from utils.data_utils import load_dataset
from utils.explainability import apply_mask, select_contiguous_window


ABLATION_VARIANTS = (
    "max_only",
    "min_only",
    "min_activation_scaled",
    "combined",
)

VARIANT_LABELS = {
    "max_only": "Max-only",
    "min_only": "Min-only",
    "min_activation_scaled": "Activation-scaled min",
    "combined": "Combined",
}

VARIANT_ORDER = {
    "max_only": 0,
    "min_only": 1,
    "min_activation_scaled": 2,
    "combined": 3,
}

EPS = 1e-6


@dataclass
class HydraSaliencyAblation:
    datasets: list[str]
    output_dir: Path | str = Path("outputs/saliency/ablation")
    variants: tuple[str, ...] = ABLATION_VARIANTS
    fraction: float = 0.10
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

    def run_dataset(self, dataset):
        out_path = self.output_dir / f"hydra_saliency_ablation_{dataset}.csv"

        if out_path.exists():
            print(f"[SKIP] Loading existing ablation results for {dataset}")
            return pd.read_csv(out_path)

        print(f"\n=== HYDRA saliency ablation: {dataset} ===")

        X_train, y_train, X_test, y_test, _ = load_dataset(dataset)

        model = HydraModelExplainable(
            input_dim=X_train.shape[-1],
            seed=self.seed,
            device=self.device,
        )
        model.fit(X_train, y_train)

        if not hasattr(model.transform, "get_saliency_map_fast"):
            raise AttributeError("get_saliency_map_fast is not available on model.transform")

        preds = model.predict(X_test)
        correct_indices = np.where(preds == y_test)[0]

        if self.max_samples_per_dataset is not None:
            correct_indices = correct_indices[:self.max_samples_per_dataset]

        rows = []

        for count, idx in enumerate(correct_indices, start=1):
            print(f"Sample {count}/{len(correct_indices)} | idx={idx}")

            x = np.asarray(X_test[idx], dtype=np.float32)
            y_true = y_test[idx]

            pred_before = int(model.predict(x[None, :])[0])
            class_index = pred_to_hydra_class_index(model, pred_before)
            x_t = model._to_tensor(x[None, :])
            score_before = get_predicted_class_score(model, x[None, :], pred_before)

            for variant in self.variants:
                saliency = np.asarray(
                    model.transform.get_saliency_map_fast(
                        x_t,
                        model.classifier,
                        model.scaler,
                        class_index=class_index,
                        saliency_variant=variant,
                    ),
                    dtype=np.float32,
                )

                importance = np.abs(saliency)
                mask = select_contiguous_window(
                    importance,
                    fraction=self.fraction,
                    mode="top",
                )

                x_masked = apply_mask(x, mask)

                score_after = get_predicted_class_score(
                    model,
                    x_masked[None, :],
                    pred_before,
                )
                pred_after = int(model.predict(x_masked[None, :])[0])

                score_drop = score_before - score_after
                bounded_relative_score_drop = np.clip(
                    score_drop / (abs(score_before) + EPS),
                    -1.0,
                    1.0,
                )

                rows.append({
                    "dataset": dataset,
                    "sample_idx": int(idx),
                    "true_label": int(y_true),
                    "pred_before": pred_before,
                    "variant": variant,
                    "variant_label": VARIANT_LABELS[variant],
                    "fraction": self.fraction,
                    "score_before": score_before,
                    "score_after": score_after,
                    "score_drop": score_drop,
                    "bounded_relative_score_drop": bounded_relative_score_drop,
                    "flipped": int(pred_after != pred_before),
                    "window_length": int(mask.sum()),
                    "window_start": int(np.where(mask)[0][0]),
                })

        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)

        del model, X_train, y_train, X_test, y_test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return df

    def run(self):
        frames = []

        for i, dataset in enumerate(self.datasets, start=1):
            print(f"\n[{i}/{len(self.datasets)}] HYDRA saliency ablation on {dataset}")
            frames.append(self.run_dataset(dataset))

        samples = pd.concat(frames, ignore_index=True)
        samples_path = self.output_dir / "hydra_saliency_ablation_samples.csv"
        samples.to_csv(samples_path, index=False)

        summary = self.summarise(samples)
        summary_path = self.output_dir / "hydra_saliency_ablation_summary.csv"
        summary.to_csv(summary_path, index=False)

        print(f"Saved samples: {samples_path}")
        print(f"Saved summary: {summary_path}")

        return {
            "samples": samples,
            "summary": summary,
        }

    def summarise(self, samples=None):
        if samples is None:
            samples_path = self.output_dir / "hydra_saliency_ablation_samples.csv"
            samples = pd.read_csv(samples_path)

        summary = (
            samples.groupby(["variant", "variant_label"], as_index=False)
            .agg(
                flip_rate=("flipped", "mean"),
                bounded_score_drop=("bounded_relative_score_drop", "mean"),
                raw_score_drop=("score_drop", "mean"),
                n_samples=("sample_idx", "count"),
                n_datasets=("dataset", "nunique"),
            )
        )

        summary["flip_rate_pct"] = 100 * summary["flip_rate"]
        summary["order"] = summary["variant"].map(VARIANT_ORDER)
        summary = summary.sort_values("order").drop(columns="order")

        return summary
