class WindowSHAPComparison:
    def __init__(
        self,
        datasets,
        output_dir,
        n_segments=100,
        shap_nsamples=500,
        max_samples_per_dataset=20,
        fractions=(0.05, 0.10, 0.20),
        seed=42,
        device=None,
    ):
        ...

    def compare_sample(self, model, x, y_true):
        ...

    def run_dataset(self, dataset):
        ...

    def run(self):
        ...

    def timing_summary(self):
        ...

    def overlap_summary(self):
        ...

    def paired_tests(self):
        ...

    def cluster_level_analysis(self, cluster_csv_path):
        ...