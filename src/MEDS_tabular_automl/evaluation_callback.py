import shutil
from pathlib import Path

import polars as pl
from hydra.experimental.callback import Callback
from loguru import logger
from omegaconf import DictConfig


class EvaluationCallback(Callback):
    def on_multirun_end(self, config: DictConfig, **kwargs):
        """Find best model based on log files and logger.info its performance and hyperparameters."""
        log_fp = Path(config.path.sweep_results_dir)

        try:
            performance = pl.read_csv(log_fp / f"*/*{config.path.performance_log_stem}.log")
        except Exception as e:
            raise FileNotFoundError(f"Log files incomplete or not found at {log_fp}") from e

        performance = performance.sort("tuning_auc", descending=True, nulls_last=True)
        logger.info(f"\nPerformance of the top 10 models:\n{performance.head(10)}")

        self.log_performance(performance[0, :])
        if hasattr(config, "delete_below_top_k") and config.delete_below_top_k >= 0:
            self.delete_below_top_k_models(
                performance, config.delete_below_top_k, config.path.sweep_results_dir
            )
        else:
            logger.info(
                "All models were saved. To automatically delete models, set delete_below_top_k in config."
            )
        best_trial_dir = Path(config.path.sweep_results_dir) / performance["trial_name"].cast(pl.String)[0]
        output_best_trial_dir = Path(config.path.best_trial_dir)
        shutil.copytree(best_trial_dir, output_best_trial_dir)
        performance.write_parquet(config.time_output_model_dir / "sweep_results_summary.parquet")

        return performance.head(1)

    def log_performance(self, best_model_performance):
        """logger.info performance of the best model with nice formatting."""
        best_model = best_model_performance["trial_name"][0]
        tuning_auc = best_model_performance["tuning_auc"][0]
        test_auc = best_model_performance["test_auc"][0]
        log_performance_message = [
            f"\nBest model can be found at {best_model}",
            "Performance of best model:",
            f"Tuning AUC: {tuning_auc}",
            f"Test AUC: {test_auc}",
        ]
        logger.info("\n".join(log_performance_message))

    def delete_below_top_k_models(self, performance, k, sweep_results_dir):
        """Save only top k models from the sweep results directory and delete all other directories.

        Args:
            performance: DataFrame containing trial_name and performance metrics.
            k: Number of top models to save.
            sweep_results_dir: Directory containing trial results.

        Example:
            >>> import tempfile
            >>> import json
            >>> import polars as pl
            >>> from pathlib import Path
            >>> performance = pl.DataFrame(
            ...     {
            ...         "trial_name": ["trial1", "trial2", "trial3", "trial4"],
            ...         "tuning_auc": [0.9, 0.8, 0.7, 0.6],
            ...         "test_auc": [0.9, 0.8, 0.7, 0.6],
            ...     }
            ... )
            >>> k = 2
            >>> with tempfile.TemporaryDirectory() as sweep_dir:
            ...     for trial in performance["trial_name"]:
            ...         trial_dir = Path(sweep_dir) / trial
            ...         trial_dir.mkdir()
            ...         with open(trial_dir / "model.json", 'w') as f:
            ...             json.dump({"model_name": trial, "content": "dummy data"}, f)
            ...     cb = EvaluationCallback()
            ...     cb.delete_below_top_k_models(performance, k, sweep_dir)
            ...     remaining_trials = sorted(p.name for p in Path(sweep_dir).iterdir())
            >>> remaining_trials
            ['trial1', 'trial2']
        """
        logger.info(f"Deleting all models except top {k} models.")
        top_k_models = performance.head(k)["trial_name"].cast(pl.String).to_list()
        logger.debug(f"Top {k} models: {top_k_models}")
        for trial_dir in Path(sweep_results_dir).iterdir():
            if trial_dir.stem not in top_k_models:
                shutil.rmtree(trial_dir)
