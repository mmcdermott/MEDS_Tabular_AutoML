from pathlib import Path

import polars as pl
from hydra.experimental.callback import Callback
from loguru import logger
from omegaconf import DictConfig


class EvaluationCallback(Callback):
    def on_multirun_end(self, config: DictConfig, **kwargs):
        """Find best model based on log files and logger.info its performance and hyperparameters."""
        log_fp = Path(config.path.model_log_dir)

        try:
            performance = pl.read_csv(log_fp / f"*/*{config.path.performance_log_stem}.log")
        except Exception as e:
            raise FileNotFoundError(f"Log files incomplete or not found at {log_fp}") from e

        performance = performance.sort("tuning_auc", descending=True, nulls_last=True)
        logger.info(f"\nPerformance of the top 10 models:\n{performance.head(10)}")

        self.log_performance(performance[0, :])
        if hasattr(config, "delete_below_top_k") and config.delete_below_top_k >= 0:
            self.delete_below_top_k_models(
                performance, config.delete_below_top_k, config.path.output_model_dir
            )
        else:
            logger.info(
                "All models were saved. To automatically delete models, set delete_below_top_k in config."
            )

        return performance.head(1)

    def log_performance(self, best_model_performance):
        """logger.info performance of the best model with nice formatting."""
        best_model = best_model_performance["model_fp"][0]
        tuning_auc = best_model_performance["tuning_auc"][0]
        test_auc = best_model_performance["test_auc"][0]
        log_performance_message = [
            f"\nBest model can be found at {best_model}",
            "Performance of best model:",
            f"Tuning AUC: {tuning_auc}",
            f"Test AUC: {test_auc}",
        ]
        logger.info("\n".join(log_performance_message))

    def delete_below_top_k_models(self, performance, k, model_dir):
        """Save only top k models from the model directory and delete all other files.

        Args:
            performance: DataFrame containing model_fp and performance metrics.
            k: Number of top models to save.
            model_dir: Directory containing models.

        Example:
            >>> import tempfile
            >>> import json
            >>> performance = pl.DataFrame(
            ...     {
            ...         "model_fp": ["model1", "model2", "model3", "model4"],
            ...         "tuning_auc": [0.9, 0.8, 0.7, 0.6],
            ...         "test_auc": [0.9, 0.8, 0.7, 0.6],
            ...     }
            ... )
            >>> k = 2
            >>> with tempfile.TemporaryDirectory() as model_dir:
            ...     for model in performance["model_fp"]:
            ...         with open(Path(model_dir) / f"{model}.json", 'w') as f:
            ...             json.dump({"model_name": model, "content": "dummy data"}, f)
            ...     cb = EvaluationCallback()
            ...     cb.delete_below_top_k_models(performance, k, model_dir)
            ...     remaining_models = sorted(p.stem for p in Path(model_dir).iterdir())
            >>> remaining_models
            ['model1', 'model2']
        """
        logger.info(f"Deleting all models except top {k} models.")
        top_k_models = performance.head(k)["model_fp"].to_list()
        for model_fp in Path(model_dir).iterdir():
            if model_fp.is_file() and model_fp.suffix != ".log" and str(model_fp.stem) not in top_k_models:
                model_fp.unlink()
