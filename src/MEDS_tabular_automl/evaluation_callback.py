from pathlib import Path

import polars as pl
from hydra.experimental.callback import Callback
from loguru import logger
from omegaconf import DictConfig


class EvaluationCallback(Callback):
    def on_multirun_end(self, config: DictConfig, **kwargs):
        """Find best model based on log files and logger.info its performance and hyperparameters."""
        log_fp = Path(config.model_logging.model_log_dir)

        try:
            perf = pl.read_csv(log_fp / f"*/*{config.model_logging.performance_log_stem}.log")
        except Exception as e:
            raise FileNotFoundError(f"Log files incomplete or not found at {log_fp}, exception {e}.")

        perf = perf.sort("tuning_auc", descending=True, nulls_last=True)
        logger.info(f"\nPerformance of the top 10 models:\n{perf.head(10)}")

        # get best model_fp
        best_model = perf[0, 0]

        logger.info(f"The best model can be found at {best_model}")
        self.log_performance(perf[0, :])
        if hasattr(config, "model_saving.delete_below_top_k") and config.delete_below_top_k >= 0:
            self.delete_below_top_k_models(
                perf, config.model_saving.delete_below_top_k, config.model_saving.model_dir
            )

        return perf.head(1)

    def log_performance(self, perf):
        """logger.info performance of the best model with nice formatting."""
        tuning_auc = perf["tuning_auc"][0]
        test_auc = perf["test_auc"][0]
        logger.info(
            f"\nPerformance of best model:\nTuning AUC: {tuning_auc}\nTest AUC: {test_auc}",
        )

    def delete_below_top_k_models(self, perf, k, model_dir):
        """Save only top k models from the model directory and delete all other files."""
        top_k_models = perf.head(k)["model_fp"].values
        for model_fp in Path(model_dir).iterdir():
            if model_fp.is_file() and model_fp.suffix != ".log" and str(model_fp) not in top_k_models:
                model_fp.unlink()
