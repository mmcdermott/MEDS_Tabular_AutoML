import ast
from pathlib import Path

import polars as pl
from hydra.experimental.callback import Callback
from loguru import logger
from omegaconf import DictConfig, OmegaConf


class EvaluationCallback(Callback):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def on_multirun_end(self, config: DictConfig, **kwargs):
        """Find best model based on log files and logger.info its performance and hyperparameters."""
        log_fp = Path(config.model_logging.model_log_dir)

        try:
            performance = pl.read_csv(log_fp / "*/*.csv")
        except Exception as e:
            raise FileNotFoundError(f"Log files incomplete or not found at {log_fp}, exception {e}.")

        performance = performance.sort("tuning_auc", descending=True, nulls_last=True)
        logger.info(performance.head(10))

        # get best model_fp
        best_model = performance[0, 0]

        best_params_fp = log_fp / best_model / f"{config.model_logging.config_log_stem}.json"

        # check if this file exists
        if not best_params_fp.is_file():
            raise FileNotFoundError(f"Best hyperparameters file not found at {best_params_fp}")

        logger.info(f"The best model can be found at {best_model}")
        # self.log_performance(performance.head(1))
        # self.log_hyperparams(best_hyperparams)
        if hasattr(config, "model_saving.delete_below_top_k") and config.delete_below_top_k >= 0:
            self.delete_below_top_k_models(
                performance, config.model_saving.delete_below_top_k, config.model_saving.model_dir
            )

        return performance.head(1)

    def log_performance(self, performance):
        """logger.info performance of the best model with nice formatting."""
        logger.info("Performance of the best model:")
        logger.info(f"Tuning AUC: {performance['tuning_auc'].values[0]}")
        logger.info(f"Test AUC: {performance['test_auc'].values[0]}")

    def log_hyperparams(self, hyperparams):
        """logger.info hyperparameters of the best model with nice formatting."""
        logger.info("Hyperparameters of the best model:")
        logger.info(
            f"Tabularization: {OmegaConf.to_yaml(ast.literal_eval(hyperparams['tabularization'].values[0]))}"
        )
        logger.info(
            f"Model parameters: {OmegaConf.to_yaml(ast.literal_eval(hyperparams['model_params'].values[0]))}"
        )

    def delete_below_top_k_models(self, performance, k, model_dir):
        """Save only top k models from the model directory and delete all other files."""
        top_k_models = performance.head(k)["model_fp"].values
        for model_fp in Path(model_dir).iterdir():
            if model_fp.is_file() and model_fp.suffix != ".log" and str(model_fp) not in top_k_models:
                model_fp.unlink()
