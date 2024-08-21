import ast
from pathlib import Path

import pandas as pd
from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf


class EvaluationCallback(Callback):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def on_multirun_end(self, config: DictConfig, **kwargs):
        """Find best model based on log files and print its performance and hyperparameters."""
        log_fp = Path(config.model_log_dir)

        performance = pd.read_csv(
            log_fp / "performance.log", sep=",", header=None
        )  # , columns=["model_fp", "tuning_auc", "test_auc"])
        performance.columns = ["model_fp", "tuning_auc", "test_auc"]
        performance.sort_values("tuning_auc", ascending=False, inplace=True)
        print(performance.head())

        hyperparams = pd.read_csv(log_fp / "hyperparameters.log", sep="\t", header=None)
        hyperparams.columns = ["model_fp", "tabularization", "model_params"]

        best_model = performance.head(1)["model_fp"].values[0]
        best_hyperparams = hyperparams[hyperparams["model_fp"] == best_model]

        print(f"The best model can be found at {best_model}")
        self.print_performance(performance.head(1))
        self.print_hyperparams(best_hyperparams)
        if hasattr(config, "save_top_k") and config.save_top_k >= 0:
            self.save_top_k_models(performance, config.save_top_k, config.model_dir)

        return performance.head(1)

    def print_performance(self, performance):
        """Print performance of the best model with nice formatting."""
        print("Performance of the best model:")
        print(f"Tuning AUC: {performance['tuning_auc'].values[0]}")
        print(f"Test AUC: {performance['test_auc'].values[0]}")

    def print_hyperparams(self, hyperparams):
        """Print hyperparameters of the best model with nice formatting."""
        print("Hyperparameters of the best model:")
        print(
            f"Tabularization: {OmegaConf.to_yaml(ast.literal_eval(hyperparams['tabularization'].values[0]))}"
        )
        print(
            f"Model parameters: {OmegaConf.to_yaml(ast.literal_eval(hyperparams['model_params'].values[0]))}"
        )

    def save_top_k_models(self, performance, k, model_dir):
        """Save only top k models from the model directory and delete all other files."""
        top_k_models = performance.head(k)["model_fp"].values
        for model_fp in Path(model_dir).iterdir():
            if model_fp.is_file() and model_fp.suffix != ".log" and str(model_fp) not in top_k_models:
                model_fp.unlink()
