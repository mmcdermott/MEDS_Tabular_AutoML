import logging
import shutil
from pathlib import Path

import hydra
import polars as pl
from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class EvaluationCallback(Callback):
    def on_multirun_end(self, config: DictConfig, **kwargs):
        """Find best model based on log files and logger.info its performance and hyperparameters.

        Args:
            config (DictConfig): Configuration dictionary containing paths and settings
            **kwargs: Additional keyword arguments

        Returns:
            polars.DataFrame: Performance of the top model

        Raises:
            FileNotFoundError: If log files are incomplete or not found

        Examples:
            >>> cb = EvaluationCallback()
            >>> with tempfile.TemporaryDirectory() as temp_dir:
            ...     temp_dir = Path(temp_dir)
            ...     sweep_results_dir = temp_dir / "sweep_results"
            ...     time_output_model_dir = temp_dir / "output"
            ...     time_output_model_dir.mkdir()
            ...
            ...     for trial, tuning_auc, test_auc in [('trial1', 0.9, 0.65), ('trial2', 0.8, 0.75)]:
            ...         trial_path = sweep_results_dir / trial
            ...         trial_path.mkdir(parents=True)
            ...
            ...         performance_log = pl.DataFrame(
            ...             {"trial_name": [trial], "tuning_auc": [tuning_auc], "test_auc": [test_auc]}
            ...         )
            ...         performance_log.write_csv(trial_path / 'performance.log')
            ...
            ...     config = DictConfig({
            ...         'path': {
            ...             'sweep_results_dir': str(sweep_results_dir),
            ...             'performance_log_stem': 'performance',
            ...             'best_trial_dir': str(temp_dir / 'best_trial'),
            ...         },
            ...         'time_output_model_dir': str(time_output_model_dir),
            ...         'prediction_splits': ['test'],
            ...         'delete_below_top_k': 1
            ...     })
            ...     best_model_performance = cb.on_multirun_end(config)
            >>> best_model_performance
            shape: (1, 3)
            ┌────────────┬────────────┬──────────┐
            │ trial_name ┆ tuning_auc ┆ test_auc │
            │ ---        ┆ ---        ┆ ---      │
            │ str        ┆ f64        ┆ f64      │
            ╞════════════╪════════════╪══════════╡
            │ trial1     ┆ 0.9        ┆ 0.65     │
            └────────────┴────────────┴──────────┘
        """
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
        else:  # pragma: no cover
            logger.info(
                "All models were saved. To automatically delete models, set delete_below_top_k in config."
            )
        best_trial_dir = Path(config.path.sweep_results_dir) / performance["trial_name"].cast(pl.String)[0]
        output_best_trial_dir = Path(config.path.best_trial_dir)
        shutil.copytree(best_trial_dir, output_best_trial_dir)
        performance.write_parquet(Path(config.time_output_model_dir) / "sweep_results_summary.parquet")

        self.store_predictions(output_best_trial_dir, config.prediction_splits)

        return performance.head(1)

    def store_predictions(self, best_trial_dir, splits):
        """Store predictions for specified data splits from the best model.

        Args:
            best_trial_dir (Path): Directory of the best trial
            splits (List[str]): Data splits to generate predictions for

        Examples:
            >>> cb = EvaluationCallback()
            >>> pred_df = pl.DataFrame({"predictions": [0.1, 0.2]})
            >>> with (
            ...     tempfile.TemporaryDirectory() as temp_dir,
            ...     patch("hydra.utils.instantiate", return_value=Mock(predict=Mock(return_value=pred_df)))
            ... ):
            ...     best_trial_dir = Path(temp_dir) / "best_trial"
            ...     best_trial_dir.mkdir()
            ...     OmegaConf.save({"model_launcher": {}}, best_trial_dir / "config.log")
            ...     (best_trial_dir / "xgboost.json").touch()
            ...     cb.store_predictions(best_trial_dir, ['test'])
            ...     # Verify predictions file was created
            ...     (best_trial_dir / 'test_predictions.parquet').exists()
            True
        """
        config = Path(best_trial_dir) / "config.log"
        xgboost_fp = Path(best_trial_dir) / "xgboost.json"
        if not xgboost_fp.exists():
            logger.warning("Prediction parquets not stored, we only support storing them for xgboost models.")
            return

        cfg = OmegaConf.load(config)
        model_launcher = hydra.utils.instantiate(cfg.model_launcher)
        model_launcher.load_model(xgboost_fp)
        model_launcher._build()

        for split in splits:
            pred_df = model_launcher.predict(split)
            pred_df.write_parquet(Path(best_trial_dir) / f"{split}_predictions.parquet")

    def log_performance(self, best_model_performance):
        """Log performance details of the best model.

        Args:
            best_model_performance (polars.DataFrame): Performance data of the best model

        Example:
            >>> cb = EvaluationCallback()
            >>> cb.log_performance(pl.DataFrame({'trial_name': ['T'], 'tuning_auc': [0], 'test_auc': [1]}))
        """
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
            >>> performance = pl.DataFrame(
            ...     {
            ...         "trial_name": ["trial1", "trial2", "trial3", "trial4"],
            ...         "tuning_auc": [0.9, 0.8, 0.7, 0.6],
            ...         "test_auc": [0.9, 0.8, 0.7, 0.6],
            ...     }
            ... )
            >>> cb = EvaluationCallback()
            >>> with tempfile.TemporaryDirectory() as sweep_dir:
            ...     for trial in performance["trial_name"]:
            ...         out_fp = Path(sweep_dir) / trial / "model.json"
            ...         out_fp.parent.mkdir(parents=True)
            ...         _ = out_fp.write_text(json.dumps({"model_name": trial, "content": "dummy data"}))
            ...     cb.delete_below_top_k_models(performance=performance, k=2, sweep_results_dir=sweep_dir)
            ...     sorted(p.name for p in Path(sweep_dir).iterdir())
            ['trial1', 'trial2']
        """
        logger.info(f"Deleting all models except top {k} models.")
        top_k_models = performance.head(k)["trial_name"].cast(pl.String).to_list()
        logger.debug(f"Top {k} models: {top_k_models}")
        for trial_dir in Path(sweep_results_dir).iterdir():
            if trial_dir.stem not in top_k_models:
                shutil.rmtree(trial_dir)
