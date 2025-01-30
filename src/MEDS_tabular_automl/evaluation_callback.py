import shutil
from pathlib import Path

import polars as pl
from hydra.experimental.callback import Callback
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf


class MockModelLauncher:  # pragma: no cover
    def load_model(self, model_path):
        pass

    def _build(self):
        pass

    def predict(self, split):
        return pl.DataFrame({"predictions": [0.1, 0.2]})


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

        Example:
            >>> import tempfile
            >>> import polars as pl
            >>> from pathlib import Path
            >>> from omegaconf import OmegaConf
            >>>
            >>> # Create a temporary directory for testing
            >>> temp_dir = tempfile.mkdtemp()
            >>> # Setup mock sweep results directory
            >>> sweep_results_dir = Path(temp_dir) / 'sweep_results'
            >>> sweep_results_dir.mkdir()
            >>>
            >>> # Create mock trial directories
            >>> trial_names = ['trial1', 'trial2']
            >>> for trial in trial_names:
            ...     trial_path = sweep_results_dir / trial
            ...     trial_path.mkdir()
            ...
            ...     # Create mock performance log
            ...     performance_log = pl.DataFrame({
            ...         'trial_name': [trial],
            ...         'tuning_auc': [0.9 if trial == 'trial1' else 0.8],
            ...         'test_auc': [0.85 if trial == 'trial1' else 0.75]
            ...     })
            ...     performance_log.write_csv(trial_path / 'performance.log')
            >>>
            >>> # Create configuration
            >>> config = OmegaConf.create({
            ...     'path': {
            ...         'sweep_results_dir': str(sweep_results_dir),
            ...         'performance_log_stem': 'performance',
            ...         'best_trial_dir': str(Path(temp_dir) / 'best_trial'),
            ...     },
            ...     'time_output_model_dir': str(Path(temp_dir) / 'output'),
            ...     'prediction_splits': ['test'],
            ...     'delete_below_top_k': 1
            ... })
            >>>
            >>> # Create a mock evaluation callback
            >>> cb = EvaluationCallback()
            >>>
            >>> # Run the method
            >>> Path(config.time_output_model_dir).mkdir()
            >>> result = cb.on_multirun_end(config)
            >>>
            >>> # Verify results
            >>> result['trial_name'][0] == 'trial1'
            True
            >>> # Clean up the temporary directory
            >>> import shutil
            >>> shutil.rmtree(temp_dir)
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
        else:
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

        Example:
        >>> import tempfile
        >>> import polars as pl
        >>> from pathlib import Path
        >>> from unittest.mock import Mock, patch
        >>> import shutil
        >>>
        >>> temp_dir = tempfile.mkdtemp()
        >>> best_trial_dir = Path(temp_dir)
        >>>
        >>> # Create mock config and xgboost files
        >>> _ = (best_trial_dir / 'config.log').write_text('''
        ... model_launcher:
        ...   _target_: MEDS_tabular_automl.evaluation_callback.MockModelLauncher
        ... ''')
        >>> (best_trial_dir / 'xgboost.json').touch()
        >>>
        >>> # Mock model launcher
        >>>
        >>> cb = EvaluationCallback()
        >>> cb.store_predictions(best_trial_dir, ['test'])
        >>> # Verify predictions file was created
        >>> (best_trial_dir / 'test_predictions.parquet').exists()
        True
        >>> shutil.rmtree(temp_dir)
        """
        config = Path(best_trial_dir) / "config.log"
        xgboost_fp = Path(best_trial_dir) / "xgboost.json"
        if not xgboost_fp.exists():
            logger.warning("Prediction parquets not stored, we only support storing them for xgboost models.")
            return

        cfg = OmegaConf.load(config)
        model_launcher = instantiate(cfg.model_launcher)
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
            >>> import polars as pl
            >>>
            >>> # Create a mock performance DataFrame
            >>> best_model_performance = pl.DataFrame({
            ...     'trial_name': ['trial1'],
            ...     'tuning_auc': [0.85],
            ...     'test_auc': [0.82]
            ... })
            >>>
            >>> # Create an instance of the evaluation callback
            >>> cb = EvaluationCallback()
            >>>
            >>> # Test the method (this will log to console)
            >>> cb.log_performance(best_model_performance)  # Doctest: +ELLIPSIS
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
