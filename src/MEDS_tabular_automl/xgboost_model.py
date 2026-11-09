import logging
from collections.abc import Callable
from pathlib import Path

import meds
import numpy as np
import scipy.sparse as sp
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf

from .gbdt_model import GradientBoostedTreesModel
from .tabular_dataset import TabularDataset

logger = logging.getLogger(__name__)


class XGBIterator(xgb.DataIter, TabularDataset):
    """XGBIterator class for loading and processing data shards for use in XGBoost models.

    This class provides functionality for iterating through data shards, loading
    feature data and labels, and processing them based on the provided configuration.

    Args:
        cfg: A configuration dictionary containing parameters for
            data processing, feature selection, and other settings.
        split: The data split to use, which can be one of "train", "tuning",
            or "held_out". This determines which subset of the data is loaded and processed.

    Attributes:
        cfg: Configuration dictionary containing parameters for
            data processing, feature selection, and other settings.
        file_name_resolver: Object for resolving file names and paths based on the configuration.
        split: The data split being used for loading and processing data shards.
        _data_shards: List of data shard names.
        valid_event_ids: Dictionary mapping shard number to a list of valid event IDs.
        labels: Dictionary mapping shard number to a list of labels for the corresponding event IDs.
        codes_set: Set of codes to include in the data.
        code_masks: Dictionary of code masks for filtering features based on aggregation.
        num_features: Total number of features in the data.
    """

    def __init__(self, cfg: DictConfig, split: str):
        """Initializes the XGBIterator with the provided configuration and data split.

        Args:
            cfg: The configuration dictionary.
            split: The data split to use.
        """
        xgb.DataIter.__init__(self, cache_prefix=cfg.path.cache_dir)
        TabularDataset.__init__(self, cfg=cfg, split=split)

        self._it = 0

    def next(self, input_data: Callable) -> int:
        """Advances the XGBIterator by one step and provides data to XGBoost for DMatrix construction.

        Args:
            input_data: A function passed by XGBoost with the same signature as `DMatrix`.

        Returns:
            0 if end of iteration, 1 otherwise.
        """
        if self._it == len(self._data_shards):
            return 0

        X, y = self._get_shard_by_index(self._it)  # self._data_shards[self._it])
        logger.debug(f"X shape: {X.shape}, y shape: {y.shape}")
        input_data(data=sp.csr_matrix(X), label=y)
        self._it += 1

        return 1

    def reset(self):
        """Resets the XGBIterator to its beginning."""
        self._it = 0


class XGBoostModel(GradientBoostedTreesModel):
    """Gradient-boosted trees with XGBoost.

    Splits are turned into `DMatrix` objects, either from matrices loaded in memory or streamed from disk
    through `XGBIterator` (`data_loading_params.keep_data_in_memory`). Iterators, training, evaluation and
    the predictions frame come from `GradientBoostedTreesModel`.

    Args:
        cfg: The configuration settings for the model, including data paths, model parameters,
            and flags for data handling.
    """

    iterator_class = XGBIterator

    def _build(self):
        """Builds the iterators and the `DMatrix` of every split."""
        self._build_iterators()
        if self.keep_data_in_memory:
            self._build_dmatrix_in_memory()
        else:
            self._build_dmatrix_from_iterators()

    def _build_dmatrix_in_memory(self):
        """Builds the DMatrix from the data in memory."""
        X_train, y_train = self.itrain.get_data()
        X_tuning, y_tuning = self.ituning.get_data()
        X_held_out, y_held_out = self.iheld_out.get_data()
        self.dtrain = xgb.DMatrix(X_train, label=y_train)
        self.dtuning = xgb.DMatrix(X_tuning, label=y_tuning)
        self.dheld_out = xgb.DMatrix(X_held_out, label=y_held_out)

    def _build_dmatrix_from_iterators(self):
        """Builds the DMatrix from the iterators."""
        self.dtrain = xgb.DMatrix(self.itrain)
        self.dtuning = xgb.DMatrix(self.ituning)
        self.dheld_out = xgb.DMatrix(self.iheld_out)

    def _train(self):
        """Trains the model."""
        self.model = xgb.train(
            OmegaConf.to_container(self.cfg.model),
            self.dtrain,
            num_boost_round=self.cfg.training_params.num_boost_round,
            early_stopping_rounds=self.cfg.training_params.early_stopping_rounds,
            evals=[(self.dtrain, "train"), (self.dtuning, "tuning")],
            verbose_eval=0,
        )

    def _predict(self, split="held_out") -> tuple[np.ndarray, np.ndarray]:
        """Helper Function that retrieves model predictions and labels."""
        if split == meds.tuning_split:
            y_pred = self.model.predict(self.dtuning)
            y_true = self.dtuning.get_label()
        elif split == meds.held_out_split:
            y_pred = self.model.predict(self.dheld_out)
            y_true = self.dheld_out.get_label()
        elif split == meds.train_split:
            y_pred = self.model.predict(self.dtrain)
            y_true = self.dtrain.get_label()
        else:
            raise ValueError(f"Invalid split for evaluation: {split}")
        return y_true, y_pred

    def save_model(self, output_fp: Path):
        """Saves the trained model to the specified file path.

        Args:
            output_fp: The file path to save the model to.
        """
        self.model.save_model(output_fp)

    def load_model(self, xgboost_json_fp: Path):
        self.model = xgb.Booster()
        self.model.load_model(str(xgboost_json_fp))
