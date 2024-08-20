from pathlib import Path

import numpy as np
import scipy.sparse as sp
from loguru import logger
from mixins import TimeableMixin
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score

from .base_model import BaseModel
from .tabular_dataset import TabularDataset


class SklearnIterator(TabularDataset, TimeableMixin):
    """SklearnIterator class for loading and processing data shards for use in SciKit-Learn models.

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
        """Initializes the SklearnIterator with the provided configuration and data split.

        Args:
            cfg: The configuration dictionary.
            split: The data split to use.
        """
        TabularDataset.__init__(self, cfg=cfg, split=split)
        TimeableMixin.__init__(self)
        self.valid_event_ids, self.labels = self._load_ids_and_labels()
        # check if the labels are empty
        if len(self.labels) == 0:
            raise ValueError("No labels found.")
        # self._it = 0

    # def iterate(self, function):
    #     for shard_idx in range(len(self._data_shards)):
    #         data, labels = self.get_data_shards(shard_idx)
    #         function(data, labels)


class SklearnMatrix(TimeableMixin):
    """SklearnMatrix class for loading and processing data shards for use in SciKit-Learn models."""

    def __init__(self, data: sp.csr_matrix, labels: np.ndarray):
        """Initializes the SklearnMatrix with the provided configuration and data split.

        Args:
            data
        """
        super().__init__()
        self.data = data
        self.labels = labels

    def get_data(self):
        return self.data

    def get_label(self):
        return self.labels


class SklearnModel(BaseModel, TimeableMixin):
    """Class for configuring, training, and evaluating an SciKit-Learn model.

    This class utilizes the configuration settings provided to manage the training and evaluation
    process of an XGBoost model, ensuring the model is trained and validated using specified parameters
    and data splits. It supports training with in-memory data handling as well as direct streaming from
    disk using iterators.

    Args:
        cfg: The configuration settings for the model, including data paths, model parameters,ß
            and flags for data handling.

    Attributes:
        cfg: Configuration object containing all settings required for model operation.
        model: The XGBoost model after being trained.
        dtrain: The training dataset in DMatrix format.
        dtuning: The tuning (validation) dataset in DMatrix format.
        dheld_out: The held-out (test) dataset in DMatrix format.
        itrain: Iterator for the training dataset.
        ituning: Iterator for the tuning dataset.
        iheld_out: Iterator for the held-out dataset.
        keep_data_in_memory: Flag indicating whether to keep all data in memory or stream from disk.
    """

    def __init__(self, cfg: DictConfig):
        """Initializes the XGBoostClassifier with the provided configuration.

        Args:
            cfg: The configuration dictionary.
        """
        super().__init__()
        self.cfg = cfg
        self.keep_data_in_memory = cfg.model_params.iterator.keep_data_in_memory

        self.itrain = None
        self.ituning = None
        self.iheld_out = None

        self.dtrain = None
        self.dtuning = None
        self.dheld_out = None

        self.model = cfg.model_params.model
        # check that self.model is a valid model
        if not hasattr(self.model, "fit"):
            raise ValueError("Model does not have a fit method.")

    @TimeableMixin.TimeAs
    def _build_data(self):
        """Builds necessary data structures for training."""
        if self.keep_data_in_memory:
            self._build_iterators()
            self._build_matrix_in_memory()
        else:
            self._build_iterators()

    def _fit_from_partial(self):
        """Fits model until convergence or maximum epochs."""
        if not hasattr(self.model, "partial_fit"):
            raise ValueError(
                f"Data is loaded in shards, but {self.model.__class__.__name__} does not support partial_fit."
            )
        classes = self.itrain.get_classes()
        best_auc = 0
        best_epoch = 0
        for epoch in range(self.cfg.model_params.epochs):
            # train on each all data
            for shard_idx in range(len(self.itrain._data_shards)):
                data, labels = self.itrain.get_data_shards(shard_idx)
                # if self.model.shuffle: # TODO: check this for speed
                #     # shuffle data
                #     indices = np.random.permutation(len(labels))
                #     data = data[indices]
                #     labels = labels[indices]
                self.model.partial_fit(data, labels, classes=classes)
            # evaluate on tuning set
            auc = self.evaluate()
            # early stopping
            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch
            if epoch - best_epoch > self.cfg.model_params.early_stopping_rounds:
                break

    @TimeableMixin.TimeAs
    def _train(self):
        """Trains the model."""
        # two cases: data is in memory or data is streamed
        if self.keep_data_in_memory:
            self.model.fit(self.dtrain.get_data(), self.dtrain.get_label())
        else:
            self._fit_from_partial()

    @TimeableMixin.TimeAs
    def train(self):
        """Trains the model."""
        self._build_data()
        self._train()

    @TimeableMixin.TimeAs
    def _build_matrix_in_memory(self):
        """Builds the DMatrix from the data in memory."""
        self.dtrain = SklearnMatrix(*self.itrain.get_data())
        self.dtuning = SklearnMatrix(*self.ituning.get_data())
        self.dheld_out = SklearnMatrix(*self.iheld_out.get_data())

    @TimeableMixin.TimeAs
    def _build_iterators(self):
        """Builds the iterators for training, validation, and testing."""
        self.itrain = SklearnIterator(self.cfg, split="train")
        self.ituning = SklearnIterator(self.cfg, split="tuning")
        self.iheld_out = SklearnIterator(self.cfg, split="held_out")

    @TimeableMixin.TimeAs
    def evaluate(self, split: str = "tuning") -> float:
        """Evaluates the model on the tuning set.

        Returns:
            The evaluation metric as the ROC AUC score.
        """
        # depending on split point to correct data
        if split == "tuning":
            dsplit = self.dtuning
            isplit = self.ituning
        elif split == "held_out":
            dsplit = self.dheld_out
            isplit = self.iheld_out
        elif split == "train":
            dsplit = self.dtrain
            isplit = self.itrain
        else:
            raise ValueError(f"Split {split} is not valid.")

        # check if model has predict_proba method
        if not hasattr(self.model, "predict_proba"):
            raise ValueError(f"Model {self.model.__class__.__name__} does not have a predict_proba method.")
        # two cases: data is in memory or data is streamed
        if self.keep_data_in_memory:
            y_pred = self.model.predict_proba(dsplit.get_data())[:, 1]
            y_true = dsplit.get_label()
        else:
            y_pred = []
            y_true = []
            for shard_idx in range(len(isplit._data_shards)):
                data, labels = isplit.get_data_shards(shard_idx)
                y_pred.extend(self.model.predict_proba(data)[:, 1])
                y_true.extend(labels)
            y_pred = np.array(y_pred)
            y_true = np.array(y_true)

        # check if y_pred and y_true are not empty
        if len(y_pred) == 0 or len(y_true) == 0:
            raise ValueError("Predictions or true labels are empty.")
        return roc_auc_score(y_true, y_pred)

    def save_model(self, output_fp: str):
        """Saves the model to the specified file path.

        Args:
            output_fp: The file path to save the model to.
        """
        output_fp = Path(output_fp)
        # check if model has save method
        if not hasattr(self.model, "save_model"):
            logger.info(f"Model {self.model.__class__.__name__} does not have a save_model method.")
            logger.info("Model will be saved using pickle dump.")
            from pickle import dump

            with open(output_fp.parent / "model.pkl", "wb") as f:
                dump(self.model, f, protocol=5)
        else:
            self.model.save_model(output_fp)
