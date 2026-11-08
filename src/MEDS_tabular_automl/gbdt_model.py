import logging
from abc import abstractmethod
from pathlib import Path

import meds
import numpy as np
import polars as pl
from meds_evaluation.schema import BINARY_CLASSIFICATION_SCHEMA_DICT
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score

from .base_model import BaseModel
from .tabular_dataset import TabularDataset

logger = logging.getLogger(__name__)


class GradientBoostedTreesModel(BaseModel):
    """Shared base of the gradient-boosted tree launchers (`XGBoostModel`, `LightGBMModel`).

    It owns what does not depend on the library: one data iterator per split (`iterator_class`), training
    (`_build` then `_train`), evaluation by AUROC, and the predictions frame aligned with the label files.
    Subclasses provide the library-specific parts: `_build` (turn the splits into the library's data
    structures), `_train`, `_predict` (labels and probabilities for a split), `save_model` and `load_model`.

    Args:
        cfg: The model launcher configuration: data paths, model parameters, training parameters, and
            data loading flags.

    Attributes:
        cfg: The configuration.
        model: The trained booster.
        dtrain, dtuning, dheld_out: The splits in the library's format.
        itrain, ituning, iheld_out: The data iterators of the splits.
        keep_data_in_memory: Whether to load all data in memory or stream it from disk, where supported.
    """

    iterator_class: type[TabularDataset] = TabularDataset

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.keep_data_in_memory = cfg.data_loading_params.keep_data_in_memory

        self.itrain = None
        self.ituning = None
        self.iheld_out = None

        self.dtrain = None
        self.dtuning = None
        self.dheld_out = None

        self.model = None

    def _build_iterators(self):
        """Builds the data iterators of the train, tuning and held-out splits."""
        self.itrain = self.iterator_class(self.cfg, split="train")
        self.ituning = self.iterator_class(self.cfg, split="tuning")
        self.iheld_out = self.iterator_class(self.cfg, split="held_out")

    @abstractmethod
    def _build(self):  # pragma: no cover
        """Builds the iterators and the library's data structures for every split."""

    @abstractmethod
    def _train(self):  # pragma: no cover
        """Trains `self.model` on `self.dtrain`, with early stopping on `self.dtuning`."""

    @abstractmethod
    def _predict(self, split: str = "held_out") -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        """Returns the labels and predicted probabilities of a split."""

    @abstractmethod
    def load_model(self, model_fp: Path):  # pragma: no cover
        """Loads a model saved by `save_model`."""

    def train(self):
        """Builds the data and trains the model."""
        self._build()
        self._train()

    def evaluate(self, split: str = "tuning") -> float:
        """Returns the AUROC of the model on a split (0 if the split has a single class)."""
        y_true, y_pred = self._predict(split)
        if len(np.unique(y_true)) != 2:
            logger.warning("There is only one class in the ground truth labels, returning 0 AUC")
            return 0.0
        return roc_auc_score(y_true, y_pred)

    def predict(self, split: str = "held_out") -> pl.DataFrame:
        """Returns the predictions of a split, one row per label, in the order of the label files.

        Raises:
            ValueError: If the split is unknown, or if the labels do not match the label files.
        """
        y_true, y_pred = self._predict(split)

        if split == meds.tuning_split:
            iterator = self.ituning
        elif split == meds.held_out_split:
            iterator = self.iheld_out
        elif split == meds.train_split:
            iterator = self.itrain
        else:
            raise ValueError(f"Invalid split for evaluation: {split}")
        _, cached_labels = iterator._load_ids_and_labels(load_ids=False, load_labels=True)
        parquet_files = [
            Path(self.cfg.path.input_label_dir) / split / f"{key}.parquet" for key in cached_labels
        ]
        labels = pl.concat([pl.read_parquet(fp) for fp in parquet_files])
        if "event_id" not in labels.schema:
            labels = labels.with_row_index("event_id")
        if "time" not in labels.schema:
            labels = labels.rename({"prediction_time": "time"})
        if "boolean_value" in labels.schema:
            labels = labels.rename({"boolean_value": "label"})
        predictions_df = pl.DataFrame(
            {
                "subject_id": labels["subject_id"],
                "prediction_time": labels["time"],
                "boolean_value": y_true,
                "predicted_boolean_value": y_pred.round(),
                "predicted_boolean_probability": y_pred,
                "event_id": labels["event_id"],
            },
            schema={**BINARY_CLASSIFICATION_SCHEMA_DICT, "event_id": pl.Int64},
        )
        if not (predictions_df["boolean_value"] == labels["label"]).all():
            mismatched_labels = predictions_df["boolean_value"] != labels["label"]
            raise ValueError(f"Label mismatch: {sum(mismatched_labels)} incorrect predictions")

        return predictions_df
