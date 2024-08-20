import numpy as np
import scipy.sparse as sp
from mixins import TimeableMixin
from omegaconf import DictConfig

from .tabular_dataset import TabularDataset


class DenseIterator(TabularDataset, TimeableMixin):
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

    def densify(self) -> np.ndarray:
        """Builds the data as a dense matrix based on column subselection."""

        # get the column indices to include
        cols = self.get_feature_indices()

        # map those to the feature names in the data
        feature_names = self.get_all_column_names()
        selected_features = [feature_names[col] for col in cols]

        # get the dense matrix by iterating through the data shards
        data = []
        labels = []
        for shard_idx in range(len(self._data_shards)):
            shard_data, shard_labels = self.get_data_shards(shard_idx)
            shard_data = shard_data[:, cols]
            data.append(shard_data)
            labels.append(shard_labels)
        data = sp.vstack(data)
        labels = np.concatenate(labels, axis=0)
        return data, labels, selected_features
