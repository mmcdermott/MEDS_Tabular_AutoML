import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import xgboost as xgb
import polars as pl
import numpy as np
import pyarrow as pa
import polars.selectors as cs
from sklearn.metrics import mean_absolute_error

import os
from typing import List, Callable

class Iterator(xgb.DataIter):
    def __init__(self, cfg: DictConfig, split: str = "train"):
        """
        Initialize the Iterator with the provided configuration and split.

        Args:
        - cfg (DictConfig): Configuration dictionary.
        - split (str): The data split to use ("train", "tuning", or "held_out").

        """

        self.cfg = cfg
        self.data_path = Path(cfg.tabularized_data_dir)
        self.dynamic_data_path = self.data_path / "summarize" / split
        self.static_data_path = self.data_path / "static" / split
        self._data_shards = [
            x.stem
            for x in self.static_data_path.iterdir()
            if x.is_file() and x.suffix == ".parquet"
        ]

        if cfg.iterator.keep_static_data_in_memory:
            self._static_shards = self._get_static_shards() # do we want to cache this differently to share across workers or iterators? 

        self._it = 0
        # XGBoost will generate some cache files under current directory with the prefix
        # "cache"
        super().__init__(cache_prefix=os.path.join(".", "cache"))
    
    def _get_static_shards(self) -> dict:
        """
        Load static shards into memory.

        Returns:
        - dict: Dictionary with shard names as keys and data frames as values.
        
        """
        static_shards = {}
        for iter in self._data_shards:
            static_shards[iter] = pl.scan_parquet(self.static_data_path / f"{iter}.parquet")
        return static_shards
    
    def _load_shard(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Load a specific shard of data from disk and concatenate with static data.

        Args:
        - idx (int): Index of the shard to load.

        Returns:
        - X (pl.DataFrame): Feature data frame.
        - y (pl.Series): Labels.
        
        """
        # concatinate with static data
        if self.cfg.iterator.keep_static_data_in_memory:
            df = self._static_shards[self._data_shards[idx]]
        else:
            df = pl.scan_parquet(self.static_data_path / f"{self._data_shards[idx]}.parquet")


        ### TODO: Add in min_code_inclusion_frequency? 

        codes_set = set(self.cfg.codes) if self.cfg.codes else None
        aggs_set = set(self.cfg.aggs) if self.cfg.aggs else None

        for window in self.cfg.window_sizes:
            dynamic_df = pl.scan_parquet(
                self.dynamic_data_path / window / f"{self._data_shards[idx]}.parquet"
            )

            ### TODO: Update this for the correct order of column names from Nassim
            columns = dynamic_df.schema.keys() # should I use df.columns instead?
            selected_columns = [
                col for col in columns
                if (parts := col.split('/')) and len(parts) > 2
                and (codes_set is None or parts[0] in codes_set)
                and (aggs_set is None or parts[-1] in aggs_set)
            ]
            selected_columns.extend(['patient_id', 'timestamp'])
            dynamic_df = dynamic_df.select(selected_columns)


            df = pl.concat([df, dynamic_df], how='align')

        ### TODO: Figure out features vs labels --> look at esgpt_baseline for loading in labels based on tasks 

        y = df.select("label")
        X = df.select([col for col in df.schema.keys() if col != "label"])

        ### TODO: Figure out best way to export this to dmatrix --> can we use scipy sparse matrix?
        ### TODO: fill nones/nulls with zero if this is needed for xgboost 
        return X.collect().to_numpy(), y.collect().to_numpy() # convert to sparse matrix instead

    def next(self, input_data: Callable):
        """
        Advance the iterator by 1 step and pass the data to XGBoost.  This function is
        called by XGBoost during the construction of ``DMatrix``

        Args:
        - input_data (Callable): A function passed by XGBoost with the same signature as `DMatrix`.

        Returns:
        - int: 0 if end of iteration, 1 otherwise.
        """
        if self._it == len(self._data_shards):
            # return 0 to let XGBoost know this is the end of iteration
            return 0

        # input_data is a function passed in by XGBoost who has the exact same signature of
        # ``DMatrix``
        X, y = self._load_shard(self._it)  # self._data_shards[self._it])
        input_data(data=X, label=y)
        self._it += 1
        # Return 1 to let XGBoost know we haven't seen all the files yet.
        return 1

    def reset(self):
        """
        Reset the iterator to its beginning.

        Example:
        >>> cfg_dict = {
        ...     "tabularize": {
        ...         "tabularized_data_dir": "/path/to/tabularized/data",
        ...     },
        ...     "iterator": {
        ...         "keep_static_data_in_memory": True
        ...     }
        ... }
        >>> cfg = OmegaConf.create(cfg_dict)
        >>> it = Iterator(cfg, split='train')
        >>> it._it = 1
        >>> it.reset()
        >>> it._it
        0
        """
        self._it = 0

class XGBoostClassifier:
    def __init__(self, cfg: DictConfig):
        """
        Initialize the XGBoostClassifier with the provided configuration.

        Args:
        - cfg (DictConfig): Configuration dictionary.
        """

        self.cfg = cfg

        self.itrain = Iterator(cfg)
        self.ival = Iterator(cfg, split="tuning")
        self.itest = Iterator(cfg, split="held_out")

        self.dtrain = xgb.DMatrix(self.ival)
        self.dval = xgb.DMatrix(self.itest)
        self.dtest = xgb.DMatrix(self.itest)

        self.model = xgb.train(OmegaConf.to_container(self.cfg.model), self.dtrain)

    def evaluate(self) -> float:
        """
        Evaluate the model on the test set.

        Returns:
        - float: Evaluation metric (mae).

        Example:
        >>> cfg_dict = {
        ...     "model": {
        ...         "booster": "gbtree",
        ...         "objective": "reg:squarederror",
        ...     }
        ... }
        >>> cfg = OmegaConf.create(cfg_dict)
        >>> classifier = XGBoostClassifier(cfg=cfg)

        >>> n_samples = 1000
        >>> n_features = 10
        >>> X_test = np.random.rand(n_samples, n_features)
        >>> y_test = np.random.rand(n_samples)

        >>> mae = classifier.evaluate(X_test, y_test)
        >>> isinstance(mae, float)
        True
        """
        ### TODO: Figure out exactly what we want to do here

        y_pred = self.model.predict(self.dtest)
        y_true = self.dtest.get_label() 
        return mean_absolute_error(y_true, y_pred)


@hydra.main(version_base=None, config_path="configs", config_name="tabularize_sweep")
def optimize(cfg: DictConfig) -> float:
    """
    Optimize the model based on the provided configuration.

    Args:
    - cfg (DictConfig): Configuration dictionary.

    Returns:
    - float: Evaluation result.

    """

    model = XGBoostClassifier(cfg)
    return model.evaluate()


if __name__ == "__main__":
    optimize()
