import json
import os
from collections.abc import Callable
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_absolute_error


class Iterator(xgb.DataIter):
    def __init__(self, cfg: DictConfig, split: str = "train"):
        """Initialize the Iterator with the provided configuration and split.

        Args:
        - cfg (DictConfig): Configuration dictionary.
        - split (str): The data split to use ("train", "tuning", or "held_out").
        """
        self.cfg = cfg
        self.data_path = Path(cfg.tabularized_data_dir)
        self.dynamic_data_path = self.data_path / "ts" / split
        self.static_data_path = self.data_path / "static" / split
        self._data_shards = [shard.stem for shard in list(self.static_data_path.glob("*.parquet"))]
        if cfg.iterator.keep_static_data_in_memory:
            self._static_shards = self._collect_static_shards_in_memory()

        self.codes_set, self.aggs_set, self.min_frequency_set, self.window_set = self._get_inclusion_sets()

        self._it = 0

        # XGBoost will generate some cache files under current directory with the prefix
        # "cache"
        super().__init__(cache_prefix=os.path.join(".", "cache"))

    def _get_inclusion_sets(self) -> tuple[set, set, set]:
        """Get the inclusion sets for codes and aggregations.

        Returns:
        - tuple[set, set, set]: Tuple of sets for codes, aggregations, and minimum code frequency.
        """
        codes_set = None
        aggs_set = None
        min_frequency_set = None
        window_set = None

        if self.cfg.codes is not None:
            codes_set = set(self.cfg.codes)

        if self.cfg.aggs is not None:
            aggs_set = set(self.cfg.aggs)

        if self.cfg.min_code_inclusion_frequency is not None:
            with open(self.data_path / "feature_freqs.json") as f:
                feature_freqs = json.load(f)
            min_frequency_set = {
                key for key, value in feature_freqs.items() if value >= self.cfg.min_code_inclusion_frequency
            }
        if self.cfg.window_sizes is not None:
            window_set = set(self.cfg.window_sizes)

        return codes_set, aggs_set, min_frequency_set, window_set

    def _collect_static_shards_in_memory(self) -> dict:
        """Load static shards into memory.

        Returns:
        - dict: Dictionary with shard names as keys and data frames as values.
        """
        static_shards = {}
        for iter in self._data_shards:
            static_shards[iter] = self._load_static_shard_by_index(iter)
        return static_shards

    def _load_static_shard_by_index(self, idx: int) -> sp.csc_matrix:
        """Load a static shard into memory.

        Args:
        - idx (int): Index of the shard to load.

        Returns:
        - sp.csc_matrix: Sparse matrix with the static shard.
        """
        return pd.read_parquet(self.static_data_path / f"{self._data_shards[int(idx)]}.parquet")

    def _get_static_shard_by_index(self, idx: int) -> pd.DataFrame:
        """Get the static shard from memory or disk.

        Args:
        - shard_name (str): Name of the shard.

        Returns:
        - pd.DataFrame: Data frame with the static shard.
        """
        if self.cfg.iterator.keep_static_data_in_memory:
            return self._static_shards[self._data_shards[idx]]
        else:
            return self._load_static_shard_by_index(self._data_shards[idx])

    def _get_task_by_index(self, idx: int) -> pd.DataFrame:
        """Get the task data for a specific shard.

        Args:
        - idx (int): Index of the shard.

        Returns:
        - pd.DataFrame: Data frame with the task data.
        """
        # TODO: replace with something real
        file = list(self.dynamic_data_path.glob(f"*/*/*/{self._data_shards[idx]}.pkl"))[0]
        shard = pd.read_pickle(file)
        shard["label"] = np.random.randint(0, 2, shard.shape[0])
        return shard[["patient_id", "timestamp", "label"]]

    def _load_dynamic_shard_from_file(self, path: Path) -> pd.DataFrame:
        """Load a sparse shard into memory. This returns a shard as a pandas dataframe, asserted that it is
        sorted on patient id and timestamp, if included.

        Args:
            - path (Path): Path to the sparse shard.

        Returns:
            - pd.DataFrame: Data frame with the sparse shard.
        """
        shard = pd.read_pickle(path)
        self._assert_correct_sorting(shard)
        return shard.drop(columns=["patient_id", "timestamp"])

    def _get_dynamic_shard_by_index(self, idx: int) -> tuple[sp.csr_matrix, np.ndarray]:
        """Load a specific shard of dynamic data from disk and return it as a sparse matrix after filtering
        column inclusion."""
        files = list(self.dynamic_data_path.glob(f"*/*/*/{self._data_shards[idx]}.pkl"))

        files = [file for file in files if self._filter_shard_files_on_window_and_aggs(file)]

        dynamic_dfs = [self._load_dynamic_shard_from_file(file) for file in files]
        dynamic_df = pd.concat(dynamic_dfs, axis=1)
        return self._filter_shard_on_codes_and_freqs(dynamic_df)

    def _get_shard_by_index(self, idx: int) -> tuple[sp.csr_matrix, np.ndarray]:
        """Load a specific shard of data from disk and concatenate with static data.

        Args:
        - idx (int): Index of the shard to load.

        Returns:
        - X (scipy.sparse.csr_matrix): Feature data frame.
        - y (numpy.ndarray): Labels.
        """

        dynamic_df = self._get_dynamic_shard_by_index(idx)

        # TODO: add in some type checking etc for safety
        static_df = self._get_static_shard_by_index(idx)

        task_df = self._get_task_by_index(idx)
        task_df = task_df.rename(
            columns={col: f"{col}/task" for col in task_df.columns if col not in ["patient_id", "timestamp"]}
        )
        df = pd.merge(task_df, static_df, on=["patient_id"], how="left")
        self._assert_correct_sorting(df)
        df = self._filter_shard_on_codes_and_freqs(df)
        df = pd.concat([df, dynamic_df], axis=1)

        return self._sparsify_shard(df)

    def _sparsify_shard(self, df: pd.DataFrame) -> tuple[sp.csc_matrix, np.ndarray]:
        """Make X and y as scipy sparse arrays for XGBoost.

        Args:
        - df (pandas.DataFrame): Data frame to sparsify.

        Returns:
        - tuple[scipy.sparse.csr_matrix, numpy.ndarray]: Tuple of feature data and labels.
        """
        labels = df.loc[:, [col for col in df.columns if col.endswith("/task")]]
        data = df.drop(columns=labels.columns)
        for col in data.columns:
            if not isinstance(data[col].dtype, pd.SparseDtype):
                data[col] = pd.arrays.SparseArray(data[col])
        sparse_matrix = data.sparse.to_coo()
        return csr_matrix(sparse_matrix), labels.values

    def _filter_shard_files_on_window_and_aggs(self, file: Path) -> bool:
        parts = file.relative_to(self.dynamic_data_path).parts
        if not parts:
            return False

        windows_part = parts[0]
        aggs_part = "/".join(parts[1:-1])

        return (self.window_set is None or windows_part in self.window_set) and (
            self.aggs_set is None or aggs_part in self.aggs_set
        )

    def _filter_shard_on_codes_and_freqs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter the dynamic data frame based on the inclusion sets.

        Args:
        - df (pd.DataFrame): Data frame to filter.

        Returns:
        - pd.DataFrame: Filtered data frame.
        """
        code_parts = ["/".join(col.split("/")[1:-2]) for col in df.columns]
        frequency_parts = ["/".join(col.split("/")[1:-1]) for col in df.columns]

        filtered_columns = [
            col
            for col, code_part, freq_part in zip(df.columns, code_parts, frequency_parts)
            if (self.codes_set is None or code_part in self.codes_set)
            and (self.min_frequency_set is None or freq_part in self.min_frequency_set)
        ]
        filtered_columns.extend([col for col in df.columns if col.endswith("/task")])

        return df[filtered_columns]

    def _assert_correct_sorting(self, shard: pd.DataFrame):
        """Assert that the shard is sorted correctly."""
        if "timestamp" in shard.columns:
            sort_columns = ["patient_id", "timestamp"]
        else:
            sort_columns = ["patient_id"]
        assert shard[sort_columns].equals(shard[sort_columns].sort_values(by=sort_columns)), (
            "Shard is not sorted on correctly. "
            "Please ensure that the data is sorted on patient_id and timestamp, if applicable."
        )

    def next(self, input_data: Callable):
        """Advance the iterator by 1 step and pass the data to XGBoost.  This function is called by XGBoost
        during the construction of ``DMatrix``

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
        X, y = self._get_shard_by_index(self._it)  # self._data_shards[self._it])
        input_data(data=X, label=y)
        self._it += 1
        # Return 1 to let XGBoost know we haven't seen all the files yet.
        return 1

    def reset(self):
        """Reset the iterator to its beginning."""
        self._it = 0

    def collect_in_memory(self) -> tuple[sp.csr_matrix, np.ndarray]:
        """Collect the data in memory.

        Returns:
        - tuple[np.ndarray, np.ndarray]: Tuple of feature data and labels.
        """
        X = []
        y = []
        for i in range(len(self._data_shards)):
            X_, y_ = self._get_shard_by_index(i)
            X.append(X_)
            y.append(y_)

        X = sp.vstack(X)
        y = np.concatenate(y, axis=0)
        return X, y


class XGBoostModel:
    def __init__(self, cfg: DictConfig):
        """Initialize the XGBoostClassifier with the provided configuration.

        Args:
        - cfg (DictConfig): Configuration dictionary.
        """

        self.cfg = cfg
        self.keep_data_in_memory = getattr(getattr(cfg, "iterator", {}), "keep_data_in_memory", True)

        self.itrain = None
        self.ival = None
        self.itest = None

        self.dtrain = None
        self.dval = None
        self.dtest = None

        self.model = None

    def train(self):
        """Train the model."""
        self._build()
        # TODO: add in eval, early stopping, etc.
        # TODO: check for Nan and inf in labels!
        self.model = xgb.train(
            OmegaConf.to_container(self.cfg.model), self.dtrain
        )  # do we want eval and things?

    def _build(self):
        """Build necessary data structures for training."""
        if self.keep_data_in_memory:
            self._build_iterators()
            self._build_dmatrix_in_memory()
        else:
            self._build_iterators()
            self._build_dmatrix_from_iterators()

    def _build_dmatrix_in_memory(self):
        """Build the DMatrix from the data in memory."""
        X_train, y_train = self.itrain.collect_in_memory()
        X_val, y_val = self.ival.collect_in_memory()
        X_test, y_test = self.itest.collect_in_memory()
        self.dtrain = xgb.DMatrix(X_train, label=y_train)
        self.dval = xgb.DMatrix(X_val, label=y_val)
        self.dtest = xgb.DMatrix(X_test, label=y_test)

    def _build_dmatrix_from_iterators(self):
        """Build the DMatrix from the iterators."""
        self.dtrain = xgb.DMatrix(self.ival)
        self.dval = xgb.DMatrix(self.itest)
        self.dtest = xgb.DMatrix(self.itest)

    def _build_iterators(self):
        """Build the iterators for training, validation, and testing."""
        self.itrain = Iterator(self.cfg, split="train")
        self.ival = Iterator(self.cfg, split="tuning")
        self.itest = Iterator(self.cfg, split="held_out")

    def evaluate(self) -> float:
        """Evaluate the model on the test set.

        Returns:
        - float: Evaluation metric (mae).
        """
        # TODO: Figure out exactly what we want to do here

        y_pred = self.model.predict(self.dtest)
        y_true = self.dtest.get_label()
        return mean_absolute_error(y_true, y_pred)


@hydra.main(version_base=None, config_path="../configs", config_name="xgboost_sweep")
def xgboost(cfg: DictConfig) -> float:
    """Optimize the model based on the provided configuration.

    Args:
    - cfg (DictConfig): Configuration dictionary.

    Returns:
    - float: Evaluation result.
    """
    model = XGBoostModel(cfg)
    model.train()
    # save model
    save_dir = (
        Path(cfg.model_dir)
        / "_".join(map(str, cfg.window_sizes))
        / "_".join([agg.replace("/", "") for agg in cfg.aggs])
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    model.model.save_model(save_dir / f"{np.random.randint(100000, 999999)}_model.json")
    return model.evaluate()


if __name__ == "__main__":
    xgboost()
