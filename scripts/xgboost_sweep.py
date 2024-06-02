import json
import os
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from timeit import timeit

import hydra
import numpy as np
import polars as pl
import scipy.sparse as sp
import xgboost as xgb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_absolute_error

from MEDS_tabular_automl.file_name import FileNameResolver
from MEDS_tabular_automl.utils import get_feature_indices, load_matrix


class Iterator(xgb.DataIter):
    def __init__(self, cfg: DictConfig, split: str = "train"):
        """Initialize the Iterator with the provided configuration and split.

        Args:
        - cfg (DictConfig): Configuration dictionary.
        - split (str): The data split to use ("train", "tuning", or "held_out").
        """
        self.cfg = cfg
        self.file_name_resolver = FileNameResolver(cfg)
        self.split = split
        # self.data_path = Path(cfg.tabularized_data_dir)
        # self.dynamic_data_path = self.data_path / "sparse" / split
        # self.task_data_path = self.data_path / "task" / split
        self._data_shards = sorted(
            [shard.stem for shard in self.file_name_resolver.list_label_files(split)]
        )  # [2, 4, 5] #
        self.valid_event_ids, self.labels = self.load_labels()
        self.codes_set, self.num_features = self._get_code_set()
        feature_columns = json.load(open(self.file_name_resolver.get_feature_columns_fp()))
        self.agg_to_feature_ids = {agg: get_feature_indices(agg, feature_columns) for agg in cfg.aggs}

        self._it = 0

        # XGBoost will generate some cache files under current directory with the prefix
        # "cache"
        super().__init__(cache_prefix=os.path.join(".", "cache"))

    def load_labels(self) -> tuple[Mapping[int, list], Mapping[int, list]]:
        """Loads valid event ids and labels for each shard.

        Returns:
        - Tuple[Mapping[int, list], Mapping[int, list]]: Tuple containing:
            dictionary from shard number to list of valid event ids -- used for indexing rows
                in the sparse matrix
            dictionary from shard number to list of labels for these valid event ids
        """
        label_fps = {
            shard: self.file_name_resolver.get_label(self.split, shard) for shard in self._data_shards
        }
        cached_labels, cached_event_ids = dict(), dict()
        for shard, label_fp in label_fps.items():
            label_df = pl.scan_parquet(label_fp)
            cached_event_ids[shard] = label_df.select(pl.col("event_id")).collect().to_series()
            cached_labels[shard] = label_df.select(pl.col("label")).collect().to_series()
        return cached_event_ids, cached_labels

    def _get_code_set(self) -> set:
        """Get the set of codes to include in the data based on the configuration."""
        with open(self.file_name_resolver.get_feature_columns_fp()) as f:
            feature_columns = json.load(f)
        feature_dict = {col: i for i, col in enumerate(feature_columns)}
        if self.cfg.codes is not None:
            codes_set = {feature_dict[code] for code in set(self.cfg.codes) if code in feature_dict}

        if self.cfg.min_code_inclusion_frequency is not None:
            with open(self.file_name_resolver.get_feature_freqs_fp()) as f:
                feature_freqs = json.load(f)
            min_frequency_set = {
                key for key, value in feature_freqs.items() if value >= self.cfg.min_code_inclusion_frequency
            }
            frequency_set = {feature_dict[code] for code in min_frequency_set if code in feature_dict}

        if self.cfg.codes is not None and self.cfg.min_code_inclusion_frequency is not None:
            codes_set = codes_set.intersection(frequency_set)
        elif self.cfg.codes is not None:
            codes_set = codes_set
        elif self.cfg.min_code_inclusion_frequency is not None:
            codes_set = frequency_set
        else:
            codes_set = None  # set(feature_columns)
        # TODO: make sure we aren't filtering out static columns!!!
        return list(codes_set), len(feature_columns)

    def _load_dynamic_shard_from_file(self, path: Path, idx: int) -> sp.csc_matrix:
        """Load a sparse shard into memory.

        Args:
            - path (Path): Path to the sparse shard.

        Returns:
            - sp.coo_matrix: Data frame with the sparse shard.
        >>> import tempfile
        >>> from types import SimpleNamespace
        >>> with tempfile.TemporaryDirectory() as tempdir:
        ...     sample_shard_path = Path(tempdir) / "sample_shard.npy"
        ...     sample_shard_data = np.array([[0, 1, 0],
        ...                               [1, 0, 1],
        ...                               [0, 1, 0]])
        ...     sample_filtered_data = np.array([[1, 0],
        ...                               [0, 1],
        ...                               [1, 0]])
        ...     np.save(sample_shard_path, sample_shard_data)
        ...     cfg = SimpleNamespace(
        ...         aggs=None,
        ...         window_sizes=None,
        ...         codes=None,
        ...         min_code_inclusion_frequency=None,
        ...         tabularized_data_dir=Path(tempdir)
        ...     )
        ...     feature_columns = ["code1/code", "code2/code", "value1/value"]
        ...     with open(Path(tempdir) / "feature_columns.json", "w") as f:
        ...         json.dump(feature_columns, f)
        ...     iterator_instance = Iterator(cfg)
        ...     iterator_instance.codes_mask = np.array([False, True, True])
        ...     loaded_shard = iterator_instance._load_dynamic_shard_from_file(sample_shard_path)
        ...     assert isinstance(loaded_shard, sp.csr_matrix)
        ...     expected_csr = sp.csr_matrix(sample_filtered_data)
        ...     assert sp.issparse(loaded_shard)
        ...     assert np.array_equal(loaded_shard.data, expected_csr.data)
        ...     assert np.array_equal(loaded_shard.indices, expected_csr.indices)
        ...     assert np.array_equal(loaded_shard.indptr, expected_csr.indptr)
        """
        # column_shard is of form event_idx, feature_idx, value
        matrix = load_matrix(path)
        if path.stem in ["first", "present"]:
            agg = f"static/{path.stem}"
        else:
            agg = f"{path.parent.stem}/{path.stem}"

        return self._filter_shard_on_codes_and_freqs(agg, sp.csc_matrix(matrix))

    def _get_dynamic_shard_by_index(self, idx: int) -> sp.csr_matrix:
        """Load a specific shard of dynamic data from disk and return it as a sparse matrix after filtering
        column inclusion.

        Args:
        - idx (int): Index of the shard to load.

        Returns:
        - sp.csr_matrix: Filtered sparse matrix.
        """
        # TODO Nassim Fix this guy
        # get all window_size x aggreagation files using the file resolver
        files = self.file_name_resolver.get_model_files(
            self.cfg.window_sizes, self.cfg.aggs, self.split, self._data_shards[idx]
        )
        assert all([file.exists() for file in files])
        shard_name = self._data_shards[idx]
        dynamic_csrs = [self._load_dynamic_shard_from_file(file, idx) for file in files]
        combined_csr = sp.hstack(dynamic_csrs, format="csr")  # TODO: check this
        # Filter Rows
        valid_indices = self.valid_event_ids[shard_name]
        return combined_csr[valid_indices, :]

    def _get_shard_by_index(self, idx: int) -> tuple[sp.csr_matrix, np.ndarray]:
        """Load a specific shard of data from disk and concatenate with static data.

        Args:
        - idx (int): Index of the shard to load.

        Returns:
        - X (scipy.sparse.csr_matrix): Feature data frame.ß
        - y (numpy.ndarray): Labels.
        """
        time = datetime.now()
        dynamic_df = self._get_dynamic_shard_by_index(idx)
        logger.debug(f"Dynamic data loading took {datetime.now() - time}")
        time = datetime.now()
        label_df = self.labels[self._data_shards[idx]]
        logger.debug(f"Task data loading took {datetime.now() - time}")
        return dynamic_df, label_df

    def _filter_shard_on_codes_and_freqs(self, agg: str, df: sp.csc_matrix) -> sp.csc_matrix:
        """Filter the dynamic data frame based on the inclusion sets. Given the codes_mask, filter the data
        frame to only include columns that are True in the mask.

        Args:
        - df (scipy.sparse.coo_matrix): Data frame to filter.

        Returns:
        - df (scipy.sparse.sp.csr_matrix): Filtered data frame.
        """
        if self.codes_set is None:
            return df
        feature_ids = self.agg_to_feature_ids[agg]
        code_mask = [True if idx in self.codes_set else False for idx in feature_ids]
        return df[:, code_mask]  # [:, list({index for index in self.codes_set if index < df.shape[1]})]

    def next(self, input_data: Callable):
        """Advance the iterator by 1 step and pass the data to XGBoost.  This function is called by XGBoost
        during the construction of ``DMatrix``

        Args:
        - input_data (Callable): A function passed by XGBoost with the same signature as `DMatrix`.

        Returns:
        - int: 0 if end of iteration, 1 otherwise.
        """
        start_time = datetime.now()
        if self._it == len(self._data_shards):
            # return 0 to let XGBoost know this is the end of iteration
            return 0

        # input_data is a function passed in by XGBoost who has the exact same signature of
        # ``DMatrix``
        X, y = self._get_shard_by_index(self._it)  # self._data_shards[self._it])
        input_data(data=X, label=y)
        self._it += 1
        # Return 1 to let XGBoost know we haven't seen all the files yet.
        logger.debug(f"******** One iteration took {datetime.now() - start_time}")
        return 1

    def reset(self):
        """Reset the iterator to its beginning."""
        self._it = 0

    def collect_in_memory(self) -> tuple[sp.coo_matrix, np.ndarray]:
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
    logger.debug("Initializing XGBoost model")
    model = XGBoostModel(cfg)
    logger.debug("Training XGBoost model")
    time = datetime.now()
    model.train()
    logger.debug(f"Training took {datetime.now() - time}")
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
    # start_time = datetime.now()
    # xgboost()
    # logger.debug(f"Total time: {datetime.now() - start_time}")
    num = 10
    time = timeit(xgboost, number=num) / num
    logger.debug(f"Training time averaged over {num} runs: {time}")
