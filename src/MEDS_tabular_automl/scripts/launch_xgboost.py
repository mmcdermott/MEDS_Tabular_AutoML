from collections.abc import Callable, Mapping
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import hydra
import numpy as np
import polars as pl
import scipy.sparse as sp
import xgboost as xgb
from loguru import logger
from mixins import TimeableMixin
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score

from MEDS_tabular_automl.describe_codes import get_feature_columns
from MEDS_tabular_automl.file_name import get_model_files, list_subdir_files
from MEDS_tabular_automl.utils import get_feature_indices, hydra_loguru_init

config_yaml = files("MEDS_tabular_automl").joinpath("configs/launch_xgboost.yaml")
if not config_yaml.is_file():
    raise FileNotFoundError("Core configuration not successfully installed!")


class Iterator(xgb.DataIter, TimeableMixin):
    """Iterator class for loading and processing data shards for use in XGBoost models.

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

    def __init__(self, cfg: DictConfig, split: str = "train"):
        """Initializes the Iterator with the provided configuration and data split.

        Args:
            cfg: A configuration dictionary containing parameters for
                data processing, feature selection, and other settings.
            split: The data split to use, which can be one of "train", "tuning",
                or "held_out". This determines which subset of the data is loaded and processed.
        """
        # generate_permutations(cfg.tabularization.window_sizes)
        # generate_permutations(cfg.tabularization.aggs)
        self.cfg = cfg
        self.split = split
        # Load shards for this split
        self._data_shards = sorted(
            [shard.stem for shard in list_subdir_files(Path(cfg.input_label_dir) / split, "parquet")]
        )
        self.valid_event_ids, self.labels = self.load_labels()
        self.codes_set, self.code_masks, self.num_features = self._get_code_set()
        self._it = 0

        super().__init__(cache_prefix=Path(cfg.cache_dir))

    @TimeableMixin.TimeAs
    def _get_code_masks(self, feature_columns: list, codes_set: set) -> Mapping[str, list[bool]]:
        """Creates boolean masks for filtering features.

        Creates a dictionary of boolean masks for each aggregation type. The masks are used to filter
        the feature columns based on the specified included codes and minimum code inclusion frequency.

        Args:
            feature_columns: List of feature columns.
            codes_set: Set of codes to include.

        Returns:
            Dictionary of code masks for each aggregation.
        """
        code_masks = {}
        for agg in set(self.cfg.tabularization.aggs):
            feature_ids = get_feature_indices(agg, feature_columns)
            code_mask = [True if idx in codes_set else False for idx in feature_ids]
            code_masks[agg] = code_mask
        return code_masks

    @TimeableMixin.TimeAs
    def _load_matrix(self, path: Path) -> sp.csc_matrix:
        """Loads a sparse matrix from disk.

        Args:
            path: Path to the sparse matrix.

        Returns:
            The sparse matrix.

        Raises:
            ValueError: If the loaded array does not have exactly 3 rows, indicating an unexpected format.
        """
        npzfile = np.load(path)
        array, shape = npzfile["array"], npzfile["shape"]
        if array.shape[0] != 3:
            raise ValueError(f"Expected array to have 3 rows, but got {array.shape[0]} rows")
        data, row, col = array
        return sp.csc_matrix((data, (row, col)), shape=shape)

    @TimeableMixin.TimeAs
    def load_labels(self) -> tuple[Mapping[int, list], Mapping[int, list]]:
        """Loads valid event ids and labels for each shard.

        Returns:
            A tuple containing two mappings: one from shard indices to lists of valid event IDs
            which is used for indexing rows in the sparse matrix, and another from shard indices
            to lists of corresponding labels.
        """
        label_fps = {
            shard: (Path(self.cfg.input_label_dir) / self.split / shard).with_suffix(".parquet")
            for shard in self._data_shards
            for shard in self._data_shards
        }
        cached_labels, cached_event_ids = dict(), dict()
        for shard, label_fp in label_fps.items():
            label_df = pl.scan_parquet(label_fp)
            cached_event_ids[shard] = label_df.select(pl.col("event_id")).collect().to_series()

            # TODO: check this for Nan or any other case we need to worry about
            cached_labels[shard] = label_df.select(pl.col("label")).collect().to_series()
            if self.cfg.model_params.iterator.binarize_task:
                cached_labels[shard] = cached_labels[shard].map_elements(
                    lambda x: 1 if x > 0 else 0, return_dtype=pl.Int8
                )

        return cached_event_ids, cached_labels

    @TimeableMixin.TimeAs
    def _get_code_set(self) -> tuple[set[int], Mapping[str, list[bool]], int]:
        """Determines the set of feature codes to include based on the configuration settings.

        Returns:
            A tuple containing:
            - A set of feature indices to be included.
            - A mapping from aggregation types to boolean masks indicating whether each feature is included.
            - The total number of features.
        """
        feature_columns = get_feature_columns(self.cfg.tabularization.filtered_code_metadata_fp)
        feature_dict = {col: i for i, col in enumerate(feature_columns)}
        allowed_codes = set(self.cfg.tabularization._resolved_codes)
        codes_set = {feature_dict[code] for code in feature_dict if code in allowed_codes}

        return (
            codes_set,
            self._get_code_masks(feature_columns, codes_set),
            len(feature_columns),
        )

    @TimeableMixin.TimeAs
    def _load_dynamic_shard_from_file(self, path: Path, idx: int) -> sp.csc_matrix:
        """Loads a specific data shard into memory as a sparse matrix.

        Args:
            path: Path to the sparse shard.
            idx: Index of the shard.

        Returns:
            The sparse matrix loaded from the file.
        """
        # column_shard is of form event_idx, feature_idx, value
        matrix = self._load_matrix(path)
        if path.stem in ["first", "present"]:
            agg = f"static/{path.stem}"
        else:
            agg = f"{path.parent.stem}/{path.stem}"

        return self._filter_shard_on_codes_and_freqs(agg, matrix)

    @TimeableMixin.TimeAs
    def _get_dynamic_shard_by_index(self, idx: int) -> sp.csc_matrix:
        """Loads a shard and returns it as a sparse matrix after applying feature inclusion filtering.

        Args:
            idx: Index of the shard to load from disk.

        Returns:
            The filtered sparse matrix.

        Raises:
            ValueError: If any of the required files for the shard do not exist.
        """
        # get all window_size x aggreagation files using the file resolver
        files = get_model_files(self.cfg, self.split, self._data_shards[idx])

        if not all(file.exists() for file in files):
            raise ValueError(f"Not all files exist for shard {self._data_shards[idx]}")

        dynamic_cscs = [self._load_dynamic_shard_from_file(file, idx) for file in files]

        combined_csc = sp.hstack(dynamic_cscs, format="csc")

        return combined_csc

    @TimeableMixin.TimeAs
    def _get_shard_by_index(self, idx: int) -> tuple[sp.csc_matrix, np.ndarray]:
        """Loads a specific shard of data from disk and concatenate with static data.

        Args:
            idx: Index of the shard to load.

        Returns:
            A tuple containing the combined feature data and the corresponding labels
            for the given shard.
        """
        dynamic_df = self._get_dynamic_shard_by_index(idx)
        label_df = self.labels[self._data_shards[idx]]
        return dynamic_df, label_df

    @TimeableMixin.TimeAs
    def _filter_shard_on_codes_and_freqs(self, agg: str, df: sp.csc_matrix) -> sp.csc_matrix:
        """Filters the given data frame based on the inclusion sets and aggregation type.

        Given the codes_mask, the method filters the dynamic data frame to only include
        columns that are True in the mask.

        Args:
            agg: The aggregation type used to determine the filtering logic.
            df: The data frame to be filtered.

        Returns:
            The filtered data frame.
        """
        if self.codes_set is None:
            return df

        ckey = f"_filter_shard_on_codes_and_freqs/{agg}"
        self._register_start(key=ckey)

        df = df[:, self.code_masks[agg]]

        self._register_end(key=ckey)

        return df

    @TimeableMixin.TimeAs
    def next(self, input_data: Callable) -> int:
        """Advances the iterator by one step and provides data to XGBoost for DMatrix construction.

        Args:
            input_data: A function passed by XGBoost with the same signature as `DMatrix`.

        Returns:
            0 if end of iteration, 1 otherwise.
        """
        if self._it == len(self._data_shards):
            # return 0 to let XGBoost know this is the end of iteration
            return 0

        # input_data is a function passed in by XGBoost who has the exact same signature of
        # ``DMatrix``
        X, y = self._get_shard_by_index(self._it)  # self._data_shards[self._it])
        input_data(data=sp.csr_matrix(X), label=y)
        self._it += 1
        # Return 1 to let XGBoost know we haven't seen all the files yet.
        return 1

    @TimeableMixin.TimeAs
    def reset(self):
        """Resets the iterator to its beginning."""
        self._it = 0

    @TimeableMixin.TimeAs
    def collect_in_memory(self) -> tuple[sp.csc_matrix, np.ndarray]:
        """Collects data from all shards into memory and returns it.

        This method iterates through all data shards, retrieves the feature data and labels
        from each shard, and then concatenates them into a single sparse matrix and a single
        array, respectively.

        Returns:
            A tuple where the first element is a sparse matrix containing the
            feature data, and the second element is a numpy array containing the labels.

        Raises:
            ValueError: If no data is found in the shards or labels, indicating an issue with input files.
        """
        X = []
        y = []
        for i in range(len(self._data_shards)):
            X_, y_ = self._get_shard_by_index(i)
            X.append(X_)
            y.append(y_)
        if len(X) == 0 or len(y) == 0:
            raise ValueError("No data found in the shards or labels. Please check input files.")
        X = sp.vstack(X)
        y = np.concatenate(y, axis=0)
        return X, y


class XGBoostModel(TimeableMixin):
    """Class for configuring, training, and evaluating an XGBoost model.

    This class utilizes the configuration settings provided to manage the training and evaluation
    process of an XGBoost model, ensuring the model is trained and validated using specified parameters
    and data splits. It supports training with in-memory data handling as well as direct streaming from
    disk using iterators.

    Args:
        cfg: The configuration settings for the model, including data paths, model parameters,
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
        self.cfg = cfg
        self.keep_data_in_memory = cfg.model_params.iterator.keep_data_in_memory

        self.itrain = None
        self.ituning = None
        self.iheld_out = None

        self.dtrain = None
        self.dtuning = None
        self.dheld_out = None

        self.model = None

    @TimeableMixin.TimeAs
    def _build(self):
        """Builds necessary data structures for training."""
        if self.keep_data_in_memory:
            self._build_iterators()
            self._build_dmatrix_in_memory()
        else:
            self._build_iterators()
            self._build_dmatrix_from_iterators()

    @TimeableMixin.TimeAs
    def _train(self):
        """Trains the model."""
        self.model = xgb.train(
            OmegaConf.to_container(self.cfg.model_params.model),
            self.dtrain,
            num_boost_round=self.cfg.model_params.num_boost_round,
            early_stopping_rounds=self.cfg.model_params.early_stopping_rounds,
            # nthreads=self.cfg.nthreads,
            evals=[(self.dtrain, "train"), (self.dtuning, "tuning")],
            verbose_eval=0,
        )

    @TimeableMixin.TimeAs
    def train(self):
        """Trains the model."""
        self._build()
        self._train()

    @TimeableMixin.TimeAs
    def _build_dmatrix_in_memory(self):
        """Builds the DMatrix from the data in memory."""
        X_train, y_train = self.itrain.collect_in_memory()
        X_tuning, y_tuning = self.ituning.collect_in_memory()
        X_held_out, y_held_out = self.iheld_out.collect_in_memory()
        self.dtrain = xgb.DMatrix(X_train, label=y_train)
        self.dtuning = xgb.DMatrix(X_tuning, label=y_tuning)
        self.dheld_out = xgb.DMatrix(X_held_out, label=y_held_out)

    @TimeableMixin.TimeAs
    def _build_dmatrix_from_iterators(self):
        """Builds the DMatrix from the iterators."""
        self.dtrain = xgb.DMatrix(self.itrain)
        self.dtuning = xgb.DMatrix(self.ituning)
        self.dheld_out = xgb.DMatrix(self.iheld_out)

    @TimeableMixin.TimeAs
    def _build_iterators(self):
        """Builds the iterators for training, validation, and testing."""
        self.itrain = Iterator(self.cfg, split="train")
        self.ituning = Iterator(self.cfg, split="tuning")
        self.iheld_out = Iterator(self.cfg, split="held_out")

    @TimeableMixin.TimeAs
    def evaluate(self) -> float:
        """Evaluates the model on the test set.

        Returns:
            The evaluation metric as the ROC AUC score.
        """
        y_pred = self.model.predict(self.dheld_out)
        y_true = self.dheld_out.get_label()
        return roc_auc_score(y_true, y_pred)


@hydra.main(version_base=None, config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig) -> float:
    """Optimizes the model based on the provided configuration.

    Args:
        cfg: The configuration dictionary specifying model and training parameters.

    Returns:
        The evaluation result as the ROC AUC score on the held-out test set.
    """

    # print(OmegaConf.to_yaml(cfg))
    if not cfg.loguru_init:
        hydra_loguru_init()
    try:
        model = XGBoostModel(cfg)
        model.train()
        auc = model.evaluate()
        logger.info(f"AUC: {auc}")

        # print(
        #     "Time Profiling for window sizes ",
        #     f"{cfg.tabularization.window_sizes} and min ",
        #     f"code frequency of {cfg.tabularization.min_code_inclusion_frequency}:",
        # )
        # print("Train Time: \n", model._profile_durations())
        # print("Train Iterator Time: \n", model.itrain._profile_durations())
        # print("Tuning Iterator Time: \n", model.ituning._profile_durations())
        # print("Held Out Iterator Time: \n", model.iheld_out._profile_durations())

        # save model
        save_dir = Path(cfg.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model_time = datetime.now().strftime("%H%M%S%f")
        model.model.save_model(save_dir / f"{auc:.4f}_model_{model_time}.json")
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        auc = 0.0
    return auc


if __name__ == "__main__":
    main()
