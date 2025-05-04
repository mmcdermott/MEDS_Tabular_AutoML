from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from mixins import TimeableMixin
from omegaconf import DictConfig
from scipy.stats import pearsonr

from .describe_codes import get_feature_columns
from .file_name import get_model_files, list_subdir_files
from .utils import get_feature_indices


class TabularDataset(TimeableMixin):
    """Tabular Dataset class for loading and processing data shards.

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
        super().__init__(cache_prefix=Path(cfg.path.cache_dir))
        self.cfg = cfg
        self.split = split
        # Load shards for this split
        self._data_shards = sorted(
            [
                shard.stem
                for shard in list_subdir_files(Path(cfg.path.input_label_cache_dir) / split, "parquet")
            ]
        )
        if len(self._data_shards) == 0:
            raise ValueError(
                "No labels found in the `input_label_cache_dir` "
                + str(Path(cfg.path.input_label_cache_dir).resolve())
            )
        self.valid_event_ids, self.labels = None, None

        self.codes_set, self.code_masks, self.num_features = self._get_code_set()

        self._set_scaler()
        self._set_imputer()

        self.valid_event_ids, self.labels = self._load_ids_and_labels()
        # check if the labels are empty
        if len(self.labels) == 0:
            raise ValueError("No labels found.")

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
            code_mask = [idx in codes_set for idx in feature_ids]
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
    def _load_ids_and_labels(
        self, load_ids: bool = True, load_labels: bool = True
    ) -> tuple[Mapping[int, list], Mapping[int, list]]:
        """Loads valid event ids and labels for each shard.

        Returns:
            A tuple containing two mappings: one from shard indices to lists of valid event IDs
            which is used for indexing rows in the sparse matrix, and another from shard indices
            to lists of corresponding labels.
        """
        label_fps = {
            shard: (Path(self.cfg.path.input_label_cache_dir) / self.split / shard).with_suffix(".parquet")
            for shard in self._data_shards
            for shard in self._data_shards
        }
        cached_labels, cached_event_ids = dict(), dict()
        for shard, label_fp in label_fps.items():
            label_df = pl.scan_parquet(label_fp)
            if "event_id" not in label_df.schema:
                label_df = label_df.with_row_index("event_id")

            if load_ids:
                cached_event_ids[shard] = label_df.select(pl.col("event_id")).collect().to_series()

            if load_labels:
                label_col = "boolean_value" if "boolean_value" in label_df.schema else "label"
                cached_labels[shard] = label_df.select(pl.col(label_col)).collect().to_series()
                if self.cfg.data_loading_params.binarize_task:
                    cached_labels[shard] = cached_labels[shard].map_elements(
                        lambda x: 1 if x > 0 else 0, return_dtype=pl.Int8
                    )

        return cached_event_ids if load_ids else None, cached_labels if load_labels else None

    def _load_labels(self) -> tuple[Mapping[int, list], Mapping[int, list]]:
        """Loads valid event ids and labels for each shard.

        Returns:
            A tuple containing two mappings: one from shard indices to lists of valid event IDs
            which is used for indexing rows in the sparse matrix, and another from shard indices
            to lists of corresponding labels.
        """
        _, cached_labels = self._load_ids_and_labels(load_ids=False)

        return cached_labels

    @TimeableMixin.TimeAs
    def _load_event_ids(self) -> tuple[Mapping[int, list], Mapping[int, list]]:
        """Loads valid event ids and labels for each shard.

        Returns:
            A tuple containing two mappings: one from shard indices to lists of valid event IDs
            which is used for indexing rows in the sparse matrix, and another from shard indices
            to lists of corresponding labels.
        """
        cached_event_ids, _ = self._load_ids_and_labels(load_labels=False)

        return cached_event_ids

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

        if (
            hasattr(self.cfg.tabularization, "max_by_correlation")
            and self.cfg.tabularization.max_by_correlation
        ):
            corrs = self._get_approximate_correlation_per_feature(*self._get_shard_by_index(0))
            corrs = np.abs(corrs)
            sorted_corrs = np.argsort(corrs)[::-1]

            codes_set = codes_set.intersection(
                set(sorted_corrs[: self.cfg.tabularization.max_by_correlation])
            )
        if hasattr(self.cfg.tabularization, "min_correlation") and self.cfg.tabularization.min_correlation:
            corrs = self._get_approximate_correlation_per_feature(*self._get_shard_by_index(0))
            corrs = np.abs(corrs)
            codes_set = codes_set.intersection(
                set(np.where(corrs > self.cfg.tabularization.min_correlation)[0])
            )

        return (
            codes_set,
            self._get_code_masks(feature_columns, codes_set),
            len(feature_columns),
        )

    def _get_approximate_correlation_per_feature(self, X: sp.csc_matrix, y: np.ndarray) -> np.ndarray:
        """Calculates the approximate correlation of each feature with the target.

        Args:
            X: The feature data.
            y: The target labels.

        Returns:
            The approximate correlation of each feature with the target.
        """
        # calculate the pearson r correlation of each feature with the target
        # this is a very rough approximation and should be used for feature selection
        # and not as a definitive measure of feature importance

        # check that y has information
        if len(np.unique(y)) == 1:
            raise ValueError("Labels have only one unique value. Cannot calculate correlation.")

        corrs = np.apply_along_axis(lambda col: pearsonr(col.flatten(), y)[0], 0, X.toarray())
        return corrs

    def _set_imputer(self):
        """Sets the imputer for the data."""
        if (
            hasattr(self.cfg.data_loading_params, "imputer")
            and self.cfg.data_loading_params.imputer.imputer_target
        ):
            imputer = self.cfg.data_loading_params.imputer.imputer_target
            if hasattr(imputer, "partial_fit"):
                for i in range(len(self._data_shards)):
                    X, _ = self._get_shard_by_index(i)
                    imputer.partial_fit(X)
            elif hasattr(imputer, "fit"):
                imputer.fit(self._get_shard_by_index(0)[0])
            else:
                raise ValueError("Imputer must have a fit or partial_fit method.")
            self.imputer = imputer
        else:
            self.imputer = None

    def _set_scaler(self):
        """Sets the scaler for the data."""
        if (
            hasattr(self.cfg.data_loading_params, "normalization")
            and self.cfg.data_loading_params.normalization.normalizer
        ):
            scaler = self.cfg.data_loading_params.normalization.normalizer
            if hasattr(scaler, "partial_fit"):
                for i in range(len(self._data_shards)):
                    X, _ = self._get_shard_by_index(i)
                    scaler.partial_fit(X)
            elif hasattr(scaler, "fit"):
                X = self._get_shard_by_index(0)[0]
                scaler.fit(X)
            else:
                raise ValueError("Scaler must have a fit or partial_fit method.")
            self.scaler = scaler
        else:
            self.scaler = None

    def _impute_and_scale_data(self, data: sp.csc_matrix) -> sp.csc_matrix:
        """Scales the data using the fitted scaler.

        Args:
            data: The data to scale.

        Returns:
            The scaled data.
        """
        if self.imputer is not None:
            data = self.imputer.transform(data)
        if self.scaler is not None:
            return self.scaler.transform(data)
        return data

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
            # find missing files
            missing_files = [file for file in files if not file.exists()]
            raise ValueError(
                f"Not all files exist for shard {self._data_shards[idx]}. Missing: {missing_files}"
            )

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
        if self.labels is None:
            self.labels = self._load_labels()
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
        if not hasattr(self, "codes_set") or self.codes_set is None:
            return df

        ckey = f"_filter_shard_on_codes_and_freqs/{agg}"
        self._register_start(key=ckey)

        df = df[:, self.code_masks[agg]]

        self._register_end(key=ckey)

        return df

    def get_data_shards(self, idx: int | list[int]) -> tuple[sp.csc_matrix, np.ndarray]:
        """Retrieves the feature data and labels for specific shards.

        Args:
            idx: Index of the shard to retrieve or list of indices.

        Returns:
            A tuple where the first element is a sparse matrix containing the
            feature data, and the second element is a numpy array containing the labels.
        """
        X = []
        y = []
        if isinstance(idx, int):
            idx = [idx]
        for i in idx:
            X_, y_ = self._get_shard_by_index(i)
            X.append(self._impute_and_scale_data(X_))
            y.append(y_)
        if len(X) == 0 or len(y) == 0:
            raise ValueError("No data found in the shards or labels. Please check input files.")
        X = sp.vstack(X)
        y = np.concatenate(y, axis=0)

        return X, y

    def get_data(self) -> tuple[sp.csc_matrix, np.ndarray]:
        """Retrieves the feature data and labels for the current split.

        Returns:
            A tuple where the first element is a sparse matrix containing the
            feature data, and the second element is a numpy array containing the labels.
        """
        return self.get_data_shards(range(len(self._data_shards)))

    def get_data_shard_count(self) -> int:
        """Retrieves the number of data shards.

        Returns:
            The number of data shards.
        """
        return len(self._data_shards)

    def get_classes(self) -> int:
        """Retrieves the unique labels in the data.

        Returns:
            The unique labels.
        """
        # get all labels in a list
        all_labels = []
        for label in self.labels.values():
            all_labels.extend(label)

        return np.unique(all_labels)

    def get_all_column_names(self) -> list[str]:
        """Retrieves the names of all columns in the data.

        Returns:
            The names of all columns.
        """
        files = get_model_files(self.cfg, self.split, self._data_shards[0])

        def extract_name(test_file):
            return str(Path(test_file.parent.parent.stem, test_file.parent.stem, test_file.stem))

        agg_wind_combos = [extract_name(test_file) for test_file in files]

        feature_columns = get_feature_columns(self.cfg.tabularization.filtered_code_metadata_fp)
        all_feats = []
        for agg_wind in agg_wind_combos:
            window, feat, agg = agg_wind.split("/")
            feature_ids = get_feature_indices(feat + "/" + agg, feature_columns)
            feature_names = [feature_columns[i] for i in feature_ids]
            for feat_name in feature_names:
                all_feats.append(f"{feat_name}/{agg}/{window}")

        return all_feats

    def get_column_names(self, indices: list[int] = None) -> list[str]:
        """Retrieves the names of the columns in the data.

        Returns:
            The names of the columns.
        """
        files = get_model_files(self.cfg, self.split, self._data_shards[0])

        def extract_name(test_file):
            return str(Path(test_file.parent.parent.stem, test_file.parent.stem, test_file.stem))

        agg_wind_combos = [extract_name(test_file) for test_file in files]

        feature_columns = get_feature_columns(self.cfg.tabularization.filtered_code_metadata_fp)
        all_feats = []
        for agg_wind in agg_wind_combos:
            window, feat, agg = agg_wind.split("/")
            feature_ids = get_feature_indices(feat + "/" + agg, feature_columns)
            feature_names = [feature_columns[i] for i in feature_ids]
            for feat_name in feature_names:
                all_feats.append(f"{feat_name}/{agg}/{window}")

        # filter by only those in the list of indices
        if indices is not None:
            all_feats = [all_feats[i] for i in indices]
        return all_feats

    def densify(self) -> np.ndarray:
        """Builds the data as a dense matrix based on column subselection."""

        # get the dense matrix by iterating through the data shards
        data = []
        labels = []
        for shard_idx in range(len(self._data_shards)):
            shard_data, shard_labels = self.get_data_shards(shard_idx)
            data.append(shard_data)
            labels.append(shard_labels)
        data = sp.vstack(data)
        labels = np.concatenate(labels, axis=0)
        return data, labels
