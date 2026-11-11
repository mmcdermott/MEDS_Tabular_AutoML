import hashlib
import json
import logging
import os
import shutil
import time
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

logger = logging.getLogger(__name__)


def select_csc_columns(X: sp.csc_matrix, cols: np.ndarray) -> sp.csc_matrix:
    """Selects columns of a CSC matrix by integer index, directly from its arrays.

    Gives the same matrix as ``X[:, cols]`` (same column order, same entry order within each column) without
    scipy's generic fancy-indexing, whose index validation dominates when many blocks are filtered.

    Args:
        X: The matrix.
        cols: Integer column indices.

    Returns:
        The selected columns, as a CSC matrix.

    Examples:
        >>> X = sp.random(6, 8, density=0.4, format="csc", random_state=0)
        >>> cols = np.array([0, 3, 4, 7])
        >>> Y = select_csc_columns(X, cols)
        >>> ref = X[:, cols]
        >>> Y.shape == ref.shape and (Y != ref).nnz == 0
        True
        >>> bool(np.array_equal(Y.indptr, ref.indptr) and np.array_equal(Y.indices, ref.indices))
        True
        >>> select_csc_columns(X, np.array([], dtype=int)).shape
        (6, 0)
    """
    cols = np.asarray(cols, dtype=np.int64)
    indptr = X.indptr
    starts = indptr[cols]
    lens = indptr[cols + 1] - starts
    new_indptr = np.zeros(len(cols) + 1, dtype=indptr.dtype)
    np.cumsum(lens, out=new_indptr[1:])
    pos = np.repeat(starts - new_indptr[:-1], lens) + np.arange(new_indptr[-1], dtype=np.int64)
    return sp.csc_matrix(
        (X.data[pos], X.indices[pos], new_indptr), shape=(X.shape[0], len(cols)), dtype=X.dtype
    )


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
        self.cfg = cfg
        self.split = split
        # Load shards for this split
        self._data_shards = sorted(
            [shard.stem for shard in list_subdir_files(Path(cfg.path.input_label_dir) / split, "parquet")]
        )
        if len(self._data_shards) == 0:
            raise ValueError(
                "No labels found in the `input_label_dir` " + str(Path(cfg.path.input_label_dir).resolve())
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
            shard: (Path(self.cfg.path.input_label_dir) / self.split / shard).with_suffix(".parquet")
            for shard in self._data_shards
        }
        cached_labels, cached_event_ids = {}, {}
        for shard, label_fp in label_fps.items():
            label_df = pl.scan_parquet(label_fp)
            if "event_id" not in label_df.collect_schema():
                label_df = label_df.with_row_index("event_id")

            if load_ids:
                cached_event_ids[shard] = label_df.select(pl.col("event_id")).collect().to_series()

            if load_labels:
                label_col = "boolean_value" if "boolean_value" in label_df.collect_schema() else "label"
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
        is_static = path.stem in ["mean", "present"]
        agg = f"static/{path.stem}" if is_static else f"{path.parent.stem}/{path.stem}"

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
        # get all window_size x aggregation files using the file resolver
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

        cols = self._code_index(agg)
        if df.shape[1] != len(self.code_masks[agg]):
            raise ValueError(
                f"{agg}: the matrix has {df.shape[1]} columns but the code mask {len(self.code_masks[agg])}"
            )
        if len(cols) < df.shape[1]:
            df = select_csc_columns(sp.csc_matrix(df), cols)

        self._register_end(key=ckey)

        return df

    def _code_index(self, agg: str) -> np.ndarray:
        """Column indices kept by the code mask of `agg` (computed once per aggregation)."""
        if not hasattr(self, "_code_index_cache"):
            self._code_index_cache = {}
        if agg not in self._code_index_cache:
            self._code_index_cache[agg] = np.flatnonzero(np.asarray(self.code_masks[agg], dtype=bool))
        return self._code_index_cache[agg]

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

        With ``data_loading_params.cache_matrices`` (and no imputer or scaler), the split's matrix with every
        tabularized column is built once and cached under ``cache_dir``; each call then only applies its own
        code filter to the cache. The result is identical either way.

        Returns:
            A tuple where the first element is a sparse matrix containing the
            feature data, and the second element is a numpy array containing the labels.
        """
        if (
            self.cfg.data_loading_params.get("cache_matrices", False)
            and self.imputer is None
            and self.scaler is None
        ):
            return self._get_data_from_matrix_cache()
        return self.get_data_shards(range(len(self._data_shards)))

    def _matrix_cache_dir(self, files_by_shard: list[list[Path]]) -> Path:
        """Cache directory for this split's unfiltered matrix, keyed on its inputs (paths, sizes, mtimes)."""
        label_dir = Path(self.cfg.path.input_label_dir) / self.split
        inputs = [f for files in files_by_shard for f in files]
        inputs += [(label_dir / shard).with_suffix(".parquet") for shard in self._data_shards]
        key = {
            "version": 1,
            "split": self.split,
            "binarize_task": bool(self.cfg.data_loading_params.get("binarize_task", False)),
            "inputs": [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in inputs],
        }
        digest = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:20]
        return Path(self.cfg.path.cache_dir) / "matrices" / f"{self.split}-{digest}"

    def _build_matrix_cache(self, files_by_shard: list[list[Path]], out: Path) -> None:
        """Stacks every shard's blocks without code filtering and writes them as .npy arrays (atomically)."""
        shards, widths = [], None
        for files in files_by_shard:
            blocks = [self._load_matrix(f) for f in files]
            shard_widths = [b.shape[1] for b in blocks]
            if widths is None:
                widths = shard_widths
            elif shard_widths != widths:
                raise ValueError(
                    f"Shards of {self.split} have different block widths: {widths} vs {shard_widths}"
                )
            shards.append(sp.hstack(blocks, format="csc"))
        X = sp.vstack(shards, format="csc")
        tmp = out.parent / f".{out.name}.{os.getpid()}.{time.time_ns()}"
        tmp.mkdir(parents=True)
        np.save(tmp / "indptr.npy", X.indptr)
        np.save(tmp / "indices.npy", X.indices)
        np.save(tmp / "data.npy", X.data)
        (tmp / "meta.json").write_text(json.dumps({"shape": list(X.shape), "widths": widths}))
        try:
            tmp.rename(out)
        except OSError:  # another trial finished it first
            shutil.rmtree(tmp, ignore_errors=True)

    def _get_data_from_matrix_cache(self) -> tuple[sp.csc_matrix, np.ndarray]:
        """`get_data` through the matrix cache: the cached unfiltered matrix, filtered by this config's
        codes."""
        files_by_shard = [get_model_files(self.cfg, self.split, shard) for shard in self._data_shards]
        for files in files_by_shard:
            missing = [f for f in files if not f.exists()]
            if missing:
                raise ValueError(f"Not all files exist for split {self.split}. Missing: {missing}")
        out = self._matrix_cache_dir(files_by_shard)
        if not out.exists():
            logger.info(f"Building the matrix cache for {self.split} in {out}")
            self._build_matrix_cache(files_by_shard, out)
        meta = json.loads((out / "meta.json").read_text())
        # Columns this config keeps: block by block, as the uncached path filters and stacks them.
        cols, offset = [], 0
        for f, width in zip(files_by_shard[0], meta["widths"], strict=True):
            agg = f"static/{f.stem}" if f.stem in ["mean", "present"] else f"{f.parent.stem}/{f.stem}"
            if not hasattr(self, "codes_set") or self.codes_set is None:
                kept = np.arange(width)
            else:
                if width != len(self.code_masks[agg]):
                    raise ValueError(
                        f"{agg}: cached block has {width} columns, the code mask {len(self.code_masks[agg])}"
                    )
                kept = self._code_index(agg)
            cols.append(kept + offset)
            offset += width
        cols = np.concatenate(cols) if cols else np.array([], dtype=np.int64)
        keep_all = len(cols) == meta["shape"][1]  # every column kept (cols is then 0..n-1)
        mode = None if keep_all else "r"  # all: one sequential read; else map and gather
        X = sp.csc_matrix(
            (
                np.load(out / "data.npy", mmap_mode=mode),
                np.load(out / "indices.npy", mmap_mode=mode),
                np.load(out / "indptr.npy", mmap_mode=mode),
            ),
            shape=tuple(meta["shape"]),
        )
        if not keep_all:
            X = select_csc_columns(X, cols)
        if self.labels is None:
            self.labels = self._load_labels()
        y = np.concatenate([self.labels[shard] for shard in self._data_shards], axis=0)
        return X, y

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
        names_by_agg = {}  # the feature names of an aggregation are the same for every window
        all_feats = []
        for agg_wind in agg_wind_combos:
            window, feat, agg = agg_wind.split("/")
            key = feat + "/" + agg
            if key not in names_by_agg:
                names_by_agg[key] = [feature_columns[i] for i in get_feature_indices(key, feature_columns)]
            suffix = f"/{agg}/{window}"
            all_feats.extend(name + suffix for name in names_by_agg[key])

        return all_feats

    def get_column_names(self, indices: list[int] | None = None) -> list[str]:
        """Retrieves the names of the columns in the data.

        Returns:
            The names of the columns.
        """
        all_feats = self.get_all_column_names()
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
