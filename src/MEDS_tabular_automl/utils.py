"""The base class for core dataset processing logic.

Attributes:
    INPUT_DF_T: This defines the type of the allowable input dataframes -- e.g., databases, filepaths,
        dataframes, etc.
    DF_T: This defines the type of internal dataframes -- e.g. polars DataFrames.
"""
import os
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from loguru import logger
from omegaconf import OmegaConf
from scipy.sparse import coo_array

DF_T = pl.LazyFrame
WRITE_USE_PYARROW = True
ROW_IDX_NAME = "__row_idx"

STATIC_CODE_AGGREGATION = "static/present"
STATIC_VALUE_AGGREGATION = "static/first"

CODE_AGGREGATIONS = [
    "code/count",
]

VALUE_AGGREGATIONS = [
    "value/count",
    "value/has_values_count",
    "value/sum",
    "value/sum_sqd",
    "value/min",
    "value/max",
]


def hydra_loguru_init() -> None:
    """Adds loguru output to the logs that hydra scrapes.

    Must be called from a hydra main!
    """
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.add(os.path.join(hydra_path, "main.log"))


def filter_to_codes(
    allowed_codes: list[str] | None,
    min_code_inclusion_frequency: int,
    code_metadata_fp: Path,
) -> list[str]:
    """Filters and returns codes based on allowed list and minimum frequency.

    Args:
        allowed_codes: List of allowed codes, None means all codes are allowed.
        min_code_inclusion_frequency: Minimum frequency a code must have to be included.
        code_metadata_fp: Path to the metadata file containing code information.

    Returns:
        Sorted list of the intersection of allowed codes (if they are specified) and filters based on
        inclusion frequency.

    Examples:
        >>> from tempfile import NamedTemporaryFile
        >>> with NamedTemporaryFile() as f:
        ...     pl.DataFrame({"code": ["E", "D", "A"], "count": [4, 3, 2]}).write_parquet(f.name)
        ...     filter_to_codes(["A", "D"], 3, f.name)
        ['D']
    """

    feature_freqs = pl.read_parquet(code_metadata_fp)

    if allowed_codes is not None:
        feature_freqs = feature_freqs.filter(pl.col("code").is_in(allowed_codes))

    feature_freqs = feature_freqs.filter(pl.col("count") >= min_code_inclusion_frequency)
    return sorted(feature_freqs["code"].to_list())


OmegaConf.register_new_resolver("filter_to_codes", filter_to_codes, replace=True)


def load_tqdm(use_tqdm: bool):
    """Conditionally loads and returns tqdm progress bar handler or a no-operation function.

    Args:
        use_tqdm: Flag indicating whether to use tqdm progress bar.

    Returns:
        A function that either encapsulates tqdm or simply returns the input it is given.
    """
    if use_tqdm:
        from tqdm import tqdm

        return tqdm
    else:

        def noop(x, **kwargs):
            return x

        return noop


def parse_static_feature_column(c: str) -> tuple[str, str, str, str]:
    """Parses a flat feature column format into component parts.

    Args:
        c: The column string in 'category/subcategory/feature' format.

    Returns:
        A tuple containing separate strings of the feature column format.

    Raises:
        ValueError: If the column string format is incorrect.

    Examples:
        >>> parse_static_feature_column("A/static/present")
        ('A', 'static', 'present')
        >>> parse_static_feature_column("A/B/static/first")
        ('A/B', 'static', 'first')
        >>> parse_static_feature_column("static/first")
        Traceback (most recent call last):
            ...
        ValueError: Column static/first is not a valid flat feature column!
    """
    parts = c.split("/")
    if len(parts) < 3:
        raise ValueError(f"Column {c} is not a valid flat feature column!")
    return ("/".join(parts[:-2]), parts[-2], parts[-1])


def array_to_sparse_matrix(array: np.ndarray, shape: tuple[int, int]) -> coo_array:
    """Converts a numpy array representation into a sparse matrix.

    Args:
        array: The array containing data, rows, and columns.
        shape: The shape of the resulting sparse matrix.

    Returns:
        The formatted sparse matrix.

    Raises:
        AssertionError: If the input array's first dimension is not 3.
    """
    assert array.shape[0] == 3
    data, row, col = array
    return coo_array((data, (row, col)), shape=shape)


def get_min_dtype(array: np.ndarray) -> np.dtype:
    """Get the minimal dtype that can represent the array.

    Args:
        array: The array to determine the minimal dtype for.

    Returns:
        The minimal dtype that can represent the array, or the array's dtype if it is non-numeric.

    Examples:
        >>> get_min_dtype(np.array([1, 2, 3])) # doctest:+ELLIPSIS
        dtype('...')
        >>> get_min_dtype(np.array([1, 2, 3, int(1e9)])) # doctest:+ELLIPSIS
        dtype('...')
        >>> get_min_dtype(np.array([1, 2, 3, int(1e18)])) # doctest:+ELLIPSIS
        dtype('...')
        >>> get_min_dtype(np.array([1, 2, 3, -128])) # doctest:+ELLIPSIS
        dtype('...')
        >>> get_min_dtype(np.array([1.0, 2.0, 3.0])) # doctest:+ELLIPSIS
        dtype('...')
        >>> get_min_dtype(np.array([1, 2, 3, np.nan])) # doctest:+ELLIPSIS
        dtype('...')
        >>> get_min_dtype(np.array([1, 2, 3, "a"])) # doctest:+ELLIPSIS
        dtype('...')
    """
    if np.issubdtype(array.dtype, np.integer):
        return np.result_type(np.min_scalar_type(array.min()), array.max())
    elif np.issubdtype(array.dtype, np.floating):
        return np.result_type(np.float32)
        # For more precision, we could do this
        # try:
        #    array.astype(np.float32, copy=False)
        #    return np.float32
        # except OverflowError:
        #    return np.float64

    return array.dtype


def sparse_matrix_to_array(coo_matrix: coo_array) -> tuple[np.ndarray, tuple[int, int]]:
    """Converts a sparse matrix to a numpy array format with shape information.

    Args:
        coo_matrix: The sparse matrix to convert.

    Returns:
        A tuple of a numpy array ([data, row, col]) and the shape of the original matrix.
    """
    data, row, col = coo_matrix.data, coo_matrix.row, coo_matrix.col
    # Remove invalid indices
    valid_indices = (data == 0) | np.isnan(data)
    data = data[~valid_indices]
    row = row[~valid_indices]
    col = col[~valid_indices]
    # reduce dtypes
    if len(data):
        data = data.astype(get_min_dtype(data), copy=False)
        row = row.astype(get_min_dtype(row), copy=False)
        col = col.astype(get_min_dtype(col), copy=False)

    return np.array([data, row, col]), coo_matrix.shape


def store_matrix(coo_matrix: coo_array, fp_path: Path) -> None:
    """Stores a sparse matrix to disk as a .npz file.

    Args:
        coo_matrix: The sparse matrix to store.
        fp_path: The file path where the matrix will be stored.
    """
    array, shape = sparse_matrix_to_array(coo_matrix)
    np.savez(fp_path, array=array, shape=shape)


def load_matrix(fp_path: Path) -> coo_array:
    """Loads a sparse matrix from a .npz file.

    Args:
        fp_path: The path to the .npz file containing the sparse matrix data.

    Returns:
        The loaded sparse matrix.
    """
    npzfile = np.load(fp_path)
    array, shape = npzfile["array"], npzfile["shape"]
    return array_to_sparse_matrix(array, shape)


def write_df(df: pl.LazyFrame | pl.DataFrame | coo_array, fp: Path, do_overwrite: bool = False) -> None:
    """Writes a sparse matrix to disk.

    Args:
        df: The sparse matrix to write.
        fp: The file path where to write the data.
        do_overwrite: A flag indicating whether to overwrite the file if it already exists.

    Raises:
        FileExistsError: If the file exists and 'do_overwrite' is not set to True.
        TypeError: If the type of 'df' is not supported for writing.

    Examples:
        >>> import tempfile
        >>> from polars.testing import assert_frame_equal
        >>> df_polars = pl.DataFrame({"a": [1, 2, 3]})
        >>> df_coo_array = coo_array(([1, 2, 3], ([0, 1, 2], [0, 0, 0])), shape=(3, 1))
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     fp = Path(tmpdir) / "test.parquet"
        ...     write_df(df_polars, fp)
        ...     assert fp.is_file()
        ...     assert_frame_equal(pl.read_parquet(fp), df_polars)
        ...     write_df(df_polars.lazy(), fp, do_overwrite=True)
        ...     assert_frame_equal(pl.read_parquet(fp), df_polars)
        ...     fp = Path(tmpdir) / "test.npz"
        ...     write_df(df_coo_array, fp, do_overwrite=True)
        ...     assert load_matrix(fp).toarray().tolist() == [[1], [2], [3]]
        ...     write_df(df_coo_array, fp, do_overwrite=False)
        Traceback (most recent call last):
            ...
        FileExistsError: ...test.npz exists and do_overwrite is False!
    """
    if fp.is_file() and not do_overwrite:
        raise FileExistsError(f"{fp} exists and do_overwrite is {do_overwrite}!")

    fp.parent.mkdir(exist_ok=True, parents=True)

    if isinstance(df, pl.LazyFrame):
        df.collect().write_parquet(fp, use_pyarrow=WRITE_USE_PYARROW)
    elif isinstance(df, pl.DataFrame):
        df.write_parquet(fp, use_pyarrow=WRITE_USE_PYARROW)
    elif isinstance(df, coo_array):
        store_matrix(df, fp)
    else:
        raise TypeError(f"Unsupported type for df: {type(df)}")


def get_events_df(shard_df: pl.LazyFrame, feature_columns) -> pl.LazyFrame:
    """Extracts and filters an Events LazyFrame with one row per observation (timestamps can be duplicated).

    Args:
        shard_df: The LazyFrame shard from which to extract events.
        feature_columns: The columns that define features used to filter the LazyFrame.

    Returns:
        A LazyFrame where each row corresponds to an event, filtered by feature columns.
    """
    # Filter out feature_columns that were not present in the training set
    raw_feature_columns = ["/".join(c.split("/")[:-1]) for c in feature_columns]
    shard_df = shard_df.filter(pl.col("code").is_in(raw_feature_columns))
    # Drop rows with missing timestamp or code to get events
    ts_shard_df = shard_df.drop_nulls(subset=["timestamp", "code"])
    return ts_shard_df


def get_unique_time_events_df(events_df: pl.LazyFrame) -> pl.LazyFrame:
    """Ensures all timestamps in the events LazyFrame are unique and sorted by patient_id and timestamp.

    Args:
        events_df: Events LazyFrame to process.

    Returns:
        A LazyFrame with unique timestamps, sorted by patient_id and timestamp.
    """
    assert events_df.select(pl.col("timestamp")).null_count().collect().item() == 0
    # Check events_df is sorted - so it aligns with the ts_matrix we generate later in the pipeline
    events_df = (
        events_df.drop_nulls("timestamp")
        .select(pl.col(["patient_id", "timestamp"]))
        .unique(maintain_order=True)
    )
    assert events_df.sort(by=["patient_id", "timestamp"]).collect().equals(events_df.collect())
    return events_df


def get_feature_names(agg: str, feature_columns: list[str]) -> str:
    """Extracts feature column names based on aggregation type from a list of column names.

    Args:
        agg: The aggregation type to filter by.
        feature_columns: The list of feature column names.

    Returns:
        The filtered list of feature column names based on the aggregation type.

    Raises:
        ValueError: If the aggregation type is unknown or unsupported.
    """
    if agg in [STATIC_CODE_AGGREGATION, STATIC_VALUE_AGGREGATION]:
        return [c for c in feature_columns if c.endswith(agg)]
    elif agg in CODE_AGGREGATIONS:
        return [c for c in feature_columns if c.endswith("/code")]
    elif agg in VALUE_AGGREGATIONS:
        return [c for c in feature_columns if c.endswith("/value")]
    else:
        raise ValueError(f"Unknown aggregation type {agg}")


def get_feature_indices(agg: str, feature_columns: list[str]) -> list[int]:
    """Generates a list of feature name indices based on the aggregation type.

    Args:
        agg: The aggregation type used to filter feature names.
        feature_columns: The list of all feature column names.

    Returns:
        Indices of the columns that match the aggregation type.
    """
    feature_to_index = {c: i for i, c in enumerate(feature_columns)}
    agg_features = get_feature_names(agg, feature_columns)
    return [feature_to_index[c] for c in agg_features]


def get_shard_prefix(base_path: Path, fp: Path) -> str:
    """Extracts the shard prefix from a file path by removing the raw_cohort_dir.

    Args:
        base_path: The base path to remove from the file path.
        fp: The full file path from which to extract the shard prefix.

    Returns:
        The shard prefix (the file path relative to the base path with the suffix removed).

    Examples:
        >>> get_shard_prefix(Path("/a/b/c"), Path("/a/b/c/d.parquet"))
        'd'
        >>> get_shard_prefix(Path("/a/b/c"), Path("/a/b/c/d/e.csv.gz"))
        'd/e'
    """

    relative_path = fp.relative_to(base_path)
    relative_parent = relative_path.parent
    file_name = relative_path.name.split(".")[0]

    return str(relative_parent / file_name)
