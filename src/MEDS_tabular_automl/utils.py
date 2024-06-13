"""The base class for core dataset processing logic.

Attributes:
    INPUT_DF_T: This defines the type of the allowable input dataframes -- e.g., databases, filepaths,
        dataframes, etc.
    DF_T: This defines the type of internal dataframes -- e.g. polars DataFrames.
"""
import os
from collections.abc import Mapping
from pathlib import Path

import hydra
import numpy as np
import polars as pl
import polars.selectors as cs
from loguru import logger
from omegaconf import DictConfig
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
        >>> get_min_dtype(np.array([1, 2, 3]))
        dtype('uint8')
        >>> get_min_dtype(np.array([1, 2, 3, int(1e9)]))
        dtype('uint32')
        >>> get_min_dtype(np.array([1, 2, 3, int(1e18)]))
        dtype('uint64')
        >>> get_min_dtype(np.array([1, 2, 3, -128]))
        dtype('int8')
        >>> get_min_dtype(np.array([1.0, 2.0, 3.0]))
        dtype('float32')
        >>> get_min_dtype(np.array([1, 2, 3, np.nan]))
        dtype('float32')
        >>> get_min_dtype(np.array([1, 2, 3, "a"]))
        dtype('<U21')
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
        ...     write_df(df_coo_array, fp, do_overwrite=True)
        ...     assert load_matrix(fp).toarray().tolist() == [[1], [2], [3]]
        ...     write_df(df_coo_array, fp, do_overwrite=False)
        Traceback (most recent call last):
            ...
        FileExistsError: test.parquet exists and do_overwrite is False!
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


def get_static_col_dtype(col: str) -> pl.DataType:
    """Determines the appropriate minimal data type for given flat representation column string based on its
    aggregation type.

    Args:
        col (str): The column name in the format 'category/type/aggregation'.

    Returns:
        pl.DataType: The appropriate Polars data type for the column.

    Raises:
        ValueError: If the column name format or aggregation type is not recognized.
    """
    code, code_type, agg = parse_static_feature_column(col)

    match agg:
        case "sum" | "sum_sqd" | "min" | "max" | "value" | "first":
            return pl.Float32
        case "present":
            return pl.Boolean
        case "count" | "has_values_count":
            return pl.UInt32
        case _:
            raise ValueError(f"Column name {col} malformed!")


def add_static_missing_cols(
    flat_df: pl.LazyFrame, feature_columns: list[str], set_count_0_to_null: bool = False
) -> pl.LazyFrame:
    """Normalizes columns in a LazyFrame so all expected columns are present and appropriately typed and
    potentially modifies zero counts to nulls based on the configuration.

    Args:
        flat_df: The LazyFrame to normalize.
        feature_columns: A list of expected column names.
        set_count_0_to_null: A flag of whether to convert zero counts to nulls.

    Returns:
        The normalized LazyFrame with all specified columns present and correctly typed and with
        zero-counts handled if specified.
    """
    cols_to_add = set(feature_columns) - set(flat_df.columns)
    cols_to_retype = set(feature_columns).intersection(set(flat_df.columns))

    cols_to_add = [(c, get_static_col_dtype(c)) for c in cols_to_add]
    cols_to_retype = [(c, get_static_col_dtype(c)) for c in cols_to_retype]

    if "timestamp" in flat_df.columns:
        key_cols = ["patient_id", "timestamp"]
    else:
        key_cols = ["patient_id"]

    flat_df = flat_df.with_columns(
        *[pl.lit(None, dtype=dt).alias(c) for c, dt in cols_to_add],
        *[pl.col(c).cast(dt).alias(c) for c, dt in cols_to_retype],
    ).select(*key_cols, *feature_columns)

    if not set_count_0_to_null:
        return flat_df

    flat_df = flat_df.collect()

    flat_df = flat_df.with_columns(
        pl.when(cs.ends_with("count") != 0).then(cs.ends_with("count")).keep_name()
    ).lazy()
    return flat_df


def get_static_feature_cols(shard_df: pl.LazyFrame) -> list[str]:
    """Generates a list of static feature column names based on data within a shard.

    This function evaluates the properties of codes within training data and applies configured
    aggregations to generate a comprehensive list of the static feature columns for modeling purposes.

    Args:
        shard_df: The LazyFrame shard to analyze.

    Returns:
        A list of column names representing static features.

    Examples:
        >>> import polars as pl
        >>> data = {'code': ['A', 'A', 'B', 'B', 'C', 'C', 'C'],
        ...         'timestamp': [
        ...             None, '2021-01-01', '2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', None
        ...         ],
        ...         'numerical_value': [1, None, 2, 2, None, None, 3]}
        >>> df = pl.DataFrame(data).lazy()
        >>> get_static_feature_cols(df)
        ['A/static/first', 'A/static/present', 'C/static/first', 'C/static/present']
    """
    feature_columns = []
    static_df = shard_df.filter(pl.col("timestamp").is_null())
    for code in static_df.select(pl.col("code").unique()).collect().to_series():
        static_aggregations = [f"{code}/static/present", f"{code}/static/first"]
        feature_columns.extend(static_aggregations)
    return sorted(feature_columns)


def get_ts_feature_cols(shard_df: pl.LazyFrame) -> list[str]:
    """Generates a list of time-series feature column names based on data within a shard.

    This function evaluates the properties of codes within training data and applies configured
    aggregations to generate a comprehensive list of the time-series feature columns for modeling
    purposes.

    Args:
        shard_df: The LazyFrame shard to analyze.

    Returns:
        A list of column names representing time-series features.
    """
    ts_df = shard_df.filter(pl.col("timestamp").is_not_null())
    feature_columns = list(ts_df.select(pl.col("code").unique()).collect().to_series())
    feature_columns = [f"{code}/code" for code in feature_columns] + [
        f"{code}/value" for code in feature_columns
    ]
    return sorted(feature_columns)


def get_prediction_ts_cols(
    aggregations: list[str], ts_feature_cols: pl.LazyFrame, window_sizes: list[str] | None = None
) -> list[str]:
    """Generates a list of feature column names for prediction tasks based on aggregations and window sizes.

    Args:
        aggregations: The list of aggregation methods to apply.
        ts_feature_cols: The list of existing time-series feature columns.
        window_sizes: The optional list of window sizes to consider.

    Returns:
        A list of feature column names formatted with aggregation and window size.
    """
    agg_feature_columns = []
    for code in ts_feature_cols:
        ts_aggregations = [f"{code}/{agg}" for agg in aggregations]
        agg_feature_columns.extend(ts_aggregations)
    if window_sizes:
        ts_aggregations = [f"{window_size}/{code}" for window_size in window_sizes]
    return sorted(ts_aggregations)


def get_flat_rep_feature_cols(cfg: DictConfig, shard_df: pl.LazyFrame) -> list[str]:
    """Combines static and time-series feature columns from a shard based on specified configurations.

    This function evaluates the properties of codes within training data and applies configured
    aggregations to generate a comprehensive list of all feature columns for modeling purposes.

    Args:
        cfg: The configuration dictionary specifying aggregation settings.
        shard_df: The LazyFrame shard in MEDS format to process.

    Returns:
        A combined list of all feature columns from both static and time-series data.
    """
    static_feature_columns = get_static_feature_cols(shard_df)
    ts_feature_columns = get_ts_feature_cols(cfg.aggs, shard_df)
    return static_feature_columns + ts_feature_columns


def load_meds_data(MEDS_cohort_dir: str, load_data: bool = True) -> Mapping[str, pl.LazyFrame]:
    """Loads the MEDS dataset from disk, structured by data splits.

    Args:
        MEDS_cohort_dir: The directory containing the MEDS datasets split by subfolders.
            We expect `train` to be a split so `MEDS_cohort_dir/train` should exist.
        load_data: If True, returns LazyFrames for each data split, otherwise returns file paths.

    Returns:
        A dictionary mapping from split name to a LazyFrame, containing the MEDS dataset for each split.

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> MEDS_cohort_dir = Path(tempfile.mkdtemp())
        >>> for split in ["train", "val", "test"]:
        ...     split_dir = MEDS_cohort_dir / split
        ...     split_dir.mkdir()
        ...     pl.DataFrame({"patient_id": [1, 2, 3]}).write_parquet(split_dir / "data.parquet")
        >>> split_to_df = load_meds_data(MEDS_cohort_dir)
        >>> assert "train" in split_to_df
        >>> assert len(split_to_df) == 3
        >>> assert len(split_to_df["train"]) == 1
        >>> assert isinstance(split_to_df["train"][0], pl.LazyFrame)
    """
    MEDS_cohort_dir = Path(MEDS_cohort_dir)
    meds_fps = list(MEDS_cohort_dir.glob("*/*.parquet"))
    splits = {fp.parent.stem for fp in meds_fps}
    split_to_fps = {split: [fp for fp in meds_fps if fp.parent.stem == split] for split in splits}
    if not load_data:
        return split_to_fps
    split_to_df = {
        split: [pl.scan_parquet(fp) for fp in split_fps] for split, split_fps in split_to_fps.items()
    }
    return split_to_df


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
