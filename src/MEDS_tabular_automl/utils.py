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
from omegaconf import DictConfig, OmegaConf
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


def load_tqdm(use_tqdm):
    if use_tqdm:
        from tqdm import tqdm

        return tqdm
    else:

        def noop(x, **kwargs):
            return x

        return noop


def parse_static_feature_column(c: str) -> tuple[str, str, str, str]:
    parts = c.split("/")
    if len(parts) < 3:
        raise ValueError(f"Column {c} is not a valid flat feature column!")
    return ("/".join(parts[:-2]), parts[-2], parts[-1])


def array_to_sparse_matrix(array: np.ndarray, shape: tuple[int, int]):
    assert array.shape[0] == 3
    data, row, col = array
    return coo_array((data, (row, col)), shape=shape)


def get_min_dtype(array):
    try:
        return np.result_type(np.min_scalar_type(array.min()), array.max())
    except:
        return array.dtype


def sparse_matrix_to_array(coo_matrix: coo_array):
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


def store_matrix(coo_matrix: coo_array, fp_path: Path):
    array, shape = sparse_matrix_to_array(coo_matrix)
    np.savez(fp_path, array=array, shape=shape)


def load_matrix(fp_path: Path):
    npzfile = np.load(fp_path)
    array, shape = npzfile["array"], npzfile["shape"]
    return array_to_sparse_matrix(array, shape)


def write_df(df: coo_array, fp: Path, **kwargs):
    """Write shard to disk."""
    do_overwrite = kwargs.get("do_overwrite", False)

    if not do_overwrite and fp.is_file():
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
    """Gets the appropriate minimal dtype for the given flat representation column string."""

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
    flat_df: DF_T, feature_columns: list[str], set_count_0_to_null: bool = False
) -> DF_T:
    """Normalizes columns in a DataFrame so all expected columns are present and appropriately typed.

    Parameters:
    - flat_df (DF_T): The DataFrame to be normalized.
    - feature_columns (list[str]): A list of feature column names that should exist in the DataFrame.
    - set_count_0_to_null (bool): A flag indicating whether counts of zero should be converted to nulls.

    Returns:
    - DF_T: The normalized DataFrame with all columns set to the correct type and zero-counts handled
        if specified.

    This function ensures that all necessary columns are added and typed correctly within
    a DataFrame, potentially modifying zero counts to nulls based on the configuration.
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


def get_static_feature_cols(shard_df) -> list[str]:
    """Generates a list of feature column names from the data within each shard based on specified
    configurations.

    Parameters:
    - cfg (dict): Configuration dictionary specifying how features should be evaluated and aggregated.
    - split_to_shard_df (dict): A dictionary of DataFrames, divided by data split (e.g., 'train', 'test').

    Returns:
    - tuple[list[str], dict]: A tuple containing a list of feature columns and a dictionary of code properties
        identified during the evaluation.

    This function evaluates the properties of codes within training data and applies configured
    aggregations to generate a comprehensive list of feature columns for modeling purposes.
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


def get_ts_feature_cols(shard_df: DF_T) -> list[str]:
    """Generates a list of feature column names from the data within each shard based on specified
    configurations.

    Parameters:
    - cfg (dict): Configuration dictionary specifying how features should be evaluated and aggregated.
    - split_to_shard_df (dict): A dictionary of DataFrames, divided by data split (e.g., 'train', 'test').

    Returns:
    - tuple[list[str], dict]: A tuple containing a list of feature columns and a dictionary of code properties
        identified during the evaluation.

    This function evaluates the properties of codes within training data and applies configured
    aggregations to generate a comprehensive list of feature columns for modeling purposes.
    Examples:
    >>> import polars as pl
    >>> data = {'code': ['A', 'A', 'B', 'B', 'C', 'C', 'C'],
    ...         'timestamp': [None, '2021-01-01', None, None, '2021-01-03', '2021-01-04', None],
    ...         'numerical_value': [1, None, 2, 2, None, None, 3]}
    >>> df = pl.DataFrame(data).lazy()
    >>> aggs = ['value/sum', 'code/count']
    >>> get_ts_feature_cols(aggs, df)
    ['A/code', 'A/value', 'C/code', 'C/value']
    """
    ts_df = shard_df.filter(pl.col("timestamp").is_not_null())
    feature_columns = list(ts_df.select(pl.col("code").unique()).collect().to_series())
    feature_columns = [f"{code}/code" for code in feature_columns] + [
        f"{code}/value" for code in feature_columns
    ]
    return sorted(feature_columns)


def get_prediction_ts_cols(
    aggregations: list[str], ts_feature_cols: DF_T, window_sizes: list[str] | None = None
) -> list[str]:
    """Generates a list of feature column names that will be used for downstream task
    Examples:
    >>> feature_cols = ['A/code', 'A/value', 'C/code', 'C/value']
    >>> window_sizes = None
    >>> aggs = ['value/sum', 'code/count']
    >>> get_prediction_ts_cols(aggs, feature_cols, window_sizes)
    error
    >>> window_sizes = ["1d"]
    >>> get_prediction_ts_cols(aggs, feature_cols, window_sizes)
    error
    """
    agg_feature_columns = []
    for code in ts_feature_cols:
        ts_aggregations = [f"{code}/{agg}" for agg in aggregations]
        agg_feature_columns.extend(ts_aggregations)
    if window_sizes:
        ts_aggregations = [f"{window_size}/{code}" for window_size in window_sizes]
    return sorted(ts_aggregations)


def get_flat_rep_feature_cols(cfg: DictConfig, shard_df: DF_T) -> list[str]:
    """Generates a list of feature column names from the data within each shard based on specified
    configurations.

    Parameters:
    - cfg (dict): Configuration dictionary specifying how features should be evaluated and aggregated.
    - shard_df (DF_T): MEDS format dataframe shard.

    Returns:
    - list[str]: list of all feature columns.

    This function evaluates the properties of codes within training data and applies configured
    aggregations to generate a comprehensive list of feature columns for modeling purposes.
    Example:
    >>> data = {'code': ['A', 'A', 'B', 'B'],
    ...         'timestamp': [None, '2021-01-01', None, None],
    ...         'numerical_value': [1, None, 2, 2]}
    >>> df = pl.DataFrame(data).lazy()
    >>> aggs = ['value/sum', 'code/count']
    >>> cfg = DictConfig({'aggs': aggs})
    >>> get_flat_rep_feature_cols(cfg, df) # doctest: +NORMALIZE_WHITESPACE
    ['A/static/first', 'A/static/present', 'B/static/first', 'B/static/present', 'A/code/count',
     'A/value/sum']
    """
    static_feature_columns = get_static_feature_cols(shard_df)
    ts_feature_columns = get_ts_feature_cols(cfg.aggs, shard_df)
    return static_feature_columns + ts_feature_columns


def load_meds_data(MEDS_cohort_dir: str, load_data: bool = True) -> Mapping[str, pl.DataFrame]:
    """Loads the MEDS dataset from disk.

    Args:
        MEDS_cohort_dir: The directory containing the MEDS datasets split by subfolders.
            We expect `train` to be a split so `MEDS_cohort_dir/train` should exist.

    Returns:
        Mapping[str, pl.DataFrame]: Mapping from split name to a polars DataFrame containing the MEDS dataset.

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


def get_events_df(shard_df: pl.DataFrame, feature_columns) -> pl.DataFrame:
    """Extracts Events DataFrame with one row per observation (timestamps can be duplicated)"""
    # Filter out feature_columns that were not present in the training set
    raw_feature_columns = ["/".join(c.split("/")[:-1]) for c in feature_columns]
    shard_df = shard_df.filter(pl.col("code").is_in(raw_feature_columns))
    # Drop rows with missing timestamp or code to get events
    ts_shard_df = shard_df.drop_nulls(subset=["timestamp", "code"])
    return ts_shard_df


def get_unique_time_events_df(events_df: pl.DataFrame):
    """Updates Events DataFrame to have unique timestamps and sorted by patient_id and timestamp."""
    assert events_df.select(pl.col("timestamp")).null_count().collect().item() == 0
    # Check events_df is sorted - so it aligns with the ts_matrix we generate later in the pipeline
    events_df = (
        events_df.drop_nulls("timestamp")
        .select(pl.col(["patient_id", "timestamp"]))
        .unique(maintain_order=True)
    )
    assert events_df.sort(by=["patient_id", "timestamp"]).collect().equals(events_df.collect())
    return events_df


def get_feature_names(agg, feature_columns) -> str:
    """Indices of columns in feature_columns list."""
    if agg in [STATIC_CODE_AGGREGATION, STATIC_VALUE_AGGREGATION]:
        return [c for c in feature_columns if c.endswith(agg)]
    elif agg in CODE_AGGREGATIONS:
        return [c for c in feature_columns if c.endswith("/code")]
    elif agg in VALUE_AGGREGATIONS:
        return [c for c in feature_columns if c.endswith("/value")]
    else:
        raise ValueError(f"Unknown aggregation type {agg}")


def get_feature_indices(agg, feature_columns) -> str:
    """Indices of columns in feature_columns list."""
    feature_to_index = {c: i for i, c in enumerate(feature_columns)}
    agg_features = get_feature_names(agg, feature_columns)
    return [feature_to_index[c] for c in agg_features]


def store_config_yaml(config_fp: Path, cfg: DictConfig):
    """Stores configuration parameters into a JSON file.

    This function writes a dictionary of parameters, which includes patient partitioning
    information and configuration details, to a specified JSON file.

    Args:
    - config_fp (Path): The file path for the JSON file where config should be stored.
    - cfg (DictConfig): A configuration object containing settings like the number of patients
      per sub-shard, minimum code inclusion frequency, and flags for updating or overwriting existing files.

    Behavior:
    - If config_fp exists and cfg.do_overwrite is False (without do_update being True), a
      FileExistsError is raised to prevent unintentional data loss.

    Raises:
    - ValueError: If there are discrepancies between old and new parameters during an update.
    - FileExistsError: If the file exists and overwriting is not allowed.

    Example:
    >>> cfg = DictConfig({
    ...     "n_patients_per_sub_shard": 100,
    ...     "min_code_inclusion_frequency": 5,
    ...     "do_overwrite": True,
    ... })
    >>> import tempfile
    >>> from pathlib import Path
    >>> with tempfile.NamedTemporaryFile() as temp_f:
    ...     config_fp = Path(temp_f.name)
    ...     store_config_yaml(config_fp, cfg)
    ...     assert config_fp.exists()
    ...     store_config_yaml(config_fp, cfg)
    ...     cfg.do_overwrite = False
    ...     try:
    ...         store_config_yaml(config_fp, cfg)
    ...     except FileExistsError as e:
    ...         print("FileExistsError Error Triggered")
    FileExistsError Error Triggered
    """
    OmegaConf.save(cfg, config_fp)


def get_shard_prefix(base_path: Path, fp: Path) -> str:
    """Extracts the shard prefix from a file path by removing the raw_cohort_dir.

    Args:
        base_path: The base path to remove.
        fp: The file path to extract the shard prefix from.

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
