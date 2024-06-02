"""The base class for core dataset processing logic.

Attributes:
    INPUT_DF_T: This defines the type of the allowable input dataframes -- e.g., databases, filepaths,
        dataframes, etc.
    DF_T: This defines the type of internal dataframes -- e.g. polars DataFrames.
"""
import json
import os
from collections.abc import Mapping
from pathlib import Path

import hydra
import numpy as np
import polars as pl
import polars.selectors as cs
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from scipy.sparse import coo_array

DF_T = pl.LazyFrame
WRITE_USE_PYARROW = True
ROW_IDX_NAME = "__row_idx"


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


def sparse_matrix_to_array(coo_matrix: coo_array):
    return np.array([coo_matrix.data, coo_matrix.row, coo_matrix.col]), coo_matrix.shape


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


def compute_feature_frequencies(cfg: DictConfig, shard_df: DF_T) -> list[str]:
    """Generates a list of feature column names from the data within each shard based on specified
    configurations.

    Parameters:
    - cfg (DictConfig): Configuration dictionary specifying how features should be evaluated and aggregated.
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
    static_df = shard_df.filter(
        pl.col("patient_id").is_not_null() & pl.col("code").is_not_null() & pl.col("timestamp").is_null()
    )
    static_code_freqs_df = static_df.groupby("code").agg(pl.count("code").alias("count")).collect()
    static_code_freqs = {
        row["code"] + "/static/present": row["count"] for row in static_code_freqs_df.iter_rows(named=True)
    }

    static_value_df = static_df.filter(pl.col("numerical_value").is_not_null())
    static_value_freqs_df = (
        static_value_df.groupby("code").agg(pl.count("numerical_value").alias("count")).collect()
    )
    static_value_freqs = {
        row["code"] + "/static/first": row["count"] for row in static_value_freqs_df.iter_rows(named=True)
    }

    ts_df = shard_df.filter(
        pl.col("patient_id").is_not_null() & pl.col("code").is_not_null() & pl.col("timestamp").is_not_null()
    )
    code_freqs_df = ts_df.groupby("code").agg(pl.count("code").alias("count")).collect()
    code_freqs = {row["code"] + "/code": row["count"] for row in code_freqs_df.iter_rows(named=True)}

    value_df = ts_df.filter(pl.col("numerical_value").is_not_null())
    value_freqs_df = value_df.groupby("code").agg(pl.count("numerical_value").alias("count")).collect()
    value_freqs = {row["code"] + "/value": row["count"] for row in value_freqs_df.iter_rows(named=True)}

    combined_freqs = {**static_code_freqs, **static_value_freqs, **code_freqs, **value_freqs}
    return combined_freqs


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


def setup_environment(cfg: DictConfig, load_data: bool = True):
    # check output dir
    flat_dir = Path(cfg.tabularized_data_dir)
    assert flat_dir.exists()

    # load MEDS data
    split_to_df = load_meds_data(cfg.MEDS_cohort_dir, load_data)
    feature_columns = json.load(open(flat_dir / "feature_columns.json"))

    # Check that the stored config matches the current config
    with open(flat_dir / "config.yaml") as file:
        yaml_config = yaml.safe_load(file)
        stored_config = OmegaConf.create(yaml_config)
    logger.info(f"Stored config: {stored_config}")
    logger.info(f"Worker config: {cfg}")
    assert cfg.keys() == stored_config.keys(), "Keys in stored config do not match current config."
    for key in cfg.keys():
        assert key in stored_config, f"Key {key} not found in stored config."
        if key == "worker":
            continue
        assert (
            cfg[key] == stored_config[key]
        ), f"Config key {key}, value is {cfg[key]} vs {stored_config[key]}"
    return flat_dir, split_to_df, feature_columns
