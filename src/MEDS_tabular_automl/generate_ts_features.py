import warnings

import numpy as np
import polars as pl
from loguru import logger
from scipy.sparse import csr_array

from MEDS_tabular_automl.utils import (
    CODE_AGGREGATIONS,
    VALUE_AGGREGATIONS,
    get_events_df,
    get_feature_names,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


def feature_name_to_code(feature_name: str) -> str:
    """Converts a feature name to a code name by removing the aggregation part.

    Args:
        feature_name: The full feature name, including aggregation.

    Returns:
        The code name without the aggregation part.

    Examples:
        >>> feature_name_to_code("A/code/count")
        'A/code'
        >>> feature_name_to_code("A/B/code/count")
        'A/B/code'
        >>> feature_name_to_code("invalid_name")
        ''
    """
    return "/".join(feature_name.split("/")[:-1])


def get_long_code_df(
    df: pl.LazyFrame, ts_columns: list[str]
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Pivots the codes data frame to a long format one-hot representation for time-series data.

    Args:
        df: The LazyFrame containing the code data.
        ts_columns: The list of time-series columns to include in the output.

    Returns:
        A tuple containing the data (1s for presence), and a tuple of row and column indices for
        the CSR sparse matrix.
    """
    column_to_int = {feature_name_to_code(col): i for i, col in enumerate(ts_columns)}
    rows = range(df.select(pl.len()).collect().item())
    cols = (
        df.with_columns(pl.col("code").cast(str).replace(column_to_int).cast(int).alias("code_index"))
        .select("code_index")
        .collect()
        .to_series()
        .to_numpy()
    )
    if not np.issubdtype(cols.dtype, np.number):
        raise ValueError("numeric_value must be a numerical type. Instead it has type: ", cols.dtype)
    data = np.ones(df.select(pl.len()).collect().item(), dtype=np.bool_)
    return data, (rows, cols)


def get_long_value_df(
    df: pl.LazyFrame, ts_columns: list[str]
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Pivots the numerical value data frame to a long format for time-series data.

    Args:
        df: The LazyFrame containing the numerical value data.
        ts_columns: The list of time-series columns that have numerical values.

    Returns:
        A tuple containing the data (numerical values), and a tuple of row and column indices for
        the CSR sparse matrix.
    """
    column_to_int = {feature_name_to_code(col): i for i, col in enumerate(ts_columns)}
    value_df = df.with_row_index("index").drop_nulls("numeric_value").filter(pl.col("code").is_in(ts_columns))
    rows = value_df.select(pl.col("index")).collect().to_series().to_numpy()
    cols = (
        value_df.with_columns(pl.col("code").cast(str).replace(column_to_int).cast(int).alias("value_index"))
        .select("value_index")
        .collect()
        .to_series()
        .to_numpy()
    )
    if not np.issubdtype(cols.dtype, np.number):
        raise ValueError("numeric_value must be a numerical type. Instead it has type: ", cols.dtype)

    data = value_df.select(pl.col("numeric_value")).collect().to_series().to_numpy()
    return data, (rows, cols)


def summarize_dynamic_measurements(
    agg: str,
    ts_columns: list[str],
    df: pl.LazyFrame,
) -> tuple[pl.DataFrame, csr_array]:
    """Summarizes dynamic measurements for feature columns that are marked as 'dynamic'.

    Args:
        agg: The aggregation method, either from CODE_AGGREGATIONS or VALUE_AGGREGATIONS.
        ts_columns: The list of time-series feature columns.
        df: The LazyFrame from which features will be extracted and summarized.

    Returns:
        A tuple containing a DataFrame with dynamic feature identifiers and a sparse matrix
        of aggregated values.
    """
    logger.info("Generating Sparse matrix for Time Series Features")
    id_cols = ["subject_id", "time"]

    # Confirm dataframe is sorted
    check_df = df.select(pl.col(id_cols))
    if not check_df.sort(by=id_cols).collect().equals(check_df.collect()):
        raise ValueError("data frame must be sorted by subject_id and time")

    # Generate sparse matrix
    if agg in CODE_AGGREGATIONS:
        code_df = df.drop(*(id_cols + ["numeric_value"]))
        data, (rows, cols) = get_long_code_df(code_df, ts_columns)
    elif agg in VALUE_AGGREGATIONS:
        value_df = df.drop(*id_cols)
        data, (rows, cols) = get_long_value_df(value_df, ts_columns)

    sp_matrix = csr_array(
        (data, (rows, cols)),
        shape=(df.select(pl.len()).collect().item(), len(ts_columns)),
    )
    return df.select(pl.col(id_cols)), sp_matrix


def get_flat_ts_rep(
    agg: str,
    feature_columns: list[str],
    shard_df: pl.LazyFrame,
) -> tuple[pl.DataFrame, csr_array]:
    """Produces a flat time-series representation from a given data frame, focusing on non-static features.

    Args:
        agg: The aggregation method to use for summarizing the data.
        feature_columns: The list of column identifiers for features involved in dynamic analysis.
        shard_df: The LazyFrame containing time-stamped data from which features will be extracted.

    Returns:
        A tuple containing a LazyFrame with consisting of the processed time series data, combining
        both code and value representations. and a sparse matrix of the flat time series data.
    """
    # Remove codes not in training set
    shard_df = get_events_df(shard_df, feature_columns)
    ts_columns = get_feature_names(agg, feature_columns)
    return summarize_dynamic_measurements(agg, ts_columns, shard_df)
