import warnings

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from scipy.sparse import csr_array

from MEDS_tabular_automl.utils import (
    CODE_AGGREGATIONS,
    DF_T,
    VALUE_AGGREGATIONS,
    get_events_df,
    get_feature_names,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


def get_long_code_df(df, ts_columns):
    """Pivots the codes data frame to a long format one-hot rep for time series data."""
    column_to_int = {col: i for i, col in enumerate(ts_columns)}
    rows = range(df.select(pl.len()).collect().item())
    cols = (
        df.with_columns(
            pl.concat_str([pl.col("code"), pl.lit("/code")]).replace(column_to_int).alias("code_index")
        )
        .select("code_index")
        .collect()
        .to_series()
        .to_numpy()
    )
    data = np.ones(df.select(pl.len()).collect().item(), dtype=np.bool_)
    return data, (rows, cols)


def get_long_value_df(df, ts_columns):
    """Pivots the numerical value data frame to a long format for time series data."""
    column_to_int = {col: i for i, col in enumerate(ts_columns)}
    value_df = df.with_row_index("index").drop_nulls("numerical_value")
    rows = value_df.select(pl.col("index")).collect().to_series().to_numpy()
    cols = (
        value_df.with_columns(
            pl.concat_str([pl.col("code"), pl.lit("/value")]).replace(column_to_int).alias("value_index")
        )
        .select("value_index")
        .collect()
        .to_series()
        .to_numpy()
    )
    data = value_df.select(pl.col("numerical_value")).collect().to_series().to_numpy()
    return data, (rows, cols)


def summarize_dynamic_measurements(
    agg: str,
    ts_columns: list[str],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize dynamic measurements for feature columns that are marked as 'dynamic'.

    Args:
    - ts_columns (list[str]): List of feature column identifiers that are specifically marked for dynamic
        analysis.
    - shard_df (DF_T): Data frame from which features will be extracted and summarized.

    Returns:
    - pl.LazyFrame: A summarized data frame containing the dynamic features.

    Example:
    >>> data = {'patient_id': [1, 1, 1, 2],
    ...         'code': ['A', 'A', 'B', 'B'],
    ...         'timestamp': ['2021-01-01', '2021-01-01', '2020-01-01', '2021-01-04'],
    ...         'numerical_value': [1, 2, 2, 2]}
    >>> df = pd.DataFrame(data)
    >>> ts_columns = ['A', 'B']
    >>> long_df = summarize_dynamic_measurements(ts_columns, df)
    >>> long_df.head()
       patient_id   timestamp  A/value  B/value  A/code  B/code
    0           1  2021-01-01        1        0       1       0
    1           1  2021-01-01        2        0       1       0
    2           1  2020-01-01        0        2       0       1
    3           2  2021-01-04        0        2       0       1
    >>> long_df.shape
    (4, 6)
    >>> long_df = summarize_dynamic_measurements(ts_columns, df[df.code == "A"])
    >>> long_df
       patient_id   timestamp  A/value  B/value  A/code  B/code
    0           1  2021-01-01        1        0       1       0
    1           1  2021-01-01        2        0       1       0
    """
    logger.info("Generating Sparse matrix for Time Series Features")
    id_cols = ["patient_id", "timestamp"]

    # Confirm dataframe is sorted
    check_df = df.select(pl.col(id_cols))
    assert check_df.sort(by=id_cols).collect().equals(check_df.collect()), "data frame must be sorted"

    # Generate sparse matrix
    if agg in CODE_AGGREGATIONS:
        code_df = df.drop(columns=id_cols + ["numerical_value"])
        data, (rows, cols) = get_long_code_df(code_df, ts_columns)
    elif agg in VALUE_AGGREGATIONS:
        value_df = df.drop(columns=id_cols)
        data, (rows, cols) = get_long_value_df(value_df, ts_columns)

    sp_matrix = csr_array(
        (data, (rows, cols)),
        shape=(df.select(pl.len()).collect().item(), len(ts_columns)),
    )
    return df.select(pl.col(id_cols)), sp_matrix


def get_flat_ts_rep(
    agg: str,
    feature_columns: list[str],
    shard_df: DF_T,
) -> pl.LazyFrame:
    """Produce a flat time series representation from a given data frame, focusing on non-static feature
    columns.

    This function filters the given data frame for non-static features based on the 'feature_columns'
    provided and generates a flat time series representation using these dynamic features. The resulting
    data frame includes both codes and values transformed and aggregated appropriately.

    Args:
        feature_columns (list[str]): A list of column identifiers that determine which features are considered
            for dynamic analysis.
        shard_df (DF_T): The data frame containing time-stamped data from which features will be extracted
            and summarized.

    Returns:
        pl.LazyFrame: A LazyFrame consisting of the processed time series data, combining both code and value
            representations.

    Example:
        >>> feature_columns = ['A/value', 'A/code', 'B/value', 'B/code',
        ...                    "C/value", "C/code", "A/static/present"]
        >>> data = {'patient_id': [1, 1, 1, 2, 2, 2],
        ...         'code': ['A', 'A', 'B', 'B', 'C', 'C'],
        ...         'timestamp': ['2021-01-01', '2021-01-01', '2020-01-01', '2021-01-04', None, None],
        ...         'numerical_value': [1, 2, 2, 2, 3, 4]}
        >>> df = pl.DataFrame(data).lazy()
        >>> pivot_df = get_flat_ts_rep(feature_columns, df)
        >>> pivot_df
           patient_id   timestamp  A/value  B/value  C/value  A/code  B/code  C/code
        0           1  2021-01-01        1        0        0       1       0       0
        1           1  2021-01-01        2        0        0       1       0       0
        2           1  2020-01-01        0        2        0       0       1       0
        3           2  2021-01-04        0        2        0       0       1       0
    """
    # Remove codes not in training set
    shard_df = get_events_df(shard_df, feature_columns)
    ts_columns = get_feature_names(agg, feature_columns)
    return summarize_dynamic_measurements(agg, ts_columns, shard_df)
