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


def feature_name_to_code(feature_name: str) -> str:
    """Converts a feature name to a code name."""
    return "/".join(feature_name.split("/")[:-1])


def get_long_code_df(df, ts_columns):
    """Pivots the codes data frame to a long format one-hot rep for time series data."""
    column_to_int = {feature_name_to_code(col): i for i, col in enumerate(ts_columns)}
    x = df.with_columns(
        pl.col("code").cast(str).replace(column_to_int).cast(int).alias("code_index"),
        pl.lit(1).alias("count"),
    ).drop("code")
    # sum up counts for same patient_id, timestamp, code_index
    x = x.group_by("patient_id", "timestamp", "code_index").sum()
    # combine codes and counts for same patient_id, timestamp
    x = x.group_by("patient_id", "timestamp", maintain_order=True).agg(pl.col("code_index", "count"))

    # repeat row_index for each code_index on that row (i.e. 1 row == 1 unique patient_id x timestamp)
    rows = (
        x.with_row_index("row_index")
        .select(pl.col("row_index").repeat_by(pl.col("code_index").list.len()))
        .select(pl.col("row_index").explode())
        .collect()
        .to_numpy()
        .T
    )[0]
    cols = x.select(pl.col("code_index").explode()).collect().to_numpy().T[0]
    data = x.select(pl.col("count").explode()).collect().to_numpy().T[0]
    shape = (x.select(pl.len()).collect().item(), len(ts_columns))
    return data, (rows, cols), shape


def get_long_value_df(df, ts_columns):
    """Pivots the numerical value data frame to a long format for time series data."""
    column_to_int = {feature_name_to_code(col): i for i, col in enumerate(ts_columns)}
    value_df = (
        df.with_row_index("index").drop_nulls("numerical_value").filter(pl.col("code").is_in(ts_columns))
    )
    rows = value_df.select(pl.col("index")).collect().to_series().to_numpy()
    cols = (
        value_df.with_columns(pl.col("code").cast(str).replace(column_to_int).cast(int).alias("value_index"))
        .select("value_index")
        .collect()
        .to_series()
        .to_numpy()
    )
    assert np.issubdtype(cols.dtype, np.number), "numerical_value must be a numerical type"
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
        code_df = df.drop(*(["numerical_value"]))
        data, (rows, cols), shape = get_long_code_df(code_df, ts_columns)
    elif agg in VALUE_AGGREGATIONS:
        value_df = df.drop(*id_cols)
        data, (rows, cols) = get_long_value_df(value_df, ts_columns)
        shape = (df.select(pl.len()).collect().item(), len(ts_columns))

    sp_matrix = csr_array(
        (data, (rows, cols)),
        shape=shape,
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
