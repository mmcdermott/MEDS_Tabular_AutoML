import warnings

import numpy as np
import pandas as pd
import polars as pl
from scipy.sparse import csc_array

from MEDS_tabular_automl.utils import DF_T

warnings.simplefilter(action="ignore", category=FutureWarning)


def get_ts_columns(feature_columns):
    def get_code_type(c):
        return c.split("/")[-2] == "code"

    def get_code_name(c):
        return "/".join(c.split("/")[0:-2])

    ts_columns = sorted(list({get_code_name(c) for c in feature_columns if not get_code_type(c) == "static"}))
    return ts_columns


def fill_missing_entries_with_nan(sparse_df, type):
    # Fill missing entries with NaN
    for col in sparse_df.columns:
        sparse_df[col] = sparse_df[col].astype(pd.SparseDtype(type, fill_value=np.nan))
    return sparse_df


def get_long_code_df(df, ts_columns):
    column_to_int = {col: i for i, col in enumerate(ts_columns)}
    rows = range(len(df))
    cols = df["code"].map(column_to_int)
    data = np.ones(len(df), dtype=np.bool_)
    sparse_matrix = csc_array((data, (rows, cols)), shape=(len(df), len(ts_columns)))
    long_code_df = pd.DataFrame.sparse.from_spmatrix(sparse_matrix, columns=ts_columns)
    # long_code_df = fill_missing_entries_with_nan(long_code_df, bool)
    return long_code_df


def get_long_value_df(df, ts_columns):
    column_to_int = {col: i for i, col in enumerate(ts_columns)}
    value_rows = range(len(df))
    value_cols = df["code"].map(column_to_int)
    value_data = df["numerical_value"]
    value_sparse_matrix = csc_array((value_data, (value_rows, value_cols)), shape=(len(df), len(ts_columns)))
    long_value_df = pd.DataFrame.sparse.from_spmatrix(value_sparse_matrix, columns=ts_columns)
    # long_value_df = fill_missing_entries_with_nan(long_value_df, np.float64)
    return long_value_df


def summarize_dynamic_measurements(
    ts_columns: list[str],
    df: DF_T,
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
    >>> df = pl.DataFrame(data).lazy()
    >>> ts_columns = ['A', 'B']
    >>> long_df = summarize_dynamic_measurements(ts_columns, df)
    >>> long_df.head()
       patient_id   timestamp  A/value  B/value  A/code  B/code
    0           1  2021-01-01      1.0      NaN    True     NaN
    1           1  2021-01-01      2.0      NaN    True     NaN
    2           1  2020-01-01      NaN      2.0     NaN    True
    3           2  2021-01-04      NaN      2.0     NaN    True
    >>> long_df.shape
    (4, 6)
    >>> long_df = summarize_dynamic_measurements(ts_columns, df.filter(pl.col("code") == "A"))
    >>> long_df
       patient_id   timestamp  A/value  B/value  A/code  B/code
    0           1  2021-01-01      1.0      NaN    True     NaN
    1           1  2021-01-01      2.0      NaN    True     NaN
    """
    df = df.collect().to_pandas()
    id_cols = ["patient_id", "timestamp"]
    code_df = df.drop(columns=id_cols + ["numerical_value"])
    long_code_df = get_long_code_df(code_df, ts_columns)

    value_df = df.drop(columns=id_cols)
    long_value_df = get_long_value_df(value_df, ts_columns)
    long_df = pd.concat(
        [
            df[["patient_id", "timestamp"]],
            long_value_df.rename(columns=lambda c: f"{c}/value"),
            long_code_df.rename(columns=lambda c: f"{c}/code"),
        ],
        axis=1,
    )
    return long_df


def get_flat_ts_rep(
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
        >>> feature_columns = ['A/value/sum', 'A/code/count', 'B/value/sum', 'B/code/count',
        ...                    "C/value/sum", "C/code/count", "A/static/present"]
        >>> data = {'patient_id': [1, 1, 1, 2, 2, 2],
        ...         'code': ['A', 'A', 'B', 'B', 'C', 'C'],
        ...         'timestamp': ['2021-01-01', '2021-01-01', '2020-01-01', '2021-01-04', None, None],
        ...         'numerical_value': [1, 2, 2, 2, 3, 4]}
        >>> df = pl.DataFrame(data).lazy()
        >>> pivot_df = get_flat_ts_rep(feature_columns, df)
        >>> pivot_df
           patient_id   timestamp  A/value  B/value  C/value  A/code  B/code  C/code
        0           1  2021-01-01      1.0      NaN      NaN    True     NaN     NaN
        1           1  2021-01-01      2.0      NaN      NaN    True     NaN     NaN
        2           1  2020-01-01      NaN      2.0      NaN     NaN    True     NaN
        3           2  2021-01-04      NaN      2.0      NaN     NaN    True     NaN
    """

    ts_columns = get_ts_columns(feature_columns)
    ts_shard_df = shard_df.drop_nulls(
        subset=["timestamp", "code"]
    )  # filter(pl.col("timestamp", "").is_not_null())
    return summarize_dynamic_measurements(ts_columns, ts_shard_df)
