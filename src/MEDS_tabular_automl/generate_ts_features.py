import warnings

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from scipy.sparse import csr_array

from MEDS_tabular_automl.generate_static_features import (
    STATIC_CODE_COL,
    STATIC_VALUE_COL,
)
from MEDS_tabular_automl.utils import DF_T

warnings.simplefilter(action="ignore", category=FutureWarning)


def get_ts_columns(feature_columns):
    def is_static(c):
        return c.endswith(STATIC_CODE_COL) or c.endswith(STATIC_VALUE_COL)

    ts_columns = sorted(list({c for c in feature_columns if not is_static(c)}))
    return ts_columns


def fill_missing_entries_with_nan(sparse_df, type, columns):
    # Fill missing entries with NaN
    for col in columns:
        sparse_df[col] = sparse_df[col].astype(pd.SparseDtype(type, fill_value=np.nan))
    return sparse_df


def get_long_code_df(df, ts_columns):
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
    column_to_int = {col: i for i, col in enumerate(ts_columns)}
    value_df = df.drop_nulls("numerical_value")
    rows = range(value_df.select(pl.len()).collect().item())
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

    # Generate sparse matrices
    value_df = df.drop(columns=id_cols)
    value_data, (value_rows, value_cols) = get_long_value_df(value_df, ts_columns)
    code_df = df.drop(columns=id_cols + ["numerical_value"])
    code_data, (code_rows, code_cols) = get_long_code_df(code_df, ts_columns)

    merge_data = np.concatenate([value_data, code_data])
    merge_rows = np.concatenate([value_rows, code_rows])
    merge_cols = np.concatenate([value_cols, code_cols])
    merge_columns = ts_columns
    sp_matrix = csr_array(
        (merge_data, (merge_rows, merge_cols)),
        shape=(value_df.select(pl.len()).collect().item(), len(merge_columns)),
    )
    return df.select(pl.col(id_cols)), sp_matrix


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
    raw_feature_columns = ["/".join(c.split("/")[:-1]) for c in feature_columns]
    shard_df = shard_df.filter(pl.col("code").is_in(raw_feature_columns))

    ts_columns = get_ts_columns(feature_columns)
    ts_shard_df = shard_df.drop_nulls(subset=["timestamp", "code"])
    return summarize_dynamic_measurements(ts_columns, ts_shard_df)
