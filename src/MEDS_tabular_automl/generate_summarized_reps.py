from collections.abc import Callable

import pandas as pd

pd.set_option("compute.use_numba", True)
import numpy as np
import polars as pl
from loguru import logger
from scipy.sparse import coo_array, csr_array, sparray, vstack

from MEDS_tabular_automl.generate_ts_features import get_feature_names, get_flat_ts_rep
from MEDS_tabular_automl.utils import CODE_AGGREGATIONS, VALUE_AGGREGATIONS, load_tqdm


def time_aggd_col_alias_fntr(window_size: str, agg: str) -> Callable[[str], str]:
    if agg is None:
        raise ValueError("Aggregation type 'agg' must be provided")

    def f(c: str) -> str:
        if c in ["patient_id", "timestamp"]:
            return c
        else:
            return "/".join([window_size] + c.split("/") + [agg])

    return f


def sparse_aggregate(sparse_matrix, agg):
    if agg == "sum":
        merged_matrix = sparse_matrix.sum(axis=0, dtype=sparse_matrix.dtype)
    elif agg == "min":
        merged_matrix = sparse_matrix.min(axis=0)
    elif agg == "max":
        merged_matrix = sparse_matrix.max(axis=0)
    elif agg == "sum_sqd":
        merged_matrix = sparse_matrix.power(2).sum(axis=0, dtype=sparse_matrix.dtype)
    elif agg == "count":
        merged_matrix = sparse_matrix.getnnz(axis=0)
    else:
        raise ValueError(f"Aggregation method '{agg}' not implemented.")
    return merged_matrix


def sum_merge_timestamps(df, sparse_matrix, agg):
    """Groups by timestamp and combines rows that are on the same date.

    The combining is done by summing the rows in the sparse matrix that correspond to the same date.

    Args:
        df (DataFrame): The DataFrame with 'timestamp' and 'patient_id'.
        sparse_matrix (csr_matrix): The corresponding sparse matrix with data.
        agg (str): Aggregation method, currently only 'sum' is implemented.

    Returns:
        DataFrame, csr_matrix: Tuple containing the DataFrame with aggregated timestamps and the corresponding
            sparse matrix.
    """
    # Assuming 'timestamp' is already sorted; if not, uncomment the next line:
    # df = df.sort_values(by='timestamp')

    # Group by timestamp and sum the data
    grouped = df.groupby("timestamp")
    indices = grouped.indices

    # Create a new sparse matrix with summed rows per unique timestamp
    patient_id = df["patient_id"].iloc[0]
    timestamps = []
    output_matrix = csr_array((0, sparse_matrix.shape[1]), dtype=sparse_matrix.dtype)

    # Loop through each group and sum
    for timestamp, rows in indices.items():
        # Combine the rows in the sparse matrix for the current group (respecting the aggregation being used)
        merged_matrix = sparse_aggregate(sparse_matrix[rows], agg)
        # Save the non-zero elements
        output_matrix = vstack([output_matrix, merged_matrix])
        timestamps.extend([timestamp])

    # Create output DataFrame
    out_df = pd.DataFrame({"patient_id": [patient_id] * len(timestamps), "timestamp": timestamps})
    return out_df, output_matrix


def sparse_rolling(df, sparse_matrix, timedelta, agg):
    """Iterates through rolling windows while maintaining sparsity.

    Example:

    >>> df = pd.DataFrame({'patient_id': {0: 1, 1: 1, 2: 1},
    ...  'timestamp': {0: pd.Timestamp('2021-01-01 00:00:00'),
    ...  1: pd.Timestamp('2021-01-01 00:00:00'), 2: pd.Timestamp('2020-01-01 00:00:00')},
    ...  'A/code': {0: 1, 1: 1, 2: 0}, 'B/code': {0: 0, 1: 0, 2: 1}, 'C/code': {0: 0, 1: 0, 2: 0}})
    >>> for col in ["A/code", "B/code", "C/code"]: df[col] = pd.arrays.SparseArray(df[col])
    >>> sparse_rolling(df, pd.Timedelta("1d"), "sum").dtypes
    A/code       Sparse[int64, 0]
    B/code       Sparse[int64, 0]
    C/code       Sparse[int64, 0]
    timestamp      datetime64[ns]
    dtype: object
    """
    patient_id = df.iloc[0].patient_id
    df = df.drop(columns="patient_id").reset_index(drop=True).reset_index()
    timestamps = []
    out_sparse_matrix = coo_array((0, sparse_matrix.shape[1]), dtype=sparse_matrix.dtype)
    for each in df[["index", "timestamp"]].rolling(on="timestamp", window=timedelta):
        timestamps.append(each.index.max())
        agg_subset_matrix = sparse_aggregate(sparse_matrix[each["index"]], agg)
        out_sparse_matrix = vstack([out_sparse_matrix, agg_subset_matrix])
    out_df = pd.DataFrame({"patient_id": [patient_id] * len(timestamps), "timestamp": timestamps})
    return out_df, out_sparse_matrix


def get_rolling_window_indicies(index_df, window_size):
    """Get the indices for the rolling windows."""
    if window_size == "full":
        newest_date = index_df.select(pl.col("timestamp")).max().collect().item()
        oldest_date = index_df.select(pl.col("timestamp")).min().collect().item()
        timedelta = newest_date - oldest_date + pd.Timedelta(days=1)
    else:
        timedelta = pd.Timedelta(window_size)
    return (
        index_df.with_row_index("index")
        .rolling(index_column="timestamp", period=timedelta, group_by="patient_id")
        .agg([pl.col("index").min().alias("min_index"), pl.col("index").max().alias("max_index")])
        .select(pl.col("min_index", "max_index"))
        .collect()
    )


def aggregate_matrix(windows, matrix, agg, num_features, use_tqdm=False):
    """Aggregate the matrix based on the windows."""
    tqdm = load_tqdm(use_tqdm)
    agg = agg.split("/")[-1]
    dtype = np.float32
    matrix = csr_array(matrix.astype(dtype))
    if agg.startswith("sum"):
        out_dtype = np.float32
    else:
        out_dtype = np.int32
    data, row, col = [], [], []
    for i, window in tqdm(enumerate(windows.iter_rows(named=True)), total=len(windows)):
        min_index = window["min_index"]
        max_index = window["max_index"]
        subset_matrix = matrix[min_index : max_index + 1, :]
        agg_matrix = sparse_aggregate(subset_matrix, agg).astype(out_dtype)
        nozero_ind = np.nonzero(agg_matrix)[0]
        col.append(nozero_ind)
        data.append(agg_matrix[nozero_ind])
        row.append(np.repeat(np.array(i, dtype=np.int32), len(nozero_ind)))
    row = np.concatenate(row)
    out_matrix = coo_array(
        (np.concatenate(data), (row, np.concatenate(col))),
        dtype=out_dtype,
        shape=(windows.shape[0], num_features),
    )
    return csr_array(out_matrix)


def compute_agg(index_df, matrix: sparray, window_size: str, agg: str, num_features: int, use_tqdm=False):
    """Applies aggreagtion to dataframe.

    Dataframe is expected to only have the relevant columns for aggregating
    It should have the patient_id and timestamp columns, and then only code columns
    if agg is a code aggregation or only value columns if it is a value aggreagation.

    Example:
    >>> from MEDS_tabular_automl.generate_ts_features import get_flat_ts_rep
    >>> feature_columns = ['A/value/sum', 'A/code/count', 'B/value/sum', 'B/code/count',
    ...                    "C/value/sum", "C/code/count", "A/static/present"]
    >>> data = {'patient_id': [1, 1, 1, 2, 2, 2],
    ...         'code': ['A', 'A', 'B', 'B', 'C', 'C'],
    ...         'timestamp': ['2021-01-01', '2021-01-01', '2020-01-01', '2021-01-04', None, None],
    ...         'numerical_value': [1, 2, 2, 2, 3, 4]}
    >>> df = pl.DataFrame(data).lazy()
    >>> df = get_flat_ts_rep(feature_columns, df)
    >>> df
       patient_id   timestamp  A/value  B/value  C/value  A/code  B/code  C/code
    0           1  2021-01-01        1        0        0       1       0       0
    1           1  2021-01-01        2        0        0       1       0       0
    2           1  2020-01-01        0        2        0       0       1       0
    3           2  2021-01-04        0        2        0       0       1       0
    >>> df['timestamp'] = pd.to_datetime(df['timestamp'])
    >>> df.dtypes
    patient_id               int64
    timestamp       datetime64[ns]
    A/value       Sparse[int64, 0]
    B/value       Sparse[int64, 0]
    C/value       Sparse[int64, 0]
    A/code        Sparse[int64, 0]
    B/code        Sparse[int64, 0]
    C/code        Sparse[int64, 0]
    dtype: object
    >>> output = compute_agg(df[['patient_id', 'timestamp', 'A/code', 'B/code', 'C/code']],
    ...  "1d", "code/count")
    >>> output
       1d/A/code/count  1d/B/code/count  1d/C/code/count  timestamp  patient_id
    0                1                0                0 2021-01-01           1
    1                2                0                0 2021-01-01           1
    2                0                1                0 2020-01-01           1
    0                0                1                0 2021-01-04           2
    >>> output.dtypes
    1d/A/code/count    Sparse[int64, 0]
    1d/B/code/count    Sparse[int64, 0]
    1d/C/code/count    Sparse[int64, 0]
    timestamp            datetime64[ns]
    patient_id                    int64
    dtype: object
    """
    group_df = (
        index_df.with_row_index("index")
        .group_by(["patient_id", "timestamp"], maintain_order=True)
        .agg([pl.col("index").min().alias("min_index"), pl.col("index").max().alias("max_index")])
        .collect()
    )
    index_df = group_df.lazy().select(pl.col("patient_id", "timestamp"))
    windows = group_df.select(pl.col("min_index", "max_index"))
    logger.info("Step 1.5: Running sparse aggregation.")
    matrix = aggregate_matrix(windows, matrix, agg, num_features, use_tqdm)
    logger.info("Step 2: computing rolling windows and aggregating.")
    windows = get_rolling_window_indicies(index_df, window_size)
    logger.info("Starting final sparse aggregations.")
    matrix = aggregate_matrix(windows, matrix, agg, num_features, use_tqdm)
    return matrix


def _generate_summary(
    ts_columns: list[str],
    index_df: pd.DataFrame,
    matrix: sparray,
    window_size: str,
    agg: str,
    num_features,
    use_tqdm=False,
) -> pl.LazyFrame:
    """Generate a summary of the data frame for a given window size and aggregation.

    Args:
    - df (DF_T): The data frame to summarize.
    - window_size (str): The window size to use for the summary.
    - agg (str): The aggregation to apply to the data frame.

    Returns:
    - pl.LazyFrame: The summarized data frame.

    Expect:
        >>> from MEDS_tabular_automl.generate_ts_features import get_flat_ts_rep
        >>> feature_columns = ['A/value/sum', 'A/code/count', 'B/value/sum', 'B/code/count',
        ...                    "C/value/sum", "C/code/count", "A/static/present"]
        >>> data = {'patient_id': [1, 1, 1, 2, 2, 2],
        ...         'code': ['A', 'A', 'B', 'B', 'C', 'C'],
        ...         'timestamp': ['2021-01-01', '2021-01-01', '2020-01-01', '2021-01-04', None, None],
        ...         'numerical_value': [1, 2, 2, 2, 3, 4]}
        >>> df = pl.DataFrame(data).lazy()
        >>> pivot_df = get_flat_ts_rep(feature_columns, df)
        >>> pivot_df['timestamp'] = pd.to_datetime(pivot_df['timestamp'])
        >>> pivot_df
           patient_id  timestamp  A/value  B/value  C/value  A/code  B/code  C/code
        0           1 2021-01-01        1        0        0       1       0       0
        1           1 2021-01-01        2        0        0       1       0       0
        2           1 2020-01-01        0        2        0       0       1       0
        3           2 2021-01-04        0        2        0       0       1       0
        >>> _generate_summary(pivot_df, "full", "value/sum")
           full/A/value/count  full/B/value/count  full/C/value/count  timestamp  patient_id
        0                   1                   0                   0 2021-01-01           1
        1                   3                   0                   0 2021-01-01           1
        2                   3                   2                   0 2021-01-01           1
        0                   0                   2                   0 2021-01-04           2
    """
    if agg not in CODE_AGGREGATIONS + VALUE_AGGREGATIONS:
        raise ValueError(
            f"Invalid aggregation: {agg}. Valid options are: {CODE_AGGREGATIONS + VALUE_AGGREGATIONS}"
        )
    out_matrix = compute_agg(index_df, matrix, window_size, agg, num_features, use_tqdm=use_tqdm)
    return out_matrix


def generate_summary(
    feature_columns: list[str], index_df: pl.LazyFrame, matrix: sparray, window_size, agg: str, use_tqdm=False
) -> pl.LazyFrame:
    """Generate a summary of the data frame for given window sizes and aggregations.

    This function processes a dataframe to apply specified aggregations over defined window sizes.
    It then joins the resulting frames on 'patient_id' and 'timestamp', and ensures all specified
    feature columns exist in the final output, adding missing ones with default values.

    Args:
        feature_columns (list[str]): List of all feature columns that must exist in the final output.
        df (list[pl.LazyFrame]): The input dataframes to process, expected to be length 2 list with code_df
            (pivoted shard with binary presence of codes) and value_df (pivoted shard with numerical values
            for each code).
        window_sizes (list[str]): List of window sizes to apply for summarization.
        aggregations (list[str]): List of aggregations to perform within each window size.

    Returns:
        pl.LazyFrame: A LazyFrame containing the summarized data with all required features present.

    Expect:
        >>> from datetime import date
        >>> wide_df = pd.DataFrame({"patient_id": [1, 1, 1, 2],
        ...     "A/code": [1, 1, 0, 0],
        ...     "B/code": [0, 0, 1, 1],
        ...     "A/value": [1, 2, 3, None],
        ...     "B/value": [None, None, None, 4.0],
        ...     "timestamp": [date(2021, 1, 1), date(2021, 1, 1),date(2020, 1, 3), date(2021, 1, 4)],
        ...     })
        >>> wide_df['timestamp'] = pd.to_datetime(wide_df['timestamp'])
        >>> for col in ["A/code", "B/code", "A/value", "B/value"]:
        ...     wide_df[col] = pd.arrays.SparseArray(wide_df[col])
        >>> feature_columns = ["A/code/count", "B/code/count", "A/value/sum", "B/value/sum"]
        >>> aggregations = ["code/count", "value/sum"]
        >>> window_sizes = ["full", "1d"]
        >>> generate_summary(feature_columns, wide_df, window_sizes, aggregations)[
        ...    ["1d/A/code/count", "full/B/code/count", "full/B/value/sum"]]
           1d/A/code/count  full/B/code/count  full/B/value/sum
        0              NaN                1.0                 0
        1              NaN                1.0                 0
        2              NaN                1.0                 0
        0              NaN                1.0                 0
        0              NaN                NaN                 0
        1              NaN                NaN                 0
        2              NaN                NaN                 0
        0              NaN                NaN                 0
        0                0                NaN                 0
        1              1.0                NaN                 0
        2              2.0                NaN                 0
        0                0                NaN                 0
        0              NaN                NaN                 0
        1              NaN                NaN                 0
        2              NaN                NaN                 0
        0              NaN                NaN                 0
    """
    assert len(feature_columns), "feature_columns must be a non-empty list"
    ts_columns = get_feature_names(agg, feature_columns)
    # Generate summaries for each window size and aggregation
    code_type, _ = agg.split("/")
    # only iterate through code_types that exist in the dataframe columns
    assert any([c.endswith(code_type) for c in ts_columns])
    logger.info(
        f"Generating aggregation {agg} for window_size {window_size}, with {len(ts_columns)} columns."
    )
    out_matrix = _generate_summary(
        ts_columns, index_df, matrix, window_size, agg, len(ts_columns), use_tqdm=use_tqdm
    )
    return out_matrix


if __name__ == "__main__":
    import json
    from pathlib import Path

    feature_columns = json.load(
        open(
            Path("/storage/shared/meds_tabular_ml/ebcl_dataset/processed/tabularize") / "feature_columns.json"
        )
    )
    df = pl.scan_parquet(
        Path("/storage/shared/meds_tabular_ml/ebcl_dataset/processed")
        / "final_cohort"
        / "held_out"
        / "7.parquet"
    )
    agg = "code/count"
    index_df, sparse_matrix = get_flat_ts_rep(agg, feature_columns, df)
    generate_summary(
        feature_columns=feature_columns,
        index_df=index_df,
        matrix=sparse_matrix,
        window_size="1d",
        agg=agg,
        use_tqdm=True,
    )
