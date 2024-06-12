import numpy as np
import pandas as pd
import polars as pl

pl.enable_string_cache()
from loguru import logger
from scipy.sparse import coo_array, csr_array, sparray

from MEDS_tabular_automl.describe_codes import get_feature_columns
from MEDS_tabular_automl.generate_ts_features import get_feature_names, get_flat_ts_rep
from MEDS_tabular_automl.utils import (
    CODE_AGGREGATIONS,
    VALUE_AGGREGATIONS,
    get_min_dtype,
    load_tqdm,
)


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


def get_rolling_window_indicies(index_df, window_size):
    """Get the indices for the rolling windows."""
    if window_size == "full":
        timedelta = pd.Timedelta(150 * 52, unit="W")  # just use 150 years as time delta
    else:
        timedelta = pd.Timedelta(window_size)
    return (
        index_df.with_row_index("index")
        .rolling(index_column="timestamp", period=timedelta, group_by="patient_id")
        .agg([pl.col("index").min().alias("min_index"), pl.col("index").max().alias("max_index")])
        .select(pl.col("min_index", "max_index"))
        .collect()
    )


def precompute_matrix(windows, matrix, agg, num_features, use_tqdm=False):
    """Aggregate the matrix based on the windows."""
    tqdm = load_tqdm(use_tqdm)
    agg = agg.split("/")[-1]
    matrix = csr_array(matrix)
    # if agg.startswith("sum"):
    #     out_dtype = np.float32
    # else:
    #     out_dtype = np.int32
    data, row, col = [], [], []
    min_data, max_data = 0, 0
    num_vals = 0
    for i, window in tqdm(enumerate(windows.iter_rows(named=True)), total=len(windows)):
        min_index = window["min_index"]
        max_index = window["max_index"]
        subset_matrix = matrix[min_index : max_index + 1, :]
        agg_matrix = sparse_aggregate(subset_matrix, agg)
        if isinstance(agg_matrix, np.ndarray):
            num_vals += np.count_nonzero(agg_matrix)
        elif isinstance(agg_matrix, coo_array):
            num_vals += len(agg_matrix.data)
        else:
            raise TypeError(f"Invalid matrix type {type(agg_matrix)}")
    import pdb

    pdb.set_trace()
    row = np.empty(shape=num_vals, dtype=get_min_dtype([0, matrix.shape[0]]))
    col = np.empty(shape=num_vals, dtype=get_min_dtype([0, matrix.shape[1]]))
    data = np.empty(shape=num_vals, dtype=get_min_dtype([min_data, max_data]))
    return data, (row, col)


def aggregate_matrix(windows, matrix, agg, num_features, use_tqdm=False):
    """Aggregate the matrix based on the windows."""
    tqdm = load_tqdm(use_tqdm)
    agg = agg.split("/")[-1]
    matrix = csr_array(matrix)
    # if agg.startswith("sum"):
    #     out_dtype = np.float32
    # else:
    #     out_dtype = np.int32
    data, row, col = [], [], []
    for i, window in tqdm(enumerate(windows.iter_rows(named=True)), total=len(windows)):
        min_index = window["min_index"]
        max_index = window["max_index"]
        subset_matrix = matrix[min_index : max_index + 1, :]
        agg_matrix = sparse_aggregate(subset_matrix, agg)
        if isinstance(agg_matrix, np.ndarray):
            nozero_ind = np.nonzero(agg_matrix)[0]
            col.append(nozero_ind)
            data.append(agg_matrix[nozero_ind])
            row.append(np.repeat(np.array(i, dtype=np.int32), len(nozero_ind)))
        elif isinstance(agg_matrix, coo_array):
            col.append(agg_matrix.col)
            data.append(agg_matrix.data)
            row.append(agg_matrix.row)
        else:
            raise TypeError(f"Invalid matrix type {type(agg_matrix)}")
    row = np.concatenate(row)
    data = np.concatenate(data)
    col = np.concatenate(col)
    row = row.astype(get_min_dtype(row), copy=False)
    col = col.astype(get_min_dtype(col), copy=False)
    data = data.astype(get_min_dtype(data), copy=False)
    out_matrix = csr_array(
        (data, (row, col)),
        shape=(windows.shape[0], num_features),
    )
    return out_matrix


def compute_agg(index_df, matrix: sparray, window_size: str, agg: str, num_features: int, use_tqdm=False):
    """Applies aggreagtion to dataframe.

    Dataframe is expected to only have the relevant columns for aggregating
    It should have the patient_id and timestamp columns, and then only code columns
    if agg is a code aggregation or only value columns if it is a value aggreagation.

    Example:
    >>> from datetime import datetime
    >>> df = pd.DataFrame({
    ...     "patient_id": [1, 1, 1, 2],
    ...     "timestamp": [
    ...         datetime(2021, 1, 1), datetime(2021, 1, 1), datetime(2020, 1, 3), datetime(2021, 1, 4)
    ...     ],
    ...     "A/code": [1, 1, 0, 0],
    ...     "B/code": [0, 0, 1, 1],
    ...     "C/code": [0, 0, 0, 0],
    ... })
    >>> output = compute_agg(df, "1d", "code/count")
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
    # windows = group_df.select(pl.col("min_index", "max_index"))
    # import pdb; pdb.set_trace()
    # logger.info("Step 1.5: Running sparse aggregation.")
    windows = get_rolling_window_indicies(index_df, window_size)
    data, (row, col) = precompute_matrix(windows, matrix, agg, num_features, use_tqdm)
    logger.info("Step 2: computing rolling windows and aggregating.")
    import pdb

    pdb.set_trace()
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
        >>> from datetime import datetime
        >>> wide_df = pd.DataFrame({
        ...     "patient_id": [1, 1, 1, 2],
        ...     "timestamp": [
        ...         datetime(2021, 1, 1), datetime(2021, 1, 1), datetime(2020, 1, 3), datetime(2021, 1, 4)
        ...     ],
        ...     "A/code": [1, 1, 0, 0],
        ...     "B/code": [0, 0, 1, 1],
        ...     "C/code": [0, 0, 0, 0],
        ...     "A/value": [1, 2, 0, 0],
        ...     "B/value": [0, 0, 2, 2],
        ...     "C/value": [0, 0, 0, 0],
        ... })
        >>> _generate_summary(wide_df, "full", "value/sum")
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
    from pathlib import Path

    # feature_columns_fp = Path("/storage/shared/meds_tabular_ml/mimiciv_dataset/mimiciv_MEDS") \
    #     / "tabularized_code_metadata.parquet"
    # shard_fp = \
    #     Path("/storage/shared/meds_tabular_ml/mimiciv_dataset/mimiciv_MEDS/final_cohort/train/0.parquet")

    feature_columns_fp = (
        Path("/storage/shared/meds_tabular_ml/ebcl_dataset/processed") / "tabularized_code_metadata.parquet"
    )
    shard_fp = Path("/storage/shared/meds_tabular_ml/ebcl_dataset/processed/final_cohort/train/0.parquet")

    feature_columns = get_feature_columns(feature_columns_fp)
    df = pl.scan_parquet(shard_fp)
    agg = "code/count"
    index_df, sparse_matrix = get_flat_ts_rep(agg, feature_columns, df)
    generate_summary(
        feature_columns=feature_columns,
        index_df=index_df,
        matrix=sparse_matrix,
        window_size="full",
        agg=agg,
        use_tqdm=True,
    )
