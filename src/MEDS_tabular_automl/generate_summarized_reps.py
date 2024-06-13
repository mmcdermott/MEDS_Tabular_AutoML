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


def sparse_aggregate(sparse_matrix, agg, dtype):
    if agg == "sum":
        merged_matrix = sparse_matrix.sum(axis=0, dtype=dtype)
    elif agg == "min":
        merged_matrix = sparse_matrix.min(axis=0)
    elif agg == "max":
        merged_matrix = sparse_matrix.max(axis=0)
    elif agg == "sum_sqd":
        merged_matrix = sparse_matrix.power(2).sum(axis=0, dtype=dtype)
    elif agg == "count":
        merged_matrix = sparse_matrix.getnnz(axis=0)
    else:
        raise ValueError(f"Aggregation method '{agg}' not implemented.")
    return merged_matrix


def sparse_assign(row_index, index, data, row, col, sparse_matrix, value, dtype):
    agg_matrix = sparse_aggregate(sparse_matrix, value, dtype)
    if isinstance(agg_matrix, np.ndarray):
        nozero_ind = agg_matrix.nonzero()[0]
        len_data = len(nozero_ind)
        col[index : index + len_data] = nozero_ind
        data[index : index + len_data] = agg_matrix[nozero_ind]
        row[index : index + len_data] = row_index
    elif isinstance(agg_matrix, coo_array):
        len_data = len(agg_matrix.data)
        col[index : index + len_data] = agg_matrix.col
        data[index : index + len_data] = agg_matrix.data
        row[index : index + len_data] = row_index
    else:
        raise TypeError(f"Invalid matrix type {type(agg_matrix)}")
    return len_data


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
    min_data, max_data = 0, 0
    num_vals = 0
    for window in tqdm(windows.iter_rows(named=True), total=len(windows)):
        min_index = window["min_index"]
        max_index = window["max_index"]
        subset_matrix = matrix[min_index : max_index + 1, :]
        agg_matrix = sparse_aggregate(subset_matrix, agg, subset_matrix.dtype)
        if isinstance(agg_matrix, np.ndarray):
            num_vals += np.count_nonzero(agg_matrix)
        elif isinstance(agg_matrix, coo_array):
            num_vals += len(agg_matrix.data)
        else:
            raise TypeError(f"Invalid matrix type {type(agg_matrix)}")
    row = np.empty(shape=num_vals, dtype=get_min_dtype(np.array([0, matrix.shape[0]])))
    col = np.empty(shape=num_vals, dtype=get_min_dtype(np.array([0, matrix.shape[1]])))
    data = np.empty(shape=num_vals, dtype=get_min_dtype(np.array([min_data, max_data])))
    row.fill(0)
    col.fill(0)
    data.fill(0)
    return data, (row, col)


def aggregate_matrix(data, row, col, windows, matrix, agg, num_features, use_tqdm=False):
    """Aggregate the matrix based on the windows."""
    tqdm = load_tqdm(use_tqdm)
    agg = agg.split("/")[-1]
    matrix = csr_array(matrix)
    index = 0
    for row_index, window in tqdm(enumerate(windows.iter_rows(named=True)), total=len(windows)):
        min_index = window["min_index"]
        max_index = window["max_index"]
        subset_matrix = matrix[min_index : max_index + 1, :]
        len_data = sparse_assign(row_index, index, data, row, col, subset_matrix, agg, matrix.dtype)
        index += len_data
    out_matrix = dict(data=data, row=row, col=col, shape=(windows.shape[0], num_features))
    return out_matrix


def compute_agg(index_df, matrix: sparray, window_size: str, agg: str, num_features: int, use_tqdm=False):
    """Applies aggreagtion to dataframe.

    Dataframe is expected to only have the relevant columns for aggregating It should have the patient_id and
    timestamp columns, and then only code columns if agg is a code aggregation or only value columns if it is
    a value aggreagation.
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
    logger.info("Step 1: computing rolling windows and aggregating.")
    windows = get_rolling_window_indicies(index_df, window_size)
    logger.info("Step 2: Preallocating memory.")
    data, (row, col) = precompute_matrix(windows, matrix, agg, num_features, use_tqdm)
    logger.info("Step 3: Starting final sparse aggregations.")
    matrix = aggregate_matrix(data, row, col, windows, matrix, agg, num_features, use_tqdm)
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
