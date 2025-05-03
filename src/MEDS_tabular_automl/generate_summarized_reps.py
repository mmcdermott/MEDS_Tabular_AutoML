import numpy as np
import pandas as pd
import polars as pl

pl.enable_string_cache()
import logging

from scipy.sparse import coo_array, csr_array, sparray

logger = logging.getLogger(__name__)

from MEDS_tabular_automl.generate_ts_features import get_feature_names
from MEDS_tabular_automl.utils import (
    CODE_AGGREGATIONS,
    VALUE_AGGREGATIONS,
    get_min_dtype,
    load_tqdm,
)


def sparse_aggregate(sparse_matrix: sparray, agg: str) -> np.ndarray | coo_array:
    """Aggregates values in a sparse matrix according to the specified method.

    Args:
        sparse_matrix: The sparse matrix to aggregate.
        agg: The aggregation method to apply, such as 'sum', 'min', 'max', 'sum_sqd', or 'count'.

    Returns:
        The aggregated sparse matrix.

    Raises:
        ValueError: If the aggregation method is not implemented.
    """
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


def get_rolling_window_indicies(
    index_df: pl.LazyFrame, window_size: str, label_df: pl.LazyFrame = None
) -> pl.LazyFrame:
    """Computes the start and end indices for rolling window operations on a LazyFrame.

    Args:
        index_df: The LazyFrame containing the indices.
        window_size: The size of the window as a string denoting time, e.g., '7d' for 7 days.

    Returns:
        A LazyFrame with columns 'min_index' and 'max_index' representing the range of each window.

    Example:
    >>> index_df = pl.DataFrame({"subject_id": [1, 1, 1, 1, 1, 1],
    ...                          "time": pl.Series(["2021-01-02",
    ...                          "2021-01-06", "2021-01-11", "2021-01-16",
    ...                          "2021-01-21", "2021-01-26"]).str.strptime(pl.Date)})
    >>> get_rolling_window_indicies(index_df.lazy(), "7d")
    shape: (6, 2)
    ┌───────────┬───────────┐
    │ min_index ┆ max_index │
    │ ---       ┆ ---       │
    │ u32       ┆ u32       │
    ╞═══════════╪═══════════╡
    │ 0         ┆ 1         │
    │ 0         ┆ 2         │
    │ 1         ┆ 3         │
    │ 2         ┆ 4         │
    │ 3         ┆ 5         │
    │ 4         ┆ 6         │
    └───────────┴───────────┘
    >>> label_df = pl.DataFrame({"subject_id": [1],
    ...                          "prediction_time": pl.Series(["2021-01-02"]).str.strptime(pl.Date)})
    >>> get_rolling_window_indicies(index_df.lazy(), "7d", label_df.lazy())
    shape: (1, 2)
    ┌───────────┬───────────┐
    │ min_index ┆ max_index │
    │ ---       ┆ ---       │
    │ u32       ┆ u32       │
    ╞═══════════╪═══════════╡
    │ 0         ┆ 1         │
    └───────────┴───────────┘

    Test label prior to all observations returns empty window
    >>> label_df = pl.DataFrame({"subject_id": [1],
    ...                          "prediction_time": pl.Series(["2021-01-01"]).str.strptime(pl.Date)})
    >>> get_rolling_window_indicies(index_df.lazy(), "7d", label_df.lazy())
    shape: (1, 2)
    ┌───────────┬───────────┐
    │ min_index ┆ max_index │
    │ ---       ┆ ---       │
    │ u32       ┆ u32       │
    ╞═══════════╪═══════════╡
    │ 0         ┆ 0         │
    └───────────┴───────────┘

    Test label after all observations returns the window ending at the last event for the patient
    >>> label_df = pl.DataFrame({"subject_id": [1],
    ...                          "prediction_time": pl.Series(["2022-01-01"]).str.strptime(pl.Date)})
    >>> get_rolling_window_indicies(index_df.lazy(), "7d", label_df.lazy())
    shape: (1, 2)
    ┌───────────┬───────────┐
    │ min_index ┆ max_index │
    │ ---       ┆ ---       │
    │ u32       ┆ u32       │
    ╞═══════════╪═══════════╡
    │ 4         ┆ 6         │
    └───────────┴───────────┘

    Confirm a label with no observations within a window from it, will extract the previous event's window.
    >>> label_df = pl.DataFrame({"subject_id": [1],
    ...                          "prediction_time": pl.Series(["2021-01-08"]).str.strptime(pl.Date)})
    >>> get_rolling_window_indicies(index_df.lazy(), "1d", label_df.lazy())
    shape: (1, 2)
    ┌───────────┬───────────┐
    │ min_index ┆ max_index │
    │ ---       ┆ ---       │
    │ u32       ┆ u32       │
    ╞═══════════╪═══════════╡
    │ 1         ┆ 2         │
    └───────────┴───────────┘

    Check if there is no data for the patient, the min_index will be 0 and max_index will be 0
    >>> index_df = pl.DataFrame({"subject_id": [],
    ...                          "time": pl.Series([]).cast(pl.Date)})
    >>> get_rolling_window_indicies(index_df.lazy(), "1d")
    shape: (0, 2)
    ┌───────────┬───────────┐
    │ min_index ┆ max_index │
    │ ---       ┆ ---       │
    │ u32       ┆ u32       │
    ╞═══════════╪═══════════╡
    └───────────┴───────────┘
    """
    if window_size == "full":
        timedelta = pd.Timedelta(150 * 52, unit="W")  # just use 150 years as time delta
    else:
        timedelta = pd.Timedelta(window_size)
    windows = (
        index_df.with_row_index("index")
        .rolling(index_column="time", period=timedelta, group_by="subject_id")
        .agg([pl.col("index").min().alias("min_index"), pl.col("index").max().alias("max_index")])
        .select(pl.col("min_index"), pl.col("max_index") + 1)
        .collect()
    )
    if label_df is not None:
        event_df = pl.concat([index_df, windows.lazy()], how="horizontal")
        windows = (
            label_df.rename({"prediction_time": "time"})
            .join_asof(event_df, by="subject_id", on="time")
            .select(windows.columns)
            .fill_null(0)
            .collect()
        )
    return windows


def aggregate_matrix(
    windows: pl.LazyFrame, matrix: sparray, agg: str, num_features: int, use_tqdm: bool = False
) -> csr_array:
    """Aggregates a matrix according to defined windows and specified aggregation method.

    Args:
        windows: The LazyFrame containing 'min_index' and 'max_index' for each window.
        matrix: The matrix to aggregate.
        agg: The aggregation method to apply.
        num_features: The number of features in the matrix.
        use_tqdm: The flag to enable progress display.

    Returns:
        Aggregated sparse matrix.

    Raises:
        TypeError: If the type of the aggregated matrix is not compatible for further operations.

    Example:
        >>> windows = pl.DataFrame({"min_index": [0, 0, 1], "max_index": [2, 3, 3]})
        >>> matrix = coo_array(([1, 2, 3, 1, 1], ([0, 1, 2, 0, 2], [0, 0, 0, 1, 2])), shape=(3, 3))
        >>> matrix.toarray()
        array([[1, 1, 0],
               [2, 0, 0],
               [3, 0, 1]])
        >>> agg = "sum"
        >>> num_features = 3
        >>> aggregated_matrix = aggregate_matrix(windows, matrix, agg, num_features)
        >>> aggregated_matrix.toarray()
        array([[3, 1, 0],
               [6, 1, 1],
               [5, 0, 1]])

        >>> windows = pl.DataFrame({"min_index": [0], "max_index": [0]})
        >>> aggregate_matrix(windows, matrix, 'sum', num_features).toarray()
        array([[0., 0., 0.]])
        >>> aggregate_matrix(windows, matrix, 'min', num_features).toarray()
        array([[0., 0., 0.]])
        >>> aggregate_matrix(windows, matrix, 'max', num_features).toarray()
        array([[0., 0., 0.]])
        >>> aggregate_matrix(windows, matrix, 'sum_sqd', num_features).toarray()
        array([[0., 0., 0.]])
        >>> aggregate_matrix(windows, matrix, 'count', num_features).toarray()
        array([[0., 0., 0.]])
    """
    tqdm = load_tqdm(use_tqdm)
    agg = agg.split("/")[-1]
    matrix = csr_array(matrix)
    data, row, col = [], [], []
    for i, window in tqdm(enumerate(windows.iter_rows(named=True)), total=len(windows)):
        min_index = window["min_index"]
        max_index = window["max_index"]
        if min_index == max_index:
            continue
        subset_matrix = matrix[min_index:max_index, :]
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
    if len(row) == 0:
        logger.warning("No data to aggregate for this shard. Returning empty aggregation matrix.")
        return csr_array(([], ([], [])), shape=(windows.shape[0], num_features))
    row = np.concatenate(row)
    data = np.concatenate(data)
    col = np.concatenate(col)
    if len(data):
        row = row.astype(get_min_dtype(row), copy=False)
        col = col.astype(get_min_dtype(col), copy=False)
        data = data.astype(get_min_dtype(data), copy=False)
    out_matrix = csr_array(
        (data, (row, col)),
        shape=(windows.shape[0], num_features),
    )
    return out_matrix


def compute_agg(
    index_df: pl.LazyFrame,
    matrix: sparray,
    window_size: str,
    agg: str,
    num_features: int,
    label_df: pl.LazyFrame | None = None,
    use_tqdm: bool = False,
) -> csr_array:
    """Applies aggregation to a sparse matrix using rolling window indices derived from a DataFrame.

    Dataframe is expected to only have the relevant columns for aggregating. It should have the subject_id and
    time columns, and then only code columns if agg is a code aggregation or only value columns if it is
    a value aggregation.

    Args:
        index_df: The DataFrame with 'subject_id' and 'time' columns used for grouping.
        matrix: The sparse matrix to be aggregated.
        window_size: The string defining the rolling window size.
        agg: The string specifying the aggregation method.
        num_features: The number of features in the matrix.
        label_df: The DataFrame with labels. If provided, only perform aggregations at the label times.
        use_tqdm: The flag to enable or disable tqdm progress bar.

    Returns:
        The aggregated sparse matrix.
    """
    group_df = (
        index_df.with_row_index("index")
        .group_by(["subject_id", "time"], maintain_order=True)
        .agg([pl.col("index").min().alias("min_index"), pl.col("index").max().alias("max_index")])
        .collect()
    )
    index_df = group_df.lazy().select(pl.col("subject_id", "time"))
    windows = group_df.select(pl.col("min_index", "max_index"))

    if label_df is not None:
        logger.info("Step 2: computing rolling windows and aggregating.")
        windows = get_rolling_window_indicies(index_df, window_size, label_df)
    else:
        logger.info("Step 1.5: Running sparse aggregation.")
        matrix = aggregate_matrix(windows, matrix, agg, num_features, use_tqdm)
        logger.info("Step 2: computing rolling windows and aggregating.")
        windows = get_rolling_window_indicies(index_df, window_size)

    logger.info("Starting final sparse aggregations.")
    matrix = aggregate_matrix(windows, matrix, agg, num_features, use_tqdm)
    return matrix


def generate_summary(
    feature_columns: list[str],
    index_df: pl.LazyFrame,
    matrix: sparray,
    window_size: str,
    agg: str,
    label_df: pl.LazyFrame | None = None,
    use_tqdm: bool = False,
) -> csr_array:
    """Generate a summary of the data frame for a given window size and aggregation.

    Args:
        feature_columns: A list of all feature columns that must exist in the final output.
        index_df: The DataFrame with index and grouping information.
        matrix: The sparse matrix containing the data to aggregate.
        window_size: The size of the rolling window used for summary.
        agg: The aggregation function to apply.
        label_df: The DataFrame with labels.
        use_tqdm: The flag to enable or disable progress display.

    Returns:
        The summary of data as a sparse matrix.

    Raises:
        ValueError: If the aggregation type is not supported.
    """
    if agg not in CODE_AGGREGATIONS + VALUE_AGGREGATIONS:
        raise ValueError(
            f"Invalid aggregation: {agg}. Valid options are: {CODE_AGGREGATIONS + VALUE_AGGREGATIONS}"
        )
    if not len(feature_columns):
        raise ValueError("No feature columns provided -- feature_columns must be a non-empty list.")

    ts_columns = get_feature_names(agg, feature_columns)
    # Generate summaries for each window size and aggregation
    code_type, _ = agg.split("/")
    # only iterate through code_types that exist in the dataframe columns
    if not any(c.endswith(code_type) for c in ts_columns):
        raise ValueError(f"No columns found for aggregation {agg} in feature_columns: {ts_columns}.")

    logger.info(
        f"Generating aggregation {agg} for window_size {window_size}, with {len(ts_columns)} columns."
    )

    out_matrix = compute_agg(index_df, matrix, window_size, agg, len(ts_columns), label_df, use_tqdm=use_tqdm)
    return out_matrix
