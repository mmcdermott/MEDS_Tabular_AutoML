import logging

import numpy as np
import pandas as pd
import polars as pl
from scipy.sparse import coo_array, csc_array, csr_array, sparray

from MEDS_tabular_automl.generate_ts_features import get_feature_names
from MEDS_tabular_automl.utils import (
    CODE_AGGREGATIONS,
    VALUE_AGGREGATIONS,
    get_min_dtype,
    load_tqdm,
)

logger = logging.getLogger(__name__)


def sparse_aggregate(sparse_matrix: sparray, agg: str) -> coo_array:
    """Aggregates the stored values of each column of a sparse matrix according to the specified method.

    Only stored entries are aggregated: a row without an entry in a column has no value there, so it is not
    a zero for that column's min or max. Every column with at least one stored entry gets a result, which is
    stored even when it is zero (e.g., a sum that cancels out). Integer values are aggregated as int64 and
    floating point values as float64.

    Args:
        sparse_matrix: The sparse matrix to aggregate.
        agg: The aggregation method to apply, such as 'sum', 'min', 'max', 'sum_sqd', or 'count'.

    Returns:
        A sparse matrix with a single row, holding the result of each column that has a stored entry.

    Raises:
        ValueError: If the aggregation method is not implemented.

    Examples:
        >>> matrix = csr_array(([5.0, 2.0, -1.0, 1.0], ([0, 2, 1, 2], [0, 0, 1, 1])), shape=(3, 3))
        >>> matrix.toarray()
        array([[ 5.,  0.,  0.],
               [ 0., -1.,  0.],
               [ 2.,  1.,  0.]])
        >>> for agg in ["sum", "min", "max", "sum_sqd", "count"]:
        ...     result = sparse_aggregate(matrix, agg)
        ...     print(agg, result.shape, result.col.tolist(), result.data.tolist())
        sum (1, 3) [0, 1] [7.0, 0.0]
        min (1, 3) [0, 1] [2.0, -1.0]
        max (1, 3) [0, 1] [5.0, 1.0]
        sum_sqd (1, 3) [0, 1] [29.0, 2.0]
        count (1, 3) [0, 1] [2, 2]
        >>> sparse_aggregate(matrix, "mean")
        Traceback (most recent call last):
            ...
        ValueError: Aggregation method 'mean' not implemented.
    """
    reductions = {"sum": np.add, "sum_sqd": np.add, "min": np.minimum, "max": np.maximum}
    if agg != "count" and agg not in reductions:
        raise ValueError(f"Aggregation method '{agg}' not implemented.")
    matrix = csc_array(sparse_matrix)
    counts = np.diff(matrix.indptr)
    columns = np.flatnonzero(counts)
    if agg == "count":
        values = counts[columns]
    else:
        data = matrix.data.astype(np.int64 if matrix.dtype.kind in "biu" else np.float64)
        if agg == "sum_sqd":
            data = data * data
        # The stored entries of each column are contiguous in CSC order and start at its indptr.
        values = reductions[agg].reduceat(data, matrix.indptr[columns])
    return coo_array((values, (np.zeros(len(columns), dtype=np.int64), columns)), shape=(1, matrix.shape[1]))


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

    The full window holds all of the subject's events up to each one, however far back they go
    >>> index_df = pl.DataFrame({
    ...     "subject_id": [1, 1, 2],
    ...     "time": pl.Series(["1850-01-01", "2010-01-01", "2010-01-01"]).str.strptime(pl.Date),
    ... })
    >>> get_rolling_window_indicies(index_df.lazy(), "full")
    shape: (3, 2)
    ┌───────────┬───────────┐
    │ min_index ┆ max_index │
    │ ---       ┆ ---       │
    │ u32       ┆ u32       │
    ╞═══════════╪═══════════╡
    │ 0         ┆ 1         │
    │ 0         ┆ 2         │
    │ 2         ┆ 3         │
    └───────────┴───────────┘
    """
    if window_size == "full":
        # Every event of the subject up to the current one. A finite period, however long, would leave out
        # older events.
        windows = (
            index_df.with_row_index("index")
            .select(
                pl.col("index").min().over("subject_id").alias("min_index"),
                (pl.col("index") + 1).alias("max_index"),
            )
            .collect()
        )
    else:
        windows = (
            index_df.with_row_index("index")
            .rolling(index_column="time", period=pd.Timedelta(window_size), group_by="subject_id")
            .agg([pl.col("index").min().alias("min_index"), pl.col("index").max().alias("max_index")])
            .select(pl.col("min_index"), pl.col("max_index") + 1)
            .collect()
        )
    if label_df is not None:
        event_df = pl.concat([index_df, windows.lazy()], how="horizontal")

        if "time" not in label_df.collect_schema():
            label_df = label_df.rename({"prediction_time": "time"})

        windows = (
            label_df.join_asof(event_df, by="subject_id", on="time")
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
        # `agg_matrix` is a column-wise aggregate over axis 0, so its own row index is always 0 -- that is
        # the row *within the aggregate*, not the window the aggregate belongs to. Using it writes every
        # window's result into row 0 of the output, where the CSR constructor then sums the duplicates.
        col.append(agg_matrix.col)
        data.append(agg_matrix.data)
        row.append(np.repeat(np.array(i, dtype=np.int32), len(agg_matrix.data)))
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
    # `aggregate_matrix` slices `matrix[min_index:max_index]`, so the upper bound is exclusive, and it
    # skips any window whose bounds are equal. `max_index` here is the *inclusive* index of the last
    # event in the (subject_id, time) group, so passing it through unchanged drops the last event of
    # every group -- and drops a group holding a single event entirely. `get_rolling_window_indicies`
    # already adds the 1 for exactly this reason; so does this.
    windows = group_df.select(pl.col("min_index"), pl.col("max_index") + 1)

    logger.info("Step 1.5: Running sparse aggregation.")
    matrix = aggregate_matrix(windows, matrix, agg, num_features, use_tqdm)
    # Each row now holds one timestamp's counts or sums of squares, which the windows below must add up.
    # Counting the rows again would count timestamps rather than events, and squaring them again would
    # square each timestamp's sum of squares. Sums, minima and maxima combine as they are.
    agg = {"code/count": "sum", "value/count": "sum", "value/sum_sqd": "sum"}.get(agg, agg)

    if label_df is not None:
        logger.info("Step 2: computing rolling windows and aggregating.")
        windows = get_rolling_window_indicies(index_df, window_size, label_df)
    else:
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
    code_type, _ = agg.split("/")  # code or value like in code/count
    # only iterate through code_types that exist in the dataframe columns
    if not any(c.endswith(code_type) for c in ts_columns):
        raise ValueError(f"No columns found for aggregation {agg} in feature_columns: {ts_columns}.")

    logger.info(
        f"Generating aggregation {agg} for window_size {window_size}, with {len(ts_columns)} columns."
    )

    out_matrix = compute_agg(index_df, matrix, window_size, agg, len(ts_columns), label_df, use_tqdm=use_tqdm)
    return out_matrix
