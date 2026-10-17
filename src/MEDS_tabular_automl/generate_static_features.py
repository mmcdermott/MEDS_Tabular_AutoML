"""This module provides functions for generating static representations of patient data for use in automated
machine learning models. It includes functionality to summarize measurements based on static features and then
transform them into a tabular format suitable for analysis. The module leverages the polars library for
efficient data manipulation.

Functions:
- convert_to_matrix: Converts a Polars DataFrame to a sparse matrix.
- get_sparse_static_rep: Merges static and time-series dataframes into a sparse representation.
- summarize_static_measurements: Summarizes static measurements from a given DataFrame.
- get_flat_static_rep: Produces a tabular representation of static data features.
"""

import logging

import numpy as np
import polars as pl
from scipy.sparse import coo_array, csr_array

from MEDS_tabular_automl.utils import (
    STATIC_CODE_AGGREGATION,
    STATIC_VALUE_AGGREGATION,
    get_events_df,
    get_feature_names,
    get_unique_time_events_df,
    parse_static_feature_column,
)

logger = logging.getLogger(__name__)


def convert_to_matrix(df: pl.DataFrame, num_events: int, num_features: int) -> csr_array:
    """Converts a Polars DataFrame to a sparse matrix.

    Args:
        df: The DataFrame to convert.
        num_events: Number of events to set matrix dimension.
        num_features: Number of features to set matrix dimension.

    Returns:
        A sparse matrix representation of the DataFrame.
    """
    dense_matrix = df.drop("subject_id").collect().to_numpy()
    data_list = []
    rows = []
    cols = []
    for row in range(dense_matrix.shape[0]):
        for col in range(dense_matrix.shape[1]):
            data = dense_matrix[row, col]
            if (data is not None) and (data != 0):
                data_list.append(data)
                rows.append(row)
                cols.append(col)
    matrix = csr_array((data_list, (rows, cols)), shape=(num_events, num_features))
    return matrix


def get_subject_specific_sparse_static_rep(static_df: pl.DataFrame, static_features: list[str]):
    """Given a DataFrame of static features, returns a sparse matrix representation of the data.

    Args:
        static_df: A DataFrame containing static features.
        static_features: A list of static feature names.

    Returns:
        A sparse matrix representation of the DataFrame.

    Example:
    >>> static_df = pl.DataFrame({"subject_id": [1, 2, 3], "A": [1, 2, 3], "B": [4, 5, 6]})
    >>> meds_df = pl.DataFrame({"subject_id": [1, 1, 1, 2, 2, 2], "code": ["A", "B", "A", "B", "A", "B"]})

    Observe that you get a sparse matrix with one row per static_df row
    >>> get_subject_specific_sparse_static_rep(static_df=static_df.lazy(), static_features=["A", "B"]).shape
    (3, 2)
    """
    return convert_to_matrix(
        static_df,
        num_events=static_df.select(pl.col("subject_id").n_unique()).collect().item(),
        num_features=len(static_features),
    )


def get_sparse_static_rep(
    static_features: list[str], static_df: pl.LazyFrame, meds_df: pl.LazyFrame, feature_columns: list[str]
) -> coo_array:
    """Merges static and time-series dataframes into a sparse representation based on the subject_id column.

    Each row of meds_df gets the static values of its subject, matched by subject_id, and each static value
    goes to the column of its feature name in static_features. Rows of subjects without static records, and
    columns of features that static_df lacks, are empty. Null, NaN and False values are not stored.

    Args:
        static_features: A list of static feature names.
        static_df: A DataFrame containing static features, with one row per subject_id, sorted by subject_id,
            and one column per static feature name.
        meds_df: A DataFrame containing time-series features.
        feature_columns (list[str]): A list of feature columns to include in the merged DataFrame.

    Returns:
        A sparse array representation of the merged static and time-series features.

    Example:
    >>> static_df = pl.DataFrame({"subject_id": [1, 2, 3], "A": [1, 2, 3], "B": [4, 5, 6]})
    >>> meds_df = pl.DataFrame({"subject_id": [1, 1, 1, 2, 2, 2], "code": ["A", "B", "A", "B", "A", "B"]})
    >>> feature_columns = ["A/static/present", "B/static/first", "A/static/present", "B/static/first"]

    Observe that you get a sparse matrix with one row per meds_df row
    >>> get_sparse_static_rep(static_features=["A", "B"], static_df=static_df.lazy(),
    ...                       meds_df=meds_df.lazy(), feature_columns=feature_columns).shape
    (6, 2)

    Rows are matched to subjects by subject_id, so a subject without static records (here, 2) gets an empty
    row, and values are matched to columns by name, so a feature missing from static_df (here, A) gets an
    empty column. Observed zeros are stored:
    >>> static_df = pl.DataFrame({"subject_id": [1, 3], "B": [4.0, 0.0]})
    >>> meds_df = pl.DataFrame({"subject_id": [1, 2, 3, 3]})
    >>> matrix = get_sparse_static_rep(static_features=["A", "B"], static_df=static_df.lazy(),
    ...                                meds_df=meds_df.lazy(), feature_columns=[])
    >>> matrix.toarray()
    array([[0., 4.],
           [0., 0.],
           [0., 0.],
           [0., 0.]])
    >>> matrix.row.tolist(), matrix.col.tolist(), matrix.data.tolist()
    ([0, 2, 3], [1, 1, 1], [4.0, 0.0, 0.0])
    """
    # Make static data sparse and merge it with the time-series data
    logger.info("Make static data sparse and merge it with the time-series data")
    # Check static_df is sorted and unique and raise error if it is not
    if not static_df.select(pl.col("subject_id")).collect().to_series().is_sorted():
        raise ValueError("static_df is not sorted by subject_id.")
    if not (
        static_df.select(pl.len()).collect().item()
        == static_df.select(pl.col("subject_id").n_unique()).collect().item()
    ):
        raise ValueError("static_df has duplicate subject_id values.")

    static_df = static_df.collect()
    static_subjects = static_df["subject_id"].to_numpy()

    # One row per subject of static_df, plus a last, empty row for subjects without static records.
    data, rows, cols = [np.zeros(0)], [np.zeros(0, dtype=np.int64)], [np.zeros(0, dtype=np.int64)]
    for col, feature in enumerate(static_features):
        if feature not in static_df.columns:
            continue
        values = static_df[feature]
        if values.dtype == pl.Boolean:
            # Presence indicators: only True is an observation.
            values = values.fill_null(False).to_numpy().astype(np.float64)
            observed = values == 1
        else:
            # Values: nulls (and NaNs) are not observations.
            values = values.cast(pl.Float64).to_numpy()
            observed = ~np.isnan(values)
        data.append(values[observed])
        rows.append(np.flatnonzero(observed))
        cols.append(np.full(observed.sum(), col))
    static_matrix = csr_array(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(static_subjects) + 1, len(static_features)),
    )

    # Repeat each subject's static row for each of its rows in meds_df.
    row_subjects = meds_df.select(pl.col("subject_id")).collect().to_series().to_numpy()
    static_rows = np.searchsorted(static_subjects, row_subjects)
    has_static = static_rows < len(static_subjects)
    has_static[has_static] = static_subjects[static_rows[has_static]] == row_subjects[has_static]
    static_rows[~has_static] = len(static_subjects)
    return coo_array(static_matrix[static_rows, :])


def summarize_static_measurements(
    agg: str,
    feature_columns: list[str],
    df: pl.LazyFrame,
) -> pl.LazyFrame:
    """Aggregates static measurements for feature columns that are marked as 'present' or 'first'.

    This function first filters for features that need to be recorded as the first occurrence
    or simply as present, then performs a pivot to reshape the data for each patient, providing
    a tabular format where each row represents a patient and each column represents a static feature.
    Only static (timeless) records are used; timed records of the same codes are not.

    Args:
        agg: The type of aggregation ('present' or 'first').
        feature_columns: A list of feature column identifiers marked for static analysis.
        df: The DataFrame from which features will be extracted and summarized.

    Returns:
        A LazyFrame containing summarized data pivoted by 'subject_id' for each static feature.

    Example:
    >>> feature_columns = ['A/static/first', 'B/static/first', 'A/static/present', 'B/static/present']
    >>> df = pl.DataFrame({'subject_id': [1, 1, 1, 1, 1, 2, 2, 2],
    ...                    'code': ['A', 'A', 'B', 'B', 'C', 'A', 'A', 'B'],
    ...                    'time': [None, None, None, None, None, None, None, datetime(2021, 1, 1)],
    ...                    'numeric_value': [1, None, 2, 3, None, None, 3, 4]})
    >>> result = summarize_static_measurements('static/first', feature_columns, df.lazy())
    >>> result
    shape: (2, 3)
    ┌────────────┬────────────────┬────────────────┐
    │ subject_id ┆ A/static/first ┆ B/static/first │
    │ ---        ┆ ---            ┆ ---            │
    │ i64        ┆ f64            ┆ f64            │
    ╞════════════╪════════════════╪════════════════╡
    │ 1          ┆ 1.0            ┆ 2.5            │
    │ 2          ┆ 3.0            ┆ null           │
    └────────────┴────────────────┴────────────────┘
    >>> result = summarize_static_measurements('static/present', feature_columns, df.lazy())
    >>> result
    shape: (2, 3)
    ┌────────────┬──────────────────┬──────────────────┐
    │ subject_id ┆ A/static/present ┆ B/static/present │
    │ ---        ┆ ---              ┆ ---              │
    │ i64        ┆ bool             ┆ bool             │
    ╞════════════╪══════════════════╪══════════════════╡
    │ 1          ┆ true             ┆ true             │
    │ 2          ┆ true             ┆ false            │
    └────────────┴──────────────────┴──────────────────┘

    Static values keep their full precision:

    >>> df = pl.DataFrame({'subject_id': [1], 'code': ['A'], 'time': [None],
    ...                    'numeric_value': [160.3953106166676]})
    >>> result = summarize_static_measurements('static/first', ['A/static/first'], df.lazy())
    >>> result['A/static/first'].to_list()
    [160.3953106166676]
    """
    if agg == STATIC_VALUE_AGGREGATION:
        static_features = get_feature_names(agg=agg, feature_columns=feature_columns)
        # Handling 'first' static values
        static_first_codes = [parse_static_feature_column(c)[0] for c in static_features]
        code_subset = df.filter(pl.col("code").is_in(static_first_codes) & pl.col("time").is_null())
        first_code_subset = code_subset.group_by(["subject_id", "code"]).mean().collect()
        static_value_pivot_df = first_code_subset.pivot(
            index=["subject_id"], on=["code"], values=["numeric_value"], aggregate_function=None
        )
        # rename code to feature name
        remap_cols = {
            input_name: output_name
            for input_name, output_name in zip(static_first_codes, static_features, strict=False)
            if input_name in static_value_pivot_df.columns
        }
        static_value_pivot_df = static_value_pivot_df.select(
            *["subject_id"], *[pl.col(k).alias(v).cast(pl.Float64) for k, v in remap_cols.items()]
        ).sort(by="subject_id")
        return static_value_pivot_df
    elif agg == STATIC_CODE_AGGREGATION:
        static_features = get_feature_names(agg=agg, feature_columns=feature_columns)
        # Handling 'present' static indicators
        static_present_codes = [parse_static_feature_column(c)[0] for c in static_features]
        static_present_pivot_df = (
            df.filter(pl.col("time").is_null())
            .select(*["subject_id", "code"])
            .filter(pl.col("code").is_in(static_present_codes))
            .with_columns(pl.lit(True).alias("__indicator"))
            .collect()
            .pivot(
                index=["subject_id"],
                on=["code"],
                values="__indicator",
                aggregate_function="sum",
            )
            .sort(by="subject_id")
        )
        remap_cols = {
            input_name: output_name
            for input_name, output_name in zip(static_present_codes, static_features, strict=False)
            if input_name in static_present_pivot_df.columns
        }
        # rename columns to final feature names
        static_present_pivot_df = static_present_pivot_df.select(
            *["subject_id"], *[pl.col(k).alias(v).cast(pl.Boolean) for k, v in remap_cols.items()]
        )
        return static_present_pivot_df
    else:
        raise ValueError(f"Invalid aggregation type: {agg}")


def get_flat_static_rep(
    agg: str, feature_columns: list[str], shard_df: pl.LazyFrame, label_df: pl.LazyFrame | None
) -> coo_array:
    """Produces a sparse representation for static data from a specified shard DataFrame.

    This function selects the appropriate static features, summarizes them using
    summarize_static_measurements, and then normalizes the resulting data to ensure
    it is suitable for further analysis or machine learning tasks.

    Args:
        agg: The aggregation method for static data.
        feature_columns: A list of feature columns to include.
        shard_df: The shard DataFrame containing the patient data.
        label_df: The label DataFrame containing the labels for the shard data.

    Returns:
        A sparse array representing the static features for the provided shard of data.
    """
    static_features = get_feature_names(agg=agg, feature_columns=feature_columns)
    static_measurements = summarize_static_measurements(agg, static_features, df=shard_df)
    if len(static_features) == 0:
        raise ValueError(f"No static features found. Remove the aggregation function {agg}")
    # convert to sparse_matrix
    if label_df is not None:
        event_df = label_df.rename({"prediction_time": "time"})
    else:
        event_df = get_unique_time_events_df(get_events_df(shard_df, feature_columns))

    matrix = get_sparse_static_rep(static_features, static_measurements.lazy(), event_df, feature_columns)
    if not matrix.shape[1] == len(static_features):
        raise ValueError(f"Expected {len(static_features)} features, got {matrix.shape[1]}")
    return matrix
