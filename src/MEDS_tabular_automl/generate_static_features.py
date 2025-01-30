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

import numpy as np
import polars as pl
from loguru import logger
from scipy.sparse import coo_array, csr_array

from MEDS_tabular_automl.utils import (
    STATIC_CODE_AGGREGATION,
    STATIC_VALUE_AGGREGATION,
    get_events_df,
    get_feature_names,
    get_unique_time_events_df,
    parse_static_feature_column,
)


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


def get_sparse_static_rep(
    static_features: list[str], static_df: pl.DataFrame, meds_df: pl.DataFrame, feature_columns: list[str]
) -> coo_array:
    """Merges static and time-series dataframes into a sparse representation based on the subject_id column.

    Args:
        static_features: A list of static feature names.
        static_df: A DataFrame containing static features.
        meds_df: A DataFrame containing time-series features.
        feature_columns (list[str]): A list of feature columns to include in the merged DataFrame.

    Returns:
        A sparse array representation of the merged static and time-series features.
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

    # load static data as sparse matrix
    static_matrix = convert_to_matrix(
        static_df, num_events=meds_df.select(pl.len()).collect().item(), num_features=len(static_features)
    )
    # Duplicate static matrix rows to match time-series data
    events_per_patient = (
        meds_df.select(pl.col("subject_id").value_counts())
        .unnest("subject_id")
        .sort(by="subject_id")
        .select(pl.col("count"))
        .collect()
        .to_series()
    )
    reindex_slices = np.repeat(range(len(events_per_patient)), events_per_patient)
    static_matrix = static_matrix[reindex_slices, :]
    return coo_array(static_matrix)


def summarize_static_measurements(
    agg: str,
    feature_columns: list[str],
    df: pl.LazyFrame,
) -> pl.LazyFrame:
    """Aggregates static measurements for feature columns that are marked as 'present' or 'first'.

    This function first filters for features that need to be recorded as the first occurrence
    or simply as present, then performs a pivot to reshape the data for each patient, providing
    a tabular format where each row represents a patient and each column represents a static feature.

    Args:
        agg: The type of aggregation ('present' or 'first').
        feature_columns: A list of feature column identifiers marked for static analysis.
        df: The DataFrame from which features will be extracted and summarized.

    Returns:
        A LazyFrame containing summarized data pivoted by 'subject_id' for each static feature.
    """
    if agg == STATIC_VALUE_AGGREGATION:
        static_features = get_feature_names(agg=agg, feature_columns=feature_columns)
        # Handling 'first' static values
        static_first_codes = [parse_static_feature_column(c)[0] for c in static_features]
        code_subset = df.filter(pl.col("code").is_in(static_first_codes))
        first_code_subset = code_subset.group_by(pl.col("subject_id")).first().collect()
        static_value_pivot_df = first_code_subset.pivot(
            index=["subject_id"], columns=["code"], values=["numeric_value"], aggregate_function=None
        )
        # rename code to feature name
        remap_cols = {
            input_name: output_name
            for input_name, output_name in zip(static_first_codes, static_features)
            if input_name in static_value_pivot_df.columns
        }
        static_value_pivot_df = static_value_pivot_df.select(
            *["subject_id"], *[pl.col(k).alias(v).cast(pl.Boolean) for k, v in remap_cols.items()]
        ).sort(by="subject_id")
        # pivot can be faster: https://stackoverflow.com/questions/73522017/replacing-a-pivot-with-a-lazy-groupby-operation # noqa: E501
        # TODO: consider casting with .cast(pl.Float32))
        return static_value_pivot_df
    elif agg == STATIC_CODE_AGGREGATION:
        static_features = get_feature_names(agg=agg, feature_columns=feature_columns)
        # Handling 'present' static indicators
        static_present_codes = [parse_static_feature_column(c)[0] for c in static_features]
        static_present_pivot_df = (
            df.select(*["subject_id", "code"])
            .filter(pl.col("code").is_in(static_present_codes))
            .with_columns(pl.lit(True).alias("__indicator"))
            .collect()
            .pivot(
                index=["subject_id"],
                columns=["code"],
                values="__indicator",
                aggregate_function=None,
            )
            .sort(by="subject_id")
        )
        remap_cols = {
            input_name: output_name
            for input_name, output_name in zip(static_present_codes, static_features)
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
