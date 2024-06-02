"""This module provides functions for generating static representations of patient data for use in automated
machine learning models. It includes functionality to summarize measurements based on static features and then
transform them into a tabular format suitable for analysis. The module leverages the polars library for
efficient data manipulation.

Functions:
- _summarize_static_measurements: Summarizes static measurements from a given DataFrame.
- get_flat_static_rep: Produces a tabular representation of static data features.
"""

import numpy as np
import polars as pl
from loguru import logger
from scipy.sparse import coo_array, csr_array

from MEDS_tabular_automl.utils import (
    DF_T,
    STATIC_CODE_AGGREGATION,
    STATIC_VALUE_AGGREGATION,
    get_events_df,
    get_feature_names,
    parse_static_feature_column,
)


def convert_to_matrix(df, num_events, num_features):
    """Converts a Polars DataFrame to a sparse matrix."""
    dense_matrix = df.drop(columns="patient_id").collect().to_numpy()
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


def get_sparse_static_rep(static_features, static_df, meds_df, feature_columns) -> coo_array:
    """Merges static and time-series dataframes.

    This function merges the static and time-series dataframes based on the patient_id column.

    Args:
    - feature_columns (List[str]): A list of feature columns to include in the merged dataframe.
    - static_df (pd.DataFrame): A dataframe containing static features.
    - ts_df (pd.DataFrame): A dataframe containing time-series features.

    Returns:
    - pd.DataFrame: A merged dataframe containing static and time-series features.
    """
    # TODO - Eventually do this duplication at the task specific stage after filtering patients and features
    # Make static data sparse and merge it with the time-series data
    logger.info("Make static data sparse and merge it with the time-series data")
    # Check static_df is sorted and unique
    assert static_df.select(pl.col("patient_id")).collect().to_series().is_sorted()
    assert (
        static_df.select(pl.len()).collect().item()
        == static_df.select(pl.col("patient_id").n_unique()).collect().item()
    )
    meds_df = get_events_df(meds_df, feature_columns)

    # load static data as sparse matrix
    static_matrix = convert_to_matrix(
        static_df, num_events=meds_df.select(pl.len()).collect().item(), num_features=len(static_features)
    )
    # Duplicate static matrix rows to match time-series data
    events_per_patient = (
        meds_df.select(pl.col("patient_id").value_counts())
        .unnest("patient_id")
        .sort(by="patient_id")
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
    df: DF_T,
) -> pl.LazyFrame:
    """Aggregates static measurements for feature columns that are marked as 'present' or 'first'.

    Parameters:
    - feature_columns (list[str]): List of feature column identifiers that are specifically marked
        for staticanalysis.
    - df (DF_T): Data frame from which features will be extracted and summarized.

    Returns:
    - pl.LazyFrame: A LazyFrame containing the summarized data pivoted by 'patient_id'
        for each static feature.

    This function first filters for features that need to be recorded as the first occurrence
    or simply as present, then performs a pivot to reshape the data for each patient, providing
    a tabular format where each row represents a patient and each column represents a static feature.
    """
    if agg == STATIC_VALUE_AGGREGATION:
        static_features = get_feature_names(agg=agg, feature_columns=feature_columns)
        # Handling 'first' static values
        static_first_codes = [parse_static_feature_column(c)[0] for c in static_features]
        code_subset = df.filter(pl.col("code").is_in(static_first_codes))
        first_code_subset = code_subset.group_by(pl.col("patient_id")).first().collect()
        static_value_pivot_df = first_code_subset.pivot(
            index=["patient_id"], columns=["code"], values=["numerical_value"], aggregate_function=None
        )
        # rename code to feature name
        remap_cols = {
            input_name: output_name
            for input_name, output_name in zip(static_first_codes, static_features)
            if input_name in static_value_pivot_df.columns
        }
        static_value_pivot_df = static_value_pivot_df.select(
            *["patient_id"], *[pl.col(k).alias(v).cast(pl.Boolean) for k, v in remap_cols.items()]
        ).sort(by="patient_id")
        # pivot can be faster: https://stackoverflow.com/questions/73522017/replacing-a-pivot-with-a-lazy-groupby-operation # noqa: E501
        # TODO: consider casting with .cast(pl.Float32))
        return static_value_pivot_df
    elif agg == STATIC_CODE_AGGREGATION:
        static_features = get_feature_names(agg=agg, feature_columns=feature_columns)
        # Handling 'present' static indicators
        static_present_codes = [parse_static_feature_column(c)[0] for c in static_features]
        static_present_pivot_df = (
            df.select(*["patient_id", "code"])
            .filter(pl.col("code").is_in(static_present_codes))
            .with_columns(pl.lit(True).alias("__indicator"))
            .collect()
            .pivot(
                index=["patient_id"],
                columns=["code"],
                values="__indicator",
                aggregate_function=None,
            )
            .sort(by="patient_id")
        )
        remap_cols = {
            input_name: output_name
            for input_name, output_name in zip(static_present_codes, static_features)
            if input_name in static_present_pivot_df.columns
        }
        # rename columns to final feature names
        static_present_pivot_df = static_present_pivot_df.select(
            *["patient_id"], *[pl.col(k).alias(v).cast(pl.Boolean) for k, v in remap_cols.items()]
        )
        return static_present_pivot_df
    else:
        raise ValueError(f"Invalid aggregation type: {agg}")


def get_flat_static_rep(
    agg: str,
    feature_columns: list[str],
    shard_df: DF_T,
) -> coo_array:
    """Produces a raw representation for static data from a specified shard DataFrame.

    Parameters:
    - feature_columns (list[str]): List of feature columns to include in the static representation.
    - shard_df (DF_T): The shard DataFrame containing patient data.

    Returns:
    - pl.LazyFrame: A LazyFrame that includes all static features for the data provided.

    This function selects the appropriate static features, summarizes them using
    _summarize_static_measurements, and then normalizes the resulting data to ensure it is
    suitable for further analysis or machine learning tasks.
    """
    static_features = get_feature_names(agg=agg, feature_columns=feature_columns)
    static_measurements = summarize_static_measurements(agg, static_features, df=shard_df)
    # convert to sparse_matrix
    matrix = get_sparse_static_rep(static_features, static_measurements.lazy(), shard_df, feature_columns)
    assert matrix.shape[1] == len(
        static_features
    ), f"Expected {len(static_features)} features, got {matrix.shape[1]}"
    return matrix
