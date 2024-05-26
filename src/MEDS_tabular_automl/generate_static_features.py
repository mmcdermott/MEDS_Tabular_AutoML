"""This module provides functions for generating static representations of patient data for use in automated
machine learning models. It includes functionality to summarize measurements based on static features and then
transform them into a tabular format suitable for analysis. The module leverages the polars library for
efficient data manipulation.

Functions:
- _summarize_static_measurements: Summarizes static measurements from a given DataFrame.
- get_flat_static_rep: Produces a tabular representation of static data features.
"""

import polars as pl

from MEDS_tabular_automl.utils import (
    DF_T,
    _normalize_flat_rep_df_cols,
    _parse_flat_feature_column,
)


def _summarize_static_measurements(
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
    static_present = [c for c in feature_columns if c.startswith("STATIC_") and c.endswith("present")]
    static_first = [c for c in feature_columns if c.startswith("STATIC_") and c.endswith("first")]

    # Handling 'first' static values
    static_first_codes = [_parse_flat_feature_column(c)[1] for c in static_first]
    code_subset = df.filter(pl.col("code").is_in(static_first_codes))
    first_code_subset = code_subset.groupby(pl.col("patient_id")).first().collect()
    static_value_pivot_df = first_code_subset.pivot(
        index=["patient_id"], columns=["code"], values=["numerical_value"], aggregate_function=None
    )
    # rename code to feature name
    remap_cols = {
        input_name: output_name
        for input_name, output_name in zip(static_first_codes, static_first)
        if input_name in static_value_pivot_df.columns
    }
    static_value_pivot_df = static_value_pivot_df.select(
        *["patient_id"], *[pl.col(k).alias(v).cast(pl.Boolean) for k, v in remap_cols.items()]
    )
    # pivot can be faster: https://stackoverflow.com/questions/73522017/replacing-a-pivot-with-a-lazy-groupby-operation # noqa: E501
    # TODO: consider casting with .cast(pl.Float32))

    # Handling 'present' static indicators
    static_present_codes = [_parse_flat_feature_column(c)[1] for c in static_present]
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
    )
    remap_cols = {
        input_name: output_name
        for input_name, output_name in zip(static_present_codes, static_present)
        if input_name in static_present_pivot_df.columns
    }
    # rename columns to final feature names
    static_present_pivot_df = static_present_pivot_df.select(
        *["patient_id"], *[pl.col(k).alias(v).cast(pl.Boolean) for k, v in remap_cols.items()]
    )
    return pl.concat([static_value_pivot_df, static_present_pivot_df], how="align")


def get_flat_static_rep(
    feature_columns: list[str],
    shard_df: DF_T,
) -> pl.LazyFrame:
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
    static_features = [c for c in feature_columns if c.startswith("STATIC_")]
    static_measurements = _summarize_static_measurements(static_features, df=shard_df)
    # fill up missing feature columns with nulls
    normalized_measurements = _normalize_flat_rep_df_cols(
        static_measurements,
        static_features,
        set_count_0_to_null=False,
    )
    return normalized_measurements
