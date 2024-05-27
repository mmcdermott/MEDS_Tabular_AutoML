import polars as pl

from MEDS_tabular_automl.utils import DF_T

VALID_AGGREGATIONS = [
    "sum",
    "sum_sqd",
    "min",
    "max",
    "value",
    "first",
    "present",
    "count",
    "has_values_count",
]


def summarize_dynamic_measurements(
    ts_columns: list[str],
    df: DF_T,
) -> pl.LazyFrame:
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
    >>> df = pl.DataFrame(data).lazy()
    >>> ts_columns = ['A', 'B']
    >>> code_df, value_df = summarize_dynamic_measurements(ts_columns, df)
    >>> code_df.collect()
    shape: (4, 4)
    ┌────────────┬────────┬────────┬────────────┐
    │ patient_id ┆ code/A ┆ code/B ┆ timestamp  │
    │ ---        ┆ ---    ┆ ---    ┆ ---        │
    │ i64        ┆ u8     ┆ u8     ┆ str        │
    ╞════════════╪════════╪════════╪════════════╡
    │ 1          ┆ 1      ┆ 0      ┆ 2021-01-01 │
    │ 1          ┆ 1      ┆ 0      ┆ 2021-01-01 │
    │ 1          ┆ 0      ┆ 1      ┆ 2020-01-01 │
    │ 2          ┆ 0      ┆ 1      ┆ 2021-01-04 │
    └────────────┴────────┴────────┴────────────┘
    >>> value_df.collect()
    shape: (3, 4)
    ┌────────────┬────────────┬─────────┬─────────┐
    │ patient_id ┆ timestamp  ┆ value/A ┆ value/B │
    │ ---        ┆ ---        ┆ ---     ┆ ---     │
    │ i64        ┆ str        ┆ f64     ┆ f64     │
    ╞════════════╪════════════╪═════════╪═════════╡
    │ 1          ┆ 2021-01-01 ┆ 1.5     ┆ null    │
    │ 1          ┆ 2020-01-01 ┆ null    ┆ 2.0     │
    │ 2          ┆ 2021-01-04 ┆ null    ┆ 2.0     │
    └────────────┴────────────┴─────────┴─────────┘
    """

    value_df = (
        df.select("patient_id", "timestamp", "code", "numerical_value")
        .collect()
        .pivot(
            index=["patient_id", "timestamp"],
            columns=["code"],
            values=["numerical_value"],
            aggregate_function="mean",  # TODO round up counts so they are binary
            separator="/",
        )
        .lazy()
    )
    value_df = value_df.rename(lambda c: f"value/{c}" if c not in ["patient_id", "timestamp"] else c)
    code_df = df.drop("numerical_value").collect().to_dummies(columns=["code"], separator="/").lazy()
    return code_df, value_df


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
        >>> feature_columns = ['A', 'B', 'C', "static/A"]
        >>> data = {'patient_id': [1, 1, 1, 2, 2, 2],
        ...         'code': ['A', 'A', 'B', 'B', 'C', 'C'],
        ...         'timestamp': ['2021-01-01', '2021-01-01', '2020-01-01', '2021-01-04', None, None],
        ...         'numerical_value': [1, 2, 2, 2, 3, 4]}
        >>> df = pl.DataFrame(data).lazy()
        >>> code_df, value_df = get_flat_ts_rep(feature_columns, df)
        >>> code_df.collect()
        shape: (4, 4)
        ┌────────────┬────────┬────────┬────────────┐
        │ patient_id ┆ code/A ┆ code/B ┆ timestamp  │
        │ ---        ┆ ---    ┆ ---    ┆ ---        │
        │ i64        ┆ u8     ┆ u8     ┆ str        │
        ╞════════════╪════════╪════════╪════════════╡
        │ 1          ┆ 1      ┆ 0      ┆ 2021-01-01 │
        │ 1          ┆ 1      ┆ 0      ┆ 2021-01-01 │
        │ 1          ┆ 0      ┆ 1      ┆ 2020-01-01 │
        │ 2          ┆ 0      ┆ 1      ┆ 2021-01-04 │
        └────────────┴────────┴────────┴────────────┘
        >>> value_df.collect()
        shape: (3, 4)
        ┌────────────┬────────────┬─────────┬─────────┐
        │ patient_id ┆ timestamp  ┆ value/A ┆ value/B │
        │ ---        ┆ ---        ┆ ---     ┆ ---     │
        │ i64        ┆ str        ┆ f64     ┆ f64     │
        ╞════════════╪════════════╪═════════╪═════════╡
        │ 1          ┆ 2021-01-01 ┆ 1.5     ┆ null    │
        │ 1          ┆ 2020-01-01 ┆ null    ┆ 2.0     │
        │ 2          ┆ 2021-01-04 ┆ null    ┆ 2.0     │
        └────────────┴────────────┴─────────┴─────────┘
    """
    ts_columns = [c for c in feature_columns if not c.startswith("static")]
    ts_shard_df = shard_df.filter(pl.col("timestamp").is_not_null())
    return summarize_dynamic_measurements(ts_columns, ts_shard_df)
