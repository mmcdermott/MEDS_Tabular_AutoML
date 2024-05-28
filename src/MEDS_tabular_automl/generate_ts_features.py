import polars as pl

from MEDS_tabular_automl.utils import DF_T, ROW_IDX_NAME


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
    >>> pivot_df = summarize_dynamic_measurements(ts_columns, df)
    >>> pivot_df.collect()
    shape: (4, 7)
    ┌───────────┬────────────┬────────────┬─────────┬─────────┬────────┬────────┐
    │ __row_idx ┆ patient_id ┆ timestamp  ┆ A/value ┆ B/value ┆ A/code ┆ B/code │
    │ ---       ┆ ---        ┆ ---        ┆ ---     ┆ ---     ┆ ---    ┆ ---    │
    │ u32       ┆ i64        ┆ str        ┆ i64     ┆ i64     ┆ bool   ┆ bool   │
    ╞═══════════╪════════════╪════════════╪═════════╪═════════╪════════╪════════╡
    │ 0         ┆ 1          ┆ 2021-01-01 ┆ 1       ┆ null    ┆ true   ┆ null   │
    │ 1         ┆ 1          ┆ 2021-01-01 ┆ 2       ┆ null    ┆ true   ┆ null   │
    │ 2         ┆ 1          ┆ 2020-01-01 ┆ null    ┆ 2       ┆ null   ┆ true   │
    │ 3         ┆ 2          ┆ 2021-01-04 ┆ null    ┆ 2       ┆ null   ┆ true   │
    └───────────┴────────────┴────────────┴─────────┴─────────┴────────┴────────┘
    """
    df = df.with_row_index(ROW_IDX_NAME)
    id_cols = [ROW_IDX_NAME, "patient_id", "timestamp"]
    pivot_df = (
        df.select(*id_cols, "code", "numerical_value")
        .with_columns(pl.lit(True).alias("__indicator"))
        .collect()
        .pivot(
            index=id_cols,  # add row index and set agg to None
            columns=["code"],
            values=["numerical_value", "__indicator"],
            aggregate_function=None,  # TODO round up counts so they are binary
            separator="/",
        )
        .lazy()
    )

    def rename(c):
        """Remove value and column prefix."""
        numerical_val_col_name = "numerical_value"
        indicator_col_name = "__indicator"
        if c.startswith(numerical_val_col_name):
            return f"{c[len(numerical_val_col_name)+6:]}/value"
        elif c.startswith(indicator_col_name):
            return f"{c[len(indicator_col_name)+6:]}/code"
        else:
            return c

    pivot_df = pivot_df.rename(rename)
    return pivot_df


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
        >>> feature_columns = ['A', 'B', 'C', "A/static/present"]
        >>> data = {'patient_id': [1, 1, 1, 2, 2, 2],
        ...         'code': ['A', 'A', 'B', 'B', 'C', 'C'],
        ...         'timestamp': ['2021-01-01', '2021-01-01', '2020-01-01', '2021-01-04', None, None],
        ...         'numerical_value': [1, 2, 2, 2, 3, 4]}
        >>> df = pl.DataFrame(data).lazy()
        >>> pivot_df = get_flat_ts_rep(feature_columns, df)
        >>> pivot_df.collect()
        shape: (4, 7)
        ┌───────────┬────────────┬────────────┬─────────┬─────────┬────────┬────────┐
        │ __row_idx ┆ patient_id ┆ timestamp  ┆ A/value ┆ B/value ┆ A/code ┆ B/code │
        │ ---       ┆ ---        ┆ ---        ┆ ---     ┆ ---     ┆ ---    ┆ ---    │
        │ u32       ┆ i64        ┆ str        ┆ i64     ┆ i64     ┆ bool   ┆ bool   │
        ╞═══════════╪════════════╪════════════╪═════════╪═════════╪════════╪════════╡
        │ 0         ┆ 1          ┆ 2021-01-01 ┆ 1       ┆ null    ┆ true   ┆ null   │
        │ 1         ┆ 1          ┆ 2021-01-01 ┆ 2       ┆ null    ┆ true   ┆ null   │
        │ 2         ┆ 1          ┆ 2020-01-01 ┆ null    ┆ 2       ┆ null   ┆ true   │
        │ 3         ┆ 2          ┆ 2021-01-04 ┆ null    ┆ 2       ┆ null   ┆ true   │
        └───────────┴────────────┴────────────┴─────────┴─────────┴────────┴────────┘
    """

    def is_static(c):
        return len(c.split("/")) > 2 and c.split("/")[-2] == "static"

    ts_columns = [c for c in feature_columns if not is_static(c)]
    ts_shard_df = shard_df.filter(pl.col("timestamp").is_not_null())
    return summarize_dynamic_measurements(ts_columns, ts_shard_df)
