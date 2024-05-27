from collections.abc import Callable

import polars as pl
import polars.selectors as cs

from MEDS_tabular_automl.utils import DF_T

VALID_AGGREGATIONS = [
    "code/count",
    "value/count",
    "value/has_values_count",
    "value/sum",
    "value/sum_sqd",
    "value/min",
    "value/max",
    "value/first",
]


def time_aggd_col_alias_fntr(window_size: str, agg: str) -> Callable[[str], str]:
    if agg is None:
        raise ValueError("Aggregation type 'agg' must be provided")

    def f(c: str) -> str:
        return "/".join([window_size] + c.split("/") + [agg])

    return f


def get_agg_pl_expr(window_size: str, agg: str):
    code_cols = cs.starts_with("code/")
    value_cols = cs.starts_with("value/")
    if window_size == "full":
        match agg:
            case "code/count":
                return code_cols.cumsum().map_alias(time_aggd_col_alias_fntr(window_size, "count"))
            case "value/count":
                return (
                    value_cols.is_not_null()
                    .cumsum()
                    .map_alias(time_aggd_col_alias_fntr(window_size, "count"))
                )
            case "value/has_values_count":
                return (
                    (value_cols.is_not_null() & value_cols.is_not_nan())
                    .cumsum()
                    .map_alias(time_aggd_col_alias_fntr(window_size, "has_values_count"))
                )
            case "value/sum":
                return value_cols.cumsum().map_alias(time_aggd_col_alias_fntr(window_size, "sum"))
            case "value/sum_sqd":
                return (value_cols**2).cumsum().map_alias(time_aggd_col_alias_fntr(window_size, "sum_sqd"))
            case "value/min":
                value_cols.cummin().map_alias(time_aggd_col_alias_fntr(window_size, "min"))
            case "value/max":
                value_cols.cummax().map_alias(time_aggd_col_alias_fntr(window_size, "max"))
            case _:
                raise ValueError(f"Invalid aggregation `{agg}` for window_size `{window_size}`")
    else:
        match agg:
            case "code/count":
                return code_cols.sum().map_alias(time_aggd_col_alias_fntr(window_size, "count"))
            case "value/count":
                return (
                    value_cols.is_not_null().sum().map_alias(time_aggd_col_alias_fntr(window_size, "count"))
                )
            case "value/has_values_count":
                return (
                    (value_cols.is_not_null() & value_cols.is_not_nan())
                    .sum()
                    .map_alias(time_aggd_col_alias_fntr(window_size, "has_values_count"))
                )
            case "value/sum":
                return value_cols.sum().map_alias(time_aggd_col_alias_fntr(window_size, "sum"))
            case "value/sum_sqd":
                return (value_cols**2).sum().map_alias(time_aggd_col_alias_fntr(window_size, "sum_sqd"))
            case "value/min":
                value_cols.min().map_alias(time_aggd_col_alias_fntr(window_size, "min"))
            case "value/max":
                value_cols.max().map_alias(time_aggd_col_alias_fntr(window_size, "max"))
            case _:
                raise ValueError(f"Invalid aggregation `{agg}` for window_size `{window_size}`")


def _generate_summary(df: DF_T, window_size: str, agg: str) -> pl.LazyFrame:
    """Generate a summary of the data frame for a given window size and aggregation.

    Args:
    - df (DF_T): The data frame to summarize.
    - window_size (str): The window size to use for the summary.
    - agg (str): The aggregation to apply to the data frame.

    Returns:
    - pl.LazyFrame: The summarized data frame.

    Expect:
        >>> from datetime import date
        >>> code_df = pl.DataFrame({"patient_id": [1, 1, 1, 2],
        ...     "code/A": [1, 1, 0, 0],
        ...     "code/B": [0, 0, 1, 1],
        ...     "timestamp": [date(2021, 1, 1), date(2021, 1, 2),date(2020, 1, 3), date(2021, 1, 4)],
        ...     }).lazy()
        >>> _generate_summary(code_df.lazy(), "full", "code/count"
        ...     ).collect().sort(["patient_id", "timestamp"])
        shape: (4, 4)
        ┌────────────┬────────────┬───────────────────┬───────────────────┐
        │ patient_id ┆ timestamp  ┆ full/code/A/count ┆ full/code/B/count │
        │ ---        ┆ ---        ┆ ---               ┆ ---               │
        │ i64        ┆ date       ┆ i64               ┆ i64               │
        ╞════════════╪════════════╪═══════════════════╪═══════════════════╡
        │ 1          ┆ 2020-01-03 ┆ 2                 ┆ 1                 │
        │ 1          ┆ 2021-01-01 ┆ 1                 ┆ 0                 │
        │ 1          ┆ 2021-01-02 ┆ 2                 ┆ 0                 │
        │ 2          ┆ 2021-01-04 ┆ 0                 ┆ 1                 │
        └────────────┴────────────┴───────────────────┴───────────────────┘
        >>> value_df = pl.DataFrame({"patient_id": [1, 1, 1, 2],
        ...     "timestamp": [date(2021, 1, 1), date(2021, 1, 2),
        ...                   date(2020, 1, 3), date(2021, 1, 4)],
        ...     "value/A": [1, 2, 3, None],
        ...     "value/B": [None, None, None, 4.0],})
        >>> _generate_summary(value_df.lazy(), "full", "value/sum").collect().sort(
        ...     ["patient_id", "timestamp"])
        shape: (4, 4)
        ┌────────────┬────────────┬──────────────────┬──────────────────┐
        │ patient_id ┆ timestamp  ┆ full/value/A/sum ┆ full/value/B/sum │
        │ ---        ┆ ---        ┆ ---              ┆ ---              │
        │ i64        ┆ date       ┆ i64              ┆ f64              │
        ╞════════════╪════════════╪══════════════════╪══════════════════╡
        │ 1          ┆ 2020-01-03 ┆ 6                ┆ null             │
        │ 1          ┆ 2021-01-01 ┆ 1                ┆ null             │
        │ 1          ┆ 2021-01-02 ┆ 3                ┆ null             │
        │ 2          ┆ 2021-01-04 ┆ null             ┆ 4.0              │
        └────────────┴────────────┴──────────────────┴──────────────────┘
        >>> _generate_summary(value_df.lazy(), "1d", "value/count").collect().sort(
        ...     ["patient_id", "timestamp"])
        shape: (4, 4)
        ┌────────────┬────────────┬──────────────────┬──────────────────┐
        │ patient_id ┆ timestamp  ┆ 1d/value/A/count ┆ 1d/value/B/count │
        │ ---        ┆ ---        ┆ ---              ┆ ---              │
        │ i64        ┆ date       ┆ u32              ┆ u32              │
        ╞════════════╪════════════╪══════════════════╪══════════════════╡
        │ 1          ┆ 2020-01-03 ┆ 1                ┆ 0                │
        │ 1          ┆ 2021-01-01 ┆ 1                ┆ 0                │
        │ 1          ┆ 2021-01-02 ┆ 1                ┆ 0                │
        │ 2          ┆ 2021-01-04 ┆ 0                ┆ 1                │
        └────────────┴────────────┴──────────────────┴──────────────────┘
    """
    if agg not in VALID_AGGREGATIONS:
        raise ValueError(f"Invalid aggregation: {agg}. Valid options are: {VALID_AGGREGATIONS}")
    if agg.split("/")[0] not in [c.split("/")[0] for c in df.columns]:
        raise ValueError(f"DataFrame is invalid, no column with prefix: `{agg.split('/')[0]}`")

    if window_size == "full":
        out_df = df.groupby("patient_id").agg(
            "timestamp",
            get_agg_pl_expr(window_size, agg),
        )
        out_df = out_df.explode(*[c for c in out_df.columns if c != "patient_id"])
    else:
        out_df = (
            df.sort(["patient_id", "timestamp"])
            .groupby_rolling(
                index_column="timestamp",
                by="patient_id",
                period=window_size,
            )
            .agg(
                get_agg_pl_expr(window_size, agg),
            )
        )

    return out_df


def generate_summary(
    feature_columns: list[str], dfs: list[pl.LazyFrame], window_sizes: list[str], aggregations: list[str]
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
        >>> value_df = pl.DataFrame({"patient_id": [1, 1, 1, 2],
        ...     "timestamp": [date(2021, 1, 1), date(2021, 1, 2),date(2020, 1, 3), date(2021, 1, 4)],
        ...     "value/A": [1, 2, 3, None],
        ...     "value/B": [None, None, None, 4.0],})
        >>> code_df = pl.DataFrame({"patient_id": [1, 1, 1, 2],
        ...     "code/A": [1, 1, 0, 0],
        ...     "code/B": [0, 0, 1, 1],
        ...     "timestamp": [date(2021, 1, 1), date(2021, 1, 2),date(2020, 1, 3), date(2021, 1, 5)],
        ...     }).lazy()
        >>> feature_columns = ["code/A", "code/B", "value/A", "value/B"]
        >>> aggregations = ["code/count", "value/sum"]
        >>> window_sizes = ["full", "1d"]
        >>> out_df = generate_summary(feature_columns, [value_df.lazy(), code_df.lazy()],
        ...     window_sizes, aggregations).collect().sort(["patient_id", "timestamp"])
        >>> print(out_df.shape)
        (5, 10)
        >>> for c in out_df.columns: print(c)
        patient_id
        timestamp
        1d/code/A/count
        1d/code/B/count
        1d/value/A/sum
        1d/value/B/sum
        full/code/A/count
        full/code/B/count
        full/value/A/sum
        full/value/B/sum
    """
    final_columns = []
    out_dfs = []
    # Generate summaries for each window size and aggregation
    for window_size in window_sizes:
        for agg in aggregations:
            code_type, agg_name = agg.split("/")
            final_columns.extend(
                [f"{window_size}/{c}/{agg_name}" for c in feature_columns if c.startswith(code_type)]
            )
            for df in dfs:
                if agg.split("/")[0] in [c.split("/")[0] for c in df.columns]:
                    out_df = _generate_summary(df, window_size, agg)
                    out_dfs.append(out_df)

    final_columns = sorted(final_columns)
    # Combine all dataframes using successive joins
    result_df = out_dfs[0]
    for df in out_dfs[1:]:
        result_df = result_df.join(df, on=["patient_id", "timestamp"], how="outer", coalesce=True)

    # Add in missing feature columns with default values
    existing_columns = result_df.columns
    for column in final_columns:
        if column not in existing_columns:
            result_df = result_df.with_columns(pl.lit(None).alias(column))
    result_df = result_df.select(pl.col(*["patient_id", "timestamp"], *final_columns))
    return result_df
