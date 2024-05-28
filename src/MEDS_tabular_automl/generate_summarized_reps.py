from collections.abc import Callable

import polars as pl
import polars.selectors as cs

from MEDS_tabular_automl.utils import DF_T, ROW_IDX_NAME

CODE_AGGREGATIONS = [
    "code/count",
]

VALUE_AGGREGATIONS = [
    "value/count",
    "value/has_values_count",
    "value/sum",
    "value/sum_sqd",
    "value/min",
    "value/max",
]

VALID_AGGREGATIONS = CODE_AGGREGATIONS + VALUE_AGGREGATIONS


def time_aggd_col_alias_fntr(window_size: str, agg: str) -> Callable[[str], str]:
    if agg is None:
        raise ValueError("Aggregation type 'agg' must be provided")

    def f(c: str) -> str:
        return "/".join([window_size] + c.split("/") + [agg])

    return f


def get_agg_pl_expr(window_size: str, agg: str):
    code_cols = cs.ends_with("code")
    value_cols = cs.ends_with("value")
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
            case "value/sum":
                return value_cols.cumsum().map_alias(time_aggd_col_alias_fntr(window_size, "sum"))
            case "value/sum_sqd":
                return (value_cols**2).cumsum().map_alias(time_aggd_col_alias_fntr(window_size, "sum_sqd"))
            case "value/min":
                value_cols.cummin().map_alias(time_aggd_col_alias_fntr(window_size, "min"))
            case "value/max":
                value_cols.cummax().map_alias(time_aggd_col_alias_fntr(window_size, "max"))
            case _:
                raise ValueError(
                    f"Invalid aggregation '{agg}' provided for window_size '{window_size}'."
                    f" Please choose from the valid options: {VALID_AGGREGATIONS}"
                )
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
        >>> pivot_df = pl.DataFrame({"patient_id": [1, 1, 1, 2],
        ...     "A/code": [True, True, None, None],
        ...     "B/code": [None, None, True, True],
        ...     "A/value": [1, 2, 3, None],
        ...     "B/value": [None, None, None, 4.0],
        ...     "timestamp": [date(2020, 1, 1), date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3)],
        ...     }).lazy().with_row_index(ROW_IDX_NAME)
        >>> _generate_summary(pivot_df.lazy(), "2d", "code/count"
        ...     ).collect().drop(ROW_IDX_NAME).sort(["patient_id", "timestamp"])
        shape: (4, 4)
        ┌────────────┬────────────┬───────────────────┬───────────────────┐
        │ patient_id ┆ timestamp  ┆ full/A/code/count ┆ full/B/code/count │
        │ ---        ┆ ---        ┆ ---               ┆ ---               │
        │ i64        ┆ date       ┆ i64               ┆ i64               │
        ╞════════════╪════════════╪═══════════════════╪═══════════════════╡
        │ 1          ┆ 2020-01-03 ┆ 2                 ┆ 1                 │
        │ 1          ┆ 2021-01-01 ┆ 1                 ┆ 0                 │
        │ 1          ┆ 2021-01-02 ┆ 2                 ┆ 0                 │
        │ 2          ┆ 2021-01-04 ┆ 0                 ┆ 1                 │
        └────────────┴────────────┴───────────────────┴───────────────────┘
        >>> _generate_summary(pivot_df.lazy(), "full", "value/sum").collect().drop(ROW_IDX_NAME).sort(
        ...     ["patient_id", "timestamp"])
        shape: (4, 4)
        ┌────────────┬────────────┬──────────────────┬──────────────────┐
        │ patient_id ┆ timestamp  ┆ 1d/A/value/sum   ┆ 1d/B/value/sum   │
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
    id_cols = [ROW_IDX_NAME, "patient_id"]
    if window_size == "full":
        out_df = df.group_by(id_cols).agg(
            "timestamp",
            get_agg_pl_expr(window_size, agg),
        )
        out_df = out_df.explode(*[c for c in out_df.columns if c not in id_cols])
    else:
        out_df = df.rolling(
            index_column="timestamp",
            by=id_cols,
            period=window_size,
        ).agg(
            get_agg_pl_expr(window_size, agg),
        )
    return out_df


def generate_summary(
    feature_columns: list[str], df: pl.LazyFrame, window_sizes: list[str], aggregations: list[str]
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
        >>> pivot_df = pl.DataFrame({"patient_id": [1, 1, 1, 2],
        ...     "A/code": [1, 1, 0, 0],
        ...     "B/code": [0, 0, 1, 1],
        ...     "A/value": [1, 2, 3, None],
        ...     "B/value": [None, None, None, 4.0],
        ...     "timestamp": [date(2021, 1, 1), date(2021, 1, 1),date(2020, 1, 3), date(2021, 1, 4)],
        ...     }).lazy()
        >>> feature_columns = ["A/code", "B/code", "A/value", "B/value"]
        >>> aggregations = ["code/count", "value/sum"]
        >>> window_sizes = ["full", "1d"]
        >>> out_df = generate_summary(feature_columns, pivot_df.lazy(),
        ...     window_sizes, aggregations).collect().sort(["patient_id", "timestamp"])
        >>> print(out_df.shape)
        (4, 10)
        >>> for c in sorted(out_df.columns): print(c)
        1d/A/code/count
        1d/A/value/sum
        1d/B/code/count
        1d/B/value/sum
        full/A/code/count
        full/A/value/sum
        full/B/code/count
        full/B/value/sum
        patient_id
        timestamp
    """
    df = df.sort(["patient_id", "timestamp"]).with_row_index(ROW_IDX_NAME)
    final_columns = []
    out_dfs = []
    # Generate summaries for each window size and aggregation
    for window_size in window_sizes:
        for agg in aggregations:
            code_type, agg_name = agg.split("/")
            final_columns.extend(
                [f"{window_size}/{c}/{agg_name}" for c in feature_columns if c.endswith(code_type)]
            )
            # only iterate through code_types that exist in the dataframe columns
            if any([c.endswith(code_type) for c in df.columns]):
                timestamp_dtype = df.dtypes[df.columns.index("timestamp")]
                assert timestamp_dtype in [
                    pl.Datetime,
                    pl.Date,
                ], f"timestamp must be of type Date, but is {timestamp_dtype}"
                out_df = _generate_summary(df, window_size, agg)
                out_dfs.append(out_df)

    final_columns = sorted(final_columns)
    # Combine all dataframes using successive joins
    result_df = pl.concat(out_dfs, how="align").drop(ROW_IDX_NAME)
    # Add in missing feature columns with default values
    missing_columns = [col for col in final_columns if col not in result_df.columns]
    result_df = result_df.with_columns([pl.lit(None).alias(col) for col in missing_columns])
    result_df = result_df.select(pl.col(*["patient_id", "timestamp"], *final_columns))
    return result_df
