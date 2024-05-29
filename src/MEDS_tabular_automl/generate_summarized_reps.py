from collections.abc import Callable

import pandas as pd
import polars as pl
import polars.selectors as cs

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


def _generate_summary(df: pd.DataFrame, window_size: str, agg: str) -> pl.LazyFrame:
    """Generate a summary of the data frame for a given window size and aggregation.

    Args:
    - df (DF_T): The data frame to summarize.
    - window_size (str): The window size to use for the summary.
    - agg (str): The aggregation to apply to the data frame.

    Returns:
    - pl.LazyFrame: The summarized data frame.

    Expect:
        >>> from MEDS_tabular_automl.generate_ts_features import get_flat_ts_rep
        >>> feature_columns = ['A/value/sum', 'A/code/count', 'B/value/sum', 'B/code/count',
        ...                    "C/value/sum", "C/code/count", "A/static/present"]
        >>> data = {'patient_id': [1, 1, 1, 2, 2, 2],
        ...         'code': ['A', 'A', 'B', 'B', 'C', 'C'],
        ...         'timestamp': ['2021-01-01', '2021-01-01', '2020-01-01', '2021-01-04', None, None],
        ...         'numerical_value': [1, 2, 2, 2, 3, 4]}
        >>> df = pl.DataFrame(data).lazy()
        >>> pivot_df = get_flat_ts_rep(feature_columns, df)
        >>> pivot_df
           patient_id   timestamp  A/value  B/value  C/value  A/code  B/code  C/code
        0           1  2021-01-01        1        0        0       1       0       0
        1           1  2021-01-01        2        0        0       1       0       0
        2           1  2020-01-01        0        2        0       0       1       0
        3           2  2021-01-04        0        2        0       0       1       0
        >>> _generate_summary(pivot_df, "full", "value/sum")
          patient_id   timestamp  A/value/sum  B/value/sum  C/value/sum
    """
    if agg not in VALID_AGGREGATIONS:
        raise ValueError(f"Invalid aggregation: {agg}. Valid options are: {VALID_AGGREGATIONS}")
    if window_size == "full":
        out_df = df.groupby("patient_id").agg(
            "timestamp",
            get_agg_pl_expr(window_size, agg),
        )
        out_df = out_df.explode(*[c for c in out_df.columns if c != "patient_id"])
    else:
        out_df = df.rolling(
            index_column="timestamp",
            by="patient_id",
            period=window_size,
        ).agg(
            get_agg_pl_expr(window_size, agg),
        )
    return out_df


def generate_summary(
    feature_columns: list[str], df: pd.DataFrame, window_sizes: list[str], aggregations: list[str]
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
    #     >>> from datetime import date
    #     >>> wide_df = pd.DataFrame({"patient_id": [1, 1, 1, 2],
    #     ...     "A/code": [1, 1, 0, 0],
    #     ...     "B/code": [0, 0, 1, 1],
    #     ...     "A/value": [1, 2, 3, None],
    #     ...     "B/value": [None, None, None, 4.0],
    #     ...     "timestamp": [date(2021, 1, 1), date(2021, 1, 1),date(2020, 1, 3), date(2021, 1, 4)],
    #     ...     }).lazy()
    #     >>> feature_columns = ["A/code", "B/code", "A/value", "B/value"]
    #     >>> aggregations = ["code/count", "value/sum"]
    #     >>> window_sizes = ["full", "1d"]
    #     >>> generate_summary(feature_columns, wide_df.lazy(), window_sizes, aggregations).collect()
    """
    df = df.sort(["patient_id", "timestamp"])
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
    result_df = pl.concat(out_dfs, how="align")
    # Add in missing feature columns with default values
    missing_columns = [col for col in final_columns if col not in result_df.columns]
    result_df = result_df.with_columns([pl.lit(None).alias(col) for col in missing_columns])
    result_df = result_df.select(pl.col(*["patient_id", "timestamp"], *final_columns))
    return result_df
