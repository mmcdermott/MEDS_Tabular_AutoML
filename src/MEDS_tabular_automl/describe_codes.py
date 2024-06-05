from collections.abc import Mapping
from pathlib import Path

import polars as pl
from omegaconf import DictConfig, OmegaConf

from MEDS_tabular_automl.utils import DF_T, get_feature_names


def convert_to_df(freq_dict):
    return pl.DataFrame([[col, freq] for col, freq in freq_dict.items()], schema=["code", "count"])


def compute_feature_frequencies(cfg: DictConfig, shard_df: DF_T) -> list[str]:
    """Generates a list of feature column names from the data within each shard based on specified
    configurations.

    Parameters:
    - cfg (DictConfig): Configuration dictionary specifying how features should be evaluated and aggregated.
    - split_to_shard_df (dict): A dictionary of DataFrames, divided by data split (e.g., 'train', 'test').

    Returns:
    - tuple[list[str], dict]: A tuple containing a list of feature columns and a dictionary of code properties
        identified during the evaluation.

    This function evaluates the properties of codes within training data and applies configured
    aggregations to generate a comprehensive list of feature columns for modeling purposes.
    Examples:
    # >>> import polars as pl
    # >>> data = {'code': ['A', 'A', 'B', 'B', 'C', 'C', 'C'],
    # ...         'timestamp': [None, '2021-01-01', None, None, '2021-01-03', '2021-01-04', None],
    # ...         'numerical_value': [1, None, 2, 2, None, None, 3]}
    # >>> df = pl.DataFrame(data).lazy()
    # >>> aggs = ['value/sum', 'code/count']
    # >>> compute_feature_frequencies(aggs, df)
    # ['A/code', 'A/value', 'C/code', 'C/value']
    """
    static_df = shard_df.filter(
        pl.col("patient_id").is_not_null() & pl.col("code").is_not_null() & pl.col("timestamp").is_null()
    )
    static_code_freqs_df = static_df.group_by("code").agg(pl.count("code").alias("count")).collect()
    static_code_freqs = {
        row["code"] + "/static/present": row["count"] for row in static_code_freqs_df.iter_rows(named=True)
    }

    static_value_df = static_df.filter(pl.col("numerical_value").is_not_null())
    static_value_freqs_df = (
        static_value_df.group_by("code").agg(pl.count("numerical_value").alias("count")).collect()
    )
    static_value_freqs = {
        row["code"] + "/static/first": row["count"] for row in static_value_freqs_df.iter_rows(named=True)
    }

    ts_df = shard_df.filter(
        pl.col("patient_id").is_not_null() & pl.col("code").is_not_null() & pl.col("timestamp").is_not_null()
    )
    code_freqs_df = ts_df.group_by("code").agg(pl.count("code").alias("count")).collect()
    code_freqs = {row["code"] + "/code": row["count"] for row in code_freqs_df.iter_rows(named=True)}

    value_df = ts_df.filter(pl.col("numerical_value").is_not_null())
    value_freqs_df = value_df.group_by("code").agg(pl.count("numerical_value").alias("count")).collect()
    value_freqs = {row["code"] + "/value": row["count"] for row in value_freqs_df.iter_rows(named=True)}

    combined_freqs = {**static_code_freqs, **static_value_freqs, **code_freqs, **value_freqs}
    return convert_to_df(combined_freqs)


def convert_to_freq_dict(df: pl.LazyFrame) -> dict:
    """Converts a DataFrame to a dictionary of frequencies.

    This function converts a DataFrame to a dictionary of frequencies, where the keys are the
    column names and the values are dictionaries of code frequencies.

    Args:
    - df (pl.DataFrame): The DataFrame to be converted.

    Returns:
    - dict: A dictionary of frequencies, where the keys are the column names and the values are
      dictionaries of code frequencies.

    Example:
    # >>> import polars as pl
    # >>> df = pl.DataFrame({
    # ...     "code": [1, 2, 3, 4, 5],
    # ...     "value": [10, 20, 30, 40, 50]
    # ... })
    # >>> convert_to_freq_dict(df)
    # {'code': {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}, 'value': {10: 1, 20: 1, 30: 1, 40: 1, 50: 1}}
    """
    if not df.columns == ["code", "count"]:
        raise ValueError(f"DataFrame must have columns 'code' and 'count', but has columns {df.columns}!")
    return dict(df.collect().iter_rows())


def get_feature_columns(fp):
    return sorted(list(convert_to_freq_dict(pl.scan_parquet(fp)).keys()))


def get_feature_freqs(fp):
    return convert_to_freq_dict(pl.scan_parquet(fp))


def filter_to_codes(
    allowed_codes: list[str] | None,
    min_code_inclusion_frequency: int,
    code_metadata_fp: Path,
):
    """Returns intersection of allowed codes if they are specified, and filters to codes based on inclusion
    frequency."""
    if allowed_codes is None:
        allowed_codes = get_feature_columns(code_metadata_fp)
    feature_freqs = get_feature_freqs(code_metadata_fp)

    code_freqs = {
        code: freq for code, freq in feature_freqs.items() if (
            freq >= min_code_inclusion_frequency and code in set(allowed_codes)
            )
    }
    return sorted([code for code, freq in code_freqs.items() if freq >= min_code_inclusion_frequency])


OmegaConf.register_new_resolver("filter_to_codes", filter_to_codes)


def clear_code_aggregation_suffix(code):
    if code.endswith("/code"):
        return code[:-5]
    elif code.endswith("/value"):
        return code[:-6]
    elif code.endswith("/static/present"):
        return code[:-15]
    elif code.endswith("/static/first"):
        return code[:-13]


def filter_parquet(fp, allowed_codes: list[str]):
    """Loads Parquet with Polars and filters to allowed codes.

    Args:
        fp: Path to the Meds cohort shard
        allowed_codes: List of codes to filter to.

    Expect:
    >>> from tempfile import NamedTemporaryFile
    >>> fp = NamedTemporaryFile()
    >>> pl.DataFrame({
    ...     "code": ["A", "A", "A", "A", "D", "D", "E", "E"],
    ...     "timestamp": [None, None, "2021-01-01", "2021-01-01", None, None, "2021-01-03", "2021-01-04"],
    ...     "numerical_value": [1, None, 2, 2, None, 5, None, 3]
    ... }).write_parquet(fp.name)
    >>> filter_parquet(fp.name, ["A/code", "D/static/present", "E/code", "E/value"]).collect()
    shape: (6, 3)
    ┌──────┬────────────┬─────────────────┐
    │ code ┆ timestamp  ┆ numerical_value │
    │ ---  ┆ ---        ┆ ---             │
    │ str  ┆ str        ┆ i64             │
    ╞══════╪════════════╪═════════════════╡
    │ A    ┆ 2021-01-01 ┆ null            │
    │ A    ┆ 2021-01-01 ┆ null            │
    │ D    ┆ null       ┆ null            │
    │ D    ┆ null       ┆ null            │
    │ E    ┆ 2021-01-03 ┆ null            │
    │ E    ┆ 2021-01-04 ┆ 3               │
    └──────┴────────────┴─────────────────┘
    >>> fp.close()
    """
    df = pl.scan_parquet(fp)
    # Drop values that are rare
    # Drop Rare Static Codes
    static_present_feature_columns = [
        clear_code_aggregation_suffix(each) for each in get_feature_names("static/present", allowed_codes)
    ]
    static_first_feature_columns = [
        clear_code_aggregation_suffix(each) for each in get_feature_names("static/first", allowed_codes)
    ]
    code_feature_columns = [
        clear_code_aggregation_suffix(each) for each in get_feature_names("code/count", allowed_codes)
    ]
    value_feature_columns = [
        clear_code_aggregation_suffix(each) for each in get_feature_names("value/sum", allowed_codes)
    ]

    is_static_code = pl.col("timestamp").is_null()
    is_numeric_code = pl.col("numerical_value").is_not_null()
    rare_static_code = is_static_code & ~pl.col("code").is_in(static_present_feature_columns)
    rare_ts_code = ~is_static_code & ~pl.col("code").is_in(code_feature_columns)
    rare_ts_value = ~is_static_code & ~pl.col("code").is_in(value_feature_columns) & is_numeric_code
    rare_static_value = is_static_code & ~pl.col("code").is_in(static_first_feature_columns) & is_numeric_code

    # Remove rare numeric values by converting them to null
    df = df.with_columns(
        pl.when(rare_static_value | rare_ts_value)
        .then(None)
        .otherwise(pl.col("numerical_value"))
        .alias("numerical_value")
    )
    # Drop rows with rare codes
    df = df.filter(~(rare_static_code | rare_ts_code))
    return df
