import polars as pl
from omegaconf import DictConfig

from MEDS_tabular_automl.utils import DF_T


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
    >>> import polars as pl
    >>> data = {'code': ['A', 'A', 'B', 'B', 'C', 'C', 'C'],
    ...         'timestamp': [None, '2021-01-01', None, None, '2021-01-03', '2021-01-04', None],
    ...         'numerical_value': [1, None, 2, 2, None, None, 3]}
    >>> df = pl.DataFrame(data).lazy()
    >>> aggs = ['value/sum', 'code/count']
    >>> get_ts_feature_cols(aggs, df)
    ['A/code', 'A/value', 'C/code', 'C/value']
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
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     "code": [1, 2, 3, 4, 5],
    ...     "value": [10, 20, 30, 40, 50]
    ... })
    >>> convert_to_freq_dict(df)
    {'code': {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}, 'value': {10: 1, 20: 1, 30: 1, 40: 1, 50: 1}}
    """
    if not df.columns == ["code", "count"]:
        raise ValueError(f"DataFrame must have columns 'code' and 'count', but has columns {df.columns}!")
    return dict(df.collect().iter_rows())


def get_feature_columns(fp):
    return sorted(list(convert_to_freq_dict(pl.scan_parquet(fp))))


def get_feature_freqs(fp):
    return convert_to_freq_dict(pl.scan_parquet(fp))
