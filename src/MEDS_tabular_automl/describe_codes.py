from pathlib import Path

import polars as pl

from MEDS_tabular_automl.utils import get_feature_names


def convert_to_df(freq_dict: dict[str, int]) -> pl.DataFrame:
    """Converts a dictionary of code frequencies to a Polars DataFrame.

    Args:
        freq_dict: A dictionary with code features and their respective frequencies.

    Returns:
        A DataFrame with two columns, "code" and "count".

    TODOs:
        - Eliminate this function and just use a DataFrame throughout. See #14
        - Use categorical types for `code` instead of strings.

    Examples:
        >>> convert_to_df({"A": 1, "B": 2, "C": 3})
        shape: (3, 2)
        ┌──────┬───────┐
        │ code ┆ count │
        │ ---  ┆ ---   │
        │ str  ┆ i64   │
        ╞══════╪═══════╡
        │ A    ┆ 1     │
        │ B    ┆ 2     │
        │ C    ┆ 3     │
        └──────┴───────┘
    """
    return pl.DataFrame([[col, freq] for col, freq in freq_dict.items()], schema=["code", "count"])


def convert_to_freq_dict(df: pl.LazyFrame) -> dict[str, dict[int, int]]:
    """Converts a DataFrame to a dictionary of frequencies.

    Args:
        df: The DataFrame to be converted.

    Returns:
        A dictionary where keys are column names and values are
        dictionaries of code frequencies.

    Raises:
        ValueError: If the DataFrame does not have the expected columns "code" and "count".

    TODOs:
        - Eliminate this function and just use a DataFrame throughout. See #14

    Example:
        >>> import polars as pl
        >>> data = pl.DataFrame({"code": [1, 2, 3, 4, 5], "count": [10, 20, 30, 40, 50]}).lazy()
        >>> convert_to_freq_dict(data)
        {1: 10, 2: 20, 3: 30, 4: 40, 5: 50}
        >>> convert_to_freq_dict(pl.DataFrame({"code": ["A", "B", "C"], "value": [1, 2, 3]}).lazy())
        Traceback (most recent call last):
            ...
        ValueError: DataFrame must have columns 'code' and 'count', but has columns ['code', 'value']!
    """
    if not df.columns == ["code", "count"]:
        raise ValueError(f"DataFrame must have columns 'code' and 'count', but has columns {df.columns}!")
    return dict(df.collect().iter_rows())


def compute_feature_frequencies(shard_df: pl.LazyFrame) -> pl.DataFrame:
    """Generates a DataFrame containing the frequencies of codes and numerical values under different
    aggregations by computing frequency counts for certain attributes and organizing the results into specific
    categories based on the dataset's features.

    Args:
        shard_df: A DataFrame containing the data to be analyzed and split (e.g., 'train', 'test').

    Returns:
        A tuple containing a list of feature columns and a dictionary of code properties identified
        during the evaluation.

    Examples:
        >>> from datetime import datetime
        >>> data = pl.DataFrame({
        ...     'subject_id': [1, 1, 2, 2, 3, 3, 3],
        ...     'code': ['A', 'A', 'B', 'B', 'C', 'C', 'C'],
        ...     'time': [
        ...         None,
        ...         datetime(2021, 1, 1),
        ...         None,
        ...         None,
        ...         datetime(2021, 1, 3),
        ...         datetime(2021, 1, 4),
        ...         None
        ...     ],
        ...     'numeric_value': [1, None, 2, 2, None, None, 3]
        ... }).lazy()
        >>> assert (
        ...     convert_to_freq_dict(compute_feature_frequencies(data).lazy()) == {
        ...         'B/static/present': 2, 'C/static/present': 1, 'A/static/present': 1, 'B/static/first': 2,
        ...         'C/static/first': 1, 'A/static/first': 1, 'A/code': 1, 'C/code': 2
        ...     }
        ... )
    """
    static_df = shard_df.filter(
        pl.col("subject_id").is_not_null() & pl.col("code").is_not_null() & pl.col("time").is_null()
    )
    static_code_freqs_df = static_df.group_by("code").agg(pl.count("code").alias("count")).collect()
    static_code_freqs = {
        row["code"] + "/static/present": row["count"] for row in static_code_freqs_df.iter_rows(named=True)
    }

    static_value_df = static_df.filter(pl.col("numeric_value").is_not_null())
    static_value_freqs_df = (
        static_value_df.group_by("code").agg(pl.count("numeric_value").alias("count")).collect()
    )
    static_value_freqs = {
        row["code"] + "/static/first": row["count"] for row in static_value_freqs_df.iter_rows(named=True)
    }

    ts_df = shard_df.filter(
        pl.col("subject_id").is_not_null() & pl.col("code").is_not_null() & pl.col("time").is_not_null()
    )
    code_freqs_df = ts_df.group_by("code").agg(pl.count("code").alias("count")).collect()
    code_freqs = {row["code"] + "/code": row["count"] for row in code_freqs_df.iter_rows(named=True)}

    value_df = ts_df.filter(pl.col("numeric_value").is_not_null())
    value_freqs_df = value_df.group_by("code").agg(pl.count("numeric_value").alias("count")).collect()
    value_freqs = {row["code"] + "/value": row["count"] for row in value_freqs_df.iter_rows(named=True)}

    combined_freqs = {**static_code_freqs, **static_value_freqs, **code_freqs, **value_freqs}
    return convert_to_df(combined_freqs)


def get_feature_columns(fp: Path) -> list[str]:
    """Retrieves feature column names from a parquet file.

    Args:
        fp: File path to the Parquet data.

    Returns:
        Sorted list of column names.

    Examples:
        >>> from tempfile import NamedTemporaryFile
        >>> with NamedTemporaryFile() as f:
        ...     pl.DataFrame({"code": ["E", "D", "A"], "count": [1, 3, 2]}).write_parquet(f.name)
        ...     get_feature_columns(f.name)
        ['A', 'D', 'E']
    """
    return sorted(list(get_feature_freqs(fp).keys()))


def get_feature_freqs(fp: Path) -> dict[str, int]:
    """Retrieves feature frequencies from a parquet file.

    Args:
        fp: File path to the Parquet data.

    Returns:
        Dictionary of feature frequencies.

    Examples:
        >>> from tempfile import NamedTemporaryFile
        >>> with NamedTemporaryFile() as f:
        ...     pl.DataFrame({"code": ["E", "D", "A"], "count": [1, 3, 2]}).write_parquet(f.name)
        ...     get_feature_freqs(f.name)
        {'E': 1, 'D': 3, 'A': 2}
    """
    return convert_to_freq_dict(pl.scan_parquet(fp))


def clear_code_aggregation_suffix(code: str) -> str:
    """Removes aggregation suffixes from code strings.

    Args:
        code: Code string to be cleared.

    Returns:
        Code string without aggregation suffixes.

    Raises:
        ValueError: If the code does not have a recognized aggregation suffix.

    Examples:
        >>> clear_code_aggregation_suffix("A/code")
        'A'
        >>> clear_code_aggregation_suffix("A/value")
        'A'
        >>> clear_code_aggregation_suffix("A/static/present")
        'A'
        >>> clear_code_aggregation_suffix("A/static/first")
        'A'
        >>> clear_code_aggregation_suffix("A")
        Traceback (most recent call last):
            ...
        ValueError: Code A does not have a recognized aggregation suffix!
    """
    if code.endswith("/code"):
        return code[:-5]
    elif code.endswith("/value"):
        return code[:-6]
    elif code.endswith("/static/present"):
        return code[:-15]
    elif code.endswith("/static/first"):
        return code[:-13]
    else:
        raise ValueError(f"Code {code} does not have a recognized aggregation suffix!")


def filter_parquet(fp: Path, allowed_codes: list[str]) -> pl.LazyFrame:
    """Loads and filters a Parquet file with Polars to include only specified codes and removes rare
    codes/values.

    Args:
        fp: Path to the Parquet file of a Meds cohort shard.
        allowed_codes: List of codes to filter by.

    Returns:
        pl.LazyFrame: A filtered LazyFrame containing only the allowed and not rare codes/values.

    Examples:
        >>> from tempfile import NamedTemporaryFile
        >>> fp = NamedTemporaryFile()
        >>> pl.DataFrame({
        ...     "code": ["A", "A", "A", "A", "D", "D", "E", "E"],
        ...     "time": [None, None, "2021-01-01", "2021-01-01", None, None, "2021-01-03", "2021-01-04"],
        ...     "numeric_value": [1, None, 2, 2, None, 5, None, 3]
        ... }).write_parquet(fp.name)
        >>> filter_parquet(fp.name, ["A/code", "D/static/present", "E/code", "E/value"]).collect()
        shape: (6, 3)
        ┌──────┬────────────┬───────────────┐
        │ code ┆ time       ┆ numeric_value │
        │ ---  ┆ ---        ┆ ---           │
        │ str  ┆ str        ┆ i64           │
        ╞══════╪════════════╪═══════════════╡
        │ A    ┆ 2021-01-01 ┆ null          │
        │ A    ┆ 2021-01-01 ┆ null          │
        │ D    ┆ null       ┆ null          │
        │ D    ┆ null       ┆ null          │
        │ E    ┆ 2021-01-03 ┆ null          │
        │ E    ┆ 2021-01-04 ┆ 3             │
        └──────┴────────────┴───────────────┘
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

    is_static_code = pl.col("time").is_null()
    is_numeric_code = pl.col("numeric_value").is_not_null()
    rare_static_code = is_static_code & ~pl.col("code").is_in(static_present_feature_columns)
    rare_ts_code = ~is_static_code & ~pl.col("code").is_in(code_feature_columns)
    rare_ts_value = ~is_static_code & ~pl.col("code").is_in(value_feature_columns) & is_numeric_code
    rare_static_value = is_static_code & ~pl.col("code").is_in(static_first_feature_columns) & is_numeric_code

    # Remove rare numeric values by converting them to null
    df = df.with_columns(
        pl.when(rare_static_value | rare_ts_value)
        .then(None)
        .otherwise(pl.col("numeric_value"))
        .alias("numeric_value")
    )
    # Drop rows with rare codes
    df = df.filter(~(rare_static_code | rare_ts_code))
    return df
