"""The base class for core dataset processing logic.

Attributes:
    INPUT_DF_T: This defines the type of the allowable input dataframes -- e.g., databases, filepaths,
        dataframes, etc.
    DF_T: This defines the type of internal dataframes -- e.g. polars DataFrames.
"""

import enum
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import polars.selectors as cs


class CodeType(enum.Enum):
    """Enum for the type of code."""

    STATIC_CATEGORICAL = "STATIC_CATEGORICAL"
    DYNAMIC_CATEGORICAL = "DYNAMIC_CATEGORICAL"
    STATIC_CONTINUOUS = "STATIC_CONTINUOUS"
    DYNAMIC_CONTINUOUS = "DYNAMIC_CONTINUOUS"


DF_T = pl.DataFrame
WRITE_USE_PYARROW = True


def _parse_flat_feature_column(c: str) -> tuple[str, str, str, str]:
    parts = c.split("/")
    if len(parts) < 3:
        raise ValueError(f"Column {c} is not a valid flat feature column!")
    return (parts[0], "/".join(parts[1:-1]), parts[-1])


def write_df(df: DF_T, fp: Path, **kwargs):
    """Write shard to disk."""
    do_overwrite = kwargs.get("do_overwrite", False)

    if not do_overwrite and fp.is_file():
        raise FileExistsError(f"{fp} exists and do_overwrite is {do_overwrite}!")

    fp.parent.mkdir(exist_ok=True, parents=True)

    if isinstance(df, pl.LazyFrame):
        df.collect().write_parquet(fp, use_pyarrow=WRITE_USE_PYARROW)
    else:
        df.write_parquet(fp, use_pyarrow=WRITE_USE_PYARROW)


def get_smallest_valid_uint_type(num: int | float | pl.Expr) -> pl.DataType:
    """Returns the smallest valid unsigned integral type for an ID variable with `num` unique options.

    Args:
        num: The number of IDs that must be uniquely expressed.

    Raises:
        ValueError: If there is no unsigned int type big enough to express the passed number of ID
            variables.

    Examples:
        >>> import polars as pl
        >>> Dataset.get_smallest_valid_uint_type(num=1)
        UInt8
        >>> Dataset.get_smallest_valid_uint_type(num=2**8-1)
        UInt16
        >>> Dataset.get_smallest_valid_uint_type(num=2**16-1)
        UInt32
        >>> Dataset.get_smallest_valid_uint_type(num=2**32-1)
        UInt64
        >>> Dataset.get_smallest_valid_uint_type(num=2**64-1)
        Traceback (most recent call last):
            ...
        ValueError: Value is too large to be expressed as an int!
    """
    if num >= (2**64) - 1:
        raise ValueError("Value is too large to be expressed as an int!")
    if num >= (2**32) - 1:
        return pl.UInt64
    elif num >= (2**16) - 1:
        return pl.UInt32
    elif num >= (2**8) - 1:
        return pl.UInt16
    else:
        return pl.UInt8


def get_flat_col_dtype(col: str) -> pl.DataType:
    """Gets the appropriate minimal dtype for the given flat representation column string."""

    code_type, code, agg = _parse_flat_feature_column(col)

    match agg:
        case "sum" | "sum_sqd" | "min" | "max" | "value" | "first":
            return pl.Float32
        case "present":
            return pl.Boolean
        case "count" | "has_values_count":
            return pl.UInt32
            # TODO: reduce the dtype to the smallest possible unsigned int type
            # return get_smallest_valid_uint_type(total_observations)
        case _:
            raise ValueError(f"Column name {col} malformed!")


def _normalize_flat_rep_df_cols(
    flat_df: DF_T, feature_columns: list[str], set_count_0_to_null: bool = False
) -> DF_T:
    """Normalizes columns in a DataFrame so all expected columns are present and appropriately typed.

    Parameters:
    - flat_df (DF_T): The DataFrame to be normalized.
    - feature_columns (list[str]): A list of feature column names that should exist in the DataFrame.
    - set_count_0_to_null (bool): A flag indicating whether counts of zero should be converted to nulls.

    Returns:
    - DF_T: The normalized DataFrame with all columns set to the correct type and zero-counts handled
        if specified.

    This function ensures that all necessary columns are added and typed correctly within
    a DataFrame, potentially modifying zero counts to nulls based on the configuration.
    """
    cols_to_add = set(feature_columns) - set(flat_df.columns)
    cols_to_retype = set(feature_columns).intersection(set(flat_df.columns))

    cols_to_add = [(c, get_flat_col_dtype(c)) for c in cols_to_add]
    cols_to_retype = [(c, get_flat_col_dtype(c)) for c in cols_to_retype]

    if "timestamp" in flat_df.columns:
        key_cols = ["patient_id", "timestamp"]
    else:
        key_cols = ["patient_id"]

    flat_df = flat_df.with_columns(
        *[pl.lit(None, dtype=dt).alias(c) for c, dt in cols_to_add],
        *[pl.col(c).cast(dt).alias(c) for c, dt in cols_to_retype],
    ).select(*key_cols, *feature_columns)

    if not set_count_0_to_null:
        return flat_df

    flat_df = flat_df.collect()

    flat_df = flat_df.with_columns(
        pl.when(cs.ends_with("count") != 0).then(cs.ends_with("count")).keep_name()
    ).lazy()
    return flat_df


def evaluate_code_properties(df, cfg):
    """Evaluates and categorizes each code in a dataframe based on its timestamp presence and numerical
    values.

    This function categorizes codes as 'dynamic' or 'static' based on the presence
    of timestamps, and as 'continuous' or 'categorical' based on the presence of
    numerical values. A code is considered:
    - Dynamic if the ratio of present timestamps to its total occurrences exceeds
    the configured dynamic threshold.
    - Continuous if the ratio of non-null numerical values to total occurrences
    exceeds the numerical value threshold
      and there is more than one unique numerical value.

    Parameters:
    - df (DataFrame): The dataframe containing the codes and their attributes.
    - cfg (dict): Configuration dictionary with keys 'dynamic_threshold', 'numerical_value_threshold',
      and 'min_code_inclusion_frequency' to determine the thresholds for categorizing codes.

    Returns:
    - dict: A dictionary with code as keys and their properties (e.g., 'dynamic_continuous') as values.
        Codes with total occurrences less than 'min_code_inclusion_frequency' are excluded.

    Examples:
    >>> import polars as pl
    >>> data = {'code': ['A', 'A', 'B', 'B', 'C', 'C', 'C'],
    ...         'timestamp': [None, '2021-01-01', None, '2021-01-02', '2021-01-03', '2021-01-04', None],
    ...         'numerical_value': [1, None, 2, 2, None, None, 3]}
    >>> df = pl.DataFrame(data)
    >>> cfg = {'dynamic_threshold': 0.5, 'numerical_value_threshold': 0.5, 'min_code_inclusion_frequency': 1}
    >>> evaluate_code_properties(df, cfg)
    {'A': 'static_categorical', 'B': 'dynamic_continuous', 'C': 'dynamic_categorical'}
    """
    code_properties = OrderedDict()
    for code in df.select(pl.col("code").unique()).collect().to_series():
        # Determine total count, timestamp count, and numerical count
        code_data = df.filter(pl.col("code") == code)
        total_count = code_data.select(pl.count("code")).collect().item()
        if total_count < cfg["min_code_inclusion_frequency"]:
            continue

        timestamp_count = code_data.select(pl.col("timestamp").count()).collect().item()
        numerical_count = code_data.select(pl.col("numerical_value").count()).collect().item()

        # Determine dynamic vs static
        is_dynamic = (timestamp_count / total_count) > cfg["dynamic_threshold"]

        # Determine categorical vs continuous
        is_continuous = (numerical_count / total_count) > cfg[
            "numerical_value_threshold"
        ] and code_data.select(pl.col("numerical_value").n_unique()).collect().item() > 1

        match (is_dynamic, is_continuous):
            case (False, False):
                code_properties[code] = CodeType.STATIC_CATEGORICAL
            case (False, True):
                code_properties[code] = CodeType.STATIC_CONTINUOUS
            case (True, False):
                code_properties[code] = CodeType.DYNAMIC_CATEGORICAL
            case (True, True):
                code_properties[code] = CodeType.DYNAMIC_CONTINUOUS

    return code_properties


def get_code_column(code: str, code_type: CodeType, aggs: Sequence[str]):
    """Constructs feature column names based on a given code, its type, and specified aggregations.

    Parameters:
    - code (str): The specific code identifier for which the feature columns are being generated.
    - code_type (CodeType): The type of the code (e.g., STATIC_CATEGORICAL, DYNAMIC_CONTINUOUS)
        that determines how the code is processed.
    - aggs (Sequence[str]): A list of aggregation operations to apply to the code, e.g.,
        "count", "sum".

    Returns:
    - list[str]: A list of fully qualified feature column names constructed based on the
        code type and applicable aggregations.

    This function builds a list of feature column names using the code and its type to apply
    the correct prefix and filters applicable aggregations based on whether they are relevant
    to the code type.
    """
    prefix = f"{code_type.value}/{code}"
    if code_type == CodeType.STATIC_CATEGORICAL:
        return [f"{prefix}/present"]
    elif code_type == CodeType.DYNAMIC_CATEGORICAL:
        valid_aggs = [agg[4:] for agg in aggs if agg.startswith("code")]
        return [f"{prefix}/{agg}" for agg in valid_aggs]
    elif code_type == CodeType.STATIC_CONTINUOUS:
        return [f"{prefix}/present", f"{prefix}/first"]
    elif code_type == CodeType.DYNAMIC_CONTINUOUS:
        valid_aggs = [agg[5:] for agg in aggs if agg.startswith("value")]
        return [f"{prefix}/{agg}" for agg in valid_aggs]
    else:
        raise ValueError(f"Invalid code type: {code_type}")


def get_flat_rep_feature_cols(cfg, split_to_shard_df) -> list[str]:
    """Generates a list of feature column names from the data within each shard based on specified
    configurations.

    Parameters:
    - cfg (dict): Configuration dictionary specifying how features should be evaluated and aggregated.
    - split_to_shard_df (dict): A dictionary of DataFrames, divided by data split (e.g., 'train', 'test').

    Returns:
    - tuple[list[str], dict]: A tuple containing a list of feature columns and a dictionary of code properties
        identified during the evaluation.

    This function evaluates the properties of codes within training data and applies configured
    aggregations to generate a comprehensive list of feature columns for modeling purposes.
    """
    feature_columns = []
    all_train_data = pl.concat(split_to_shard_df["train"])
    code_properties = evaluate_code_properties(all_train_data, cfg)
    for code, code_type in code_properties.items():
        feature_columns.extend(get_code_column(code, code_type, cfg.aggs))
    return feature_columns, code_properties
