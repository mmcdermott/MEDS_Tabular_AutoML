"""The base class for core dataset processing logic and script utilities."""

from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import ListConfig
from scipy.sparse import coo_array

WRITE_USE_PYARROW = True
ROW_IDX_NAME = "__row_idx"

STATIC_CODE_AGGREGATION = "static/present"
STATIC_VALUE_AGGREGATION = "static/mean"

CODE_AGGREGATIONS = [
    "code/count",
    "code/recency",
]

# Code-group roll-ups (opt-in): a code's group is the code without its last "//" segment (MEDS codes are
# "//"-separated paths, e.g. MEDICATION//Insulin//Administered -> MEDICATION//Insulin); a group of at least
# two described codes is a feature "<group>/group". code/group_count counts its members' records per window
# and code/group_recency is the recency of its latest member record, as code/count and code/recency do for a
# code.
GROUP_AGGREGATIONS = [
    "code/group_count",
    "code/group_recency",
]
GROUP_SUFFIX = "/group"

# Aggregations computed once over the full history up to each row, whatever the window sizes; their files are
# written under the "full" window. Recency over a shorter window w is a threshold of the full-history value
# (the code occurred within w exactly when its recency is at least 1 / (1 + w in days)), so it is redundant.
FULL_HISTORY_AGGREGATIONS = ["code/recency", "code/group_recency"]


def code_group(code: str) -> str | None:
    """The group of a code: the code without its last "//" segment, or None for a code without one.

    Examples:
        >>> code_group("MEDICATION//Insulin//Administered")
        'MEDICATION//Insulin'
        >>> code_group("LAB//RESULT//50822//mEq/L")
        'LAB//RESULT//50822'
        >>> code_group("MEDS_DEATH") is None
        True
    """
    head, sep, _ = code.rpartition("//")
    return head if sep else None


def with_groups(freqs: dict[str, int]) -> dict[str, int]:
    """Adds the "<group>/group" features to described feature counts: every group of at least two described
    codes ("<code>/code" features), counted as the sum of its members' counts. Counts that already list groups
    are returned as they are.

    Examples:
        >>> with_groups({"A//x/code": 3, "A//y/code": 2, "A//y/value": 2, "B//z/code": 5, "C/code": 1})
        {'A//x/code': 3, 'A//y/code': 2, 'A//y/value': 2, 'B//z/code': 5, 'C/code': 1, 'A/group': 5}
    """
    if any(feature.endswith(GROUP_SUFFIX) for feature in freqs):
        return dict(freqs)
    members: dict[str, list[int]] = {}
    for feature, count in freqs.items():
        if feature.endswith("/code"):
            group = code_group(feature[: -len("/code")])
            if group is not None:
                members.setdefault(group, []).append(count)
    groups = {f"{g}{GROUP_SUFFIX}": sum(counts) for g, counts in members.items() if len(counts) >= 2}
    return {**freqs, **groups}


def select_feature_counts(
    counts: dict[str, int],
    allowed_codes,
    min_code_inclusion_count: int | None,
    min_code_inclusion_frequency: float | None,
    max_included_codes: int | None,
) -> list[str]:
    """The features kept by the code filters, sorted. Groups ("<group>/group") follow allowed_codes and the
    count and frequency thresholds, with frequencies relative to the other features' total, but take no place
    under max_included_codes, so they leave the other features' selection unchanged.

    Examples:
        >>> counts = {"A//x/code": 3, "A//y/code": 2, "B/code": 4, "A/group": 5}
        >>> select_feature_counts(counts, None, 3, None, None)
        ['A//x/code', 'A/group', 'B/code']
        >>> select_feature_counts(counts, None, None, None, 1)
        ['A/group', 'B/code']
        >>> select_feature_counts(counts, None, None, 0.4, None)
        ['A/group', 'B/code']
    """
    items = dict(counts)
    if allowed_codes is not None:
        allowed = set(allowed_codes)
        items = {k: v for k, v in items.items() if k in allowed}
    if min_code_inclusion_frequency is not None:
        if min_code_inclusion_frequency < 0 or min_code_inclusion_frequency > 1:
            raise ValueError("min_code_inclusion_frequency must be between 0 and 1.")
        total = sum(v for k, v in items.items() if not k.endswith(GROUP_SUFFIX))
        items = {k: v for k, v in items.items() if v / total >= min_code_inclusion_frequency}
    if min_code_inclusion_count is not None:
        items = {k: v for k, v in items.items() if v >= min_code_inclusion_count}
    if max_included_codes is not None:
        # Break ties by code, so that the codes kept at the cutoff do not depend on the row order.
        ranked = sorted(
            ((k, v) for k, v in items.items() if not k.endswith(GROUP_SUFFIX)), key=lambda kv: (-kv[1], kv[0])
        )
        groups = {k: v for k, v in items.items() if k.endswith(GROUP_SUFFIX)}
        items = {**dict(ranked[:max_included_codes]), **groups}
    return sorted(items)


def agg_windows(agg: str, window_sizes) -> list[str]:
    """The windows an aggregation is computed for.

    Examples:
        >>> agg_windows("code/count", ["1d", "7d"])
        ['1d', '7d']
        >>> agg_windows("code/recency", ["1d", "7d"])
        ['full']
    """
    return ["full"] if agg in FULL_HISTORY_AGGREGATIONS else list(window_sizes)


VALUE_AGGREGATIONS = [
    "value/count",
    "value/sum",
    "value/sum_sqd",
    "value/min",
    "value/max",
]


def filter_to_codes(
    code_metadata_fp: Path,
    allowed_codes: list[str] | None,
    min_code_inclusion_count: int | None,
    min_code_inclusion_frequency: float | None,
    max_include_codes: int | None,
) -> ListConfig[str]:
    """Filters and returns codes based on allowed list and minimum frequency.

    Args:
        code_metadata_fp: Path to the metadata file containing code information.
        allowed_codes: List of allowed codes, None means all codes are allowed.
        min_code_inclusion_count: Minimum count a code must have to be included.
        min_code_inclusion_frequency: The minimum frequency a code must have,
            normalized by dividing its count by the total number of observations
            across all codes in the dataset, to be included.
        max_include_codes: Maximum number of codes to include (selecting the most
            prevelent codes; codes with equal counts are taken in name order).

    Returns:
        Sorted list of the intersection of allowed codes (if they are specified) and filters based on
        inclusion frequency.

    Examples:
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     pl.DataFrame({"code": ["E", "D", "A"], "count": [4, 3, 2]}).write_parquet(f.name)
        ...     filter_to_codes( f.name, ["A", "D"], 3, None, None)
        ['D']
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     pl.DataFrame({"code": ["E", "D", "A"], "count": [4, 3, 2]}).write_parquet(f.name)
        ...     filter_to_codes( f.name, None, None, .35, None)
        ['E']
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     pl.DataFrame({"code": ["E", "D", "A"], "count": [4, 3, 2]}).write_parquet(f.name)
        ...     filter_to_codes( f.name, None, None, None, 1)
        ['E']
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     pl.DataFrame({"code": ["E", "D", "C", "A"], "count": [4, 3, 3, 2]}).write_parquet(f.name)
        ...     filter_to_codes( f.name, None, None, None, 2)
        ['C', 'E']
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     pl.DataFrame({"code": ["E", "D", "A"], "count": [4, 3, 2]}).write_parquet(f.name)
        ...     filter_to_codes( f.name, ["A", "D"], 10, None, None)
        Traceback (most recent call last):
        ...
        ValueError: Code filtering criteria ...
        ...
    """
    frame = pl.read_parquet(code_metadata_fp, columns=["code", "count"])
    counts = with_groups(dict(zip(frame["code"].to_list(), frame["count"].to_list(), strict=True)))
    selected = select_feature_counts(
        counts, allowed_codes, min_code_inclusion_count, min_code_inclusion_frequency, max_include_codes
    )

    if not selected:
        raise ValueError(
            f"Code filtering criteria leaves only 0 codes. Note that {len(counts)} "
            "codes are read in, try modifying the following kwargs:"
            f"\n- tabularization.allowed_codes: {allowed_codes}"
            f"\n- tabularization.min_code_inclusion_count: {min_code_inclusion_count}"
            f"\n- tabularization.min_code_inclusion_frequency: {min_code_inclusion_frequency}"
            f"\n- tabularization.max_include_codes: {max_include_codes}"
        )
    return ListConfig(selected)


def load_tqdm(use_tqdm: bool):
    """Conditionally loads and returns tqdm progress bar handler or a no-operation function.

    Args:
        use_tqdm: Flag indicating whether to use tqdm progress bar.

    Returns:
        A function that either encapsulates tqdm or simply returns the input it is given.
    """
    if use_tqdm:
        from tqdm import tqdm

        return tqdm
    else:

        def noop(x, **kwargs):
            return x

        return noop


def parse_static_feature_column(c: str) -> tuple[str, str, str, str]:
    """Parses a flat feature column format into component parts.

    Args:
        c: The column string in 'category/subcategory/feature' format.

    Returns:
        A tuple containing separate strings of the feature column format.

    Raises:
        ValueError: If the column string format is incorrect.

    Examples:
        >>> parse_static_feature_column("A/static/present")
        ('A', 'static', 'present')
        >>> parse_static_feature_column("A/B/static/mean")
        ('A/B', 'static', 'mean')
        >>> parse_static_feature_column("static/mean")
        Traceback (most recent call last):
            ...
        ValueError: Column static/mean is not a valid flat feature column!
    """
    parts = c.split("/")
    if len(parts) < 3:
        raise ValueError(f"Column {c} is not a valid flat feature column!")
    return ("/".join(parts[:-2]), parts[-2], parts[-1])


def array_to_sparse_matrix(array: np.ndarray, shape: tuple[int, int]) -> coo_array:
    """Converts a numpy array representation into a sparse matrix.

    Args:
        array: The array containing data, rows, and columns.
        shape: The shape of the resulting sparse matrix.

    Returns:
        The formatted sparse matrix.

    Raises:
        AssertionError: If the input array's first dimension is not 3.
    """
    if not array.shape[0] == 3:
        raise AssertionError(
            f"Array must have 3 dimensions: [data, row, col], currently has {array.shape[0]}"
        )
    data, row, col = array
    return coo_array((data, (row, col)), shape=shape)


def get_min_dtype(array: np.ndarray) -> np.dtype:
    """Get the minimal dtype that can represent the array.

    Args:
        array: The array to determine the minimal dtype for.

    Returns:
        The minimal dtype that can represent the array, or the array's dtype if it is non-numeric.

    Examples:
        >>> get_min_dtype(np.array([1, 2, 3]))
        dtype('...')
        >>> get_min_dtype(np.array([1, 2, 3, int(1e9)]))
        dtype('...')
        >>> get_min_dtype(np.array([1, 2, 3, int(1e18)]))
        dtype('...')
        >>> get_min_dtype(np.array([1, 2, 3, -128]))
        dtype('...')
        >>> get_min_dtype(np.array([1.0, 2.0, 3.0]))
        dtype('...')
        >>> get_min_dtype(np.array([1, 2, 3, np.nan]))
        dtype('...')
        >>> get_min_dtype(np.array([1, 2, 3, "a"]))
        dtype('...')

    Floating point values are reduced to float32 only if that represents every value exactly:

        >>> get_min_dtype(np.array([0.5, -2.0, 3.0, np.nan]))
        dtype('float32')
        >>> get_min_dtype(np.array([0.1, 2.0]))
        dtype('float64')
        >>> get_min_dtype(np.array([1e300]))
        dtype('float64')
    """
    if np.issubdtype(array.dtype, np.integer):
        return np.result_type(np.min_scalar_type(array.min()), array.max())
    elif np.issubdtype(array.dtype, np.floating):
        with np.errstate(over="ignore"):
            narrowed = array.astype(np.float32)
        if np.array_equal(narrowed, array, equal_nan=True):
            return np.dtype(np.float32)

    return array.dtype


def sparse_matrix_to_array(coo_matrix: coo_array) -> tuple[np.ndarray, tuple[int, int]]:
    """Converts a sparse matrix to a numpy array format with shape information.

    Args:
        coo_matrix: The sparse matrix to convert.

    Returns:
        A tuple of a numpy array ([data, row, col]) and the shape of the original matrix.

    Examples:
        Stored zeros are kept and NaN entries are dropped:

        >>> array, shape = sparse_matrix_to_array(
        ...     coo_array(([0.0, 2.0, np.nan], ([0, 1, 1], [0, 0, 1])), shape=(2, 2))
        ... )
        >>> array.tolist(), shape
        ([[0.0, 2.0], [0.0, 1.0], [0.0, 0.0]], (2, 2))
    """
    data, row, col = coo_matrix.data, coo_matrix.row, coo_matrix.col
    # Remove NaN entries. Stored zeros are kept: they are observed values (e.g., a measurement of 0 or a sum
    # that cancels out), whereas a feature without any observation has no entry at all.
    valid_indices = ~np.isnan(data)
    data = data[valid_indices]
    row = row[valid_indices]
    col = col[valid_indices]
    # reduce dtypes
    if len(data):
        data = data.astype(get_min_dtype(data), copy=False)
        row = row.astype(get_min_dtype(row), copy=False)
        col = col.astype(get_min_dtype(col), copy=False)

    return np.array([data, row, col]), coo_matrix.shape


def store_matrix(coo_matrix: coo_array, fp_path: Path, do_compress: bool) -> None:
    """Stores a sparse matrix to disk as a .npz file.

    Args:
        coo_matrix: The sparse matrix to store.
        fp_path: The file path where the matrix will be stored.
    """
    array, shape = sparse_matrix_to_array(coo_matrix)
    if do_compress:
        np.savez_compressed(fp_path, array=array, shape=shape)
    else:
        np.savez(fp_path, array=array, shape=shape)


def load_matrix(fp_path: Path) -> coo_array:
    """Loads a sparse matrix from a .npz file.

    Args:
        fp_path: The path to the .npz file containing the sparse matrix data.

    Returns:
        The loaded sparse matrix.
    """
    npzfile = np.load(fp_path)
    array, shape = npzfile["array"], npzfile["shape"]
    return array_to_sparse_matrix(array, shape)


def write_df(
    df: pl.LazyFrame | pl.DataFrame | coo_array,
    fp: Path,
    do_compress: bool = False,
    do_overwrite: bool = False,
) -> None:
    """Writes a sparse matrix to disk.

    Args:
        df: The sparse matrix to write.
        fp: The file path where to write the data.
        do_compress: A flag indicating whether to compress the data.
        do_overwrite: A flag indicating whether to overwrite the file if it already exists.

    Raises:
        FileExistsError: If the file exists and 'do_overwrite' is not set to True.
        TypeError: If the type of 'df' is not supported for writing.

    Examples:
        >>> from polars.testing import assert_frame_equal
        >>> df_polars = pl.DataFrame({"a": [1, 2, 3]})
        >>> df_coo_array = coo_array(([1, 2, 3], ([0, 1, 2], [0, 0, 0])), shape=(3, 1))
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     fp = Path(tmpdir) / "test.parquet"
        ...     write_df(df_polars, fp)
        ...     assert fp.is_file()
        ...     assert_frame_equal(pl.read_parquet(fp), df_polars)
        ...     write_df(df_polars.lazy(), fp, do_overwrite=True)
        ...     assert_frame_equal(pl.read_parquet(fp), df_polars)
        ...     fp = Path(tmpdir) / "test.npz"
        ...     write_df(df_coo_array, fp, do_compress=False, do_overwrite=True)
        ...     assert load_matrix(fp).toarray().tolist() == [[1], [2], [3]]
        ...     fp = Path(tmpdir) / "test_compressed.npz"
        ...     write_df(df_coo_array, fp, do_compress=True, do_overwrite=True)
        ...     assert load_matrix(fp).toarray().tolist() == [[1], [2], [3]]

    Errors are raised if the file already exists and 'do_overwrite' is not set to True.

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     fp = Path(tmpdir) / "test.parquet"
        ...     fp.touch()
        ...     write_df(df_coo_array, fp, do_overwrite=False)
        Traceback (most recent call last):
            ...
        FileExistsError: ...test.parquet exists and do_overwrite is False!
    """
    if fp.is_file() and not do_overwrite:
        raise FileExistsError(f"{fp} exists and do_overwrite is {do_overwrite}!")

    fp.parent.mkdir(exist_ok=True, parents=True)

    if isinstance(df, pl.LazyFrame):
        df.collect().write_parquet(fp, use_pyarrow=WRITE_USE_PYARROW)
    elif isinstance(df, pl.DataFrame):
        df.write_parquet(fp, use_pyarrow=WRITE_USE_PYARROW)
    elif isinstance(df, coo_array):
        store_matrix(df, fp, do_compress)
    else:
        raise TypeError(f"Unsupported type for df: {type(df)}")


def get_events_df(shard_df: pl.LazyFrame, feature_columns) -> pl.LazyFrame:
    """Extracts and filters an Events LazyFrame with one row per observation (times can be duplicated).

    Args:
        shard_df: The LazyFrame shard from which to extract events.
        feature_columns: The columns that define features used to filter the LazyFrame.

    Returns:
        A LazyFrame where each row corresponds to an event, filtered by feature columns.
    """
    # Filter out feature_columns that were not present in the training set
    raw_feature_columns = ["/".join(c.split("/")[:-1]) for c in feature_columns]
    shard_df = shard_df.filter(pl.col("code").is_in(raw_feature_columns))
    # Drop rows with missing time or code to get events
    ts_shard_df = shard_df.drop_nulls(subset=["time", "code"])
    return ts_shard_df


def get_unique_time_events_df(events_df: pl.LazyFrame) -> pl.LazyFrame:
    """Ensures all times in the events LazyFrame are unique and sorted by subject_id and time.

    Args:
        events_df: Events LazyFrame to process.

    Returns:
        A LazyFrame with unique times, sorted by subject_id and time.
    """
    if not events_df.select(pl.col("time")).null_count().collect().item() == 0:
        raise ValueError("Time column must not have null values for time series data.")
    # Check events_df is sorted - so it aligns with the ts_matrix we generate later in the pipeline
    events_df = (
        events_df.drop_nulls("time").select(pl.col(["subject_id", "time"])).unique(maintain_order=True)
    )
    if not events_df.sort(by=["subject_id", "time"]).collect().equals(events_df.collect()):
        raise ValueError("Data frame must be sorted by subject_id and time")
    return events_df


def get_feature_names(agg: str, feature_columns: list[str]) -> str:
    """Extracts feature column names based on aggregation type from a list of column names.

    Args:
        agg: The aggregation type to filter by.
        feature_columns: The list of feature column names.

    Returns:
        The filtered list of feature column names based on the aggregation type.

    Raises:
        ValueError: If the aggregation type is unknown or unsupported.
    """
    if agg in [STATIC_CODE_AGGREGATION, STATIC_VALUE_AGGREGATION]:
        return [c for c in feature_columns if c.endswith(agg)]
    elif agg in CODE_AGGREGATIONS:
        return [c for c in feature_columns if c.endswith("/code")]
    elif agg in GROUP_AGGREGATIONS:
        return [c for c in feature_columns if c.endswith(GROUP_SUFFIX)]
    elif agg in VALUE_AGGREGATIONS:
        return [c for c in feature_columns if c.endswith("/value")]
    else:
        raise ValueError(f"Unknown aggregation type {agg}")


def get_feature_indices(agg: str, feature_columns: list[str]) -> list[int]:
    """Generates a list of feature name indices based on the aggregation type.

    Args:
        agg: The aggregation type used to filter feature names.
        feature_columns: The list of all feature column names.

    Returns:
        Indices of the columns that match the aggregation type.
    """
    feature_to_index = {c: i for i, c in enumerate(feature_columns)}
    agg_features = get_feature_names(agg, feature_columns)
    return [feature_to_index[c] for c in agg_features]


def get_shard_prefix(base_path: Path, fp: Path) -> str:
    """Extracts the shard prefix from a file path by removing the raw_cohort_dir.

    Args:
        base_path: The base path to remove from the file path.
        fp: The full file path from which to extract the shard prefix.

    Returns:
        The shard prefix (the file path relative to the base path with the suffix removed).

    Examples:
        >>> get_shard_prefix(Path("/a/b/c"), Path("/a/b/c/d.parquet"))
        'd'
        >>> get_shard_prefix(Path("/a/b/c"), Path("/a/b/c/d/e.csv.gz"))
        'd/e'
    """

    relative_path = fp.relative_to(base_path)
    relative_parent = relative_path.parent
    file_name = relative_path.name.split(".")[0]

    return (relative_parent / file_name).as_posix()
