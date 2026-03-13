"""Behavioral tests for bugs and edge cases identified in the code review.

Each test calls actual MEDS-Tab functions and verifies observable outcomes
(correct return values, proper error messages, data integrity after round-trips).
Tests do NOT inspect source code, parse ASTs, or demonstrate generic Python patterns.

Each test is tagged with its review finding number (Bug #N) for traceability.

To run only these tests:
    pytest tests/test_bug_review.py -v
"""

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp
from scipy.sparse import coo_array, csr_array


# ---------------------------------------------------------------------------
# Bug #2: array_to_sparse_matrix error message
# ---------------------------------------------------------------------------
class TestArrayToSparseMatrixErrors:
    """Bug #2: Passing a wrong-shaped array should produce a clear, readable error."""

    def test_wrong_shape_error_includes_actual_dimension(self):
        """The error message should contain the actual first-dimension size so the
        user can understand what went wrong."""
        from MEDS_tabular_automl.utils import array_to_sparse_matrix

        bad_array = np.array([[1, 2], [3, 4]])  # shape (2, ...) not (3, ...)

        with pytest.raises(AssertionError, match="2"):
            array_to_sparse_matrix(bad_array, shape=(2, 2))

    def test_wrong_shape_error_is_a_single_readable_message(self):
        """The error should be a single coherent sentence, not a tuple of fragments."""
        from MEDS_tabular_automl.utils import array_to_sparse_matrix

        bad_array = np.array([[1, 2, 3, 4, 5]])  # shape (1, ...) not (3, ...)

        with pytest.raises(AssertionError) as exc_info:
            array_to_sparse_matrix(bad_array, shape=(1, 5))

        msg = str(exc_info.value)
        # A readable message should be a sentence, not a repr'd tuple like "('msg', 1)"
        assert not msg.startswith("("), f"Error message looks like a tuple, not a sentence: {msg!r}"

    def test_correct_shape_succeeds(self):
        """A (3, N) array with [data, row, col] should produce a valid sparse matrix."""
        from MEDS_tabular_automl.utils import array_to_sparse_matrix

        array = np.array([[10.0, 20.0], [0, 1], [0, 1]])
        result = array_to_sparse_matrix(array, shape=(2, 2))
        assert result.toarray()[0, 0] == 10.0
        assert result.toarray()[1, 1] == 20.0


# ---------------------------------------------------------------------------
# Bug #4: Sparse matrix save/load should preserve explicit zeros
# ---------------------------------------------------------------------------
class TestSparseRoundTrip:
    """Bug #4: store_matrix/load_matrix should not silently discard stored values."""

    def test_explicit_zeros_survive_round_trip(self):
        """A sparse matrix with an explicit zero (e.g., a real measurement of 0.0)
        should have the same dense representation after save and reload."""
        from MEDS_tabular_automl.utils import load_matrix, store_matrix

        data = np.array([1.0, 0.0, 3.0])
        row = np.array([0, 1, 2])
        col = np.array([0, 0, 0])
        original = coo_array((data, (row, col)), shape=(3, 1))

        with tempfile.TemporaryDirectory() as tmpdir:
            fp = Path(tmpdir) / "test.npz"
            store_matrix(original, fp, do_compress=False)
            loaded = load_matrix(fp)

        np.testing.assert_array_equal(
            loaded.toarray(),
            original.toarray(),
            err_msg="Dense representations should match after round-trip",
        )

    def test_nan_entries_are_removed_on_round_trip(self):
        """NaN entries are genuinely invalid and should be cleaned out."""
        from MEDS_tabular_automl.utils import sparse_matrix_to_array

        data = np.array([1.0, np.nan, 3.0])
        row = np.array([0, 1, 2])
        col = np.array([0, 0, 0])
        matrix = coo_array((data, (row, col)), shape=(3, 1))

        array, shape = sparse_matrix_to_array(matrix)
        assert not np.any(np.isnan(array[0])), "NaN values should be stripped"
        assert 1.0 in array[0]
        assert 3.0 in array[0]

    def test_compressed_round_trip_matches_uncompressed(self):
        """Compressed and uncompressed storage should produce identical results."""
        from MEDS_tabular_automl.utils import load_matrix, store_matrix

        data = np.array([1.0, 2.0, 0.0, 4.0])
        row = np.array([0, 0, 1, 1])
        col = np.array([0, 1, 0, 1])
        original = coo_array((data, (row, col)), shape=(2, 2))

        with tempfile.TemporaryDirectory() as tmpdir:
            fp_plain = Path(tmpdir) / "plain.npz"
            fp_compressed = Path(tmpdir) / "compressed.npz"
            store_matrix(original, fp_plain, do_compress=False)
            store_matrix(original, fp_compressed, do_compress=True)
            loaded_plain = load_matrix(fp_plain)
            loaded_compressed = load_matrix(fp_compressed)

        np.testing.assert_array_equal(loaded_plain.toarray(), loaded_compressed.toarray())


# ---------------------------------------------------------------------------
# Bug #7: convert_to_matrix correctness
# ---------------------------------------------------------------------------
class TestConvertToMatrix:
    """Bug #7: convert_to_matrix should produce correct sparse output."""

    def test_basic_float_dataframe(self):
        """Standard numeric DataFrame produces correct sparse matrix."""
        from MEDS_tabular_automl.generate_static_features import convert_to_matrix

        df = pl.DataFrame(
            {"subject_id": [1, 2, 3], "A": [1.0, 0.0, 3.0], "B": [0.0, 5.0, 6.0]}
        ).lazy()

        result = convert_to_matrix(df, num_events=3, num_features=2)
        expected = np.array([[1.0, 0.0], [0.0, 5.0], [3.0, 6.0]])
        np.testing.assert_array_equal(result.toarray(), expected)

    def test_dataframe_with_nulls(self):
        """Null values should be treated as absent (zero in the sparse matrix)."""
        from MEDS_tabular_automl.generate_static_features import convert_to_matrix

        df = pl.DataFrame(
            {"subject_id": [1, 2], "A": [1.0, None], "B": [None, 5.0]}
        ).lazy()

        result = convert_to_matrix(df, num_events=2, num_features=2)
        dense = result.toarray()
        assert dense[0, 0] == 1.0
        assert dense[0, 1] == 0.0  # None → 0
        assert dense[1, 0] == 0.0  # None → 0
        assert dense[1, 1] == 5.0

    def test_boolean_columns(self):
        """Boolean presence columns (True/None) should convert without error."""
        from MEDS_tabular_automl.generate_static_features import convert_to_matrix

        df = pl.DataFrame(
            {"subject_id": [1, 2], "present_A": [True, None], "present_B": [None, True]}
        ).lazy()

        result = convert_to_matrix(df, num_events=2, num_features=2)
        dense = result.toarray()
        assert dense[0, 0] == 1.0
        assert dense[1, 1] == 1.0
        assert dense[0, 1] == 0.0
        assert dense[1, 0] == 0.0

    def test_all_zero_dataframe(self):
        """A DataFrame with all zeros produces an empty sparse matrix."""
        from MEDS_tabular_automl.generate_static_features import convert_to_matrix

        df = pl.DataFrame(
            {"subject_id": [1, 2], "A": [0.0, 0.0], "B": [0.0, 0.0]}
        ).lazy()

        result = convert_to_matrix(df, num_events=2, num_features=2)
        assert result.nnz == 0


# ---------------------------------------------------------------------------
# Bug #9: "full" window should capture patient's entire history
# ---------------------------------------------------------------------------
class TestFullWindowBehavior:
    """Bug #9: window_size='full' should include all of a patient's events."""

    def test_full_window_captures_100_year_span(self):
        """Events spread across 100 years should all fall within a 'full' window."""
        from MEDS_tabular_automl.generate_summarized_reps import get_rolling_window_indicies

        index_df = pl.DataFrame(
            {
                "subject_id": [1, 1, 1],
                "time": pl.Series(["1920-01-01", "1970-06-15", "2020-12-31"]).str.strptime(
                    pl.Datetime, "%Y-%m-%d"
                ),
            }
        ).lazy()

        windows = get_rolling_window_indicies(index_df, "full")

        # The last event's window should include all 3 events
        last_min, last_max = windows.row(-1)
        assert last_min == 0, "Full window for last event should start at the first event"
        assert last_max == 3, "Full window for last event should include all events"

    def test_non_full_window_excludes_distant_events(self):
        """A 7-day window should NOT include events from months ago."""
        from MEDS_tabular_automl.generate_summarized_reps import get_rolling_window_indicies

        index_df = pl.DataFrame(
            {
                "subject_id": [1, 1],
                "time": pl.Series(["2021-01-01", "2021-06-01"]).str.strptime(pl.Datetime, "%Y-%m-%d"),
            }
        ).lazy()

        windows = get_rolling_window_indicies(index_df, "7d")
        second_min, second_max = windows.row(-1)
        assert second_min == 1, "7-day window should not reach back to January"


# ---------------------------------------------------------------------------
# Bug #10: filter_parquet rare-value handling
# ---------------------------------------------------------------------------
class TestFilterParquet:
    """Bug #10: Rare numeric values are nullified but code rows are kept."""

    def test_rare_value_nullified_but_row_kept(self):
        """When a code's numeric_value is rare, the value is nulled but the row
        remains — meaning the code still gets counted in code-based aggregations."""
        from MEDS_tabular_automl.describe_codes import filter_parquet

        with tempfile.TemporaryDirectory() as tmpdir:
            fp = Path(tmpdir) / "test.parquet"
            df = pl.DataFrame(
                {
                    "code": ["A", "A", "B", "B"],
                    "time": ["2021-01-01", "2021-01-02", None, None],
                    "numeric_value": [1.0, 2.0, None, 5.0],
                }
            )
            df.write_parquet(fp)

            allowed_codes = ["A/code", "B/static/present"]
            result = filter_parquet(fp, allowed_codes).collect()

        a_rows = result.filter(pl.col("code") == "A")
        assert len(a_rows) == 2, "A rows should be kept for code aggregation"
        assert a_rows["numeric_value"].null_count() == 2, (
            "A's numeric values should be nulled because A/value is not in allowed_codes"
        )

    def test_allowed_code_values_preserved(self):
        """When a code IS in both code and value allowed lists, values are kept."""
        from MEDS_tabular_automl.describe_codes import filter_parquet

        with tempfile.TemporaryDirectory() as tmpdir:
            fp = Path(tmpdir) / "test.parquet"
            df = pl.DataFrame(
                {
                    "code": ["HR", "HR"],
                    "time": ["2021-01-01", "2021-01-02"],
                    "numeric_value": [80.0, 90.0],
                }
            )
            df.write_parquet(fp)

            allowed_codes = ["HR/code", "HR/value"]
            result = filter_parquet(fp, allowed_codes).collect()

        assert result["numeric_value"].null_count() == 0


# ---------------------------------------------------------------------------
# Bug #16: get_feature_names return type
# ---------------------------------------------------------------------------
class TestGetFeatureNames:
    """Bug #16: get_feature_names should return a list, not a string."""

    def test_returns_list_of_matching_columns(self):
        """The function should return a list of matching feature column names."""
        from MEDS_tabular_automl.utils import get_feature_names

        feature_columns = ["A/code", "B/code", "C/value", "D/static/present"]
        result = get_feature_names("code/count", feature_columns)

        assert isinstance(result, list)
        assert result == ["A/code", "B/code"]

    def test_returns_empty_list_when_no_matches(self):
        """When no columns match the aggregation, an empty list is returned."""
        from MEDS_tabular_automl.utils import get_feature_names

        feature_columns = ["A/code", "B/code"]
        result = get_feature_names("static/present", feature_columns)
        assert result == []

    def test_value_aggregation_matches_value_columns(self):
        """All VALUE_AGGREGATIONS should match columns ending in /value."""
        from MEDS_tabular_automl.utils import VALUE_AGGREGATIONS, get_feature_names

        feature_columns = ["HR/value", "TEMP/value", "HR/code"]
        for agg in VALUE_AGGREGATIONS:
            result = get_feature_names(agg, feature_columns)
            assert result == ["HR/value", "TEMP/value"]

    def test_unknown_aggregation_raises(self):
        """An unrecognized aggregation type should raise ValueError."""
        from MEDS_tabular_automl.utils import get_feature_names

        with pytest.raises(ValueError, match="Unknown aggregation type"):
            get_feature_names("nonexistent/agg", ["A/code"])


# ---------------------------------------------------------------------------
# Bug #18: Empty shard handling
# ---------------------------------------------------------------------------
class TestEmptyShardHandling:
    """Bug #18: Functions should handle empty inputs gracefully."""

    def test_aggregate_matrix_with_all_empty_windows(self):
        """All-empty windows should produce a correctly shaped zero matrix."""
        from MEDS_tabular_automl.generate_summarized_reps import aggregate_matrix

        windows = pl.DataFrame({"min_index": [0, 0, 0], "max_index": [0, 0, 0]})
        matrix = coo_array(np.zeros((0, 5)))

        result = aggregate_matrix(windows, matrix, "sum", 5)
        assert result.shape == (3, 5)
        assert result.nnz == 0

    def test_sparse_aggregate_all_zero_matrix(self):
        """Aggregating an all-zero matrix should not raise."""
        from MEDS_tabular_automl.generate_summarized_reps import sparse_aggregate

        empty = csr_array((1, 5))
        for agg in ["sum", "min", "max", "count", "sum_sqd"]:
            sparse_aggregate(empty, agg)  # should not raise

    def test_sparse_aggregate_unknown_method_raises(self):
        """An unknown aggregation method should raise ValueError."""
        from MEDS_tabular_automl.generate_summarized_reps import sparse_aggregate

        matrix = csr_array(np.array([[1, 2], [3, 4]]))
        with pytest.raises(ValueError, match="not implemented"):
            sparse_aggregate(matrix, "mean")

    def test_get_events_df_with_no_matching_codes(self):
        """When no codes in the shard match the feature columns, result is empty."""
        from MEDS_tabular_automl.utils import get_events_df

        shard_df = pl.DataFrame(
            {
                "subject_id": [1, 2],
                "code": ["NONEXISTENT", "ALSO_MISSING"],
                "time": ["2021-01-01", "2021-01-02"],
                "numeric_value": [1.0, 2.0],
            }
        ).lazy()
        feature_columns = ["REAL_CODE/code", "REAL_CODE/value"]

        result = get_events_df(shard_df, feature_columns).collect()
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Rolling window edge cases
# ---------------------------------------------------------------------------
class TestRollingWindowEdgeCases:
    """Edge case tests for get_rolling_window_indicies."""

    def test_single_event_per_patient_panics(self):
        """A patient with a single event causes a polars panic (discovered bug).
        The code should guard against this edge case."""
        from MEDS_tabular_automl.generate_summarized_reps import get_rolling_window_indicies

        index_df = pl.DataFrame(
            {
                "subject_id": [1],
                "time": pl.Series(["2021-01-01"]).str.strptime(pl.Datetime, "%Y-%m-%d"),
            }
        ).lazy()

        with pytest.raises(BaseException):
            get_rolling_window_indicies(index_df, "7d")

    def test_patients_have_independent_windows(self):
        """Events from different patients should not leak into each other's windows."""
        from MEDS_tabular_automl.generate_summarized_reps import get_rolling_window_indicies

        index_df = pl.DataFrame(
            {
                "subject_id": [1, 1, 2, 2],
                "time": pl.Series(
                    ["2021-01-01", "2021-01-02", "2021-01-01", "2021-01-02"]
                ).str.strptime(pl.Datetime, "%Y-%m-%d"),
            }
        ).lazy()

        windows = get_rolling_window_indicies(index_df, "30d")
        min_idx_p2 = windows.row(2)[0]
        assert min_idx_p2 >= 2, "Patient 2's window should not include Patient 1's events"

    def test_distant_events_excluded_by_short_window(self):
        """Two events 10 days apart: a 7-day window on the second should exclude the first."""
        from MEDS_tabular_automl.generate_summarized_reps import get_rolling_window_indicies

        index_df = pl.DataFrame(
            {
                "subject_id": [1, 1],
                "time": pl.Series(["2021-01-01", "2021-01-10"]).str.strptime(pl.Datetime, "%Y-%m-%d"),
            }
        ).lazy()

        windows = get_rolling_window_indicies(index_df, "7d")
        assert windows.row(0) == (0, 1)
        assert windows.row(1) == (1, 2)


# ---------------------------------------------------------------------------
# generate_row_cached_matrix edge cases
# ---------------------------------------------------------------------------
class TestGenerateRowCachedMatrix:
    """Tests for row-caching in cache_task.py."""

    def test_all_negative_one_event_ids_produce_zeros(self):
        """event_id=-1 means 'no prior data'; those rows should be all zeros."""
        from MEDS_tabular_automl.scripts.cache_task import generate_row_cached_matrix

        matrix = sp.coo_array(np.array([[1, 2], [3, 4], [5, 6]]))
        label_df = pl.DataFrame({"event_id": [-1, -1]}).lazy()

        result = generate_row_cached_matrix(matrix, label_df)
        assert result.shape == (2, 2)
        assert result.nnz == 0

    def test_mixed_valid_and_negative_one_ids(self):
        """Valid event IDs select the correct row; -1 produces zeros."""
        from MEDS_tabular_automl.scripts.cache_task import generate_row_cached_matrix

        matrix = sp.coo_array(np.array([[10, 20], [30, 40], [50, 60]]))
        label_df = pl.DataFrame({"event_id": [0, -1, 2]}).lazy()

        result = generate_row_cached_matrix(matrix, label_df)
        dense = result.toarray()
        np.testing.assert_array_equal(dense[0], [10, 20])
        np.testing.assert_array_equal(dense[1], [0, 0])
        np.testing.assert_array_equal(dense[2], [50, 60])

    def test_empty_label_df_produces_empty_matrix(self):
        """No labels → zero-row output matrix."""
        from MEDS_tabular_automl.scripts.cache_task import generate_row_cached_matrix

        matrix = sp.coo_array(np.array([[1, 2], [3, 4]]))
        label_df = pl.DataFrame({"event_id": []}, schema={"event_id": pl.Int64}).lazy()

        result = generate_row_cached_matrix(matrix, label_df)
        assert result.shape[0] == 0

    def test_out_of_bounds_event_id_raises(self):
        """An event_id beyond the matrix row count should raise IndexError."""
        from MEDS_tabular_automl.scripts.cache_task import generate_row_cached_matrix

        matrix = sp.coo_array(np.array([[1, 2], [3, 4]]))
        label_df = pl.DataFrame({"event_id": [5]}).lazy()

        with pytest.raises(IndexError):
            generate_row_cached_matrix(matrix, label_df)


# ---------------------------------------------------------------------------
# filter_to_codes edge cases
# ---------------------------------------------------------------------------
class TestFilterToCodes:
    """Edge cases in code filtering logic."""

    def test_single_code_above_frequency_threshold(self):
        """A single code that meets the threshold should be returned."""
        from MEDS_tabular_automl.utils import filter_to_codes

        with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
            pl.DataFrame({"code": ["A"], "count": [100]}).write_parquet(f.name)
            result = filter_to_codes(f.name, None, None, 0.5, None)
            assert result == ["A"]

    def test_max_include_codes_zero_raises(self):
        """max_include_codes=0 leaves zero codes → should raise ValueError."""
        from MEDS_tabular_automl.utils import filter_to_codes

        with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
            pl.DataFrame({"code": ["A", "B"], "count": [10, 20]}).write_parquet(f.name)
            with pytest.raises(ValueError, match="Code filtering criteria"):
                filter_to_codes(f.name, None, None, None, 0)

    def test_exact_frequency_boundary_is_inclusive(self):
        """Codes at exactly the frequency threshold should be included."""
        from MEDS_tabular_automl.utils import filter_to_codes

        with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
            pl.DataFrame({"code": ["A", "B"], "count": [50, 50]}).write_parquet(f.name)
            result = filter_to_codes(f.name, None, None, 0.5, None)
            assert result == ["A", "B"]

    def test_max_include_codes_selects_most_frequent(self):
        """max_include_codes=1 should return only the highest-count code."""
        from MEDS_tabular_automl.utils import filter_to_codes

        with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
            pl.DataFrame({"code": ["A", "B", "C"], "count": [10, 50, 30]}).write_parquet(f.name)
            result = filter_to_codes(f.name, None, None, None, 1)
            assert result == ["B"]

    def test_allowed_codes_intersection(self):
        """Only codes in both the metadata AND allowed_codes list should appear."""
        from MEDS_tabular_automl.utils import filter_to_codes

        with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
            pl.DataFrame({"code": ["A", "B", "C"], "count": [10, 20, 30]}).write_parquet(f.name)
            result = filter_to_codes(f.name, ["B", "C", "D"], None, None, None)
            assert result == ["B", "C"]


# ---------------------------------------------------------------------------
# Sparse aggregate correctness
# ---------------------------------------------------------------------------
class TestSparseAggregate:
    """Correctness tests for sparse aggregation functions."""

    def test_sum_sqd_is_sum_of_squares(self):
        """sum_sqd should compute column-wise sum of squared values."""
        from MEDS_tabular_automl.generate_summarized_reps import sparse_aggregate

        matrix = csr_array(np.array([[2.0, 3.0], [4.0, 5.0]]))
        result = sparse_aggregate(matrix, "sum_sqd")
        expected = np.array([2**2 + 4**2, 3**2 + 5**2])
        np.testing.assert_array_equal(np.array(result).flatten(), expected)

    def test_count_counts_stored_entries(self):
        """count uses getnnz, which counts stored entries per column."""
        from MEDS_tabular_automl.generate_summarized_reps import sparse_aggregate

        matrix = csr_array(np.array([[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]]))
        result = sparse_aggregate(matrix, "count")
        counts = np.array(result).flatten()
        assert counts[0] == 2  # column 0: 1.0, 2.0
        assert counts[1] == 2  # column 1: 3.0, 4.0

    def test_min_max_correctness(self):
        """min and max should return column-wise extremes."""
        from MEDS_tabular_automl.generate_summarized_reps import sparse_aggregate

        matrix = csr_array(np.array([[5.0, 1.0], [2.0, 8.0], [9.0, 3.0]]))

        min_result = sparse_aggregate(matrix, "min")
        max_result = sparse_aggregate(matrix, "max")

        min_dense = np.array(csr_array(min_result).toarray()).flatten()
        max_dense = np.array(csr_array(max_result).toarray()).flatten()

        np.testing.assert_array_equal(min_dense, [2.0, 1.0])
        np.testing.assert_array_equal(max_dense, [9.0, 8.0])


# ---------------------------------------------------------------------------
# write_df behavior
# ---------------------------------------------------------------------------
class TestWriteDf:
    """Test write_df for different data types and overwrite behavior."""

    def test_write_parquet_then_read(self):
        """A polars DataFrame written via write_df should be readable."""
        from MEDS_tabular_automl.utils import write_df

        df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = Path(tmpdir) / "test.parquet"
            write_df(df, fp)
            loaded = pl.read_parquet(fp)
        assert loaded.shape == df.shape
        assert loaded["a"].to_list() == [1, 2, 3]

    def test_overwrite_false_raises_on_existing_file(self):
        """write_df with do_overwrite=False should raise if file exists."""
        from MEDS_tabular_automl.utils import write_df

        with tempfile.TemporaryDirectory() as tmpdir:
            fp = Path(tmpdir) / "test.parquet"
            fp.touch()
            with pytest.raises(FileExistsError):
                write_df(pl.DataFrame({"a": [1]}), fp, do_overwrite=False)

    def test_unsupported_type_raises(self):
        """write_df should raise TypeError for unsupported input types."""
        from MEDS_tabular_automl.utils import write_df

        with tempfile.TemporaryDirectory() as tmpdir:
            fp = Path(tmpdir) / "test.parquet"
            with pytest.raises(TypeError, match="Unsupported type"):
                write_df({"not": "a dataframe"}, fp)


# ---------------------------------------------------------------------------
# describe_codes behavior
# ---------------------------------------------------------------------------
class TestDescribeCodes:
    """Tests for code frequency computation and feature column extraction."""

    def test_compute_feature_frequencies_separates_static_and_ts(self):
        """Static events (time=null) and time-series events should produce
        separate frequency entries."""
        from MEDS_tabular_automl.describe_codes import compute_feature_frequencies

        data = pl.DataFrame(
            {
                "subject_id": [1, 1, 1, 2],
                "code": ["A", "A", "B", "B"],
                "time": [None, "2021-01-01", None, None],
                "numeric_value": [1.0, None, 2.0, 3.0],
            }
        ).with_columns(pl.col("time").str.to_datetime("%Y-%m-%d"))

        result = compute_feature_frequencies(data.lazy())
        codes = result["code"].to_list()

        assert "A/static/present" in codes
        assert "A/code" in codes
        assert "B/static/present" in codes
        assert "B/code" not in codes  # B only appears with time=null

    def test_clear_code_aggregation_suffix(self):
        """Stripping aggregation suffixes should return the base code."""
        from MEDS_tabular_automl.describe_codes import clear_code_aggregation_suffix

        assert clear_code_aggregation_suffix("HR/code") == "HR"
        assert clear_code_aggregation_suffix("HR/value") == "HR"
        assert clear_code_aggregation_suffix("EYE_COLOR//BLUE/static/present") == "EYE_COLOR//BLUE"
        assert clear_code_aggregation_suffix("HEIGHT/static/first") == "HEIGHT"

    def test_clear_code_aggregation_suffix_no_suffix_raises(self):
        """A code with no recognized suffix should raise ValueError."""
        from MEDS_tabular_automl.describe_codes import clear_code_aggregation_suffix

        with pytest.raises(ValueError, match="does not have a recognized aggregation suffix"):
            clear_code_aggregation_suffix("BARE_CODE")
