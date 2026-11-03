"""Every measurement must reach a time-series aggregation.

`compute_agg` collapses measurements to one row per `(subject_id, time)` before applying the rolling window.
That collapse is index-based: it hands `aggregate_matrix` the first and last row index of each group,
and `aggregate_matrix` slices `matrix[min_index:max_index]`. The upper bound is exclusive and equal
bounds are skipped, so an inclusive `max_index` silently loses the last measurement of every timestamp and
loses a single-measurement timestamp completely.

The loss is invisible in normal use: the output matrix has the right shape, the run exits 0, and the
only symptom is that features are smaller than the data. On a dataset where every measurement has its own
timestamp it is total. Measured on the MIMIC-IV demo extract, 1,022,455 timestamped measurements fall into
96,849 `(subject_id, time)` groups, so 9.5% of measurements were dropped.
"""

from datetime import datetime

import polars as pl
import pytest

from MEDS_tabular_automl.describe_codes import filter_parquet
from MEDS_tabular_automl.generate_summarized_reps import generate_summary
from MEDS_tabular_automl.generate_ts_features import get_flat_ts_rep

FEATURE_COLUMNS = ["A/code", "A/value", "B/code", "B/value", "C/code", "C/value"]
DATA_SCHEMA = {"subject_id": pl.Int64, "time": pl.Datetime, "code": pl.String, "numeric_value": pl.Float64}
AGGS = ["code/count", "value/count", "value/sum", "value/sum_sqd", "value/min", "value/max"]

# Every case is summarized over a full window ending at one label, at INDEX_TIME.
PRE_INDEX_T0 = datetime(2020, 1, 1, 0)
PRE_INDEX_T1 = datetime(2020, 1, 1, 1)
PRE_INDEX_T2 = datetime(2020, 1, 1, 2)
INDEX_TIME = datetime(2020, 1, 2)
POST_INDEX_T0 = datetime(2020, 1, 3)

# Each case is one subject's data and the expected summary for each aggregation, by code. 0 means no
# stored entry.
CASES = {
    # Each timestamp holds a single measurement; none of them may be dropped.
    "each measurement at its own timestamp": {
        "data": {
            "time": [PRE_INDEX_T0, PRE_INDEX_T1, PRE_INDEX_T2],
            "code": ["A", "A", "B"],
            "numeric_value": [1, 4, 2],
        },
        "expected": {
            "code/count": {"A": 2, "B": 1, "C": 0},
            "value/count": {"A": 2, "B": 1, "C": 0},
            "value/sum": {"A": 1 + 4, "B": 2, "C": 0},
            "value/sum_sqd": {"A": 1**2 + 4**2, "B": 2**2, "C": 0},
            "value/min": {"A": 1, "B": 2, "C": 0},
            "value/max": {"A": 4, "B": 2, "C": 0},
        },
    },
    # A has three measurements at two timestamps: counts must count measurements, not timestamps, and
    # sum_sqd must square each value once, not square the per-timestamp sums of squares again.
    "measurements sharing a timestamp": {
        "data": {
            "time": [PRE_INDEX_T0, PRE_INDEX_T0, PRE_INDEX_T0, PRE_INDEX_T1],
            "code": ["A", "A", "B", "A"],
            "numeric_value": [1, 2, 5, 4],
        },
        "expected": {
            "code/count": {"A": 3, "B": 1, "C": 0},
            "value/count": {"A": 3, "B": 1, "C": 0},
            "value/sum": {"A": 1 + 2 + 4, "B": 5, "C": 0},
            "value/sum_sqd": {"A": 1**2 + 2**2 + 4**2, "B": 5**2, "C": 0},
            "value/min": {"A": 1, "B": 5, "C": 0},
            "value/max": {"A": 4, "B": 5, "C": 0},
        },
    },
    # A is absent at T1 and B at T0. An absence is not a value of 0, so A's min stays positive and B's max
    # stays negative.
    "codes absent at some timestamps": {
        "data": {
            "time": [PRE_INDEX_T0, PRE_INDEX_T1, PRE_INDEX_T2, PRE_INDEX_T2],
            "code": ["A", "B", "A", "B"],
            "numeric_value": [5, -1, 7, -3],
        },
        "expected": {
            "code/count": {"A": 2, "B": 2, "C": 0},
            "value/count": {"A": 2, "B": 2, "C": 0},
            "value/sum": {"A": 5 + 7, "B": -1 - 3, "C": 0},
            "value/sum_sqd": {"A": 5**2 + 7**2, "B": 1**2 + 3**2, "C": 0},
            "value/min": {"A": 5, "B": -3, "C": 0},
            "value/max": {"A": 7, "B": -1, "C": 0},
        },
    },
    # A's first measurement has no numeric value: it counts as an occurrence of A, but not as a value.
    "a measurement with no numeric value": {
        "data": {
            "time": [PRE_INDEX_T0, PRE_INDEX_T1],
            "code": ["A", "A"],
            "numeric_value": [None, 2],
        },
        "expected": {
            "code/count": {"A": 2, "B": 0, "C": 0},
            "value/count": {"A": 1, "B": 0, "C": 0},
            "value/sum": {"A": 2, "B": 0, "C": 0},
            "value/sum_sqd": {"A": 2**2, "B": 0, "C": 0},
            "value/min": {"A": 2, "B": 0, "C": 0},
            "value/max": {"A": 2, "B": 0, "C": 0},
        },
    },
    # B, exactly at the index time, is in the window; C, after it, is not.
    "measurements at and after the index time": {
        "data": {
            "time": [PRE_INDEX_T0, PRE_INDEX_T1, INDEX_TIME, POST_INDEX_T0],
            "code": ["A", "A", "B", "C"],
            "numeric_value": [1, 4, 2, 3],
        },
        "expected": {
            "code/count": {"A": 2, "B": 1, "C": 0},
            "value/count": {"A": 2, "B": 1, "C": 0},
            "value/sum": {"A": 1 + 4, "B": 2, "C": 0},
            "value/sum_sqd": {"A": 1**2 + 4**2, "B": 2**2, "C": 0},
            "value/min": {"A": 1, "B": 2, "C": 0},
            "value/max": {"A": 4, "B": 2, "C": 0},
        },
    },
    # Every measurement is after the index time, so nothing is summarized.
    "only measurements after the index time": {
        "data": {
            "time": [POST_INDEX_T0, POST_INDEX_T0],
            "code": ["A", "B"],
            "numeric_value": [1, 2],
        },
        "expected": {
            "code/count": {"A": 0, "B": 0, "C": 0},
            "value/count": {"A": 0, "B": 0, "C": 0},
            "value/sum": {"A": 0, "B": 0, "C": 0},
            "value/sum_sqd": {"A": 0, "B": 0, "C": 0},
            "value/min": {"A": 0, "B": 0, "C": 0},
            "value/max": {"A": 0, "B": 0, "C": 0},
        },
    },
}


@pytest.mark.parametrize("agg", AGGS)
@pytest.mark.parametrize("case", CASES)
def test_full_window_summary(case, agg):
    data = pl.LazyFrame({"subject_id": 1, **CASES[case]["data"]}, schema=DATA_SCHEMA)
    labels = pl.LazyFrame({"subject_id": [1], "prediction_time": [INDEX_TIME]})

    index_df, matrix = get_flat_ts_rep(agg, FEATURE_COLUMNS, data)
    summary = generate_summary(FEATURE_COLUMNS, index_df, matrix, "full", agg, labels)

    got = dict(zip(["A", "B", "C"], summary.toarray()[0].tolist(), strict=True))
    assert got == CASES[case]["expected"][agg]


def stored_entries(summary, row):
    """Return `{code: value}` for the entries *stored* in one output row.

    A missing entry is absent from the dict, while a stored 0 is present, which `toarray()` can't tell apart.
    """
    entries = summary.tocsr()[[row], :].tocoo()
    codes = ["A", "B", "C"]
    return {codes[c]: v for c, v in zip(entries.col.tolist(), entries.data.tolist(), strict=True)}


# Two subjects, three labels, a 7d window. Every label is exactly at a measurement time. There,
# anchoring the window at the prediction time and anchoring it at the last event at or before the
# label define the same window, (label - 7d, label], so these expectations hold under both.
WINDOWED_DATA = {
    "subject_id": [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
    "time": [
        datetime(2021, 1, 1),  # subject 1: before both of its windows
        datetime(2021, 1, 6),  # subject 1: A twice at one timestamp, and B
        datetime(2021, 1, 6),
        datetime(2021, 1, 6),
        datetime(2021, 1, 12),  # subject 1: A, and B without a numeric value
        datetime(2021, 1, 12),
        datetime(2021, 1, 20),  # subject 1: C, alone in the Jan 20 window
        datetime(2021, 1, 10),  # subject 2
        datetime(2021, 1, 10),
        datetime(2021, 1, 11),
        datetime(2021, 1, 15),  # subject 2: after its label
    ],
    "code": ["A", "A", "A", "B", "A", "B", "C", "A", "B", "B", "A"],
    "numeric_value": [1, 2, 3, 10, 4, None, 7, -5, 3, -1, 100],
}
WINDOWED_LABELS = {
    "subject_id": [1, 1, 2],
    "prediction_time": [datetime(2021, 1, 12), datetime(2021, 1, 20), datetime(2021, 1, 11)],
}
# One dict per label row, holding its stored entries only.
WINDOWED_EXPECTED = {
    "code/count": [{"A": 3, "B": 2}, {"C": 1}, {"A": 1, "B": 2}],
    "value/count": [{"A": 3, "B": 1}, {"C": 1}, {"A": 1, "B": 2}],
    "value/sum": [{"A": 2 + 3 + 4, "B": 10}, {"C": 7}, {"A": -5, "B": 3 - 1}],
    "value/sum_sqd": [{"A": 2**2 + 3**2 + 4**2, "B": 10**2}, {"C": 7**2}, {"A": 5**2, "B": 3**2 + 1**2}],
    "value/min": [{"A": 2, "B": 10}, {"C": 7}, {"A": -5, "B": -1}],
    "value/max": [{"A": 4, "B": 10}, {"C": 7}, {"A": -5, "B": 3}],
}


@pytest.mark.parametrize("agg", AGGS)
def test_windowed_summary_several_subjects_and_labels(agg):
    """Each output row summarizes only its own subject's measurements inside its own 7d window.

    Subject 1's Jan 1 measurement is outside both of its windows, and its Jan 6 and 12 measurements are
    outside its Jan 20 window. Subject 2's Jan 15 measurement is after its label. If a row picks up another
    label's window or another subject's data, or an aggregate lands in the wrong row, at least one of these
    entries changes.
    """
    data = pl.LazyFrame(WINDOWED_DATA, schema=DATA_SCHEMA)
    labels = pl.LazyFrame(WINDOWED_LABELS)

    index_df, matrix = get_flat_ts_rep(agg, FEATURE_COLUMNS, data)
    summary = generate_summary(FEATURE_COLUMNS, index_df, matrix, "7d", agg, labels)

    assert summary.shape == (3, 3)
    got = [stored_entries(summary, row) for row in range(3)]
    assert got == WINDOWED_EXPECTED[agg]


def test_has_values_count_is_rejected_up_front():
    """`value/has_values_count` has no implementation, so `generate_summary` must reject it up front.

    It used to pass validation and then fail inside `sparse_aggregate` with "not implemented". `value/count`
    already counts only the measurements that have a value.
    """
    data = pl.LazyFrame(WINDOWED_DATA, schema=DATA_SCHEMA)
    index_df, matrix = get_flat_ts_rep("value/count", FEATURE_COLUMNS, data)
    with pytest.raises(ValueError, match="Invalid aggregation: value/has_values_count"):
        generate_summary(FEATURE_COLUMNS, index_df, matrix, "full", "value/has_values_count")


@pytest.mark.xfail(strict=True, reason="fixed by #173 1c60b23")
@pytest.mark.parametrize("agg", ["value/sum", "value/min", "value/max"])
def test_nan_numeric_value_is_missing(tmp_path, agg):
    """A NaN `numeric_value` is a missing value, so it must not turn the window's sum, min or max into NaN.

    The data is read through `filter_parquet`, the way the tabularization scripts read a shard.
    """
    shard_fp = tmp_path / "shard.parquet"
    pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time": [PRE_INDEX_T0, PRE_INDEX_T1, PRE_INDEX_T2],
            "code": ["A", "A", "A"],
            "numeric_value": [1.0, float("nan"), 3.0],
        },
        schema=DATA_SCHEMA,
    ).write_parquet(shard_fp)
    labels = pl.LazyFrame({"subject_id": [1], "prediction_time": [INDEX_TIME]})

    data = filter_parquet(shard_fp, FEATURE_COLUMNS)
    index_df, matrix = get_flat_ts_rep(agg, FEATURE_COLUMNS, data)
    summary = generate_summary(FEATURE_COLUMNS, index_df, matrix, "full", agg, labels)

    expected = {"value/sum": 1.0 + 3.0, "value/min": 1.0, "value/max": 3.0}[agg]
    assert stored_entries(summary, 0) == {"A": expected}
