"""Every event must reach a time-series aggregation.

`compute_agg` collapses events to one row per `(subject_id, time)` before applying the rolling window.
That collapse is index-based: it hands `aggregate_matrix` the first and last row index of each group,
and `aggregate_matrix` slices `matrix[min_index:max_index]`. The upper bound is exclusive and equal
bounds are skipped, so an inclusive `max_index` silently loses the last event of every timestamp and
loses a single-event timestamp completely.

The loss is invisible in normal use: the output matrix has the right shape, the run exits 0, and the
only symptom is that features are smaller than the data. On a dataset where every event has its own
timestamp it is total. Measured on the MIMIC-IV demo extract, 1,022,455 timestamped events fall into
96,849 `(subject_id, time)` groups, so 9.5% of events were dropped.
"""

from datetime import datetime

import polars as pl
import pytest

from MEDS_tabular_automl.generate_summarized_reps import generate_summary
from MEDS_tabular_automl.generate_ts_features import get_flat_ts_rep

FEATURE_COLUMNS = ["A/code", "A/value", "B/code", "B/value", "C/code", "C/value"]
EVENT_SCHEMA = {"subject_id": pl.Int64, "time": pl.Datetime, "code": pl.String, "numeric_value": pl.Float64}

T0 = datetime(2020, 1, 1, 0)
T1 = datetime(2020, 1, 1, 1)
T2 = datetime(2020, 1, 1, 2)
BEFORE = datetime(2019, 12, 31)
AFTER = datetime(2020, 1, 2)

# Each case is one subject's events, the prediction time of its single label, and the expected output of
# a full-window summary at that time for each aggregation. 0 means no stored entry.
CASES = {
    "one event per timestamp": (
        {"time": [T0, T1, T2], "code": ["A", "B", "C"], "numeric_value": [1, 2, 3]},
        AFTER,
        {
            "code/count": {"A": 1, "B": 1, "C": 1},
            "value/count": {"A": 1, "B": 1, "C": 1},
            "value/sum": {"A": 1, "B": 2, "C": 3},
            "value/sum_sqd": {"A": 1, "B": 4, "C": 9},
            "value/min": {"A": 1, "B": 2, "C": 3},
            "value/max": {"A": 1, "B": 2, "C": 3},
        },
    ),
    "two events share a timestamp": (
        {"time": [T0, T0, T1], "code": ["A", "B", "C"], "numeric_value": [1, 2, 3]},
        AFTER,
        {
            "code/count": {"A": 1, "B": 1, "C": 1},
            "value/count": {"A": 1, "B": 1, "C": 1},
            "value/sum": {"A": 1, "B": 2, "C": 3},
            "value/sum_sqd": {"A": 1, "B": 4, "C": 9},
            "value/min": {"A": 1, "B": 2, "C": 3},
            "value/max": {"A": 1, "B": 2, "C": 3},
        },
    ),
    "all events share a timestamp": (
        {"time": [T0, T0, T0], "code": ["A", "B", "C"], "numeric_value": [1, 2, 3]},
        AFTER,
        {
            "code/count": {"A": 1, "B": 1, "C": 1},
            "value/count": {"A": 1, "B": 1, "C": 1},
            "value/sum": {"A": 1, "B": 2, "C": 3},
            "value/sum_sqd": {"A": 1, "B": 4, "C": 9},
            "value/min": {"A": 1, "B": 2, "C": 3},
            "value/max": {"A": 1, "B": 2, "C": 3},
        },
    ),
    "a lone event": (
        {"time": [T0], "code": ["A"], "numeric_value": [3.25]},
        AFTER,
        {
            "code/count": {"A": 1, "B": 0, "C": 0},
            "value/count": {"A": 1, "B": 0, "C": 0},
            "value/sum": {"A": 3.25, "B": 0, "C": 0},
            "value/sum_sqd": {"A": 10.5625, "B": 0, "C": 0},
            "value/min": {"A": 3.25, "B": 0, "C": 0},
            "value/max": {"A": 3.25, "B": 0, "C": 0},
        },
    ),
    # Three events at two timestamps. Counts must count events, and sum_sqd must square each value once:
    # 1² + 2² + 4² = 21, not (1² + 2²)² + (4²)² = 281.
    "the same code repeated": (
        {"time": [T0, T0, T1], "code": ["A", "A", "A"], "numeric_value": [1, 2, 4]},
        AFTER,
        {
            "code/count": {"A": 3, "B": 0, "C": 0},
            "value/count": {"A": 3, "B": 0, "C": 0},
            "value/sum": {"A": 7, "B": 0, "C": 0},
            "value/sum_sqd": {"A": 21, "B": 0, "C": 0},
            "value/min": {"A": 1, "B": 0, "C": 0},
            "value/max": {"A": 4, "B": 0, "C": 0},
        },
    ),
    # A is only ever 5. Its absence at T1 is not a value of 0, so its min is 5.
    "a positive code absent at a later timestamp": (
        {"time": [T0, T1], "code": ["A", "B"], "numeric_value": [5, 1]},
        AFTER,
        {
            "code/count": {"A": 1, "B": 1, "C": 0},
            "value/count": {"A": 1, "B": 1, "C": 0},
            "value/sum": {"A": 5, "B": 1, "C": 0},
            "value/sum_sqd": {"A": 25, "B": 1, "C": 0},
            "value/min": {"A": 5, "B": 1, "C": 0},
            "value/max": {"A": 5, "B": 1, "C": 0},
        },
    ),
    # A is only ever -5. Its absence at T1 is not a value of 0, so its max is -5.
    "a negative code absent at a later timestamp": (
        {"time": [T0, T1], "code": ["A", "B"], "numeric_value": [-5, -1]},
        AFTER,
        {
            "code/count": {"A": 1, "B": 1, "C": 0},
            "value/count": {"A": 1, "B": 1, "C": 0},
            "value/sum": {"A": -5, "B": -1, "C": 0},
            "value/sum_sqd": {"A": 25, "B": 1, "C": 0},
            "value/min": {"A": -5, "B": -1, "C": 0},
            "value/max": {"A": -5, "B": -1, "C": 0},
        },
    ),
    # The code is counted, but with no numeric value it has no value/* entries.
    "a code with no numeric value": (
        {"time": [T0], "code": ["A"], "numeric_value": [None]},
        AFTER,
        {
            "code/count": {"A": 1, "B": 0, "C": 0},
            "value/count": {"A": 0, "B": 0, "C": 0},
            "value/sum": {"A": 0, "B": 0, "C": 0},
            "value/sum_sqd": {"A": 0, "B": 0, "C": 0},
            "value/min": {"A": 0, "B": 0, "C": 0},
            "value/max": {"A": 0, "B": 0, "C": 0},
        },
    ),
    # Predicted at T1: B at T1 counts, as the window includes its end, and C at T2 is in the future.
    "events after the prediction time": (
        {"time": [T0, T1, T2], "code": ["A", "B", "C"], "numeric_value": [1, 2, 3]},
        T1,
        {
            "code/count": {"A": 1, "B": 1, "C": 0},
            "value/count": {"A": 1, "B": 1, "C": 0},
            "value/sum": {"A": 1, "B": 2, "C": 0},
            "value/sum_sqd": {"A": 1, "B": 4, "C": 0},
            "value/min": {"A": 1, "B": 2, "C": 0},
            "value/max": {"A": 1, "B": 2, "C": 0},
        },
    ),
    # Every event is after the prediction time, so nothing is summarized.
    "a prediction time before any event": (
        {"time": [T0], "code": ["A"], "numeric_value": [1]},
        BEFORE,
        {
            "code/count": {"A": 0, "B": 0, "C": 0},
            "value/count": {"A": 0, "B": 0, "C": 0},
            "value/sum": {"A": 0, "B": 0, "C": 0},
            "value/sum_sqd": {"A": 0, "B": 0, "C": 0},
            "value/min": {"A": 0, "B": 0, "C": 0},
            "value/max": {"A": 0, "B": 0, "C": 0},
        },
    ),
}
AGGS = ["code/count", "value/count", "value/sum", "value/sum_sqd", "value/min", "value/max"]


@pytest.mark.parametrize("agg", AGGS)
@pytest.mark.parametrize("case", CASES)
def test_full_window_summary(case, agg):
    events, prediction_time, expected = CASES[case]
    events = pl.LazyFrame({"subject_id": 1, **events}, schema=EVENT_SCHEMA)
    labels = pl.LazyFrame({"subject_id": [1], "prediction_time": [prediction_time]})

    index_df, matrix = get_flat_ts_rep(agg, FEATURE_COLUMNS, events)
    summary = generate_summary(FEATURE_COLUMNS, index_df, matrix, "full", agg, labels)

    got = dict(zip(["A", "B", "C"], summary.toarray()[0].tolist(), strict=True))
    assert got == expected[agg]
