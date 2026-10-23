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
