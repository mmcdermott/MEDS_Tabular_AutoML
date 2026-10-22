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

import numpy as np
import polars as pl
import pytest

from MEDS_tabular_automl.generate_summarized_reps import generate_summary
from MEDS_tabular_automl.generate_ts_features import get_flat_ts_rep

FEATURE_COLUMNS = ["A/code", "A/value", "B/code", "B/value", "C/code", "C/value"]

# One subject per pattern. Every subject has a single label at 2020-01-02, after all of its events, and
# is summarized over a full window, so each output row is that subject's whole history.
EVENTS = pl.LazyFrame(
    [
        # 1: one event per timestamp
        (1, datetime(2020, 1, 1, 0), "A", 1.0),
        (1, datetime(2020, 1, 1, 1), "B", 2.0),
        (1, datetime(2020, 1, 1, 2), "C", 3.0),
        # 2: A and B share the first timestamp
        (2, datetime(2020, 1, 1, 0), "A", 1.0),
        (2, datetime(2020, 1, 1, 0), "B", 2.0),
        (2, datetime(2020, 1, 1, 1), "C", 3.0),
        # 3: all three share one timestamp
        (3, datetime(2020, 1, 1, 0), "A", 1.0),
        (3, datetime(2020, 1, 1, 0), "B", 2.0),
        (3, datetime(2020, 1, 1, 0), "C", 3.0),
        # 4: a lone event
        (4, datetime(2020, 1, 1, 0), "A", 3.25),
        # 5: the same code twice at one timestamp and once more later
        (5, datetime(2020, 1, 1, 0), "A", 1.0),
        (5, datetime(2020, 1, 1, 0), "A", 2.0),
        (5, datetime(2020, 1, 1, 1), "A", 4.0),
        # 6: positive A, then a timestamp where A is absent
        (6, datetime(2020, 1, 1, 0), "A", 5.0),
        (6, datetime(2020, 1, 1, 1), "B", 1.0),
        # 7: negative A, then a timestamp where A is absent
        (7, datetime(2020, 1, 1, 0), "A", -5.0),
        (7, datetime(2020, 1, 1, 1), "B", -1.0),
        # 8: a code with no numeric value
        (8, datetime(2020, 1, 1, 0), "A", None),
    ],
    schema={"subject_id": pl.Int64, "time": pl.Datetime, "code": pl.String, "numeric_value": pl.Float64},
    orient="row",
)

LABELS = pl.LazyFrame({"subject_id": [1, 2, 3, 4, 5, 6, 7, 8], "prediction_time": [datetime(2020, 1, 2)] * 8})

# Expected [A, B, C] for each subject; 0 means no stored entry.
EXPECTED = {
    "code/count": [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 0, 0],
        [3, 0, 0],  # three events, though only two timestamps
        [1, 1, 0],
        [1, 1, 0],
        [1, 0, 0],
    ],
    "value/count": [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 0, 0],
        [3, 0, 0],
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 0],  # no numeric value, so no value/* entry
    ],
    "value/sum": [
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [3.25, 0, 0],
        [7, 0, 0],
        [5, 1, 0],
        [-5, -1, 0],
        [0, 0, 0],
    ],
    "value/sum_sqd": [
        [1, 4, 9],
        [1, 4, 9],
        [1, 4, 9],
        [10.5625, 0, 0],
        [21, 0, 0],  # 1² + 2² + 4², not (1² + 2²)² + (4²)² = 281
        [25, 1, 0],
        [25, 1, 0],
        [0, 0, 0],
    ],
    "value/min": [
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [3.25, 0, 0],
        [1, 0, 0],
        [5, 1, 0],  # not 0: A's absence at 01:00 is not a value of 0
        [-5, -1, 0],
        [0, 0, 0],
    ],
    "value/max": [
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [3.25, 0, 0],
        [4, 0, 0],
        [5, 1, 0],
        [-5, -1, 0],  # not 0: A's absence at 01:00 is not a value of 0
        [0, 0, 0],
    ],
}


@pytest.mark.parametrize("agg", EXPECTED)
def test_full_window_summary(agg):
    index_df, matrix = get_flat_ts_rep(agg, FEATURE_COLUMNS, EVENTS)
    summary = generate_summary(FEATURE_COLUMNS, index_df, matrix, "full", agg, LABELS)
    np.testing.assert_array_equal(summary.toarray(), EXPECTED[agg])
