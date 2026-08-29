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
EVENT_SCHEMA = {
    "subject_id": pl.Int64,
    "time": pl.Datetime("us"),
    "code": pl.String,
    "numeric_value": pl.Float32,
}
LABEL_SCHEMA = {"subject_id": pl.Int64, "prediction_time": pl.Datetime("us")}

T1 = datetime(2020, 1, 1, 0)
T2 = datetime(2020, 1, 1, 1)
T3 = datetime(2020, 1, 1, 2)
PREDICTION_TIME = datetime(2020, 1, 2)


def summarize(times, codes, values, agg):
    events = pl.LazyFrame(
        {
            "subject_id": [1] * len(codes),
            "time": times,
            "code": codes,
            "numeric_value": values,
        },
        schema=EVENT_SCHEMA,
    )
    labels = pl.LazyFrame(
        {"subject_id": [1], "prediction_time": [PREDICTION_TIME]}, schema=LABEL_SCHEMA
    )
    index_df, matrix = get_flat_ts_rep(agg, FEATURE_COLUMNS, events)
    summary = generate_summary(FEATURE_COLUMNS, index_df, matrix, "full", agg, labels)
    return np.asarray(summary.todense())[0]


@pytest.mark.parametrize(
    "times, description",
    [
        ([T1, T2, T3], "one event per timestamp"),
        ([T1, T1, T2], "two events sharing the first timestamp"),
        ([T1, T1, T1], "all three events sharing one timestamp"),
    ],
)
def test_code_count_sees_every_event(times, description):
    """One count per event, whatever the timestamps look like."""
    counts = summarize(times, ["A", "B", "C"], [None, None, None], "code/count")
    np.testing.assert_array_equal(counts, [1, 1, 1], err_msg=description)


@pytest.mark.parametrize("times", [[T1, T2, T3], [T1, T1, T2], [T1, T1, T1]])
def test_value_sum_sees_every_value(times):
    """`value/sum` over a full window is the sum of the numeric values, with nothing dropped."""
    sums = summarize(times, ["A", "A", "B"], [1.5, 2.5, -4.0], "value/sum")
    np.testing.assert_allclose(sums, [4.0, -4.0, 0.0], rtol=1e-6)


def test_a_lone_event_is_not_dropped():
    """The smallest case, and the one an inclusive upper bound loses completely."""
    np.testing.assert_array_equal(
        summarize([T1], ["A"], [None], "code/count"), [1, 0, 0]
    )
    np.testing.assert_allclose(summarize([T1], ["A"], [3.25], "value/sum"), [3.25, 0.0, 0.0])
