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

import numpy as np
import polars as pl
import pytest

from MEDS_tabular_automl.generate_summarized_reps import generate_summary
from MEDS_tabular_automl.generate_ts_features import get_flat_ts_rep

FEATURE_COLUMNS = ["A/code", "A/value", "B/code", "B/value", "C/code", "C/value"]
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
# Three distinct event times and a prediction time after all of them, as strings so that polars parses
# them the way the rest of the suite does.
EVENT_TIMES = ["2020-01-01T00:00:00", "2020-01-01T01:00:00", "2020-01-01T02:00:00"]
PREDICTION_TIME = "2020-01-02T00:00:00"


def summarize(time_indices, codes, values, agg):
    """Aggregate `codes` observed at `EVENT_TIMES[i] for i in time_indices` over a full window."""
    events = pl.LazyFrame(
        {
            "subject_id": [1] * len(codes),
            "time": [EVENT_TIMES[index] for index in time_indices],
            "code": codes,
            "numeric_value": values,
        },
        schema={"subject_id": pl.Int64, "time": pl.String, "code": pl.String, "numeric_value": pl.Float32},
    ).with_columns(pl.col("time").str.to_datetime(TIME_FORMAT))
    labels = pl.LazyFrame(
        {"subject_id": [1], "prediction_time": [PREDICTION_TIME]},
        schema={"subject_id": pl.Int64, "prediction_time": pl.String},
    ).with_columns(pl.col("prediction_time").str.to_datetime(TIME_FORMAT))

    index_df, matrix = get_flat_ts_rep(agg, FEATURE_COLUMNS, events)
    summary = generate_summary(FEATURE_COLUMNS, index_df, matrix, "full", agg, labels)
    return np.asarray(summary.todense())[0]


@pytest.mark.parametrize(
    "time_indices",
    [
        pytest.param([0, 1, 2], id="one-event-per-timestamp"),
        pytest.param([0, 0, 1], id="two-events-share-the-first-timestamp"),
        pytest.param([0, 0, 0], id="all-three-share-one-timestamp"),
    ],
)
def test_code_count_sees_every_event(time_indices):
    """One count per event, whatever the timestamps look like."""
    counts = summarize(time_indices, ["A", "B", "C"], [None, None, None], "code/count")
    np.testing.assert_array_equal(counts, [1, 1, 1])


@pytest.mark.parametrize("time_indices", [[0, 1, 2], [0, 0, 1], [0, 0, 0]])
def test_value_sum_sees_every_value(time_indices):
    """`value/sum` over a full window is the sum of the numeric values, with nothing dropped."""
    sums = summarize(time_indices, ["A", "A", "B"], [1.5, 2.5, -4.0], "value/sum")
    np.testing.assert_allclose(sums, [4.0, -4.0, 0.0], rtol=1e-6)


def test_a_lone_event_is_not_dropped():
    """The smallest case, and the one an inclusive upper bound loses completely."""
    np.testing.assert_array_equal(summarize([0], ["A"], [None], "code/count"), [1, 0, 0])
    np.testing.assert_allclose(summarize([0], ["A"], [3.25], "value/sum"), [3.25, 0.0, 0.0])


@pytest.mark.parametrize(
    "agg, expected",
    [
        ("code/count", 3),
        ("value/count", 3),
        ("value/sum", 1.0 + 2.0 + 4.0),
        ("value/sum_sqd", 1.0**2 + 2.0**2 + 4.0**2),
    ],
)
def test_events_sharing_a_timestamp_count_and_square_individually(agg, expected):
    """Counts are of events, not timestamps, and every value is squared once.

    The first two events share a timestamp, so the per-timestamp step yields counts of 2 and 1 and sums of
    squares of 5 and 16; the window must add these up rather than count or square them again.
    """
    np.testing.assert_allclose(summarize([0, 0, 1], ["A", "A", "A"], [1.0, 2.0, 4.0], agg), [expected, 0, 0])


@pytest.mark.parametrize(
    "codes, values, agg, expected",
    [
        pytest.param(["A", "B", "A"], [5.0, -1.0, 7.0], "value/min", [5.0, -1.0, 0.0], id="min"),
        pytest.param(["A", "B", "A"], [5.0, -1.0, 7.0], "value/max", [7.0, -1.0, 0.0], id="max"),
        pytest.param(["A", "A", "A"], [0.0, 5.0, 7.0], "value/min", [0.0, 0.0, 0.0], id="observed-zero"),
    ],
)
def test_min_and_max_see_observed_values_only(codes, values, agg, expected):
    """A timestamp without a value for a code is not a zero for that code's min or max.

    Each event has its own timestamp, so `A` (positive) and `B` (negative) are each absent from some rows of
    the window. Reducing over those absent entries as zeros gave `A` a minimum and `B` a maximum of 0, which
    were then not stored at all. An observed 0, on the other hand, is a value like any other.
    """
    np.testing.assert_array_equal(summarize([0, 1, 2], codes, values, agg), expected)
