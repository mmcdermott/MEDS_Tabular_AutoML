"""`value/min` and `value/max` must land in the row of the window that produced them.

`sparse_aggregate` returns an ndarray for sum/count/sum_sqd and a `coo_array` for min/max, and
`aggregate_matrix` handles the two branches differently. The ndarray branch tags each result with `i`,
the index of the window being aggregated. The coo branch uses `agg_matrix.row` instead -- the row index
*inside* the aggregate, which for a column-wise reduction over axis 0 is always 0.

So every window's min and max is written into row 0 of the output, where the CSR constructor sums the
duplicates. Row 0 receives the sum of every window's extremum; every other row receives nothing. It
does not raise: the matrix has the right shape and min/max simply look like very sparse features.

Measured on two independent runs of the same configuration, on different machines and commits, over the
same input: `value/sum` had 12,387,109 non-zero entries spread over 2,361 rows, while `value/max` had
8,243 and `value/min` 35 -- every one of them in row 0.

Introduced by `cb21821`, whose message records it as fixing a crash for min and max "due to a coo
matrix being returned rather than a dense matrix as with sum and count operations". The crash was
fixed; the row index was not carried across.
"""

import numpy as np
import polars as pl
import pytest

from MEDS_tabular_automl.generate_summarized_reps import generate_summary
from MEDS_tabular_automl.generate_ts_features import get_feature_names, get_flat_ts_rep

FEATURE_COLUMNS = ["LAB/code", "LAB/value", "PAD/code", "PAD/value"]
EVENT_SCHEMA = {
    "subject_id": pl.Int64,
    "time": pl.String,
    "code": pl.String,
    "numeric_value": pl.Float32,
}
LABEL_SCHEMA = {"subject_id": pl.Int64, "prediction_time": pl.String}
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
# Timestamps as strings parsed by polars, the way the rest of the suite builds them.
OBSERVATION_TIMES = [f"2020-01-01T{hour:02d}:00:00" for hour in range(4)]
PREDICTION_TIME = "2020-01-02T00:00:00"
OBSERVATIONS = len(OBSERVATION_TIMES)
# One subject per base value, four observations each, all positive and all distinct.
SUBJECTS = {1: 10.0, 2: 20.0, 3: 30.0}


def summarize(agg):
    """Aggregate over a full window and return the `LAB/value` column, one entry per subject."""
    events, labels = [], []
    for subject, base in SUBJECTS.items():
        for step, time in enumerate(OBSERVATION_TIMES):
            events.append({"subject_id": subject, "time": time, "code": "LAB", "numeric_value": base + step})
            # A second event at the same timestamp: events are collapsed per (subject_id, time)
            # before windowing, and a timestamp holding a single event contributes nothing.
            events.append({"subject_id": subject, "time": time, "code": "PAD", "numeric_value": None})
        labels.append({"subject_id": subject, "prediction_time": PREDICTION_TIME})

    event_frame = pl.LazyFrame(events, schema=EVENT_SCHEMA).with_columns(
        pl.col("time").str.to_datetime(TIME_FORMAT)
    )
    label_frame = pl.LazyFrame(labels, schema=LABEL_SCHEMA).with_columns(
        pl.col("prediction_time").str.to_datetime(TIME_FORMAT)
    )
    index_df, matrix = get_flat_ts_rep(agg, FEATURE_COLUMNS, event_frame)
    summary = generate_summary(FEATURE_COLUMNS, index_df, matrix, "full", agg, label_frame)
    # The summarized matrix carries only the columns of this aggregation's own type, so LAB/value's
    # index in it is not its index in FEATURE_COLUMNS. Ask upstream rather than assume a layout.
    return np.asarray(summary.todense())[:, get_feature_names(agg, FEATURE_COLUMNS).index("LAB/value")]


@pytest.mark.parametrize(
    "agg, expected",
    [
        ("value/max", [base + OBSERVATIONS - 1 for base in SUBJECTS.values()]),
        ("value/min", list(SUBJECTS.values())),
    ],
)
def test_extremum_lands_in_its_own_row(agg, expected):
    """Each subject's own extremum, in that subject's own row -- not all of them summed into row 0."""
    np.testing.assert_allclose(summarize(agg), expected)


def test_every_row_is_populated_not_just_the_first():
    """The symptom that makes this look like sparsity rather than misplacement."""
    for agg in ("value/min", "value/max"):
        column = summarize(agg)
        assert all(value != 0 for value in column), f"{agg} left rows empty: {column.tolist()}"
