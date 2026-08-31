"""`value/min` and `value/max` must land in the row of the window that produced them.

`sparse_aggregate` returns an ndarray for sum/count/sum_sqd and a `coo_array` for min/max, and
`aggregate_matrix` handles the two branches differently. The ndarray branch tags each result with `i`,
the index of the window being aggregated. The `coo_array` branch uses `agg_matrix.row` instead -- the
row index *inside* the aggregate, which for a column-wise reduction over axis 0 is always 0.

So every window's min and max is written into row 0 of the output, where the CSR constructor sums the
duplicates. Row 0 receives the sum of every window's extremum; every other row receives nothing. It
does not raise: the matrix has the right shape and min/max simply look like very sparse features.

Measured on two independent runs of the same configuration, on different machines and commits, over the
same input: `value/sum` had 12,387,109 non-zero entries spread over 2,361 rows, while `value/max` had
8,243 and `value/min` 35 -- every one of them in row 0.

Introduced by `cb21821`, whose message records it as fixing a crash for min and max caused by a sparse
return type where sum and count return a dense one. The crash was fixed; the row index was not carried
across.
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
# One subject per band, with disjoint value ranges so an extremum identifies its own subject.
SUBJECTS = {1: 10.0, 2: 20.0, 3: 30.0}
BAND = 10.0


def summarize(agg):
    """Aggregate over a full window and return the `LAB/value` column, one entry per subject.

    Two `LAB` observations at each timestamp, not one `LAB` and one filler. That matters: events are
    collapsed per `(subject_id, time)` before windowing, so a timestamp holding a single event
    contributes nothing -- but a filler carrying no numeric value would leave the `LAB/value` column
    with an implicit zero on its row, and scipy's `min(axis=0)` reduces over implicit zeros. The
    assertions below would then depend on whether that collapse is in effect, which is a *different*
    defect from the one this file is about.
    """
    events, labels = [], []
    for subject, base in SUBJECTS.items():
        for step, time in enumerate(OBSERVATION_TIMES):
            for offset in (0.0, 0.5):
                events.append(
                    {
                        "subject_id": subject,
                        "time": time,
                        "code": "LAB",
                        "numeric_value": base + step + offset,
                    }
                )
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


@pytest.mark.parametrize("agg", ["value/min", "value/max"])
def test_extremum_lands_in_its_own_row(agg):
    """Each subject's extremum must fall inside that subject's own value band.

    Asserted as a band rather than an exact number on purpose. Exactly which observations reach the
    window depends on the per-timestamp collapse, which is a separate concern; what this file is about
    is *where the result is written*. Under the defect every window's extremum is summed into row 0, so
    row 0 lands far above its band and the other rows are empty -- which no band can accommodate.
    """
    column = summarize(agg)
    for (subject, base), value in zip(SUBJECTS.items(), column, strict=True):
        assert base <= value < base + BAND, (
            f"subject {subject}: {agg} is {value}, outside its band [{base}, {base + BAND}) -- "
            f"full column {column.tolist()}"
        )


def test_every_row_is_populated_not_just_the_first():
    """The symptom that makes this look like sparsity rather than misplacement."""
    for agg in ("value/min", "value/max"):
        column = summarize(agg)
        assert all(value != 0 for value in column), f"{agg} left rows empty: {column.tolist()}"
