"""Regression tests for the feature-value decisions taken on #173.

- Stored zeros (#173 fix #8): an observed value of 0, or a window whose values cancel to 0, is a stored entry.
  A code that was never observed has no entry. The two must stay distinct through aggregation and writing.
- Task-specific windows (#173 fix #12) end at the prediction time, not at the subject's last event before it,
  both in `generate_summary` and end to end through the tabularization scripts.
- `static/mean` is the mean of the subject's static values of the code (#173 fix #15, renamed from
  `static/first`).

Assertions are on stored sparse entries, not `toarray()`, which shows a stored 0 and a missing entry alike.
"""

from datetime import datetime
from pathlib import Path

import meds
import polars as pl
import pytest
from hydra import compose, initialize
from scipy.sparse import coo_array, csr_array

from MEDS_tabular_automl.generate_static_features import get_flat_static_rep
from MEDS_tabular_automl.generate_summarized_reps import generate_summary
from MEDS_tabular_automl.generate_ts_features import get_flat_ts_rep
from MEDS_tabular_automl.scripts import describe_codes, tabularize_static, tabularize_time_series
from MEDS_tabular_automl.utils import load_matrix, store_matrix

DATA_SCHEMA = {"subject_id": pl.Int64, "time": pl.Datetime, "code": pl.String, "numeric_value": pl.Float64}
VALUE_AGGS = ["value/sum", "value/min", "value/max"]


def jan(day: int, year: int = 2021) -> datetime:
    """Midnight on the given day of January, as a naive datetime like the MEDS times it stands for."""
    return datetime.fromisoformat(f"{year}-01-{day:02d}")


def entries(matrix) -> list[tuple[int, int, float]]:
    """The stored entries of a sparse matrix, as sorted (row, col, value) triples."""
    stored = coo_array(matrix)
    stored.sum_duplicates()
    return sorted(zip(stored.row.tolist(), stored.col.tolist(), stored.data.tolist(), strict=True))


# ---------------------------------------------------------------------------------------------------------
# Stored zeros (#8)
# ---------------------------------------------------------------------------------------------------------

# Columns: A (observed, its values summarize to 0), B (observed, nonzero), C (never observed).
ZERO_FEATURES = ["A/code", "A/value", "B/code", "B/value", "C/code", "C/value"]
ZERO_CASES = {
    # A is observed twice with the value 0.0.
    "observed zeros": {"A": [0.0, 0.0], "expected": {"value/sum": 0.0, "value/min": 0.0, "value/max": 0.0}},
    # A's values cancel: its sum is 0, while its min and max are not.
    "values cancelling to zero": {
        "A": [-2.0, 2.0],
        "expected": {"value/sum": 0.0, "value/min": -2.0, "value/max": 2.0},
    },
}


def zero_case_summary(case: str, agg: str):
    """Summarize one subject's A and B measurements over a full window ending at one label."""
    a_values = ZERO_CASES[case]["A"]
    data = pl.LazyFrame(
        {
            "subject_id": [1, 1, 1],
            "time": [jan(1), jan(2), jan(2)],
            "code": ["A", "A", "B"],
            "numeric_value": [a_values[0], a_values[1], 5.0],
        },
        schema=DATA_SCHEMA,
    )
    labels = pl.LazyFrame({"subject_id": [1], "prediction_time": [jan(3)]})
    index_df, matrix = get_flat_ts_rep(agg, ZERO_FEATURES, data)
    return generate_summary(ZERO_FEATURES, index_df, matrix, "full", agg, labels)


@pytest.mark.parametrize("agg", VALUE_AGGS)
@pytest.mark.parametrize("case", ZERO_CASES)
def test_stored_zero_survives_aggregation_and_writing(tmp_path, case, agg):
    """A's result is stored, B's is stored, and C has no entry, at every stage."""
    expected_a = ZERO_CASES[case]["expected"][agg]
    expected = [(0, 0, expected_a), (0, 1, 5.0)]  # columns: A=0, B=1, C=2

    # 1. generate_summary stores A's result even when it is 0, and nothing for C.
    summary = zero_case_summary(case, agg)
    assert summary.shape == (1, 3)
    assert summary.nnz == 2
    assert entries(summary) == expected

    # 2. Writing and reloading the matrix (sparse_matrix_to_array, then load) keeps the stored zero.
    fp = tmp_path / "summary.npz"
    store_matrix(coo_array(summary), fp, do_compress=False)
    reloaded = load_matrix(fp)
    assert reloaded.nnz == 2
    assert entries(reloaded) == expected


# ---------------------------------------------------------------------------------------------------------
# Task-specific windows end at the prediction time (#12)
# ---------------------------------------------------------------------------------------------------------

EVENT_DAYS = [2, 6, 11, 16, 21, 26]
# Powers of 2, so every sum identifies the set of days it came from.
EVENT_VALUES = {day: float(2**i) for i, day in enumerate(EVENT_DAYS)}  # Jan 2: 1, Jan 6: 2, Jan 11: 4, ...


def anchoring_summary(prediction_time: datetime, window: str):
    data = pl.LazyFrame(
        {
            "subject_id": [1] * len(EVENT_DAYS),
            "time": [jan(day) for day in EVENT_DAYS],
            "code": ["A"] * len(EVENT_DAYS),
            "numeric_value": list(EVENT_VALUES.values()),
        },
        schema=DATA_SCHEMA,
    )
    labels = pl.LazyFrame({"subject_id": [1], "prediction_time": [prediction_time]})
    features = ["A/code", "A/value"]
    index_df, matrix = get_flat_ts_rep("value/sum", features, data)
    return generate_summary(features, index_df, matrix, window, "value/sum", labels)


@pytest.mark.parametrize(
    "prediction_time,window,expected",
    [
        # (Jan 3, Jan 10]: only Jan 6. Anchored at the last event (Jan 6) it would also hold Jan 2.
        (jan(10), "7d", [(0, 0, EVENT_VALUES[6])]),
        # (Dec 25 2021, Jan 1 2022]: no event, so no entry. Anchored at the last event (Jan 26) it would
        # hold Jan 21 and 26.
        (jan(1, year=2022), "7d", []),
        # Everything up to Jan 10: Jan 2 and Jan 6.
        (jan(10), "full", [(0, 0, EVENT_VALUES[2] + EVENT_VALUES[6])]),
    ],
)
def test_task_specific_window_ends_at_the_prediction_time(prediction_time, window, expected):
    summary = anchoring_summary(prediction_time, window)
    assert summary.shape == (1, 1)
    assert entries(summary) == expected


# ---------------------------------------------------------------------------------------------------------
# Task-specific tabularization end to end (#12)
# ---------------------------------------------------------------------------------------------------------

LABEL_BETWEEN_EVENTS = jan(10)
LABEL_AT_EVENT = jan(11)
WINDOWS = ["7d", "full"]
AGGS = [
    "static/present",
    "static/mean",
    "code/count",
    "value/count",
    "value/sum",
    "value/sum_sqd",
    "value/min",
    "value/max",
]


TABULARIZATION = {
    "tabularization.min_code_inclusion_count": 1,
    "tabularization.window_sizes": f"[{','.join(WINDOWS)}]",
    "tabularization.aggs": f"[{','.join(AGGS)}]",
}


def run_stage(stage, config_name: str, overrides: dict):
    with initialize(version_base=None, config_path="../src/MEDS_tabular_automl/configs/"):
        cfg = compose(config_name=config_name, overrides=[f"{k}={v}" for k, v in overrides.items()])
    stage.main(cfg)
    return cfg


def write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """One subject: a static HEIGHT, A on each EVENT_DAY, and three more measurements on Jan 11.

    Labels on Jan 10 (between two events) and Jan 11 (at an event time, which holds A twice and B twice).
    """
    records = [(1, None, "HEIGHT", 170.0)]
    records += [(1, jan(day), "A", value) for day, value in EVENT_VALUES.items()]
    records += [(1, jan(11), "A", 0.5), (1, jan(11), "B", 3.0)]
    records += [(1, jan(11), "B", None)]
    data_dir, label_dir = tmp_path / "data", tmp_path / "labels"
    (data_dir / "train").mkdir(parents=True)
    pl.DataFrame(records, schema=DATA_SCHEMA, orient="row").sort("subject_id", "time").write_parquet(
        data_dir / "train/0.parquet"
    )
    # meds-tab-describe fits code metadata on the training subjects of the MEDS root's subject splits.
    splits_fp = tmp_path / meds.subject_splits_filepath
    splits_fp.parent.mkdir(parents=True)
    pl.DataFrame(
        {meds.SubjectSplitSchema.subject_id_name: [1], meds.SubjectSplitSchema.split_name: [meds.train_split]}
    ).write_parquet(splits_fp)
    (label_dir / "train").mkdir(parents=True)
    pl.DataFrame(
        {
            "subject_id": [1, 1],
            "prediction_time": [LABEL_BETWEEN_EVENTS, LABEL_AT_EVENT],
            "boolean_value": [False, True],
        }
    ).write_parquet(label_dir / "train/0.parquet")
    return data_dir, label_dir


def row(matrix: csr_array, i: int) -> tuple[list[int], list[float]]:
    """The stored column indices and values of one row."""
    start, end = matrix.indptr[i], matrix.indptr[i + 1]
    return matrix.indices[start:end].tolist(), matrix.data[start:end].tolist()


@pytest.fixture(scope="module")
def features(tmp_path_factory) -> dict[str, csr_array]:
    """Run describe, static and time-series tabularization with the labels; return each block by name."""
    tmp_path = tmp_path_factory.mktemp("task_specific")
    data_dir, label_dir = write_inputs(tmp_path)
    output_dir = tmp_path / "output"
    shared = {"input_dir": str(data_dir), "output_dir": str(output_dir), "tqdm": False}
    run_stage(describe_codes, "describe_codes", shared)
    labelled = {**shared, **TABULARIZATION, "input_label_dir": str(label_dir)}
    run_stage(tabularize_static, "tabularization", labelled)
    run_stage(tabularize_time_series, "tabularization", labelled)
    root = output_dir / "tabularize/train/0"
    return {fp.relative_to(root).as_posix(): csr_array(load_matrix(fp)) for fp in root.glob("**/*.npz")}


def test_every_block_has_one_row_per_label(features):
    expected = {"none/static/present.npz", "none/static/mean.npz"} | {
        f"{window}/{agg}.npz" for window in WINDOWS for agg in AGGS if not agg.startswith("static")
    }
    assert set(features) == expected
    for name, matrix in features.items():
        assert matrix.shape[0] == 2, name


def test_finite_window_between_events_ends_at_the_prediction_time(features):
    """Label on Jan 10, 7d window (Jan 3, Jan 10]: Jan 6 only, not Jan 2 (which a window ending at the last
    event, Jan 6, would hold)."""
    a_col = 0  # columns of value features: A, B
    assert row(features["7d/value/sum.npz"], 0) == ([a_col], [EVENT_VALUES[6]])
    assert row(features["7d/code/count.npz"], 0) == ([a_col], [1])


def test_full_window_holds_all_history_up_to_the_prediction_time(features):
    """Label on Jan 10: Jan 2 and Jan 6. Label on Jan 11: also everything on Jan 11."""
    assert row(features["full/value/sum.npz"], 0) == ([0], [EVENT_VALUES[2] + EVENT_VALUES[6]])
    assert row(features["full/code/count.npz"], 0) == ([0], [2])
    a_sum = EVENT_VALUES[2] + EVENT_VALUES[6] + EVENT_VALUES[11] + 0.5
    assert row(features["full/value/sum.npz"], 1) == ([0, 1], [a_sum, 3.0])
    assert row(features["full/code/count.npz"], 1) == ([0, 1], [4, 2])


def test_label_at_an_event_time_holds_every_measurement_of_that_time(features):
    """7d window ending Jan 11: Jan 6 (A=2) and Jan 11 (A=4, A=0.5, B=3.0, and B without a value)."""
    assert row(features["7d/value/sum.npz"], 1) == ([0, 1], [2.0 + 4.0 + 0.5, 3.0])
    assert row(features["7d/code/count.npz"], 1) == ([0, 1], [3, 2])
    assert row(features["7d/value/count.npz"], 1) == ([0, 1], [3, 1])
    assert row(features["7d/value/min.npz"], 1) == ([0, 1], [0.5, 3.0])


@pytest.mark.parametrize(
    "name, expected", [("none/static/present.npz", 1.0), ("none/static/mean.npz", 170.0)]
)
def test_every_label_gets_its_subjects_static_features(features, name, expected):
    assert entries(features[name]) == [(0, 0, expected), (1, 0, expected)]


# ---------------------------------------------------------------------------------------------------------
# static/mean (#15)
# ---------------------------------------------------------------------------------------------------------


def test_static_mean_is_the_mean_of_the_subjects_static_values(tmp_path):
    """HEIGHT has static values null, 171.0 and 170.5: their mean, 170.75, not the first value, 171.0.

    WEIGHT has a single static value, which must come back unchanged at full float64 precision, also after
    writing and reloading the matrix.
    """
    df = pl.LazyFrame(
        [
            (1, None, "HEIGHT", None),
            (1, None, "HEIGHT", 171.0),
            (1, None, "HEIGHT", 170.5),
            (1, None, "WEIGHT", 70.3953106166676),
            (1, jan(1), "ADMIT", None),
        ],
        schema=DATA_SCHEMA,
        orient="row",
    )
    features = ["ADMIT/code", "HEIGHT/static/mean", "WEIGHT/static/mean"]
    matrix = get_flat_static_rep("static/mean", features, df, None)
    assert entries(matrix) == [(0, 0, 170.75), (0, 1, 70.3953106166676)]

    fp = tmp_path / "mean.npz"
    store_matrix(coo_array(matrix), fp, do_compress=False)
    assert entries(load_matrix(fp)) == [(0, 0, 170.75), (0, 1, 70.3953106166676)]
