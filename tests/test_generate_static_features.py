"""Static features must land in the rows of their own subject and in the columns of their own feature.

The static representation has one row per subject in a shard, and is then expanded to one row per event time
(or per label). Each output row must take its own subject's static values, whether or not every subject has
both static and timed records and in whatever order the labels come; each value must take its own feature's
column, whether or not the shard has a record of every static feature.
"""

import polars as pl

from MEDS_tabular_automl.generate_static_features import get_flat_static_rep

TIME_FORMAT = "%Y-%m-%d"


def shard(records: list[tuple]) -> pl.LazyFrame:
    """A MEDS shard from (subject_id, time or None, code, numeric_value) records."""
    return pl.LazyFrame(
        records,
        schema={"subject_id": pl.Int64, "time": pl.String, "code": pl.String, "numeric_value": pl.Float32},
        orient="row",
    ).with_columns(pl.col("time").str.to_datetime(TIME_FORMAT))


def test_subject_without_static_records_gets_an_empty_row():
    """Subject 2 has no static record; the rows of subject 1 keep theirs."""
    df = shard(
        [
            (1, None, "SEX//F", None),
            (1, "2020-01-01", "ADMIT", None),
            (1, "2020-01-02", "ADMIT", None),
            (2, "2020-01-01", "ADMIT", None),
        ]
    )
    matrix = get_flat_static_rep("static/present", ["ADMIT/code", "SEX//F/static/present"], df, None)
    assert matrix.toarray().tolist() == [[1.0], [1.0], [0.0]]


def test_static_values_follow_their_subject():
    """Subject 2 has no static record and subject 3 no timed one: subject 3's value must not reach 2."""
    df = shard(
        [
            (1, None, "HEIGHT", 160.0),
            (1, "2020-01-01", "ADMIT", None),
            (2, "2020-01-01", "ADMIT", None),
            (3, None, "HEIGHT", 170.0),
        ]
    )
    matrix = get_flat_static_rep("static/first", ["ADMIT/code", "HEIGHT/static/first"], df, None)
    assert matrix.toarray().tolist() == [[160.0], [0.0]]


def test_label_rows_take_their_own_subjects_static_values():
    """Label rows need not be grouped or sorted by subject."""
    df = shard([(1, None, "HEIGHT", 160.0), (2, None, "HEIGHT", 170.0), (2, "2020-01-01", "ADMIT", None)])
    labels = pl.LazyFrame(
        {"subject_id": [2, 1, 2], "prediction_time": ["2020-01-02", "2020-01-02", "2020-01-03"]}
    ).with_columns(pl.col("prediction_time").str.to_datetime(TIME_FORMAT))
    matrix = get_flat_static_rep("static/first", ["ADMIT/code", "HEIGHT/static/first"], df, labels)
    assert matrix.toarray().tolist() == [[170.0], [160.0], [170.0]]


def test_static_values_follow_their_feature():
    """The shard has no record of A, so B's indicator must stay in B's column rather than shift into A's."""
    df = shard([(1, None, "B", None), (1, "2020-01-01", "E", None)])
    features = ["A/static/present", "B/static/present", "E/code"]
    matrix = get_flat_static_rep("static/present", features, df, None)
    assert matrix.toarray().tolist() == [[0.0, 1.0]]


def test_timed_records_are_not_static():
    """A timed record of a static code is neither a static value nor a static presence."""
    df = shard(
        [
            (1, None, "HEIGHT", 170.0),
            (1, "2020-01-01", "HEIGHT", 180.0),
            (2, "2020-01-01", "HEIGHT", 175.0),
        ]
    )
    features = ["HEIGHT/code", "HEIGHT/static/first", "HEIGHT/static/present"]
    assert get_flat_static_rep("static/first", features, df, None).toarray().tolist() == [[170.0], [0.0]]
    assert get_flat_static_rep("static/present", features, df, None).toarray().tolist() == [[1.0], [0.0]]
