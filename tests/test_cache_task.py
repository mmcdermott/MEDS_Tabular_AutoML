"""`meds-tab-cache-task` must keep every label of a shard's subjects, in a deterministic order, and give each
label its subject's static features."""

from pathlib import Path

import polars as pl
from hydra import compose, initialize

from MEDS_tabular_automl.scripts import cache_task, describe_codes, tabularize_static, tabularize_time_series
from MEDS_tabular_automl.utils import load_matrix

TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"


def run_stage(stage, config_name: str, overrides: dict):
    with initialize(version_base=None, config_path="../src/MEDS_tabular_automl/configs/"):
        cfg = compose(config_name=config_name, overrides=[f"{k}={v}" for k, v in overrides.items()])
    stage.main(cfg)
    return cfg


def test_cache_task_keeps_every_label_with_its_static_features(tmp_path):
    input_dir, output_dir, label_dir = tmp_path / "data", tmp_path / "output", tmp_path / "labels"
    # Subject 1 has a static record and two event times, subject 2 only a static record, subject 3 one event
    # time and no static record.
    (input_dir / "train").mkdir(parents=True)
    pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 2, 3],
            "time": [None, "2020-01-01T00:00:00", "2020-01-02T00:00:00", None, "2020-01-01T00:00:00"],
            "code": ["S", "E", "E", "G", "E"],
            "numeric_value": [None, 1.0, None, None, None],
        },
        schema={"subject_id": pl.Int64, "time": pl.String, "code": pl.String, "numeric_value": pl.Float32},
    ).with_columns(pl.col("time").str.to_datetime(TIME_FORMAT)).write_parquet(input_dir / "train/0.parquet")
    # Unsorted labels, one of them before subject 1's first event.
    label_dir.mkdir()
    labels = pl.DataFrame(
        {
            "subject_id": [3, 1, 2, 1],
            "prediction_time": [
                "2020-01-02T00:00:00",
                "2020-01-01T12:00:00",
                "2020-01-02T00:00:00",
                "2019-12-31T00:00:00",
            ],
            "boolean_value": [True, False, True, False],
        }
    )
    labels.with_columns(pl.col("prediction_time").str.to_datetime(TIME_FORMAT)).write_parquet(
        label_dir / "0.parquet"
    )

    shared_config = {
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "do_overwrite": False,
        "seed": 1,
        "tqdm": False,
    }
    tabularization_config = {
        **shared_config,
        "tabularization.min_code_inclusion_count": 1,
        "tabularization.window_sizes": "[full]",
        "tabularization.aggs": "[static/present,code/count]",
    }
    run_stage(describe_codes, "describe_codes", shared_config)
    run_stage(tabularize_static, "tabularization", tabularization_config)
    run_stage(tabularize_time_series, "tabularization", tabularization_config)
    cfg = run_stage(
        cache_task,
        "task_specific_caching",
        {**tabularization_config, "task_name": "task", "input_label_dir": str(label_dir.resolve())},
    )

    labels = pl.read_parquet(Path(cfg.output_label_cache_dir) / "train/0.parquet")
    # Sorted by subject and time. Subject 2 has no timed event but keeps its label, without an event.
    assert labels.select("subject_id", "event_id", "label").rows() == [
        (1, -1, False),
        (1, 0, False),
        (2, -1, True),
        (3, 2, True),
    ]
    task_cache = Path(cfg.output_tabularized_cache_dir) / "train/0"
    # Columns G and S: every row has its subject's static features, with or without past data.
    static = load_matrix(task_cache / "none/static/present.npz").toarray()
    assert static.tolist() == [[0, 1], [0, 1], [1, 0], [0, 0]]
    # Column E: only rows with past data have dynamic features.
    assert load_matrix(task_cache / "full/code/count.npz").toarray().tolist() == [[0], [1], [0], [1]]
