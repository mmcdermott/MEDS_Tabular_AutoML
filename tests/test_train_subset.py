"""train_subset: reproducible subsets of the training split (scaling studies), through every stage."""

import json
from io import StringIO
from pathlib import Path

import meds
import polars as pl
import pytest
from omegaconf import OmegaConf

from MEDS_tabular_automl import train_subset
from MEDS_tabular_automl.describe_codes import get_feature_columns
from MEDS_tabular_automl.native import engine
from MEDS_tabular_automl.utils import load_matrix
from tests.test_integration import run_command
from tests.test_tabularize import MEDS_OUTPUTS, SPLITS_JSON


def engine_built() -> bool:
    try:
        engine.library()
    except (OSError, RuntimeError):
        return False
    return True


def write_dataset(root: Path) -> tuple[Path, Path, dict[str, pl.DataFrame]]:
    """The test MEDS dataset under root/data (+ metadata/subject_splits.parquet), with a label at every timed
    event of every subject under root/labels, laid out like the data."""
    input_dir, label_dir = root / "data", root / "labels"
    frames = {}
    for shard, text in MEDS_OUTPUTS.items():
        df = pl.read_csv(StringIO(text)).with_columns(pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f"))
        (input_dir / shard).parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(input_dir / f"{shard}.parquet")
        frames[shard] = df
        labels = (
            df.filter(pl.col("time").is_not_null())
            .select("subject_id", pl.col("time").alias("prediction_time"))
            .unique(maintain_order=True)
            .with_columns((pl.int_range(pl.len()) % 2 == 0).alias("boolean_value"))
        )
        (label_dir / shard).parent.mkdir(parents=True, exist_ok=True)
        labels.write_parquet(label_dir / f"{shard}.parquet")
    splits_fp = root / meds.subject_splits_filepath
    splits_fp.parent.mkdir(parents=True, exist_ok=True)
    pl.concat(
        [
            df.select(pl.col("subject_id").unique()).with_columns(pl.lit(shard.split("/")[0]).alias("split"))
            for shard, df in frames.items()
        ]
    ).write_parquet(splits_fp)
    (input_dir / ".shards.json").write_text(SPLITS_JSON)
    return input_dir, label_dir, frames


def subset_cfg(tmp_path: Path, label_dir: Path, **subset):
    return OmegaConf.create(
        {
            "output_dir": str(tmp_path / "out"),
            "input_label_dir": str(label_dir),
            "train_subset": {"by": None, "n": None, "fraction": None, "salt": "0", **subset},
        }
    )


def read_split(label_dir: Path, split: str) -> dict[str, pl.DataFrame]:
    return {f.stem: pl.read_parquet(f) for f in sorted((Path(label_dir) / split).glob("*.parquet"))}


def subjects_of(frames: dict[str, pl.DataFrame]) -> set[int]:
    return {s for frame in frames.values() for s in frame["subject_id"].to_list()}


def test_subjects_subsets_are_nested_and_keep_tuning_and_held_out(tmp_path):
    _, label_dir, _ = write_dataset(tmp_path)
    source_train = read_split(label_dir, "train")
    all_train = subjects_of(source_train)
    assert len(all_train) == 4

    previous: set[int] = set()
    for n in range(1, len(all_train) + 1):
        cfg = subset_cfg(tmp_path, label_dir, by="subjects", n=n)
        train_subset.apply(cfg)
        cropped = Path(cfg.input_label_dir)
        kept = read_split(cropped, "train")
        selected = subjects_of(kept)
        assert len(selected) == n and previous < selected  # exact size, nested in the smaller subsets
        previous = selected
        for shard, frame in source_train.items():  # every label of a selected subject, in the original order
            assert kept[shard].equals(frame.filter(pl.col("subject_id").is_in(sorted(selected))))
        for split in ("tuning", "held_out"):  # evaluation splits are copied unchanged
            for f in (label_dir / split).glob("*.parquet"):
                assert (cropped / split / f.name).read_bytes() == f.read_bytes()
        manifest = pl.read_parquet(cropped.parent / "subjects.parquet")["subject_id"].to_list()
        assert set(manifest) == selected
        record = json.loads((cropped.parent / "subset.json").read_text())
        assert record["train_subjects"] == {"total": 4, "kept": n}

        # Applying again (another stage, or the same one twice) reuses the same labels.
        again = subset_cfg(tmp_path, label_dir, by="subjects", n=n)
        train_subset.apply(again)
        assert again.input_label_dir == cfg.input_label_dir
        train_subset.apply(cfg)
        assert Path(cfg.input_label_dir) == cropped

    other_salt = subset_cfg(tmp_path, label_dir, by="subjects", n=2, salt="1")
    train_subset.apply(other_salt)
    assert "salt1" in Path(other_salt.input_label_dir).parent.name


def test_labels_subsets(tmp_path):
    _, label_dir, _ = write_dataset(tmp_path)
    source_train = read_split(label_dir, "train")
    total = sum(len(frame) for frame in source_train.values())

    kept_rows = []
    for fraction in (0.25, 0.5, 1.0):
        cfg = subset_cfg(tmp_path, label_dir, by="labels", fraction=fraction)
        train_subset.apply(cfg)
        kept = read_split(Path(cfg.input_label_dir), "train")
        assert sum(len(frame) for frame in kept.values()) == max(1, round(fraction * total))
        rows = set()
        for shard, frame in kept.items():
            source = source_train[shard].with_row_index("row")
            matched = frame.join(source, on=["subject_id", "prediction_time", "boolean_value"], how="left")
            assert matched["row"].null_count() == 0 and matched["row"].is_sorted()  # original rows, in order
            rows |= {(shard, row) for row in matched["row"].to_list()}
        assert all(previous <= rows for previous in kept_rows)  # nested
        kept_rows.append(rows)


def test_bad_requests(tmp_path):
    _, label_dir, _ = write_dataset(tmp_path)
    with pytest.raises(ValueError, match="only 4 subjects"):
        train_subset.apply(subset_cfg(tmp_path, label_dir, by="subjects", n=5))
    cfg = subset_cfg(tmp_path, label_dir, by="subjects", n=1)
    cfg.input_label_dir = None
    with pytest.raises(ValueError, match="needs input_label_dir"):
        train_subset.apply(cfg)
    cfg = subset_cfg(tmp_path, label_dir)
    assert train_subset.apply(cfg) is None and cfg.input_label_dir == str(label_dir)


@pytest.mark.parametrize(
    "backend",
    [
        "python",
        pytest.param(
            "native", marks=pytest.mark.skipif(not engine_built(), reason="The native engine is not built.")
        ),
    ],
)
def test_pipeline_uses_only_the_selected_subjects(tmp_path, backend):
    input_dir, label_dir, frames = write_dataset(tmp_path)
    output_dir, output_model_dir = tmp_path / "out", tmp_path / "models"
    shared = {
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "input_label_dir": str(label_dir.resolve()),
        "train_subset.by": "subjects",
        "train_subset.n": 1,
        "backend": backend,
        "tabularization.min_code_inclusion_count": 1,
        "tabularization.window_sizes": "[30d,365d,full]",
        "do_overwrite": False,
        "tqdm": False,
        "hydra.verbose": True,
    }
    describe = {k: v for k, v in shared.items() if not k.startswith("tabularization.") and k != "backend"}
    run_command("meds-tab-describe", [], describe, "describe_codes with train_subset")
    run_command("meds-tab-tabularize-static", [], shared, "tabularize_static with train_subset")
    run_command("meds-tab-tabularize-time-series", [], shared, "tabularize_time_series with train_subset")

    (subset_dir,) = list((output_dir / "train_subset").iterdir())
    (selected,) = pl.read_parquet(subset_dir / "subjects.parquet")["subject_id"].to_list()
    cropped = subset_dir / "labels"

    # Code frequencies come from the selected training subject only.
    train = pl.concat([df for shard, df in frames.items() if shard.startswith("train/")])
    selected_codes = set(train.filter(pl.col("subject_id") == selected)["code"].to_list())
    other_codes = set(train.filter(pl.col("subject_id") != selected)["code"].to_list()) - selected_codes
    features = get_feature_columns(output_dir / "metadata" / "codes.parquet")
    assert not [f for f in features if any(f.startswith(f"{code}/") for code in other_codes)]
    assert features

    # One tabularized row per kept label, in every shard: the other training shard has no labels left.
    row_counts = {}
    for shard in MEDS_OUTPUTS:
        labels = pl.read_parquet(cropped / f"{shard}.parquet")
        blocks = list((output_dir / "tabularize" / shard).glob("**/*.npz"))
        assert blocks, f"no tabularized blocks for {shard}"
        for block in blocks:
            assert load_matrix(block).shape[0] == len(labels), f"{block}: rows != labels"
        row_counts[shard] = len(labels)
    assert min(row_counts["train/0"], row_counts["train/1"]) == 0
    for split in ("tuning/0", "held_out/0"):
        assert row_counts[split] == len(pl.read_parquet(label_dir / f"{split}.parquet"))

    model = {
        **shared,
        "task_name": "test_task",
        "output_model_dir": str(output_model_dir.resolve()),
        "model_launcher": "xgboost",
        "hydra.sweeper.n_trials": 1,
        "data_loading_params.keep_data_in_memory": True,
    }
    model.pop("backend")
    _, stdout = run_command("meds-tab-model", ["--multirun"], model, "launch_model with train_subset")
    assert "Performance of best model:" in stdout
    (run_dir,) = list(output_model_dir.iterdir())
    held_out = pl.read_parquet(run_dir / "best_trial" / "held_out_predictions.parquet")
    assert len(held_out) == len(pl.read_parquet(label_dir / "held_out" / "0.parquet"))
