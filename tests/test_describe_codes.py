from datetime import UTC, datetime
from pathlib import Path

import meds
import polars as pl
import pytest
from omegaconf import DictConfig

from MEDS_tabular_automl.describe_codes import get_feature_freqs
from MEDS_tabular_automl.scripts import describe_codes
from MEDS_tabular_automl.utils import filter_to_codes


def write_events(path: Path, codes: list[str], subject_id: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "subject_id": [subject_id] * len(codes),
            "code": codes,
            "time": [datetime(2020, 1, 1, tzinfo=UTC)] * len(codes),
            "numeric_value": pl.Series([None] * len(codes), dtype=pl.Float64),
        }
    ).write_parquet(path)


def write_subject_splits(meds_root_dir: Path, splits: dict[int, str]) -> None:
    splits_fp = meds_root_dir / meds.subject_splits_filepath
    splits_fp.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            meds.SubjectSplitSchema.subject_id_name: list(splits),
            meds.SubjectSplitSchema.split_name: list(splits.values()),
        }
    ).write_parquet(splits_fp)


def run_describe(input_dir: Path, output_dir: Path, meds_root_dir: Path | None = None) -> Path:
    output_filepath = output_dir / "metadata" / "codes.parquet"
    describe_codes.main(
        DictConfig(
            {
                "input_dir": str(input_dir),
                "meds_root_dir": str(meds_root_dir) if meds_root_dir else None,
                "cache_dir": str(output_dir / ".cache"),
                "output_filepath": str(output_filepath),
                "do_overwrite": False,
                "tqdm": False,
            }
        )
    )
    return output_filepath


def test_describe_codes_fits_split_sharded_metadata_on_train_only(tmp_path):
    input_dir = tmp_path / "data"
    write_events(input_dir / "train" / "0.parquet", ["A"] * 8 + ["B"] * 10, subject_id=1)
    write_events(input_dir / "held_out" / "0.parquet", ["A"] * 5 + ["HELD_OUT_ONLY"], subject_id=2)
    write_subject_splits(tmp_path, {1: meds.train_split, 2: meds.held_out_split})

    output_filepath = run_describe(input_dir, tmp_path / "output")

    frequencies = get_feature_freqs(output_filepath)
    assert frequencies["A/code"] == 8
    assert "HELD_OUT_ONLY/code" not in frequencies
    assert filter_to_codes(output_filepath, None, 10, None, None) == ["B/code"]


def test_describe_codes_uses_subject_splits_not_shard_directory_names(tmp_path):
    input_dir = tmp_path / "data"
    write_events(input_dir / "train" / "0.parquet", ["TRAIN_ONLY"], subject_id=1)
    write_events(input_dir / "train" / "1.parquet", ["HELD_OUT_IN_TRAIN_DIR"], subject_id=2)
    write_events(input_dir / "held_out" / "0.parquet", ["TRAIN_IN_HELD_OUT_DIR"], subject_id=3)
    write_subject_splits(tmp_path, {1: meds.train_split, 2: meds.held_out_split, 3: meds.train_split})

    frequencies = get_feature_freqs(run_describe(input_dir, tmp_path / "output"))

    assert "TRAIN_ONLY/code" in frequencies
    assert "TRAIN_IN_HELD_OUT_DIR/code" in frequencies
    assert "HELD_OUT_IN_TRAIN_DIR/code" not in frequencies


def test_describe_codes_reads_subject_splits_from_meds_root_dir(tmp_path):
    input_dir = tmp_path / "data"
    write_events(input_dir / "0.parquet", ["TRAIN_ONLY"], subject_id=1)
    write_events(input_dir / "1.parquet", ["HELD_OUT_ONLY"], subject_id=2)
    meds_root_dir = tmp_path / "elsewhere"
    write_subject_splits(meds_root_dir, {1: meds.train_split, 2: meds.held_out_split})

    frequencies = get_feature_freqs(run_describe(input_dir, tmp_path / "output", meds_root_dir))

    assert "TRAIN_ONLY/code" in frequencies
    assert "HELD_OUT_ONLY/code" not in frequencies


def test_describe_codes_errors_when_subject_splits_file_is_missing(tmp_path):
    input_dir = tmp_path / "data"
    write_events(input_dir / "train" / "0.parquet", ["A"])

    with pytest.raises(FileNotFoundError, match=r"metadata/subject_splits\.parquet"):
        run_describe(input_dir, tmp_path / "output")
