from datetime import UTC, datetime
from pathlib import Path

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


def run_describe(input_dir: Path, output_dir: Path) -> Path:
    output_filepath = output_dir / "metadata" / "codes.parquet"
    describe_codes.main(
        DictConfig(
            {
                "input_dir": str(input_dir),
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
    write_events(input_dir / "train" / "0.parquet", ["A"] * 8 + ["B"] * 10)
    write_events(input_dir / "held_out" / "0.parquet", ["A"] * 5 + ["HELD_OUT_ONLY"])

    output_filepath = run_describe(input_dir, tmp_path / "output")

    frequencies = get_feature_freqs(output_filepath)
    assert frequencies["A/code"] == 8
    assert "HELD_OUT_ONLY/code" not in frequencies
    assert filter_to_codes(output_filepath, None, 10, None, None) == ["B/code"]


def test_describe_codes_filters_unsharded_data_using_subject_splits(tmp_path):
    input_dir = tmp_path / "data"
    write_events(input_dir / "0.parquet", ["TRAIN_ONLY"], subject_id=1)
    write_events(input_dir / "1.parquet", ["HELD_OUT_ONLY"], subject_id=2)
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    pl.DataFrame({"subject_id": [1, 2], "split": ["train", "held_out"]}).write_parquet(
        metadata_dir / "subject_splits.parquet"
    )

    frequencies = get_feature_freqs(run_describe(input_dir, tmp_path / "output"))

    assert "TRAIN_ONLY/code" in frequencies
    assert "HELD_OUT_ONLY/code" not in frequencies


def test_describe_codes_errors_when_train_only_fitting_cannot_be_guaranteed(tmp_path):
    input_dir = tmp_path / "data"
    write_events(input_dir / "0.parquet", ["A"])

    with pytest.raises(ValueError, match=r"no train/ directory or metadata/subject_splits\.parquet"):
        run_describe(input_dir, tmp_path / "output")
