"""Tests for Hydra-wrapped script entry points.

These tests exercise validation paths and error handling in the scripts/ directory, including cache_task,
tabularize_static, tabularize_time_series, and launch_autogluon.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

# ============================================================================
# scripts/cache_task.py
# ============================================================================


def test_cache_task_no_tabularized_data(tmp_path):
    """cache_task.main raises when no tabularized data found (line 127)."""
    from MEDS_tabular_automl.scripts.cache_task import main

    cfg = MagicMock()
    cfg.tqdm = False
    cfg.input_tabularized_dir = str(tmp_path / "empty")
    (tmp_path / "empty").mkdir()

    with pytest.raises(FileNotFoundError, match="No tabularized data found"):
        main.__wrapped__(cfg)  # bypass hydra decorator


def test_cache_task_no_label_dir(tmp_path):
    """cache_task.main raises when label directory missing (line 136)."""
    from MEDS_tabular_automl.scripts.cache_task import main

    # Create a fake .npz file so tabularization_tasks is non-empty
    tab_dir = tmp_path / "tab" / "train" / "0" / "30d" / "code"
    tab_dir.mkdir(parents=True)
    np.savez(tab_dir / "count.npz", array=np.zeros((3, 1)), shape=np.array([1, 1]))

    cfg = MagicMock()
    cfg.tqdm = False
    cfg.input_tabularized_dir = str(tmp_path / "tab")
    cfg.input_label_dir = str(tmp_path / "nonexistent_labels")

    with pytest.raises(FileNotFoundError, match="Label directory"):
        main.__wrapped__(cfg)


def test_cache_task_missing_numeric_value(tmp_path):
    """cache_task inner read_meds_data_df raises when numeric_value missing (line 165)."""
    from MEDS_tabular_automl.scripts.cache_task import main

    # Create tabularized data
    tab_dir = tmp_path / "tab" / "train" / "0" / "30d" / "code"
    tab_dir.mkdir(parents=True)
    np.savez(tab_dir / "count.npz", array=np.zeros((3, 1)), shape=np.array([1, 1]))

    # Create labels
    label_dir = tmp_path / "labels" / "train"
    label_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "subject_id": [1],
            "prediction_time": pl.Series(["2021-01-01"]).str.strptime(pl.Datetime),
            "boolean_value": [True],
        }
    ).write_parquet(label_dir / "0.parquet")

    # Create MEDS data WITHOUT numeric_value column
    meds_dir = tmp_path / "meds" / "train"
    meds_dir.mkdir(parents=True)
    pl.DataFrame({"subject_id": [1], "code": ["A"]}).write_parquet(meds_dir / "0.parquet")

    cfg = MagicMock()
    cfg.tqdm = False
    cfg.input_tabularized_dir = str(tmp_path / "tab")
    cfg.input_label_dir = str(tmp_path / "labels")
    cfg.label_column = "boolean_value"
    cfg.tabularization.filtered_code_metadata_fp = str(tmp_path / "codes.parquet")
    cfg.input_dir = str(tmp_path / "meds")
    cfg.output_label_cache_dir = str(tmp_path / "label_cache")
    cfg.output_tabularized_cache_dir = str(tmp_path / "tab_cache")

    # Create code metadata
    pl.DataFrame({"code": ["A"], "count": [1]}).write_parquet(tmp_path / "codes.parquet")

    with pytest.raises(ValueError, match=r"numeric_value.*column not found"):
        main.__wrapped__(cfg)


# ============================================================================
# scripts/tabularize_static.py and tabularize_time_series.py
# ============================================================================


def test_aggregate_matrix_unsupported_agg_type():
    """aggregate_matrix raises TypeError when sparse_aggregate returns unexpected type (line 272)."""
    from MEDS_tabular_automl.generate_summarized_reps import aggregate_matrix

    matrix = sp.csr_array(np.eye(3))
    windows = pl.DataFrame({"min_index": [0], "max_index": [3]})

    with (
        patch(
            "MEDS_tabular_automl.generate_summarized_reps.sparse_aggregate",
            return_value="not a matrix",
        ),
        pytest.raises(TypeError, match="Invalid matrix type"),
    ):
        aggregate_matrix(windows, matrix, "sum", 3)


def test_tabularize_static_invalid_label_dir():
    """tabularize_static raises when input_label_dir is set but not a directory (line 78)."""
    from MEDS_tabular_automl.scripts.tabularize_static import main

    cfg = MagicMock()
    cfg.tqdm = False
    cfg.do_overwrite = False
    cfg.input_label_dir = "/nonexistent/path"

    with pytest.raises(ValueError, match="not a directory"):
        main.__wrapped__(cfg)


def test_tabularize_static_with_overwrite(tmp_path):
    """Exercise the compute_fn/write_fn closures in tabularize_static (lines 84-102).

    Uses the same pipeline as test_tabularize.py but with do_overwrite=True to force rwlock_wrap to call
    compute_fn instead of reading the cached result.
    """
    import json
    from io import StringIO
    from pathlib import Path

    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    from MEDS_tabular_automl.scripts import describe_codes, tabularize_static

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    data = (
        "subject_id,code,time,numeric_value\n"
        "1,HEIGHT,,175.0\n"
        "1,EYE_COLOR//BROWN,,\n"
        "1,TEMP,2021-01-01T00:00:00.000000,98.6\n"
    )
    fp = input_dir / "train" / "0.parquet"
    fp.parent.mkdir(parents=True)
    pl.read_csv(StringIO(data)).with_columns(
        pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f")
    ).write_parquet(fp)
    json.dump({"train/0": [1]}, (input_dir / ".shards.json").open("w"))

    shared = {
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "do_overwrite": False,
        "seed": 1,
        "tqdm": False,
    }
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../src/MEDS_tabular_automl/configs/"):
        cfg = compose(config_name="describe_codes", overrides=[f"{k}={v}" for k, v in shared.items()])
    describe_codes.main(cfg)

    tab_config = {
        **shared,
        "tabularization.min_code_inclusion_count": 1,
        "tabularization.window_sizes": "[full]",
        # Set filtered_code_metadata_fp to a different path than input_code_metadata_fp
        # so rwlock_wrap sees a missing output and actually runs compute_fn
        "tabularization.filtered_code_metadata_fp": str(
            (output_dir / "metadata" / "filtered_codes.parquet").resolve()
        ),
    }
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../src/MEDS_tabular_automl/configs/"):
        cfg = compose(config_name="tabularization", overrides=[f"{k}={v}" for k, v in tab_config.items()])

    assert Path(cfg.input_code_metadata_fp).exists(), f"Missing: {cfg.input_code_metadata_fp}"
    tabularize_static.main(cfg)
    output_files = list((output_dir / "tabularize").glob("**/*.npz"))
    assert len(output_files) > 0

    # Now test with input_label_dir set (lines 126-127)
    label_dir = tmp_path / "labels" / "train"
    label_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "subject_id": [1],
            "prediction_time": pl.Series(["2021-01-01"]).str.strptime(pl.Datetime),
            "boolean_value": [True],
        }
    ).write_parquet(label_dir / "0.parquet")

    tab_config_with_labels = {
        **tab_config,
        "do_overwrite": True,
        "input_label_dir": str((tmp_path / "labels").resolve()),
    }
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../src/MEDS_tabular_automl/configs/"):
        cfg2 = compose(
            config_name="tabularization",
            overrides=[f"{k}={v}" for k, v in tab_config_with_labels.items()],
        )
    tabularize_static.main(cfg2)


def test_tabularize_time_series_empty_summary(tmp_path):
    """Exercise the empty summary_df check (line 110) by mocking generate_summary to return empty."""
    import json
    from io import StringIO

    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    from MEDS_tabular_automl.scripts import describe_codes, tabularize_time_series

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    data = (
        "subject_id,code,time,numeric_value\n"
        "1,A,2021-01-01T00:00:00.000000,1.0\n"
        "1,B,2021-01-02T00:00:00.000000,2.0\n"
    )
    fp = input_dir / "train" / "0.parquet"
    fp.parent.mkdir(parents=True)
    pl.read_csv(StringIO(data)).with_columns(
        pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f")
    ).write_parquet(fp)
    json.dump({"train/0": [1]}, (input_dir / ".shards.json").open("w"))

    shared = {
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "do_overwrite": False,
        "seed": 1,
        "tqdm": False,
    }
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../src/MEDS_tabular_automl/configs/"):
        cfg = compose(config_name="describe_codes", overrides=[f"{k}={v}" for k, v in shared.items()])
    describe_codes.main(cfg)

    tab_config = {
        **shared,
        "tabularization.min_code_inclusion_count": 1,
        "tabularization.window_sizes": "[full]",
    }
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../src/MEDS_tabular_automl/configs/"):
        cfg = compose(config_name="tabularization", overrides=[f"{k}={v}" for k, v in tab_config.items()])

    # Mock generate_summary to return an empty matrix (0 columns)
    empty_matrix = sp.csr_matrix((0, 0))
    with (
        patch(
            "MEDS_tabular_automl.scripts.tabularize_time_series.generate_summary",
            return_value=empty_matrix,
        ),
        pytest.raises(ValueError, match="No data found in the summarized dataframe"),
    ):
        tabularize_time_series.main(cfg)


def test_tabularize_time_series_invalid_label_dir():
    """tabularize_time_series raises when input_label_dir not a directory (line 64)."""
    from MEDS_tabular_automl.scripts.tabularize_time_series import main

    cfg = MagicMock()
    cfg.tqdm = False
    cfg.do_overwrite = False
    cfg.input_label_dir = "/nonexistent/path"

    with pytest.raises(ValueError, match="not a directory"):
        main.__wrapped__(cfg)


# ============================================================================
# launch_autogluon.py
# ============================================================================


def test_launch_autogluon_import_error():
    """launch_autogluon raises ImportError when autogluon not available (lines 26-27)."""
    import sys

    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"task_name": "test"})

    # Temporarily remove autogluon from sys.modules
    saved = {}
    for key in list(sys.modules):
        if "autogluon" in key:
            saved[key] = sys.modules.pop(key)

    with (
        patch.dict(sys.modules, {"autogluon": None, "autogluon.tabular": None}),
        pytest.raises(ImportError, match="AutoGluon could not be imported"),
    ):
        # Need to reimport to pick up the patched sys.modules
        import importlib

        import MEDS_tabular_automl.scripts.launch_autogluon as ag_mod

        importlib.reload(ag_mod)
        ag_mod.main.__wrapped__(cfg)

    # Restore
    sys.modules.update(saved)


def test_launch_autogluon_full_flow(tmp_path):
    """Exercise the entire launch_autogluon main function with mocked dependencies."""
    import json
    import sys
    from types import ModuleType

    from omegaconf import OmegaConf

    # Create a fake autogluon module
    fake_ag = ModuleType("autogluon")
    fake_ag_tab = ModuleType("autogluon.tabular")

    mock_predictor = MagicMock()
    mock_predictor.predict = MagicMock(return_value=MagicMock())
    mock_predictor.evaluate = MagicMock(return_value=0.85)
    fake_ag_tab.TabularPredictor = MagicMock(return_value=mock_predictor)
    mock_predictor.fit = MagicMock(return_value=mock_predictor)
    fake_ag_tab.TabularDataset = MagicMock(side_effect=lambda df: df)
    fake_ag.tabular = fake_ag_tab

    # Create fake dense data that DenseIterator.densify() would return
    fake_data = sp.csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    fake_labels = np.array([0, 1])
    mock_iterator = MagicMock()
    mock_iterator.densify = MagicMock(return_value=(fake_data, fake_labels))

    # Build config
    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    cfg = OmegaConf.create(
        {
            "task_name": "test_task",
            "time_output_model_dir": str(tmp_path / "model"),
            "path": {
                "sweep_results_dir": str(sweep_dir),
                "config_log_stem": "config",
                "performance_log_stem": "perf",
                "time_output_model_dir": str(tmp_path / "model"),
            },
            "tabularization": {},
            "model_launcher": {},
        }
    )

    with (
        patch.dict(sys.modules, {"autogluon": fake_ag, "autogluon.tabular": fake_ag_tab}),
        patch(
            "MEDS_tabular_automl.scripts.launch_autogluon.DenseIterator",
            return_value=mock_iterator,
        ),
    ):
        from MEDS_tabular_automl.scripts.launch_autogluon import main

        main.__wrapped__(cfg)

    # Verify the performance log was written
    perf_log = sweep_dir / "perf.json"
    assert perf_log.exists()

    perf = json.loads(perf_log.read_text())
    assert perf["score"] == 0.85
