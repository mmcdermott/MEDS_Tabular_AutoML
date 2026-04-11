"""Tests for all remaining uncovered code paths to reach 100% coverage.

Each test targets specific uncovered lines identified via coverage reporting.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp
from mixins import TimeableMixin

from MEDS_tabular_automl.tabular_dataset import TabularDataset


def _make_dataset(labels=None, num_shards=1):
    """Create a minimal TabularDataset for unit testing."""
    ds = TabularDataset.__new__(TabularDataset)
    ds.cfg = MagicMock()
    ds.split = "train"
    ds._data_shards = [f"shard_{i}" for i in range(num_shards)]
    ds.codes_set = {0, 1, 2}
    ds.code_masks = {"code/count": [True, True, True]}
    ds.num_features = 3
    ds.valid_event_ids = None
    ds.labels = labels
    ds.imputer = None
    ds.scaler = None
    TimeableMixin.__init__(ds)
    return ds


# ============================================================================
# tabular_dataset.py — remaining uncovered lines
# ============================================================================


def test_load_matrix_bad_array_shape(tmp_path):
    """_load_matrix raises ValueError when npz has wrong array dimensions (line 113)."""
    ds = _make_dataset()
    fp = tmp_path / "bad.npz"
    bad_array = np.zeros((2, 5))  # should be 3 rows
    np.savez(fp, array=bad_array, shape=np.array([5, 5]))
    with pytest.raises(ValueError, match="Expected array to have 3 rows"):
        ds._load_matrix(fp)


def test_init_empty_labels_after_loading(tmp_path):
    """__init__ raises ValueError when labels dict is empty after loading (line 74)."""
    label_dir = tmp_path / "labels" / "train"
    label_dir.mkdir(parents=True)
    pl.DataFrame({"subject_id": [1], "boolean_value": [True]}).write_parquet(label_dir / "0.parquet")

    cfg = MagicMock()
    cfg.path.input_label_cache_dir = str(tmp_path / "labels")

    with (
        patch.object(TabularDataset, "_get_code_set", return_value=({0}, {"code/count": [True]}, 1)),
        patch.object(TabularDataset, "_set_scaler"),
        patch.object(TabularDataset, "_set_imputer"),
        patch.object(TabularDataset, "_load_ids_and_labels", return_value=({}, {})),
        pytest.raises(ValueError, match="No labels found"),
    ):
        TabularDataset(cfg, "train")


def test_get_code_set_with_max_by_correlation():
    """_get_code_set correlation-based feature selection (lines 195-201)."""
    ds = _make_dataset()
    ds.cfg.tabularization.aggs = ["code/count"]
    ds.cfg.tabularization.filtered_code_metadata_fp = "fake.parquet"
    ds.cfg.tabularization.max_by_correlation = 2
    ds.cfg.tabularization.min_correlation = None

    # Mock dependencies
    feature_columns = ["A/code", "B/code", "C/code"]
    with (
        patch("MEDS_tabular_automl.tabular_dataset.get_feature_columns", return_value=feature_columns),
        patch("MEDS_tabular_automl.tabular_dataset.get_feature_indices", return_value=[0, 1, 2]),
    ):
        ds.cfg.tabularization._resolved_codes = {"A/code", "B/code", "C/code"}
        # Mock _get_shard_by_index to return data with known correlations
        X = sp.csc_matrix(np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]))
        y = np.array([1, 0, 1, 0])
        ds._get_shard_by_index = MagicMock(return_value=(X, y))

        codes_set, _code_masks, _num_features = ds._get_code_set()
        # Should select top 2 by correlation
        assert len(codes_set) <= 2


def test_get_code_set_with_min_correlation():
    """_get_code_set min_correlation filtering (lines 203-207)."""
    ds = _make_dataset()
    ds.cfg.tabularization.aggs = ["code/count"]
    ds.cfg.tabularization.filtered_code_metadata_fp = "fake.parquet"
    ds.cfg.tabularization.max_by_correlation = None
    ds.cfg.tabularization.min_correlation = 0.5

    feature_columns = ["A/code", "B/code"]
    with (
        patch("MEDS_tabular_automl.tabular_dataset.get_feature_columns", return_value=feature_columns),
        patch("MEDS_tabular_automl.tabular_dataset.get_feature_indices", return_value=[0, 1]),
    ):
        ds.cfg.tabularization._resolved_codes = {"A/code", "B/code"}
        X = sp.csc_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]))
        y = np.array([1, 0, 1, 0])
        ds._get_shard_by_index = MagicMock(return_value=(X, y))

        codes_set, _code_masks, _num_features = ds._get_code_set()
        # Both features have abs(corr) > 0.5
        assert len(codes_set) == 2


def test_set_imputer_with_partial_fit():
    """_set_imputer partial_fit branch (lines 263-265)."""
    ds = _make_dataset(num_shards=2)
    mock_imputer = MagicMock()
    mock_imputer.partial_fit = MagicMock()
    ds.cfg.data_loading_params.imputer.imputer_target = mock_imputer
    ds._get_shard_by_index = MagicMock(return_value=(sp.csc_matrix(np.eye(3)), np.array([1, 0, 1])))

    ds._set_imputer()
    assert mock_imputer.partial_fit.call_count == 2


def test_set_scaler_with_fit():
    """_set_scaler fit-only branch (lines 286-287)."""
    ds = _make_dataset()
    mock_scaler = MagicMock()
    mock_scaler.fit = MagicMock()
    del mock_scaler.partial_fit
    ds.cfg.data_loading_params.normalization.normalizer = mock_scaler
    ds._get_shard_by_index = MagicMock(return_value=(sp.csc_matrix(np.eye(3)), np.array([1, 0, 1])))

    ds._set_scaler()
    mock_scaler.fit.assert_called_once()


def test_get_shard_reloads_labels(tmp_path):
    """_get_shard_by_index reloads labels when None (line 371)."""
    ds = _make_dataset()
    ds.labels = None
    ds._load_labels = MagicMock(return_value={"shard_0": np.array([1, 0])})
    ds._get_dynamic_shard_by_index = MagicMock(return_value=sp.csc_matrix(np.eye(2)))

    _x, y = ds._get_shard_by_index(0)
    ds._load_labels.assert_called_once()
    assert ds.labels is not None
    np.testing.assert_array_equal(y, np.array([1, 0]))


def test_get_data_shards_empty_raises():
    """get_data_shards with empty index list raises ValueError (line 420)."""
    ds = _make_dataset()
    with pytest.raises(ValueError, match="No data found"):
        ds.get_data_shards([])


def test_get_data_shard_count():
    """get_data_shard_count returns number of shards (line 441)."""
    ds = _make_dataset(num_shards=3)
    assert ds.get_data_shard_count() == 3


def test_get_all_column_names():
    """get_all_column_names constructs feature names from file structure (lines 462-478)."""
    ds = _make_dataset()
    ds.cfg.tabularization.filtered_code_metadata_fp = "fake.parquet"

    # Create fake file structure: window/code_type/agg_name.npz
    fake_file = Path("/fake/30d/code/count.npz")
    with (
        patch("MEDS_tabular_automl.tabular_dataset.get_model_files", return_value=[fake_file]),
        patch("MEDS_tabular_automl.tabular_dataset.get_feature_columns", return_value=["A/code", "B/code"]),
        patch("MEDS_tabular_automl.tabular_dataset.get_feature_indices", return_value=[0, 1]),
    ):
        names = ds.get_all_column_names()
        assert names == ["A/code/count/30d", "B/code/count/30d"]


def test_get_column_names_with_indices():
    """get_column_names with indices parameter filters results (lines 486-505)."""
    ds = _make_dataset()
    ds.cfg.tabularization.filtered_code_metadata_fp = "fake.parquet"

    fake_file = Path("/fake/30d/code/count.npz")
    with (
        patch("MEDS_tabular_automl.tabular_dataset.get_model_files", return_value=[fake_file]),
        patch("MEDS_tabular_automl.tabular_dataset.get_feature_columns", return_value=["A/code", "B/code"]),
        patch("MEDS_tabular_automl.tabular_dataset.get_feature_indices", return_value=[0, 1]),
    ):
        # Without indices — returns all
        all_names = ds.get_column_names()
        assert len(all_names) == 2

        # With indices — returns subset
        filtered = ds.get_column_names(indices=[0])
        assert filtered == ["A/code/count/30d"]


def test_densify():
    """Densify() iterates shards and returns dense data + labels (lines 511-519)."""
    ds = _make_dataset(labels={"shard_0": np.array([1, 0])}, num_shards=1)
    shard_data = sp.csc_matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    ds._get_shard_by_index = MagicMock(return_value=(shard_data, np.array([1, 0])))

    data, labels = ds.densify()
    assert data.shape == (2, 2)
    np.testing.assert_array_equal(labels, [1, 0])


# ============================================================================
# sklearn_model.py — remaining uncovered lines
# ============================================================================


def test_sklearn_evaluate_train_split():
    """SklearnModel.evaluate with train split (lines 152-153)."""
    from MEDS_tabular_automl.sklearn_model import SklearnMatrix, SklearnModel

    sklearn_model = SklearnModel.__new__(SklearnModel)
    sklearn_model.keep_data_in_memory = True
    sklearn_model.model = MagicMock()
    sklearn_model.model.predict_proba = MagicMock(
        return_value=np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.7, 0.3]])
    )
    sklearn_model.dtrain = SklearnMatrix(sp.csr_matrix(np.eye(4)), np.array([1, 0, 1, 0]))
    sklearn_model.dtuning = None
    sklearn_model.dheld_out = None
    sklearn_model.itrain = None
    sklearn_model.ituning = None
    sklearn_model.iheld_out = None

    auc = sklearn_model.evaluate(split="train")
    assert 0.0 <= auc <= 1.0


def test_sklearn_evaluate_empty_predictions():
    """SklearnModel.evaluate raises on empty predictions (line 177)."""
    from MEDS_tabular_automl.sklearn_model import SklearnMatrix, SklearnModel

    sklearn_model = SklearnModel.__new__(SklearnModel)
    sklearn_model.keep_data_in_memory = True
    sklearn_model.model = MagicMock()
    sklearn_model.model.predict_proba = MagicMock(return_value=np.array([]).reshape(0, 2))
    sklearn_model.dtuning = SklearnMatrix(sp.csr_matrix((0, 3)), np.array([]))
    sklearn_model.ituning = None

    with pytest.raises(ValueError, match="Predictions or true labels are empty"):
        sklearn_model.evaluate(split="tuning")


def test_sklearn_save_with_save_model_method(tmp_path):
    """SklearnModel.save_model calls model.save_model when available (line 195)."""
    from MEDS_tabular_automl.sklearn_model import SklearnModel

    mock_model = MagicMock()
    mock_model.save_model = MagicMock()
    sklearn_model = SklearnModel.__new__(SklearnModel)
    sklearn_model.model = mock_model

    fp = tmp_path / "model.json"
    sklearn_model.save_model(fp)
    mock_model.save_model.assert_called_once_with(fp)


# ============================================================================
# xgboost_model.py — remaining uncovered lines
# ============================================================================


def test_xgboost_predict_train_split():
    """XGBoostModel._predict with train split (lines 146-147)."""
    from MEDS_tabular_automl.xgboost_model import XGBoostModel

    model = XGBoostModel.__new__(XGBoostModel)
    model.model = MagicMock()
    model.model.predict = MagicMock(return_value=np.array([0.9, 0.1]))
    mock_dmatrix = MagicMock()
    mock_dmatrix.get_label = MagicMock(return_value=np.array([1, 0]))
    model.dtrain = mock_dmatrix
    model.dtuning = None
    model.dheld_out = None

    y_true, y_pred = model._predict(split="train")
    np.testing.assert_array_equal(y_true, [1, 0])
    np.testing.assert_array_equal(y_pred, [0.9, 0.1])


def _setup_xgb_predict(split_name):
    """Helper to set up XGBoostModel for predict() testing."""
    from MEDS_tabular_automl.xgboost_model import XGBoostModel

    model = XGBoostModel.__new__(XGBoostModel)
    model.cfg = MagicMock()
    model.cfg.path.input_label_cache_dir = "/fake"
    model.model = MagicMock()
    model.model.predict = MagicMock(return_value=np.array([0.9, 0.1]))

    mock_dmatrix = MagicMock()
    mock_dmatrix.get_label = MagicMock(return_value=np.array([1.0, 0.0]))

    mock_iter = MagicMock()
    mock_iter._load_ids_and_labels = MagicMock(return_value=(None, {"shard0": [1, 0]}))

    # Set all split attributes
    for attr in ("dtrain", "dtuning", "dheld_out"):
        setattr(model, attr, mock_dmatrix)
    for attr in ("itrain", "ituning", "iheld_out"):
        setattr(model, attr, mock_iter)

    return model


def _run_xgb_predict(model, split_name):
    """Run predict() with mocked parquet reading."""
    labels_df = pl.DataFrame(
        {
            "subject_id": [1, 2],
            "prediction_time": pl.Series(["2021-01-01", "2021-01-02"]).str.strptime(pl.Datetime),
            "boolean_value": [True, False],
        }
    )
    with patch("MEDS_tabular_automl.xgboost_model.pl.read_parquet", return_value=labels_df):
        return model.predict(split=split_name)


def test_xgboost_predict_held_out():
    """XGBoostModel.predict with held_out split (line 163)."""
    model = _setup_xgb_predict("held_out")
    result = _run_xgb_predict(model, "held_out")
    assert result.shape[0] == 2


def test_xgboost_predict_tuning():
    """XGBoostModel.predict with tuning split (line 161)."""
    model = _setup_xgb_predict("tuning")
    result = _run_xgb_predict(model, "tuning")
    assert result.shape[0] == 2


def test_xgboost_predict_train():
    """XGBoostModel.predict with train split (line 165)."""
    model = _setup_xgb_predict("train")
    result = _run_xgb_predict(model, "train")
    assert result.shape[0] == 2


def test_xgboost_predict_invalid_split():
    """XGBoostModel.predict with invalid split (line 167)."""
    model = _setup_xgb_predict("held_out")
    model._predict = MagicMock(return_value=(np.array([1, 0]), np.array([0.9, 0.1])))
    with pytest.raises(ValueError, match="Invalid split"):
        model.predict(split="invalid")


def test_xgboost_predict_label_mismatch():
    """XGBoostModel.predict raises on label mismatch (lines 191-192)."""
    model = _setup_xgb_predict("held_out")
    # Make predictions not match labels: predict returns [0, 1] but labels are [True, False]
    model.model.predict = MagicMock(return_value=np.array([0.0, 1.0]))
    mock_dmatrix = MagicMock()
    mock_dmatrix.get_label = MagicMock(return_value=np.array([0.0, 1.0]))  # swapped
    model.dheld_out = mock_dmatrix

    with pytest.raises(ValueError, match="Label mismatch"):
        _run_xgb_predict(model, "held_out")


# ============================================================================
# generate_static_features.py — remaining uncovered lines
# ============================================================================


def test_get_flat_static_rep_no_features():
    """get_flat_static_rep raises when no static features found (line 252)."""
    from MEDS_tabular_automl.generate_static_features import get_flat_static_rep

    shard_df = pl.DataFrame({"subject_id": [1], "code": ["A"], "numeric_value": [1.0]}).lazy()
    # Feature columns with no matching static features for the given agg
    with (
        patch("MEDS_tabular_automl.generate_static_features.get_feature_names", return_value=[]),
        pytest.raises(ValueError, match="No static features found"),
    ):
        get_flat_static_rep("static/first", ["A/code"], shard_df, None)


def test_get_flat_static_rep_shape_mismatch():
    """get_flat_static_rep raises on feature count mismatch (line 261)."""
    from MEDS_tabular_automl.generate_static_features import get_flat_static_rep

    shard_df = pl.DataFrame({"subject_id": [1, 2], "code": ["A", "B"], "numeric_value": [1.0, 2.0]}).lazy()

    # Return 2 features but mock matrix to have wrong shape
    bad_matrix = sp.coo_array(np.zeros((2, 1)))  # 1 col instead of 2
    with (
        patch(
            "MEDS_tabular_automl.generate_static_features.get_feature_names",
            return_value=["A/static/first", "B/static/first"],
        ),
        patch(
            "MEDS_tabular_automl.generate_static_features.summarize_static_measurements",
            return_value=pl.DataFrame({"subject_id": [1, 2]}),
        ),
        patch(
            "MEDS_tabular_automl.generate_static_features.get_sparse_static_rep",
            return_value=bad_matrix,
        ),
        patch(
            "MEDS_tabular_automl.generate_static_features.get_unique_time_events_df",
            return_value=pl.DataFrame({"subject_id": [1, 2], "time": ["2021", "2021"]}).lazy(),
        ),
        patch(
            "MEDS_tabular_automl.generate_static_features.get_events_df",
            return_value=shard_df,
        ),
        pytest.raises(ValueError, match="Expected 2 features, got 1"),
    ):
        get_flat_static_rep("static/first", ["A/static/first", "B/static/first"], shard_df, None)


# ============================================================================
# scripts/cache_task.py — remaining uncovered lines
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
# These are Hydra-wrapped functions that are exercised by integration tests
# via subprocess but not counted in coverage. We test the validation paths.
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
# launch_autogluon.py — full function body with mocked autogluon
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
    import sys
    from types import ModuleType
    from unittest.mock import patch

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
    import json

    perf = json.loads(perf_log.read_text())
    assert perf["score"] == 0.85
