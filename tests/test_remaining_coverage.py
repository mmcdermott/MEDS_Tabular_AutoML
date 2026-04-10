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
from scipy.sparse import csr_array

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

        codes_set, code_masks, num_features = ds._get_code_set()
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

        codes_set, code_masks, num_features = ds._get_code_set()
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

    X, y = ds._get_shard_by_index(0)
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
    """densify() iterates shards and returns dense data + labels (lines 511-519)."""
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
    labels_df = pl.DataFrame({
        "subject_id": [1, 2],
        "prediction_time": pl.Series(["2021-01-01", "2021-01-02"]).str.strptime(pl.Datetime),
        "boolean_value": [True, False],
    })
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
        patch(
            "MEDS_tabular_automl.generate_static_features.get_feature_names", return_value=[]
        ),
        pytest.raises(ValueError, match="No static features found"),
    ):
        get_flat_static_rep("static/first", ["A/code"], shard_df, None)


def test_get_flat_static_rep_shape_mismatch():
    """get_flat_static_rep raises on feature count mismatch (line 261)."""
    from MEDS_tabular_automl.generate_static_features import get_flat_static_rep

    shard_df = pl.DataFrame(
        {"subject_id": [1, 2], "code": ["A", "B"], "numeric_value": [1.0, 2.0]}
    ).lazy()

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
    pl.DataFrame({
        "subject_id": [1],
        "prediction_time": pl.Series(["2021-01-01"]).str.strptime(pl.Datetime),
        "boolean_value": [True],
    }).write_parquet(label_dir / "0.parquet")

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

    with pytest.raises(ValueError, match="numeric_value.*column not found"):
        main.__wrapped__(cfg)


# ============================================================================
# scripts/tabularize_static.py and tabularize_time_series.py
# These are Hydra-wrapped functions that are exercised by integration tests
# via subprocess but not counted in coverage. We test the validation paths.
# ============================================================================


def test_tabularize_static_invalid_label_dir():
    """tabularize_static raises when input_label_dir is set but not a directory (line 78)."""
    from MEDS_tabular_automl.scripts.tabularize_static import main

    cfg = MagicMock()
    cfg.tqdm = False
    cfg.do_overwrite = False
    cfg.input_label_dir = "/nonexistent/path"

    with pytest.raises(ValueError, match="not a directory"):
        main.__wrapped__(cfg)


def test_tabularize_time_series_invalid_label_dir():
    """tabularize_time_series raises when input_label_dir not a directory (line 64)."""
    from MEDS_tabular_automl.scripts.tabularize_time_series import main

    cfg = MagicMock()
    cfg.tqdm = False
    cfg.do_overwrite = False
    cfg.input_label_dir = "/nonexistent/path"

    with pytest.raises(ValueError, match="not a directory"):
        main.__wrapped__(cfg)
