"""Tests for error handling and validation that can't easily be expressed as doctests.

These tests cover error paths that require complex setup (mocking, file I/O, multi-step construction) that
would be unwieldy as inline doctests.
"""

from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

# ============================================================================
# sklearn_model.py
#
# Note: error path tests for module-level functions in generate_static_features,
# generate_summarized_reps, and generate_ts_features have been moved to
# doctests on the source functions for documentation value.
# ============================================================================


def test_sklearn_model_rejects_model_without_fit():
    """SklearnModel.__init__ validates the model has a fit method."""
    from MEDS_tabular_automl.sklearn_model import SklearnModel

    cfg = MagicMock()
    cfg.model = object()
    with pytest.raises(ValueError, match="does not have a fit method"):
        SklearnModel(cfg)


def test_sklearn_evaluate_rejects_invalid_split():
    """SklearnModel.evaluate raises for splits other than train/tuning/held_out."""
    from MEDS_tabular_automl.sklearn_model import SklearnModel

    sklearn_model = SklearnModel.__new__(SklearnModel)
    sklearn_model.model = MagicMock()
    sklearn_model.keep_data_in_memory = True
    with pytest.raises(ValueError, match="not valid"):
        sklearn_model.evaluate(split="nonexistent_split")


def test_sklearn_evaluate_rejects_model_without_predict_proba():
    """SklearnModel.evaluate validates the model has predict_proba."""
    from MEDS_tabular_automl.sklearn_model import SklearnModel

    model = MagicMock(spec=[])
    sklearn_model = SklearnModel.__new__(SklearnModel)
    sklearn_model.model = model
    sklearn_model.keep_data_in_memory = True
    sklearn_model.dtuning = MagicMock()
    sklearn_model.ituning = MagicMock()
    with pytest.raises(ValueError, match="does not have a predict_proba method"):
        sklearn_model.evaluate(split="tuning")


def test_sklearn_partial_fit_rejects_model_without_method():
    """SklearnModel._fit_from_partial raises when model lacks partial_fit."""
    from MEDS_tabular_automl.sklearn_model import SklearnModel

    model = MagicMock(spec=["fit", "predict_proba"])
    sklearn_model = SklearnModel.__new__(SklearnModel)
    sklearn_model.model = model
    sklearn_model.cfg = MagicMock()
    sklearn_model.itrain = MagicMock()
    with pytest.raises(ValueError, match="does not support partial_fit"):
        sklearn_model._fit_from_partial()


def test_sklearn_save_model_pickle_fallback(tmp_path):
    """When model has no save_model method, falls back to pickle with .pkl extension."""
    from sklearn.linear_model import SGDClassifier

    from MEDS_tabular_automl.sklearn_model import SklearnModel

    sklearn_model = SklearnModel.__new__(SklearnModel)
    sklearn_model.model = SGDClassifier()

    with pytest.raises(ValueError, match=r"Model file extension must be \.pkl"):
        sklearn_model.save_model(tmp_path / "model.json")

    sklearn_model.save_model(tmp_path / "model.pkl")
    assert (tmp_path / "model.pkl").exists()


# ============================================================================
# xgboost_model.py
# ============================================================================


def test_xgboost_predict_rejects_invalid_split():
    """XGBoostModel._predict raises for invalid split names."""
    from MEDS_tabular_automl.xgboost_model import XGBoostModel

    model = XGBoostModel.__new__(XGBoostModel)
    model.model = MagicMock()
    with pytest.raises(ValueError, match="Invalid split"):
        model._predict(split="invalid_split")


def test_xgboost_evaluate_returns_zero_for_single_class():
    """When ground truth has only one class, AUC is undefined; returns 0.0."""
    from MEDS_tabular_automl.xgboost_model import XGBoostModel

    model = XGBoostModel.__new__(XGBoostModel)
    model._predict = MagicMock(return_value=(np.array([1, 1, 1]), np.array([0.9, 0.8, 0.7])))
    assert model.evaluate(split="tuning") == 0.0


# ============================================================================
# evaluation_callback.py
# ============================================================================


def test_evaluation_callback_raises_on_missing_logs(tmp_path):
    """EvaluationCallback.on_multirun_end raises FileNotFoundError for missing logs."""
    from MEDS_tabular_automl.evaluation_callback import EvaluationCallback

    cb = EvaluationCallback()
    config = MagicMock()
    config.path.sweep_results_dir = str(tmp_path / "nonexistent")
    config.path.performance_log_stem = "perf"
    with pytest.raises(FileNotFoundError, match="Log files incomplete"):
        cb.on_multirun_end(config)


# ============================================================================
# Tests moved from test_remaining_coverage.py
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
    from unittest.mock import patch

    labels_df = pl.DataFrame(
        {
            "subject_id": [1, 2],
            "prediction_time": pl.Series(["2021-01-01", "2021-01-02"]).str.strptime(pl.Datetime),
            "boolean_value": [True, False],
        }
    )
    with patch("MEDS_tabular_automl.xgboost_model.pl.read_parquet", return_value=labels_df):
        return model.predict(split=split_name)


@pytest.mark.parametrize("split", ["held_out", "tuning", "train"])
def test_xgboost_predict_by_split(split):
    """XGBoostModel.predict dispatches correctly for each valid split."""
    model = _setup_xgb_predict(split)
    result = _run_xgb_predict(model, split)
    assert result.shape[0] == 2
    assert "predicted_boolean_probability" in result.columns


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


def test_get_flat_static_rep_no_features():
    """get_flat_static_rep raises when no static features found (line 252)."""
    from unittest.mock import patch

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
    from unittest.mock import patch

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
