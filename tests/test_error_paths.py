"""Tests for error handling and validation that can't easily be expressed as doctests.

These tests cover error paths that require complex setup (mocking, file I/O, multi-step construction) that
would be unwieldy as inline doctests.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.sparse import csr_array

from MEDS_tabular_automl.generate_static_features import (
    get_sparse_static_rep,
    summarize_static_measurements,
)
from MEDS_tabular_automl.generate_summarized_reps import generate_summary
from MEDS_tabular_automl.generate_ts_features import summarize_dynamic_measurements

# ============================================================================
# generate_static_features.py
# ============================================================================


def test_get_sparse_static_rep_unsorted():
    """Static data must be sorted by subject_id; unsorted raises ValueError."""
    import polars as pl

    static_df = pl.DataFrame({"subject_id": [2, 1], "A": [1.0, 2.0], "B": [3.0, 4.0]}).lazy()
    meds_df = pl.DataFrame({"subject_id": [1, 2], "code": ["A", "B"]}).lazy()
    with pytest.raises(ValueError, match="not sorted by subject_id"):
        get_sparse_static_rep(["A", "B"], static_df, meds_df, ["A/static/present", "B/static/present"])


def test_get_sparse_static_rep_duplicate_subjects():
    """Static data must have unique subject_ids; duplicates raise ValueError."""
    import polars as pl

    static_df = pl.DataFrame({"subject_id": [1, 1], "A": [1.0, 2.0], "B": [3.0, 4.0]}).lazy()
    meds_df = pl.DataFrame({"subject_id": [1, 1], "code": ["A", "B"]}).lazy()
    with pytest.raises(ValueError, match="duplicate subject_id"):
        get_sparse_static_rep(["A", "B"], static_df, meds_df, ["A/static/present", "B/static/present"])


def test_summarize_static_invalid_aggregation():
    """Aggregation type must be static/first or static/present."""
    import polars as pl

    df = pl.DataFrame({"subject_id": [1], "code": ["A"], "numeric_value": [1.0]}).lazy()
    with pytest.raises(ValueError, match="Invalid aggregation type"):
        summarize_static_measurements("invalid_agg", ["A/static/first"], df)


# ============================================================================
# generate_summarized_reps.py
# ============================================================================


def test_generate_summary_rejects_invalid_aggregation():
    """generate_summary validates aggregation type against CODE/VALUE_AGGREGATIONS."""
    import polars as pl

    m = csr_array(np.eye(3))
    df = pl.DataFrame({"subject_id": [1], "time": ["2021-01-01"]}).lazy()
    with pytest.raises(ValueError, match="Invalid aggregation"):
        generate_summary(
            agg="invalid/agg", feature_columns=["A/code"], index_df=df, matrix=m, window_size="full"
        )


def test_generate_summary_rejects_empty_features():
    """generate_summary requires a non-empty feature_columns list."""
    import polars as pl

    m = csr_array(np.eye(3))
    df = pl.DataFrame({"subject_id": [1], "time": ["2021-01-01"]}).lazy()
    with pytest.raises(ValueError, match="No feature columns provided"):
        generate_summary(agg="code/count", feature_columns=[], index_df=df, matrix=m, window_size="full")


def test_generate_summary_rejects_mismatched_columns():
    """generate_summary raises when no feature columns match the aggregation type."""
    import polars as pl

    m = csr_array(np.eye(3))
    df = pl.DataFrame({"subject_id": [1], "time": ["2021-01-01"]}).lazy()
    with pytest.raises(ValueError, match="No columns found for aggregation"):
        generate_summary(
            agg="code/count", feature_columns=["A/value"], index_df=df, matrix=m, window_size="full"
        )


# ============================================================================
# generate_ts_features.py
# ============================================================================


def test_summarize_dynamic_unsorted():
    """Time-series data must be sorted by subject_id and time."""
    import polars as pl

    df = pl.DataFrame(
        {
            "subject_id": [2, 1],
            "time": pl.Series(["2021-01-02", "2021-01-01"]).str.strptime(pl.Date),
            "code": ["A", "B"],
            "numeric_value": [1.0, 2.0],
        }
    ).lazy()
    with pytest.raises(ValueError, match="must be sorted by subject_id and time"):
        summarize_dynamic_measurements("code/count", ["A/code", "B/code"], df)


# ============================================================================
# sklearn_model.py
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
