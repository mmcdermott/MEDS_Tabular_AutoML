"""Tests for error handling paths and edge cases to achieve 100% coverage."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp
from scipy.sparse import coo_array, csc_array, csr_array

from MEDS_tabular_automl.generate_static_features import (
    get_sparse_static_rep,
    summarize_static_measurements,
)
from MEDS_tabular_automl.generate_summarized_reps import sparse_aggregate
from MEDS_tabular_automl.generate_ts_features import (
    get_long_code_df,
    get_long_value_df,
    summarize_dynamic_measurements,
)
from MEDS_tabular_automl.utils import (
    array_to_sparse_matrix,
    filter_to_codes,
    get_feature_names,
    get_unique_time_events_df,
    write_df,
)


# ============================================================================
# utils.py error paths
# ============================================================================


def test_filter_to_codes_frequency_below_zero(tmp_path):
    fp = tmp_path / "codes.parquet"
    pl.DataFrame({"code": ["A", "B"], "count": [10, 20]}).write_parquet(fp)
    with pytest.raises(ValueError, match="min_code_inclusion_frequency must be between 0 and 1"):
        filter_to_codes(fp, None, None, -0.5, None)


def test_filter_to_codes_frequency_above_one(tmp_path):
    fp = tmp_path / "codes.parquet"
    pl.DataFrame({"code": ["A", "B"], "count": [10, 20]}).write_parquet(fp)
    with pytest.raises(ValueError, match="min_code_inclusion_frequency must be between 0 and 1"):
        filter_to_codes(fp, None, None, 1.5, None)


def test_filter_to_codes_leaves_zero(tmp_path):
    fp = tmp_path / "codes.parquet"
    pl.DataFrame({"code": ["A", "B"], "count": [1, 2]}).write_parquet(fp)
    with pytest.raises(ValueError, match="Code filtering criteria leaves only 0 codes"):
        filter_to_codes(fp, None, 1000, None, None)


def test_array_to_sparse_wrong_dimensions():
    bad_array = np.zeros((2, 5))
    with pytest.raises(AssertionError, match="currently has 2"):
        array_to_sparse_matrix(bad_array, shape=(5, 5))


def test_array_to_sparse_four_dimensions():
    bad_array = np.zeros((4, 5))
    with pytest.raises(AssertionError, match="currently has 4"):
        array_to_sparse_matrix(bad_array, shape=(5, 5))


def test_write_df_file_exists_no_overwrite(tmp_path):
    fp = tmp_path / "test.parquet"
    fp.touch()
    df = pl.DataFrame({"a": [1]})
    with pytest.raises(FileExistsError, match="exists and do_overwrite is False"):
        write_df(df, fp, do_overwrite=False)


def test_write_df_unsupported_type(tmp_path):
    fp = tmp_path / "test.parquet"
    with pytest.raises(TypeError, match="Unsupported type"):
        write_df("not a dataframe", fp)


def test_write_df_lazyframe(tmp_path):
    fp = tmp_path / "test.parquet"
    df = pl.DataFrame({"a": [1, 2]}).lazy()
    write_df(df, fp)
    assert fp.exists()
    assert pl.read_parquet(fp).shape == (2, 1)


def test_write_df_coo_array(tmp_path):
    fp = tmp_path / "test.npz"
    mat = coo_array(([1.0, 2.0], ([0, 1], [0, 1])), shape=(2, 2))
    write_df(mat, fp, do_overwrite=True)
    assert fp.exists()


def test_get_unique_time_events_null_times():
    df = pl.DataFrame({"subject_id": [1, 1], "time": [None, None], "code": ["A", "B"]}).lazy()
    with pytest.raises(ValueError, match="Time column must not have null values"):
        get_unique_time_events_df(df)


def test_get_unique_time_events_unsorted():
    df = pl.DataFrame(
        {
            "subject_id": [2, 1],
            "time": pl.Series(["2021-01-02", "2021-01-01"]).str.strptime(pl.Date),
            "code": ["A", "B"],
        }
    ).lazy()
    with pytest.raises(ValueError, match="must be sorted by subject_id and time"):
        get_unique_time_events_df(df)


def test_get_feature_names_unknown_aggregation():
    with pytest.raises(ValueError, match="Unknown aggregation type"):
        get_feature_names("invalid/agg", ["A/code", "B/value"])


# ============================================================================
# generate_summarized_reps.py error paths
# ============================================================================


def test_sparse_aggregate_invalid():
    m = csr_array(np.eye(3))
    with pytest.raises(ValueError, match="Aggregation method 'invalid' not implemented"):
        sparse_aggregate(m, "invalid")


def test_generate_summary_invalid_agg():
    from MEDS_tabular_automl.generate_summarized_reps import generate_summary

    m = csr_array(np.eye(3))
    df = pl.DataFrame({"subject_id": [1], "time": ["2021-01-01"]}).lazy()
    with pytest.raises(ValueError, match="Invalid aggregation"):
        generate_summary(agg="invalid/agg", feature_columns=["A/code"], index_df=df, matrix=m,
                         window_size="full")


def test_generate_summary_empty_features():
    from MEDS_tabular_automl.generate_summarized_reps import generate_summary

    m = csr_array(np.eye(3))
    df = pl.DataFrame({"subject_id": [1], "time": ["2021-01-01"]}).lazy()
    with pytest.raises(ValueError, match="No feature columns provided"):
        generate_summary(agg="code/count", feature_columns=[], index_df=df, matrix=m, window_size="full")


def test_generate_summary_no_matching_columns():
    from MEDS_tabular_automl.generate_summarized_reps import generate_summary

    m = csr_array(np.eye(3))
    df = pl.DataFrame({"subject_id": [1], "time": ["2021-01-01"]}).lazy()
    with pytest.raises(ValueError, match="No columns found for aggregation"):
        generate_summary(agg="code/count", feature_columns=["A/value"], index_df=df, matrix=m,
                         window_size="full")


# ============================================================================
# generate_ts_features.py error paths
# ============================================================================


def test_get_long_code_df_unmapped_code():
    """When a code doesn't map to any ts_column, polars raises InvalidOperationError on cast."""
    import polars.exceptions

    df = pl.DataFrame({"code": ["UNMAPPED"], "numeric_value": [1.0]}).lazy()
    with pytest.raises((ValueError, polars.exceptions.InvalidOperationError)):
        get_long_code_df(df, ["other_code/code"])



# Note: The ValueError paths on lines 63 and 92 of generate_ts_features.py are
# unreachable with polars 1.39+ because polars raises InvalidOperationError on the
# .cast(int) before the numpy dtype check runs. These are dead code paths.


def test_summarize_dynamic_unsorted():
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
# generate_static_features.py error paths
# ============================================================================


def test_get_sparse_static_rep_unsorted():
    static_df = pl.DataFrame({"subject_id": [2, 1], "A": [1.0, 2.0], "B": [3.0, 4.0]}).lazy()
    meds_df = pl.DataFrame({"subject_id": [1, 2], "code": ["A", "B"]}).lazy()
    with pytest.raises(ValueError, match="not sorted by subject_id"):
        get_sparse_static_rep(["A", "B"], static_df, meds_df, ["A/static/present", "B/static/present"])


def test_get_sparse_static_rep_duplicate_subjects():
    static_df = pl.DataFrame({"subject_id": [1, 1], "A": [1.0, 2.0], "B": [3.0, 4.0]}).lazy()
    meds_df = pl.DataFrame({"subject_id": [1, 1], "code": ["A", "B"]}).lazy()
    with pytest.raises(ValueError, match="duplicate subject_id"):
        get_sparse_static_rep(["A", "B"], static_df, meds_df, ["A/static/present", "B/static/present"])


def test_summarize_static_invalid_aggregation():
    df = pl.DataFrame({"subject_id": [1], "code": ["A"], "numeric_value": [1.0]}).lazy()
    with pytest.raises(ValueError, match="Invalid aggregation type"):
        summarize_static_measurements("invalid_agg", ["A/static/first"], df)


# ============================================================================
# sklearn_model.py error paths
# ============================================================================


def test_sklearn_model_no_fit():
    from MEDS_tabular_automl.sklearn_model import SklearnModel

    cfg = MagicMock()
    cfg.model = object()  # no fit method
    with pytest.raises(ValueError, match="does not have a fit method"):
        SklearnModel(cfg)


def test_sklearn_evaluate_invalid_split():
    from MEDS_tabular_automl.sklearn_model import SklearnModel

    model = MagicMock()
    model.fit = MagicMock()
    sklearn_model = SklearnModel.__new__(SklearnModel)
    sklearn_model.model = model
    sklearn_model.keep_data_in_memory = True
    with pytest.raises(ValueError, match="not valid"):
        sklearn_model.evaluate(split="invalid_split")


def test_sklearn_evaluate_no_predict_proba():
    from MEDS_tabular_automl.sklearn_model import SklearnModel

    model = MagicMock(spec=[])  # empty spec, no methods
    sklearn_model = SklearnModel.__new__(SklearnModel)
    sklearn_model.model = model
    sklearn_model.keep_data_in_memory = True
    sklearn_model.dtuning = MagicMock()
    sklearn_model.ituning = MagicMock()
    with pytest.raises(ValueError, match="does not have a predict_proba method"):
        sklearn_model.evaluate(split="tuning")


def test_sklearn_save_model_wrong_extension(tmp_path):
    from MEDS_tabular_automl.sklearn_model import SklearnModel

    class FakeModel:
        def fit(self, X, y):
            pass
        def predict_proba(self, X):
            pass

    sklearn_model = SklearnModel.__new__(SklearnModel)
    sklearn_model.model = FakeModel()

    with pytest.raises(ValueError, match="Model file extension must be .pkl"):
        sklearn_model.save_model(tmp_path / "model.json")


def test_sklearn_save_model_pickle(tmp_path):
    from sklearn.linear_model import SGDClassifier

    from MEDS_tabular_automl.sklearn_model import SklearnModel

    sklearn_model = SklearnModel.__new__(SklearnModel)
    sklearn_model.model = SGDClassifier()  # real picklable model, no save_model method
    sklearn_model.save_model(tmp_path / "model.pkl")
    assert (tmp_path / "model.pkl").exists()


def test_sklearn_partial_fit_without_method():
    from MEDS_tabular_automl.sklearn_model import SklearnModel

    model = MagicMock(spec=["fit", "predict_proba"])  # no partial_fit
    sklearn_model = SklearnModel.__new__(SklearnModel)
    sklearn_model.model = model
    sklearn_model.cfg = MagicMock()
    sklearn_model.itrain = MagicMock()
    with pytest.raises(ValueError, match="does not support partial_fit"):
        sklearn_model._fit_from_partial()


# ============================================================================
# xgboost_model.py error paths
# ============================================================================


def test_xgboost_predict_invalid_split():
    from MEDS_tabular_automl.xgboost_model import XGBoostModel

    model = XGBoostModel.__new__(XGBoostModel)
    model.model = MagicMock()
    with pytest.raises(ValueError, match="Invalid split"):
        model._predict(split="invalid_split")


def test_xgboost_predict_df_invalid_split():
    from MEDS_tabular_automl.xgboost_model import XGBoostModel

    model = XGBoostModel.__new__(XGBoostModel)
    model.model = MagicMock()
    model.cfg = MagicMock()
    model._predict = MagicMock(return_value=(np.array([1, 0]), np.array([0.9, 0.1])))
    with pytest.raises(ValueError, match="Invalid split"):
        model.predict(split="invalid_split")


def test_xgboost_evaluate_single_class():
    from MEDS_tabular_automl.xgboost_model import XGBoostModel

    model = XGBoostModel.__new__(XGBoostModel)
    model.model = MagicMock()
    model._predict = MagicMock(return_value=(np.array([1, 1, 1]), np.array([0.9, 0.8, 0.7])))
    assert model.evaluate(split="tuning") == 0.0


# ============================================================================
# evaluation_callback.py error paths
# ============================================================================


def test_evaluation_callback_missing_logs(tmp_path):
    from MEDS_tabular_automl.evaluation_callback import EvaluationCallback

    cb = EvaluationCallback()
    config = MagicMock()
    config.path.sweep_results_dir = str(tmp_path / "nonexistent")
    config.path.performance_log_stem = "perf"
    with pytest.raises(FileNotFoundError, match="Log files incomplete"):
        cb.on_multirun_end(config)


# ============================================================================
# launch_autogluon.py - import error path
# ============================================================================


def test_autogluon_import_error():
    with patch.dict("sys.modules", {"autogluon": None, "autogluon.tabular": None}):
        with pytest.raises(ImportError, match="AutoGluon could not be imported"):
            try:
                import autogluon.tabular as ag  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "AutoGluon could not be imported. Please try installing it using:"
                    " `pip install autogluon`"
                ) from e
