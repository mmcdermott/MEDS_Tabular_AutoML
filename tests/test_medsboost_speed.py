"""Training-speed defaults of medsboost (GOSS row sampling) and trial-written predictions."""

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from MEDS_tabular_automl import medsboost
from MEDS_tabular_automl.evaluation_callback import EvaluationCallback

needs_library = pytest.mark.skipif(not medsboost.is_available(), reason="medsboost is not built")


def _data(seed: int = 0):
    X = sp.random(3000, 60, density=0.3, format="csc", random_state=seed, dtype=np.float32)
    y = (X[:, 0].toarray().ravel() + X[:, 1].toarray().ravel() > 0.6).astype(np.float32)
    return X, y


def _auroc(y, p) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


@needs_library
@pytest.mark.parametrize("rows", [{}, {"goss_top": 0.0}, {"goss_top": 0.2, "goss_other": 0.1}])
def test_bagging_and_goss_train(rows):
    """GOSS (the default), plain bagging and a stronger GOSS all learn the task."""
    X, y = _data()
    model = medsboost.MedsBoost(threads=2, max_rounds=100, **rows)
    model.fit(X[:2000], y[:2000], X[2000:2500], y[2000:2500])
    assert _auroc(y[2500:], model.predict_proba(X[2500:])) > 0.95
    assert model.info["trees_trained"] > 0
    assert model.info["grow_s"] >= 0


def test_goss_is_the_default():
    params = medsboost.MedsBoost().params
    assert (params["goss_top"], params["goss_other"]) == (0.3, 0.2)


@pytest.mark.parametrize(
    "goss_top, goss_other",
    [(-0.1, 0.2), (1.0, 0.0), (0.6, 0.5), (0.3, 0.0)],
)
def test_invalid_goss_is_rejected(goss_top, goss_other):
    with pytest.raises(ValueError):
        medsboost.MedsBoost(goss_top=goss_top, goss_other=goss_other)


def test_store_predictions_keeps_the_trial_predictions(tmp_path, monkeypatch):
    """When the best trial already has every prediction file, the callback neither reloads the model nor
    rewrites them."""
    want = pl.DataFrame({"subject_id": [1, 2], "predicted_boolean_probability": [0.1, 0.9]})
    for split in ("held_out", "tuning"):
        want.write_parquet(tmp_path / f"{split}_predictions.parquet")

    def fail(*args, **kwargs):
        raise AssertionError("the model should not be reloaded")

    monkeypatch.setattr("hydra.utils.instantiate", fail)
    EvaluationCallback().store_predictions(tmp_path, ["held_out", "tuning"])
    for split in ("held_out", "tuning"):
        assert pl.read_parquet(tmp_path / f"{split}_predictions.parquet").equals(want)
