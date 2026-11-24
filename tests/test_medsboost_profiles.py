"""The default hyperparameters of medsboost by training prevalence (MedsBoost(profile=...))."""

import numpy as np
import pytest
import scipy.sparse as sp
from omegaconf import OmegaConf

from MEDS_tabular_automl import medsboost
from MEDS_tabular_automl.medsboost import (
    LOW_PREVALENCE_MAX,
    LOW_PREVALENCE_PROFILE,
    STANDARD_PROFILE,
    MedsBoost,
)

needs_library = pytest.mark.skipif(not medsboost.is_available(), reason="medsboost is not built")


def data(prevalence: float, n: int = 4000, seed: int = 0):
    """Rows with a signal in columns 0 and 1 and exactly round(prevalence * n) positives."""
    rng = np.random.default_rng(seed)
    X = sp.random(n, 30, density=0.3, format="csc", random_state=seed, dtype=np.float32)
    score = X[:, 0].toarray().ravel() + X[:, 1].toarray().ravel() + rng.normal(0, 0.05, n)
    y = np.zeros(n, dtype=np.float32)
    y[np.argsort(-score)[: round(prevalence * n)]] = 1
    return X, y


def fit(model: MedsBoost, prevalence: float) -> MedsBoost:
    X, y = data(prevalence)
    return model.fit(X[:3000], y[:3000], X[3000:], y[3000:])


def profile_part(params: dict) -> dict:
    return {k: params[k] for k in STANDARD_PROFILE}


def test_params_show_the_standard_values_before_fitting():
    assert profile_part(MedsBoost().params) == STANDARD_PROFILE
    assert profile_part(MedsBoost(profile="low_prevalence").params) == LOW_PREVALENCE_PROFILE


@needs_library
@pytest.mark.parametrize(
    "prevalence, expected",
    [(0.01, "low_prevalence"), (LOW_PREVALENCE_MAX, "low_prevalence"), (0.05, "standard"), (0.5, "standard")],
)
def test_auto_profile_follows_the_training_prevalence(prevalence, expected):
    model = fit(MedsBoost(threads=2, max_rounds=30), prevalence)
    assert model.info["profile"] == expected
    assert model.info["prevalence"] == pytest.approx(prevalence, abs=0.002)
    want = LOW_PREVALENCE_PROFILE if expected == "low_prevalence" else STANDARD_PROFILE
    assert profile_part(model.params) == {**want, "max_rounds": 30}
    model.free()


@needs_library
def test_explicit_values_win_over_the_profile():
    model = fit(MedsBoost(threads=2, num_leaves=63, max_rounds=30, goss_top=0.0), 0.01)
    assert model.info["profile"] == "low_prevalence"
    assert model.params["num_leaves"] == 63 and model.params["goss_top"] == 0.0
    assert model.params["learning_rate"] == LOW_PREVALENCE_PROFILE["learning_rate"]
    model.free()


@needs_library
@pytest.mark.parametrize("profile, prevalence", [("standard", 0.01), ("low_prevalence", 0.5)])
def test_forced_profiles(profile, prevalence):
    model = fit(MedsBoost(threads=2, max_rounds=30, profile=profile), prevalence)
    assert model.info["profile"] == profile
    model.free()


@needs_library
def test_refitting_resolves_the_profile_again():
    model = MedsBoost(threads=2, max_rounds=30)
    fit(model, 0.01)
    assert model.params["num_leaves"] == LOW_PREVALENCE_PROFILE["num_leaves"]
    fit(model, 0.5)
    assert model.info["profile"] == "standard"
    assert model.params["num_leaves"] == STANDARD_PROFILE["num_leaves"]
    model.free()


def test_invalid_profile_is_rejected():
    with pytest.raises(ValueError, match="profile"):
        MedsBoost(profile="rare")


def test_launcher_config_leaves_profile_parameters_to_the_profile():
    from MEDS_tabular_automl import LAUNCH_MODEL_CFG

    cfg = OmegaConf.load(LAUNCH_MODEL_CFG.parent / "model_launcher" / "medsboost.yaml")
    model = cfg.model_launcher.model
    assert model.profile == "auto"
    assert all(model[k] is None for k in STANDARD_PROFILE)
