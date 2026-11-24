"""Tests of the medsboost ensemble (MEDS_tabular_automl.medsboost.ensemble) and its launcher option."""

import numpy as np
import pytest
import scipy.sparse as sp
from omegaconf import OmegaConf

from MEDS_tabular_automl import medsboost
from MEDS_tabular_automl.medsboost.ensemble import MIX6_MEMBER_PARAMS, MedsBoostEnsemble, subject_split
from MEDS_tabular_automl.medsboost_model import MedsBoostModel

pytestmark = pytest.mark.skipif(not medsboost.is_available(), reason="medsboost library not built")

SMALL = {"num_leaves": 7, "min_child_samples": 5, "max_rounds": 30, "early_stopping": 5, "threads": 2}


def _data(n_subjects=400, rows_per_subject=3, seed=0):
    rng = np.random.default_rng(seed)
    n = n_subjects * rows_per_subject
    X = sp.random(n, 30, density=0.3, format="csr", random_state=seed, dtype=np.float32)
    signal = X[:, 0].toarray().ravel() + X[:, 1].toarray().ravel()
    y = (signal + 0.2 * rng.standard_normal(n) > 0.8).astype(np.float32)
    groups = np.repeat(np.arange(n_subjects), rows_per_subject)
    return X, y, groups


def test_subject_split_keeps_subjects_together_and_stratifies():
    _, y, groups = _data()
    pos_subjects = set(groups[y > 0])
    for seed in range(5):
        fit, stop = subject_split(y, groups, 0.11, seed)
        assert not set(groups[fit]) & set(groups[stop])
        assert len(fit) + len(stop) == len(y)
        share_pos = len(set(groups[stop]) & pos_subjects) / len(pos_subjects)
        assert abs(share_pos - 0.11) < 0.02, "positive subjects are stratified"


def test_members_use_different_stopping_subjects_and_seeds():
    X, y, groups = _data()
    ens = MedsBoostEnsemble(4, [{"learning_rate": 0.1}, {}], 0.2, {**SMALL, "seed": 7}).fit(X, y, groups)
    assert [c["seed"] for c in ens.member_configs] == [7, 8, 9, 10]
    stops = [set(groups[subject_split(y, groups, 0.2, s)[1]]) for s in (7, 8, 9, 10)]
    assert all(a != b for i, a in enumerate(stops) for b in stops[i + 1 :])
    assert len(ens.models) == 4 and ens.num_trees == sum(m.num_trees for m in ens.models)


def test_prediction_is_member_mean_and_deterministic():
    X, y, groups = _data()
    a = MedsBoostEnsemble(3, MIX6_MEMBER_PARAMS, 0.2, SMALL).fit(X, y, groups)
    b = MedsBoostEnsemble(3, MIX6_MEMBER_PARAMS, 0.2, SMALL).fit(X, y, groups)
    np.testing.assert_allclose(a.predict_proba(X), np.mean([m.predict_proba(X) for m in a.models], axis=0))
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X))


def test_save_load_round_trip(tmp_path):
    X, y, groups = _data()
    ens = MedsBoostEnsemble(2, [], 0.2, SMALL).fit(X, y, groups)
    fp = tmp_path / "medsboost.txt"
    ens.save(fp)
    assert MedsBoostEnsemble.is_ensemble_file(fp)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["medsboost.member0.txt", "medsboost.member1.txt", "medsboost.txt"]
    assert not MedsBoostEnsemble.is_ensemble_file(tmp_path / "medsboost.member0.txt")
    np.testing.assert_allclose(MedsBoostEnsemble.load(fp, threads=2).predict_proba(X), ens.predict_proba(X))


def test_invalid_arguments():
    with pytest.raises(ValueError):
        MedsBoostEnsemble(0)
    with pytest.raises(ValueError):
        MedsBoostEnsemble(2, stop_fraction=1.0)
    X, y, groups = _data()
    with pytest.raises(ValueError):
        MedsBoostEnsemble(2, base_params=SMALL).fit(X, y, groups[:-1])


def _launcher(members, pool_tuning):
    cfg = OmegaConf.create(
        {
            "data_loading_params": {"keep_data_in_memory": True},
            "model": {**SMALL, "seed": 1},
            "ensemble": {
                "members": members,
                "member_params": [],
                "stop_fraction": 0.2,
                "pool_tuning": pool_tuning,
            },
        }
    )
    model = MedsBoostModel(cfg)
    X, y, groups = _data()
    tr, tu = slice(0, 900), slice(900, 1050)
    model._set_data((X[tr], y[tr]), (X[tu], y[tu]), (X[1050:], y[1050:]), slots=None)
    model.groups = (groups[tr], groups[tu])
    return model


def test_launcher_single_model_unchanged():
    model = _launcher(members=1, pool_tuning=False)
    model._train()
    assert isinstance(model.model, medsboost.MedsBoost)


@pytest.mark.parametrize("pool_tuning", [False, True])
def test_launcher_ensemble_training_rows(pool_tuning):
    model = _launcher(members=2, pool_tuning=pool_tuning)
    model._train()
    assert isinstance(model.model, MedsBoostEnsemble)
    # the members train on train rows (300 subjects), or on train+tuning rows (350 subjects) when pooled
    assert model.model.info["rows"] == (1050 if pool_tuning else 900)
    assert model.model.info["subjects"] == (350 if pool_tuning else 300)
    assert 0.5 <= model.evaluate("held_out") <= 1.0
