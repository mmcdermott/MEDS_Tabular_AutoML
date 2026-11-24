"""An average of medsboost models (bagged boosting), each early-stopping on its own share of the subjects.

``MedsBoostEnsemble`` trains ``members`` medsboost models on the same rows. Each member sets aside a share
(``stop_fraction``) of the *subjects* for early stopping, drawn with its own seed and stratified by whether a
subject has a positive label, and trains on the other rows; its parameters are ``base_params`` updated by
``member_params[k % len(member_params)]``. The prediction is the mean of the members' probabilities.

Why: with few positives, a single model's early-stopping point depends on a small tuning set, and its trees
fit columns that separate a handful of positives by chance. Averaging members that each stop on different
subjects removes most of that variance, and alternating strongly regularized members with default ones adds
diversity.
``MIX6_MEMBER_PARAMS`` is the validated configuration (``model_launcher=medsboost_mix6``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import MedsBoost

# Strongly regularized members: few, large leaves and strong L2 so that trees cannot isolate a few positives
# on rare columns; a low learning rate with more patience lets them train for hundreds of trees, not dozens.
STRONG_PARAMS = {
    "num_leaves": 15,
    "min_child_samples": 300,
    "lambda_l2": 10.0,
    "learning_rate": 0.02,
    "max_rounds": 3000,
    "early_stopping": 100,
}

# The "mix6" ensemble (model_launcher=medsboost_mix6): 6 members alternating the strong parameters and the
# profile defaults ({}), trained on train+tuning, each stopping on its own 11% of subjects. Chosen by
# subject-grouped 5-fold cross-validation on train+tuning of 3 MIMIC-IV phenotypes (AMI, MASLD, stroke; 0.9-4%
# prevalence), where it beat the single default model on all 30 fold comparisons (two fold seeds), then
# validated once on held-out data of those tasks and of 3 patient-outcome tasks never used to choose it
# (docs/usage_guide.md).
MIX6_MEMBER_PARAMS = [STRONG_PARAMS, {}]


def subject_split(y, groups, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Splits rows by subject: about `fraction` of the subjects, drawn separately among those with and without
    a positive label, go to the second part (early stopping); all rows of a subject stay on one side.

    Examples:
        >>> y = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        >>> g = np.array([1, 1, 2, 3, 3, 4, 5, 6, 7, 8])
        >>> fit, stop = subject_split(y, g, 0.5, seed=0)
        >>> sorted(set(g[fit]) & set(g[stop]))
        []
        >>> len(fit) + len(stop) == len(y), len(set(g[stop][y[stop] > 0]))
        (True, 1)
    """
    y, groups = np.asarray(y), np.asarray(groups)
    rng = np.random.default_rng(seed)
    subjects = np.unique(groups)
    positive = np.isin(subjects, np.unique(groups[y > 0]))
    pick = np.zeros(len(subjects), dtype=bool)
    for stratum in (positive, ~positive):
        idx = np.flatnonzero(stratum)
        rng.shuffle(idx)
        pick[idx[: round(fraction * len(idx))]] = True
    stop = np.isin(groups, subjects[pick])
    return np.flatnonzero(~stop), np.flatnonzero(stop)


class MedsBoostEnsemble:
    """Mean of `members` medsboost models, each early-stopping on its own share of the subjects.

    Args:
        members: number of models.
        member_params: per-member overrides of `base_params`, cycled (member k uses entry k % len); empty: all
            members use `base_params`. ``{}`` means "the base parameters" (e.g. the profile defaults).
        stop_fraction: share of the subjects each member early-stops on.
        base_params: MedsBoost keyword arguments shared by all members; member k's seed is ``seed + k``.

    Examples:
        >>> import scipy.sparse as sp, tempfile
        >>> X = sp.random(900, 30, density=0.3, format="csr", random_state=0, dtype=np.float32)
        >>> y = (X[:, 0].toarray().ravel() + X[:, 1].toarray().ravel() > 0.6).astype(np.float32)
        >>> groups = np.repeat(np.arange(300), 3)
        >>> small = {"num_leaves": 7, "min_child_samples": 5, "max_rounds": 30, "early_stopping": 5,
        ...          "threads": 2}
        >>> ens = MedsBoostEnsemble(3, [{"learning_rate": 0.1}, {}], 0.2, small).fit(X, y, groups)
        >>> p = ens.predict_proba(X)
        >>> bool(np.allclose(p, np.mean([m.predict_proba(X) for m in ens.models], axis=0)))
        True
        >>> [c["seed"] for c in ens.member_configs], [c.get("learning_rate") for c in ens.member_configs]
        ([1, 2, 3], [0.1, None, 0.1])
        >>> with tempfile.TemporaryDirectory() as d:
        ...     ens.save(Path(d) / "medsboost.txt")
        ...     MedsBoostEnsemble.is_ensemble_file(Path(d) / "medsboost.txt")
        ...     reloaded = MedsBoostEnsemble.load(Path(d) / "medsboost.txt", threads=2)
        ...     bool(np.allclose(reloaded.predict_proba(X), p))
        True
        True
    """

    MANIFEST_FORMAT = "medsboost-ensemble-v1"

    def __init__(
        self,
        members: int = 6,
        member_params: list[dict] | None = None,
        stop_fraction: float = 0.11,
        base_params: dict | None = None,
    ):
        if members < 1:
            raise ValueError(f"members must be at least 1, got {members}")
        if not 0 < stop_fraction < 1:
            raise ValueError(f"stop_fraction must be in (0, 1), got {stop_fraction}")
        self.members = int(members)
        self.member_params = [dict(p or {}) for p in (member_params or [])]
        self.stop_fraction = float(stop_fraction)
        base = dict(base_params or {})
        seed = int(base.pop("seed", None) or 1)
        self.member_configs = []
        for k in range(self.members):
            cfg = dict(base)
            if self.member_params:
                cfg.update(self.member_params[k % len(self.member_params)])
            cfg["seed"] = seed + k
            self.member_configs.append(cfg)
        self.models: list[MedsBoost] = []
        self.info: dict = {}

    def fit(self, X, y, groups=None, slots: np.ndarray | None = None) -> MedsBoostEnsemble:
        """Trains every member on (X, y); `groups` are subject ids (None: every row is its own subject)."""
        y = np.asarray(y, dtype=np.float32)
        groups = np.arange(len(y)) if groups is None else np.asarray(groups)
        if len(groups) != len(y):
            raise ValueError(f"groups has {len(groups)} entries for {len(y)} rows")
        self.free()
        self.models = []
        for cfg in self.member_configs:
            fit_rows, stop_rows = subject_split(y, groups, self.stop_fraction, cfg["seed"])
            if len(np.unique(y[stop_rows])) < 2:
                raise ValueError(
                    "an early-stopping share has a single class; raise stop_fraction or add labels"
                )
            model = MedsBoost(**cfg).fit(X[fit_rows], y[fit_rows], X[stop_rows], y[stop_rows], slots=slots)
            self.models.append(model)
        self.info = {
            "members": self.members,
            "rows": len(y),
            "subjects": len(np.unique(groups)),
            "trees": [m.num_trees for m in self.models],
            "profiles": [m.info.get("profile") for m in self.models],
        }
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Mean of the members' predicted probabilities."""
        if not self.models:
            raise RuntimeError("the ensemble is not fitted")
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)

    @property
    def num_trees(self) -> int:
        return int(sum(m.num_trees for m in self.models))

    def feature_importance(self, importance_type: str = "split") -> np.ndarray:
        """Per-column importance summed over the members (see MedsBoost.feature_importance)."""
        return np.sum([m.feature_importance(importance_type) for m in self.models], axis=0)

    def save(self, path, save_stats: bool | None = None) -> None:
        """Writes one medsboost model file per member next to `path`, and a JSON manifest at `path`."""
        path = Path(path)
        files = []
        for k, model in enumerate(self.models):
            member_fp = path.with_name(f"{path.stem}.member{k}{path.suffix}")
            model.save(member_fp, save_stats=save_stats)
            files.append(member_fp.name)
        manifest = {
            "format": self.MANIFEST_FORMAT,
            "members": files,
            "stop_fraction": self.stop_fraction,
            "member_configs": self.member_configs,
        }
        path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    @classmethod
    def is_ensemble_file(cls, path) -> bool:
        """True if `path` is an ensemble manifest (medsboost model files are not JSON)."""
        try:
            with open(path, encoding="utf-8") as f:
                head = f.read(256)
        except (OSError, UnicodeDecodeError):
            return False
        return head.lstrip().startswith("{") and cls.MANIFEST_FORMAT in head

    @classmethod
    def load(cls, path, threads: int = 0) -> MedsBoostEnsemble:
        path = Path(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("format") != cls.MANIFEST_FORMAT:
            raise ValueError(f"{path} is not a medsboost ensemble manifest")
        ens = cls(members=len(manifest["members"]), stop_fraction=manifest.get("stop_fraction", 0.11))
        ens.member_configs = manifest.get("member_configs", ens.member_configs)
        ens.models = [MedsBoost.load(path.with_name(name), threads=threads) for name in manifest["members"]]
        return ens

    def free(self) -> None:
        for model in self.models:
            model.free()
