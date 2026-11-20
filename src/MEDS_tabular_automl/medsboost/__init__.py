"""medsboost: gradient-boosted trees for wide, sparse tabularized data (binary classification).

The same algorithm as LightGBM in MEDS-Tab's configuration (Newton boosting on the logistic loss,
per-feature histograms, leaf-wise growth, `min_child_samples` / `lambda_l2` constraints, bagging, early
stopping on tuning log-loss), engineered for matrices with ~1e5 columns and ~1e3 non-zeros per row: absent
entries are never stored or scanned, features that can no longer split in a node are dropped from it and its
descendants, and each split is one parallel pass. Columns can be grouped into "slots" (one code and
aggregation, all windows); each tree then uses one column per slot, drawn from precomputed layouts.

The C++ library (csrc/medsboost) is built into this package as ``_medsboost`` when MEDS-Tab is installed
with a C++17 compiler and OpenMP; nothing loads it unless this module is used. Until MEDS-Tab 0.4.1 it was
called sparsegb.

Examples:
    >>> from sklearn.metrics import roc_auc_score
    >>> X = sp.random(600, 40, density=0.3, format="csc", random_state=0, dtype=np.float32)
    >>> y = (X[:, 0].toarray().ravel() + X[:, 1].toarray().ravel() > 0.6).astype(np.float32)
    >>> model = MedsBoost(threads=2, num_leaves=15, min_child_samples=5, max_rounds=40, early_stopping=10)
    >>> model = model.fit(X[:400], y[:400], X[400:500], y[400:500])
    >>> p = model.predict_proba(X[500:])
    >>> p.shape, bool(((p > 0) & (p < 1)).all())
    ((100,), True)
    >>> bool(roc_auc_score(y[500:], p) > 0.9)
    True
    >>> with tempfile.TemporaryDirectory() as d:
    ...     model.save(Path(d) / "model.txt")
    ...     reloaded = MedsBoost.load(Path(d) / "model.txt")
    >>> bool(np.allclose(reloaded.predict_proba(X[500:]), p))
    True
    >>> model.num_trees > 0
    True

Models saved under the name medsboost had before MEDS-Tab 0.4.1, sparsegb, still load:

    >>> with tempfile.TemporaryDirectory() as d:
    ...     model.save(Path(d) / "model.txt")
    ...     text = (Path(d) / "model.txt").read_text()
    ...     _ = (Path(d) / "old.txt").write_text(text.replace("medsboost 1", "sparsegb 1", 1))
    ...     old = MedsBoost.load(Path(d) / "old.txt")
    >>> text.startswith("medsboost 1"), bool(np.allclose(old.predict_proba(X[500:]), p))
    (True, True)

Feature importance, like LightGBM's: how often each column is split on, or the total gain of those splits.
The informative columns 0 and 1 come first:

    >>> gain = model.feature_importance("gain")
    >>> splits = model.feature_importance("split")
    >>> gain.shape, splits.dtype, int(splits.sum()) > 0
    ((40,), dtype('int64'), True)
    >>> sorted(np.argsort(-gain)[:2].tolist())
    [0, 1]

Split counts survive any save; gains only a save with ``save_stats`` (format version 2, which predictions
ignore), so a default save is unchanged:

    >>> with tempfile.TemporaryDirectory() as d:
    ...     model.save(Path(d) / "v1.txt")
    ...     model.save(Path(d) / "v2.txt", save_stats=True)
    ...     v1_text = (Path(d) / "v1.txt").read_text()
    ...     v2_text = (Path(d) / "v2.txt").read_text()
    ...     v1 = MedsBoost.load(Path(d) / "v1.txt")
    ...     v2 = MedsBoost.load(Path(d) / "v2.txt")
    >>> v1_text == text, v2_text.startswith("medsboost 2")
    (True, True)
    >>> bool(np.array_equal(v1.feature_importance("split"), splits))
    True
    >>> bool(np.allclose(v2.feature_importance("gain"), gain))
    True
    >>> bool(np.allclose(v2.predict_proba(X[500:]), p))
    True
    >>> v1.feature_importance("gain")
    Traceback (most recent call last):
        ...
    ValueError: this model was loaded from a file saved without node statistics; save it with save_stats=True
    >>> model.feature_importance("cover")
    Traceback (most recent call last):
        ...
    ValueError: importance_type must be one of ['gain', 'split'], got 'cover'

Format version 2 adds, per tree, ``gain hess count`` to each node line (``feature threshold left right``;
a child ``c < 0`` is leaf ``~c``) and, after the leaf values line, a line of leaf hessian sums and a line of
leaf row counts: the node cover that exact TreeSHAP needs, on the tree's (bagged) training rows. Every
node's hessian sum and row count are those of its two children added:

    >>> def stat(nodes, leaves, child, k):  # k: 5 = hessian sum, 6 = row count
    ...     return float(nodes[child][k]) if child >= 0 else leaves[k - 5][~child]
    >>> lines, consistent = v2_text.splitlines(), []
    >>> i = lines.index(f"trees {model.num_trees}") + 1
    >>> while i < len(lines):
    ...     n_nodes, n_leaves = (int(v) for v in lines[i].split()[1:])
    ...     nodes = [line.split() for line in lines[i + 1 : i + 1 + n_nodes]]
    ...     leaves = [[float(v) for v in lines[i + n_nodes + j].split()] for j in (2, 3)]
    ...     consistent += [
    ...         n_leaves == 1
    ...         or all(
    ...             np.isclose(
    ...                 stat(nodes, leaves, int(n[2]), k) + stat(nodes, leaves, int(n[3]), k), float(n[k])
    ...             )
    ...             for n in nodes
    ...             for k in (5, 6)
    ...         )
    ...     ]
    ...     i += n_nodes + 4
    >>> len(consistent) == model.num_trees, all(consistent)
    (True, True)

SHAP values (exact path-dependent TreeSHAP, from the node row counts): for each row, the contributions of the
columns the trees use plus the expected value give the row's log-odds. The informative columns 0 and 1 have
the largest mean absolute values:

    >>> values, columns, expected = model.shap_values(X[500:])
    >>> values.shape == (100, len(columns))
    True
    >>> bool(np.allclose(values.sum(1) + expected, model.predict_raw(X[500:])))
    True
    >>> sorted(columns[np.argsort(-np.abs(values).mean(0))[:2]].tolist())
    [0, 1]
    >>> v2_values, _, v2_expected = v2.shap_values(X[500:])
    >>> bool(np.allclose(v2_values, values)), bool(np.isclose(v2_expected, expected))
    (True, True)
    >>> v1.shap_values(X[500:])
    Traceback (most recent call last):
        ...
    ValueError: this model was loaded from a file saved without node statistics; save it with save_stats=True
"""

from __future__ import annotations

import contextlib
import ctypes as ct
import importlib.util
import os
from functools import cache
from pathlib import Path

import numpy as np
import scipy.sparse as sp

NOT_BUILT = (
    "medsboost is not built. The compiled wheels (`pip install meds-tab` on Linux, Windows or Apple-silicon "
    "macOS 15+, Python 3.11-3.13) include it; elsewhere install MEDS-Tab from source with a C++17 compiler "
    "and OpenMP (`pip install --no-binary meds-tab meds-tab`; MSVC on Windows)."
)


def check_inputs(X_train, y_train, X_tuning, y_tuning) -> None:
    """Raises ValueError unless the training and tuning data can be passed to the library.

    The library trusts its inputs: mismatched lengths would make it read past the end of an array, NaN
    values make its sorting undefined, and an empty tuning split gives a model with no trees.

    Examples:
        >>> X = sp.csc_matrix(np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]], dtype=np.float32))
        >>> y = np.array([0, 1, 1])
        >>> check_inputs(X, y, X, y)
        >>> check_inputs(X, y[:2], X, y)
        Traceback (most recent call last):
            ...
        ValueError: y_train has 2 labels for 3 rows of X_train
        >>> check_inputs(X, y, X[:, :1], y)
        Traceback (most recent call last):
            ...
        ValueError: X_tuning has 1 columns, X_train 2
        >>> check_inputs(X, y, X[:0], y[:0])
        Traceback (most recent call last):
            ...
        ValueError: the tuning split is empty; medsboost early-stops on it
        >>> X_nan = X.copy()
        >>> X_nan.data[0] = np.nan
        >>> check_inputs(X_nan, y, X, y)
        Traceback (most recent call last):
            ...
        ValueError: X_train has non-finite values (NaN or inf)
        >>> check_inputs(X, np.array([0, 1, 2]), X, y)
        Traceback (most recent call last):
            ...
        ValueError: y_train must hold binary labels (0/1 or booleans)
    """
    for name, X, y in (("train", X_train, y_train), ("tuning", X_tuning, y_tuning)):
        if X.ndim != 2:
            raise ValueError(f"X_{name} must be 2-dimensional, got shape {X.shape}")
        y = np.asarray(y)
        if y.shape != (X.shape[0],):
            raise ValueError(f"y_{name} has {y.size} labels for {X.shape[0]} rows of X_{name}")
        if X.shape[0] == 0:
            hint = "; medsboost early-stops on it" if name == "tuning" else ""
            raise ValueError(f"the {name} split is empty{hint}")
        values = X.data if sp.issparse(X) else np.asarray(X)
        if not np.isfinite(values).all():
            raise ValueError(f"X_{name} has non-finite values (NaN or inf)")
        if not np.isin(y, (0, 1)).all():
            raise ValueError(f"y_{name} must hold binary labels (0/1 or booleans)")
    if X_tuning.shape[1] != X_train.shape[1]:
        raise ValueError(f"X_tuning has {X_tuning.shape[1]} columns, X_train {X_train.shape[1]}")


I64P, I32P, F32P, F64P = (ct.POINTER(t) for t in (ct.c_int64, ct.c_int32, ct.c_float, ct.c_double))


def library_path() -> Path:
    spec = importlib.util.find_spec(f"{__package__}._medsboost")
    if spec is None or spec.origin is None:
        raise FileNotFoundError(NOT_BUILT)
    return Path(spec.origin)


def default_threads() -> int:
    """Threads used for ``threads=0``: ``OMP_NUM_THREADS`` when set, else the CPUs this process may run on.

    Hydra's joblib launcher starts each worker with ``OMP_NUM_THREADS`` set to the CPU count divided by
    ``hydra.launcher.n_jobs``; honoring it keeps parallel sweep trials from each taking every CPU.

    Examples:
        >>> from unittest import mock
        >>> with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "3"}):
        ...     default_threads()
        3
        >>> with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "not a number"}):
        ...     default_threads() >= 1
        True
    """
    with contextlib.suppress(ValueError):
        n = int(os.environ.get("OMP_NUM_THREADS", "").split(",")[0])
        if n > 0:
            return n
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return os.cpu_count() or 1


def is_available() -> bool:
    """Whether the medsboost library is built."""
    try:
        library()
    except (FileNotFoundError, OSError):
        return False
    return True


@cache
def library() -> ct.CDLL:
    lib = ct.CDLL(str(library_path()))
    lib.medsboost_train.restype = ct.c_void_p
    lib.medsboost_train.argtypes = [
        ct.c_int64,
        ct.c_int64,
        I64P,
        I32P,
        F32P,
        F32P,
        ct.c_int64,
        I64P,
        I32P,
        F32P,
        F32P,
        ct.c_int64,
        I64P,
        I32P,
        F32P,
        I32P,
        F64P,
        F64P,
        F64P,
        F64P,
    ]
    lib.medsboost_last_error.restype = ct.c_char_p
    lib.medsboost_last_error.argtypes = []
    lib.medsboost_predict.restype = ct.c_int
    lib.medsboost_predict.argtypes = [ct.c_void_p, ct.c_int64, I64P, I32P, F32P, ct.c_int, F64P]
    lib.medsboost_save.restype = ct.c_int
    lib.medsboost_save.argtypes = [ct.c_void_p, ct.c_char_p]
    lib.medsboost_save_ex.restype = ct.c_int
    lib.medsboost_save_ex.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_int]
    lib.medsboost_feature_importance.restype = ct.c_int
    lib.medsboost_feature_importance.argtypes = [ct.c_void_p, ct.c_int, F64P]
    lib.medsboost_num_used_features.restype = ct.c_int64
    lib.medsboost_num_used_features.argtypes = [ct.c_void_p]
    lib.medsboost_used_features.restype = ct.c_int
    lib.medsboost_used_features.argtypes = [ct.c_void_p, I32P]
    lib.medsboost_shap.restype = ct.c_int
    lib.medsboost_shap.argtypes = [ct.c_void_p, ct.c_int64, I64P, I32P, F32P, ct.c_int, F64P]
    lib.medsboost_load.restype = ct.c_void_p
    lib.medsboost_load.argtypes = [ct.c_char_p]
    lib.medsboost_free.restype = None
    lib.medsboost_free.argtypes = [ct.c_void_p]
    lib.medsboost_num_trees.restype = ct.c_int64
    lib.medsboost_num_trees.argtypes = [ct.c_void_p]
    lib.medsboost_num_features.restype = ct.c_int64
    lib.medsboost_num_features.argtypes = [ct.c_void_p]
    return lib


def _ptr(a: np.ndarray, t):
    return a.ctypes.data_as(t)


def _csr(X) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    X = sp.csr_matrix(X, dtype=np.float32)
    X.sort_indices()
    return (
        X.indptr.astype(np.int64),
        X.indices.astype(np.int32, copy=False),
        X.data.astype(np.float32),
        X.shape[0],
    )


# Default hyperparameters by training prevalence. Parameters left unset (None) take the "standard" values, or
# the "low_prevalence" ones when the training prevalence is at most LOW_PREVALENCE_MAX (profile="auto").
# The low-prevalence values (kept to 3 significant figures: rounding further cost AUROC) come from a
# multi-objective Optuna search (338 trials, 5 MIMIC-IV tasks at 0.9-71% prevalence), confirmed on held-out
# data the search never saw, 8 unseen NWICU tasks and prevalence ladders down to 0.1%: at <= 3% prevalence,
# AUPRC +6-17% at about the same AUROC (no task more than 0.003 lower). Above 3% they matched the standard
# values but trained about 2x slower, so the standard values stay there.
STANDARD_PROFILE = {
    "max_bin": 15,
    "num_leaves": 127,
    "min_child_samples": 50,
    "lambda_l2": 1.0,
    "learning_rate": 0.05,
    "max_rounds": 1000,
    "min_sum_hessian": 1e-3,
    "layouts": 14,
    "goss_top": 0.3,
    "goss_other": 0.2,
}
LOW_PREVALENCE_PROFILE = {
    "max_bin": 63,
    "num_leaves": 41,
    "min_child_samples": 95,
    "lambda_l2": 0.0211,
    "learning_rate": 0.0202,
    "max_rounds": 2000,
    "min_sum_hessian": 0.00182,
    "layouts": 28,
    "goss_top": 0.186,
    "goss_other": 0.105,
}
LOW_PREVALENCE_MAX = 0.03
PROFILES = ("auto", "standard", "low_prevalence")


def _check_goss(goss_top: float, goss_other: float) -> None:
    if not (0 <= goss_top < 1 and 0 <= goss_other <= 1 - goss_top):
        raise ValueError(
            f"Need 0 <= goss_top < 1 and 0 <= goss_other <= 1 - goss_top; got {goss_top}, {goss_other}"
        )
    if goss_top > 0 and goss_other <= 0:
        raise ValueError("goss_other must be > 0 when goss_top > 0 (the rest would never be sampled)")


class MedsBoost:
    """Binary gradient-boosted trees; see the module docstring.

    Hyperparameters left as None follow ``profile``: ``"auto"`` (the default) uses ``STANDARD_PROFILE``, or
    ``LOW_PREVALENCE_PROFILE`` when the training prevalence is at most ``LOW_PREVALENCE_MAX`` (3%);
    ``"standard"`` and ``"low_prevalence"`` force one. Explicitly given values always win. ``params`` shows
    standard values until ``fit`` resolves the profile; ``info["profile"]`` records the one used.

    Args:
        threads: worker threads (0: ``OMP_NUM_THREADS`` if set, else every available CPU).
        max_bin: bins per feature (zero has its own bin), at most 255.
        num_leaves: leaves per tree (leaf-wise growth).
        min_child_samples: minimum rows in each child of a split.
        lambda_l2: L2 penalty on leaf values.
        learning_rate: shrinkage.
        bagging_fraction: rows sampled (without replacement) per tree; used only when GOSS is off
            (``goss_top=0``).
        max_rounds: maximum number of trees.
        early_stopping: stop after this many trees without a better tuning log-loss; the best tree count is
            kept.
        seed: random seed (bagging and column sampling).
        min_sum_hessian: minimum hessian sum in each child.
        layouts: number of precomputed column layouts (multiple of 7); 0 draws a fresh column sample every
            `refresh` trees instead. Only matters when slots are given.
        refresh: trees per column sample (or per layout draw).
        slot_k: columns of each slot a sample uses (``layouts=0`` only); 1 = one window per (code,
            aggregation).
        goss_top: gradient-based one-side sampling (GOSS, as in LightGBM): each tree keeps the ``goss_top``
            share of rows with the largest |gradient| ...
        goss_other: ... plus a random ``goss_other`` share of the rest, up-weighted by
            ``(1 - goss_top) / goss_other``. ``goss_top=0`` uses plain bagging (``bagging_fraction``)
            instead. With single-precision histograms, the default 0.3 / 0.2 (half the rows per tree) trained
            22% faster in total than double-precision bagging 0.8 on 10 MIMIC-IV tasks, with the same held-out
            AUROC.
        profile: ``"auto"``, ``"standard"`` or ``"low_prevalence"`` (see above).
        save_stats: also write node statistics when saving (format version 2): each split's gain, and the
            hessian sum and row count (cover) of every node and leaf. Then ``feature_importance("gain")``
            works on the reloaded model, and the file has what exact TreeSHAP needs. Off by default: the
            saved file is then the version 1 format that every MEDS-Tab release reads. Does not affect
            training.
    """

    def __init__(
        self,
        threads: int = 0,
        max_bin: int | None = None,
        num_leaves: int | None = None,
        min_child_samples: int | None = None,
        lambda_l2: float | None = None,
        learning_rate: float | None = None,
        bagging_fraction: float = 0.8,
        max_rounds: int | None = None,
        early_stopping: int = 20,
        seed: int = 1,
        min_sum_hessian: float | None = None,
        layouts: int | None = None,
        refresh: int = 1,
        save_stats: bool = False,
        slot_k: int = 1,
        goss_top: float | None = None,
        goss_other: float | None = None,
        profile: str = "auto",
    ):
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {PROFILES}, got {profile!r}")
        given = {
            "max_bin": max_bin,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "lambda_l2": lambda_l2,
            "learning_rate": learning_rate,
            "max_rounds": max_rounds,
            "min_sum_hessian": min_sum_hessian,
            "layouts": layouts,
            "goss_top": goss_top,
            "goss_other": goss_other,
        }
        self.profile = profile
        self._explicit = {k: v for k, v in given.items() if v is not None}
        resolved = {**self._profile_values("low_prevalence" if profile == "low_prevalence" else "standard")}
        for name in ("standard", "low_prevalence"):  # every combination a fit may use must be valid
            if profile in ("auto", name):
                values = self._profile_values(name)
                _check_goss(values["goss_top"], values["goss_other"])
        self.params = {
            "threads": threads or default_threads(),
            **{
                k: resolved[k]
                for k in ("max_bin", "num_leaves", "min_child_samples", "lambda_l2", "learning_rate")
            },
            "bagging_fraction": bagging_fraction,
            "max_rounds": resolved["max_rounds"],
            "early_stopping": early_stopping,
            "seed": seed,
            "min_sum_hessian": resolved["min_sum_hessian"],
            "layouts": resolved["layouts"],
            "refresh": refresh,
            "slot_k": slot_k,
            "goss_top": resolved["goss_top"],
            "goss_other": resolved["goss_other"],
        }
        self.save_stats = bool(save_stats)  # an output option, not a training parameter
        self._handle = None
        self.info: dict = {}

    def _profile_values(self, name: str) -> dict:
        """The profile's values, overridden by the explicitly given ones."""
        base = LOW_PREVALENCE_PROFILE if name == "low_prevalence" else STANDARD_PROFILE
        return {**base, **self._explicit}

    def _resolve_profile(self, prevalence: float) -> str:
        """Sets the profile-dependent params for training data of this prevalence; returns the profile
        used."""
        if self.profile == "auto":
            name = "low_prevalence" if prevalence <= LOW_PREVALENCE_MAX else "standard"
        else:
            name = self.profile
        self.params.update(self._profile_values(name))
        return name

    def _param_array(self) -> np.ndarray:
        p = self.params
        return np.array(
            [
                p["threads"],
                p["max_bin"],
                p["num_leaves"],
                p["min_child_samples"],
                p["lambda_l2"],
                p["learning_rate"],
                p["bagging_fraction"],
                p["max_rounds"],
                p["early_stopping"],
                p["seed"],
                p["min_sum_hessian"],
                p["slot_k"],
                p["refresh"],
                1,
                p["layouts"],
                p["goss_top"],  # GOSS top share (0: bagging)
                p["goss_other"],  # GOSS other share
            ],
            dtype=np.float64,
        )

    def fit(self, X_train, y_train, X_tuning, y_tuning, slots: np.ndarray | None = None) -> MedsBoost:
        """Trains on (X_train, y_train) with early stopping on (X_tuning, y_tuning).

        Args:
            slots: int array, one per column: columns sharing a slot are alternatives (one is used per
                tree); -1 means always used. None: every column in its own slot (no column sampling).
        """
        check_inputs(X_train, y_train, X_tuning, y_tuning)
        lib = library()
        X = sp.csc_matrix(X_train, dtype=np.float32)
        X.sort_indices()
        n, m = X.shape
        cptr, cidx, cval = X.indptr.astype(np.int64), X.indices.astype(np.int32, copy=False), X.data
        y = np.ascontiguousarray(y_train, dtype=np.float32)
        prevalence = float(y.mean()) if len(y) else 0.0
        profile = self._resolve_profile(prevalence)
        tptr, tidx, tval, n_tun = _csr(X_tuning)
        yt = np.ascontiguousarray(y_tuning, dtype=np.float32)
        slots = np.arange(m, dtype=np.int32) if slots is None else np.ascontiguousarray(slots, dtype=np.int32)
        if slots.shape != (m,):
            raise ValueError(f"slots must have one entry per column ({m}), got {slots.shape}")
        empty_i64, empty_i32, empty_f32 = (
            np.zeros(1, np.int64),
            np.zeros(1, np.int32),
            np.zeros(1, np.float32),
        )
        out_tun, out_held, info = np.zeros(max(n_tun, 1)), np.zeros(1), np.zeros(16)
        self.free()
        self._handle = lib.medsboost_train(
            n,
            m,
            _ptr(cptr, I64P),
            _ptr(cidx, I32P),
            _ptr(cval, F32P),
            _ptr(y, F32P),
            n_tun,
            _ptr(tptr, I64P),
            _ptr(tidx, I32P),
            _ptr(tval, F32P),
            _ptr(yt, F32P),
            0,
            _ptr(empty_i64, I64P),
            _ptr(empty_i32, I32P),
            _ptr(empty_f32, F32P),
            _ptr(slots, I32P),
            _ptr(self._param_array(), F64P),
            _ptr(out_tun, F64P),
            _ptr(out_held, F64P),
            _ptr(info, F64P),
        )
        if not self._handle:  # null: training failed; the library kept the reason
            self._handle = None
            raise RuntimeError(f"medsboost training failed: {lib.medsboost_last_error().decode()}")
        self.info = {
            "profile": profile,
            "prevalence": prevalence,
            "best_iteration": int(info[0]),
            "trees_trained": int(info[1]),
            "binning_s": float(info[2]),
            "boosting_s": float(info[4]),
            # Phase timers of the C++ trainer, in seconds, for profiling: binning the tuning/held-out rows,
            # building column layouts, growing trees (of which root histograms, partitioning and the
            # histogram passes of splits) and updating scores.
            "eval_binning_s": float(info[3]),
            "subset_s": float(info[5]),
            "grow_s": float(info[6]),
            "root_s": float(info[9]),
            "partition_s": float(info[10]),
            "split_hist_s": float(info[11] + info[12]),
            "score_s": float(info[7]),
        }
        return self

    def predict_raw(self, X) -> np.ndarray:
        """Log-odds for each row of X (same column layout as training)."""
        if self._handle is None:
            raise RuntimeError("model is not trained or loaded")
        lib = library()
        ptr, idx, val, n = _csr(X)
        if X.shape[1] != lib.medsboost_num_features(self._handle):
            raise ValueError(
                f"X has {X.shape[1]} columns, the model {lib.medsboost_num_features(self._handle)}"
            )
        out = np.zeros(n)
        lib.medsboost_predict(
            self._handle,
            n,
            _ptr(ptr, I64P),
            _ptr(idx, I32P),
            _ptr(val, F32P),
            int(self.params["threads"]),
            _ptr(out, F64P),
        )
        return out

    def predict_proba(self, X) -> np.ndarray:
        """Probability of the positive class for each row of X."""
        return 1.0 / (1.0 + np.exp(-self.predict_raw(X)))

    @property
    def num_trees(self) -> int:
        return int(library().medsboost_num_trees(self._handle)) if self._handle is not None else 0

    def save(self, path, save_stats: bool | None = None) -> None:
        """Writes the model in medsboost's text format.

        Args:
            save_stats: write node statistics: split gains and node / leaf cover (format version 2); None uses
                the ``save_stats`` the model was created with.

        Raises:
            ValueError: statistics were requested for a model loaded from a file saved without them.
            OSError: the file could not be written.
        """
        if self._handle is None:
            raise OSError(f"could not save the model to {path}: it is not trained or loaded")
        stats = self.save_stats if save_stats is None else bool(save_stats)
        status = library().medsboost_save_ex(self._handle, str(path).encode(), int(stats))
        if status == 2:
            raise ValueError("this model was loaded from a file saved without node statistics")
        if status != 0:
            raise OSError(f"could not save the model to {path}")

    def feature_importance(self, importance_type: str = "split") -> np.ndarray:
        """Importance of each column, like LightGBM's ``Booster.feature_importance``.

        Args:
            importance_type: "split" (int array: how many splits use the column) or "gain" (float array: the
                total gain of those splits).

        Raises:
            ValueError: an unknown importance type, or "gain" for a model loaded from a file saved without
                node statistics (see ``save_stats``).
        """
        kinds = {"split": 0, "gain": 1}
        if importance_type not in kinds:
            raise ValueError(f"importance_type must be one of {sorted(kinds)}, got {importance_type!r}")
        if self._handle is None:
            raise RuntimeError("model is not trained or loaded")
        lib = library()
        out = np.zeros(int(lib.medsboost_num_features(self._handle)))
        status = lib.medsboost_feature_importance(self._handle, kinds[importance_type], _ptr(out, F64P))
        if status == 2:
            raise ValueError(
                "this model was loaded from a file saved without node statistics; "
                "save it with save_stats=True"
            )
        return out.astype(np.int64) if importance_type == "split" else out

    def shap_values(self, X) -> tuple[np.ndarray, np.ndarray, float]:
        """Exact SHAP values of each row of X (same column layout as training), in log-odds.

        Path-dependent TreeSHAP (Lundberg, Erion and Lee, 2018), like ``shap.TreeExplainer`` with
        ``feature_perturbation="tree_path_dependent"`` and LightGBM's ``pred_contrib``: a feature's value is
        averaged out over the training rows that reach each node (the node row counts saved with
        ``save_stats``). Each row's values and the expected value add up to ``predict_raw``.

        Only the columns the trees split on can have a non-zero value, so only those are returned: on wide
        tabularized data they are a small part of the columns.

        Returns:
            values: (rows, k) contributions of the k used columns.
            columns: (k,) their column indices in X, ascending.
            expected_value: the model's expected log-odds over the training rows, the same for every row.

        Raises:
            ValueError: the model was loaded from a file saved without node statistics (see ``save_stats``).
        """
        if self._handle is None:
            raise RuntimeError("model is not trained or loaded")
        lib = library()
        ptr, idx, val, n = _csr(X)
        if X.shape[1] != lib.medsboost_num_features(self._handle):
            raise ValueError(
                f"X has {X.shape[1]} columns, the model {lib.medsboost_num_features(self._handle)}"
            )
        k = int(lib.medsboost_num_used_features(self._handle))
        columns = np.zeros(k, dtype=np.int32)
        lib.medsboost_used_features(self._handle, _ptr(columns, I32P))
        out = np.zeros((n, k + 1))
        status = lib.medsboost_shap(
            self._handle,
            n,
            _ptr(ptr, I64P),
            _ptr(idx, I32P),
            _ptr(val, F32P),
            int(self.params["threads"]),
            _ptr(out, F64P),
        )
        if status == 2:
            raise ValueError(
                "this model was loaded from a file saved without node statistics; "
                "save it with save_stats=True"
            )
        expected = float(out[0, k]) if n else float("nan")
        return out[:, :k], columns.astype(np.int64), expected

    @classmethod
    def load(cls, path, threads: int = 0) -> MedsBoost:
        model = cls(threads=threads)
        model._handle = library().medsboost_load(str(path).encode())
        if not model._handle:
            raise OSError(f"could not load a medsboost model from {path}")
        return model

    def free(self) -> None:
        if self._handle is not None:
            library().medsboost_free(self._handle)
            self._handle = None

    def __del__(self):
        with contextlib.suppress(Exception):  # interpreter shutdown
            self.free()
