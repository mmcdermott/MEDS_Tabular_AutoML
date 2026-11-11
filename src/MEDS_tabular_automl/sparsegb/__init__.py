"""sparsegb: gradient-boosted trees for wide, sparse tabularized data (binary classification).

The same algorithm as LightGBM in MEDS-Tab's configuration (Newton boosting on the logistic loss,
per-feature histograms, leaf-wise growth, `min_child_samples` / `lambda_l2` constraints, bagging, early
stopping on tuning log-loss), engineered for matrices with ~1e5 columns and ~1e3 non-zeros per row: absent
entries are never stored or scanned, features that can no longer split in a node are dropped from it and its
descendants, and each split is one parallel pass. Columns can be grouped into "slots" (one code and
aggregation, all windows); each tree then uses one column per slot, drawn from precomputed layouts.

The C++ library (csrc/sparsegb) is built into this package as ``_sparsegb`` when MEDS-Tab is installed with a
C++17 compiler and OpenMP; nothing loads it unless this module is used.

Examples:
    >>> from sklearn.metrics import roc_auc_score
    >>> X = sp.random(600, 40, density=0.3, format="csc", random_state=0, dtype=np.float32)
    >>> y = (X[:, 0].toarray().ravel() + X[:, 1].toarray().ravel() > 0.6).astype(np.float32)
    >>> model = SparseGB(threads=2, num_leaves=15, min_child_samples=5, max_rounds=40, early_stopping=10)
    >>> model = model.fit(X[:400], y[:400], X[400:500], y[400:500])
    >>> p = model.predict_proba(X[500:])
    >>> p.shape, bool(((p > 0) & (p < 1)).all())
    ((100,), True)
    >>> bool(roc_auc_score(y[500:], p) > 0.9)
    True
    >>> with tempfile.TemporaryDirectory() as d:
    ...     model.save(Path(d) / "model.txt")
    ...     reloaded = SparseGB.load(Path(d) / "model.txt")
    >>> bool(np.allclose(reloaded.predict_proba(X[500:]), p))
    True
    >>> model.num_trees > 0
    True
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
    "sparsegb is not built. The compiled wheels (`pip install meds-tab` on Linux or Apple-silicon macOS 15+, "
    "Python 3.11-3.13) include it; elsewhere install MEDS-Tab from source with a C++17 compiler and OpenMP "
    "(`pip install --no-binary meds-tab meds-tab`). Windows is not supported yet."
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
        ValueError: the tuning split is empty; sparsegb early-stops on it
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
            hint = "; sparsegb early-stops on it" if name == "tuning" else ""
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
    spec = importlib.util.find_spec(f"{__package__}._sparsegb")
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
    """Whether the sparsegb library is built."""
    try:
        library()
    except (FileNotFoundError, OSError):
        return False
    return True


@cache
def library() -> ct.CDLL:
    lib = ct.CDLL(str(library_path()))
    lib.sgb_train.restype = ct.c_void_p
    lib.sgb_train.argtypes = [
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
    lib.sgb_last_error.restype = ct.c_char_p
    lib.sgb_last_error.argtypes = []
    lib.sgb_predict.restype = ct.c_int
    lib.sgb_predict.argtypes = [ct.c_void_p, ct.c_int64, I64P, I32P, F32P, ct.c_int, F64P]
    lib.sgb_save.restype = ct.c_int
    lib.sgb_save.argtypes = [ct.c_void_p, ct.c_char_p]
    lib.sgb_load.restype = ct.c_void_p
    lib.sgb_load.argtypes = [ct.c_char_p]
    lib.sgb_free.restype = None
    lib.sgb_free.argtypes = [ct.c_void_p]
    lib.sgb_num_trees.restype = ct.c_int64
    lib.sgb_num_trees.argtypes = [ct.c_void_p]
    lib.sgb_num_features.restype = ct.c_int64
    lib.sgb_num_features.argtypes = [ct.c_void_p]
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


class SparseGB:
    """Binary gradient-boosted trees; see the module docstring.

    Args:
        threads: worker threads (0: ``OMP_NUM_THREADS`` if set, else every available CPU).
        max_bin: bins per feature (zero has its own bin), at most 255.
        num_leaves: leaves per tree (leaf-wise growth).
        min_child_samples: minimum rows in each child of a split.
        lambda_l2: L2 penalty on leaf values.
        learning_rate: shrinkage.
        bagging_fraction: rows sampled (without replacement) per tree.
        max_rounds: maximum number of trees.
        early_stopping: stop after this many trees without a better tuning log-loss; the best tree count is
            kept.
        seed: random seed (bagging and column sampling).
        min_sum_hessian: minimum hessian sum in each child.
        layouts: number of precomputed column layouts (multiple of 7); 0 draws a fresh column sample every
            `refresh` trees instead. Only matters when slots are given.
        refresh: trees per column sample (or per layout draw).
    """

    def __init__(
        self,
        threads: int = 0,
        max_bin: int = 15,
        num_leaves: int = 127,
        min_child_samples: int = 50,
        lambda_l2: float = 1.0,
        learning_rate: float = 0.05,
        bagging_fraction: float = 0.8,
        max_rounds: int = 1000,
        early_stopping: int = 20,
        seed: int = 1,
        min_sum_hessian: float = 1e-3,
        layouts: int = 14,
        refresh: int = 1,
    ):
        self.params = {
            "threads": threads or default_threads(),
            "max_bin": max_bin,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "lambda_l2": lambda_l2,
            "learning_rate": learning_rate,
            "bagging_fraction": bagging_fraction,
            "max_rounds": max_rounds,
            "early_stopping": early_stopping,
            "seed": seed,
            "min_sum_hessian": min_sum_hessian,
            "layouts": layouts,
            "refresh": refresh,
        }
        self._handle = None
        self.info: dict = {}

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
                1,
                p["refresh"],
                1,
                p["layouts"],
                0.0,  # GOSS top share (0: bagging)
                0.0,  # GOSS other share
            ],
            dtype=np.float64,
        )

    def fit(self, X_train, y_train, X_tuning, y_tuning, slots: np.ndarray | None = None) -> SparseGB:
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
        self._handle = lib.sgb_train(
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
            raise RuntimeError(f"sparsegb training failed: {lib.sgb_last_error().decode()}")
        self.info = {
            "best_iteration": int(info[0]),
            "trees_trained": int(info[1]),
            "binning_s": float(info[2]),
            "boosting_s": float(info[4]),
        }
        return self

    def predict_raw(self, X) -> np.ndarray:
        """Log-odds for each row of X (same column layout as training)."""
        if self._handle is None:
            raise RuntimeError("model is not trained or loaded")
        lib = library()
        ptr, idx, val, n = _csr(X)
        if X.shape[1] != lib.sgb_num_features(self._handle):
            raise ValueError(f"X has {X.shape[1]} columns, the model {lib.sgb_num_features(self._handle)}")
        out = np.zeros(n)
        lib.sgb_predict(
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
        return int(library().sgb_num_trees(self._handle)) if self._handle is not None else 0

    def save(self, path) -> None:
        """Writes the model in sparsegb's text format."""
        if self._handle is None or library().sgb_save(self._handle, str(path).encode()) != 0:
            raise OSError(f"could not save the model to {path}")

    @classmethod
    def load(cls, path, threads: int = 0) -> SparseGB:
        model = cls(threads=threads)
        model._handle = library().sgb_load(str(path).encode())
        if not model._handle:
            raise OSError(f"could not load a sparsegb model from {path}")
        return model

    def free(self) -> None:
        if self._handle is not None:
            library().sgb_free(self._handle)
            self._handle = None

    def __del__(self):
        with contextlib.suppress(Exception):  # interpreter shutdown
            self.free()
