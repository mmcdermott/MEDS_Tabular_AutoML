import logging
import time
from pathlib import Path

import meds
import numpy as np
import polars as pl
import scipy.sparse as sp

from . import medsboost
from .file_name import get_model_files
from .gbdt_model import GradientBoostedTreesModel
from .medsboost import MedsBoost
from .medsboost.ensemble import MedsBoostEnsemble
from .tabular_dataset import TabularDataset

logger = logging.getLogger(__name__)


# Aggregations that share another's slot: a code's recency is one more alternative to its windowed counts.
SLOT_OF_AGG = {"code/recency": "code/count", "code/group_recency": "code/group_count"}


def _matrix_columns(it: TabularDataset):
    """(name, aggregation, is_static) of each column of the loaded matrix, in its order: the loader's, model
    files in `get_model_files` order, each filtered by its code mask."""
    names = it.get_all_column_names()  # f"{feature}/{agg}/{window}", before the code masks
    pos = 0
    for f in get_model_files(it.cfg, it.split, it._data_shards[0]):
        is_static = f.stem in ("mean", "present")
        agg = f"static/{f.stem}" if is_static else f"{f.parent.stem}/{f.stem}"
        mask = np.asarray(it.code_masks[agg])
        for name, keep in zip(names[pos : pos + len(mask)], mask, strict=True):
            if keep:
                yield name, agg, is_static
        pos += len(mask)


def column_slots(it: TabularDataset) -> np.ndarray:
    """One slot per (code, aggregation): all windows of a code's aggregation are alternatives; static columns
    -1. A code's code/recency column joins its code/count slot (see SLOT_OF_AGG), so each tree uses either one
    of its windowed counts or its recency.

    The column order is the loader's: model files in `get_model_files` order, each filtered by its code mask.
    """
    slot_of, keys = [], {}
    for name, agg, is_static in _matrix_columns(it):
        if is_static:
            slot_of.append(-1)
        else:
            slot_of.append(keys.setdefault((name.rsplit("/", 3)[0], SLOT_OF_AGG.get(agg, agg)), len(keys)))
    return np.asarray(slot_of, dtype=np.int32)


def _kept_columns(X_train, min_nnz: int) -> np.ndarray:
    """Indices of the columns with at least min_nnz stored training entries (see min_column_nnz)."""
    return np.flatnonzero(np.diff(sp.csc_matrix(X_train, dtype=np.float32).indptr) >= min_nnz)


def matrix_column_names(it: TabularDataset, min_column_nnz: int = 0) -> list[str]:
    """Names of the columns a model is trained on, in the order of ``MedsBoost.feature_importance``.

    Each name is ``f"{feature}/{agg}/{window}"``. Unlike ``it.get_all_column_names()``, it leaves out the
    columns the code masks (``tabularization.min_code_inclusion_count`` etc.) drop and, given the trial's
    ``model_launcher.min_column_nnz``, the rare columns medsboost drops (``it`` must then be the train split).
    """
    names = [name for name, _, _ in _matrix_columns(it)]
    if min_column_nnz > 1:
        X, _ = it.get_data()
        names = [names[i] for i in _kept_columns(X, min_column_nnz)]
    return names


class MedsBoostModel(GradientBoostedTreesModel):
    """Gradient-boosted trees with medsboost (`MEDS_tabular_automl.medsboost`), MEDS-Tab's sparse booster.

    Same data, splits and outputs as the XGBoost and LightGBM launchers. Training uses one column per
    (code, aggregation) per tree, drawn from precomputed layouts (`model.layouts`), and early-stops on the
    tuning split's log-loss. The data is loaded in memory.

    Examples:
        >>> rng = np.random.default_rng(0)
        >>> X = sp.random(600, 40, density=0.3, format="csc", random_state=0, dtype=np.float32)
        >>> y = (X[:, 0].toarray().ravel() + X[:, 1].toarray().ravel() > 0.6).astype(np.float32)
        >>> cfg = OmegaConf.create({
        ...     "data_loading_params": {"keep_data_in_memory": True},
        ...     "model": {"threads": 2, "num_leaves": 15, "min_child_samples": 5, "max_rounds": 40,
        ...               "early_stopping": 10},
        ... })
        >>> model = MedsBoostModel(cfg)
        >>> model._set_data((X[:400], y[:400]), (X[400:500], y[400:500]), (X[500:], y[500:]), slots=None)
        >>> model._train()
        >>> model.evaluate("held_out") > 0.9
        True
        >>> with tempfile.TemporaryDirectory() as d:
        ...     model.save_model(Path(d) / "medsboost.txt")
        ...     reloaded = MedsBoostModel(cfg)
        ...     reloaded.load_model(Path(d) / "medsboost.txt")
        ...     splits = ((X[:400], y[:400]), (X[400:500], y[400:500]), (X[500:], y[500:]))
        ...     reloaded._set_data(*splits, slots=None)
        ...     bool(np.allclose(reloaded._predict("held_out")[1], model._predict("held_out")[1]))
        True
        >>> model._predict("train")
        Traceback (most recent call last):
            ...
        ValueError: Invalid split for evaluation: train
    """

    def _build(self):
        """Loads every split in memory and computes the column slots."""
        if not medsboost.is_available():  # fail before spending minutes loading the data
            raise ImportError(
                f"{medsboost.NOT_BUILT} Or choose a launcher that needs no compiled library, e.g. "
                "model_launcher=lightgbm."
            )
        if not self.keep_data_in_memory:
            logger.warning("medsboost loads the data in memory; ignoring keep_data_in_memory=False.")
        start = time.perf_counter()
        self._build_iterators()
        self._set_data(
            self.itrain.get_data(),
            self.ituning.get_data(),
            self.iheld_out.get_data(),
            slots=column_slots(self.itrain),
        )
        self._drop_rare_columns(int(self.cfg.get("min_column_nnz") or 0))
        if self._ensemble_members() > 1:
            self.groups = (self._subject_ids(self.itrain), self._subject_ids(self.ituning))
        logger.info(f"medsboost: loaded train/tuning/held_out in {time.perf_counter() - start:.1f}s")

    @staticmethod
    def _subject_ids(it: TabularDataset) -> np.ndarray:
        """The subject id of each loaded row of a split, in the loader's order (label shards, then rows)."""
        label_dir = Path(it.cfg.path.input_label_dir) / it.split
        return np.concatenate(
            [
                pl.read_parquet((label_dir / shard).with_suffix(".parquet"), columns=["subject_id"])[
                    "subject_id"
                ].to_numpy()
                for shard in it._data_shards
            ]
        )

    def _ensemble_members(self) -> int:
        ensemble = self.cfg.get("ensemble")
        return int(ensemble.get("members") or 1) if ensemble is not None else 1

    def _drop_rare_columns(self, min_nnz: int):
        """Drops the columns with fewer than min_nnz nonzero training rows from every split (and their slots).

        Such columns are rarely split on, yet each tree scans them; dropping them speeds training up.

        Examples:
            >>> X = sp.csc_matrix(np.array([[1, 0, 2], [1, 0, 0], [1, 3, 0]], dtype=np.float32))
            >>> cfg = OmegaConf.create({"data_loading_params": {"keep_data_in_memory": True}})
            >>> model = MedsBoostModel(cfg)
            >>> model._set_data((X, [0, 1, 0]), (X, [0, 1, 0]), (X, [0, 1, 0]), slots=np.array([0, 1, 1]))
            >>> model._drop_rare_columns(2)
            >>> model.dtrain[0].toarray().tolist(), model.slots.tolist()
            ([[1.0], [1.0], [1.0]], [0])
        """
        if min_nnz <= 1:
            return
        keep = _kept_columns(self.dtrain[0], min_nnz)
        logger.info(
            f"medsboost: keeping {len(keep)} of {self.dtrain[0].shape[1]} columns (min_column_nnz={min_nnz})"
        )
        self.dtrain = (self.dtrain[0][:, keep], self.dtrain[1])
        self.dtuning = (self.dtuning[0][:, keep], self.dtuning[1])
        self.dheld_out = (self.dheld_out[0][:, keep], self.dheld_out[1])
        if self.slots is not None:
            self.slots = np.asarray(self.slots)[keep]

    def _set_data(self, train, tuning, held_out, slots):
        def as_float(X):
            return sp.csc_matrix(X, dtype=np.float32)

        self.dtrain = (as_float(train[0]), np.asarray(train[1]))
        self.dtuning = (as_float(tuning[0]), np.asarray(tuning[1]))
        self.dheld_out = (as_float(held_out[0]), np.asarray(held_out[1]))
        self.slots = slots
        # Subject ids of the (train, tuning) rows; set by _build when an ensemble is configured.
        self.groups = None

    def _train(self):
        """Trains with early stopping on the tuning split; keeps the best tree count.

        With ``ensemble.members`` > 1, trains a ``MedsBoostEnsemble`` instead: every member early-stops on its
        own ``ensemble.stop_fraction`` of the subjects of the training rows (train, or train+tuning with
        ``ensemble.pool_tuning``) rather than on the tuning split.

        Examples:
            >>> rng = np.random.default_rng(0)
            >>> X = sp.random(900, 40, density=0.3, format="csc", random_state=0, dtype=np.float32)
            >>> y = (X[:, 0].toarray().ravel() + X[:, 1].toarray().ravel() > 0.6).astype(np.float32)
            >>> cfg = OmegaConf.create({
            ...     "data_loading_params": {"keep_data_in_memory": True},
            ...     "model": {"threads": 2, "num_leaves": 15, "min_child_samples": 5, "max_rounds": 40,
            ...               "early_stopping": 10, "seed": 1},
            ...     "ensemble": {"members": 4, "member_params": [{"num_leaves": 7}, {}], "stop_fraction": 0.2,
            ...                  "pool_tuning": True},
            ... })
            >>> model = MedsBoostModel(cfg)
            >>> model._set_data((X[:600], y[:600]), (X[600:750], y[600:750]), (X[750:], y[750:]), slots=None)
            >>> model.groups = (np.repeat(np.arange(200), 3), np.repeat(np.arange(200, 250), 3))
            >>> model._train()
            >>> len(model.model.models), model.evaluate("held_out") > 0.9
            (4, True)
            >>> with tempfile.TemporaryDirectory() as d:
            ...     model.save_model(Path(d) / "medsboost.txt")
            ...     reloaded = MedsBoostModel(cfg)
            ...     reloaded.load_model(Path(d) / "medsboost.txt")
            ...     reloaded._set_data((X[:600], y[:600]), (X[600:750], y[600:750]), (X[750:], y[750:]), None)
            ...     bool(np.allclose(reloaded._predict("held_out")[1], model._predict("held_out")[1]))
            True
        """
        params = dict(self.cfg.model) if self.cfg.get("model") is not None else {}
        start = time.perf_counter()
        members = self._ensemble_members()
        if members <= 1:
            self.model = MedsBoost(**params).fit(*self.dtrain, *self.dtuning, slots=self.slots)
            logger.info(
                f"medsboost: kept {self.model.num_trees} trees (best on tuning) of "
                f"{self.model.info['trees_trained']} after {time.perf_counter() - start:.1f}s"
            )
            return

        ens = self.cfg.ensemble
        (X_tr, y_tr), (X_tu, y_tu) = self.dtrain, self.dtuning
        g_tr, g_tu = self.groups if self.groups is not None else (None, None)
        if g_tr is None:
            logger.warning("medsboost ensemble: no subject ids; each row is treated as its own subject.")
            g_tr = np.arange(X_tr.shape[0])
            g_tu = np.arange(X_tr.shape[0], X_tr.shape[0] + X_tu.shape[0])
        if ens.get("pool_tuning", False):
            logger.warning(
                "medsboost ensemble: pool_tuning trains on train+tuning, so the reported tuning AUROC is "
                "in-sample. Use it for single runs, not to compare sweep trials."
            )
            X = sp.vstack([X_tr, X_tu]).tocsr()
            y = np.concatenate([np.asarray(y_tr), np.asarray(y_tu)])
            groups = np.concatenate([np.asarray(g_tr), np.asarray(g_tu)])
        else:
            X, y, groups = sp.csr_matrix(X_tr), np.asarray(y_tr), np.asarray(g_tr)
        member_params = [dict(p) for p in (ens.get("member_params") or [])]
        self.model = MedsBoostEnsemble(
            members=members,
            member_params=member_params,
            stop_fraction=float(ens.get("stop_fraction", 0.11)),
            base_params=params,
        ).fit(X, y, groups, slots=self.slots)
        logger.info(
            f"medsboost ensemble: {members} members (learning rates "
            f"{self.model.info['learning_rates']}) kept {self.model.info['trees']} trees "
            f"(profiles {self.model.info['profiles']}) after {time.perf_counter() - start:.1f}s"
        )

    def _predict(self, split="held_out") -> tuple[np.ndarray, np.ndarray]:
        """Returns labels and predicted probabilities."""
        if split == meds.tuning_split:
            X, y = self.dtuning
        elif split == meds.held_out_split:
            X, y = self.dheld_out
        else:
            raise ValueError(f"Invalid split for evaluation: {split}")
        return y, self.model.predict_proba(X)

    def save_model(self, output_fp: Path):
        """Saves the model in medsboost's text format (an ensemble: one file per member and a manifest)."""
        self.model.save(output_fp)

    def load_model(self, model_fp: Path):
        threads = int(self.cfg.model.get("threads", 0)) if self.cfg.get("model") is not None else 0
        if MedsBoostEnsemble.is_ensemble_file(model_fp):
            self.model = MedsBoostEnsemble.load(model_fp, threads=threads)
        else:
            self.model = MedsBoost.load(model_fp, threads=threads)
