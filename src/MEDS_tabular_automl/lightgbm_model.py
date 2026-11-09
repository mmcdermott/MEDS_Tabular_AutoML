import logging
import time
from pathlib import Path

import lightgbm as lgb
import meds
import numpy as np
from omegaconf import OmegaConf

from .gbdt_model import GradientBoostedTreesModel

logger = logging.getLogger(__name__)


class LightGBMModel(GradientBoostedTreesModel):
    """Gradient-boosted trees with LightGBM, on the same data, splits and outputs as `XGBoostModel`.

    LightGBM bins each feature once, when its training `Dataset` is constructed, and bundles sparse features
    that are rarely non-zero together (exclusive feature bundling), which suits the wide, sparse matrices
    that tabularization produces. The data is always loaded in memory: LightGBM has no sparse streaming
    input, so `data_loading_params.keep_data_in_memory=False` only logs a warning.

    Unlike XGBoost, LightGBM reads absent sparse entries as 0, so an observed value of 0 and no measurement
    look the same in value aggregations; `value/count` carries the distinction (see the usage guide).

    Iterators (plain `TabularDataset`s), training, evaluation and the predictions frame come from
    `GradientBoostedTreesModel`.

    Examples:
        >>> rng = np.random.default_rng(0)
        >>> X = sp.random(200, 30, density=0.2, format="csc", random_state=0)
        >>> y = (X[:, 0].toarray().ravel() > 0).astype(np.float32)
        >>> cfg = OmegaConf.create({
        ...     "data_loading_params": {"keep_data_in_memory": True},
        ...     "model": {"objective": "binary", "num_threads": 1, "verbosity": -1, "min_child_samples": 5},
        ...     "training_params": {"num_boost_round": 20, "early_stopping_rounds": 5},
        ... })
        >>> model = LightGBMModel(cfg)
        >>> model._set_data((X, y), (X, y), (X, y))
        >>> model._train()
        >>> model.evaluate("tuning") > 0.9
        True
        >>> with tempfile.TemporaryDirectory() as d:
        ...     model.save_model(Path(d) / "lightgbm.txt")
        ...     reloaded = LightGBMModel(cfg)
        ...     reloaded.load_model(Path(d) / "lightgbm.txt")
        ...     reloaded._set_data((X, y), (X, y), (X, y))
        ...     bool(np.allclose(reloaded._predict("held_out")[1], model._predict("held_out")[1]))
        True

        Integer matrices, as the native backend writes for counts and presence, are accepted too:

        >>> X_int = X.astype(np.uint8)
        >>> model_int = LightGBMModel(cfg)
        >>> model_int._set_data((X_int, y), (X_int, y), (X_int, y))
        >>> model_int._train()
        >>> model_int.dtuning[0].dtype
        dtype('float32')
        >>> model._predict("train")
        Traceback (most recent call last):
            ...
        ValueError: Invalid split for evaluation: train
    """

    def _build(self):
        """Loads every split in memory and wraps the training split in a LightGBM `Dataset`."""
        if not self.keep_data_in_memory:
            logger.warning("LightGBM loads the data in memory; ignoring keep_data_in_memory=False.")
        start = time.perf_counter()
        self._build_iterators()
        self._set_data(self.itrain.get_data(), self.ituning.get_data(), self.iheld_out.get_data())
        logger.info(f"LightGBM: loaded train/tuning/held_out in {time.perf_counter() - start:.1f}s")

    def _set_data(self, train, tuning, held_out):
        """Keeps tuning and held-out matrices for prediction; the training one only lives in the `Dataset`.

        LightGBM's sparse input must be float32 or float64, while tabularized blocks can be integer (the
        native backend writes counts and presence as uint8), so integer matrices are cast to float32.
        """

        def as_float(X):
            return X if X.dtype in (np.float32, np.float64) else X.astype(np.float32)

        X_train, y_train = train
        self.dtrain = lgb.Dataset(as_float(X_train), label=y_train)
        self.dtuning = (as_float(tuning[0]), tuning[1])
        self.dheld_out = (as_float(held_out[0]), held_out[1])

    def _train(self):
        """Trains with early stopping on the tuning split; the binned `Dataset` is built on first use."""
        params = OmegaConf.to_container(self.cfg.model, resolve=True)
        X_tuning, y_tuning = self.dtuning
        start = time.perf_counter()
        self.model = lgb.train(
            params,
            self.dtrain,
            num_boost_round=self.cfg.training_params.num_boost_round,
            valid_sets=[lgb.Dataset(X_tuning, label=y_tuning, reference=self.dtrain)],
            callbacks=[lgb.early_stopping(self.cfg.training_params.early_stopping_rounds, verbose=False)],
        )
        # lgb.train returns the model truncated at its best early-stopping round.
        logger.info(
            f"LightGBM: kept {self.model.current_iteration()} rounds (best on tuning) "
            f"after {time.perf_counter() - start:.1f}s of training"
        )

    def _predict(self, split="held_out") -> tuple[np.ndarray, np.ndarray]:
        """Returns labels and predicted probabilities, at the best early-stopping iteration."""
        if split == meds.tuning_split:
            X, y = self.dtuning
        elif split == meds.held_out_split:
            X, y = self.dheld_out
        else:
            raise ValueError(f"Invalid split for evaluation: {split}")
        return np.asarray(y), self.model.predict(X)

    def save_model(self, output_fp: Path):
        """Saves the model in LightGBM's text format (best iteration included)."""
        self.model.save_model(str(output_fp))

    def load_model(self, model_fp: Path):
        self.model = lgb.Booster(model_file=str(model_fp))
