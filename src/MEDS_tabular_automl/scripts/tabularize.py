"""Tabularizes a task's data in one command: the static features, then the time-series features."""

import hydra
from omegaconf import DictConfig

from .. import TABULARIZATION_CFG, native
from . import tabularize_static, tabularize_time_series


@hydra.main(
    version_base=None, config_path=str(TABULARIZATION_CFG.parent), config_name=TABULARIZATION_CFG.stem
)
def main(cfg: DictConfig):
    """``meds-tab-tabularize``: the static then the time-series tabularization, in one command.

    It runs ``meds-tab-tabularize-static`` and then ``meds-tab-tabularize-time-series`` with the same
    arguments. The native backend writes every block during the static step, so it skips the second; the
    Python backend runs both. Each output file is written under its lock, so ``--multirun`` workers share
    the work as with the separate commands.
    """
    tabularize_static.run(cfg)
    if not native.use_native(cfg):
        tabularize_time_series.run(cfg)
