"""Deprecated alias: the sparsegb launcher was renamed medsboost in MEDS-Tab 0.4.1.

``model_launcher=sparsegb`` still works through this class, which is ``MedsBoostModel`` with a warning.
"""

import logging

from .medsboost_model import MedsBoostModel, column_slots  # noqa: F401

logger = logging.getLogger(__name__)


class SparseGBModel(MedsBoostModel):
    """``MedsBoostModel`` under its 0.4.0 name; warns once per instance that the launcher was renamed."""

    def __init__(self, cfg):
        logger.warning(
            "model_launcher=sparsegb was renamed model_launcher=medsboost; the old name will be removed in a "
            "future release."
        )
        super().__init__(cfg)
