"""Deprecated alias of :mod:`MEDS_tabular_automl.medsboost`: sparsegb was renamed medsboost in MEDS-Tab 0.4.1.

Everything here is the medsboost module's; ``SparseGB`` is ``MedsBoost``. Models saved by sparsegb load with
either name.

Examples:
    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     import importlib, MEDS_tabular_automl.sparsegb as old
    ...     old = importlib.reload(old)
    >>> any(issubclass(w.category, DeprecationWarning) and "medsboost" in str(w.message) for w in caught)
    True
    >>> from MEDS_tabular_automl.medsboost import MedsBoost
    >>> old.SparseGB is MedsBoost
    True
"""

import warnings

from ..medsboost import (
    NOT_BUILT,
    MedsBoost,
    check_inputs,
    default_threads,
    is_available,
    library,
    library_path,
)

SparseGB = MedsBoost

warnings.warn(
    "MEDS_tabular_automl.sparsegb was renamed MEDS_tabular_automl.medsboost (SparseGB is now MedsBoost); "
    "the old names will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
