"""Checks an installed compiled MEDS-Tab wheel (run by cibuildwheel after each build).

Both compiled libraries are optional extensions, so a failed compile still produces a wheel; this makes such a
wheel fail instead. It loads the native engine, checks that medsboost is built, and runs medsboost's doctests
(train, predict, save and reload, input checks) against the installed library.
"""

import doctest
import tempfile

from MEDS_tabular_automl import medsboost
from MEDS_tabular_automl.native import engine

engine.library()
assert medsboost.is_available(), medsboost.NOT_BUILT
result = doctest.testmod(medsboost, extraglobs={"tempfile": tempfile})
assert result.attempted > 0 and result.failed == 0, result
print(f"native engine and medsboost OK: {result.attempted} doctest examples passed")
