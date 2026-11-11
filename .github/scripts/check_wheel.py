"""Checks an installed compiled MEDS-Tab wheel (run by cibuildwheel after each build).

Both compiled libraries are optional extensions, so a failed compile still produces a wheel; this makes such
a wheel fail instead. It loads the native engine, checks that sparsegb is built, and runs sparsegb's doctests
(train, predict, save and reload, input checks) against the installed library.
"""

import doctest
import tempfile

from MEDS_tabular_automl import sparsegb
from MEDS_tabular_automl.native import engine

engine.library()
assert sparsegb.is_available(), sparsegb.NOT_BUILT
result = doctest.testmod(sparsegb, extraglobs={"tempfile": tempfile})
assert result.attempted > 0 and result.failed == 0, result
print(f"native engine and sparsegb OK: {result.attempted} doctest examples passed")
