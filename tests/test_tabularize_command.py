"""The one tabularization command writes exactly what the static and time-series commands write.

With either backend, ``meds-tab-tabularize`` writes the blocks that ``meds-tab-tabularize-static`` followed by
``meds-tab-tabularize-time-series`` write.
"""

from pathlib import Path

import pytest
from scipy.sparse import coo_array

from MEDS_tabular_automl.scripts import describe_codes, tabularize, tabularize_static, tabularize_time_series
from MEDS_tabular_automl.utils import load_matrix
from tests.test_native_backend import engine_built
from tests.test_task_specific_features import TABULARIZATION, run_stage, write_inputs

BACKENDS = [
    "python",
    pytest.param(
        "native", marks=pytest.mark.skipif(not engine_built(), reason="The native engine is not built.")
    ),
]


def blocks(output_dir: Path) -> dict[str, list]:
    """Every tabularized block, by its path under the output, as its sorted stored entries."""
    root = output_dir / "tabularize"
    out = {}
    for fp in sorted(root.glob("**/*.npz")):
        stored = coo_array(load_matrix(fp))
        stored.sum_duplicates()
        out[fp.relative_to(root).as_posix()] = (
            stored.shape,
            sorted(zip(stored.row.tolist(), stored.col.tolist(), stored.data.tolist(), strict=True)),
        )
    return out


def tabularized(tmp_path: Path, backend: str, stages) -> dict[str, list]:
    data_dir, label_dir = write_inputs(tmp_path)
    shared = {"input_dir": str(data_dir), "output_dir": str(tmp_path / "output"), "tqdm": False}
    run_stage(describe_codes, "describe_codes", shared)
    labelled = {**shared, **TABULARIZATION, "input_label_dir": str(label_dir), "backend": backend}
    for stage in stages:
        run_stage(stage, "tabularization", labelled)
    return blocks(tmp_path / "output")


@pytest.mark.parametrize("backend", BACKENDS)
def test_one_command_writes_what_the_two_commands_write(tmp_path, backend):
    two = tabularized(tmp_path / "two", backend, [tabularize_static, tabularize_time_series])
    one = tabularized(tmp_path / "one", backend, [tabularize])
    assert two, "the two commands wrote no blocks"
    # Static blocks are written under ".../none/<agg>", time-series blocks under ".../<window>/<agg>".
    assert any("/none/" in name for name in two), "no static block"
    assert any("/none/" not in name for name in two), "no time-series block"
    assert one == two
