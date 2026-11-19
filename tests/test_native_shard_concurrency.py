"""Shards computed concurrently (Pipeline shards_in_flight) and a shard's blocks written in one engine call
(engine.write_blocks): the same files as one shard and one block at a time."""

import json
import shutil

import numpy as np
import pytest
from scipy.sparse import csr_array

from MEDS_tabular_automl.native import engine
from MEDS_tabular_automl.native.contract import TabularConfig
from MEDS_tabular_automl.native.pipeline import Pipeline
from tests.test_native_backend import engine_built, run, write_labels, write_meds
from tests.test_native_engine import INTEGER, VALUES, random_csr, with_data

pytestmark = pytest.mark.skipif(not engine_built(), reason="The native engine is not built.")


def test_write_blocks_matches_write_block_bytes(tmp_path):
    rng = np.random.default_rng(0)
    blocks = []
    for i, (kind, values) in enumerate(VALUES.items()):
        for index in (np.int32, np.int64):
            matrix = random_csr(rng, 40, 300, 900, index)
            matrix = with_data(matrix, values(rng, matrix.nnz))
            blocks.append(
                (tmp_path / "many" / f"w{i % 3}" / f"{kind}_{index.__name__}.npz", matrix, kind in INTEGER)
            )
    blocks.append((tmp_path / "many" / "empty.npz", csr_array((0, 0)), False))
    sizes = engine.write_blocks(blocks, threads=4)
    for (path, matrix, integer), size in zip(blocks, sizes, strict=True):
        one = tmp_path / "one.npz"
        assert engine.write_block(one, matrix, integer=integer) == size == path.stat().st_size
        assert path.read_bytes() == one.read_bytes(), path.name
    assert engine.write_blocks([], threads=4) == []


def test_write_blocks_reports_failures(tmp_path):
    good = csr_array(np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError, match="finite"):
        engine.write_blocks(
            [(tmp_path / "a.npz", good, False), (tmp_path / "b.npz", csr_array(np.array([[np.inf]])), False)],
            2,
        )
    (tmp_path / "file").write_text("")
    with pytest.raises(OSError):
        engine.write_blocks([(tmp_path / "file/a.npz", good, False)], 2)


def _tree(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("compress", [False, True])
def test_shards_in_flight_write_the_same_files(tmp_path, compress):
    meds, labels = tmp_path / "data", tmp_path / "labels"
    write_meds(meds)
    write_labels(meds, labels, duplicates=True)
    config = TabularConfig(min_code_inclusion_count=2, do_compress=compress)
    outputs = {}
    for shards in (1, 2, 5):
        output = tmp_path / f"out{shards}"
        run("describe_codes", "describe_codes", {"input_dir": str(meds), "output_dir": str(output)})
        pipeline = Pipeline(meds, output, config, threads=3, shards_in_flight=shards)
        report = pipeline.tabularize_at_labels(labels)
        assert report and not any(r["reused"] for r in report.values())
        manifests = {
            name: json.loads(text)["files"]
            for name, text in _tree(output / ".meds_tab_native" / "manifests").items()
        }
        outputs[shards] = (_tree(output / "tabularize"), manifests)
    assert outputs[1][0], "No blocks were written."
    assert outputs[2] == outputs[1]
    assert outputs[5] == outputs[1]


def test_a_failing_shard_publishes_nothing_partial(tmp_path, monkeypatch):
    """A failure while shards are in flight leaves no staged directories and no manifest for unfinished
    shards; the rerun completes."""
    meds, output, labels = tmp_path / "data", tmp_path / "out", tmp_path / "labels"
    write_meds(meds)
    write_labels(meds, labels)
    run("describe_codes", "describe_codes", {"input_dir": str(meds), "output_dir": str(output)})
    config = TabularConfig(min_code_inclusion_count=2)
    calls = []
    real = engine.write_blocks

    def fail_second(blocks, threads):
        calls.append(len(blocks))
        if len(calls) == 2:
            raise OSError("disk full")
        return real(blocks, threads)

    monkeypatch.setattr(engine, "write_blocks", fail_second)
    with pytest.raises(OSError, match="disk full"):
        Pipeline(meds, output, config, threads=2, shards_in_flight=3).tabularize_at_labels(labels)
    staging = output / ".meds_tab_native" / "staging"
    assert not staging.exists() or not any(staging.iterdir())
    monkeypatch.setattr(engine, "write_blocks", real)
    rerun = Pipeline(meds, output, config, threads=2, shards_in_flight=3)
    rerun.tabularize_at_labels(labels)
    reference = tmp_path / "reference"
    shutil.copytree(output / "metadata", reference / "metadata")
    Pipeline(meds, reference, config, threads=2, shards_in_flight=1).tabularize_at_labels(labels)
    assert _tree(output / "tabularize") == _tree(reference / "tabularize")


@pytest.mark.parametrize("shards", [0, -1, 1.5])
def test_invalid_shards_in_flight_is_rejected(tmp_path, shards):
    with pytest.raises(ValueError, match="shards_in_flight"):
        Pipeline(tmp_path, tmp_path, TabularConfig(), shards_in_flight=shards)
