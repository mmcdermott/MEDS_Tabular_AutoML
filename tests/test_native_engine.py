"""The native engine's C ABI, block files and failure handling (MEDS_tabular_automl.native)."""

import gc
import json
import math
import threading
import weakref

import numpy as np
import pytest
from scipy.sparse import csr_array

from MEDS_tabular_automl.native import engine
from MEDS_tabular_automl.native.contract import TabularConfig
from MEDS_tabular_automl.native.pipeline import Pipeline, _gather
from MEDS_tabular_automl.native.storage import output_lock, write_csr_block
from MEDS_tabular_automl.utils import load_matrix
from tests.test_native_backend import engine_built, run, write_labels, write_meds

pytestmark = pytest.mark.skipif(not engine_built(), reason="The native engine is not built.")

IDENTITY = np.array([0, 1], dtype=np.int32)  # Value columns of the two codes in events().


def events():
    subjects = np.array([1, 1, 1, 2], dtype=np.int64)
    times = np.array([0, 1, 1, 5], dtype=np.int64)
    codes = np.array([0, 1, 1, 0], dtype=np.int32)
    values = np.array([1.0, np.nan, 2.5, -1.0])
    return subjects, times, codes, values


def random_csr(rng, rows, columns, entries, index=np.int64) -> csr_array:
    if not rows or not columns:
        return csr_array((rows, columns))
    keys = np.unique(rng.integers(0, rows, entries) * columns + rng.integers(0, columns, entries))
    matrix = csr_array((np.ones(len(keys)), (keys // columns, keys % columns)), (rows, columns))
    matrix.sort_indices()
    return csr_array((matrix.data, matrix.indices.astype(index), matrix.indptr.astype(index)), matrix.shape)


VALUES = {
    "float32": lambda rng, n: rng.integers(-50, 50, n) / 4.0,
    "float64": lambda rng, n: rng.normal(size=n),
    "zeros": lambda rng, n: np.where(rng.random(n) < 0.3, 0.0, rng.normal(size=n)),
    "near_float32_max": lambda rng, n: np.full(n, 3.4e38) * rng.choice([1, 1.0000001], n),
    "uint16": lambda rng, n: rng.integers(0, 60000, n).astype(np.int64),
    "negative": lambda rng, n: rng.integers(-300, 300, n).astype(np.int64),
    "negative_and_zero": lambda rng, n: np.where(rng.random(n) < 0.5, 0, -1).astype(np.int64),
    "int32_counts": lambda rng, n: rng.integers(0, 70000, n).astype(np.int32),
    "float_counts": lambda rng, n: rng.integers(0, 1000, n).astype(np.float64),
    "large": lambda rng, n: rng.integers(0, 2**40, n).astype(np.int64),
}
INTEGER = {"uint16", "negative", "negative_and_zero", "int32_counts", "float_counts", "large"}


def with_data(matrix: csr_array, data) -> csr_array:
    return csr_array((data, matrix.indices, matrix.indptr), matrix.shape)


@pytest.mark.parametrize(
    "shape",
    [(0, 0), (0, 5), (1, 1), (5, 3), (300, 70), (70000, 3), (20, 2**25), (2**24 + 1, 2)],
)
def test_native_writer_matches_python_writer_bytes(tmp_path, shape):
    """Every value type and index width; past 2^24 rows or columns the homogeneous array must be float64 so
    that float32 cannot round an index."""
    rng = np.random.default_rng(0)
    for index in (np.int32, np.int64):
        matrix = random_csr(rng, *shape, min(shape[0] * shape[1] // 3 + 1, 3000), index)
        for kind, values in VALUES.items():
            block = with_data(matrix, values(rng, matrix.nnz))
            integer = kind in INTEGER
            write_csr_block(tmp_path / "python.npz", block, integer=integer, compress=False)
            size = engine.write_block(tmp_path / "native.npz", block, integer=integer)
            written = (tmp_path / "native.npz").read_bytes()
            assert written == (tmp_path / "python.npz").read_bytes(), (kind, index)
            assert size == len(written)


def test_native_writer_rejects_invalid_blocks(tmp_path):
    with pytest.raises(ValueError, match="finite"):
        engine.write_block(tmp_path / "a.npz", csr_array(np.array([[1.0, np.inf]])), integer=False)
    matrix = csr_array(np.array([[1.0, 0.0]]))
    matrix.indices[0] = 5
    with pytest.raises(ValueError, match="columns"):
        engine.write_block(tmp_path / "a.npz", matrix, integer=False)
    (tmp_path / "file").write_text("")
    with pytest.raises(OSError):
        engine.write_block(tmp_path / "file/a.npz", csr_array((1, 1)), integer=False)


@pytest.mark.parametrize("integer", [False, True])
def test_native_extraction_matches_gathered_rows(tmp_path, integer):
    rng = np.random.default_rng(1)
    matrix = random_csr(rng, 4000, 600, 200000)
    matrix = with_data(matrix, VALUES["int32_counts" if integer else "zeros"](rng, matrix.nnz))
    source = tmp_path / "cache.npz"
    engine.write_block(source, matrix, integer=integer)
    # Unsorted, repeated, empty (-1) and adjacent rows.
    rows = np.concatenate([[-1, 3999, 0, 0], rng.integers(0, 4000, 300), np.arange(10, 40), [-1]])
    expected = _gather(matrix, rows)
    write_csr_block(tmp_path / "python.npz", expected, integer=integer, compress=False)
    written, size = engine.extract_rows(
        source, matrix.indptr, rows, 600, tmp_path / "native.npz", integer=integer
    )
    assert (tmp_path / "native.npz").read_bytes() == (tmp_path / "python.npz").read_bytes()
    assert size == (tmp_path / "native.npz").stat().st_size
    np.testing.assert_array_equal(written, expected.indptr)


def test_native_extraction_rejects_mismatched_blocks(tmp_path):
    matrix = csr_array(np.array([[0.0, 1.5], [2.0, 0.0], [0.0, 0.0]]))
    engine.write_block(tmp_path / "a.npz", matrix, integer=False)
    write_csr_block(tmp_path / "c.npz", matrix, integer=False, compress=True)
    target, rows, offsets = tmp_path / "out.npz", np.array([1, -1, 0]), matrix.indptr
    engine.extract_rows(tmp_path / "a.npz", offsets, rows, 2, target, integer=False)
    np.testing.assert_array_equal(load_matrix(target).toarray(), [[2.0, 0.0], [0.0, 0.0], [0.0, 1.5]])
    for source, recorded, columns, selected, match in [
        (tmp_path / "c.npz", offsets, 2, rows, "not a stored block"),
        (tmp_path / "a.npz", np.array([0, 1, 2, 3]), 2, rows, "row offsets"),
        (tmp_path / "a.npz", offsets, 2, np.array([3]), "outside"),
        (tmp_path / "a.npz", offsets, 1, rows, "outside"),
    ]:
        with pytest.raises(ValueError, match=match):
            engine.extract_rows(source, recorded, selected, columns, target, integer=False)
    with pytest.raises(OSError):
        engine.extract_rows(tmp_path / "missing.npz", offsets, rows, 2, target, integer=False)
    (tmp_path / "a.npz").write_bytes((tmp_path / "a.npz").read_bytes()[:150])
    with pytest.raises(ValueError, match="truncated"):
        engine.extract_rows(tmp_path / "a.npz", offsets, rows, 2, target, integer=False)


def with_npy_shape(path, shape: str):
    """The block at path with its array's .npy shape replaced in place (the header keeps its length)."""
    data = bytearray(path.read_bytes())
    start = data.index(b"'shape': (")
    end = data.index(b"\n", start)
    text = (b"'shape': " + shape.encode() + b", }").ljust(end - start)
    assert len(text) == end - start
    data[start:end] = text
    path.write_bytes(bytes(data))


def test_native_extraction_rejects_impossible_dimensions(tmp_path):
    """Huge dimensions are rejected, not overflowed: a 19-digit one in the header, and an 18-digit one that
    does not fit in the file."""
    matrix = csr_array(np.array([[1.0, 2.0]]))
    target = tmp_path / "out.npz"
    for shape, offsets in [
        ("(3, 9999999999999999999)", [0, 2]),
        ("(3, 999999999999999999)", [0, 999999999999999999]),
    ]:
        source = tmp_path / "a.npz"
        engine.write_block(source, matrix, integer=False)
        with_npy_shape(source, shape)
        with pytest.raises(ValueError, match=r"shape|truncated"):
            engine.extract_rows(source, np.array(offsets), np.array([0]), 2, target, integer=False)


def _owner(array: np.ndarray):
    """The native result behind a view: arrays chain through .base, memoryviews through .obj."""
    while not hasattr(array, "_owner"):
        array = array.obj if isinstance(array, memoryview) else array.base
    return array._owner


def test_views_outlive_shard_and_result_is_freed_once():
    subjects, times, codes, values = events()
    shard = engine.Shard(subjects, times, codes, values, 2, 1)
    rows = (np.array([1, 2], dtype=np.int64), np.array([1, 5], dtype=np.int64))
    matrices = shard.aggregate(*rows, [-1], 63, IDENTITY, 2, 2)
    owner = weakref.ref(_owner(matrices[0].data))
    sums = matrices[2]
    shard.close()
    del shard, matrices, subjects, times, codes, values
    gc.collect()
    # One owner per result: alive while any view (or array derived from one) survives.
    assert owner() is not None
    np.testing.assert_array_equal(sums.toarray(), [[1.0, 2.5], [-1.0, 0.0]])
    indices = sums.indices
    del sums
    gc.collect()
    assert owner() is not None and indices.tolist() == [0, 1, 0]
    del indices
    gc.collect()
    assert owner() is None


@pytest.mark.parametrize(
    "change",
    [
        lambda a: (a[0][:-1], *a[1:]),  # Unequal lengths.
        lambda a: (a[0].astype(np.int32), *a[1:]),  # Wrong dtype.
        lambda a: (a[0], a[1][::-1].copy(), *a[2:]),  # Unsorted times.
        lambda a: (*a[:2], np.array([0, 1, 1, 9], dtype=np.int32), a[3]),  # Code out of range.
        lambda a: (*a[:3], np.array([1.0, np.inf, 0.0, 0.0])),  # Infinite value.
        lambda a: (np.repeat(a[0], 2)[::2], *a[1:]),  # Non-contiguous view.
    ],
)
def test_invalid_shard_arguments_are_rejected(change):
    with pytest.raises(ValueError):
        engine.Shard(*change(events()), 2, 1)


def test_invalid_aggregation_arguments_are_rejected():
    shard = engine.Shard(*events(), 2, 1)
    rows = (np.array([1], dtype=np.int64), np.array([1], dtype=np.int64))
    for mask in (0, 64):
        with pytest.raises(ValueError, match="mask"):
            shard.aggregate(*rows, [-1], mask, IDENTITY, 2, 1)
    with pytest.raises(ValueError, match="Window"):
        shard.aggregate(*rows, [0], 1, IDENTITY, 2, 1)
    with pytest.raises(ValueError, match="threads"):
        shard.aggregate(*rows, [-1], 1, IDENTITY, 2, 0)
    with pytest.raises(ValueError, match="code_count entries"):
        shard.aggregate(*rows, [-1], 2, IDENTITY[:1], 1, 1)
    with pytest.raises(ValueError, match="no value column"):
        shard.aggregate(*rows, [-1], 2, np.array([0, -1], dtype=np.int32), 1, 4)
    with pytest.raises(ValueError, match="sorted"):
        unsorted = (np.array([2, 1], dtype=np.int64), np.array([0, 0], dtype=np.int64))
        shard.aggregate(*unsorted, [-1], 1, IDENTITY, 2, 1)


def window_sums(times, values, rows, width):
    """Value/sum and value/sum_sqd of one subject's single code, at the given row times."""
    n = len(times)
    shard = engine.Shard(
        np.zeros(n, dtype=np.int64),
        np.asarray(times, dtype=np.int64),
        np.zeros(n, dtype=np.int32),
        np.asarray(values, dtype=np.float64),
        1,
        1,
    )
    rows = (np.zeros(len(rows), dtype=np.int64), np.asarray(rows, dtype=np.int64))
    sums, squares = shard.aggregate(*rows, [width], 0b1100, np.zeros(1, dtype=np.int32), 1, 1)
    return sums.toarray().ravel(), squares.toarray().ravel()


def test_window_sums_are_not_disturbed_by_values_leaving_the_window():
    """A value leaving the window never touches the sum of the values that remain."""
    values = np.array([1e100, 1e50, 1.0])
    sums, squares = window_sums([0, 1, 2], values, [0, 1, 2], 1)
    np.testing.assert_array_equal(sums, values)
    np.testing.assert_array_equal(squares, values * values)
    # Values in disjoint windows are never summed together, so their sum cannot overflow.
    n = 2
    shard = engine.Shard(
        np.zeros(n, dtype=np.int64),
        np.array([0, 10]),
        np.zeros(n, dtype=np.int32),
        np.array([1e308, 1e308]),
        1,
        1,
    )
    (sums,) = shard.aggregate(
        np.zeros(2, dtype=np.int64), np.array([0, 10]), [5], 0b100, np.zeros(1, dtype=np.int32), 1, 1
    )
    np.testing.assert_array_equal(sums.toarray().ravel(), [1e308, 1e308])


@pytest.mark.parametrize("width", [3, 50, -1])
def test_window_sums_are_accurate_for_every_window(width):
    """Each window's sum errs by at most the summation bound of the values in that window."""
    rng = np.random.default_rng(width + 1)
    times = np.sort(rng.integers(0, 400, 300))
    values = rng.choice([-1, 1], 300) * 10.0 ** rng.uniform(-10, 10, 300)
    rows = np.unique(times)
    sums, squares = window_sums(times, values, rows, width)
    for row, time in enumerate(rows):
        window = values[(times <= time) & ((width == -1) | (times > time - width))]
        for got, terms in ((sums[row], window), (squares[row], window * window)):
            bound = 2 * len(terms) * 2.0**-53 * np.abs(terms).sum()
            assert abs(got - math.fsum(terms)) <= bound, (time, got, math.fsum(terms))


def test_output_lock_makes_concurrent_runs_take_turns(tmp_path):
    meds = tmp_path / "data"
    write_meds(meds)
    output = tmp_path / "out"
    run("describe_codes", "describe_codes", {"input_dir": str(meds), "output_dir": str(output)})
    pipeline = Pipeline(meds, output, TabularConfig(min_code_inclusion_count=2))
    with output_lock(output):
        other = threading.Thread(target=pipeline.tabularize)
        other.start()
        other.join(timeout=1.0)
        assert other.is_alive(), "A second run must wait for the lock."
        assert not (output / "tabularize").exists()
    other.join()
    assert pipeline.report["tabularize"]["train/0"]["reused"] is False


def test_cache_task_reads_the_tabularized_config_under_the_lock(tmp_path):
    """A waiting cache_task must use the blocks as the run before it left them, not as they were when it
    started waiting."""
    meds, output, labels = tmp_path / "data", tmp_path / "out", tmp_path / "labels"
    write_meds(meds)
    write_labels(meds, labels)
    run("describe_codes", "describe_codes", {"input_dir": str(meds), "output_dir": str(output)})
    full = TabularConfig(min_code_inclusion_count=2, window_sizes=("full",))
    week = TabularConfig(min_code_inclusion_count=2, window_sizes=("7d",))
    Pipeline(meds, output, full).tabularize()
    with output_lock(output):
        caching = threading.Thread(target=Pipeline(meds, output, full).cache_task, args=("task", labels))
        caching.start()
        caching.join(timeout=1.0)
        assert caching.is_alive(), "cache_task must wait for the lock."
        Pipeline(meds, output, week)._tabularize(reuse=True)  # Another run, which holds the lock.
    caching.join()
    assert sorted(p.name for p in (output / "tabularize/train/0").iterdir()) == ["7d", "none"]
    assert sorted(p.name for p in (output / "task/task_cache/train/0").iterdir()) == ["7d", "none"]


def test_corrupt_manifest_forces_recomputation(tmp_path):
    meds = tmp_path / "data"
    write_meds(meds)
    output = tmp_path / "out"
    run("describe_codes", "describe_codes", {"input_dir": str(meds), "output_dir": str(output)})
    config = TabularConfig(min_code_inclusion_count=2)
    Pipeline(meds, output, config).tabularize()
    manifest = output / ".meds_tab_native/manifests/tabularize/train/0.json"
    manifest.write_text("{not json")
    rerun = Pipeline(meds, output, config)
    rerun.tabularize()
    assert rerun.report["tabularize"]["train/0"]["reused"] is False
    record = json.loads(manifest.read_text())
    record["identity"] = "0" * 64
    manifest.write_text(json.dumps(record))
    again = Pipeline(meds, output, config)
    again.tabularize()
    assert again.report["tabularize"]["train/0"]["reused"] is False
    unchanged = Pipeline(meds, output, config)
    unchanged.tabularize()
    assert unchanged.report["tabularize"]["train/0"]["reused"] is True
