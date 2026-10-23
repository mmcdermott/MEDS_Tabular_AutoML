"""The native backend writes the same tabularized files as the Python backend.

Every stage script runs on the same MEDS data with ``backend=python`` and with ``backend=native``, and
every block and label file is compared: shapes, stored entries (including stored zeros), labels, counts,
extrema and static values exactly; ``value/sum`` and ``value/sum_sqd`` within a floating-point bound for
summation order (the engine uses compensated summation).
"""

import importlib
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf
from scipy.sparse import csr_array

from MEDS_tabular_automl.native import engine
from MEDS_tabular_automl.tabular_dataset import TabularDataset
from MEDS_tabular_automl.utils import load_matrix


def engine_built() -> bool:
    try:
        engine.library()
    except (OSError, RuntimeError):
        return False
    return True


pytestmark = pytest.mark.skipif(not engine_built(), reason="The native engine is not built.")

CONFIGS = "../src/MEDS_tabular_automl/configs/"
SPLITS = {
    "train/0": range(0, 12),
    "train/1": range(12, 24),
    "tuning/0": range(24, 30),
    "held_out/0": range(30, 36),
    "held_out/1": range(36, 38),  # Only static records: a shard without timed events.
}
WINDOWS = "[1d,7d,30d,full]"
UNIT_ROUNDOFF = 2.0**-53
START_US = 1_577_836_800_000_000  # 2020-01-01T00:00:00 in microseconds.
HOUR_US = 3_600_000_000
DAY_US = 24 * HOUR_US


def frame(rows, schema: dict) -> pl.DataFrame:
    """Rows with times in integer microseconds (None: timeless) as a MEDS-typed frame."""
    raw = pl.DataFrame(
        rows, schema={k: (pl.Int64 if v == "time" else v) for k, v in schema.items()}, orient="row"
    )
    return raw.with_columns(pl.col(k).cast(pl.Datetime("us")) for k, v in schema.items() if v == "time")


def write_meds(root: Path, seed: int = 0):
    """Seeded MEDS shards with the cases the stages must agree on.

    Duplicate timestamps, repeated codes in one timestamp, null, NaN and zero values, negative values, rare
    codes (filtered by min_code_inclusion_count), subjects without static records, a subject with only static
    records, and repeated static codes with differing values.
    """
    rng = np.random.default_rng(seed)
    codes = [f"LAB//{i}" for i in range(10)] + ["RARE//0", "RARE//1"]
    for name, subjects in SPLITS.items():
        rows = []
        for subject in subjects:
            if subject % 7 != 3:  # Some subjects have no static records.
                rows.append((subject, None, f"GENDER//{'F' if subject % 2 else 'M'}", None))
                rows.append((subject, None, "AGE", float(rng.integers(20, 90))))
                if subject % 5 == 0:
                    rows.append((subject, None, "AGE", float(rng.integers(20, 90))))
            if subject % 11 == 10 or name == "held_out/1":
                continue  # Only static records.
            offsets = np.sort(rng.integers(0, 400 * 24, size=int(rng.integers(1, 40))))
            for offset in offsets:
                time = START_US + int(offset) * HOUR_US
                for _ in range(int(rng.integers(1, 4))):
                    code = codes[int(rng.integers(0, 10))] if rng.random() > 0.02 else codes[10 + subject % 2]
                    draw = rng.random()
                    if draw < 0.2:
                        value = None
                    elif draw < 0.25:
                        value = float("nan")
                    elif draw < 0.32:
                        value = 0.0
                    else:
                        value = float(np.round(rng.normal(1.0, 3.0), 3))
                    rows.append((subject, time, code, value))
        shard = frame(
            rows, {"subject_id": pl.Int64, "time": "time", "code": pl.String, "numeric_value": pl.Float32}
        )
        path = root / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        shard.sort("subject_id", "time", nulls_last=False, maintain_order=True).write_parquet(path)


def write_labels(meds: Path, root: Path, seed: int = 1, duplicates: bool = False):
    """Labels per shard (<split>/<shard>.parquet): times between events, before any history, of a subject with
    only static records, and unsorted; optionally with repeated (subject, time) keys."""
    rng = np.random.default_rng(seed)
    for path in sorted(meds.glob("*/*.parquet")):
        shard = pl.read_parquet(path).with_columns(pl.col("time").dt.epoch("us"))
        rows = []
        for subject in shard["subject_id"].unique().sort().to_list():
            times = shard.filter(pl.col("subject_id") == subject)["time"].drop_nulls()
            first = times.min() if len(times) else START_US
            last = times.max() if len(times) else first
            rows.append((subject, first - DAY_US, bool(rng.random() < 0.5)))
            for _ in range(int(rng.integers(1, 4))):
                rows.append((subject, first + int((last - first) * rng.random()), bool(rng.random() < 0.5)))
            if duplicates:
                rows.append((*rows[-1][:2], not rows[-1][2]))
        labels = frame(rows, {"subject_id": pl.Int64, "prediction_time": "time", "boolean_value": pl.Boolean})
        target = root / path.relative_to(meds)
        target.parent.mkdir(parents=True, exist_ok=True)
        labels.sample(fraction=1.0, shuffle=True, seed=seed).write_parquet(target)


def run(script: str, config: str, values: dict):
    module = importlib.import_module(f"MEDS_tabular_automl.scripts.{script}")
    with initialize(version_base=None, config_path=CONFIGS):
        cfg = compose(config_name=config, overrides=[f"{k}={v}" for k, v in values.items()])
    module.main(cfg)


def shared(meds: Path, output: Path, backend: str, **extra) -> dict:
    return {
        "input_dir": str(meds),
        "output_dir": str(output),
        "backend": backend,
        "native.threads": 3,
        "tabularization.min_code_inclusion_count": 2,
        "tabularization.window_sizes": WINDOWS,
        **extra,
    }


def tabularize(meds: Path, output: Path, backend: str, **extra):
    run("describe_codes", "describe_codes", {"input_dir": str(meds), "output_dir": str(output)})
    values = shared(meds, output, backend, **extra)
    run("tabularize_static", "tabularization", values)
    run("tabularize_time_series", "tabularization", values)


def cache_task(meds: Path, output: Path, labels: Path, backend: str, task: str = "task", **extra):
    values = {**shared(meds, output, backend, **extra), "task_name": task, "input_label_dir": str(labels)}
    run("cache_task", "task_specific_caching", values)


def blocks(root: Path) -> dict[str, csr_array]:
    found = {}
    for path in sorted(root.rglob("*.npz")):
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        matrix = csr_array(load_matrix(path))
        matrix.sum_duplicates()
        matrix.sort_indices()
        found[path.relative_to(root).as_posix()] = matrix
    return found


def assert_same_blocks(python: Path, native: Path):
    expected, actual = blocks(python), blocks(native)
    assert sorted(actual) == sorted(expected)
    assert expected, "No blocks were written."
    for name, want in expected.items():
        got = actual[name]
        assert got.shape == want.shape, name
        assert np.array_equal(got.indptr, want.indptr) and np.array_equal(got.indices, want.indices), name
        statistic = Path(name).stem
        if name.endswith(("value/sum.npz", "value/sum_sqd.npz")):
            # A float64 sum of n terms errs by <= n*u*sum|x| <= n*u*sqrt(n*sum x^2) (sum: x, sum_sqd:
            # x^2); 4x covers both sides' errors. n and sum x^2 come from the same window's blocks.
            window = name.rsplit("/", 2)[0]
            rows = np.repeat(np.arange(got.shape[0]), np.diff(got.indptr))
            n = expected[f"{window}/value/count.npz"].toarray()[rows, got.indices]
            squares = expected[f"{window}/value/sum_sqd.npz"].toarray()[rows, got.indices]
            spread = np.sqrt(n * squares) if statistic == "sum" else squares
            assert np.all(np.abs(got.data - want.data) <= 4 * n * UNIT_ROUNDOFF * spread), name
        else:
            assert np.array_equal(got.data.astype(np.float64), want.data.astype(np.float64)), name


def assert_same_labels(python: Path, native: Path):
    expected = sorted(p.relative_to(python) for p in python.rglob("*.parquet"))
    assert sorted(p.relative_to(native) for p in native.rglob("*.parquet")) == expected
    for name in expected:
        want, got = pl.read_parquet(python / name), pl.read_parquet(native / name)
        assert got.columns == want.columns, name
        assert got.equals(want.cast(got.schema)), name


def loaded(output: Path, task: str, split: str):
    """X and y of one split as the model launchers load them."""
    with initialize(version_base=None, config_path=CONFIGS):
        tabularization = compose(
            config_name="tabularization",
            overrides=[
                "input_dir=unused",
                f"output_dir={output}",
                "tabularization.min_code_inclusion_count=2",
                f"tabularization.window_sizes={WINDOWS}",
            ],
        ).tabularization
    cfg = OmegaConf.create(
        {
            "tabularization": OmegaConf.to_container(tabularization, resolve=True),
            "path": {
                "input_label_cache_dir": str(output / task / "labels"),
                "input_tabularized_cache_dir": str(output / task / "task_cache"),
            },
            "data_loading_params": {"keep_data_in_memory": True, "binarize_task": True},
        }
    )
    dataset = TabularDataset(cfg, split)
    X, y = dataset.get_data()
    return csr_array(X), np.asarray(y)


@pytest.fixture(scope="module")
def meds(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("meds") / "data"
    write_meds(root)
    return root


def test_event_time_blocks_and_task_cache_match(meds, tmp_path):
    labels = tmp_path / "labels"
    write_labels(meds, labels, duplicates=True)
    for backend in ("python", "native"):
        tabularize(meds, tmp_path / backend, backend)
        cache_task(meds, tmp_path / backend, labels, backend)
    assert_same_blocks(tmp_path / "python/tabularize", tmp_path / "native/tabularize")
    assert_same_blocks(tmp_path / "python/task/task_cache", tmp_path / "native/task/task_cache")
    assert_same_labels(tmp_path / "python/task/labels", tmp_path / "native/task/labels")
    for split in ("train", "tuning", "held_out"):
        (Xp, yp), (Xn, yn) = (loaded(tmp_path / backend, "task", split) for backend in ("python", "native"))
        assert np.array_equal(yp, yn), split
        assert Xp.shape == Xn.shape, split
        assert abs(Xp - Xn).max() <= 1e-9 * max(abs(Xp).max(), 1), split


def test_compressed_blocks_match(meds, tmp_path):
    labels = tmp_path / "labels"
    write_labels(meds, labels)
    for backend in ("python", "native"):
        tabularize(meds, tmp_path / backend, backend, **{"tabularization.do_compress": True})
        cache_task(meds, tmp_path / backend, labels, backend, **{"tabularization.do_compress": True})
    assert_same_blocks(tmp_path / "python/tabularize", tmp_path / "native/tabularize")
    assert_same_blocks(tmp_path / "python/task/task_cache", tmp_path / "native/task/task_cache")


def test_compressed_cache_of_uncompressed_blocks_matches(meds, tmp_path):
    """cache_task compresses as it is told, whatever the tabularized blocks it reads."""
    labels = tmp_path / "labels"
    write_labels(meds, labels)
    for backend in ("python", "native"):
        tabularize(meds, tmp_path / backend, backend)
        cache_task(meds, tmp_path / backend, labels, backend, **{"tabularization.do_compress": True})
    assert_same_blocks(tmp_path / "python/task/task_cache", tmp_path / "native/task/task_cache")
    for path in (tmp_path / "native/task/task_cache").rglob("*.npz"):
        with zipfile.ZipFile(path) as archive:
            assert all(m.compress_type == zipfile.ZIP_DEFLATED for m in archive.infolist()), path


def test_label_time_blocks_match(meds, tmp_path):
    labels = tmp_path / "labels"
    write_labels(meds, labels, duplicates=True)
    for backend in ("python", "native"):
        tabularize(meds, tmp_path / backend, backend, input_label_dir=str(labels))
    assert_same_blocks(tmp_path / "python/tabularize", tmp_path / "native/tabularize")
    shapes = {name: m.shape[0] for name, m in blocks(tmp_path / "native/tabularize").items()}
    for name, rows in shapes.items():
        shard = "/".join(name.split("/")[:2])
        assert rows == len(pl.read_parquet(labels / f"{shard}.parquet")), name


def test_reuse_and_overwrite(meds, tmp_path):
    output = tmp_path / "native"
    tabularize(meds, output, "native")
    block = output / "tabularize/train/0/full/code/count.npz"
    written = block.stat().st_mtime_ns
    run("tabularize_time_series", "tabularization", shared(meds, output, "native"))
    assert block.stat().st_mtime_ns == written, "Matching outputs are reused."
    run("tabularize_time_series", "tabularization", shared(meds, output, "native", do_overwrite=True))
    assert block.stat().st_mtime_ns != written, "do_overwrite recomputes."


def test_cache_task_extracts_the_tabularized_blocks(meds, tmp_path):
    """As in the Python backend, cache_task extracts every tabularized block, whatever its own aggs and window
    sizes say, and neither they, the label column nor do_overwrite re-tabularize."""
    labels = tmp_path / "labels"
    write_labels(meds, labels)
    output = tmp_path / "native"
    tabularize(meds, output, "native")
    block = output / "tabularize/train/0/full/code/count.npz"
    written = block.stat().st_mtime_ns
    values = {
        **shared(meds, output, "native"),
        "task_name": "task",
        "input_label_dir": str(labels),
        "tabularization.aggs": "[static/present]",
        "tabularization.window_sizes": "[1d]",
    }
    run("cache_task", "task_specific_caching", values)
    run("cache_task", "task_specific_caching", {**values, "do_overwrite": True})
    assert block.stat().st_mtime_ns == written
    cached = {name.split("/", 2)[2] for name in blocks(output / "task/task_cache")}
    assert cached == {name.split("/", 2)[2] for name in blocks(output / "tabularize")}


def test_label_times_finer_than_a_microsecond_are_rejected(meds, tmp_path):
    """The engine works in microseconds; truncating would merge distinct labels."""
    output = tmp_path / "native"
    tabularize(meds, output, "native")
    labels = tmp_path / "labels"
    for path in sorted(meds.glob("*/*.parquet")):
        target = labels / path.relative_to(meds)
        target.parent.mkdir(parents=True, exist_ok=True)
        subject = pl.read_parquet(path)["subject_id"][0]
        nanoseconds = [START_US * 1000 + 1, START_US * 1000 + 2]
        pl.DataFrame(
            {
                "subject_id": [subject, subject],
                "prediction_time": pl.Series(nanoseconds).cast(pl.Datetime("ns")),
                "boolean_value": [False, True],
            }
        ).write_parquet(target)
    with pytest.raises(ValueError, match="finer than a microsecond"):
        cache_task(meds, output, labels, "native")


def test_other_workers_do_nothing(meds, tmp_path):
    run("describe_codes", "describe_codes", {"input_dir": str(meds), "output_dir": str(tmp_path)})
    run("tabularize_time_series", "tabularization", shared(meds, tmp_path, "native", worker=1))
    assert not (tmp_path / "tabularize").exists()


def test_backend_selection(monkeypatch):
    from MEDS_tabular_automl.native import stages

    monkeypatch.delenv("MEDS_TAB_BACKEND", raising=False)
    with initialize(version_base=None, config_path=CONFIGS):
        cfg = compose(config_name="tabularization", overrides=["input_dir=in", "output_dir=out"])
    assert cfg.backend == "python" and not stages.use_native(cfg)
    cfg.backend = "native"
    assert stages.use_native(cfg)
    cfg.tabularization.aggs = ["code/count", "value/has_values_count"]
    with pytest.raises(RuntimeError, match="does not implement"):
        stages.use_native(cfg)
    cfg.tabularization.aggs = ["code/count"]
    cfg.tabularization.window_sizes = ["1mo"]
    with pytest.raises(RuntimeError, match="window size"):
        stages.use_native(cfg)
    cfg.tabularization.window_sizes = ["full"]

    def not_built():
        raise FileNotFoundError(engine.NOT_BUILT)

    monkeypatch.setattr(engine, "library_path", not_built)
    engine.library.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="not built"):
            stages.use_native(cfg)
        cfg.backend = "python"
        assert not stages.use_native(cfg)
    finally:
        monkeypatch.undo()
        engine.library.cache_clear()
