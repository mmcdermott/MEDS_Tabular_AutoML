"""The native backend writes the same tabularized files as the Python backend.

Every stage script runs on the same MEDS data with ``backend=python`` and with ``backend=native``, and
every block is compared: shapes, stored entries (including stored zeros), labels, counts,
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
from meds import SubjectSplitSchema, subject_splits_filepath
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
# Every supported aggregation, so both backends are compared on all of them (not only the defaults).
AGGS = (
    "[static/present,static/mean,code/count,code/recency,value/count,value/sum,value/sum_sqd,value/min,"
    "value/max]"
)
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
    # meds-tab-describe fits the code metadata on the training subjects of the MEDS root's subject splits.
    splits = [(s, name.split("/")[0]) for name, subjects in SPLITS.items() for s in subjects]
    splits_fp = root.parent / subject_splits_filepath
    splits_fp.parent.mkdir(parents=True, exist_ok=True)
    schema = {SubjectSplitSchema.subject_id_name: pl.Int64, SubjectSplitSchema.split_name: pl.String}
    frame(splits, schema).write_parquet(splits_fp)


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


def override(key: str, value) -> str:
    """A Hydra override; paths are quoted with forward slashes, as Windows paths can hold characters Hydra's
    override grammar rejects unquoted (e.g. the ``~`` of 8.3 short names such as ``RUNNER~1``).

    Examples:
        >>> override("input_dir", r"C:\\Users\\RUNNER~1\\meds")
        "input_dir='C:/Users/RUNNER~1/meds'"
        >>> override("tabularization.window_sizes", "[1d,full]")
        'tabularization.window_sizes=[1d,full]'
    """
    if isinstance(value, str) and ("/" in value or "\\" in value) and not value.startswith("["):
        value = "'" + value.replace("\\", "/") + "'"
    return f"{key}={value}"


def run(script: str, config: str, values: dict):
    module = importlib.import_module(f"MEDS_tabular_automl.scripts.{script}")
    with initialize(version_base=None, config_path=CONFIGS):
        cfg = compose(config_name=config, overrides=[override(k, v) for k, v in values.items()])
    module.main(cfg)


def shared(meds: Path, output: Path, backend: str, **extra) -> dict:
    return {
        "input_dir": str(meds),
        "output_dir": str(output),
        "backend": backend,
        "native.threads": 3,
        "tabularization.min_code_inclusion_count": 2,
        "tabularization.window_sizes": WINDOWS,
        "tabularization.aggs": AGGS,
        **extra,
    }


def tabularize(meds: Path, output: Path, labels: Path, backend: str, **extra):
    """Task-specific tabularization of meds at the label times in labels, with the given backend."""
    run("describe_codes", "describe_codes", {"input_dir": str(meds), "output_dir": str(output)})
    values = shared(meds, output, backend, input_label_dir=str(labels), **extra)
    run("tabularize_static", "tabularization", values)
    run("tabularize_time_series", "tabularization", values)


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
        elif statistic == "recency":
            # 1 / (1 + days): the backends compute the days from different time origins.
            assert np.allclose(got.data, want.data, rtol=1e-12, atol=0), name
        else:
            assert np.array_equal(got.data.astype(np.float64), want.data.astype(np.float64)), name


def loaded(
    output: Path, labels: Path, split: str, threshold: int = 2, cache_dir: Path | None = None, raw=False
):
    """X and y of one split as the model launchers load them (through the matrix cache when cache_dir is
    given; raw: X exactly as get_data returns it)."""
    with initialize(version_base=None, config_path=CONFIGS):
        tabularization = compose(
            config_name="tabularization",
            overrides=[
                "input_dir=unused",
                override("output_dir", str(output)),
                override("input_label_dir", str(labels)),
                f"tabularization.min_code_inclusion_count={threshold}",
                f"tabularization.window_sizes={WINDOWS}",
                f"tabularization.aggs={AGGS}",
            ],
        ).tabularization
    cfg = OmegaConf.create(
        {
            "tabularization": OmegaConf.to_container(tabularization, resolve=True),
            "path": {
                "input_label_dir": str(labels),
                "input_tabularized_dir": str(output / "tabularize"),
                "cache_dir": str(cache_dir),
            },
            "data_loading_params": {
                "keep_data_in_memory": True,
                "binarize_task": True,
                "cache_matrices": cache_dir is not None,
            },
        }
    )
    dataset = TabularDataset(cfg, split)
    X, y = dataset.get_data()
    return (X if raw else csr_array(X)), np.asarray(y)


@pytest.fixture(scope="module")
def meds(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("meds") / "data"
    write_meds(root)
    return root


def test_blocks_and_loaded_data_match(meds, tmp_path):
    """Every block, with one row per label (duplicates included, in label-file order), and the X and y the
    model launchers load."""
    labels = tmp_path / "labels"
    write_labels(meds, labels, duplicates=True)
    for backend in ("python", "native"):
        tabularize(meds, tmp_path / backend, labels, backend)
    assert_same_blocks(tmp_path / "python/tabularize", tmp_path / "native/tabularize")
    for name, matrix in blocks(tmp_path / "native/tabularize").items():
        shard = "/".join(name.split("/")[:2])
        assert matrix.shape[0] == len(pl.read_parquet(labels / f"{shard}.parquet")), name
    for split in ("train", "tuning", "held_out"):
        (Xp, yp), (Xn, yn) = (loaded(tmp_path / backend, labels, split) for backend in ("python", "native"))
        assert np.array_equal(yp, yn), split
        assert Xp.shape == Xn.shape, split
        assert abs(Xp - Xn).max() <= 1e-9 * max(abs(Xp).max(), 1), split


def test_matrix_cache_gives_the_same_data(meds, tmp_path, monkeypatch):
    """With data_loading_params.cache_matrices, every code threshold loads the same X and y as without it,
    from one cache per split built on first use; changed inputs build a new cache."""
    labels = tmp_path / "labels"
    write_labels(meds, labels, duplicates=True)
    tabularize(meds, tmp_path / "tab", labels, "python")
    cache = tmp_path / "cache"
    for threshold in (2, 3, 4):
        for split in ("train", "tuning", "held_out"):
            X, y = loaded(tmp_path / "tab", labels, split, threshold, raw=True)
            Xc, yc = loaded(tmp_path / "tab", labels, split, threshold, cache, raw=True)
            assert Xc.format == X.format == "csc" and Xc.dtype == X.dtype and Xc.shape == X.shape, split
            assert np.array_equal(Xc.indptr, X.indptr) and np.array_equal(Xc.indices, X.indices), split
            assert np.array_equal(Xc.data, X.data) and np.array_equal(yc, y), split
    built = sorted(p.name for p in (cache / "matrices").iterdir())
    assert len(built) == 3 and not any(p.startswith(".") for p in built)

    # Reused, not rebuilt.
    def no_rebuild(*args, **kwargs):
        raise AssertionError("the matrix cache was rebuilt")

    monkeypatch.setattr(TabularDataset, "_build_matrix_cache", no_rebuild)
    loaded(tmp_path / "tab", labels, "train", 3, cache)
    monkeypatch.undo()

    # A changed block file gets a new cache.
    block = next((tmp_path / "tab" / "tabularize" / "train").rglob("*.npz"))
    block.touch()
    loaded(tmp_path / "tab", labels, "train", 3, cache)
    assert len(list((cache / "matrices").iterdir())) == 4


def test_static_mean_is_the_mean_of_static_values(meds, tmp_path):
    """Some subjects have two different static AGE values: static/mean is their mean, in both backends."""
    labels = tmp_path / "labels"
    write_labels(meds, labels)
    for backend in ("python", "native"):
        tabularize(meds, tmp_path / backend, labels, backend)
    assert_same_blocks(tmp_path / "python/tabularize", tmp_path / "native/tabularize")
    data = pl.concat([pl.read_parquet(p) for p in sorted(meds.glob("*/*.parquet"))])
    ages = data.filter(pl.col("time").is_null() & (pl.col("code") == "AGE"))
    repeated = ages.group_by("subject_id").agg(pl.len().alias("n"), pl.col("numeric_value").n_unique())
    assert repeated.filter((pl.col("n") > 1) & (pl.col("numeric_value") > 1)).height > 0, "No case tested."


def test_compressed_blocks_match(meds, tmp_path):
    labels = tmp_path / "labels"
    write_labels(meds, labels)
    for backend in ("python", "native"):
        tabularize(meds, tmp_path / backend, labels, backend, **{"tabularization.do_compress": True})
    assert_same_blocks(tmp_path / "python/tabularize", tmp_path / "native/tabularize")
    for path in (tmp_path / "native/tabularize").rglob("*.npz"):
        with zipfile.ZipFile(path) as archive:
            assert all(m.compress_type == zipfile.ZIP_DEFLATED for m in archive.infolist()), path


def test_reuse_and_overwrite(meds, tmp_path):
    labels = tmp_path / "labels"
    write_labels(meds, labels)
    output = tmp_path / "native"
    tabularize(meds, output, labels, "native")
    block = output / "tabularize/train/0/full/code/count.npz"
    written = block.stat().st_mtime_ns
    values = shared(meds, output, "native", input_label_dir=str(labels))
    run("tabularize_time_series", "tabularization", values)
    assert block.stat().st_mtime_ns == written, "Matching outputs are reused."
    run("tabularize_time_series", "tabularization", {**values, "do_overwrite": True})
    assert block.stat().st_mtime_ns != written, "do_overwrite recomputes."


def test_label_times_finer_than_a_microsecond_are_rejected(meds, tmp_path):
    """The engine works in microseconds; truncating would merge distinct labels."""
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
        tabularize(meds, tmp_path / "native", labels, "native")


def test_other_workers_do_nothing(meds, tmp_path):
    labels = tmp_path / "labels"
    write_labels(meds, labels)
    run("describe_codes", "describe_codes", {"input_dir": str(meds), "output_dir": str(tmp_path)})
    values = shared(meds, tmp_path, "native", worker=1, input_label_dir=str(labels))
    run("tabularize_time_series", "tabularization", values)
    assert not (tmp_path / "tabularize").exists()


def test_backend_selection(monkeypatch):
    from MEDS_tabular_automl.native import stages

    monkeypatch.delenv("MEDS_TAB_BACKEND", raising=False)
    with initialize(version_base=None, config_path=CONFIGS):
        cfg = compose(config_name="tabularization", overrides=["input_dir=in", "output_dir=out"])
    # auto (the default) runs the native backend when it can, and falls back to Python where native raises.
    assert cfg.backend == "auto" and stages.use_native(cfg)
    cfg.backend = "native"
    assert stages.use_native(cfg)
    cfg.backend = "python"
    assert not stages.use_native(cfg)

    def cases():
        cfg.tabularization.aggs = ["code/count", "value/has_values_count"]
        yield "does not implement"
        cfg.tabularization.aggs = ["code/count"]
        cfg.tabularization.window_sizes = ["1mo"]
        yield "window size"
        cfg.tabularization.window_sizes = ["full"]
        default_out = cfg.output_tabularized_dir
        cfg.output_tabularized_dir = "elsewhere"
        yield "default location"
        cfg.output_tabularized_dir = default_out

    for match in cases():
        cfg.backend = "native"
        with pytest.raises(RuntimeError, match=match):
            stages.use_native(cfg)
        cfg.backend = "auto"
        assert not stages.use_native(cfg)
    cfg.backend = "auto"
    assert stages.use_native(cfg)

    def not_built():
        raise FileNotFoundError(engine.NOT_BUILT)

    monkeypatch.setattr(engine, "library_path", not_built)
    engine.library.cache_clear()
    try:
        cfg.backend = "native"
        with pytest.raises(RuntimeError, match="not built"):
            stages.use_native(cfg)
        cfg.backend = "auto"
        assert not stages.use_native(cfg)
        cfg.backend = "python"
        assert not stages.use_native(cfg)
    finally:
        monkeypatch.undo()
        engine.library.cache_clear()
