"""Performance benchmarks for meds-tab core operations.

These benchmarks track computational performance of the key operations that define
meds-tab's value proposition: fast tabularization of MEDS-format medical time-series.

Run locally:
    uv run pytest benchmark/ --benchmark-enable -v
    uv run pytest benchmark/ --benchmark-enable --benchmark-json=benchmark/output.json

Benchmarks are organized by pipeline stage:
1. Sparse aggregation primitives (the inner loop of summarization)
2. Windowed aggregation (rolling window over sparse matrices)
3. Time-series feature generation (MEDS DataFrame -> sparse matrix)
4. Static feature generation (MEDS DataFrame -> sparse pivot)
5. Sparse matrix construction
6. End-to-end pipeline (describe_codes -> tabularize -> summarize)
"""

import json

import numpy as np
import polars as pl
import pytest
from scipy.sparse import csr_array

from MEDS_tabular_automl.generate_static_features import summarize_static_measurements
from MEDS_tabular_automl.generate_summarized_reps import aggregate_matrix, sparse_aggregate
from MEDS_tabular_automl.generate_ts_features import get_flat_ts_rep
from MEDS_tabular_automl.utils import array_to_sparse_matrix

# ============================================================================
# Fixtures — synthetic data at multiple scales
# ============================================================================

CODES_TS = ["HR", "TEMP", "BP_SYS", "BP_DIA", "O2_SAT", "ADMISSION//CARDIAC", "DISCHARGE", "WEIGHT"]
CODES_STATIC = ["EYE_COLOR//BROWN", "EYE_COLOR//BLUE", "DOB", "HEIGHT"]

FEATURE_COLUMNS = [
    *[f"{c}/code" for c in CODES_TS],
    *[f"{c}/value" for c in ["HR", "TEMP", "BP_SYS", "BP_DIA", "O2_SAT"]],
    *[f"{c}/static/present" for c in CODES_STATIC],
    "HEIGHT/static/first",
]


@pytest.fixture(params=[1_000, 10_000], ids=["1K_rows", "10K_rows"])
def n_rows(request):
    return request.param


@pytest.fixture
def sparse_matrix(n_rows):
    """Sparse matrix simulating tabularized patient features."""
    rng = np.random.default_rng(42)
    n_cols = 500
    density = 0.02
    nnz = int(n_rows * n_cols * density)
    r = rng.integers(0, n_rows, nnz)
    c = rng.integers(0, n_cols, nnz)
    d = rng.standard_normal(nnz).astype(np.float32)
    return csr_array((d, (r, c)), shape=(n_rows, n_cols))


@pytest.fixture
def rolling_windows_nonoverlapping(n_rows):
    """Non-overlapping rolling windows (each row in exactly one window)."""
    window_size = 20
    n_windows = n_rows // window_size
    return pl.DataFrame(
        {
            "min_index": list(range(0, n_windows * window_size, window_size)),
            "max_index": list(range(window_size, (n_windows + 1) * window_size, window_size)),
        }
    )


@pytest.fixture
def rolling_windows_overlapping(n_rows):
    """Overlapping rolling windows (stride=5, window=20, so rows appear in ~4 windows each).

    This better reflects real usage where e.g. a 30-day window slides daily.
    """
    stride = 5
    window_size = 20
    n_windows = (n_rows - window_size) // stride + 1
    return pl.DataFrame(
        {
            "min_index": [i * stride for i in range(n_windows)],
            "max_index": [i * stride + window_size for i in range(n_windows)],
        }
    )


def _make_meds_df(n_events, seed=42):
    """Create a sorted MEDS-format DataFrame with realistic structure."""
    rng = np.random.default_rng(seed)
    n_subjects = max(n_events // 10, 10)
    subjects = sorted(rng.integers(1, n_subjects + 1, n_events).tolist())
    # Generate timestamps as microseconds from epoch, then cast
    base_us = int(pl.Series(["2020-01-01"]).str.strptime(pl.Datetime("us"), "%Y-%m-%d")[0].timestamp() * 1e6)
    range_us = 365 * 2 * 24 * 3600 * 1_000_000  # 2 years in microseconds
    offsets = np.sort(rng.integers(0, range_us, n_events))
    times = pl.Series("time", [base_us + int(o) for o in offsets], dtype=pl.Datetime("us"))
    codes = rng.choice(CODES_TS, n_events).tolist()
    values = [float(rng.standard_normal()) if rng.random() > 0.3 else None for _ in range(n_events)]
    return pl.DataFrame({"subject_id": subjects, "time": times, "code": codes, "numeric_value": values}).sort(
        "subject_id", "time"
    )


@pytest.fixture(params=[1_000, 10_000], ids=["1K_events", "10K_events"])
def meds_df(request):
    """Synthetic MEDS-format DataFrame at varying sizes."""
    return _make_meds_df(request.param)


def _make_static_df(n_subjects, seed=42):
    """Create static MEDS data (one row per subject per static code)."""
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(1, n_subjects + 1):
        for code in CODES_STATIC:
            val = float(rng.normal(170, 10)) if code == "HEIGHT" else None
            rows.append({"subject_id": sid, "code": code, "numeric_value": val})
    return pl.DataFrame(rows).sort("subject_id")


@pytest.fixture(params=[500, 2_000], ids=["500_subj", "2K_subj"])
def static_df(request):
    """Synthetic static MEDS data at varying sizes."""
    return _make_static_df(request.param)


# ============================================================================
# 1. Sparse aggregation primitives
# ============================================================================


@pytest.mark.parametrize("agg", ["sum", "count", "min", "max", "sum_sqd"])
def test_sparse_aggregate(benchmark, sparse_matrix, agg):
    """Benchmark each sparse aggregation method."""
    benchmark(sparse_aggregate, sparse_matrix, agg)


# ============================================================================
# 2. Windowed aggregation
# ============================================================================


@pytest.mark.parametrize("agg", ["sum", "count", "max"])
def test_aggregate_matrix_nonoverlap(benchmark, sparse_matrix, rolling_windows_nonoverlapping, agg):
    """Benchmark windowed aggregation with non-overlapping windows."""
    benchmark(aggregate_matrix, rolling_windows_nonoverlapping, sparse_matrix, agg, sparse_matrix.shape[1])


@pytest.mark.parametrize("agg", ["sum", "count"])
def test_aggregate_matrix_overlap(benchmark, sparse_matrix, rolling_windows_overlapping, agg):
    """Benchmark windowed aggregation with overlapping windows (closer to real usage)."""
    benchmark(aggregate_matrix, rolling_windows_overlapping, sparse_matrix, agg, sparse_matrix.shape[1])


# ============================================================================
# 3. Time-series feature generation (DataFrame -> sparse)
# ============================================================================


def test_get_flat_ts_rep_code(benchmark, meds_df):
    """Benchmark time-series code feature generation (includes sort check)."""
    benchmark(get_flat_ts_rep, "code/count", FEATURE_COLUMNS, meds_df.lazy())


def test_get_flat_ts_rep_value(benchmark, meds_df):
    """Benchmark time-series value feature generation (includes sort check)."""
    benchmark(get_flat_ts_rep, "value/sum", FEATURE_COLUMNS, meds_df.lazy())


# ============================================================================
# 4. Static feature generation
# ============================================================================


def test_summarize_static_present(benchmark, static_df):
    """Benchmark static present feature summarization."""
    feature_columns = [f"{c}/static/present" for c in static_df["code"].unique().sort().to_list()]
    benchmark(summarize_static_measurements, "static/present", feature_columns, static_df.lazy())


def test_summarize_static_first(benchmark, static_df):
    """Benchmark static first-value feature summarization."""
    feature_columns = [f"{c}/static/first" for c in static_df["code"].unique().sort().to_list()]
    benchmark(summarize_static_measurements, "static/first", feature_columns, static_df.lazy())


# ============================================================================
# 5. Sparse matrix construction
# ============================================================================


@pytest.mark.parametrize("n", [10_000, 100_000], ids=["10K_nnz", "100K_nnz"])
def test_array_to_sparse_matrix(benchmark, n):
    """Benchmark converting numpy arrays to sparse COO format."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(n).astype(np.float32)
    rows = rng.integers(0, 5000, n)
    cols = rng.integers(0, 500, n)
    array = np.array([data, rows, cols])
    benchmark(array_to_sparse_matrix, array, shape=(5000, 500))


# ============================================================================
# 6. End-to-end pipeline benchmark
# ============================================================================


def test_end_to_end_pipeline(benchmark, tmp_path):
    """Benchmark the full describe -> tabularize_static -> tabularize_ts pipeline.

    This is the benchmark most relevant to users — it measures the total time to go from raw MEDS parquet to
    tabularized features. Hydra config composition and filesystem cleanup are excluded from the timed region.
    """
    import shutil

    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    from MEDS_tabular_automl.scripts import describe_codes, tabularize_static, tabularize_time_series

    # Build a dataset large enough that computation dominates I/O
    input_dir = tmp_path / "input"
    rng = np.random.default_rng(42)

    n_subjects = 200
    n_ts = 50_000
    ts_subjects = sorted(rng.integers(1, n_subjects + 1, n_ts).tolist())
    base_us = int(pl.Series(["2020-01-01"]).str.strptime(pl.Datetime("us"), "%Y-%m-%d")[0].timestamp() * 1e6)
    range_us = 365 * 2 * 24 * 3600 * 1_000_000
    offsets = np.sort(rng.integers(0, range_us, n_ts))
    ts_times = pl.Series("time", [base_us + int(o) for o in offsets], dtype=pl.Datetime("us"))
    ts_df = pl.DataFrame(
        {
            "subject_id": ts_subjects,
            "time": ts_times,
            "code": rng.choice(CODES_TS, n_ts).tolist(),
            "numeric_value": [
                float(rng.standard_normal()) if rng.random() > 0.3 else None for _ in range(n_ts)
            ],
        }
    )

    static_rows = []
    for sid in sorted(set(ts_subjects)):
        static_rows.append(
            {"subject_id": sid, "time": None, "code": "EYE_COLOR//BROWN", "numeric_value": None}
        )
        static_rows.append(
            {"subject_id": sid, "time": None, "code": "HEIGHT", "numeric_value": float(rng.normal(170, 10))}
        )
    static_df_data = pl.DataFrame(
        static_rows,
        schema={"subject_id": pl.Int64, "time": pl.Datetime, "code": pl.Utf8, "numeric_value": pl.Float64},
    )

    all_data = pl.concat([ts_df, static_df_data], how="diagonal_relaxed").sort("subject_id", "time")
    fp = input_dir / "train" / "0.parquet"
    fp.parent.mkdir(parents=True, exist_ok=True)
    all_data.write_parquet(fp)
    json.dump({"train/0": sorted(set(ts_subjects))}, (input_dir / ".shards.json").open("w"))

    # Pre-compose Hydra configs outside the timed region
    base_config = {
        "input_dir": str(input_dir.resolve()),
        "output_dir": str((tmp_path / "output").resolve()),
        "do_overwrite": False,
        "seed": 1,
        "tqdm": False,
    }
    tab_config = {
        **base_config,
        "tabularization.min_code_inclusion_count": 1,
        "tabularization.window_sizes": "[30d,full]",
    }

    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../src/MEDS_tabular_automl/configs/"):
        desc_cfg = compose(
            config_name="describe_codes", overrides=[f"{k}={v}" for k, v in base_config.items()]
        )

    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../src/MEDS_tabular_automl/configs/"):
        tab_cfg = compose(config_name="tabularization", overrides=[f"{k}={v}" for k, v in tab_config.items()])

    def run_pipeline():
        output_dir = tmp_path / "output"
        if output_dir.exists():
            shutil.rmtree(output_dir)

        describe_codes.main(desc_cfg)
        tabularize_static.main(tab_cfg)
        tabularize_time_series.main(tab_cfg)

    benchmark.pedantic(run_pipeline, rounds=3, warmup_rounds=1)
