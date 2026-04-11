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
5. End-to-end pipeline (describe_codes -> tabularize -> summarize)
"""

import json

import numpy as np
import polars as pl
import pytest
from scipy.sparse import csr_array

from MEDS_tabular_automl.generate_static_features import (
    summarize_static_measurements,
)
from MEDS_tabular_automl.generate_summarized_reps import (
    aggregate_matrix,
    sparse_aggregate,
)
from MEDS_tabular_automl.generate_ts_features import get_flat_ts_rep
from MEDS_tabular_automl.utils import (
    array_to_sparse_matrix,
)

# ============================================================================
# Fixtures — synthetic data at multiple scales
# ============================================================================


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
def rolling_windows(n_rows):
    """Rolling window indices matching the sparse matrix size."""
    window_size = 20
    n_windows = n_rows // window_size
    return pl.DataFrame(
        {
            "min_index": list(range(0, n_windows * window_size, window_size)),
            "max_index": list(range(window_size, (n_windows + 1) * window_size, window_size)),
        }
    )


CODES = [
    "ADMISSION//CARDIAC",
    "ADMISSION//ORTHOPEDIC",
    "DISCHARGE",
    "HR",
    "TEMP",
    "BP_SYS",
    "BP_DIA",
    "O2_SAT",
    "WEIGHT",
    "HEIGHT",
    "EYE_COLOR//BROWN",
    "EYE_COLOR//BLUE",
    "DOB",
]

FEATURE_COLUMNS = [
    "ADMISSION//CARDIAC/code",
    "ADMISSION//ORTHOPEDIC/code",
    "DISCHARGE/code",
    "HR/code",
    "TEMP/code",
    "BP_SYS/code",
    "BP_DIA/code",
    "O2_SAT/code",
    "WEIGHT/code",
    "HEIGHT/code",
    "HR/value",
    "TEMP/value",
    "BP_SYS/value",
    "BP_DIA/value",
    "O2_SAT/value",
    "EYE_COLOR//BROWN/static/present",
    "EYE_COLOR//BLUE/static/present",
    "DOB/static/present",
    "HEIGHT/static/first",
]


@pytest.fixture(params=[1_000, 5_000], ids=["1K_events", "5K_events"])
def meds_df(request):
    """Synthetic MEDS-format DataFrame at varying sizes."""
    n_events = request.param
    rng = np.random.default_rng(42)
    n_subjects = n_events // 10

    subjects = sorted(rng.integers(1, n_subjects + 1, n_events).tolist())
    times = pl.Series(
        [f"2021-01-{(i % 28) + 1:02d}T{(i % 24):02d}:00:00.000000" for i in range(n_events)]
    ).str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.f")
    codes = rng.choice(
        [c for c in CODES if c not in ("EYE_COLOR//BROWN", "EYE_COLOR//BLUE", "DOB")], n_events
    ).tolist()
    values = [float(rng.standard_normal()) if rng.random() > 0.3 else None for _ in range(n_events)]

    return pl.DataFrame(
        {
            "subject_id": subjects,
            "time": times,
            "code": codes,
            "numeric_value": values,
        }
    ).sort("subject_id", "time")


@pytest.fixture
def static_df():
    """Synthetic static MEDS data (one row per subject)."""
    rng = np.random.default_rng(42)
    n_subjects = 500
    static_codes = ["EYE_COLOR//BROWN", "EYE_COLOR//BLUE", "DOB", "HEIGHT"]
    rows = []
    for sid in range(1, n_subjects + 1):
        for code in static_codes:
            val = float(rng.standard_normal()) if code == "HEIGHT" else None
            rows.append({"subject_id": sid, "code": code, "numeric_value": val})
    return pl.DataFrame(rows).sort("subject_id")


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
def test_aggregate_matrix(benchmark, sparse_matrix, rolling_windows, agg):
    """Benchmark rolling window aggregation over sparse matrix."""
    n_rows = min(sparse_matrix.shape[0], rolling_windows.shape[0] * 20)
    matrix = sparse_matrix[:n_rows, :]
    windows = rolling_windows.head(n_rows // 20)
    benchmark(aggregate_matrix, windows, matrix, agg, matrix.shape[1])


# ============================================================================
# 3. Time-series feature generation (DataFrame -> sparse)
# ============================================================================


def test_get_flat_ts_rep_code(benchmark, meds_df):
    """Benchmark time-series code feature generation."""
    benchmark(get_flat_ts_rep, "code/count", FEATURE_COLUMNS, meds_df.lazy())


def test_get_flat_ts_rep_value(benchmark, meds_df):
    """Benchmark time-series value feature generation."""
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


@pytest.mark.parametrize("n", [10_000, 50_000], ids=["10K_nnz", "50K_nnz"])
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
    tabularized features.
    """
    import shutil

    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    from MEDS_tabular_automl.scripts import describe_codes, tabularize_static, tabularize_time_series

    # Set up test data once outside the benchmark loop
    input_dir = tmp_path / "input"
    rng = np.random.default_rng(42)

    # Time-series events
    ts_codes = ["HR", "TEMP", "BP_SYS", "ADMISSION//CARDIAC", "DISCHARGE"]
    n_ts = 400
    ts_subjects = sorted(rng.integers(1, 51, n_ts).tolist())
    ts_times = pl.Series(
        [f"2021-01-{(i % 28) + 1:02d}T{(i % 24):02d}:00:00.000000" for i in range(n_ts)]
    ).str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.f")
    ts_code_vals = rng.choice(ts_codes, n_ts).tolist()
    ts_values = [float(rng.standard_normal()) if rng.random() > 0.3 else None for _ in range(n_ts)]
    ts_df = pl.DataFrame(
        {
            "subject_id": ts_subjects,
            "time": ts_times,
            "code": ts_code_vals,
            "numeric_value": ts_values,
        }
    )

    # Static events (no time, includes HEIGHT with value for static/first)
    static_rows = []
    for sid in sorted(set(ts_subjects)):
        static_rows.append(
            {"subject_id": sid, "time": None, "code": "EYE_COLOR//BROWN", "numeric_value": None}
        )
        static_rows.append(
            {"subject_id": sid, "time": None, "code": "HEIGHT", "numeric_value": float(rng.normal(170, 10))}
        )
    static_df = pl.DataFrame(
        static_rows,
        schema={
            "subject_id": pl.Int64,
            "time": pl.Datetime,
            "code": pl.Utf8,
            "numeric_value": pl.Float64,
        },
    )

    all_data = pl.concat([ts_df, static_df], how="diagonal_relaxed").sort("subject_id", "time")
    fp = input_dir / "train" / "0.parquet"
    fp.parent.mkdir(parents=True, exist_ok=True)
    all_data.write_parquet(fp)
    json.dump({"train/0": list(set(ts_subjects))}, (input_dir / ".shards.json").open("w"))

    def run_pipeline():
        output_dir = tmp_path / "output"
        if output_dir.exists():
            shutil.rmtree(output_dir)

        base_config = {
            "input_dir": str(input_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
            "do_overwrite": False,
            "seed": 1,
            "tqdm": False,
        }
        tab_config = {
            **base_config,
            "tabularization.min_code_inclusion_count": 1,
            "tabularization.window_sizes": "[full]",
        }

        GlobalHydra.instance().clear()
        with initialize(version_base=None, config_path="../src/MEDS_tabular_automl/configs/"):
            cfg = compose(
                config_name="describe_codes", overrides=[f"{k}={v}" for k, v in base_config.items()]
            )
        describe_codes.main(cfg)

        GlobalHydra.instance().clear()
        with initialize(version_base=None, config_path="../src/MEDS_tabular_automl/configs/"):
            cfg = compose(config_name="tabularization", overrides=[f"{k}={v}" for k, v in tab_config.items()])
        tabularize_static.main(cfg)
        tabularize_time_series.main(cfg)

    benchmark.pedantic(run_pipeline, rounds=3, warmup_rounds=1)
