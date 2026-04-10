"""Performance benchmarks for meds-tab core operations.

Run with: uv run pytest benchmark/ --benchmark-enable --benchmark-json=benchmark/output.json
"""

import numpy as np
import polars as pl
import pytest
from scipy.sparse import csr_array

from MEDS_tabular_automl.generate_static_features import summarize_static_measurements
from MEDS_tabular_automl.generate_summarized_reps import aggregate_matrix, sparse_aggregate
from MEDS_tabular_automl.utils import array_to_sparse_matrix

# ============================================================================
# Fixtures for benchmark data
# ============================================================================


@pytest.fixture(params=[1_000, 10_000])
def sparse_matrix_size(request):
    """Parametrize sparse matrix sizes for benchmarks."""
    return request.param


@pytest.fixture
def sparse_matrix(sparse_matrix_size):
    """Create a realistic sparse matrix (patients x features)."""
    rng = np.random.default_rng(42)
    rows, cols = sparse_matrix_size, 500
    density = 0.02
    nnz = int(rows * cols * density)
    r = rng.integers(0, rows, nnz)
    c = rng.integers(0, cols, nnz)
    d = rng.standard_normal(nnz).astype(np.float32)
    return csr_array((d, (r, c)), shape=(rows, cols))


@pytest.fixture
def static_df():
    """Create a realistic static DataFrame for benchmarking."""
    rng = np.random.default_rng(42)
    n_subjects = 1000
    codes = ["BP_SYS", "BP_DIA", "HR", "TEMP", "O2_SAT", "WEIGHT", "HEIGHT", "BMI"]
    n_events = n_subjects * 5  # 5 events per subject on average
    return pl.DataFrame(
        {
            "subject_id": sorted(rng.integers(1, n_subjects + 1, n_events).tolist()),
            "code": rng.choice(codes, n_events).tolist(),
            "numeric_value": rng.standard_normal(n_events).astype(float).tolist(),
        }
    )


@pytest.fixture
def rolling_windows():
    """Create rolling window indices for aggregation benchmarks."""
    n_windows = 500
    window_size = 20
    return pl.DataFrame(
        {
            "min_index": list(range(0, n_windows * window_size, window_size)),
            "max_index": list(range(window_size, (n_windows + 1) * window_size, window_size)),
        }
    )


# ============================================================================
# Sparse matrix operation benchmarks
# ============================================================================


def test_sparse_aggregate_sum(benchmark, sparse_matrix):
    """Benchmark sparse matrix sum aggregation."""
    benchmark(sparse_aggregate, sparse_matrix, "sum")


def test_sparse_aggregate_count(benchmark, sparse_matrix):
    """Benchmark sparse matrix count aggregation (uses csc indptr)."""
    benchmark(sparse_aggregate, sparse_matrix, "count")


def test_sparse_aggregate_min(benchmark, sparse_matrix):
    """Benchmark sparse matrix min aggregation."""
    benchmark(sparse_aggregate, sparse_matrix, "min")


def test_sparse_aggregate_max(benchmark, sparse_matrix):
    """Benchmark sparse matrix max aggregation."""
    benchmark(sparse_aggregate, sparse_matrix, "max")


def test_sparse_aggregate_sum_sqd(benchmark, sparse_matrix):
    """Benchmark sparse matrix sum-of-squares aggregation."""
    benchmark(sparse_aggregate, sparse_matrix, "sum_sqd")


# ============================================================================
# Windowed aggregation benchmark
# ============================================================================


def test_aggregate_matrix_sum(benchmark, sparse_matrix, rolling_windows):
    """Benchmark rolling window aggregation (sum) over sparse matrix."""
    n_rows = min(sparse_matrix.shape[0], rolling_windows.shape[0] * 20)
    matrix = sparse_matrix[:n_rows, :]
    windows = rolling_windows.head(n_rows // 20)
    benchmark(aggregate_matrix, windows, matrix, "sum", matrix.shape[1])


def test_aggregate_matrix_count(benchmark, sparse_matrix, rolling_windows):
    """Benchmark rolling window aggregation (count) over sparse matrix."""
    n_rows = min(sparse_matrix.shape[0], rolling_windows.shape[0] * 20)
    matrix = sparse_matrix[:n_rows, :]
    windows = rolling_windows.head(n_rows // 20)
    benchmark(aggregate_matrix, windows, matrix, "count", matrix.shape[1])


# ============================================================================
# Static feature benchmarks
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
# Sparse matrix construction benchmark
# ============================================================================


def test_array_to_sparse_matrix(benchmark):
    """Benchmark converting numpy arrays to sparse COO format."""
    rng = np.random.default_rng(42)
    n = 50_000
    data = rng.standard_normal(n).astype(np.float32)
    rows = rng.integers(0, 5000, n)
    cols = rng.integers(0, 500, n)
    array = np.array([data, rows, cols])
    benchmark(array_to_sparse_matrix, array, shape=(5000, 500))
