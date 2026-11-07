"""Property-based parity between the native and the Python backends (Hypothesis).

`test_native_backend.py` compares the two backends on one seeded dataset. Here Hypothesis generates many
small MEDS datasets and label sets, aimed at the cases where two implementations of the same features tend to
disagree, runs both tabularization stages with each backend, and requires the same blocks. When a dataset
makes them differ, Hypothesis shrinks it to a minimal one, which can be kept as a regular regression test.

Comparing the backends finds where they disagree, not bugs they share; the Python backend's own tests cover
the feature definitions.

Two Hypothesis profiles: "ci" (the default; few examples, derandomized, so CI is reproducible) and "deep"
(many examples), selected with the HYPOTHESIS_PROFILE environment variable.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, event, given, note, settings
from hypothesis import strategies as st
from meds import SubjectSplitSchema, subject_splits_filepath
from scipy.sparse import csr_array

from MEDS_tabular_automl.native import engine
from MEDS_tabular_automl.native.storage import write_csr_block
from tests.test_native_backend import (
    START_US,
    assert_same_blocks,
    blocks,
    engine_built,
    frame,
    tabularize,
)

pytestmark = pytest.mark.skipif(not engine_built(), reason="The native engine is not built.")

settings.register_profile(
    "ci",
    max_examples=25,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
settings.register_profile(
    "deep",
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))

HOUR_US = 3_600_000_000
STEPS_US = {"1h": HOUR_US, "1d": 24 * HOUR_US}
CODES = ["A", "B", "C", "R"]  # R is rare, so min_code_inclusion_count can filter it out.
SHARDS = ["train/0", "train/1", "held_out/0"]
MEDS_SCHEMA = {"subject_id": pl.Int64, "time": "time", "code": pl.String, "numeric_value": pl.Float32}
LABEL_SCHEMA = {"subject_id": pl.Int64, "prediction_time": "time", "boolean_value": pl.Boolean}

# Values: missing (None, NaN), zero, and small multiples of 0.25, exact in float32 so that sums are exact;
# sometimes a large one, to exercise the backends' summation-order tolerance.
VALUES = st.one_of(
    st.none(),
    st.just(float("nan")),
    st.just(0.0),
    st.integers(-12, 12).map(lambda k: k / 4),
    st.just(1e6 + 0.25),
)


@st.composite
def meds_cases(draw):
    """(MEDS rows per shard, label rows per shard), times in integer microseconds (None: timeless).

    Times are k steps after START_US with k in 0..20, so subjects share timestamps and windows often end
    exactly on an event. Labels fall on events, between them, before the first, after the last, and are
    sometimes duplicated; each shard's label file is shuffled.
    """
    step = STEPS_US[draw(st.sampled_from(sorted(STEPS_US)))]
    n_subjects = draw(st.integers(1, 5))
    rows = {shard: [] for shard in SHARDS}
    labels = {shard: [] for shard in SHARDS}
    # An anchor subject in a training shard, so that every code family has features after describe (which
    # fits on training subjects): A and B with values, and two different static values of S2 (static/mean).
    anchor = 1000
    rows["train/0"] += [(anchor, None, "S1", None), (anchor, None, "S2", 1.5), (anchor, None, "S2", 2.25)]
    rows["train/0"] += [(anchor, START_US + k * step, code, 1.0) for k in (0, 1) for code in ("A", "B")]
    labels["train/0"].append((anchor, START_US + 2 * step, True))
    for subject in range(1, n_subjects + 1):
        shard = draw(st.sampled_from(SHARDS))
        if draw(st.booleans()):
            rows[shard].append((subject, None, "S1", None))
        static_values = draw(st.lists(VALUES, max_size=3))  # Several values of S2: static/mean.
        rows[shard] += [(subject, None, "S2", value) for value in static_values]
        if len([v for v in static_values if v is not None and not np.isnan(v)]) > 1:
            event("subject with several static values")
        times = []
        for _ in range(draw(st.integers(0, 12))):
            time = START_US + draw(st.integers(0, 20)) * step
            if time in times:
                event("measurements sharing a timestamp")
            times.append(time)
            value = draw(VALUES)
            if value is not None and np.isnan(value):
                event("NaN value")
            rows[shard].append((subject, time, draw(st.sampled_from(CODES)), value))
        for _ in range(draw(st.integers(0, 4))):
            kind = draw(st.sampled_from(["at_event", "between", "before", "after"]))
            if kind == "at_event" and times:
                time = draw(st.sampled_from(times))
            elif kind == "between" and times:
                time = draw(st.sampled_from(times)) + step // 2
            elif kind == "before":
                time = START_US - 3 * step
            else:
                time = START_US + 25 * step
            labels[shard].append((subject, time, draw(st.booleans())))
            event(f"label {kind}" if kind in ("before", "after") or times else "label before")
            if draw(st.integers(0, 9)) == 0:  # A duplicated label.
                labels[shard].append(labels[shard][-1])
                event("duplicated label")
    for shard in SHARDS:
        order = draw(st.permutations(range(len(labels[shard]))))
        labels[shard] = [labels[shard][i] for i in order]
    return rows, labels


@st.composite
def tabularization_configs(draw):
    windows = draw(st.lists(st.sampled_from(["1h", "1d", "7d", "full"]), min_size=1, max_size=4, unique=True))
    return {
        "tabularization.window_sizes": f"[{','.join(windows)}]",
        "tabularization.min_code_inclusion_count": draw(st.sampled_from([1, 2])),
        "tabularization.do_compress": draw(st.booleans()),
    }


def write_case(root: Path, rows: dict, labels: dict) -> tuple[Path, Path]:
    """Writes the shards that have data, a label file for each of them, and the subject splits."""
    data, label_dir = root / "meds" / "data", root / "labels"
    splits = []
    for shard, shard_rows in rows.items():
        if not shard_rows:
            continue
        path = data / f"{shard}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame(shard_rows, MEDS_SCHEMA).sort("subject_id", "time", nulls_last=False).write_parquet(path)
        label_path = label_dir / f"{shard}.parquet"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        frame(labels[shard], LABEL_SCHEMA).write_parquet(label_path)
        split = shard.split("/")[0]
        splits += [(subject, split) for subject in sorted({r[0] for r in shard_rows})]
    splits_path = root / "meds" / subject_splits_filepath
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    schema = {SubjectSplitSchema.subject_id_name: pl.Int64, SubjectSplitSchema.split_name: pl.String}
    frame(splits, schema).write_parquet(splits_path)
    return data, label_dir


# This test runs both backends itself, whatever MEDS_TAB_BACKEND says. CI runs the suite once per backend
# (MEDS_TAB_BACKEND=python or native), so it runs only where the backend is native, not twice per Python
# version. Locally, with MEDS_TAB_BACKEND unset, it always runs.
@pytest.mark.skipif(
    os.environ.get("MEDS_TAB_BACKEND") == "python",
    reason="Compares both backends; runs in the MEDS_TAB_BACKEND=native jobs.",
)
@given(case=meds_cases(), config=tabularization_configs())
def test_native_matches_python_at_label_times(case, config):
    """Same files, shapes, stored entries and values (sums within the summation-order bound), and one row per
    label in every block."""
    rows, labels = case
    note(f"config={config}")
    with tempfile.TemporaryDirectory() as directory:  # Per example: pytest's tmp_path is shared by examples.
        root = Path(directory)
        data, label_dir = write_case(root, rows, labels)
        for backend in ("python", "native"):
            tabularize(data, root / backend, label_dir, backend, **config)
        assert_same_blocks(root / "python/tabularize", root / "native/tabularize")
        for name, matrix in blocks(root / "native/tabularize").items():
            shard = "/".join(name.split("/")[:2])
            assert matrix.shape[0] == len(labels[shard]), name


@st.composite
def sparse_blocks(draw):
    """A CSR matrix with sorted indices of either index width, and whether its values are integers.

    Values cover the stored-type choice of both writers: integers at type boundaries, negative values,
    stored zeros (also -0.0), floats exact in float32 or not, and very large floats.
    """
    n_rows, n_columns = draw(st.integers(0, 40)), draw(st.integers(0, 40))
    integer = draw(st.booleans())
    if n_rows and n_columns:
        positions = draw(st.sets(st.integers(0, n_rows * n_columns - 1), max_size=200))
    else:
        positions = set()
    keys = np.array(sorted(positions), dtype=np.int64)
    if integer:
        boundaries = [0, 1, -1, 127, 128, -128, -129, 255, 256, 32767, 32768, 65535, 65536, 2**31 - 1, 2**31]
        value = st.one_of(st.sampled_from(boundaries), st.integers(-(2**40), 2**40))
    else:
        value = st.one_of(
            st.just(0.0),
            st.just(-0.0),
            st.integers(-64, 64).map(lambda k: k / 8),  # Exact in float32.
            st.floats(allow_nan=False, allow_infinity=False, width=64),
            st.floats(allow_nan=False, allow_infinity=False, width=32),
        )
    data = np.array([draw(value) for _ in keys], dtype=np.int64 if integer else np.float64)
    matrix = csr_array(
        (data, (keys // max(n_columns, 1), keys % max(n_columns, 1))), shape=(n_rows, n_columns)
    )
    matrix.sort_indices()
    index = draw(st.sampled_from([np.int32, np.int64]))
    matrix = csr_array((matrix.data, matrix.indices.astype(index), matrix.indptr.astype(index)), matrix.shape)
    return matrix, integer


@given(block=sparse_blocks())
def test_native_writer_matches_python_writer_bytes(block):
    """Byte for byte, for any block: values, index width, and the stored type each writer chooses."""
    matrix, integer = block
    note(f"shape={matrix.shape} nnz={matrix.nnz} integer={integer} dtype={matrix.data.dtype}")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_csr_block(root / "python.npz", matrix, integer=integer, compress=False)
        size = engine.write_block(root / "native.npz", matrix, integer=integer)
        written = (root / "native.npz").read_bytes()
        assert written == (root / "python.npz").read_bytes()
        assert size == len(written)
