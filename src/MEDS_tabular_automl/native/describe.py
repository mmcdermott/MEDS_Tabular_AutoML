"""The native backend of meds-tab-describe: code frequencies of the training subjects' records.

The Python backend scans each shard four times (one group-by on code strings per kind of count) and caches
per-shard counts. Here a shard's subjects are read first, and a shard without a training subject is skipped.
The other columns are read once by PyArrow, with codes kept dictionary-encoded as Parquet stores them, and the
engine counts every record batch from the Arrow buffers (``engine.CodeCounter``, ``csrc/describe.cc``): each
distinct code string of a batch's dictionary is looked up once (or not at all when the dictionary repeats the
previous batch's), and each record costs a few comparisons. Several shards are read at once, each worker
thread with its own counter; the counters are merged by code name at the end. Nothing is cached per shard.

The counts are the Python backend's: for every code, records with a subject and a code, split into timed
records (``<code>/code``), timed records with a non-NaN value (``<code>/value``), timeless records
(``<code>/static/present``) and timeless records with a non-NaN value (``<code>/static/mean``); kinds without
records are left out.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from . import engine

SUFFIXES = ("/code", "/value", "/static/present", "/static/mean")  # The engine's count order.
COLUMNS = ["subject_id", "time", "code", "numeric_value"]
CODES = pa.dictionary(pa.int32(), pa.string())


def _columns(table: pa.Table) -> pa.Table:
    """The shard's columns in the types the engine reads: int64 subjects, string codes dictionary-encoded with
    int32 indices, float32 or float64 values (time: only its nulls are read)."""
    subjects, codes, values = table["subject_id"], table["code"], table["numeric_value"]
    if subjects.type != pa.int64():
        subjects = subjects.cast(pa.int64())
    if codes.type not in (CODES, pa.dictionary(pa.int32(), pa.large_string())):
        # Not a string column read as a dictionary (e.g. stored as another type): encode it here.
        codes = codes.cast(pa.string()).dictionary_encode()
    if values.type not in (pa.float32(), pa.float64()):
        values = values.cast(pa.float64())
    return pa.table({"subject_id": subjects, "time": table["time"], "code": codes, "numeric_value": values})


def _frame(names: list[str], counts: np.ndarray) -> pl.DataFrame:
    """(code, count) rows of every non-zero count (codes by row of counts), sorted by code."""
    rows, kinds = np.nonzero(counts)
    return pl.DataFrame(
        {
            "code": [names[i] + SUFFIXES[k] for i, k in zip(rows.tolist(), kinds.tolist(), strict=True)],
            "count": counts[rows, kinds],
        },
        schema={"code": pl.String, "count": pl.Int64},
    ).sort("code")


def describe_codes(shards: list[Path], train_subjects, shards_in_flight: int = 8) -> pl.DataFrame:
    """The (code, count) frame of meds-tab-describe for the training subjects' records in shards, sorted by
    code."""
    train = np.unique(np.asarray(train_subjects, dtype=np.int64))
    local = threading.local()
    counters: list[engine.CodeCounter] = []
    lock = threading.Lock()

    def count(path: Path):
        counter = getattr(local, "counter", None)
        if counter is None:
            counter = local.counter = engine.CodeCounter(train)
            with lock:
                counters.append(counter)
        # The subjects first: a shard with no training subject (e.g. a tuning or held-out shard) is not read
        # further.
        subjects = pq.read_table(path, columns=["subject_id"])["subject_id"]
        if subjects.type != pa.int64():
            subjects = subjects.cast(pa.int64())
        if not counter.has_train_subject(subjects):
            return
        rest = pq.read_table(path, columns=COLUMNS[1:], read_dictionary=["code"])
        counter.add(_columns(rest.append_column("subject_id", subjects).select(COLUMNS)).to_batches())

    with ThreadPoolExecutor(max(1, shards_in_flight), thread_name_prefix="meds-tab-describe") as pool:
        futures = [pool.submit(count, Path(s)) for s in shards]
        try:
            for future in futures:
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    if not counters:
        return _frame([], np.zeros((0, 4), dtype=np.int64))
    total = counters[0]
    for other in counters[1:]:
        total.merge(other)
    return _frame(*total.result())
