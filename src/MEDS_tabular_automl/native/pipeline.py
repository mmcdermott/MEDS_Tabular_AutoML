"""MEDS-Tab's tabularization and task caching on the native engine.

Python reads each Parquet shard once, encodes its code strings to integer IDs, and hands the engine
integer arrays. The engine computes every window and aggregation of the shard from that one copy, and
Python writes MEDS-Tab's block files on a pool of writer threads while the next shard is computed. Each
shard's outputs are staged, renamed into place and recorded in a manifest, so a later run can reuse
exactly the outputs whose inputs, configuration and engine are unchanged.

``docs/native_backend.md`` lists every large allocation and copy.
"""

from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.sparse import csr_array

from . import engine
from .contract import DYNAMIC_AGGS, SUFFIXES, TabularConfig, window_microseconds
from .storage import (
    Publisher,
    atomic_write_json,
    block_path,
    digest,
    file_fingerprint,
    manifest_matches,
    manifest_path,
    output_lock,
    read_manifest,
    write_csr_block,
)

STATISTIC_BITS = {agg: 1 << bit for bit, agg in enumerate(DYNAMIC_AGGS)}  # The engine's statistic bits.
INTEGER_AGGS = ("code/count", "value/count", "static/present")


def shard_names(input_dir: Path) -> list[str]:
    """'<split>/<shard>' for every <input_dir>/<split>/<shard>.parquet, sorted."""
    deeper = list(Path(input_dir).glob("*/*/**/*.parquet"))
    if deeper:
        raise ValueError(f"Shards must be exactly <split>/<shard>.parquet; found {deeper[0]}.")
    names = sorted(
        p.relative_to(input_dir).with_suffix("").as_posix() for p in Path(input_dir).glob("*/*.parquet")
    )
    if not names:
        raise FileNotFoundError(f"No <split>/<shard>.parquet files under {input_dir}.")
    return names


def select_features(counts: dict[str, int], config: TabularConfig) -> frozenset[str]:
    """The features the Python backend's filter_to_codes keeps, in its order of filters; ties at
    max_included_codes are broken by name."""
    items = dict(counts)
    if config.allowed_codes is not None:
        allowed = set(config.allowed_codes)
        items = {k: v for k, v in items.items() if k in allowed}
    if config.min_code_inclusion_frequency is not None:
        total = sum(items.values())
        items = {k: v for k, v in items.items() if v / total >= config.min_code_inclusion_frequency}
    if config.min_code_inclusion_count is not None:
        items = {k: v for k, v in items.items() if v >= config.min_code_inclusion_count}
    if config.max_included_codes is not None:
        ranked = sorted(items.items(), key=lambda item: (-item[1], item[0]))
        items = dict(ranked[: config.max_included_codes])
    if not items:
        raise ValueError("Code filtering criteria leaves only 0 codes.")
    return frozenset(items)


@dataclass(frozen=True)
class Vocabulary:
    """Columns of every block family, from the features meds-tab-describe found.

    Each family's columns are its described features in sorted order, as in the Python backend; a column that
    is not selected stays in the block but is never filled.
    """

    code_names: tuple[str, ...]  # Raw code of each code/count column; code IDs index this.
    code_keep: np.ndarray  # bool per code ID: '<code>/code' is selected.
    value_columns: np.ndarray  # int32 per code ID: its value/* column if '<code>/value' is selected, else -1.
    value_count: int  # Columns of value/* blocks.
    present_names: tuple[str, ...]
    present_keep: np.ndarray
    mean_names: tuple[str, ...]
    mean_keep: np.ndarray

    @classmethod
    def build(cls, counts: dict[str, int], selected: frozenset[str]) -> Vocabulary:
        families = {suffix: [] for suffix in dict.fromkeys(SUFFIXES.values())}
        for feature in sorted(counts):
            for suffix, members in families.items():
                if feature.endswith(suffix):
                    members.append(feature)
                    break

        def family(agg: str) -> tuple[tuple[str, ...], np.ndarray]:
            features = families[SUFFIXES[agg]]
            names = tuple(f[: -len(SUFFIXES[agg])] for f in features)
            return names, np.array([f in selected for f in features], dtype=bool)

        code_names, code_keep = family("code/count")
        value_names, value_keep = family("value/sum")
        value_index = {name: i for i, name in enumerate(value_names) if value_keep[i]}
        value_columns = np.array([value_index.get(name, -1) for name in code_names], dtype=np.int32)
        present_names, present_keep = family("static/present")
        mean_names, mean_keep = family("static/mean")
        return cls(
            code_names,
            code_keep,
            value_columns,
            len(value_names),
            present_names,
            present_keep,
            mean_names,
            mean_keep,
        )

    def block_width(self, agg: str) -> int:
        if agg == "code/count":
            return len(self.code_names)
        if agg in DYNAMIC_AGGS:
            return self.value_count
        return len(self.present_names if agg == "static/present" else self.mean_names)


@dataclass
class EncodedShard:
    """One shard as integer arrays.

    Timed arrays hold the selected timed events, sorted by (subject, time); static arrays hold the selected
    timeless records in file order.
    """

    subjects: np.ndarray
    times: np.ndarray
    codes: np.ndarray
    values: np.ndarray  # NaN when missing, or when the code's values are not a selected feature.
    static_subjects: np.ndarray
    static_present: np.ndarray  # static/present column.
    static_mean: np.ndarray  # static/mean column, or -1 when not a selected feature.
    static_values: np.ndarray
    shard_subjects: np.ndarray  # Every subject with any record, sorted.


def _numpy(series: pl.Series, dtype) -> np.ndarray:
    """A column as a contiguous array of dtype, copied only if needed."""
    return np.ascontiguousarray(series.to_numpy(), dtype=dtype)


def _microseconds(frame: pl.LazyFrame, column: str, source) -> pl.Expr:
    """A datetime or date column of frame as integer microseconds since the epoch, the engine's unit.

    Raises if the column holds times finer than a microsecond, which would otherwise be truncated.
    """
    dtype = frame.collect_schema()[column]
    if dtype == pl.Date:
        return pl.col(column).cast(pl.Datetime("us")).dt.epoch("us")
    if not isinstance(dtype, pl.Datetime):
        raise ValueError(f"{source}: {column} must be a datetime column; found {dtype}.")
    if (
        dtype.time_unit == "ns"
        and frame.select((pl.col(column).dt.epoch("ns") % 1000 != 0).any()).collect().item()
    ):
        raise ValueError(f"{source}: {column} has times finer than a microsecond, which are unsupported.")
    return pl.col(column).dt.epoch("us")


def _code_ids(names: tuple[str, ...]) -> pl.Expr:
    """Each code's index in names, or null.

    Inside a Parquet scan, Polars maps each distinct string of a dictionary-encoded column once rather than
    once per row.
    """
    if not names:
        return pl.lit(None, dtype=pl.UInt32)
    return pl.col("code").cast(pl.Enum(names), strict=False).to_physical()


def read_shard(path: Path, vocab: Vocabulary) -> EncodedShard:
    """Reads, validates and encodes one MEDS shard."""
    schema = pl.read_parquet_schema(path)
    missing = {"subject_id", "time", "code", "numeric_value"} - set(schema)
    if missing:
        raise ValueError(f"{path}: missing MEDS columns {sorted(missing)}.")
    value = pl.col("numeric_value").cast(pl.Float64).fill_null(np.nan)  # Missing is NaN from here on.

    frame = (
        pl.scan_parquet(path)
        .select(
            pl.col("subject_id").cast(pl.Int64),
            _microseconds(pl.scan_parquet(path), "time", path).alias("time"),
            pl.col("code").is_null().alias("code_null"),
            _code_ids(vocab.code_names).alias("code_id"),
            value.alias("numeric_value"),
        )
        .collect()
    )
    if frame["subject_id"].null_count() or frame["code_null"].any():
        raise ValueError(f"{path}: subject_id and code must not be null.")
    if frame["numeric_value"].is_infinite().any():
        raise ValueError(f"{path}: infinite numeric values are unsupported.")
    shard_subjects = _numpy(frame["subject_id"].unique().sort(), np.int64)

    # Timed events of selected codes, sorted by (subject, time) unless they already are.
    selected = pl.lit(pl.Series(vocab.code_keep, dtype=pl.Boolean)).gather(pl.col("code_id"))
    timed = frame.filter(pl.col("time").is_not_null() & selected) if vocab.code_names else frame.clear()
    subjects, times = _numpy(timed["subject_id"], np.int64), _numpy(timed["time"], np.int64)
    if not engine.rows_sorted(subjects, times):
        timed = timed.sort(["subject_id", "time"], maintain_order=True)
        subjects, times = _numpy(timed["subject_id"], np.int64), _numpy(timed["time"], np.int64)
    codes = _numpy(timed["code_id"], np.int32)
    values = np.where(vocab.value_columns[codes] < 0, np.nan, timed["numeric_value"].to_numpy())

    # Timeless records of selected static/present codes. Timeless rows are few, so they are read again.
    static = (
        pl.scan_parquet(path)
        .filter(pl.col("time").is_null())
        .select(
            pl.col("subject_id").cast(pl.Int64),
            _code_ids(vocab.present_names).fill_null(-1).cast(pl.Int32).alias("present"),
            _code_ids(vocab.mean_names).fill_null(-1).cast(pl.Int32).alias("mean"),
            value.alias("numeric_value"),
        )
        .collect()
    )
    present = static["present"].to_numpy()
    kept = present >= 0
    kept[kept] = vocab.present_keep[present[kept]]
    mean = static["mean"].to_numpy()[kept]
    selected_mean = mean >= 0
    selected_mean[selected_mean] = vocab.mean_keep[mean[selected_mean]]
    return EncodedShard(
        subjects=subjects,
        times=times,
        codes=codes,
        values=values,
        static_subjects=static["subject_id"].to_numpy()[kept],
        static_present=present[kept].astype(np.int32),
        static_mean=np.where(selected_mean, mean, -1).astype(np.int32),
        static_values=np.where(selected_mean, static["numeric_value"].to_numpy()[kept], np.nan),
        shard_subjects=shard_subjects,
    )


def _gather(matrix: csr_array, rows: np.ndarray) -> csr_array:
    """Rows of a CSR matrix by index; -1 selects an empty row."""
    rows = np.asarray(rows, dtype=np.int64)
    # -1 selects an extra empty row past the last one, so it needs no special case.
    bounds = np.append(matrix.indptr, matrix.indptr[-1])
    starts = bounds[np.where(rows >= 0, rows, matrix.shape[0])]
    lengths = bounds[np.where(rows >= 0, rows + 1, matrix.shape[0] + 1)] - starts
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    positions = np.repeat(starts - indptr[:-1], lengths) + np.arange(indptr[-1])
    return csr_array(
        (matrix.data[positions], matrix.indices[positions], indptr), shape=(len(rows), matrix.shape[1])
    )


class ShardOutput:
    """The blocks of one staged shard directory, written by a pool of writer threads.

    Native writes release the GIL, so blocks are written concurrently. Compressed blocks are written by
    storage.write_csr_block. Submitted matrices stay referenced until wait() returns.
    """

    def __init__(self, directory: Path, compress: bool, pool: ThreadPoolExecutor):
        self.directory = directory
        self.compress = compress
        self.pool = pool
        self.pending = []
        self.files: dict[str, int] = {}  # Block name -> file size.
        self.entries = 0

    def _write(self, window: str, agg: str, matrix: csr_array):
        path = block_path(self.directory, window, agg)
        integer = agg in INTEGER_AGGS
        if self.compress:
            write_csr_block(path, matrix, integer=integer, compress=True)
            size = path.stat().st_size
        else:
            size = engine.write_block(path, matrix, integer=integer)
        return f"{window}/{agg}.npz", int(matrix.indptr[-1]), size

    def write(self, window: str, agg: str, matrix: csr_array):
        self.pending.append(self.pool.submit(self._write, window, agg, matrix))

    def wait(self):
        """Waits for every submitted block; raises the first failure after all have finished."""
        error = None
        for future in self.pending:
            try:
                name, entries, size = future.result()
            except BaseException as exception:
                error = error or exception
                continue
            self.files[name] = size
            self.entries += entries
        self.pending.clear()
        if error is not None:
            raise error

    def abandon(self):
        """Cancels pending blocks and waits for running ones, so the directory can be discarded."""
        for future in self.pending:
            future.cancel()
        for future in self.pending:
            if not future.cancelled():
                future.exception()
        self.pending.clear()


class ShardJob:
    """A shard whose blocks are being written.

    finish() waits for them and publishes the shard; after a failure, everything staged is discarded.
    """

    def __init__(self, output: ShardOutput, staged: Path, publisher: Publisher, publish):
        self.output = output
        self.staged = staged
        self.publisher = publisher
        self.publish = publish

    def finish(self):
        try:
            self.output.wait()
            self.publish()
        except BaseException:
            self.abandon()
            raise

    def abandon(self):
        self.output.abandon()
        self.publisher.discard(self.staged)


class Pipeline:
    """MEDS shards to MEDS-Tab's tabularized blocks, at the label times of a task.

    code_metadata is meds-tab-describe's codes.parquet (default <output_dir>/metadata/codes.parquet). Outputs
    whose recorded identity (inputs, configuration, feature metadata, engine build) still matches are reused
    unless reuse is False. threads bounds both the engine and the writer pool. Each public method holds the
    output directory's lock while it runs.
    """

    def __init__(
        self,
        input_dir,
        output_dir,
        config: TabularConfig,
        *,
        code_metadata=None,
        threads: int = 1,
        reuse: bool = True,
    ):
        if not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer.")
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.config = config
        self.code_metadata = Path(code_metadata or self.output_dir / "metadata/codes.parquet")
        self.threads = threads
        self.reuse = reuse
        self.publisher = Publisher(self.output_dir)
        # Per stage and shard: {"reused": True}, or {"reused": False, "seconds": ..., "entries": ...}.
        self.report: dict[str, dict] = defaultdict(dict)
        self._vocabulary: Vocabulary | None = None
        self._metadata_digest: str | None = None

    # -- Feature metadata ------------------------------------------------------------------------------

    def _describe(self):
        if self._vocabulary is not None:
            return
        if not self.code_metadata.is_file():
            raise FileNotFoundError(f"{self.code_metadata} not found: run meds-tab-describe first.")
        frame = pl.read_parquet(self.code_metadata, columns=["code", "count"])
        counts = dict(zip(frame["code"].to_list(), frame["count"].to_list(), strict=True))
        if len(counts) != len(frame):
            raise ValueError(f"{self.code_metadata} lists a feature more than once.")
        self._metadata_digest = digest(counts)
        self._vocabulary = Vocabulary.build(counts, select_features(counts, self.config))

    @property
    def vocabulary(self) -> Vocabulary:
        self._describe()
        return self._vocabulary

    @property
    def metadata_digest(self) -> str:
        self._describe()
        return self._metadata_digest

    # -- Identity and publication ----------------------------------------------------------------------

    def _blocks_identity(self, stage: str, name: str, **extra) -> str:
        """Everything a shard's tabularized blocks depend on."""
        return digest(
            {
                "stage": stage,
                "config": self.config.block_identity(),
                "input": file_fingerprint(self.input_dir / f"{name}.parquet"),
                "metadata": self.metadata_digest,
                "engine": engine.library_sha256(),
                **extra,
            }
        )

    def _publish(self, job: ShardJob, target: Path, manifest: Path, record: dict):
        """Publishes a staged shard directory, then records its manifest.

        The manifest is removed first and written last, so a crash at any step leaves no manifest and the
        shard is recomputed. Readers see the complete old directory, the complete new one, or none.
        """
        manifest.unlink(missing_ok=True)
        self.publisher.publish_dir(job.staged, target)
        prefix = target.relative_to(self.output_dir).as_posix()
        files = {f"{prefix}/{name}": size for name, size in job.output.files.items()}
        atomic_write_json(manifest, {**record, "files": files})

    def _reusable(self, manifest: Path, identity: str, reuse: bool) -> bool:
        return reuse and manifest_matches(read_manifest(manifest), identity, self.output_dir)

    def _run_shards(self, names: list[str], start):
        """Runs start(name) for every shard; it computes the shard, submits its blocks and returns a ShardJob,
        or None when the shard is reused.

        Each shard is finished after the next one is computed, so computing overlaps writing.
        """
        pending: list[ShardJob] = []
        try:
            for name in names:
                job = start(name)
                if job is not None:
                    pending.append(job)
                while len(pending) > 1:
                    pending.pop(0).finish()
            while pending:
                pending.pop(0).finish()
        except BaseException:
            for job in pending:
                job.abandon()
            raise

    def _by_size(self, names: list[str]) -> list[str]:
        """Shards smallest first, so the first writes start as early as possible."""
        return sorted(names, key=lambda n: ((self.input_dir / f"{n}.parquet").stat().st_size, n))

    def _writers(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix="meds-tab-writer")

    # -- Shared computation ----------------------------------------------------------------------------

    def _write_blocks(self, encoded: EncodedShard, row_subjects, row_times, output):
        """Submits every configured block of a shard at the given sorted rows to output."""
        config, vocab = self.config, self.vocabulary
        dynamic = config.dynamic_aggs
        if dynamic:
            shard = engine.Shard(
                encoded.subjects,
                encoded.times,
                encoded.codes,
                encoded.values,
                len(vocab.code_names),
                self.threads,
            )
            matrices = shard.aggregate(
                row_subjects,
                row_times,
                [window_microseconds(w) for w in config.window_sizes],
                sum(STATISTIC_BITS[agg] for agg in dynamic),
                vocab.value_columns,
                vocab.value_count,
                self.threads,
            )
            shard.close()
            # Window-major, then statistics in DYNAMIC_AGGS order, as config.dynamic_aggs lists them.
            blocks = [(window, agg) for window in config.window_sizes for agg in dynamic]
            for (window, agg), matrix in zip(blocks, matrices, strict=True):
                output.write(window, agg, matrix)
        if config.static_aggs:
            present, mean = self._static(encoded, row_subjects, row_times)
            for agg, matrix in (("static/present", present), ("static/mean", mean)):
                if agg in config.static_aggs:
                    output.write("none", agg, matrix)

    def _static(self, encoded: EncodedShard, row_subjects, row_times) -> list[csr_array]:
        vocab = self.vocabulary
        return engine.static_features(
            encoded.static_subjects,
            encoded.static_present,
            encoded.static_mean,
            encoded.static_values,
            row_subjects,
            row_times,
            len(vocab.present_names),
            len(vocab.mean_names),
        )

    # -- Tabularization at label times -----------------------------------------------------------------

    def tabularize_at_labels(self, label_dir) -> dict:
        """Blocks at label times, as the Python backend writes them for each shard <split>/<shard>: the rows
        of <label_dir>/<split>/<shard>.parquet in file order, duplicates included, written to
        tabularize/<split>/<shard>."""
        with output_lock(self.output_dir), self._writers() as pool:
            names = self._by_size(shard_names(self.input_dir))
            self._run_shards(names, lambda name: self._label_shard(name, Path(label_dir), pool))
        return self.report["tabularize"]

    def _label_shard(self, name: str, label_dir: Path, pool: ThreadPoolExecutor) -> ShardJob | None:
        label_file = label_dir / f"{name}.parquet"
        if not label_file.is_file():
            raise FileNotFoundError(f"No labels for shard {name}: {label_file} does not exist.")
        identity = self._blocks_identity("tabularize_at_labels", name, labels=file_fingerprint(label_file))
        manifest = manifest_path(self.output_dir, "tabularize", name)
        if self._reusable(manifest, identity, self.reuse):
            self.report["tabularize"].setdefault(name, {"reused": True})
            return None
        started = time.perf_counter()
        frame = pl.read_parquet(label_file, columns=["subject_id", "prediction_time"])
        if frame.null_count().sum_horizontal().item():
            raise ValueError(f"{label_file}: subject_id and prediction_time must not be null.")
        labels = frame.select(
            pl.col("subject_id").cast(pl.Int64),
            _microseconds(frame.lazy(), "prediction_time", label_file).alias("time"),
        )
        # The engine takes every label row, sorted by (subject, time); label-file order is restored as the
        # blocks are written: output row i is sorted row order[i].
        rows = labels.with_row_index("label").sort("subject_id", "time", maintain_order=True)
        row_subjects, row_times = _numpy(rows["subject_id"], np.int64), _numpy(rows["time"], np.int64)
        order = np.empty(len(rows), dtype=np.int64)
        order[rows["label"].to_numpy()] = np.arange(len(rows))
        encoded = read_shard(self.input_dir / f"{name}.parquet", self.vocabulary)
        staged = self.publisher.stage()
        output = ShardOutput(staged, self.config.do_compress, pool)

        def publish():
            self._publish(job, self.output_dir / "tabularize" / name, manifest, {"identity": identity})
            self.report["tabularize"][name] = {
                "reused": False,
                "seconds": time.perf_counter() - started,
                "entries": output.entries,
            }

        job = ShardJob(output, staged, self.publisher, publish)
        in_order = np.array_equal(order, np.arange(len(rows)))
        try:
            self._write_blocks(
                encoded, row_subjects, row_times, output if in_order else _Reordered(output, order)
            )
        except BaseException:
            job.abandon()
            raise
        del encoded  # Freed while the blocks are written.
        return job


class _Reordered:
    """A ShardOutput whose blocks are written with their rows in the given order."""

    def __init__(self, output: ShardOutput, order: np.ndarray):
        self.output = output
        self.order = order

    def write(self, window: str, agg: str, matrix: csr_array):
        self.output.write(window, agg, _gather(matrix, self.order))
