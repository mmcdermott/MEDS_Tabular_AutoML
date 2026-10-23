"""MEDS-Tab's block files, and atomic publication of the native backend's outputs.

A block file is an NPZ holding ``array``, one homogeneous 3 x nnz array of values, row IDs and column IDs,
and ``shape``, as ``utils.store_matrix`` writes them and ``utils.load_matrix`` reads them. Blocks written
here keep their entries in CSR order and store each in the smallest type that holds every value and index
exactly.

Each shard's outputs are staged under ``<output_dir>/.meds_tab_native/`` and renamed into place, then
recorded in a manifest of what produced them.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from scipy.sparse import csr_array

logger = logging.getLogger(__name__)

INTERNAL_DIR = ".meds_tab_native"
CHUNK_ENTRIES = 1 << 20  # Entries converted per write, which bounds the writer's buffers.


def block_path(shard_dir: Path, window: str, agg: str) -> Path:
    return shard_dir / window / f"{agg}.npz"


def stored_dtype(data: np.ndarray, rows: int, columns: int, *, integer: bool) -> np.dtype:
    """The type of a block's homogeneous array.

    Values get the smallest exact type: an integer type for count statistics, and float32 when it is
    lossless. That type is promoted with the smallest types of the row and column IDs, and widened to
    float64 when float32 cannot hold every ID exactly.
    """
    if not len(data):
        values = np.dtype(np.uint8 if integer else np.float32)
    elif integer:
        values = np.result_type(np.min_scalar_type(int(data.min())), np.min_scalar_type(int(data.max())))
    elif not np.all(np.isfinite(data)):
        raise ValueError("Stored feature values must be finite.")
    else:
        values = np.dtype(np.float32 if np.array_equal(data.astype(np.float32), data) else np.float64)
    max_row, max_column = max(rows - 1, 0), max(columns - 1, 0)
    common = np.result_type(values, np.min_scalar_type(max_row), np.min_scalar_type(max_column))
    if common.kind == "f" and max(max_row, max_column) >= 2 ** (np.finfo(common).nmant + 1):
        common = np.dtype(np.float64)
    if common.kind == "f" and max(max_row, max_column) >= 2**53:
        raise OverflowError("Row and column IDs exceed exact float64 integers.")
    return common


def write_csr_block(path: Path, matrix: csr_array, *, integer: bool, compress: bool):
    """Writes a CSR matrix as a block file.

    The 3 x nnz array is streamed in chunks of CHUNK_ENTRIES, so it is never materialized. Uncompressed, the
    file is byte for byte what the engine's write_block writes.
    """
    indptr, indices, data = matrix.indptr, matrix.indices, matrix.data
    nnz = len(data)
    common = stored_dtype(data, *matrix.shape, integer=integer)
    path.parent.mkdir(parents=True, exist_ok=True)
    method = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", compression=method, allowZip64=True) as archive:
        with archive.open(_member("array.npy", method), "w", force_zip64=True) as handle:
            header = {
                "descr": np.lib.format.dtype_to_descr(common),
                "fortran_order": False,
                "shape": (3, nnz),
            }
            np.lib.format.write_array_header_1_0(handle, header)
            for start in range(0, nnz, CHUNK_ENTRIES):
                handle.write(np.ascontiguousarray(data[start : start + CHUNK_ENTRIES], dtype=common).data)
            for start in range(0, nnz, CHUNK_ENTRIES):
                stop = min(start + CHUNK_ENTRIES, nnz)
                first = int(np.searchsorted(indptr, start, side="right")) - 1
                last = int(np.searchsorted(indptr, stop - 1, side="right"))
                counts = np.minimum(indptr[first + 1 : last + 1], stop) - np.maximum(
                    indptr[first:last], start
                )
                handle.write(np.repeat(np.arange(first, last, dtype=common), counts).data)
            for start in range(0, nnz, CHUNK_ENTRIES):
                handle.write(np.ascontiguousarray(indices[start : start + CHUNK_ENTRIES], dtype=common).data)
        shape = io.BytesIO()
        np.lib.format.write_array(shape, np.asarray(matrix.shape, dtype=np.int64))
        archive.writestr(_member("shape.npy", method), shape.getvalue())


def _member(name: str, method: int) -> zipfile.ZipInfo:
    """A zip member with a fixed timestamp, so identical blocks give identical files."""
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = method
    return info


def label_cache_frame(subjects, times, labels, event_ids, label_dtype) -> pl.DataFrame:
    """A task's label cache, (subject_id, time, label, event_id), as the Python backend writes it."""
    return pl.DataFrame(
        {
            "subject_id": pl.Series(subjects, dtype=pl.Int64),
            "time": pl.Series(np.asarray(times, dtype=np.int64), dtype=pl.Int64).cast(pl.Datetime("us")),
            "label": pl.Series(labels, dtype=label_dtype),
            "event_id": pl.Series(event_ids, dtype=pl.Int64),
        }
    )


def file_fingerprint(path: Path) -> dict:
    """An input file's identity: its path, size, modification time and a hash of its last 64 KiB (for Parquet,
    the footer)."""
    status = path.stat()
    with open(path, "rb") as handle:
        handle.seek(max(status.st_size - 65536, 0))
        tail = hashlib.sha256(handle.read()).hexdigest()
    return {"path": str(path), "bytes": status.st_size, "mtime_ns": status.st_mtime_ns, "tail_sha256": tail}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _temporary(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def atomic_write_json(path: Path, value):
    temporary = _temporary(path)
    try:
        with open(temporary, "w") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_parquet(frame: pl.DataFrame, path: Path):
    temporary = _temporary(path)
    try:
        frame.write_parquet(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def output_lock(output_dir: Path):
    """Holds an exclusive lock on output_dir's native outputs, so concurrent native runs on one output
    directory take turns: no run sees another's partial publication or manifest.

    Holding the lock also means no other run is staging, so leftovers of crashed runs are removed.
    """
    import fcntl  # POSIX only; imported here so that importing this module never needs it.

    internal = output_dir / INTERNAL_DIR
    internal.mkdir(parents=True, exist_ok=True)
    with open(internal / "lock", "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info(f"Waiting for another native run on {output_dir} to finish.")
            fcntl.flock(handle, fcntl.LOCK_EX)
        for leftovers in ("staging", "trash"):
            shutil.rmtree(internal / leftovers, ignore_errors=True)
        yield


@dataclass
class Publisher:
    """Stages shard directories and publishes them by rename.

    Staging lives under ``<output_dir>/.meds_tab_native/staging``, on the same filesystem as the outputs, so
    renames are atomic, and MEDS-Tab's globs never descend there. Replacing a directory removes the old one
    first, so readers see the complete old directory, the complete new one, or none.
    """

    output_dir: Path

    def stage(self) -> Path:
        path = self.output_dir / INTERNAL_DIR / "staging" / uuid.uuid4().hex
        path.mkdir(parents=True)
        return path

    def publish_dir(self, staged: Path, target: Path):
        target.parent.mkdir(parents=True, exist_ok=True)
        self.withdraw(target)
        os.replace(staged, target)

    def withdraw(self, target: Path):
        """Removes a published directory: renamed aside, then deleted."""
        if target.exists():
            trash = self.output_dir / INTERNAL_DIR / "trash" / uuid.uuid4().hex
            trash.parent.mkdir(parents=True, exist_ok=True)
            os.replace(target, trash)
            shutil.rmtree(trash, ignore_errors=True)

    def discard(self, staged: Path):
        shutil.rmtree(staged, ignore_errors=True)


def manifest_path(output_dir: Path, *parts: str) -> Path:
    return output_dir / INTERNAL_DIR / "manifests" / Path(*parts).with_suffix(".json")


def read_manifest(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def manifest_matches(manifest: dict | None, identity: str, root: Path) -> bool:
    """Whether a manifest records identity and every file it lists is present at its recorded size."""
    if not manifest or manifest.get("identity") != identity:
        return False
    files = manifest.get("files", {}).items()
    return all((root / name).is_file() and (root / name).stat().st_size == size for name, size in files)
