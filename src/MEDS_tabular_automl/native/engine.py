"""Bindings for the native engine's C ABI (``csrc/meds_tab.h``), loaded with ctypes.

The engine is built into this package as ``_engine`` when MEDS-Tab is installed (``setup.py``).
``ctypes.CDLL`` releases the GIL during every call, so block writes on several threads run in parallel.
Arrays passed to native code must be aligned, C-contiguous and of the declared dtype (checked here).
Result matrices are returned as SciPy CSR views of the native buffers; the views keep their result alive.
"""

from __future__ import annotations

import ctypes as ct
import hashlib
import importlib.util
import os
from functools import cache
from pathlib import Path

import numpy as np
from scipy.sparse import csr_array

ABI_VERSION = 1
I64P = ct.POINTER(ct.c_int64)
I32P = ct.POINTER(ct.c_int32)
F64P = ct.POINTER(ct.c_double)

NOT_BUILT = (
    "MEDS-Tab's native engine is not built. It is compiled only by a source install with a C++17 compiler "
    "and the zlib headers available, e.g. `pip install --force-reinstall --no-deps --no-cache-dir "
    "--no-binary meds-tab meds-tab`, or `uv sync --reinstall-package meds-tab` in a checkout. See "
    "docs/native_backend.md."
)


class Events(ct.Structure):
    _fields_ = (("subjects", I64P), ("times", I64P), ("codes", I32P), ("values", F64P), ("size", ct.c_int64))


class Rows(ct.Structure):
    _fields_ = (("subjects", I64P), ("times", I64P), ("size", ct.c_int64))


class StaticRecords(ct.Structure):
    _fields_ = (
        ("subjects", I64P),
        ("present", I32P),
        ("first", I32P),
        ("values", F64P),
        ("size", ct.c_int64),
    )


class CSR(ct.Structure):
    _fields_ = (
        ("rows", ct.c_int64),
        ("columns", ct.c_int64),
        ("nnz", ct.c_int64),
        ("index_64", ct.c_int32),
        ("values_integer", ct.c_int32),
        ("outer", ct.c_void_p),
        ("inner", ct.c_void_p),
        ("values", ct.c_void_p),
    )


SIGNATURES = {
    "meds_tab_rows_sorted": (ct.c_int, [ct.POINTER(Rows)]),
    "meds_tab_shard_new": (ct.c_void_p, [ct.POINTER(Events), ct.c_int64, ct.c_int32]),
    "meds_tab_shard_free": (None, [ct.c_void_p]),
    "meds_tab_shard_aggregate": (
        ct.c_void_p,
        [ct.c_void_p, ct.POINTER(Rows), I64P, ct.c_int64, ct.c_uint32, I32P, ct.c_int64, ct.c_int32],
    ),
    "meds_tab_static": (
        ct.c_void_p,
        [ct.POINTER(StaticRecords), ct.POINTER(Rows), ct.c_int64, ct.c_int64],
    ),
    "meds_tab_result_size": (ct.c_int64, [ct.c_void_p]),
    "meds_tab_result_matrix": (ct.c_int, [ct.c_void_p, ct.c_int64, ct.POINTER(CSR)]),
    "meds_tab_result_free": (None, [ct.c_void_p]),
    "meds_tab_write_block": (ct.c_int, [ct.c_char_p, ct.POINTER(CSR), ct.c_int32, I64P]),
    "meds_tab_extract_rows": (
        ct.c_int,
        [ct.c_char_p, I64P, ct.c_int64, ct.c_int64, I64P, ct.c_int64, ct.c_int32, ct.c_char_p, I64P, I64P],
    ),
}


def library_path() -> Path:
    spec = importlib.util.find_spec(f"{__package__}._engine")
    if spec is None or spec.origin is None:
        raise FileNotFoundError(NOT_BUILT)
    return Path(spec.origin)


@cache
def library() -> ct.CDLL:
    """The loaded engine; raises FileNotFoundError if it is not built."""
    lib = ct.CDLL(str(library_path()))
    lib.meds_tab_abi_version.restype = ct.c_int
    if lib.meds_tab_abi_version() != ABI_VERSION:
        raise RuntimeError(f"The native engine was built for another ABI version (expected {ABI_VERSION}).")
    lib.meds_tab_last_error.restype = ct.c_char_p
    for name, (restype, argtypes) in SIGNATURES.items():
        function = getattr(lib, name)
        function.restype = restype
        function.argtypes = argtypes
    return lib


@cache
def library_sha256() -> str:
    """Identifies the engine build in cache manifests."""
    return hashlib.sha256(library_path().read_bytes()).hexdigest()


def _error() -> str:
    return library().meds_tab_last_error().decode()


def _array(values, dtype, name: str) -> np.ndarray:
    """Checks that an array can be borrowed by native code."""
    if not isinstance(values, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array.")
    if values.ndim != 1 or values.dtype != dtype or not values.flags.c_contiguous or not values.flags.aligned:
        raise ValueError(f"{name} must be an aligned, contiguous one-dimensional {np.dtype(dtype)} array.")
    return values


def _pointer(values: np.ndarray | None, kind):
    return None if values is None else values.ctypes.data_as(kind)


def _rows(subjects: np.ndarray, times: np.ndarray) -> Rows:
    _array(subjects, np.int64, "row subjects")
    _array(times, np.int64, "row times")
    if len(subjects) != len(times):
        raise ValueError("Row arrays must have equal lengths.")
    return Rows(_pointer(subjects, I64P), _pointer(times, I64P), len(subjects))


def rows_sorted(subjects: np.ndarray, times: np.ndarray) -> bool:
    """Whether (subject, time) rows are sorted."""
    return bool(library().meds_tab_rows_sorted(ct.byref(_rows(subjects, times))))


class _Result:
    """Owns a native result; every NumPy view of its buffers holds a reference."""

    __slots__ = ("__weakref__", "free", "handle")

    def __init__(self, handle):
        self.handle = handle
        self.free = library().meds_tab_result_free  # Held, so freeing never needs module globals.

    def __del__(self):
        self.free(self.handle)


def _view(address: int, count: int, scalar, owner: _Result) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=scalar)
    buffer = (scalar * count).from_address(address)
    buffer._owner = owner
    return np.ctypeslib.as_array(buffer)


def _matrices(handle) -> list[csr_array]:
    """Every matrix of a result handle as CSR views of the native buffers."""
    lib = library()
    if not handle:
        raise ValueError(_error())
    owner = _Result(handle)
    matrices = []
    for index in range(lib.meds_tab_result_size(handle)):
        csr = CSR()
        if not lib.meds_tab_result_matrix(handle, index, ct.byref(csr)):
            raise RuntimeError(_error())
        indptr = _view(csr.outer, csr.rows + 1, ct.c_int64, owner)
        indices = _view(csr.inner, csr.nnz, ct.c_int64, owner)
        data = _view(csr.values, csr.nnz, ct.c_double, owner)
        matrices.append(csr_array((data, indices, indptr), shape=(csr.rows, csr.columns), copy=False))
    return matrices


class Shard:
    """One shard's timed events, indexed natively.

    Keeps the borrowed arrays alive.
    """

    def __init__(self, subjects, times, codes, values, code_count: int, threads: int):
        self.arrays = (
            _array(subjects, np.int64, "event subjects"),
            _array(times, np.int64, "event times"),
            _array(codes, np.int32, "event codes"),
            _array(values, np.float64, "event values"),
        )
        if any(len(a) != len(subjects) for a in self.arrays):
            raise ValueError("Event arrays must have equal lengths.")
        events = Events(
            _pointer(subjects, I64P),
            _pointer(times, I64P),
            _pointer(codes, I32P),
            _pointer(values, F64P),
            len(subjects),
        )
        self.code_count = code_count
        self.handle = library().meds_tab_shard_new(ct.byref(events), code_count, threads)
        if not self.handle:
            raise ValueError(_error())

    def close(self):
        if getattr(self, "handle", None):
            library().meds_tab_shard_free(self.handle)
            self.handle = None

    __del__ = close

    def aggregate(
        self, subjects, times, windows, statistics: int, value_columns, value_column_count: int, threads: int
    ) -> list[csr_array]:
        """One matrix per window and selected statistic, window-major (see meds_tab.h)."""
        widths = np.ascontiguousarray(windows, dtype=np.int64)
        _array(value_columns, np.int32, "value columns")
        if len(value_columns) != self.code_count:
            raise ValueError("value_columns must map every code: it needs code_count entries.")
        handle = library().meds_tab_shard_aggregate(
            self.handle,
            ct.byref(_rows(subjects, times)),
            _pointer(widths, I64P),
            len(widths),
            statistics,
            _pointer(value_columns, I32P),
            value_column_count,
            threads,
        )
        return _matrices(handle)


def static_features(
    subjects, present, first, values, row_subjects, row_times, present_columns: int, first_columns: int
) -> list[csr_array]:
    """[static/present, static/first] for the given sorted rows."""
    arrays = (
        _array(subjects, np.int64, "static subjects"),
        _array(present, np.int32, "static present columns"),
        _array(first, np.int32, "static first columns"),
        _array(values, np.float64, "static values"),
    )
    if any(len(a) != len(subjects) for a in arrays):
        raise ValueError("Static record arrays must have equal lengths.")
    records = StaticRecords(
        _pointer(subjects, I64P),
        _pointer(present, I32P),
        _pointer(first, I32P),
        _pointer(values, F64P),
        len(subjects),
    )
    rows = _rows(row_subjects, row_times)
    handle = library().meds_tab_static(ct.byref(records), ct.byref(rows), present_columns, first_columns)
    return _matrices(handle)


def _check(status: int):
    if status != 1:
        raise (OSError if status < 0 else ValueError)(_error())


def write_block(path: Path, matrix: csr_array, *, integer: bool) -> int:
    """Writes matrix as an uncompressed block file, byte for byte as storage.write_csr_block does; returns the
    file size."""
    indptr, indices = matrix.indptr, matrix.indices
    if indptr.dtype != indices.dtype or indptr.dtype not in (np.int32, np.int64):
        indptr, indices = indptr.astype(np.int64), indices.astype(np.int64)
    data = matrix.data
    if data.dtype not in (np.float64, np.int64):
        data = data.astype(np.int64 if data.dtype.kind in "iub" else np.float64)
    indptr, indices, data = (np.ascontiguousarray(a) for a in (indptr, indices, data))
    block = CSR(
        matrix.shape[0],
        matrix.shape[1],
        len(data),
        indptr.dtype == np.int64,
        data.dtype == np.int64,
        indptr.ctypes.data,
        indices.ctypes.data,
        data.ctypes.data,
    )
    size = ct.c_int64()
    _check(library().meds_tab_write_block(os.fsencode(path), ct.byref(block), integer, ct.byref(size)))
    return size.value


def extract_rows(
    source: Path, indptr: np.ndarray, rows: np.ndarray, columns: int, target: Path, *, integer: bool
) -> tuple[np.ndarray, int]:
    """Writes the given rows of a block written by write_block (-1 selects an empty row) to target, reading
    only their entries; indptr is the source's row offsets.

    Returns the target's row offsets and size.
    """
    indptr = np.ascontiguousarray(indptr, dtype=np.int64)
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    written = np.empty(len(rows) + 1, dtype=np.int64)
    size = ct.c_int64()
    status = library().meds_tab_extract_rows(
        os.fsencode(source),
        _pointer(indptr, I64P),
        len(indptr) - 1,
        columns,
        _pointer(rows, I64P),
        len(rows),
        integer,
        os.fsencode(target),
        _pointer(written, I64P),
        ct.byref(size),
    )
    _check(status)
    return written, size.value
