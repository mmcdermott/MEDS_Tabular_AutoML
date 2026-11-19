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

ABI_VERSION = 2
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
        ("mean", I32P),
        ("values", F64P),
        ("size", ct.c_int64),
    )


class ArrowColumn(ct.Structure):
    _fields_ = (("validity", ct.c_void_p), ("data", ct.c_void_p), ("offset", ct.c_int64))


class ArrowRecords(ct.Structure):
    _fields_ = (
        ("length", ct.c_int64),
        ("subjects", ArrowColumn),
        ("times", ArrowColumn),
        ("code_indices", ArrowColumn),
        ("code_offsets", ArrowColumn),
        ("code_bytes", ct.c_void_p),
        ("dictionary_length", ct.c_int64),
        ("large_offsets", ct.c_int32),
        ("values", ArrowColumn),
        ("values_64", ct.c_int32),
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
    "meds_tab_code_counter_new": (ct.c_void_p, [I64P, ct.c_int64]),
    "meds_tab_code_counter_free": (None, [ct.c_void_p]),
    "meds_tab_code_counter_add": (ct.c_int, [ct.c_void_p, ct.POINTER(ArrowRecords), ct.c_int64]),
    "meds_tab_code_counter_merge": (ct.c_int, [ct.c_void_p, ct.c_void_p]),
    "meds_tab_code_counter_has_train_subjects": (ct.c_int, [ct.c_void_p, I64P, ct.c_int64, I32P]),
    "meds_tab_code_counter_size": (ct.c_int, [ct.c_void_p, I64P, I64P]),
    "meds_tab_code_counter_export": (ct.c_int, [ct.c_void_p, I64P, ct.c_char_p, I64P]),
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
    subjects, present, mean, values, row_subjects, row_times, present_columns: int, mean_columns: int
) -> list[csr_array]:
    """[static/present, static/mean] for the given sorted rows."""
    arrays = (
        _array(subjects, np.int64, "static subjects"),
        _array(present, np.int32, "static present columns"),
        _array(mean, np.int32, "static mean columns"),
        _array(values, np.float64, "static values"),
    )
    if any(len(a) != len(subjects) for a in arrays):
        raise ValueError("Static record arrays must have equal lengths.")
    records = StaticRecords(
        _pointer(subjects, I64P),
        _pointer(present, I32P),
        _pointer(mean, I32P),
        _pointer(values, F64P),
        len(subjects),
    )
    rows = _rows(row_subjects, row_times)
    handle = library().meds_tab_static(ct.byref(records), ct.byref(rows), present_columns, mean_columns)
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


def _arrow_column(array) -> ArrowColumn:
    """An Arrow array's validity bitmap (None when it has no nulls) and data buffer, with its offset."""
    buffers = array.buffers()
    validity = buffers[0].address if buffers[0] is not None and array.null_count else None
    data = buffers[1].address if len(buffers) > 1 and buffers[1] is not None else None
    return ArrowColumn(validity, data, array.offset)


class CodeCounter:
    """Meds-tab-describe's code frequencies of the training subjects' records, counted by the engine from
    PyArrow record batches (see csrc/describe.h).

    Not thread-safe: use one counter per thread and merge.
    """

    def __init__(self, train_subjects: np.ndarray):
        train = _array(np.ascontiguousarray(train_subjects, dtype=np.int64), np.int64, "train_subjects")
        handle = library().meds_tab_code_counter_new(_pointer(train, I64P), len(train))
        if not handle:
            raise ValueError(_error())
        self.handle = handle
        self.free = library().meds_tab_code_counter_free

    def __del__(self):
        if getattr(self, "handle", None):
            self.free(self.handle)

    def add(self, batches):
        """Adds PyArrow record batches (or a table's batches), counted in one engine call.

        Each has columns subject_id (int64), time (any type: only its nulls are read), code (a dictionary of
        strings with int32 indices) and numeric_value (float32 or float64).
        """
        import pyarrow as pa

        records, keep = [], []
        for batch in batches:
            subjects, times, codes, values = (
                batch["subject_id"],
                batch["time"],
                batch["code"],
                batch["numeric_value"],
            )
            if subjects.type != pa.int64() or values.type not in (pa.float32(), pa.float64()):
                raise ValueError("subject_id must be int64 and numeric_value float32 or float64.")
            if not (pa.types.is_dictionary(codes.type) and codes.type.index_type == pa.int32()):
                raise ValueError("code must be a dictionary array with int32 indices.")
            dictionary = codes.dictionary
            if dictionary.type not in (pa.string(), pa.large_string()):
                raise ValueError("The code dictionary must hold strings.")
            if not times.null_count:
                time_column = ArrowColumn(None, None, 0)  # Every record is timed.
            elif times.buffers()[0] is None:  # A null-typed column: every record is static.
                no_times = np.zeros((len(times) + 7) // 8, dtype=np.uint8)
                keep.append(no_times)
                time_column = ArrowColumn(no_times.ctypes.data, None, 0)
            else:
                time_column = _arrow_column(times)
            dictionary_bytes = dictionary.buffers()[2]
            records.append(
                ArrowRecords(
                    batch.num_rows,
                    _arrow_column(subjects),
                    time_column,
                    _arrow_column(codes.indices),
                    _arrow_column(dictionary),
                    dictionary_bytes.address if dictionary_bytes is not None else None,
                    len(dictionary),
                    dictionary.type == pa.large_string(),
                    _arrow_column(values),
                    values.type == pa.float64(),
                )
            )
            keep.append(batch)
        array = (ArrowRecords * len(records))(*records)
        _check(library().meds_tab_code_counter_add(self.handle, array, len(records)))
        del keep

    def has_train_subject(self, subjects) -> bool:
        """Whether a PyArrow int64 array or chunked array of subjects holds a training subject (null slots may
        give a false True, never a false False)."""
        found = ct.c_int32()
        for chunk in getattr(subjects, "chunks", [subjects]):
            data = chunk.buffers()[1]
            if data is None or not len(chunk):
                continue
            address = data.address + 8 * chunk.offset
            _check(
                library().meds_tab_code_counter_has_train_subjects(
                    self.handle, ct.cast(address, I64P), len(chunk), ct.byref(found)
                )
            )
            if found.value:
                return True
        return False

    def merge(self, other: CodeCounter):
        """Adds other's counts to this counter's, matching codes by name."""
        _check(library().meds_tab_code_counter_merge(self.handle, other.handle))

    def result(self) -> tuple[list[str], np.ndarray]:
        """Every code seen (in first-seen order) and its (code, value, static/present, static/mean) counts."""
        codes, name_bytes = ct.c_int64(), ct.c_int64()
        _check(library().meds_tab_code_counter_size(self.handle, ct.byref(codes), ct.byref(name_bytes)))
        offsets = np.zeros(codes.value + 1, dtype=np.int64)
        names = ct.create_string_buffer(max(name_bytes.value, 1))
        counts = np.zeros((codes.value, 4), dtype=np.int64)
        _check(
            library().meds_tab_code_counter_export(
                self.handle, _pointer(offsets, I64P), names, _pointer(counts, I64P)
            )
        )
        raw = names.raw[: name_bytes.value]
        return [
            raw[a:b].decode() for a, b in zip(offsets[:-1].tolist(), offsets[1:].tolist(), strict=True)
        ], counts
