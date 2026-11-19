"""The native backend of meds-tab-describe (MEDS_tabular_automl.native.describe and engine.CodeCounter): the
Python backend's code frequencies, counted from PyArrow record batches."""

import math

import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from MEDS_tabular_automl.native import engine
from MEDS_tabular_automl.native.describe import describe_codes
from tests.test_native_backend import engine_built, run, write_meds

pytestmark = pytest.mark.skipif(not engine_built(), reason="The native engine is not built.")


def batch(subjects, times, codes, values, *, dictionary=None, large=False, float64=False) -> pa.RecordBatch:
    strings = pa.large_string() if large else pa.string()
    if dictionary is None:
        code_array = pa.array(codes, type=strings).dictionary_encode()
    else:
        indices = pa.array([None if c is None else dictionary.index(c) for c in codes], type=pa.int32())
        code_array = pa.DictionaryArray.from_arrays(indices, pa.array(dictionary, type=strings))
    return pa.record_batch(
        {
            "subject_id": pa.array(subjects, type=pa.int64()),
            "time": pa.array(times, type=pa.int64()),
            "code": code_array,
            "numeric_value": pa.array(values, type=pa.float64() if float64 else pa.float32()),
        }
    )


def counted(counter: engine.CodeCounter) -> dict[str, list[int]]:
    names, counts = counter.result()
    return {name: row for name, row in zip(names, counts.tolist(), strict=True) if any(row)}


def test_counts_training_subjects_by_kind():
    # Subjects 1 and 3 are training subjects; 1's records are not grouped. Columns: timed, timed with a value,
    # timeless, timeless with a value (null and NaN are missing values; inf is a value).
    nan = math.nan
    counter = engine.CodeCounter(np.array([1, 3]))
    counter.add(
        [
            batch(
                [1, 1, 2, 3, 1, 3, 3, None, 3],
                [None, 5, 5, 6, 7, None, 8, 9, 9],
                ["A", "B", "B", "C", "B", "A", "C", "A", None],
                [5.0, nan, 1.0, math.inf, None, nan, -1.0, 1.0, 1.0],
            )
        ]
    )
    assert counted(counter) == {"A": [0, 0, 2, 1], "B": [2, 0, 0, 0], "C": [2, 2, 0, 0]}


@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("float64", [False, True])
def test_batches_with_repeated_reordered_and_new_dictionaries(large, float64):
    """The same dictionary again (reused), the same codes reordered, and a dictionary with new codes."""
    counter = engine.CodeCounter(np.array([1, 2]))
    first = ["A", "B", "C"]
    counter.add(
        [
            batch([1, 1], [1, 2], ["A", "C"], [1.0, None], dictionary=first, large=large, float64=float64),
            batch([2, 2], [3, None], ["A", "B"], [None, 2.0], dictionary=first, large=large, float64=float64),
            batch([2], [4], ["C"], [3.0], dictionary=["C", "B", "A"], large=large, float64=float64),
        ]
    )
    counter.add([batch([1, 2], [5, 6], ["D", "A"], [1.0, 1.0], dictionary=["A", "D"], large=large)])
    assert counted(counter) == {
        "A": [3, 2, 0, 0],
        "B": [0, 0, 1, 1],
        "C": [2, 1, 0, 0],
        "D": [1, 1, 0, 0],
    }


def test_null_dictionary_entries_and_all_static_batches():
    indices = pa.array([0, 1, 1], type=pa.int32())
    codes = pa.DictionaryArray.from_arrays(indices, pa.array(["A", None]))
    static = pa.record_batch(
        {
            "subject_id": pa.array([1, 1, 1], type=pa.int64()),
            "time": pa.nulls(3),
            "code": codes,
            "numeric_value": pa.array([1.0, 2.0, None], type=pa.float32()),
        }
    )
    counter = engine.CodeCounter(np.array([1]))
    counter.add([static])
    assert counted(counter) == {"A": [0, 0, 1, 1]}


def test_merge_adds_counts_by_name_and_subject_check():
    left, right = engine.CodeCounter(np.array([1])), engine.CodeCounter(np.array([1]))
    left.add([batch([1, 1], [1, 2], ["A", "B"], [1.0, None])])
    right.add([batch([1, 1], [1, 2], ["B", "C"], [None, 1.0])])
    left.merge(right)
    assert counted(left) == {"A": [1, 1, 0, 0], "B": [2, 0, 0, 0], "C": [1, 1, 0, 0]}
    assert left.has_train_subject(pa.chunked_array([[5, 5, 7], [1]]))
    assert not left.has_train_subject(pa.chunked_array([[5, 5, 7], [9]]))


def test_invalid_input_is_rejected():
    with pytest.raises(ValueError, match="sorted"):
        engine.CodeCounter(np.array([3, 1]))
    counter = engine.CodeCounter(np.array([1]))
    good = batch([1], [1], ["A"], [1.0])
    with pytest.raises(ValueError, match="int64"):
        counter.add([good.set_column(0, "subject_id", pa.array([1], type=pa.int32()))])
    with pytest.raises(ValueError, match="dictionary"):
        counter.add([good.set_column(2, "code", pa.array(["A"]))])
    out_of_range = pa.DictionaryArray.from_arrays(pa.array([3], type=pa.int32()), pa.array(["A"]), safe=False)
    with pytest.raises(ValueError, match="dictionary"):
        counter.add([good.set_column(2, "code", out_of_range)])


def test_describe_backends_write_the_same_frequencies(tmp_path):
    meds = tmp_path / "data"
    write_meds(meds)
    outputs = {}
    for backend in ("python", "native"):
        output = tmp_path / backend
        values = {"input_dir": str(meds), "output_dir": str(output), "backend": backend}
        run("describe_codes", "describe_codes", values)
        outputs[backend] = pl.read_parquet(output / "metadata/codes.parquet")
    native = outputs["native"]
    assert native.schema == outputs["python"].schema
    assert native["code"].is_sorted() and native["count"].min() > 0
    assert dict(native.iter_rows()) == dict(outputs["python"].iter_rows())
    kinds = set(native["code"].str.extract(r"/(code|value|static/present|static/mean)$").to_list())
    assert kinds == {"code", "value", "static/present", "static/mean"}


def test_describe_keeps_an_existing_output_unless_overwriting(tmp_path):
    meds, output = tmp_path / "data", tmp_path / "out"
    write_meds(meds)
    codes = output / "metadata/codes.parquet"
    codes.parent.mkdir(parents=True)
    pl.DataFrame({"code": ["X/code"], "count": [1]}).write_parquet(codes)
    values = {"input_dir": str(meds), "output_dir": str(output), "backend": "native"}
    run("describe_codes", "describe_codes", values)
    assert pl.read_parquet(codes)["code"].to_list() == ["X/code"]
    run("describe_codes", "describe_codes", {**values, "do_overwrite": True})
    assert len(pl.read_parquet(codes)) > 1


def test_describe_codes_matches_across_shards_in_flight_and_skips_other_subjects(tmp_path):
    meds = tmp_path / "data"
    write_meds(meds)
    shards = sorted(meds.rglob("*.parquet"))
    subjects = np.unique(np.concatenate([pl.read_parquet(s)["subject_id"].to_numpy() for s in shards]))
    frames = [describe_codes(shards, subjects[::2], shards_in_flight=k) for k in (1, 3)]
    assert frames[0].equals(frames[1]) and len(frames[0]) > 0
    assert describe_codes(shards, np.array([-1]), shards_in_flight=2).is_empty()
