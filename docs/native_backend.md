# Native Backend

`meds-tab-tabularize-static`, `meds-tab-tabularize-time-series` and `meds-tab-cache-task` can run on a
C++ engine instead of the Python implementation. The commands, their configuration and the files they
write are unchanged: `TabularDataset` and the model launchers read the native backend's output exactly as
they read the Python backend's. `meds-tab-describe` and model training are not affected.

## Using it

The engine is compiled when MEDS-Tab is installed from source with a C++17 compiler and the zlib headers
available (see below). It is used only when selected:

```bash
meds-tab-describe input_dir=... output_dir=...
meds-tab-tabularize-static input_dir=... output_dir=... backend=native "tabularization.window_sizes=[1d,30d,365d,full]"
meds-tab-tabularize-time-series input_dir=... output_dir=... backend=native "tabularization.window_sizes=[1d,30d,365d,full]"
meds-tab-cache-task input_dir=... output_dir=... backend=native task_name=... input_label_dir=...
```

| Setting          | Default            | Meaning                                                                            |
| ---------------- | ------------------ | ---------------------------------------------------------------------------------- |
| `backend`        | `python`           | `python` or `native`; the `MEDS_TAB_BACKEND` environment variable sets the default |
| `native.threads` | all available CPUs | Engine and writer threads (respects Linux CPU affinity, e.g. Slurm allocations)    |

With `backend=native`, a command fails with an explanation if the engine is not built or the
configuration uses something it does not implement. To check that the engine is built, run
`python -c "from MEDS_tabular_automl.native import engine; engine.library()"`.

The wheel published on PyPI is pure Python and never contains the engine, so building it takes a source
install, with a C++17 compiler and the zlib headers available:

- From PyPI: `pip install --no-binary meds-tab meds-tab`.
- To rebuild an existing installation (installed from the wheel, or built before a compiler was available):
    `pip install --force-reinstall --no-deps --no-cache-dir --no-binary meds-tab meds-tab`, or
    `uv pip install --reinstall-package meds-tab --no-binary meds-tab meds-tab`.
- From a checkout of this repository: `uv sync` builds it; after installing a compiler, run
    `uv sync --reinstall-package meds-tab`.

## Same outputs

`tests/test_native_backend.py` runs every stage with both backends on seeded MEDS data. The data includes:

- duplicate timestamps and repeated codes;
- null, NaN, zero and negative values;
- rare codes;
- subjects without static records, and subjects with only static records;
- labels before any history, unsorted and duplicated.

The test compares every block of the event-time and label-time tabularizations and of the task cache,
compressed and not, as well as the label caches and the `X`, `y` that `TabularDataset` loads. Shapes,
stored entries (including stored zeros), labels, counts, extrema and static values must be identical.
`value/sum` and `value/sum_sqd` may differ only by summation order: the engine uses compensated summation,
and each entry is checked against its window's floating-point error bound. CI runs the whole test suite
with each backend.

## What the engine does differently

The Python backend schedules one job per (shard, window, aggregation): with the default configuration, 30
time-series jobs per shard. Each job reads the shard, rebuilds its sparse representation and reduces every
event row in Python. The native backend reads each shard once, encodes codes to integer IDs during the
Parquet scan, and indexes each subject's events once. From that index it computes every window and
aggregation:

- it sizes each row's output;
- it allocates each final sparse matrix once;
- it fills disjoint (subject, window) spans in parallel.

The numeric aggregations of one window share one set of stored entries. Blocks are written while the next
shard is computed. Task caching reads only the selected rows of each block, at row offsets recorded when
the block was written.

How the work is scheduled also differs:

- **Static and time-series blocks are computed together**, from one read of each shard. Whichever of the
    two tabularization commands runs first writes every configured block; the other finds them complete.
- **One process does the work.** In a Hydra multirun over `worker`, workers other than 0 return at once;
    worker 0 uses `native.threads` threads. The time-series command needs no `--multirun`.
- **Reuse is by identity, not by file existence.** Each shard's outputs are published atomically (staged,
    then renamed into place) together with a manifest of what produced them: the input files (each
    identified by path, size, modification time and a hash of its last 64 KiB, which holds the Parquet
    footer), configuration, feature metadata and engine build. A later run reuses a shard only when all of
    these match and every recorded file is present. A file rewritten in place with the same size and
    modification time is not detected; use `do_overwrite=True`, which recomputes everything. Outputs
    without a manifest (for example written by the Python backend) are recomputed rather than trusted.
- **Concurrent native runs on one output directory take turns.** Each holds a lock on
    `<output_dir>/.meds_tab_native/lock` while it runs. The lock covers the native backend's outputs, not
    the filtered feature metadata that `meds-tab-tabularize-static` writes for both backends before the
    backend runs. As with the Python backend, don't run commands with different feature-selection settings
    on one output directory at the same time.
- **Default output locations only.** `output_tabularized_dir`, `input_tabularized_dir`,
    `output_tabularized_cache_dir` and `output_label_cache_dir` must keep their defaults; the native backend
    raises an error otherwise.
- Manifests, staging directories and each shard's row-offset index live in `<output_dir>/.meds_tab_native/`.

The native engine implements the aggregations `static/present`, `static/first`, `code/count` and
`value/{count,sum,sum_sqd,min,max}`, and window sizes `full` and `<n><us|ms|s|m|h|d|w>`. The Python
backend lists `value/has_values_count` but does not implement it either.

## Performance

These measurements were taken on an Apple M4 Pro (14 CPUs, 48 GiB, macOS 15.6) on the public
[MIMIC-IV demo](https://physionet.org/content/mimic-iv-demo/2.2/) written as MEDS. The demo has 100
subjects and 776,589 events in four shards (two training, tuning, held-out), and there are two admission
tasks with 220 and 245 labels. The runs used the default aggregations, windows `[1d,7d,30d,365d,full]` and
`min_code_inclusion_count=10`. They are the tutorial's commands, each run as its own process into a fresh
output directory, and each time is the median of three alternating runs. The Python backend runs the
time-series stage as the tutorial does (`--multirun worker="range(0,8)" hydra/launcher=joblib`); the
native backend uses its default threads.

| Command                                                   |        Python |        Native |
| --------------------------------------------------------- | ------------: | ------------: |
| `meds-tab-describe` (same code)                           |        0.34 s |        0.34 s |
| `meds-tab-tabularize-static` (native: writes every block) |        0.75 s |        0.95 s |
| `meds-tab-tabularize-time-series`, 8-worker multirun      |       19.04 s |        1.30 s |
| `meds-tab-tabularize-time-series`, single run             |             — |        0.44 s |
| `meds-tab-cache-task`, two tasks                          | 3.94 + 3.48 s | 0.36 + 0.37 s |
| Total (native with a single time-series run)              |        27.6 s | 3.3 s (2.5 s) |

Each command spends about 0.35 s starting Python, Hydra and Polars. On data this small, the native total
is therefore mostly start-up: tabularizing all four shards takes about 0.6 s and each task about 0.03 s.
Peak memory of the whole demo pipeline was 10.5 GiB with the Python backend (all worker processes) and
2.9 GiB with the native backend. The native pipeline is bound by writing its output: the demo's event-time
blocks are 3.2 GiB uncompressed, which takes at least 0.49 s on this SSD.

## Memory and data flow

This table lists every large allocation, scan and copy of one shard, in order. Python owns inputs,
configuration, scheduling and files; C++ owns aggregation buffers. Byte counts are per item.

| Step       | Where                                         | Scans and copies                                                                                                                                                                     | Allocation (owner)                                                                                                                         |
| ---------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Read shard | `pipeline.read_shard`                         | Parquet decode of four columns; codes become IDs inside the scan (each distinct string of a dictionary-encoded column is mapped once); timeless rows are read again with their codes | Polars frame of subjects, times, code IDs and values (Polars)                                                                              |
| Validate   | `pipeline.read_shard`                         | Null counts; one infinite-value scan                                                                                                                                                 | —                                                                                                                                          |
| Encode     | `pipeline.read_shard`                         | One filter (timed rows with selected codes); stable sort by (subject, time) only when a native check finds the rows out of order                                                     | Timed: 28 B/event; static: 24 B/record (`EncodedShard`, Python)                                                                            |
| Rows       | `EncodedShard.event_rows`                     | One scan for (subject, time) changes                                                                                                                                                 | 16 B/row (Python)                                                                                                                          |
| Index      | `Shard` constructor (`csrc/shard.cc`)         | Validation scan; subjects indexed in parallel                                                                                                                                        | 4 B/event, 8 B/distinct subject code, 24 B/subject (native; borrows the Python arrays, which `engine.Shard` keeps alive)                   |
| Size       | `Shard::Aggregate`                            | The same scan, recording per-row sizes                                                                                                                                               | 16 B per row and window (kept through the fill)                                                                                            |
| Allocate   | `Shard::Aggregate`                            | —                                                                                                                                                                                    | Final CSR per window: code/count 16 B/entry; numeric 16 B/entry + 8 B/entry per further statistic; 8 B/row per structure (native `Result`) |
| Fill       | `Shard::Aggregate`                            | Each event enters and leaves each window once; each entry is written once, in place                                                                                                  | Per worker, reused: up to 80 B per subject code and 48 B per subject event                                                                 |
| Views      | `engine.Shard.aggregate`                      | None (SciPy CSR views of the native buffers)                                                                                                                                         | One owner per result frees it when the last view is released                                                                               |
| Static     | `AggregateStatic` (`csrc/static_features.cc`) | Stable sort of record indices by subject                                                                                                                                             | 8 B/record index; output CSR 16 B/entry (native `Result`)                                                                                  |
| Write      | `engine.write_block` (`csrc/npz.cc`)          | Per block: one scan to choose the stored type; per 1 MiB chunk: value conversion, row-ID expansion, column conversion (bounds-checked); CRC-32 of every byte                         | At most `threads` blocks in flight, a 1 MiB buffer each                                                                                    |
| Publish    | `Publisher`, manifests                        | Rename of the staged directory; row-offset index moved into place; manifest written last                                                                                             | Row-offset index: rows, shard subjects, per-subject static CSR, 8 B per row per block                                                      |

A shard is computed while the previous shard's blocks are written (smallest shards first), so up to two
shards' outputs are resident. To bound memory, use smaller shards, as with the Python backend.
Label-time tabularization computes every label row, duplicates included, in (subject, time) order and
writes the blocks in label-file order. Compressed blocks (`do_compress`) are written by Python's `zipfile`.

## Source layout

| Path                                         | Contents                                                                      |
| -------------------------------------------- | ----------------------------------------------------------------------------- |
| `csrc/meds_tab.h`                            | The engine's C ABI: types, ownership, errors                                  |
| `csrc/shard.{h,cc}`                          | Shard index and aggregation                                                   |
| `csrc/static_features.{h,cc}`                | `static/present` and `static/first`                                           |
| `csrc/types.h`, `csrc/parallel.h`            | Events, rows, the CSR and result types; the thread loop                       |
| `csrc/npz.{h,cc}`                            | Block writing and row extraction                                              |
| `csrc/c_api.cc`, `csrc/python_module.cc`     | The C ABI's implementation; the stub that lets setuptools build the library   |
| `src/MEDS_tabular_automl/native/stages.py`   | Backend selection and the three commands' entry points                        |
| `src/MEDS_tabular_automl/native/pipeline.py` | Shard reading and encoding, scheduling, writing, reuse                        |
| `src/MEDS_tabular_automl/native/engine.py`   | ctypes bindings                                                               |
| `src/MEDS_tabular_automl/native/storage.py`  | Block files (Python side), atomic publication, manifests                      |
| `src/MEDS_tabular_automl/native/contract.py` | Supported configurations, window parsing, output identity                     |
| `setup.py`                                   | Builds `csrc/` as the optional extension `MEDS_tabular_automl.native._engine` |
