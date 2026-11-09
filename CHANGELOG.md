# Changelog

## 0.3.0

Feature values differ from 0.2.0: several aggregation bugs are fixed (see Fixed) and the workflow changed (see
Removed and Changed). Re-run `meds-tab-describe` and tabularization, and retrain models, rather than mixing
outputs of the two versions.

### Added

- An optional C++ backend for `meds-tab-tabularize-static` and `meds-tab-tabularize-time-series`
    (#174), selected with `backend=native` (or `MEDS_TAB_BACKEND=native`). It writes the same files as the
    Python backend, which stays the default. It is compiled when MEDS-Tab is installed from source with a
    C++17 compiler and the zlib headers; the PyPI wheel stays pure Python. On a MIMIC-IV cohort of 3,562
    subjects and 4,591 labels it tabularizes in about 12 s, against 26.5 min for the Python backend with 16
    workers, with identical outputs. See `docs/native_backend.md`.
- A LightGBM model launcher, `model_launcher=lightgbm`, with its own sweep space (#181); LightGBM is now a
    dependency. It trains on the same in-memory data as XGBoost, and best-trial predictions are now stored
    for any model that can be reloaded (XGBoost and LightGBM). On the 10 MIMIC-IV "MG" tasks with 2 trials
    each, it matched XGBoost's held-out AUROC (mean 0.9512 against 0.9511) with lower peak memory. The usage
    guide explains how `hydra.launcher.n_jobs` caps model threads under `--multirun`, and where XGBoost and
    LightGBM read an observed 0 differently from no measurement.

### Changed (breaking)

- `meds-tab-describe` fits the code metadata (and so every feature-selection threshold) on the training
    subjects only (#167), read from `<meds_root_dir>/metadata/subject_splits.parquet`. `meds_root_dir`
    defaults to the parent of `input_dir`; a missing splits file is an error. Codes seen only in tuning or
    held-out subjects are no longer selected.
- The static value feature `static/first` is renamed `static/mean` (#176), since it computes the mean of a
    subject's static (timeless) values of the code, not the first one. There is no alias:
    - replace `static/first` with `static/mean` in any `tabularization.aggs` override;
    - re-run `meds-tab-describe` and tabularization. Static value features are now written to
        `.../none/static/mean.npz` and named `<code>/static/mean`, and code metadata counts them under
        `<code>/static/mean`. Code metadata, tabularized outputs and task caches from earlier versions use
        the old name and must be regenerated.
    - Add `static/mean` only if the data has static numeric values: otherwise tabularization stops with
        "No static features found". The MEDS extractions of MIMIC-IV, eICU and NWICU have none, so the
        documented examples leave it out.

### Removed (breaking)

- Full tabularization and `meds-tab-cache-task` are removed (#178). Task-specific tabularization is now the
    only workflow: features are computed at the prediction times of a task's labels, with every window
    ending at the prediction time.
    - `meds-tab-tabularize-static` and `meds-tab-tabularize-time-series` require `input_label_dir`.
    - `meds-tab-cache-task` and its `task_specific_caching` config no longer exist. Instead of
        tabularizing once and caching per task, tabularize each task with its own `input_label_dir` and
        `output_dir`.
    - `meds-tab-model` (and `meds-tab-xgboost`, `meds-tab-autogluon`) reads the labels from the new
        required `input_label_dir`, the same directory given to tabularization, and the tabularized data
        from `input_tabularized_dir` (default `${output_dir}/tabularize`). The model config keys
        `input_tabularized_cache_dir` and `input_label_cache_dir` are removed: rename the first to
        `input_tabularized_dir`, or drop it to use the default, and replace the second with
        `input_label_dir`.
    - Why: computing features at every event time doesn't scale to large datasets, and `cache_task`'s
        finite windows ended at the subject's last event before the prediction time rather than at the
        prediction time, so the two workflows gave different features for the same label.
    - The last release with the cached workflow is 0.2.0.
- `value/has_values_count` is no longer an aggregation option (#170). It has had no implementation since
    tabularization moved to sparse matrices, and was not in the default config. Passing it in
    `tabularization.aggs` got through validation and then failed with "Aggregation method
    'has_values_count' not implemented"; it is now rejected up front as an invalid aggregation. Use
    `value/count`, which counts only the measurements that have a numeric value.

### Fixed

Time-series aggregation (#170, #172, #173):

- The last measurement of every timestamp was dropped, and a timestamp with a single measurement was lost
    entirely.
- `value/min` and `value/max` were written into the first row instead of their window's row, so most
    min/max features were missing.
- `code/count`, `value/count` and `value/sum_sqd` counted timestamps or squared per-timestamp sums instead
    of adding up every measurement of the window.
- `value/min` and `value/max` reduced over absent entries as if they were zeros.
- Observed values of 0, and windows whose values cancel to 0, are stored entries, distinct from "not
    observed", through aggregation, writing and loading.
- Task-specific windows end at the prediction time, not at the subject's last event before it.
- `full` windows cover the whole history, not the last 150 years.
- A NaN `numeric_value` is treated as missing instead of turning sums, minima and maxima into NaN.
- Floating-point values are stored as float32 only when that is exact.

Static features (#173):

- Static features are matched to rows by subject and to columns by name, so a subject without static
    records, unsorted labels or a shard missing a static feature no longer shift values to other subjects
    or features.
- Static features are built from timeless records only.

Code selection and metadata (#173):

- Ties at the `max_included_codes` cutoff are broken by code, so the selected codes don't depend on row
    order.
- A frequency table with exactly two codes no longer crashes `meds-tab-describe`.

Hyperparameter search (#171):

- `min_code_inclusion_count` is searched log-uniformly over integers, as its `tag(log, ...)` declares;
    it was searched uniformly (median ~500,000, selecting no code in most trials on small datasets).

Tutorial (#177):

- `MIMICIV_TUTORIAL/task_tabularize_meds.sh`'s help text lists the six arguments the script takes.
