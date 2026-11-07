# Changelog

## Unreleased

### Removed (breaking)

- Full tabularization and `meds-tab-cache-task` are removed. Task-specific tabularization is now the only
    workflow: features are computed at the prediction times of a task's labels, with every window ending at
    the prediction time.
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
- `value/has_values_count` is no longer an aggregation option. It has had no implementation since
    tabularization moved to sparse matrices, and was not in the default config. Passing it in
    `tabularization.aggs` got through validation and then failed with "Aggregation method
    'has_values_count' not implemented"; it is now rejected up front as an invalid aggregation. Use
    `value/count`, which counts only the measurements that have a numeric value.

### Changed (breaking)

- The static value feature `static/first` is renamed `static/mean`, since it computes the mean of a
    subject's static (timeless) values of the code, not the first one. There is no alias:
    - replace `static/first` with `static/mean` in any `tabularization.aggs` override;
    - re-run `meds-tab-describe` and tabularization. Static value features are now written to
        `.../none/static/mean.npz` and named `<code>/static/mean`, and code metadata counts them under
        `<code>/static/mean`. Code metadata, tabularized outputs and task caches from earlier versions use
        the old name and must be regenerated.
