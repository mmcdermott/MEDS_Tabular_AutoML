# Changelog

## Unreleased

### Removed

- `value/has_values_count` is no longer an aggregation option. It has had no implementation since
    tabularization moved to sparse matrices, and was not in the default config. Passing it in
    `tabularization.aggs` got through validation and then failed with "Aggregation method
    'has_values_count' not implemented"; it is now rejected up front as an invalid aggregation. Use
    `value/count`, which counts only the measurements that have a numeric value.
