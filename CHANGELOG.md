# Changelog

## Unreleased

### Changed (breaking)

- The static value feature `static/first` is renamed `static/mean`, since it computes the mean of a
    subject's static (timeless) values of the code, not the first one. There is no alias:
    - replace `static/first` with `static/mean` in any `tabularization.aggs` override;
    - re-run `meds-tab-describe` and tabularization. Static value features are now written to
        `.../none/static/mean.npz` and named `<code>/static/mean`, and code metadata counts them under
        `<code>/static/mean`. Code metadata, tabularized outputs and task caches from earlier versions use
        the old name and must be regenerated.
