"""WIP.

This file will be used to generate time series features from the raw data.
"""
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs

from MEDS_tabular_automl.utils import (
    DF_T,
    _normalize_flat_rep_df_cols,
    _parse_flat_feature_column,
)


def _summarize_dynamic_measurements(
    self,
    feature_columns: list[str],
    include_only_subjects: set[int] | None = None,
) -> pl.LazyFrame:
    if include_only_subjects is None:
        df = self.dynamic_measurements_df
    else:
        df = self.dynamic_measurements_df.join(
            self.events_df.filter(pl.col("subject_id").is_in(list(include_only_subjects))).select("event_id"),
            on="event_id",
            how="inner",
        )

    valid_measures = {}
    for feat_col in feature_columns:
        temp, meas, feat = _parse_flat_feature_column(feat_col)

        if temp != "dynamic":
            continue

        if meas not in valid_measures:
            valid_measures[meas] = set()
        valid_measures[meas].add(feat)

    out_dfs = {}
    for m, allowed_vocab in valid_measures.items():
        cfg = self.measurement_configs[m]

        total_observations = int(
            np.ceil(
                cfg.observation_rate_per_case
                * cfg.observation_rate_over_cases
                * sum(self.n_events_per_subject.values())
            )
        )

        count_type = self.get_smallest_valid_uint_type(total_observations)

        if cfg.modality == "univariate_regression" and cfg.vocabulary is None:
            prefix = f"dynamic/{m}/{m}"

            key_col = pl.col(m)
            val_col = pl.col(m).drop_nans().cast(pl.Float32)

            out_dfs[m] = (
                df.lazy()
                .select("measurement_id", "event_id", m)
                .filter(pl.col(m).is_not_null())
                .groupby("event_id")
                .agg(
                    pl.col(m).is_not_null().sum().cast(count_type).alias(f"{prefix}/count"),
                    (
                        (pl.col(m).is_not_nan() & pl.col(m).is_not_null())
                        .sum()
                        .cast(count_type)
                        .alias(f"{prefix}/has_values_count")
                    ),
                    val_col.sum().alias(f"{prefix}/sum"),
                    (val_col**2).sum().alias(f"{prefix}/sum_sqd"),
                    val_col.min().alias(f"{prefix}/min"),
                    val_col.max().alias(f"{prefix}/max"),
                )
            )
            continue
        elif cfg.modality == "multivariate_regression":
            column_cols = [m, m]
            values_cols = [m, cfg.values_column]
            key_prefix = f"{m}_{m}_"
            val_prefix = f"{cfg.values_column}_{m}_"

            key_col = cs.starts_with(key_prefix)
            val_col = cs.starts_with(val_prefix).drop_nans().cast(pl.Float32)

            aggs = [
                key_col.is_not_null()
                .sum()
                .cast(count_type)
                .map_alias(lambda c: f"dynamic/{m}/{c.replace(key_prefix, '')}/count"),
                (
                    (cs.starts_with(val_prefix).is_not_null() & cs.starts_with(val_prefix).is_not_nan())
                    .sum()
                    .map_alias(lambda c: f"dynamic/{m}/{c.replace(val_prefix, '')}/has_values_count")
                ),
                val_col.sum().map_alias(lambda c: f"dynamic/{m}/{c.replace(val_prefix, '')}/sum"),
                (val_col**2).sum().map_alias(lambda c: f"dynamic/{m}/{c.replace(val_prefix, '')}/sum_sqd"),
                val_col.min().map_alias(lambda c: f"dynamic/{m}/{c.replace(val_prefix, '')}/min"),
                val_col.max().map_alias(lambda c: f"dynamic/{m}/{c.replace(val_prefix, '')}/max"),
            ]
        else:
            column_cols = [m]
            values_cols = [m]
            aggs = [
                pl.all().is_not_null().sum().cast(count_type).map_alias(lambda c: f"dynamic/{m}/{c}/count")
            ]

        ID_cols = ["measurement_id", "event_id"]
        out_dfs[m] = (
            df.select(*ID_cols, *set(column_cols + values_cols))
            .filter(pl.col(m).is_in(allowed_vocab))
            .pivot(
                index=ID_cols,
                columns=column_cols,
                values=values_cols,
                aggregate_function=None,
            )
            .lazy()
            .drop("measurement_id")
            .groupby("event_id")
            .agg(*aggs)
        )

    return pl.concat(list(out_dfs.values()), how="align")


def _summarize_over_window(df: DF_T, window_size: str) -> pl.LazyFrame:
    """Apply aggregations to the raw representation over a window size."""
    if isinstance(df, Path):
        df = pl.scan_parquet(df)

    def time_aggd_col_alias_fntr(new_agg: str | None = None) -> Callable[[str], str]:
        if new_agg is None:

            def f(c: str) -> str:
                return "/".join([window_size] + c.split("/")[1:])

        else:

            def f(c: str) -> str:
                return "/".join([window_size] + c.split("/")[1:-1] + [new_agg])

        return f

    # Columns to convert to counts:
    present_indicator_cols = cs.ends_with("/present")

    # Columns to convert to value aggregations:
    value_cols = cs.ends_with("/value")

    # Columns to aggregate via other operations
    cnt_cols = (cs.ends_with("/count") | cs.ends_with("/has_values_count")).fill_null(0)

    cols_to_sum = cs.ends_with("/sum") | cs.ends_with("/sum_sqd")
    cols_to_min = cs.ends_with("/min")
    cols_to_max = cs.ends_with("/max")

    if window_size == "FULL":
        df = df.groupby("subject_id").agg(
            "timestamp",
            # present to counts
            present_indicator_cols.cumsum().map_alias(time_aggd_col_alias_fntr("count")),
            # values to stats
            value_cols.is_not_null().cumsum().map_alias(time_aggd_col_alias_fntr("count")),
            (
                (value_cols.is_not_null() & value_cols.is_not_nan())
                .cumsum()
                .map_alias(time_aggd_col_alias_fntr("has_values_count"))
            ),
            value_cols.cumsum().map_alias(time_aggd_col_alias_fntr("sum")),
            (value_cols**2).cumsum().map_alias(time_aggd_col_alias_fntr("sum_sqd")),
            value_cols.cummin().map_alias(time_aggd_col_alias_fntr("min")),
            value_cols.cummax().map_alias(time_aggd_col_alias_fntr("max")),
            # Raw aggregations
            cnt_cols.cumsum().map_alias(time_aggd_col_alias_fntr()),
            cols_to_sum.cumsum().map_alias(time_aggd_col_alias_fntr()),
            cols_to_min.cummin().map_alias(time_aggd_col_alias_fntr()),
            cols_to_max.cummax().map_alias(time_aggd_col_alias_fntr()),
        )
        df = df.explode(*[c for c in df.columns if c != "subject_id"])
    else:
        df = df.groupby_rolling(
            index_column="timestamp",
            by="subject_id",
            period=window_size,
        ).agg(
            # present to counts
            present_indicator_cols.sum().map_alias(time_aggd_col_alias_fntr("count")),
            # values to stats
            value_cols.is_not_null().sum().map_alias(time_aggd_col_alias_fntr("count")),
            (
                (value_cols.is_not_null() & value_cols.is_not_nan())
                .sum()
                .map_alias(time_aggd_col_alias_fntr("has_values_count"))
            ),
            value_cols.sum().map_alias(time_aggd_col_alias_fntr("sum")),
            (value_cols**2).sum().map_alias(time_aggd_col_alias_fntr("sum_sqd")),
            value_cols.min().map_alias(time_aggd_col_alias_fntr("min")),
            value_cols.max().map_alias(time_aggd_col_alias_fntr("max")),
            # Raw aggregations
            cnt_cols.sum().map_alias(time_aggd_col_alias_fntr()),
            cols_to_sum.sum().map_alias(time_aggd_col_alias_fntr()),
            cols_to_min.min().map_alias(time_aggd_col_alias_fntr()),
            cols_to_max.max().map_alias(time_aggd_col_alias_fntr()),
        )

    return _normalize_flat_rep_df_cols(df, set_count_0_to_null=True)


def get_flat_ts_rep(
    feature_columns: list[str],
    **kwargs,
) -> pl.LazyFrame:
    """Produce raw representation for dynamic data."""

    return _normalize_flat_rep_df_cols(
        _summarize_dynamic_measurements(feature_columns, **kwargs)
        .sort(by=["subject_id", "timestamp"])
        .collect()
        .lazy(),
        [c for c in feature_columns if c.startswith("dynamic")],
    )
    # The above .collect().lazy() shouldn't be necessary but it appears to be for some reason...
