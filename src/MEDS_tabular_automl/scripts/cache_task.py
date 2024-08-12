#!/usr/bin/env python

"""Aggregates time-series data for feature columns across different window sizes."""
from functools import partial
from importlib.resources import files
from pathlib import Path

import hydra
import numpy as np
import polars as pl
import scipy.sparse as sp
from omegaconf import DictConfig

from ..describe_codes import filter_parquet, get_feature_columns
from ..file_name import list_subdir_files
from ..mapper import wrap as rwlock_wrap
from ..utils import (
    CODE_AGGREGATIONS,
    STATIC_CODE_AGGREGATION,
    STATIC_VALUE_AGGREGATION,
    VALUE_AGGREGATIONS,
    get_events_df,
    get_shard_prefix,
    get_unique_time_events_df,
    hydra_loguru_init,
    load_matrix,
    load_tqdm,
    write_df,
)

config_yaml = files("MEDS_tabular_automl").joinpath("configs/task_specific_caching.yaml")
if not config_yaml.is_file():
    raise FileNotFoundError("Core configuration not successfully installed!")


VALID_AGGREGATIONS = [
    *VALUE_AGGREGATIONS,
    *CODE_AGGREGATIONS,
    STATIC_CODE_AGGREGATION,
    STATIC_VALUE_AGGREGATION,
]


def write_lazyframe(df: pl.LazyFrame, fp: Path):
    df.collect().write_parquet(fp, use_pyarrow=True)


def generate_row_cached_matrix(matrix: sp.coo_array, label_df: pl.LazyFrame) -> sp.coo_array:
    """Generates row-cached matrix for a given matrix and label DataFrame.

    Args:
        matrix: The input sparse matrix.
        label_df: A LazyFrame with an 'event_id' column indicating valid row indices in the matrix.

    Returns:
        A COOrdinate formatted sparse matrix containing only the rows specified by label_df's event_ids.

    Raises:
        ValueError: If the maximum event_id in label_df exceeds the number of rows in the matrix.
    """
    label_len = label_df.select(pl.col("event_id").max()).collect().item()
    if matrix.shape[0] <= label_len:
        raise ValueError(
            f"Label_df event_ids must be valid indexes of sparse matrix: {matrix.shape[0]} <= {label_len}"
        )
    csr = sp.csr_array(matrix)
    valid_ids = label_df.select(pl.col("event_id")).collect().to_series().to_numpy()
    csr = csr[valid_ids, :]
    return sp.coo_array(csr)


@hydra.main(version_base=None, config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig):
    """Performs row splicing of tabularized data for a specific task based on configuration.

    Uses Hydra to manage configurations and logging. The function processes data files based on specified
    task configurations, loading matrices, applying transformations, and writing results.

    Args:
        cfg: The configuration for processing, loaded from a YAML file.
    """
    iter_wrapper = load_tqdm(cfg.tqdm)
    if not cfg.loguru_init:
        hydra_loguru_init()
    # Produce ts representation

    # shuffle tasks
    tabularization_tasks = list_subdir_files(cfg.input_dir, "npz")
    np.random.shuffle(tabularization_tasks)

    label_dir = Path(cfg.input_label_dir)
    label_df = pl.scan_parquet(label_dir / "**/*.parquet").rename(
        {
            "prediction_time": "time",
            cfg.label_column: "label",
        }
    )

    feature_columns = get_feature_columns(cfg.tabularization.filtered_code_metadata_fp)

    # iterate through them
    for data_fp in iter_wrapper(tabularization_tasks):
        # parse as time series agg
        split, shard_num, window_size, code_type, agg_name = Path(data_fp).with_suffix("").parts[-5:]

        raw_data_fp = Path(cfg.output_cohort_dir) / "data" / split / f"{shard_num}.parquet"
        raw_data_df = filter_parquet(raw_data_fp, cfg.tabularization._resolved_codes)
        raw_data_df = (
            get_unique_time_events_df(get_events_df(raw_data_df, feature_columns))
            .with_row_index("event_id")
            .select("patient_id", "time", "event_id")
        )
        shard_label_df = label_df.join(
            raw_data_df.select("patient_id").unique(), on="patient_id", how="inner"
        ).join_asof(other=raw_data_df, by="patient_id", on="time")

        shard_label_fp = Path(cfg.output_label_dir) / split / f"{shard_num}.parquet"
        rwlock_wrap(
            raw_data_fp,
            shard_label_fp,
            pl.scan_parquet,
            write_lazyframe,
            lambda df: shard_label_df,
            do_overwrite=cfg.do_overwrite,
            do_return=False,
        )

        out_fp = (Path(cfg.output_dir) / get_shard_prefix(cfg.input_dir, data_fp)).with_suffix(".npz")
        compute_fn = partial(generate_row_cached_matrix, label_df=shard_label_df)
        write_fn = partial(write_df, do_overwrite=cfg.do_overwrite)

        rwlock_wrap(
            data_fp,
            out_fp,
            load_matrix,
            write_fn,
            compute_fn,
            do_overwrite=cfg.do_overwrite,
            do_return=False,
        )


if __name__ == "__main__":
    main()
