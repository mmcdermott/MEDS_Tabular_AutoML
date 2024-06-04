#!/usr/bin/env python

"""Aggregates time-series data for feature columns across different window sizes."""
import hydra
import numpy as np
import polars as pl
import scipy.sparse as sp
from omegaconf import DictConfig

from MEDS_tabular_automl.mapper import wrap as rwlock_wrap
from MEDS_tabular_automl.utils import (
    CODE_AGGREGATIONS,
    STATIC_CODE_AGGREGATION,
    STATIC_VALUE_AGGREGATION,
    VALUE_AGGREGATIONS,
    hydra_loguru_init,
    load_matrix,
    load_tqdm,
    write_df,
)

VALID_AGGREGATIONS = [
    *VALUE_AGGREGATIONS,
    *CODE_AGGREGATIONS,
    STATIC_CODE_AGGREGATION,
    STATIC_VALUE_AGGREGATION,
]


def generate_row_cached_matrix(matrix, label_df):
    """Generates row-cached matrix for a given matrix and label_df."""
    label_len = label_df.select(pl.len()).collect().item()
    if not matrix.shape[0] == label_len:
        raise ValueError(
            f"Matrix and label_df must have the same number of rows: {matrix.shape[0]} != {label_len}"
        )
    csr = sp.csr_array(matrix)
    valid_ids = label_df.select(pl.col("event_id")).collect().to_series().to_numpy()
    csr = csr[valid_ids, :]
    return sp.coo_array(csr)


@hydra.main(version_base=None, config_path="../configs", config_name="tabularize")
def main(
    cfg: DictConfig,
):
    """Performs row splicing of tabularized data for a specific task."""
    iter_wrapper = load_tqdm(cfg.tqdm)
    if not cfg.loguru_init:
        hydra_loguru_init()
    f_name_resolver = cfg
    # Produce ts representation

    # shuffle tasks
    tabularization_tasks = f_name_resolver.list_static_files() + f_name_resolver.list_ts_files()
    np.random.shuffle(tabularization_tasks)

    # iterate through them
    for data_fp in iter_wrapper(tabularization_tasks):
        # parse as time series agg
        try:
            split, shard_num, agg = f_name_resolver.parse_static_file_path(data_fp)
            window_size = None
        except ValueError:
            split, shard_num, window_size, agg = f_name_resolver.parse_ts_file_path(data_fp)
        label_fp = f_name_resolver.get_label(split, shard_num)
        out_fp = f_name_resolver.get_task_specific_path(split, shard_num, window_size, agg)
        assert label_fp.exists(), f"Output file {label_fp} does not exist."

        def read_fn(fps):
            matrix_fp, label_fp = fps
            return load_matrix(matrix_fp), pl.scan_parquet(label_fp)

        def compute_fn(shard_dfs):
            matrix, label_df = shard_dfs
            cache_matrix = generate_row_cached_matrix(matrix, label_df)
            return cache_matrix

        def write_fn(cache_matrix, out_fp):
            write_df(cache_matrix, out_fp, do_overwrite=cfg.do_overwrite)

        in_fps = [data_fp, label_fp]
        rwlock_wrap(
            in_fps,
            out_fp,
            read_fn,
            write_fn,
            compute_fn,
            do_overwrite=cfg.do_overwrite,
            do_return=False,
        )


if __name__ == "__main__":
    main()
