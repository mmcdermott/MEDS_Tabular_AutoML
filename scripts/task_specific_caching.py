#!/usr/bin/env python

"""Aggregates time-series data for feature columns across different window sizes."""
import json

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig

from MEDS_tabular_automl.file_name import FileNameResolver
from MEDS_tabular_automl.mapper import wrap as rwlock_wrap
from MEDS_tabular_automl.utils import (
    hydra_loguru_init,
    load_tqdm,
    write_df,
)


def generate_row_cached_matrix(matrix, label_df, feature_columns):
    """Generates row-cached matrix for a given matrix and label_df."""
    return None  # TODO


@hydra.main(version_base=None, config_path="../configs", config_name="tabularize")
def task_specific_cache(
    cfg: DictConfig,
):
    """Performs row splicing of tabularized data for a specific task."""
    iter_wrapper = load_tqdm(cfg.tqdm)
    if not cfg.test:
        hydra_loguru_init()
    f_name_resolver = FileNameResolver(cfg)
    # Produce ts representation
    meds_shard_fps = f_name_resolver.list_meds_files()
    feature_columns = json.load(open(f_name_resolver.get_feature_columns_fp()))

    # shuffle tasks
    tabularization_tasks = f_name_resolver.list_static_files() + f_name_resolver.list_ts_files()
    np.random.shuffle(tabularization_tasks)

    # iterate through them
    for shard_fp, window_size, agg in iter_wrapper(tabularization_tasks):
        agg, window_size = 0, 0  # TODO: fix
        shard_num = shard_fp.stem
        split = shard_fp.parent.stem
        out_fp = f_name_resolver.get_task_specific_output(
            split, shard_num, window_size, agg
        )  # TODO make this function

        def read_fn(fps):
            matrix_fp, label_fp = fps
            return load_matrix(fp), pl.scan_parquet(label_fp)

        def compute_fn(shard_dfs):
            matrix, label_df = shard_dfs
            cache_matrix = generate_row_cached_matrix(matrix, label_df, feature_columns)
            return cache_matrix

        def write_fn(cache_matrix, out_fp):
            write_df(cache_matrix, out_fp, do_overwrite=cfg.do_overwrite)

        rwlock_wrap(
            shard_fp,
            ts_fp,
            read_fn,
            write_fn,
            compute_fn,
            do_overwrite=cfg.do_overwrite,
            do_return=False,
        )


if __name__ == "__main__":
    task_specific_cache()
