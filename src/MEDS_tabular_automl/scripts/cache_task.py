#!/usr/bin/env python

"""Aggregates time-series data for feature columns across different window sizes."""
from importlib.resources import files
from pathlib import Path

import hydra
import numpy as np
import polars as pl
import scipy.sparse as sp
from omegaconf import DictConfig

from MEDS_tabular_automl.file_name import list_subdir_files
from MEDS_tabular_automl.mapper import wrap as rwlock_wrap
from MEDS_tabular_automl.utils import (
    CODE_AGGREGATIONS,
    STATIC_CODE_AGGREGATION,
    STATIC_VALUE_AGGREGATION,
    VALUE_AGGREGATIONS,
    get_shard_prefix,
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
def main(cfg: DictConfig) -> None:
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

    # iterate through them
    for data_fp in iter_wrapper(tabularization_tasks):
        # parse as time series agg
        split, shard_num, window_size, code_type, agg_name = Path(data_fp).with_suffix("").parts[-5:]
        label_fp = Path(cfg.input_label_dir) / split / f"{shard_num}.parquet"
        out_fp = (Path(cfg.output_dir) / get_shard_prefix(cfg.input_dir, data_fp)).with_suffix(".npz")
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
