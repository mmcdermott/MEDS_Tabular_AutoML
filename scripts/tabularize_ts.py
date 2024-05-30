#!/usr/bin/env python
"""Tabularizes time-series data in MEDS format into tabular representations."""

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from MEDS_tabular_automl.generate_ts_features import get_flat_ts_rep
from MEDS_tabular_automl.mapper import wrap as rwlock_wrap
from MEDS_tabular_automl.utils import load_tqdm, setup_environment, write_df


@hydra.main(version_base=None, config_path="../configs", config_name="tabularize")
def tabularize_ts_data(
    cfg: DictConfig,
):
    """Processes a medical dataset to generates and stores flat representatiosn of time-series data.

    This function handles MEDS format data and pivots tables to create two types of data files
    with patient_id and timestamp indexes:
        code data: containing a column for every code and 1 and 0 values indicating presence
        value data: containing a column for every code which the numerical value observed.

    Args:
        cfg: configuration dictionary containing the necessary parameters for tabularizing the data.
    """
    iter_wrapper = load_tqdm(cfg.tqdm)
    flat_dir, split_to_fp, feature_columns = setup_environment(cfg, load_data=False)

    # Produce ts representation
    ts_subdir = flat_dir / "ts"

    for sp, shard_fps in split_to_fp.items():
        sp_dir = ts_subdir / sp

        for i, shard_fp in enumerate(iter_wrapper(shard_fps)):
            out_fp = sp_dir / f"{i}.pkl"

            def read_fn(in_fp):
                return pl.scan_parquet(in_fp)

            def compute_fn(shard_df):
                return get_flat_ts_rep(
                    feature_columns=feature_columns,
                    shard_df=shard_df,
                )

            def write_fn(data, out_df):
                write_df(data, out_df, do_overwrite=cfg.do_overwrite)

            rwlock_wrap(
                shard_fp,
                out_fp,
                read_fn,
                write_fn,
                compute_fn,
                do_overwrite=cfg.do_overwrite,
                do_return=False,
            )
    logger.info("Generated TS flat representations.")


if __name__ == "__main__":
    tabularize_ts_data()
