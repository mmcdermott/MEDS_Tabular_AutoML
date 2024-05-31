#!/usr/bin/env python
"""Tabularizes time-series data in MEDS format into tabular representations."""
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from omegaconf import DictConfig
from scipy.sparse import coo_matrix, csc_matrix, hstack

from MEDS_tabular_automl.mapper import wrap as rwlock_wrap
from MEDS_tabular_automl.utils import load_tqdm, setup_environment, write_df


def merge_dfs(feature_columns, static_df, ts_df):
    """Merges static and time-series dataframes.

    This function merges the static and time-series dataframes based on the patient_id column.

    Args:
    - feature_columns (List[str]): A list of feature columns to include in the merged dataframe.
    - static_df (pd.DataFrame): A dataframe containing static features.
    - ts_df (pd.DataFrame): A dataframe containing time-series features.

    Returns:
    - pd.DataFrame: A merged dataframe containing static and time-series features.
    """
    # Make static data sparse and merge it with the time-series data
    logger.info("Make static data sparse and merge it with the time-series data")
    static_df[static_df.columns[1:]] = (
        static_df[static_df.columns[1:]].fillna(0).astype(pd.SparseDtype("float64", fill_value=0))
    )
    merge_df = pd.merge(ts_df, static_df, on=["patient_id"], how="left")
    # indexes_df = merge_df[["patient_id", "timestamp"]]
    # drop indexes
    merge_df = merge_df.drop(columns=["patient_id", "timestamp"])
    # TODO: fix naming convention, we are generating value rows with zero frequency so remove those
    merge_df = merge_df.rename(
        columns={
            c: "/".join(c.split("/")[1:-1]) for c in merge_df.columns if c.split("/")[-2] in ["code", "value"]
        }
    )

    # Convert to sparse matrix and remove 0 frequency columns (i.e. columns not in feature_columns)
    logger.info(
        "Convert to sparse matrix and remove 0 frequency columns (i.e. columns not in feature_columns)"
    )
    original_sparse_matrix = merge_df.sparse.to_coo()
    missing_columns = [col for col in feature_columns if col not in merge_df.columns]

    # reorder columns to be in order of feature_columns
    logger.info("Reorder columns to be in order of feature_columns")
    final_sparse_matrix = hstack(
        [original_sparse_matrix, coo_matrix((merge_df.shape[0], len(missing_columns)))]
    )
    index_map = {name: index for index, name in enumerate(feature_columns)}
    reverse_map = [index_map[col] for col in feature_columns]
    final_sparse_matrix = coo_matrix(csc_matrix(final_sparse_matrix)[:, reverse_map])

    # convert to np matrix of data, row, col
    logger.info(f"Final sparse matrix shape: {final_sparse_matrix.shape}")
    data, row, col = final_sparse_matrix.data, final_sparse_matrix.row, final_sparse_matrix.col
    final_matrix = np.matrix([data, row, col])
    return final_matrix


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
    med_dir = Path(cfg.tabularized_data_dir)
    ts_dir = med_dir / "ts"
    static_dir = med_dir / "static"
    shard_fps = list(ts_dir.glob("*/*/*/*/*.pkl"))

    # Produce ts representation
    out_subdir = flat_dir / "sparse"

    for shard_fp in iter_wrapper(shard_fps):
        split = shard_fp.parts[-5]
        in_ts_fp = shard_fp
        assert in_ts_fp.exists(), f"{in_ts_fp} does not exist!"
        in_static_fp = static_dir / split / f"{shard_fp.stem}.parquet"
        assert in_static_fp.exists(), f"{in_static_fp} does not exist!"
        out_fp = out_subdir / "/".join(shard_fp.parts[-5:-1]) / f"{shard_fp.stem}"
        out_fp.parent.mkdir(parents=True, exist_ok=True)

        def read_fn(in_fps):
            in_static_fp, in_ts_fp = in_fps
            static_df = pl.read_parquet(in_static_fp)
            ts_df = pd.read_pickle(in_ts_fp)
            return [static_df, ts_df]

        def compute_fn(shards):
            static_df, shard_df = shards
            return merge_dfs(
                feature_columns=feature_columns,
                static_df=static_df.to_pandas(),
                ts_df=shard_df,
            )

        def write_fn(data, out_df):
            write_df(data, out_df, do_overwrite=cfg.do_overwrite)

        in_fps = in_static_fp, in_ts_fp
        logger.info(f"Processing {in_static_fp} and\n{in_ts_fp}")
        logger.info(f"Writing to {out_fp}...")
        rwlock_wrap(
            in_fps,
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
