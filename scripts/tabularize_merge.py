#!/usr/bin/env python
"""Tabularizes time-series data in MEDS format into tabular representations."""
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from omegaconf import DictConfig
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, hstack

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
    # TODO - store static and ts data as numpy matrices
    # TODO - Eventually do this duplication at the task specific stage after filtering patients and features
    # Make static data sparse and merge it with the time-series data
    logger.info("Make static data sparse and merge it with the time-series data")
    assert static_df.patient_id.is_monotonic_increasing
    assert ts_df.patient_id.is_monotonic_increasing
    sparse_time_series = ts_df.drop(columns=["patient_id", "timestamp"]).sparse.to_coo()

    num_patients = max(static_df.patient_id.nunique(), ts_df.patient_id.nunique())

    # load static data as sparse matrix
    static_matrix = static_df.drop(columns="patient_id").values
    data_list = []
    rows = []
    cols = []
    for row in range(static_matrix.shape[0]):
        for col in range(static_matrix.shape[1]):
            data = static_matrix[row, col]
            if (data is not None) and (data != 0):
                data_list.append(data)
                rows.append(row)
                cols.append(col)
    static_matrix = csr_matrix((data_list, (rows, cols)), shape=(num_patients, static_matrix.shape[1]))
    # Duplicate static matrix rows to match time-series data
    duplication_index = ts_df["patient_id"].value_counts().sort_index().reset_index(drop=True)
    reindex_slices = np.repeat(duplication_index.index.values, duplication_index.values)
    static_matrix = static_matrix[reindex_slices, :]

    # TODO: fix naming convention, we are generating value rows with zero frequency so remove those
    ts_columns = ["/".join(c.split("/")[1:-1]) for c in ts_df.columns]
    sparse_columns = ts_columns + list(static_df.columns)

    # Convert to sparse matrix and remove 0 frequency columns (i.e. columns not in feature_columns)
    logger.info(
        "Convert to sparse matrix and remove 0 frequency columns (i.e. columns not in feature_columns)"
    )
    set_sparse_cols = set(sparse_columns)
    missing_columns = [col for col in feature_columns if col not in set_sparse_cols]

    # reorder columns to be in order of feature_columns
    logger.info("Reorder columns to be in order of feature_columns")
    final_sparse_matrix = hstack(
        [sparse_time_series, static_matrix, coo_matrix((sparse_time_series.shape[0], len(missing_columns)))]
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
def merge_data(
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
    merge_data()
