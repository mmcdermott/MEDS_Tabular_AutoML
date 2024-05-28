#!/usr/bin/env python
"""Tabularizes time-series data in MEDS format into tabular representations."""
import hydra
from omegaconf import DictConfig

from MEDS_tabular_automl.generate_ts_features import get_flat_ts_rep
from MEDS_tabular_automl.utils import setup_environment, write_df


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
    flat_dir, split_to_df, feature_columns = setup_environment(cfg)
    # Produce ts representation
    ts_subdir = flat_dir / "ts"

    for sp, subjects_dfs in split_to_df.items():
        sp_dir = ts_subdir / sp

        for i, shard_df in enumerate(subjects_dfs):
            pivot_fp = sp_dir / f"{i}.parquet"
            if pivot_fp.exists() and not cfg.do_overwrite:
                raise FileExistsError(f"do_overwrite is {cfg.do_overwrite} and {pivot_fp.exists()} exists!")

            pivot_df = get_flat_ts_rep(
                feature_columns=feature_columns,
                shard_df=shard_df,
            )
            write_df(pivot_df, pivot_fp, do_overwrite=cfg.do_overwrite)


if __name__ == "__main__":
    tabularize_ts_data()
