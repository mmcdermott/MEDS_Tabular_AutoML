#!/usr/bin/env python

"""Aggregates time-series data for feature columns across different window sizes."""
import os

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from MEDS_tabular_automl.generate_summarized_reps import generate_summary
from MEDS_tabular_automl.generate_ts_features import get_flat_ts_rep
from MEDS_tabular_automl.mapper import wrap as rwlock_wrap
from MEDS_tabular_automl.utils import setup_environment, write_df


def hydra_loguru_init() -> None:
    """Adds loguru output to the logs that hydra scrapes.

    Must be called from a hydra main!
    """
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.add(os.path.join(hydra_path, "main.log"))


@hydra.main(version_base=None, config_path="../configs", config_name="tabularize")
def summarize_ts_data_over_windows(
    cfg: DictConfig,
):
    """Processes time-series data by summarizing it across different windows, creating a flat, summarized
    representation of the data for analysis.

    This function orchestrates the data processing pipeline for summarizing time-series data. It loads
    data from the tabularize_ts stage, iterates through the pivoted wide dataframes for each split and
    shards and then applies a range aggregations across different window sizes defined in the config
    The summarized data is then written to disk in a structured directory format.

    Args:
        cfg: A configuration dictionary derived from Hydra, containing parameters such as the input data
             directory, output directory, and specifics regarding the summarization process (like window
             sizes and aggregation functions).

    Workflow:
        1. Set up the environment based on configuration settings.
        2. Load and categorize time-series file paths by their data splits.
        3. Pair code and value files for each split.
        4. For each pair of files in each split:
            - Load the dataframes in a lazy manner.
            - Summarize the dataframes based on predefined window sizes and aggregation methods.
            - Write the summarized dataframe to disk.

    Raises:
        FileNotFoundError: If specified directories or files in the configuration are not found.
        ValueError: If required columns like 'code' or 'value' are missing in the data files.
    """
    if not cfg.test:
        hydra_loguru_init()
    flat_dir, split_to_fps, feature_columns = setup_environment(cfg, load_data=False)
    # Produce ts representation
    ts_subdir = flat_dir / "ts"

    for sp, shard_fps in split_to_fps.items():
        sp_dir = ts_subdir / sp

        for i, shard_fp in enumerate(shard_fps):
            for window_size in cfg.window_sizes:
                for agg in cfg.aggs:
                    pivot_fp = sp_dir / window_size / agg / f"{i}.pkl"
                    if pivot_fp.exists() and not cfg.do_overwrite:
                        raise FileExistsError(
                            f"do_overwrite is {cfg.do_overwrite} and {pivot_fp.exists()} exists!"
                        )

                    def read_fn(fp):
                        return pl.scan_parquet(fp)

                    def compute_fn(shard_df):
                        # Load Sparse DataFrame
                        pivot_df = get_flat_ts_rep(
                            feature_columns=feature_columns,
                            shard_df=shard_df,
                        )

                        # Summarize data -- applying aggregations on various window sizes
                        summary_df = generate_summary(
                            feature_columns,
                            pivot_df,
                            window_size,
                            agg,
                        )
                        assert summary_df.shape[1] > 2, "No data found in the summarized dataframe"

                        logger.info("Writing pivot file")
                        return summary_df

                    def write_fn(out_df, out_fp):
                        write_df(out_df, out_fp, do_overwrite=cfg.do_overwrite)

                    rwlock_wrap(
                        shard_fp,
                        pivot_fp,
                        read_fn,
                        write_fn,
                        compute_fn,
                        do_overwrite=cfg.do_overwrite,
                        do_return=False,
                    )


if __name__ == "__main__":
    summarize_ts_data_over_windows()
