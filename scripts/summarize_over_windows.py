#!/usr/bin/env python

"""Aggregates time-series data for feature columns across different window sizes."""
import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from MEDS_tabular_automl.generate_summarized_reps import generate_summary
from MEDS_tabular_automl.generate_ts_features import get_flat_ts_rep
from MEDS_tabular_automl.utils import setup_environment, write_df


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
    flat_dir, split_to_df, feature_columns = setup_environment(cfg)
    # Produce ts representation
    ts_subdir = flat_dir / "ts"

    for sp, subjects_dfs in split_to_df.items():
        sp_dir = ts_subdir / sp
        if sp != "train":
            continue

        for i, shard_df in enumerate(subjects_dfs):
            pivot_fp = sp_dir / f"{i}.parquet"
            if pivot_fp.exists() and not cfg.do_overwrite:
                raise FileExistsError(f"do_overwrite is {cfg.do_overwrite} and {pivot_fp.exists()} exists!")
            if sp != "train":
                # remove codes not in training set
                shard_df = shard_df.filter(pl.col("code").is_in(feature_columns))

            # Load Sparse DataFrame
            pivot_df = get_flat_ts_rep(
                feature_columns=feature_columns,
                shard_df=shard_df,
            )

            # Summarize data -- applying aggregations on various window sizes
            summary_df = generate_summary(
                feature_columns,
                pivot_df,
                cfg.window_sizes,
                cfg.aggs,
            )

            logger.info("Writing pivot file")
            write_df(summary_df, pivot_fp, do_overwrite=cfg.do_overwrite)


if __name__ == "__main__":
    summarize_ts_data_over_windows()
