#!/usr/bin/env python

"""Aggregates time-series data for feature columns across different window sizes."""


from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from MEDS_tabular_automl.generate_summarized_reps import generate_summary
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
    flat_dir, _, feature_columns = setup_environment(cfg)

    # Assuming MEDS_cohort_dir is correctly defined somewhere above this snippet
    ts_dir = Path(cfg.tabularized_data_dir) / "ts"
    # TODO: Use patient splits here instead
    ts_fps = list(ts_dir.glob("*/*.parquet"))
    splits = {fp.parent.stem for fp in ts_fps}

    split_to_pair_fps = {}
    for split in splits:
        # Categorize files by identifier (base name without '_code' or '_value') using a list comprehension
        categorized_files = {
            file.stem.rsplit("_", 1)[0]: {"code": None, "value": None}
            for file in ts_fps
            if file.parent.stem == split
        }
        for file in ts_fps:
            if file.parent.stem == split:
                identifier = file.stem.rsplit("_", 1)[0]
                suffix = file.stem.split("_")[-1]  # 'code' or 'value'
                categorized_files[identifier][suffix] = file

        # Process categorized files into pairs ensuring code is first and value is second
        code_value_pairs = [
            (info["code"], info["value"])
            for info in categorized_files.values()
            if info["code"] is not None and info["value"] is not None
        ]

        split_to_pair_fps[split] = code_value_pairs

    # Summarize data and store
    summary_dir = flat_dir / "summary"
    for split, pairs in split_to_pair_fps.items():
        logger.info(f"Processing {split}:")
        for code_file, value_file in pairs:
            logger.info(f" - Code file: {code_file}, Value file: {value_file}")
            summary_df = generate_summary(
                feature_columns,
                [pl.scan_parquet(code_file), pl.scan_parquet(value_file)],
                cfg.window_sizes,
                cfg.aggs,
            )

            shard_number = code_file.stem.rsplit("_", 1)[0]
            write_df(summary_df, summary_dir / split / f"{shard_number}.parquet")


if __name__ == "__main__":
    summarize_ts_data_over_windows()
