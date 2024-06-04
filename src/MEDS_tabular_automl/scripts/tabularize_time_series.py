#!/usr/bin/env python

"""Aggregates time-series data for feature columns across different window sizes."""
import polars as pl
pl.enable_string_cache()

from itertools import product
from pathlib import Path
import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig



from MEDS_tabular_automl.describe_codes import filter_parquet, get_feature_columns
from MEDS_tabular_automl.file_name import list_subdir_files
from MEDS_tabular_automl.generate_summarized_reps import generate_summary
from MEDS_tabular_automl.generate_ts_features import get_flat_ts_rep
from MEDS_tabular_automl.mapper import wrap as rwlock_wrap
from MEDS_tabular_automl.utils import (
    STATIC_CODE_AGGREGATION,
    STATIC_VALUE_AGGREGATION,
    get_shard_prefix,
    hydra_loguru_init,
    load_tqdm,
    write_df,
)



@hydra.main(version_base=None, config_path="../configs", config_name="tabularization")
def main(
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
    iter_wrapper = load_tqdm(cfg.tqdm)
    if not cfg.loguru_init:
        hydra_loguru_init()
    # Produce ts representation
    meds_shard_fps = list_subdir_files(cfg.input_dir, "parquet")
    feature_columns = get_feature_columns(cfg.input_code_metadata)

    # shuffle tasks
    aggs = [
        agg
        for agg in cfg.tabularization.aggs
        if agg not in [STATIC_CODE_AGGREGATION, STATIC_VALUE_AGGREGATION]
    ]
    tabularization_tasks = list(product(meds_shard_fps, cfg.tabularization.window_sizes, aggs))
    np.random.shuffle(tabularization_tasks)

    # iterate through them
    for shard_fp, window_size, agg in iter_wrapper(tabularization_tasks):
        out_fp = (
            Path(cfg.output_dir) / get_shard_prefix(cfg.input_dir, shard_fp) / window_size / agg
        ).with_suffix(".npz")

        def read_fn(in_fp):
            return filter_parquet(in_fp, cfg.tabularization._resolved_codes)

        def compute_fn(shard_df):
            # Load Sparse DataFrame
            index_df, sparse_matrix = get_flat_ts_rep(agg, feature_columns, shard_df)

            # Summarize data -- applying aggregations on a specific window size + aggregation combination
            summary_df = generate_summary(
                feature_columns,
                index_df,
                sparse_matrix,
                window_size,
                agg,
            )
            assert summary_df.shape[1] > 0, "No data found in the summarized dataframe"

            logger.info("Writing pivot file")
            return summary_df

        def write_fn(out_matrix, out_fp):
            coo_matrix = out_matrix.tocoo()
            write_df(coo_matrix, out_fp, do_overwrite=cfg.do_overwrite)

        rwlock_wrap(
            shard_fp,
            out_fp,
            read_fn,
            write_fn,
            compute_fn,
            do_overwrite=cfg.do_overwrite,
            do_return=False,
        )


if __name__ == "__main__":
    main()
