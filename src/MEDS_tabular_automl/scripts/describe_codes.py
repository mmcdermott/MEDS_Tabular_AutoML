#!/usr/bin/env python
"""This Python script, stores the configuration parameters and feature columns used in the output."""
from collections import defaultdict
from importlib.resources import files
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from ..describe_codes import (
    compute_feature_frequencies,
    convert_to_df,
    convert_to_freq_dict,
)
from ..file_name import list_subdir_files
from ..mapper import wrap as rwlock_wrap
from ..utils import get_shard_prefix, hydra_loguru_init, load_tqdm, stage_init, write_df

config_yaml = files("MEDS_tabular_automl").joinpath("configs/describe_codes.yaml")
if not config_yaml.is_file():
    raise FileNotFoundError("Core configuration not successfully installed!")


@hydra.main(version_base=None, config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig):
    """Computes feature frequencies and stores them to disk.

    Args:
        cfg: The configuration object for the tabularization process, loaded from a Hydra
            YAML configuration file.
    """
    stage_init(cfg, ["input_dir"])
    iter_wrapper = load_tqdm(cfg.tqdm)
    if not cfg.loguru_init:
        hydra_loguru_init()

    # 0. Identify Output Columns and Frequencies
    logger.info("Iterating through shards and caching feature frequencies.")

    def write_fn(df, out_fp):
        write_df(df, out_fp)

    def read_fn(in_fp):
        return pl.scan_parquet(in_fp)

    # Map: Iterates through shards and caches feature frequencies
    train_shards = list_subdir_files(cfg.input_dir, "parquet")
    np.random.shuffle(train_shards)
    for shard_fp in iter_wrapper(train_shards):
        out_fp = (Path(cfg.cache_dir) / get_shard_prefix(cfg.input_dir, shard_fp)).with_suffix(
            shard_fp.suffix
        )
        rwlock_wrap(
            shard_fp,
            out_fp,
            read_fn,
            write_fn,
            compute_feature_frequencies,
            do_overwrite=cfg.do_overwrite,
            do_return=False,
        )

    logger.info("Summing frequency computations.")
    # Reduce: sum the frequency computations

    def compute_fn(freq_df_list):
        feature_freqs = defaultdict(int)
        for shard_freq_df in freq_df_list:
            shard_freq_dict = convert_to_freq_dict(shard_freq_df)
            for feature, freq in shard_freq_dict.items():
                feature_freqs[feature] += freq
        feature_df = convert_to_df(feature_freqs)
        return feature_df

    def write_fn(df, out_fp):
        write_df(df, out_fp)

    def read_fn(feature_dir):
        files = list_subdir_files(feature_dir, "parquet")
        return [pl.scan_parquet(fp) for fp in files]

    rwlock_wrap(
        Path(cfg.cache_dir),
        Path(cfg.output_filepath),
        read_fn,
        write_fn,
        compute_fn,
        do_overwrite=cfg.do_overwrite,
        do_return=False,
    )
    logger.info("Stored feature columns and frequencies.")


if __name__ == "__main__":
    main()
