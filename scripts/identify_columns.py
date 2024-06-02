#!/usr/bin/env python
"""This Python script, stores the configuration parameters and feature columns used in the output."""
import json
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_tabular_automl.file_name import FileNameResolver
from MEDS_tabular_automl.mapper import wrap as rwlock_wrap
from MEDS_tabular_automl.utils import compute_feature_frequencies, load_tqdm


def store_config_yaml(config_fp: Path, cfg: DictConfig):
    """Stores configuration parameters into a JSON file.

    This function writes a dictionary of parameters, which includes patient partitioning
    information and configuration details, to a specified JSON file.

    Args:
    - config_fp (Path): The file path for the JSON file where config should be stored.
    - cfg (DictConfig): A configuration object containing settings like the number of patients
      per sub-shard, minimum code inclusion frequency, and flags for updating or overwriting existing files.

    Behavior:
    - If config_fp exists and cfg.do_overwrite is False (without do_update being True), a
      FileExistsError is raised to prevent unintentional data loss.

    Raises:
    - ValueError: If there are discrepancies between old and new parameters during an update.
    - FileExistsError: If the file exists and overwriting is not allowed.

    Example:
    >>> cfg = DictConfig({
    ...     "n_patients_per_sub_shard": 100,
    ...     "min_code_inclusion_frequency": 5,
    ...     "do_overwrite": True,
    ... })
    >>> import tempfile
    >>> from pathlib import Path
    >>> with tempfile.NamedTemporaryFile() as temp_f:
    ...     config_fp = Path(temp_f.name)
    ...     store_config_yaml(config_fp, cfg)
    ...     assert config_fp.exists()
    ...     store_config_yaml(config_fp, cfg)
    ...     cfg.do_overwrite = False
    ...     try:
    ...         store_config_yaml(config_fp, cfg)
    ...     except FileExistsError as e:
    ...         print("FileExistsError Error Triggered")
    FileExistsError Error Triggered
    """
    if config_fp.exists():
        if not cfg.do_overwrite:
            raise FileExistsError(f"do_overwrite is {cfg.do_overwrite} and {config_fp} exists!")
    OmegaConf.save(cfg, config_fp)


@hydra.main(version_base=None, config_path="../configs", config_name="tabularize")
def store_columns(
    cfg: DictConfig,
):
    """Stores the configuration parameters and feature columns tabularized data we will be generated for.

    Args:
        cfg: The configuration object for the tabularization process.
    """
    iter_wrapper = load_tqdm(cfg.tqdm)
    # create output dir
    f_name_resolver = FileNameResolver(cfg)
    flat_dir = f_name_resolver.tabularize_dir
    flat_dir.mkdir(exist_ok=True, parents=True)

    # store params in json file
    config_fp = f_name_resolver.get_config_path()
    store_config_yaml(config_fp, cfg)

    # 0. Identify Output Columns and Frequencies
    logger.info("Iterating through shards and caching feature frequencies.")

    def compute_fn(shard_df):
        return compute_feature_frequencies(cfg, shard_df)

    def write_fn(data, out_fp):
        json.dump(data, open(out_fp, "w"))

    def read_fn(in_fp):
        return pl.scan_parquet(in_fp)

    # Map: Iterates through shards and caches feature frequencies
    train_shards = f_name_resolver.list_meds_files(split="train")
    np.random.shuffle(train_shards)
    feature_dir = f_name_resolver.tabularize_dir
    for shard_fp in iter_wrapper(train_shards):
        out_fp = feature_dir / "identify_train_columns" / f"{shard_fp.stem}.json"
        rwlock_wrap(
            shard_fp,
            out_fp,
            read_fn,
            write_fn,
            compute_fn,
            do_overwrite=cfg.do_overwrite,
            do_return=False,
        )

    logger.info("Summing frequency computations.")
    # Reduce: sum the frequency computations

    def compute_fn(feature_freq_list):
        feature_freqs = defaultdict(int)
        for shard_feature_freq in feature_freq_list:
            for feature, freq in shard_feature_freq.items():
                feature_freqs[feature] += freq
        return feature_freqs, sorted(list(feature_freqs.keys()))

    def write_fn(data, out_fp):
        feature_freqs, feature_columns = data
        json.dump(feature_columns, open(f_name_resolver.get_feature_columns_fp(), "w"))
        json.dump(feature_freqs, open(f_name_resolver.get_feature_freqs_fp(), "w"))

    def read_fn(feature_dir):
        files = list(feature_dir.glob("*.json"))
        return [json.load(open(fp)) for fp in files]

    rwlock_wrap(
        feature_dir / "identify_train_columns",
        feature_dir,
        read_fn,
        write_fn,
        compute_fn,
        do_overwrite=cfg.do_overwrite,
        do_return=False,
    )
    logger.info("Stored feature columns and frequencies.")


if __name__ == "__main__":
    store_columns()
