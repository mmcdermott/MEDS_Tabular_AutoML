"""This Python script, stores the configuration parameters and feature columns used in the output."""

import logging
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from MEDS_transforms.mapreduce.rwlock import rwlock_wrap
from omegaconf import DictConfig

from .. import DESCRIBE_CODES_CFG
from ..describe_codes import (
    compute_feature_frequencies,
    convert_to_df,
    convert_to_freq_dict,
)
from ..file_name import list_subdir_files
from ..utils import get_shard_prefix, load_tqdm, write_df

logger = logging.getLogger(__name__)


def get_training_shards(input_dir: Path | str) -> tuple[list[Path], pl.LazyFrame | None]:
    """Find data shards and an optional subject filter for train-only fitting.

    Split-sharded MEDS datasets are handled without reading split metadata. For
    unsharded datasets, ``metadata/subject_splits.parquet`` is used to restrict
    every shard to training subjects.
    """
    input_dir = Path(input_dir)
    train_dir = input_dir / "train"
    if train_dir.is_dir():
        train_shards = list_subdir_files(train_dir, "parquet")
        if not train_shards:
            raise ValueError(f"Training split directory contains no Parquet shards: {train_dir}")
        return train_shards, None

    subject_splits_candidates = (
        input_dir / "metadata" / "subject_splits.parquet",
        input_dir.parent / "metadata" / "subject_splits.parquet",
    )
    subject_splits_fp = next((fp for fp in subject_splits_candidates if fp.is_file()), None)
    if subject_splits_fp is None:
        raise ValueError(
            "meds-tab-describe requires train-only data when fitting code metadata, but no "
            f"train/ directory or metadata/subject_splits.parquet was found for {input_dir}."
        )

    data_shards = [
        fp
        for fp in list_subdir_files(input_dir, "parquet")
        if fp != subject_splits_fp and "metadata" not in fp.relative_to(input_dir).parts
    ]
    if not data_shards:
        raise ValueError(f"No MEDS data Parquet shards found under {input_dir}")

    subject_splits = pl.scan_parquet(subject_splits_fp)
    required_columns = {"subject_id", "split"}
    actual_columns = set(subject_splits.collect_schema().names())
    if not required_columns.issubset(actual_columns):
        raise ValueError(
            f"{subject_splits_fp} must contain columns {sorted(required_columns)}, "
            f"but has {sorted(actual_columns)}."
        )
    train_subjects = subject_splits.filter(pl.col("split") == "train").select("subject_id").unique()
    if train_subjects.select(pl.len()).collect().item() == 0:
        raise ValueError(f"No training subjects found in {subject_splits_fp}")
    return data_shards, train_subjects


@hydra.main(
    version_base=None, config_path=str(DESCRIBE_CODES_CFG.parent), config_name=DESCRIBE_CODES_CFG.stem
)
def main(cfg: DictConfig):
    """Computes feature frequencies and stores them to disk.

    Args:
        cfg: The configuration object for the tabularization process, loaded from a Hydra
            YAML configuration file.
    """
    iter_wrapper = load_tqdm(cfg.tqdm)

    # 0. Identify Output Columns and Frequencies
    logger.info("Iterating through shards and caching feature frequencies.")

    # Map: Iterates through shards and caches feature frequencies
    train_shards, train_subjects = get_training_shards(cfg.input_dir)
    np.random.shuffle(train_shards)
    train_cache_files = []

    def compute_train_feature_frequencies(shard_df):
        if train_subjects is not None:
            shard_df = shard_df.join(train_subjects, on="subject_id", how="semi")
        return compute_feature_frequencies(shard_df)

    for shard_fp in iter_wrapper(train_shards):
        out_fp = (Path(cfg.cache_dir) / "train_only" / get_shard_prefix(cfg.input_dir, shard_fp)).with_suffix(
            shard_fp.suffix
        )
        train_cache_files.append(out_fp)

        rwlock_wrap(
            shard_fp,
            out_fp,
            pl.scan_parquet,
            write_df,
            compute_train_feature_frequencies,
            do_overwrite=cfg.do_overwrite,
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

    def read_fn(_feature_dir):
        return [pl.scan_parquet(fp) for fp in train_cache_files]

    rwlock_wrap(
        Path(cfg.cache_dir),
        Path(cfg.output_filepath),
        read_fn,
        write_df,
        compute_fn,
        do_overwrite=cfg.do_overwrite,
    )
    logger.info("Stored feature columns and frequencies.")
