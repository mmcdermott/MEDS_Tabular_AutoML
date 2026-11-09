"""This Python script, stores the configuration parameters and feature columns used in the output."""

import logging
from collections import defaultdict
from pathlib import Path

import hydra
import meds
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


def get_training_shards(cfg: DictConfig) -> tuple[list[Path], pl.LazyFrame]:
    """Find data shards and the training subjects to restrict them to.

    The split assignment is always read from ``meds.subject_splits_filepath`` under the MEDS root directory
    (``cfg.meds_root_dir``, defaulting to the parent of ``cfg.input_dir``). Shard directory names are not used
    to decide which subjects are in the training split.
    """
    input_dir = Path(cfg.input_dir)
    meds_root_dir = Path(cfg.meds_root_dir) if cfg.get("meds_root_dir") else input_dir.parent
    subject_splits_fp = meds_root_dir / meds.subject_splits_filepath
    if not subject_splits_fp.is_file():
        raise FileNotFoundError(
            "meds-tab-describe fits code metadata on training subjects only and requires the subject "
            f"splits file, but it was not found at {subject_splits_fp}."
        )

    data_shards = [
        fp
        for fp in list_subdir_files(input_dir, "parquet")
        if "metadata" not in fp.relative_to(input_dir).parts
    ]
    if not data_shards:
        raise ValueError(f"No MEDS data Parquet shards found under {input_dir}")

    subject_id_col = meds.SubjectSplitSchema.subject_id_name
    split_col = meds.SubjectSplitSchema.split_name

    subject_splits = pl.scan_parquet(subject_splits_fp)
    required_columns = {subject_id_col, split_col}
    actual_columns = set(subject_splits.collect_schema().names())
    if not required_columns.issubset(actual_columns):
        raise ValueError(
            f"{subject_splits_fp} must contain columns {sorted(required_columns)}, "
            f"but has {sorted(actual_columns)}."
        )
    train_subjects = (
        subject_splits.filter(pl.col(split_col) == meds.train_split).select(subject_id_col).unique()
    )
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
    train_shards, train_subjects = get_training_shards(cfg)
    np.random.shuffle(train_shards)
    train_cache_files = []

    def compute_train_feature_frequencies(shard_df):
        shard_df = shard_df.join(train_subjects, on=meds.SubjectSplitSchema.subject_id_name, how="semi")
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
