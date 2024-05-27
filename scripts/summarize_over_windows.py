"""WIP."""


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
    """Writes a flat (historically summarized) representation of the dataset to disk.

    This file caches a set of files useful for building flat representations of the dataset to disk,
    suitable for, e.g., sklearn style modeling for downstream tasks. It will produce a few sets of files:

    * A new directory ``self.config.save_dir / "flat_reps"`` which contains the following:
    * A subdirectory ``raw`` which contains: (1) a json file with the configuration arguments and (2) a
        set of parquet files containing flat (e.g., wide) representations of summarized events per subject,
        broken out by split and subject chunk.
    * A set of subdirectories ``past/*`` which contains summarized views over the past ``*`` time period
        per subject per event, for all time periods in ``window_sizes``, if any.

    Args:
        cfg:
            MEDS_cohort_dir: directory of MEDS format dataset that is ingested.
            tabularized_data_dir: output directory of tabularized data.
            min_code_inclusion_frequency: The base feature inclusion frequency that should be used to dictate
                what features can be included in the flat representation. It can either be a float, in which
                case it applies across all measurements, or `None`, in which case no filtering is applied, or
                a dictionary from measurement type to a float dictating a per-measurement-type inclusion
                cutoff.
            window_sizes: Beyond writing out a raw, per-event flattened representation, the dataset also has
                the capability to summarize these flattened representations over the historical windows
                specified in this argument. These are strings specifying time deltas, using this syntax:
                `link`_. Each window size will be summarized to a separate directory, and will share the same
                subject file split as is used in the raw representation files.
            codes: A list of codes to include in the flat representation. If `None`, all codes will be included
                in the flat representation.
            aggs: A list of aggregations to apply to the raw representation. Must have length greater than 0.
            n_patients_per_sub_shard: The number of subjects that should be included in each output file.
                Lowering this number increases the number of files written, making the process of creating and
                leveraging these files slower but more memory efficient.
            do_overwrite: If `True`, this function will overwrite the data already stored in the target save
                directory.
            do_update: bool = True
            seed: The seed to use for random number generation.

    .. _link: https://pola-rs.github.io/polars/py-polars/html/reference/dataframe/api/polars.DataFrame.groupby_rolling.html # noqa: E501
    """
    flat_dir, _, feature_columns = setup_environment(cfg)

    # Assuming MEDS_cohort_dir is correctly defined somewhere above this snippet
    ts_dir = Path(cfg.tabularized_data_dir) / "ts"
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

    # Example use of split_to_pair_fps
    for split, pairs in split_to_pair_fps.items():
        logger.info(f"Processing {split}:")
        for code_file, value_file in pairs:
            logger.info(f" - Code file: {code_file}, Value file: {value_file}")
            summary_df = generate_summary(pl.scan_parquet(code_file), pl.scan_parquet(value_file))
            shard_number = code_file.stem.rsplit("_", 1)[0]
            write_df(summary_df, flat_dir / split / f"{shard_number}.parquet")
