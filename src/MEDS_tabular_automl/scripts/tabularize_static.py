#!/usr/bin/env python
"""Tabularizes static data in MEDS format into tabular representations."""

from itertools import product
from pathlib import Path

import hydra
import numpy as np
import polars as pl

pl.enable_string_cache()

from importlib.resources import files

from omegaconf import DictConfig

from ..describe_codes import (
    convert_to_df,
    filter_parquet,
    get_feature_columns,
    get_feature_freqs,
)
from ..file_name import list_subdir_files
from ..generate_static_features import get_flat_static_rep
from ..mapper import wrap as rwlock_wrap
from ..utils import (
    STATIC_CODE_AGGREGATION,
    STATIC_VALUE_AGGREGATION,
    filter_to_codes,
    get_shard_prefix,
    hydra_loguru_init,
    load_tqdm,
    stage_init,
    write_df,
)

config_yaml = files("MEDS_tabular_automl").joinpath("configs/tabularization.yaml")
if not config_yaml.is_file():
    raise FileNotFoundError("Core configuration not successfully installed!")


@hydra.main(version_base=None, config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(
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
            input_dir: directory of MEDS format dataset that is ingested.
            tabularized_data_dir: output directory of tabularized data.
            min_code_inclusion_count: The base feature inclusion count that should be used to dictate
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
    stage_init(
        cfg,
        [
            "input_code_metadata_fp",
            "input_dir",
            "tabularization.filtered_code_metadata_fp",
        ],
    )

    if cfg.input_label_dir:
        if not Path(cfg.input_label_dir).is_dir():
            raise ValueError(f"input_label_dir: {cfg.input_label_dir} is not a directory.")

    iter_wrapper = load_tqdm(cfg.tqdm)
    if not cfg.loguru_init:
        hydra_loguru_init()

    # Step 1: Cache the filtered features that will be used in the tabularization process and modeling
    def read_fn(_):
        return _

    def compute_fn(_):
        filtered_feature_columns = filter_to_codes(
            cfg.input_code_metadata_fp,
            cfg.tabularization.allowed_codes,
            cfg.tabularization.min_code_inclusion_count,
            cfg.tabularization.min_code_inclusion_frequency,
            cfg.tabularization.max_included_codes,
        )
        feature_freqs = get_feature_freqs(cfg.input_code_metadata_fp)
        filtered_feature_columns_set = set(filtered_feature_columns)
        filtered_feature_freqs = {
            code: count for code, count in feature_freqs.items() if code in filtered_feature_columns_set
        }
        return convert_to_df(filtered_feature_freqs)

    def write_fn(data, out_fp):
        data.write_parquet(out_fp)

    in_fp = Path(cfg.input_code_metadata_fp)
    out_fp = Path(cfg.tabularization.filtered_code_metadata_fp)
    rwlock_wrap(
        in_fp,
        out_fp,
        read_fn,
        write_fn,
        compute_fn,
        do_overwrite=cfg.do_overwrite,
        do_return=False,
    )

    # Step 2: Produce static data representation
    meds_shard_fps = list_subdir_files(cfg.input_dir, "parquet")
    feature_columns = get_feature_columns(cfg.tabularization.filtered_code_metadata_fp)

    # shuffle tasks
    aggs = cfg.tabularization.aggs
    static_aggs = [agg for agg in aggs if agg in [STATIC_CODE_AGGREGATION, STATIC_VALUE_AGGREGATION]]
    tabularization_tasks = list(product(meds_shard_fps, static_aggs))
    np.random.shuffle(tabularization_tasks)
    for shard_fp, agg in iter_wrapper(tabularization_tasks):
        if cfg.input_label_dir:
            label_fp = Path(cfg.input_label_dir) / shard_fp.relative_to(shard_fp.parents[1])
            label_df = pl.scan_parquet(label_fp)
        else:
            label_df = None
        out_fp = (
            Path(cfg.output_tabularized_dir) / get_shard_prefix(cfg.input_dir, shard_fp) / "none" / agg
        ).with_suffix(".npz")

        def read_fn(in_fp):
            return filter_parquet(in_fp, cfg.tabularization._resolved_codes)

        def compute_fn(shard_df):
            return get_flat_static_rep(
                agg=agg,
                feature_columns=feature_columns,
                shard_df=shard_df,
                label_df=label_df,
            )

        def write_fn(data, out_df):
            write_df(data, out_df, do_overwrite=cfg.do_overwrite)

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
