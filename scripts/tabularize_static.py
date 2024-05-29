#!/usr/bin/env python
"""Tabularizes static data in MEDS format into tabular representations."""

from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig, OmegaConf

from MEDS_tabular_automl.generate_static_features import get_flat_static_rep
from MEDS_tabular_automl.utils import setup_environment, write_df

pl.enable_string_cache()


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
    - FileExistsError: If the file exists and neither updating nor overwriting is allowed.

    Example:
    >>> cfg = DictConfig({
    ...     "n_patients_per_sub_shard": 100,
    ...     "min_code_inclusion_frequency": 5,
    ...     "do_update": False,
    ...     "do_overwrite": True
    ... })
    >>> import tempfile
    >>> from pathlib import Path
    >>> with tempfile.TemporaryDirectory() as d:
    ...     config_fp = Path(d) / "config.yaml"
    ...     store_config_yaml(config_fp, cfg)
    ...     assert config_fp.exists()
    """
    if config_fp.exists():
        if not cfg.do_overwrite:
            raise FileExistsError(f"do_overwrite is {cfg.do_overwrite} and {config_fp} exists!")
    OmegaConf.save(cfg, config_fp)


@hydra.main(version_base=None, config_path="../configs", config_name="tabularize")
def tabularize_static_data(
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
    flat_dir, split_to_df, feature_columns = setup_environment(cfg)

    # Produce static representation
    static_subdir = flat_dir / "static"

    static_dfs = {}
    for sp, subjects_dfs in split_to_df.items():
        static_dfs[sp] = []
        sp_dir = static_subdir / sp

        for i, shard_df in enumerate(subjects_dfs):
            fp = sp_dir / f"{i}.parquet"
            static_dfs[sp].append(fp)
            if fp.exists() and not cfg.do_overwrite:
                raise FileExistsError(f"do_overwrite is {cfg.do_overwrite} and {fp} exists!")

            df = get_flat_static_rep(
                feature_columns=feature_columns,
                shard_df=shard_df,
            )

            write_df(df, fp, do_overwrite=cfg.do_overwrite, pandas=True)


if __name__ == "__main__":
    tabularize_static_data()
