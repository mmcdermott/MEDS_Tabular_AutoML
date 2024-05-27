"""This Python script, utilizing the Hydra and Polars libraries, automates the creation of flat
representations of medical datasets for machine learning modeling.

It includes functions to store configuration parameters in a JSON file and write summarized dataset
representations to disk based on configurable parameters such as inclusion frequencies and historical window
sizes. The script ensures data integrity through conditional checks on overwriting and updating existing
files, and enhances traceability by recording configuration details and feature columns used in the output.
"""
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from MEDS_tabular_automl.utils import get_flat_rep_feature_cols, load_meds_data


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
def store_columns(
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
    # create output dir
    flat_dir = Path(cfg.tabularized_data_dir) / "flat_reps"
    flat_dir.mkdir(exist_ok=True, parents=True)

    # load MEDS data
    split_to_df = load_meds_data(cfg.MEDS_cohort_dir)

    # store params in json file
    config_fp = flat_dir / "config.yaml"
    store_config_yaml(config_fp, cfg)

    # 0. Identify Output Columns
    # We set window_sizes to None here because we want to get the feature column names for the raw flat
    # representation, not the summarized one.
    feature_columns = set()
    for shard_df in split_to_df["train"]:
        feature_columns.update(get_flat_rep_feature_cols(cfg, shard_df))
    feature_columns = sorted(list(feature_columns))
    json.dump(feature_columns, open(flat_dir / "feature_columns.json", "w"))
