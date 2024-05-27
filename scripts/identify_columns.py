#!/usr/bin/env python
"""This Python script, stores the configuration parameters and feature columns used in the output."""
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
    # create output dir
    flat_dir = Path(cfg.tabularized_data_dir)
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
