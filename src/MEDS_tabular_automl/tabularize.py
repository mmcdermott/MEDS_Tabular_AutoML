"""The base class for core dataset processing logic.

Attributes:
    INPUT_DF_T: This defines the type of the allowable input dataframes -- e.g., databases, filepaths,
        dataframes, etc.
    DF_T: This defines the type of internal dataframes -- e.g. polars DataFrames.
"""
from collections.abc import Mapping
from pathlib import Path

import polars as pl
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from MEDS_tabular_automl.generate_static_features import get_flat_static_rep
from MEDS_tabular_automl.generate_ts_features import get_flat_ts_rep
from MEDS_tabular_automl.utils import get_flat_rep_feature_cols, write_df


def load_meds_data(MEDS_cohort_dir: str) -> Mapping[str, pl.DataFrame]:
    """Loads the MEDS dataset from disk.

    Args:
        MEDS_cohort_dir: The directory containing the MEDS datasets split by subfolders.
            We expect `train` to be a split so `MEDS_cohort_dir/train` should exist.

    Returns:
        Mapping[str, pl.DataFrame]: Mapping from split name to a polars DataFrame containing the MEDS dataset.

    Example:
    >>> import tempfile
    >>> from pathlib import Path
    >>> MEDS_cohort_dir = Path(tempfile.mkdtemp())
    >>> for split in ["train", "val", "test"]:
    ...     split_dir = MEDS_cohort_dir / split
    ...     split_dir.mkdir()
    ...     pl.DataFrame({"patient_id": [1, 2, 3]}).write_parquet(split_dir / "data.parquet")
    >>> split_to_df = load_meds_data(MEDS_cohort_dir)
    >>> assert "train" in split_to_df
    >>> assert len(split_to_df) == 3
    >>> assert len(split_to_df["train"]) == 1
    >>> assert isinstance(split_to_df["train"][0], pl.LazyFrame)
    """
    MEDS_cohort_dir = Path(MEDS_cohort_dir)
    meds_fps = list(MEDS_cohort_dir.glob("*/*.parquet"))
    splits = {fp.parent.stem for fp in meds_fps}
    split_to_fps = {split: [fp for fp in meds_fps if fp.parent.stem == split] for split in splits}
    split_to_df = {
        split: [pl.scan_parquet(fp) for fp in split_fps] for split, split_fps in split_to_fps.items()
    }
    return split_to_df


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


def cache_flat_representation(
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
    config_fp = flat_dir / "config.json"
    store_config_yaml(config_fp, cfg)

    # 0. Identify Output Columns
    # We set window_sizes to None here because we want to get the feature column names for the raw flat
    # representation, not the summarized one.
    feature_columns = set()
    for shard_df in split_to_df["train"]:
        feature_columns.update(get_flat_rep_feature_cols(cfg, shard_df))
    feature_columns = sorted(list(feature_columns))

    # 1. Produce static representation
    static_subdir = flat_dir / "static"

    static_dfs = {}
    actual_num_patients = 0
    for sp, subjects_dfs in tqdm(list(split_to_df.items()), desc="Flattening Splits"):
        static_dfs[sp] = []
        sp_dir = static_subdir / sp

        for i, shard_df in enumerate(tqdm(subjects_dfs, desc="Subject chunks", leave=False)):
            fp = sp_dir / f"{i}.parquet"
            static_dfs[sp].append(fp)
            if fp.exists():
                if cfg.do_update:
                    continue
                elif not cfg.do_overwrite:
                    raise FileExistsError(f"do_overwrite is {cfg.do_overwrite} and {fp} exists!")

            df = get_flat_static_rep(
                feature_columns=feature_columns,
                shard_df=shard_df,
            )

            write_df(df, fp, do_overwrite=cfg.do_overwrite)
            actual_num_patients += df.shape[0]
    # expected_num_patients = sum(len(ids) for split_ids in sp_subjects.values() for ids in split_ids)
    # assert (
    #     actual_num_patients == expected_num_patients
    # ), f"Expected {expected_num_patients} patients, got {actual_num_patients}."

    # 2. Produce raw representation
    ts_subdir = flat_dir / "at_ts"

    ts_dfs = {}
    for sp, subjects_dfs in tqdm(list(split_to_df.items()), desc="Flattening Splits"):
        ts_dfs[sp] = []
        sp_dir = ts_subdir / sp

        for i, shard_df in enumerate(tqdm(subjects_dfs, desc="Subject chunks", leave=False)):
            fp = sp_dir / f"{i}.parquet"
            ts_dfs[sp].append(fp)
            if fp.exists():
                if cfg.do_update:
                    continue
                elif not cfg.do_overwrite:
                    raise FileExistsError(f"do_overwrite is {cfg.do_overwrite} and {fp} exists!")

            df = get_flat_ts_rep(
                feature_columns=feature_columns,
                shard_df=shard_df,
            )

            write_df(df, fp, do_overwrite=cfg.do_overwrite)

    if cfg.window_sizes is None:
        return

    # # 3. Produce summarized history representations
    # history_subdir = flat_dir / "over_history"

    # for window_size in tqdm(cfg.window_sizes, desc="History window sizes"):
    #     for sp, df_fps in tqdm(list(ts_dfs.items()), desc="Windowing Splits", leave=False):
    #         for i, df_fp in enumerate(tqdm(df_fps, desc="Subject chunks", leave=False)):
    #             fp = history_subdir / sp / window_size / f"{i}.parquet"
    #             if fp.exists():
    #                 if cfg.do_update:
    #                     continue
    #                 elif not cfg.do_overwrite:
    #                     raise FileExistsError(f"do_overwrite is {cfg.do_overwrite} and {fp} exists!")

    #             df = _summarize_over_window(df_fp, window_size)
    #             write_df(df, fp)
