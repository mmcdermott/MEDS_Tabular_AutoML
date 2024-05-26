"""The base class for core dataset processing logic.

Attributes:
    INPUT_DF_T: This defines the type of the allowable input dataframes -- e.g., databases, filepaths,
        dataframes, etc.
    DF_T: This defines the type of internal dataframes -- e.g. polars DataFrames.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import DictConfig
from tqdm.auto import tqdm

from MEDS_tabular_automl.generate_static_features import get_flat_static_rep
from MEDS_tabular_automl.utils import get_flat_rep_feature_cols, write_df


def load_meds_data(MEDS_cohort_dir: str) -> Mapping[str, pl.DataFrame]:
    """Loads the MEDS dataset from disk.

    Args:
        MEDS_cohort_dir: The directory containing the MEDS datasets split by subfolders.
            We expect `train` to be a split so `MEDS_cohort_dir/train` should exist.

    Returns:
        Mapping[str, pl.DataFrame]: Mapping from split name to a polars DataFrame containing the MEDS dataset.
    """
    MEDS_cohort_dir = Path(MEDS_cohort_dir)
    meds_fps = list(MEDS_cohort_dir.glob("*/*.parquet"))
    splits = {fp.parent.stem for fp in meds_fps}
    assert "train" in splits, f"Expected 'train' split in {splits}."
    split_to_fps = {split: [fp for fp in meds_fps if fp.parent.stem == split] for split in splits}
    split_to_df = {
        split: pl.concat([pl.scan_parquet(fp) for fp in split_fps])
        for split, split_fps in split_to_fps.items()
    }
    return split_to_df


def store_params_json(params_fp: Path, cfg: DictConfig, sp_subjects: Mapping[str, Sequence[Sequence[int]]]):
    """Stores configuration parameters into a JSON file.

    This function writes a dictionary of parameters, which includes patient partitioning
    information and configuration details, to a specified JSON file. If the file already exists,
    the function can update it with new values depending on the configuration settings provided.

    Parameters:
    - params_fp (Path): The file path for the JSON file where parameters should be stored.
    - cfg (DictConfig): A configuration object containing settings like the number of patients
      per sub-shard, minimum code inclusion frequency, and flags for updating or overwriting existing files.
    - sp_subjects (Mapping[str, Sequence[Sequence[int]]]): A mapping of split names to sequences
      representing patient IDs, structured in sub-shards.

    Behavior:
    - If params_fp exists and cfg.do_update is True, the function checks for differences
      between existing and new parameters. If discrepancies are found, it will raise an error detailing
      the differences. The number of patients per sub-shard will be standardized to match the existing record.
    - If params_fp exists and cfg.do_overwrite is False (without do_update being True), a
      FileExistsError is raised to prevent unintentional data loss.

    Raises:
    - ValueError: If there are discrepancies between old and new parameters during an update.
    - FileExistsError: If the file exists and neither updating nor overwriting is allowed.

    Example:
    >>> cfg = DictConfig({
    >>>     "n_patients_per_sub_shard": 100,
    >>>     "min_code_inclusion_frequency": 5,
    >>>     "do_update": False,
    >>>     "do_overwrite": True
    >>> })
    >>> sp_subjects = {"train": [[1, 2, 3], [4, 5]], "test": [[6, 7]]}
    >>> params = store_params_json(Path("/path/to/params.json"), cfg, sp_subjects)
    """
    params = {
        "n_patients_per_sub_shard": cfg.n_patients_per_sub_shard,
        "min_code_inclusion_frequency": cfg.min_code_inclusion_frequency,
        "patient_shard_by_split": sp_subjects,
    }
    if params_fp.exists():
        if cfg.do_update:
            with open(params_fp) as f:
                old_params = json.load(f)

            if old_params["n_patients_per_sub_shard"] != params["n_patients_per_sub_shard"]:
                print(
                    "Standardizing chunk size to existing record "
                    f"({old_params['n_patients_per_sub_shard']})."
                )
                params["n_patients_per_sub_shard"] = old_params["n_patients_per_sub_shard"]
                params["patient_shard_by_split"] = old_params["patient_shard_by_split"]

            if old_params != params:
                err_strings = ["Asked to update but parameters differ:"]
                old = set(old_params.keys())
                new = set(params.keys())
                if old != new:
                    err_strings.append("Keys differ: ")
                    if old - new:
                        err_strings.append(f"  old - new = {old - new}")
                    if new - old:
                        err_strings.append(f"  new - old = {old - new}")

                for k in old & new:
                    old_val = old_params[k]
                    new_val = params[k]

                    if old_val != new_val:
                        err_strings.append(f"Values differ for {k}:")
                        err_strings.append(f"  Old: {old_val}")
                        err_strings.append(f"  New: {new_val}")

                raise ValueError("\n".join(err_strings))
        elif not cfg.do_overwrite:
            raise FileExistsError(f"do_overwrite is {cfg.do_overwrite} and {params_fp} exists!")
    with open(params_fp, mode="w") as f:
        json.dump(params, f)
    return params


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
    # setup rng seed
    rng = np.random.default_rng(cfg.seed)

    # create output dir
    flat_dir = Path(cfg.tabularized_data_dir) / "flat_reps"
    flat_dir.mkdir(exist_ok=True, parents=True)

    # load MEDS data
    split_to_df = load_meds_data(cfg.MEDS_cohort_dir)

    # for every dataset split, create shards to output flat representations to
    sp_subjects = {}
    sp_dfs = {}
    for split_name, split_df in split_to_df.items():
        split_patient_ids = (
            split_df.select(pl.col("patient_id").cast(pl.Int32).unique()).collect().to_series().to_list()
        )
        print(len(split_patient_ids))
        if cfg.n_patients_per_sub_shard is None:
            sp_subjects[split_name] = split_patient_ids
            sp_dfs[split_name] = [split_df]
        else:
            shuffled_patient_ids = rng.permutation(split_patient_ids)
            num_shards = max(len(split_patient_ids) // cfg.n_patients_per_sub_shard, 1)  # must be 1 or larger
            sharded_patient_ids = np.array_split(shuffled_patient_ids, num_shards)
            sp_subjects[split_name] = [shard.tolist() for shard in sharded_patient_ids]
            sp_dfs[split_name] = [
                split_df.filter(pl.col("patient_id").is_in(set(shard))) for shard in sharded_patient_ids
            ]

    # store params in json file
    params_fp = flat_dir / "params.json"
    store_params_json(params_fp, cfg, sp_subjects)

    # 0. Identify Output Columns
    # We set window_sizes to None here because we want to get the feature column names for the raw flat
    # representation, not the summarized one.
    feature_columns, code_properties = get_flat_rep_feature_cols(cfg, sp_dfs)

    # 1. Produce static representation
    static_subdir = flat_dir / "static"

    static_dfs = {}
    actual_num_patients = 0
    for sp, subjects_dfs in tqdm(list(sp_dfs.items()), desc="Flattening Splits"):
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
    expected_num_patients = sum(len(ids) for split_ids in sp_subjects.values() for ids in split_ids)
    assert (
        actual_num_patients == expected_num_patients
    ), f"Expected {expected_num_patients} patients, got {actual_num_patients}."

    # # 2. Produce raw representation
    # ts_subdir = flat_dir / "at_ts"
    # import pdb; pdb.set_trace()

    # ts_dfs = {}
    # for sp, subjects_dfs in tqdm(list(sp_dfs.items()), desc="Flattening Splits"):
    #     ts_dfs[sp] = []
    #     sp_dir = ts_subdir / sp

    #     for i, shard_df in enumerate(tqdm(subjects_dfs, desc="Subject chunks", leave=False)):
    #         fp = sp_dir / f"{i}.parquet"
    #         ts_dfs[sp].append(fp)
    #         if fp.exists():
    #             if cfg.do_update:
    #                 continue
    #             elif not cfg.do_overwrite:
    #                 raise FileExistsError(f"do_overwrite is {cfg.do_overwrite} and {fp} exists!")

    #         df = get_flat_ts_rep(
    #             feature_columns=feature_columns,
    #             shard_df=shard_df,
    #         )

    #         write_df(df, fp, do_overwrite=cfg.do_overwrite)

    # if cfg.window_sizes is None:
    #     return

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
