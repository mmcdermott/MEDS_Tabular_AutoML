"""The base class for core dataset processing logic.

Attributes:
    INPUT_DF_T: This defines the type of the allowable input dataframes -- e.g., databases, filepaths,
        dataframes, etc.
    DF_T: This defines the type of internal dataframes -- e.g. polars DataFrames.
"""

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs
from omegaconf import DictConfig
from tqdm.auto import tqdm

DF_T = pl.DataFrame


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


def _write_df(cls, df: DF_T, fp: Path, **kwargs):
    """Write shard to disk."""
    do_overwrite = kwargs.get("do_overwrite", False)

    if not do_overwrite and fp.is_file():
        raise FileExistsError(f"{fp} exists and do_overwrite is {do_overwrite}!")

    fp.parent.mkdir(exist_ok=True, parents=True)

    if isinstance(df, pl.LazyFrame):
        df.collect().write_parquet(fp, use_pyarrow=cls.WRITE_USE_PYARROW)
    else:
        df.write_parquet(fp, use_pyarrow=cls.WRITE_USE_PYARROW)


def _get_flat_col_dtype(self, col: str) -> pl.DataType:
    """Gets the appropriate minimal dtype for the given flat representation column string."""

    parts = col.split("/")
    if len(parts) < 4:
        raise ValueError(f"Malformed column {col}. Should be temporal/measurement/feature/agg")

    temp, meas = parts[0], parts[1]
    agg = parts[-1]
    feature = "/".join(parts[2:-1])

    cfg = self.measurement_configs[meas]

    match agg:
        case "sum" | "sum_sqd" | "min" | "max" | "value":
            return pl.Float32
        case "present":
            return pl.Boolean
        case "count" | "has_values_count":
            # config.observation_rate_over_cases = total_observed / total_possible
            # config.observation_rate_per_case = raw_total_observed / total_observed

            match temp:
                case "static":
                    n_possible = len(self.subject_ids)
                case str() | "dynamic":
                    n_possible = sum(self.n_events_per_subject.values())
                case _:
                    raise ValueError(
                        f"Column name {col} malformed: Temporality {temp} not in `static` or `dynamic"
                    )

            if cfg.vocabulary is None:
                observation_frequency = cfg.observation_rate_per_case * cfg.observation_rate_over_cases
            else:
                if feature not in cfg.vocabulary.idxmap:
                    raise ValueError(f"Column name {col} malformed: Feature {feature} not in {meas}!")
                else:
                    observation_frequency = cfg.vocabulary.obs_frequencies[cfg.vocabulary[feature]]

            total_observations = int(np.ceil(observation_frequency * n_possible))

            return self.get_smallest_valid_uint_type(total_observations)
        case _:
            raise ValueError(f"Column name {col} malformed!")


def _normalize_flat_rep_df_cols(
    flat_df: DF_T, feature_columns: list[str] | None = None, set_count_0_to_null: bool = False
) -> DF_T:
    if feature_columns is None:
        feature_columns = [x for x in flat_df.columns if x not in ("subject_id", "timestamp")]
        cols_to_add = set()
        cols_to_retype = set(feature_columns)
    else:
        cols_to_add = set(feature_columns) - set(flat_df.columns)
        cols_to_retype = set(feature_columns).intersection(set(flat_df.columns))

    cols_to_add = [(c, _get_flat_col_dtype(c)) for c in cols_to_add]
    cols_to_retype = [(c, _get_flat_col_dtype(c)) for c in cols_to_retype]

    if "timestamp" in flat_df.columns:
        key_cols = ["subject_id", "timestamp"]
    else:
        key_cols = ["subject_id"]

    flat_df = flat_df.with_columns(
        *[pl.lit(None, dtype=dt).alias(c) for c, dt in cols_to_add],
        *[pl.col(c).cast(dt).alias(c) for c, dt in cols_to_retype],
    ).select(*key_cols, *feature_columns)

    if not set_count_0_to_null:
        return flat_df

    flat_df = flat_df.collect()

    flat_df = flat_df.with_columns(
        pl.when(cs.ends_with("count") != 0).then(cs.ends_with("count")).keep_name()
    ).lazy()
    return flat_df


def _summarize_dynamic_measurements(
    self,
    feature_columns: list[str],
    include_only_subjects: set[int] | None = None,
) -> pl.LazyFrame:
    if include_only_subjects is None:
        df = self.dynamic_measurements_df
    else:
        df = self.dynamic_measurements_df.join(
            self.events_df.filter(pl.col("subject_id").is_in(list(include_only_subjects))).select("event_id"),
            on="event_id",
            how="inner",
        )

    valid_measures = {}
    for feat_col in feature_columns:
        temp, meas, feat, _ = self._parse_flat_feature_column(feat_col)

        if temp != "dynamic":
            continue

        if meas not in valid_measures:
            valid_measures[meas] = set()
        valid_measures[meas].add(feat)

    out_dfs = {}
    for m, allowed_vocab in valid_measures.items():
        cfg = self.measurement_configs[m]

        total_observations = int(
            np.ceil(
                cfg.observation_rate_per_case
                * cfg.observation_rate_over_cases
                * sum(self.n_events_per_subject.values())
            )
        )

        count_type = self.get_smallest_valid_uint_type(total_observations)

        if cfg.modality == "univariate_regression" and cfg.vocabulary is None:
            prefix = f"dynamic/{m}/{m}"

            key_col = pl.col(m)
            val_col = pl.col(m).drop_nans().cast(pl.Float32)

            out_dfs[m] = (
                df.lazy()
                .select("measurement_id", "event_id", m)
                .filter(pl.col(m).is_not_null())
                .groupby("event_id")
                .agg(
                    pl.col(m).is_not_null().sum().cast(count_type).alias(f"{prefix}/count"),
                    (
                        (pl.col(m).is_not_nan() & pl.col(m).is_not_null())
                        .sum()
                        .cast(count_type)
                        .alias(f"{prefix}/has_values_count")
                    ),
                    val_col.sum().alias(f"{prefix}/sum"),
                    (val_col**2).sum().alias(f"{prefix}/sum_sqd"),
                    val_col.min().alias(f"{prefix}/min"),
                    val_col.max().alias(f"{prefix}/max"),
                )
            )
            continue
        elif cfg.modality == "multivariate_regression":
            column_cols = [m, m]
            values_cols = [m, cfg.values_column]
            key_prefix = f"{m}_{m}_"
            val_prefix = f"{cfg.values_column}_{m}_"

            key_col = cs.starts_with(key_prefix)
            val_col = cs.starts_with(val_prefix).drop_nans().cast(pl.Float32)

            aggs = [
                key_col.is_not_null()
                .sum()
                .cast(count_type)
                .map_alias(lambda c: f"dynamic/{m}/{c.replace(key_prefix, '')}/count"),
                (
                    (cs.starts_with(val_prefix).is_not_null() & cs.starts_with(val_prefix).is_not_nan())
                    .sum()
                    .map_alias(lambda c: f"dynamic/{m}/{c.replace(val_prefix, '')}/has_values_count")
                ),
                val_col.sum().map_alias(lambda c: f"dynamic/{m}/{c.replace(val_prefix, '')}/sum"),
                (val_col**2).sum().map_alias(lambda c: f"dynamic/{m}/{c.replace(val_prefix, '')}/sum_sqd"),
                val_col.min().map_alias(lambda c: f"dynamic/{m}/{c.replace(val_prefix, '')}/min"),
                val_col.max().map_alias(lambda c: f"dynamic/{m}/{c.replace(val_prefix, '')}/max"),
            ]
        else:
            column_cols = [m]
            values_cols = [m]
            aggs = [
                pl.all().is_not_null().sum().cast(count_type).map_alias(lambda c: f"dynamic/{m}/{c}/count")
            ]

        ID_cols = ["measurement_id", "event_id"]
        out_dfs[m] = (
            df.select(*ID_cols, *set(column_cols + values_cols))
            .filter(pl.col(m).is_in(allowed_vocab))
            .pivot(
                index=ID_cols,
                columns=column_cols,
                values=values_cols,
                aggregate_function=None,
            )
            .lazy()
            .drop("measurement_id")
            .groupby("event_id")
            .agg(*aggs)
        )

    return pl.concat(list(out_dfs.values()), how="align")


def _summarize_over_window(df: DF_T, window_size: str) -> pl.LazyFrame:
    """Apply aggregations to the raw representation over a window size."""
    if isinstance(df, Path):
        df = pl.scan_parquet(df)

    def time_aggd_col_alias_fntr(new_agg: str | None = None) -> Callable[[str], str]:
        if new_agg is None:

            def f(c: str) -> str:
                return "/".join([window_size] + c.split("/")[1:])

        else:

            def f(c: str) -> str:
                return "/".join([window_size] + c.split("/")[1:-1] + [new_agg])

        return f

    # Columns to convert to counts:
    present_indicator_cols = cs.ends_with("/present")

    # Columns to convert to value aggregations:
    value_cols = cs.ends_with("/value")

    # Columns to aggregate via other operations
    cnt_cols = (cs.ends_with("/count") | cs.ends_with("/has_values_count")).fill_null(0)

    cols_to_sum = cs.ends_with("/sum") | cs.ends_with("/sum_sqd")
    cols_to_min = cs.ends_with("/min")
    cols_to_max = cs.ends_with("/max")

    if window_size == "FULL":
        df = df.groupby("subject_id").agg(
            "timestamp",
            # present to counts
            present_indicator_cols.cumsum().map_alias(time_aggd_col_alias_fntr("count")),
            # values to stats
            value_cols.is_not_null().cumsum().map_alias(time_aggd_col_alias_fntr("count")),
            (
                (value_cols.is_not_null() & value_cols.is_not_nan())
                .cumsum()
                .map_alias(time_aggd_col_alias_fntr("has_values_count"))
            ),
            value_cols.cumsum().map_alias(time_aggd_col_alias_fntr("sum")),
            (value_cols**2).cumsum().map_alias(time_aggd_col_alias_fntr("sum_sqd")),
            value_cols.cummin().map_alias(time_aggd_col_alias_fntr("min")),
            value_cols.cummax().map_alias(time_aggd_col_alias_fntr("max")),
            # Raw aggregations
            cnt_cols.cumsum().map_alias(time_aggd_col_alias_fntr()),
            cols_to_sum.cumsum().map_alias(time_aggd_col_alias_fntr()),
            cols_to_min.cummin().map_alias(time_aggd_col_alias_fntr()),
            cols_to_max.cummax().map_alias(time_aggd_col_alias_fntr()),
        )
        df = df.explode(*[c for c in df.columns if c != "subject_id"])
    else:
        df = df.groupby_rolling(
            index_column="timestamp",
            by="subject_id",
            period=window_size,
        ).agg(
            # present to counts
            present_indicator_cols.sum().map_alias(time_aggd_col_alias_fntr("count")),
            # values to stats
            value_cols.is_not_null().sum().map_alias(time_aggd_col_alias_fntr("count")),
            (
                (value_cols.is_not_null() & value_cols.is_not_nan())
                .sum()
                .map_alias(time_aggd_col_alias_fntr("has_values_count"))
            ),
            value_cols.sum().map_alias(time_aggd_col_alias_fntr("sum")),
            (value_cols**2).sum().map_alias(time_aggd_col_alias_fntr("sum_sqd")),
            value_cols.min().map_alias(time_aggd_col_alias_fntr("min")),
            value_cols.max().map_alias(time_aggd_col_alias_fntr("max")),
            # Raw aggregations
            cnt_cols.sum().map_alias(time_aggd_col_alias_fntr()),
            cols_to_sum.sum().map_alias(time_aggd_col_alias_fntr()),
            cols_to_min.min().map_alias(time_aggd_col_alias_fntr()),
            cols_to_max.max().map_alias(time_aggd_col_alias_fntr()),
        )

    return _normalize_flat_rep_df_cols(df, set_count_0_to_null=True)


def _get_flat_ts_rep(
    self,
    feature_columns: list[str],
    **kwargs,
) -> pl.LazyFrame:
    """Produce raw representation for dynamic data."""
    return _normalize_flat_rep_df_cols(
        _summarize_dynamic_measurements(feature_columns, **kwargs)
        .drop("event_id")
        .sort(by=["subject_id", "timestamp"])
        .collect()
        .lazy(),
        [c for c in feature_columns if not c.startswith("static/")],
    )
    # The above .collect().lazy() shouldn't be necessary but it appears to be for some reason...


def _get_flat_static_rep(
    self,
    feature_columns: list[str],
    **kwargs,
) -> pl.LazyFrame:
    """Produce raw representation for static data."""
    static_features = [c for c in feature_columns if c.startswith("static/")]
    return self._normalize_flat_rep_df_cols(
        self._summarize_static_measurements(static_features, **kwargs).collect().lazy(),
        static_features,
        set_count_0_to_null=False,
    )


def _get_flat_rep_feature_cols(
    self,
    feature_inclusion_frequency: float | dict[str, float] | None = None,
    window_sizes: list[str] | None = None,
    include_only_measurements: set[str] | None = None,
) -> list[str]:
    """
    process aggregations and select which columns get which aggregations
        1. static
                1. code & no numerical_values
                2. numerical_values
        2. dynamic
                1. codes -> aggs applied to all codes
                2. numerical_values -> continuous aggs
    """
    feature_inclusion_frequency, include_only_measurements = self._resolve_flat_rep_cache_params(
        feature_inclusion_frequency, include_only_measurements
    )
    feature_columns = []
    for m, cfg in self.measurement_configs.items():
        if m not in include_only_measurements:
            continue

        features = None
        if cfg.vocabulary is not None:
            vocab = copy.deepcopy(cfg.vocabulary)
            if feature_inclusion_frequency is not None:
                m_freq = feature_inclusion_frequency[m]
                vocab.filter(total_observations=None, min_valid_element_freq=m_freq)
            features = vocab.vocabulary
        # elif cfg.modality == DataModality.UNIVARIATE_REGRESSION:
        #     features = [m]
        else:
            raise ValueError(f"Config with modality {cfg.modality} should have a Vocabulary!")
        temps = []
        aggs = []

        # match cfg.temporality:
        #     case TemporalityType.STATIC:
        #         temps = [str(cfg.temporality)]
        #         match cfg.modality:
        #             case DataModality.UNIVARIATE_REGRESSION:
        #                 aggs = ["value"]
        #             case DataModality.SINGLE_LABEL_CLASSIFICATION:
        #                 aggs = ["present"]
        #             case _:
        #                 raise ValueError(f"{cfg.modality} invalid with {cfg.temporality}")
        #     case TemporalityType.FUNCTIONAL_TIME_DEPENDENT if window_sizes is None:
        #         temps = [str(cfg.temporality)]
        #         match cfg.modality:
        #             case DataModality.UNIVARIATE_REGRESSION:
        #                 aggs = ["value"]
        #             case DataModality.SINGLE_LABEL_CLASSIFICATION:
        #                 aggs = ["present"]
        #             case _:
        #                 raise ValueError(f"{cfg.modality} invalid with {cfg.temporality}")
        #     case TemporalityType.FUNCTIONAL_TIME_DEPENDENT if window_sizes is not None:
        #         temps = window_sizes
        #         match cfg.modality:
        #             case DataModality.UNIVARIATE_REGRESSION:
        #                 aggs = ["count", "has_values_count", "sum", "sum_sqd", "min", "max"]
        #             case DataModality.SINGLE_LABEL_CLASSIFICATION:
        #                 aggs = ["count"]
        #             case _:
        #                 raise ValueError(f"{cfg.modality} invalid with {cfg.temporality}")
        #     case TemporalityType.DYNAMIC:
        #         temps = [str(cfg.temporality)] if window_sizes is None else window_sizes
        #         match cfg.modality:
        #             case DataModality.UNIVARIATE_REGRESSION | DataModality.MULTIVARIATE_REGRESSION:
        #                 aggs = ["count", "has_values_count", "sum", "sum_sqd", "min", "max"]
        #             case DataModality.MULTI_LABEL_CLASSIFICATION:
        #                 aggs = ["count"]
        #             case _:
        #                 raise ValueError(f"{cfg.modality} invalid with {cfg.temporality}")

        for temp in temps:
            for feature in features:
                for agg in aggs:
                    feature_columns.append(f"{temp}/{m}/{feature}/{agg}")

    return sorted(feature_columns)


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
    for split_name, split_df in split_to_df.items():
        split_patient_ids = (
            split_df.select(pl.col("patient_id").cast(pl.Int32).unique()).collect().to_series().to_list()
        )
        print(len(split_patient_ids))
        if cfg.n_patients_per_sub_shard is None:
            sp_subjects[split_name] = split_patient_ids
        else:
            shuffled_patient_ids = rng.permutation(split_patient_ids)
            num_shards = max(len(split_patient_ids) // cfg.n_patients_per_sub_shard, 1)  # must be 1 or larger
            sharded_patient_ids = np.array_split(shuffled_patient_ids, num_shards)
            sp_subjects[split_name] = [shard.tolist() for shard in sharded_patient_ids]

    # store params in json file
    params_fp = flat_dir / "params.json"
    params = store_params_json(params_fp, cfg, sp_subjects)

    # 0. Identify Output Columns
    # We set window_sizes to None here because we want to get the feature column names for the raw flat
    # representation, not the summarized one.
    feature_columns = _get_flat_rep_feature_cols(
        min_code_inclusion_frequency=cfg.min_code_inclusion_frequency,
        window_sizes=None,
    )

    # 1. Produce static representation
    static_subdir = flat_dir / "static"

    static_dfs = {}
    for sp, subjects in tqdm(list(params["patient_shard_by_split"].items()), desc="Flattening Splits"):
        static_dfs[sp] = []
        sp_dir = static_subdir / sp

        for i, subjects_list in enumerate(tqdm(subjects, desc="Subject chunks", leave=False)):
            fp = sp_dir / f"{i}.parquet"
            static_dfs[sp].append(fp)
            if fp.exists():
                if cfg.do_update:
                    continue
                elif not cfg.do_overwrite:
                    raise FileExistsError(f"do_overwrite is {cfg.do_overwrite} and {fp} exists!")

            df = _get_flat_static_rep(
                feature_columns=feature_columns,
                include_only_subjects=subjects_list,
            )

            _write_df(df, fp, do_overwrite=cfg.do_overwrite)

    # 2. Produce raw representation
    ts_subdir = flat_dir / "at_ts"

    ts_dfs = {}
    for sp, subjects in tqdm(list(params["patient_shard_by_split"].items()), desc="Flattening Splits"):
        ts_dfs[sp] = []
        sp_dir = ts_subdir / sp

        for i, subjects_list in enumerate(tqdm(subjects, desc="Subject chunks", leave=False)):
            fp = sp_dir / f"{i}.parquet"
            ts_dfs[sp].append(fp)
            if fp.exists():
                if cfg.do_update:
                    continue
                elif not cfg.do_overwrite:
                    raise FileExistsError(f"do_overwrite is {cfg.do_overwrite} and {fp} exists!")

            df = _get_flat_ts_rep(
                feature_columns=feature_columns,
                include_only_subjects=subjects_list,
            )

            _write_df(df, fp, do_overwrite=cfg.do_overwrite)

    if cfg.window_sizes is None:
        return

    # 3. Produce summarized history representations
    history_subdir = flat_dir / "over_history"

    for window_size in tqdm(cfg.window_sizes, desc="History window sizes"):
        for sp, df_fps in tqdm(list(ts_dfs.items()), desc="Windowing Splits", leave=False):
            for i, df_fp in enumerate(tqdm(df_fps, desc="Subject chunks", leave=False)):
                fp = history_subdir / sp / window_size / f"{i}.parquet"
                if fp.exists():
                    if cfg.do_update:
                        continue
                    elif not cfg.do_overwrite:
                        raise FileExistsError(f"do_overwrite is {cfg.do_overwrite} and {fp} exists!")

                df = _summarize_over_window(df_fp, window_size)
                _write_df(df, fp)
