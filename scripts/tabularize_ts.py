import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from MEDS_tabular_automl.generate_ts_features import get_flat_ts_rep
from MEDS_tabular_automl.utils import setup_environment, write_df


@hydra.main(version_base=None, config_path="../configs", config_name="tabularize")
def tabularize_ts_data(
    cfg: DictConfig,
):
    """Processes a medical dataset to generates and stores flat representatiosn of time-series data.

    This function handles MEDS format data and pivots tables to create two types of data files
    with patient_id and timestamp indexes:
        code data: containing a column for every code and 1 and 0 values indicating presence
        value data: containing a column for every code which the numerical value observed.

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
            codes: A list of codes to include in the flat representation. If `None`, all codes will be
                included in the flat representation.
            aggs: A list of aggregations to apply to the raw representation. Must have length greater than 0.
            n_patients_per_sub_shard: The number of subjects that should be included in each output file.
                Lowering this number increases the number of files written, making the process of creating and
                leveraging these files slower but more memory efficient.
            do_overwrite: If `True`, this function will overwrite the data already stored in the target save
                directory.
            do_update: bool = True
            seed: The seed to use for random number generation.
    """
    flat_dir, split_to_df, feature_columns = setup_environment(cfg)
    # Produce ts representation
    ts_subdir = flat_dir / "ts"

    for sp, subjects_dfs in tqdm(list(split_to_df.items()), desc="Flattening Splits"):
        sp_dir = ts_subdir / sp

        for i, shard_df in enumerate(tqdm(subjects_dfs, desc="Subject chunks", leave=False)):
            code_fp = sp_dir / f"{i}_code.parquet"
            value_fp = sp_dir / f"{i}_value.parquet"
            if code_fp.exists() or value_fp.exists():
                if cfg.do_update:
                    continue
                elif not cfg.do_overwrite:
                    raise FileExistsError(
                        f"do_overwrite is {cfg.do_overwrite} and {code_fp.exists()}"
                        f" or {value_fp.exists()} exists!"
                    )

            code_df, value_df = get_flat_ts_rep(
                feature_columns=feature_columns,
                shard_df=shard_df,
            )
            write_df(code_df, code_fp, do_overwrite=cfg.do_overwrite)
            write_df(value_df, value_fp, do_overwrite=cfg.do_overwrite)
