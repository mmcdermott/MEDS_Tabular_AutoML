"""
Setup Conda environment as described here: https://github.com/justin13601/ACES
"""
from pathlib import Path

import hydra
import polars as pl
from aces import config, predicates, query
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../configs", config_name="tabularize")
def main(cfg):
    # create task configuration object
    task_cfg = config.TaskExtractorConfig.load(config_path="hf_cohort/task.yaml")

    # setup directories
    med_dir = Path(cfg.tabularized_data_dir)

    # location of MEDS format Data
    cohort_dir = med_dir.parent / "final_cohort"
    # output directory for tables with event_ids and labels
    output_dir = med_dir / "task"

    shard_fps = list(cohort_dir.glob("*/*.parquet"))

    for in_fp in tqdm(shard_fps):
        out_fp = output_dir / "/".join(in_fp.parts[-2:])
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        # one of the following
        predicates_df = predicates.generate_predicates_df(task_cfg, in_fp, "meds")

        # execute query
        df_result = query.query(task_cfg, predicates_df)
        label_df = (
            df_result.select(pl.col(["subject_id", "trigger", "label"]))
            .rename({"trigger": "timestamp", "subject_id": "patient_id"})
            .sort(by=["patient_id", "timestamp"])
        )
        data_df = pl.scan_parquet(in_fp)
        data_df = data_df.unique(subset=["patient_id", "timestamp"]).sort(by=["patient_id", "timestamp"])
        data_df = data_df.with_row_index("event_id")
        data_df = data_df.drop(["code", "numerical_value"])
        output_df = label_df.lazy().join_asof(other=data_df, by="patient_id", on="timestamp")

        # store it
        output_df.collect().write_parquet(out_fp)


if __name__ == "__main__":
    main()
