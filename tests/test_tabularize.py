import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import json
import os
import tempfile
from io import StringIO
from pathlib import Path

import polars as pl
from hydra import compose, initialize

from MEDS_tabular_automl.describe_codes import get_feature_columns
from MEDS_tabular_automl.file_name import list_subdir_files
from MEDS_tabular_automl.scripts import (
    cache_task,
    describe_codes,
    launch_xgboost,
    sweep_xgboost,
    tabularize_static,
    tabularize_time_series,
)
from MEDS_tabular_automl.utils import (
    VALUE_AGGREGATIONS,
    get_events_df,
    get_feature_names,
    get_shard_prefix,
    get_unique_time_events_df,
    load_matrix,
)

SPLITS_JSON = """{"train/0": [239684, 1195293], "train/1": [68729, 814703], "tuning/0": [754281], "held_out/0": [1500733]}"""  # noqa: E501

MEDS_TRAIN_0 = """
patient_id,code,timestamp,numerical_value
239684,HEIGHT,,175.271115221764
239684,EYE_COLOR//BROWN,,
239684,DOB,1980-12-28T00:00:00.000000,
239684,TEMP,2010-05-11T17:41:51.000000,96.0
239684,ADMISSION//CARDIAC,2010-05-11T17:41:51.000000,
239684,HR,2010-05-11T17:41:51.000000,102.6
239684,TEMP,2010-05-11T17:48:48.000000,96.2
239684,HR,2010-05-11T17:48:48.000000,105.1
239684,TEMP,2010-05-11T18:25:35.000000,95.8
239684,HR,2010-05-11T18:25:35.000000,113.4
239684,HR,2010-05-11T18:57:18.000000,112.6
239684,TEMP,2010-05-11T18:57:18.000000,95.5
239684,DISCHARGE,2010-05-11T19:27:19.000000,
1195293,HEIGHT,,164.6868838269085
1195293,EYE_COLOR//BLUE,,
1195293,DOB,1978-06-20T00:00:00.000000,
1195293,TEMP,2010-06-20T19:23:52.000000,100.0
1195293,ADMISSION//CARDIAC,2010-06-20T19:23:52.000000,
1195293,HR,2010-06-20T19:23:52.000000,109.0
1195293,TEMP,2010-06-20T19:25:32.000000,100.0
1195293,HR,2010-06-20T19:25:32.000000,114.1
1195293,HR,2010-06-20T19:45:19.000000,119.8
1195293,TEMP,2010-06-20T19:45:19.000000,99.9
1195293,HR,2010-06-20T20:12:31.000000,112.5
1195293,TEMP,2010-06-20T20:12:31.000000,99.8
1195293,HR,2010-06-20T20:24:44.000000,107.7
1195293,TEMP,2010-06-20T20:24:44.000000,100.0
1195293,TEMP,2010-06-20T20:41:33.000000,100.4
1195293,HR,2010-06-20T20:41:33.000000,107.5
1195293,DISCHARGE,2010-06-20T20:50:04.000000,
"""
MEDS_TRAIN_1 = """
patient_id,code,timestamp,numerical_value
68729,EYE_COLOR//HAZEL,,
68729,HEIGHT,,160.3953106166676
68729,DOB,1978-03-09T00:00:00.000000,
68729,HR,2010-05-26T02:30:56.000000,86.0
68729,ADMISSION//PULMONARY,2010-05-26T02:30:56.000000,
68729,TEMP,2010-05-26T02:30:56.000000,97.8
68729,DISCHARGE,2010-05-26T04:51:52.000000,
814703,EYE_COLOR//HAZEL,,
814703,HEIGHT,,156.48559093209357
814703,DOB,1976-03-28T00:00:00.000000,
814703,TEMP,2010-02-05T05:55:39.000000,100.1
814703,HR,2010-02-05T05:55:39.000000,170.2
814703,ADMISSION//ORTHOPEDIC,2010-02-05T05:55:39.000000,
814703,DISCHARGE,2010-02-05T07:02:30.000000,
"""
MEDS_HELD_OUT_0 = """
patient_id,code,timestamp,numerical_value
1500733,HEIGHT,,158.60131573580904
1500733,EYE_COLOR//BROWN,,
1500733,DOB,1986-07-20T00:00:00.000000,
1500733,TEMP,2010-06-03T14:54:38.000000,100.0
1500733,HR,2010-06-03T14:54:38.000000,91.4
1500733,ADMISSION//ORTHOPEDIC,2010-06-03T14:54:38.000000,
1500733,HR,2010-06-03T15:39:49.000000,84.4
1500733,TEMP,2010-06-03T15:39:49.000000,100.3
1500733,HR,2010-06-03T16:20:49.000000,90.1
1500733,TEMP,2010-06-03T16:20:49.000000,100.1
1500733,DISCHARGE,2010-06-03T16:44:26.000000,
"""
MEDS_TUNING_0 = """
patient_id,code,timestamp,numerical_value
754281,EYE_COLOR//BROWN,,
754281,HEIGHT,,166.22261567137025
754281,DOB,1988-12-19T00:00:00.000000,
754281,ADMISSION//PULMONARY,2010-01-03T06:27:59.000000,
754281,TEMP,2010-01-03T06:27:59.000000,99.8
754281,HR,2010-01-03T06:27:59.000000,142.0
754281,DISCHARGE,2010-01-03T08:22:13.000000,
"""

MEDS_OUTPUTS = {
    "train/0": MEDS_TRAIN_0,
    "train/1": MEDS_TRAIN_1,
    "held_out/0": MEDS_HELD_OUT_0,
    "tuning/0": MEDS_TUNING_0,
}

CODE_COLS = [
    "ADMISSION//CARDIAC/code",
    "ADMISSION//ORTHOPEDIC/code",
    "ADMISSION//PULMONARY/code",
    "DISCHARGE/code",
    "DOB/code",
    "HR/code",
    "TEMP/code",
]
VALUE_COLS = ["HR/value", "TEMP/value"]
STATIC_PRESENT_COLS = [
    "EYE_COLOR//BLUE/static/present",
    "EYE_COLOR//BROWN/static/present",
    "EYE_COLOR//HAZEL/static/present",
    "HEIGHT/static/present",
]
STATIC_FIRST_COLS = ["HEIGHT/static/first"]

EXPECTED_STATIC_FILES = [
    "held_out/0/none/static/first.npz",
    "held_out/0/none/static/present.npz",
    "train/0/none/static/first.npz",
    "train/0/none/static/present.npz",
    "train/1/none/static/first.npz",
    "train/1/none/static/present.npz",
    "tuning/0/none/static/first.npz",
    "tuning/0/none/static/present.npz",
]

SUMMARIZE_EXPECTED_FILES = [
    "train/1/365d/value/sum.npz",
    "train/1/365d/code/count.npz",
    "train/1/full/value/sum.npz",
    "train/1/full/code/count.npz",
    "train/1/30d/value/sum.npz",
    "train/1/30d/code/count.npz",
    "train/0/365d/value/sum.npz",
    "train/0/365d/code/count.npz",
    "train/0/full/value/sum.npz",
    "train/0/full/code/count.npz",
    "train/0/30d/value/sum.npz",
    "train/0/30d/code/count.npz",
    "held_out/0/365d/value/sum.npz",
    "held_out/0/365d/code/count.npz",
    "held_out/0/full/value/sum.npz",
    "held_out/0/full/code/count.npz",
    "held_out/0/30d/value/sum.npz",
    "held_out/0/30d/code/count.npz",
    "tuning/0/365d/value/sum.npz",
    "tuning/0/365d/code/count.npz",
    "tuning/0/full/value/sum.npz",
    "tuning/0/full/code/count.npz",
    "tuning/0/30d/value/sum.npz",
    "tuning/0/30d/code/count.npz",
]

MERGE_EXPECTED_FILES = [
    "train/365d/value/sum/0.npz",
    "train/365d/value/sum/1.npz",
    "train/365d/code/count/0.npz",
    "train/365d/code/count/1.npz",
    "train/full/value/sum/0.npz",
    "train/full/value/sum/1.npz",
    "train/full/code/count/0.npz",
    "train/full/code/count/1.npz",
    "train/30d/value/sum/0.npz",
    "train/30d/value/sum/1.npz",
    "train/30d/code/count/0.npz",
    "train/30d/code/count/1.npz",
    "held_out/365d/value/sum/0.npz",
    "held_out/365d/code/count/0.npz",
    "held_out/full/value/sum/0.npz",
    "held_out/full/code/count/0.npz",
    "held_out/30d/value/sum/0.npz",
    "held_out/30d/code/count/0.npz",
    "tuning/365d/value/sum/0.npz",
    "tuning/365d/code/count/0.npz",
    "tuning/full/value/sum/0.npz",
    "tuning/full/code/count/0.npz",
    "tuning/30d/value/sum/0.npz",
    "tuning/30d/code/count/0.npz",
]


def test_tabularize():
    with tempfile.TemporaryDirectory() as d:
        MEDS_cohort_dir = Path(d) / "processed"

        describe_codes_config = {
            "MEDS_cohort_dir": str(MEDS_cohort_dir.resolve()),
            "do_overwrite": False,
            "seed": 1,
            "hydra.verbose": True,
            "tqdm": False,
            "loguru_init": True,
        }

        with initialize(
            version_base=None, config_path="../src/MEDS_tabular_automl/configs/"
        ):  # path to config.yaml
            overrides = [f"{k}={v}" for k, v in describe_codes_config.items()]
            cfg = compose(config_name="describe_codes", overrides=overrides)  # config.yaml

        # Create the directories
        (MEDS_cohort_dir / "final_cohort").mkdir(parents=True, exist_ok=True)

        # Store MEDS outputs
        for split, data in MEDS_OUTPUTS.items():
            file_path = MEDS_cohort_dir / "final_cohort" / f"{split}.parquet"
            file_path.parent.mkdir(exist_ok=True)
            df = pl.read_csv(StringIO(data))
            df.with_columns(pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f")).write_parquet(
                file_path
            )

        # Check the files are not empty
        meds_files = list_subdir_files(Path(cfg.input_dir), "parquet")
        assert (
            len(list_subdir_files(Path(cfg.input_dir).parent, "parquet")) == 4
        ), "MEDS train split Data Files Should be 4!"
        for f in meds_files:
            assert pl.read_parquet(f).shape[0] > 0, "MEDS Data Tabular Dataframe Should not be Empty!"
        split_json = json.load(StringIO(SPLITS_JSON))
        splits_fp = MEDS_cohort_dir / "splits.json"
        json.dump(split_json, splits_fp.open("w"))
        # Step 1: Describe Codes - compute code frequencies
        describe_codes.main(cfg)

        assert (Path(cfg.output_dir) / "config.yaml").is_file()
        assert Path(cfg.output_filepath).is_file()

        feature_columns = get_feature_columns(cfg.output_filepath)
        assert get_feature_names("code/count", feature_columns) == sorted(CODE_COLS)
        assert get_feature_names("static/present", feature_columns) == sorted(STATIC_PRESENT_COLS)
        assert get_feature_names("static/first", feature_columns) == sorted(STATIC_FIRST_COLS)
        for value_agg in VALUE_AGGREGATIONS:
            assert get_feature_names(value_agg, feature_columns) == sorted(VALUE_COLS)

        # Step 2: Tabularization
        tabularize_static_config = {
            "MEDS_cohort_dir": str(MEDS_cohort_dir.resolve()),
            "do_overwrite": False,
            "seed": 1,
            "hydra.verbose": True,
            "tqdm": False,
            "loguru_init": True,
            "tabularization.min_code_inclusion_frequency": 1,
            "tabularization.aggs": "[static/present,static/first,code/count,value/sum]",
            "tabularization.window_sizes": "[30d,365d,full]",
        }

        with initialize(
            version_base=None, config_path="../src/MEDS_tabular_automl/configs/"
        ):  # path to config.yaml
            overrides = [f"{k}={v}" for k, v in tabularize_static_config.items()]
            cfg = compose(config_name="tabularization", overrides=overrides)  # config.yaml
        tabularize_static.main(cfg)
        output_files = list(Path(cfg.output_dir).glob("**/static/**/*.npz"))
        actual_files = [get_shard_prefix(Path(cfg.output_dir), each) + ".npz" for each in output_files]
        assert set(actual_files) == set(EXPECTED_STATIC_FILES)
        # Check the files are not empty
        for f in output_files:
            static_matrix = load_matrix(f)
            assert static_matrix.shape[0] > 0, "Static Data Tabular Dataframe Should not be Empty!"
            expected_num_cols = len(get_feature_names(f"static/{f.stem}", feature_columns))
            assert static_matrix.shape[1] == expected_num_cols, (
                f"Static Data Tabular Dataframe Should have {expected_num_cols}"
                f"Columns but has {static_matrix.shape[1]}!"
            )
            split = f.parts[-5]
            shard_num = f.parts[-4]
            med_shard_fp = (Path(cfg.input_dir) / split / shard_num).with_suffix(".parquet")
            expected_num_rows = (
                get_unique_time_events_df(get_events_df(pl.scan_parquet(med_shard_fp), feature_columns))
                .collect()
                .shape[0]
            )
            assert static_matrix.shape[0] == expected_num_rows, (
                f"Static Data matrix Should have {expected_num_rows}"
                f" rows but has {static_matrix.shape[0]}!"
            )
        allowed_codes = cfg.tabularization._resolved_codes
        num_allowed_codes = len(allowed_codes)
        feature_columns = get_feature_columns(cfg.tabularization.filtered_code_metadata_fp)
        assert num_allowed_codes == len(
            feature_columns
        ), f"Should have {len(feature_columns)} codes but has {num_allowed_codes}"

        tabularize_time_series.main(cfg)

        # confirm summary files exist:
        output_files = list_subdir_files(cfg.output_dir, "npz")
        actual_files = [
            get_shard_prefix(Path(cfg.output_dir), each) + ".npz"
            for each in output_files
            if "none/static" not in str(each)
        ]
        assert set(actual_files) == set(SUMMARIZE_EXPECTED_FILES)
        for f in output_files:
            ts_matrix = load_matrix(f)
            assert ts_matrix.shape[0] > 0, "Time-Series Tabular Dataframe Should not be Empty!"
            expected_num_cols = len(get_feature_names(f"{f.parent.stem}/{f.stem}", feature_columns))
            assert ts_matrix.shape[1] == expected_num_cols, (
                f"Time-Series Tabular Dataframe Should have {expected_num_cols}"
                f"Columns but has {ts_matrix.shape[1]}!"
            )
            split = f.parts[-5]
            shard_num = f.parts[-4]
            med_shard_fp = (Path(cfg.input_dir) / split / shard_num).with_suffix(".parquet")
            expected_num_rows = (
                get_unique_time_events_df(get_events_df(pl.scan_parquet(med_shard_fp), feature_columns))
                .collect()
                .shape[0]
            )
            assert ts_matrix.shape[0] == expected_num_rows, (
                f"Time-Series Data matrix Should have {expected_num_rows}"
                f" rows but has {ts_matrix.shape[0]}!"
            )

        # Step 3: Cache Task data
        cache_config = {
            "MEDS_cohort_dir": str(MEDS_cohort_dir.resolve()),
            "do_overwrite": False,
            "seed": 1,
            "hydra.verbose": True,
            "tqdm": False,
            "loguru_init": True,
            "tabularization.min_code_inclusion_frequency": 1,
            "tabularization.aggs": "[static/present,static/first,code/count,value/sum]",
            "tabularization.window_sizes": "[30d,365d,full]",
        }

        with initialize(
            version_base=None, config_path="../src/MEDS_tabular_automl/configs/"
        ):  # path to config.yaml
            overrides = [f"{k}={v}" for k, v in cache_config.items()]
            cfg = compose(config_name="task_specific_caching", overrides=overrides)  # config.yaml

        # Create fake labels
        for f in list_subdir_files(Path(cfg.MEDS_cohort_dir) / "final_cohort", "parquet"):
            df = pl.scan_parquet(f)
            df = get_unique_time_events_df(get_events_df(df, feature_columns)).collect()
            pseudo_labels = pl.Series(([0, 1] * df.shape[0])[: df.shape[0]])
            df = df.with_columns(pl.Series(name="label", values=pseudo_labels))
            df = df.select(pl.col(["patient_id", "timestamp", "label"]))
            df = df.with_row_index("event_id")

            split = f.parent.stem
            shard_num = f.stem
            out_f = Path(cfg.input_label_dir) / Path(
                get_shard_prefix(Path(cfg.MEDS_cohort_dir) / "final_cohort", f)
            ).with_suffix(".parquet")
            out_f.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(out_f)

        cache_task.main(cfg)

        xgboost_config_kwargs = {
            "MEDS_cohort_dir": str(MEDS_cohort_dir.resolve()),
            "do_overwrite": False,
            "seed": 1,
            "hydra.verbose": True,
            "tqdm": False,
            "loguru_init": True,
            "tabularization.min_code_inclusion_frequency": 1,
            "tabularization.aggs": "[static/present,static/first,code/count,value/sum]",
            "tabularization.window_sizes": "[30d,365d,full]",
        }

        with initialize(
            version_base=None, config_path="../src/MEDS_tabular_automl/configs/"
        ):  # path to config.yaml
            overrides = [f"{k}={v}" for k, v in xgboost_config_kwargs.items()]
            cfg = compose(config_name="launch_xgboost", overrides=overrides)  # config.yaml

        launch_xgboost.main(cfg)
        output_files = list(Path(cfg.output_dir).glob("**/*.json"))
        assert len(output_files) == 1
        assert output_files[0] == Path(cfg.output_dir) / "model.json"
        os.remove(Path(cfg.output_dir) / "model.json")

        xgboost_config_kwargs = {
            "MEDS_cohort_dir": str(MEDS_cohort_dir.resolve()),
            "do_overwrite": False,
            "seed": 1,
            "hydra.verbose": True,
            "tqdm": False,
            "loguru_init": True,
            "tabularization.min_code_inclusion_frequency": 1,
            "tabularization.aggs": "[static/present,static/first,code/count,value/sum]",
            "tabularization.window_sizes": "[30d,365d,full]",
        }

        with initialize(
            version_base=None, config_path="../src/MEDS_tabular_automl/configs/"
        ):  # path to config.yaml
            overrides = [f"{k}={v}" for k, v in xgboost_config_kwargs.items()]
            cfg = compose(config_name="launch_xgboost", overrides=overrides)  # config.yaml

        sweep_xgboost.main(cfg)
        output_files = list(Path(cfg.output_dir).glob("**/*.json"))
        assert len(output_files) == 1
        assert output_files[0] == Path(cfg.output_dir) / "model.json"
