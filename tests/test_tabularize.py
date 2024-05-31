import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import json
import shutil
import tempfile
from io import StringIO
from pathlib import Path

import pandas as pd
import polars as pl
from hydra import compose, initialize
from loguru import logger

from scripts.identify_columns import store_columns
from scripts.summarize_over_windows import summarize_ts_data_over_windows
from scripts.tabularize_static import tabularize_static_data
from scripts.tabularize_ts import tabularize_ts_data
from scripts.xgboost_sweep import xgboost

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


def test_tabularize():
    with tempfile.TemporaryDirectory() as d:
        MEDS_cohort_dir = Path(d) / "MEDS_cohort"
        tabularized_data_dir = Path(d) / "flat_reps"
        model_dir = Path(d) / "save_model"

        # Create the directories
        MEDS_cohort_dir.mkdir()

        # Store MEDS outputs
        for split, data in MEDS_OUTPUTS.items():
            file_path = MEDS_cohort_dir / f"{split}.parquet"
            file_path.parent.mkdir(exist_ok=True)
            df = pl.read_csv(StringIO(data))
            df.with_columns(pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S.%f")).write_parquet(
                file_path
            )

        split_json = json.load(StringIO(SPLITS_JSON))
        splits_fp = MEDS_cohort_dir / "splits.json"
        json.dump(split_json, splits_fp.open("w"))

        tabularize_config_kwargs = {
            "MEDS_cohort_dir": str(MEDS_cohort_dir.resolve()),
            "tabularized_data_dir": str(tabularized_data_dir.resolve()),
            "min_code_inclusion_frequency": 1,
            "window_sizes": ["30d", "365d", "full"],
            "aggs": ["code/count", "value/sum"],
            "codes": "null",
            "n_patients_per_sub_shard": 2,
            "do_overwrite": True,
            "do_update": True,
            "seed": 1,
            "hydra.verbose": True,
            "tqdm": False,
        }

        with initialize(version_base=None, config_path="../configs/"):  # path to config.yaml
            overrides = [f"{k}={v}" for k, v in tabularize_config_kwargs.items()]
            cfg = compose(config_name="tabularize", overrides=overrides)  # config.yaml
        logger.info("caching flat representation of MEDS data")
        store_columns(cfg)
        assert (tabularized_data_dir / "config.yaml").is_file()
        assert (tabularized_data_dir / "feature_columns.json").is_file()
        assert (tabularized_data_dir / "feature_freqs.json").is_file()
        tabularize_static_data(cfg)
        actual_files = [
            (f.parent.stem, f.stem) for f in list(tabularized_data_dir.glob("static/*/*.parquet"))
        ]
        expected_files = [("train", "1"), ("train", "0"), ("held_out", "0"), ("tuning", "0")]
        assert set(actual_files) == set(expected_files)

        # Check the files are not empty
        for f in list(tabularized_data_dir.glob("static/*/*.parquet")):
            assert pl.read_parquet(f).shape[0] > 0, "Static Data Tabular Dataframe Should not be Empty!"

        tabularize_ts_data(cfg)
        # confirm the time series files exist:
        actual_files = [(f.parent.stem, f.stem) for f in list(tabularized_data_dir.glob("ts/*/*.pkl"))]
        expected_files = [
            ("train", "1"),
            ("train", "0"),
            ("held_out", "0"),
            ("tuning", "0"),
        ]
        assert set(actual_files) == set(expected_files)
        for f in list(tabularized_data_dir.glob("ts/*/*.pkl")):
            assert pd.read_pickle(f).shape[0] > 0, "Time-Series Tabular Dataframe Should not be Empty!"
        shutil.rmtree(tabularized_data_dir / "ts")

        summarize_ts_data_over_windows(cfg)
        # confirm summary files exist:
        output_files = list(tabularized_data_dir.glob("ts/*/*/*/*/*.pkl"))
        actual_files = [str(Path(*f.parts[-5:])) for f in output_files]
        expected_files = [
            "train/365d/value/sum/0.pkl",
            "train/365d/value/sum/1.pkl",
            "train/365d/code/count/0.pkl",
            "train/365d/code/count/1.pkl",
            "train/full/value/sum/0.pkl",
            "train/full/value/sum/1.pkl",
            "train/full/code/count/0.pkl",
            "train/full/code/count/1.pkl",
            "train/30d/value/sum/0.pkl",
            "train/30d/value/sum/1.pkl",
            "train/30d/code/count/0.pkl",
            "train/30d/code/count/1.pkl",
            "held_out/365d/value/sum/0.pkl",
            "held_out/365d/code/count/0.pkl",
            "held_out/full/value/sum/0.pkl",
            "held_out/full/code/count/0.pkl",
            "held_out/30d/value/sum/0.pkl",
            "held_out/30d/code/count/0.pkl",
            "tuning/365d/value/sum/0.pkl",
            "tuning/365d/code/count/0.pkl",
            "tuning/full/value/sum/0.pkl",
            "tuning/full/code/count/0.pkl",
            "tuning/30d/value/sum/0.pkl",
            "tuning/30d/code/count/0.pkl",
        ]
        assert set(actual_files) == set(expected_files)
        for f in output_files:
            df = pd.read_pickle(f)
            assert df.shape[0] > 0

        xgboost_config_kwargs = {
            "model_dir": str(model_dir.resolve()),
            "hydra.mode": "MULTIRUN",
        }
        xgboost_config_kwargs = {**tabularize_config_kwargs, **xgboost_config_kwargs}
        with initialize(version_base=None, config_path="../configs/"):  # path to config.yaml
            overrides = [f"{k}={v}" for k, v in xgboost_config_kwargs.items()]
            cfg = compose(config_name="xgboost_sweep", overrides=overrides)  # config.yaml
        xgboost(cfg)
        output_files = list(model_dir.glob("*/*/*_model.json"))
        assert len(output_files) == 1
