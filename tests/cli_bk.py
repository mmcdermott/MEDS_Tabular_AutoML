import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import json
import subprocess
import tempfile
from io import StringIO
from pathlib import Path

import polars as pl
from loguru import logger
from omegaconf import DictConfig
from scripts.identify_columns import store_columns
from scripts.tabularize_static import tabularize_static_data
from test_tabularize import (
    CODE_COLS,
    EXPECTED_STATIC_FILES,
    MEDS_OUTPUTS,
    SPLITS_JSON,
    STATIC_FIRST_COLS,
    STATIC_PRESENT_COLS,
    SUMMARIZE_EXPECTED_FILES,
    VALUE_COLS,
)

from MEDS_tabular_automl.file_name import FileNameResolver
from MEDS_tabular_automl.utils import (
    VALUE_AGGREGATIONS,
    get_events_df,
    get_feature_names,
    load_matrix,
)


def run_command(script: str, args: list[str], hydra_kwargs: dict[str, str], test_name: str):
    command_parts = [script] + args + [f"{k}={v}" for k, v in hydra_kwargs.items()]
    command_out = subprocess.run(" ".join(command_parts), shell=True, capture_output=True)
    stderr = command_out.stderr.decode()
    stdout = command_out.stdout.decode()
    if command_out.returncode != 0:
        raise AssertionError(f"{test_name} failed!\nstdout:\n{stdout}\nstderr:\n{stderr}")
    return stderr, stdout


def test_tabularize():
    # Step 0: Setup Environment
    with tempfile.TemporaryDirectory() as d:
        MEDS_cohort_dir = Path(d) / "processed"
        tabularized_data_dir = Path(d) / "processed" / "tabularize"
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

        tabularize_config_kwargs = {
            "MEDS_cohort_dir": str(MEDS_cohort_dir.resolve()),
            "tabularized_data_dir": str(tabularized_data_dir.resolve()),
            "min_code_inclusion_frequency": 1,
            "model_dir": str(Path(d) / "save_model"),
            "window_sizes": "[30d,365d,full]",
            "aggs": "[code/count,value/sum,static/present,static/first]",
            "codes": "null",
            "n_patients_per_sub_shard": 2,
            "do_overwrite": True,
            "do_update": True,
            "seed": 1,
            "hydra.verbose": True,
            "tqdm": False,
            "test": True,
            "task_dir": str((tabularized_data_dir / "task").resolve()),
        }
        cfg = DictConfig(tabularize_config_kwargs)
        f_name_resolver = FileNameResolver(cfg)
        meds_files = f_name_resolver.list_meds_files()
        assert len(meds_files) == 4, "MEDS Data Files Should be 4!"
        for f in meds_files:
            assert pl.read_parquet(f).shape[0] > 0, "MEDS Data Tabular Dataframe Should not be Empty!"

        split_json = json.load(StringIO(SPLITS_JSON))
        splits_fp = MEDS_cohort_dir / "splits.json"
        json.dump(split_json, splits_fp.open("w"))
        logger.info("caching flat representation of MEDS data")

        # Step 1: Run the describe_codes script
        stderr, stdout = run_command(
            "meds_tab describe_codes",
            [],
            tabularize_config_kwargs,
            "describe_codes",
        )

        store_columns(cfg)
        assert (tabularized_data_dir / "config.yaml").is_file()
        assert (tabularized_data_dir / "feature_columns.json").is_file()
        assert (tabularized_data_dir / "feature_freqs.json").is_file()

        feature_columns = json.load(open(f_name_resolver.get_feature_columns_fp()))
        assert get_feature_names("code/count", feature_columns) == sorted(CODE_COLS)
        assert get_feature_names("static/present", feature_columns) == sorted(STATIC_PRESENT_COLS)
        assert get_feature_names("static/first", feature_columns) == sorted(STATIC_FIRST_COLS)
        for value_agg in VALUE_AGGREGATIONS:
            assert get_feature_names(value_agg, feature_columns) == sorted(VALUE_COLS)

        # Step 2: Run the tabularization script
        n_workers = "1"
        stderr, stdout = run_command(
            "meds_tab tabularization",
            [n_workers],
            tabularize_config_kwargs,
            "tabularization",
        )
        # Check Static File Generation
        tabularize_static_data(cfg)
        actual_files = [str(Path(*f.parts[-5:])) for f in f_name_resolver.list_static_files()]
        assert set(actual_files) == set(EXPECTED_STATIC_FILES)
        # Check the files are not empty
        for f in f_name_resolver.list_static_files():
            static_matrix = load_matrix(f)
            assert static_matrix.shape[0] > 0, "Static Data Tabular Dataframe Should not be Empty!"
            expected_num_cols = len(get_feature_names(f"static/{f.stem}", feature_columns))
            logger.info((static_matrix.shape[1], expected_num_cols))
            logger.info(f_name_resolver.list_static_files())
            assert static_matrix.shape[1] == expected_num_cols, (
                f"Static Data Tabular Dataframe Should have {expected_num_cols}"
                f"Columns but has {static_matrix.shape[1]}!"
            )
        static_first_fp = f_name_resolver.get_flat_static_rep("tuning", "0", "static/first")
        static_present_fp = f_name_resolver.get_flat_static_rep("tuning", "0", "static/present")
        assert (
            load_matrix(static_first_fp).shape[0] == load_matrix(static_present_fp).shape[0]
        ), "static data first and present aggregations have different numbers of rows"

        # Check Time Series File Generation
        output_files = f_name_resolver.list_ts_files()
        f_name_resolver.list_ts_files()
        actual_files = [str(Path(*f.parts[-5:])) for f in output_files]

        assert set(actual_files) == set(SUMMARIZE_EXPECTED_FILES)
        for f in output_files:
            sparse_array = load_matrix(f)
            assert sparse_array.shape[0] > 0
            assert sparse_array.shape[1] > 0
        ts_code_fp = f_name_resolver.get_flat_ts_rep("tuning", "0", "365d", "code/count")
        ts_value_fp = f_name_resolver.get_flat_ts_rep("tuning", "0", "365d", "value/sum")
        assert (
            load_matrix(ts_code_fp).shape[0] == load_matrix(ts_value_fp).shape[0]
        ), "time series code and value have different numbers of rows"
        assert (
            load_matrix(static_first_fp).shape[0] == load_matrix(ts_value_fp).shape[0]
        ), "static data and time series have different numbers of rows"

        # Create Fake Labels
        feature_columns = json.load(open(f_name_resolver.get_feature_columns_fp()))
        for f in f_name_resolver.list_meds_files():
            df = pl.read_parquet(f)
            df = get_events_df(df, feature_columns)
            pseudo_labels = pl.Series(([0, 1] * df.shape[0])[: df.shape[0]])
            df = df.with_columns(pl.Series(name="label", values=pseudo_labels))
            df = df.select(pl.col(["patient_id", "timestamp", "label"]))
            df = df.unique(subset=["patient_id", "timestamp"])
            df = df.with_row_index("event_id")

            split = f.parent.stem
            shard_num = f.stem
            out_f = f_name_resolver.get_label(split, shard_num)
            out_f.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(out_f)

        # Step 3: Run the task_specific_caching script
        stderr, stdout = run_command(
            "meds_tab task_specific_caching",
            [],
            tabularize_config_kwargs,
            "task_specific_caching",
        )
        # Check the files are not empty

        # Step 4: Run the xgboost script
        xgboost_config_kwargs = {
            "hydra.mode": "MULTIRUN",
        }
        xgboost_config_kwargs = {**tabularize_config_kwargs, **xgboost_config_kwargs}
        stderr, stdout = run_command(
            "meds_tab xgboost",
            [],
            xgboost_config_kwargs,
            "xgboost",
        )
        output_files = list(Path(cfg.model_dir).glob("*.json"))
        assert len(output_files) == 1
        assert output_files[0] == Path(cfg.model_dir) / "model.json"
