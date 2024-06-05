import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import json
import subprocess
import tempfile
from io import StringIO
from pathlib import Path

import polars as pl
from hydra import compose, initialize
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

from MEDS_tabular_automl.describe_codes import get_feature_columns
from MEDS_tabular_automl.file_name import list_subdir_files
from MEDS_tabular_automl.utils import (
    VALUE_AGGREGATIONS,
    get_events_df,
    get_feature_names,
    get_shard_prefix,
    get_unique_time_events_df,
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

        # Step 1: Run the describe_codes script
        stderr, stdout = run_command(
            "meds-tab-describe",
            [],
            describe_codes_config,
            "describe_codes",
        )
        assert (Path(cfg.output_dir) / "config.yaml").is_file()
        assert Path(cfg.output_filepath).is_file()

        feature_columns = get_feature_columns(cfg.output_filepath)
        assert get_feature_names("code/count", feature_columns) == sorted(CODE_COLS)
        assert get_feature_names("static/present", feature_columns) == sorted(STATIC_PRESENT_COLS)
        assert get_feature_names("static/first", feature_columns) == sorted(STATIC_FIRST_COLS)
        for value_agg in VALUE_AGGREGATIONS:
            assert get_feature_names(value_agg, feature_columns) == sorted(VALUE_COLS)

        # Step 2: Run the static data tabularization script
        tabularize_config = {
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
        stderr, stdout = run_command(
            "meds-tab-tabularize-static",
            [],
            tabularize_config,
            "tabularization",
        )
        with initialize(
            version_base=None, config_path="../src/MEDS_tabular_automl/configs/"
        ):  # path to config.yaml
            overrides = [f"{k}={v}" for k, v in tabularize_config.items()]
            cfg = compose(config_name="tabularization", overrides=overrides)  # config.yaml

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

        # Step 3: Run the time series tabularization script
        tabularize_config = {
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

        stderr, stdout = run_command(
            "meds-tab-tabularize-time-series",
            ["--multirun", 'worker="range(0,1)"', "hydra/launcher=joblib"],
            tabularize_config,
            "tabularization",
        )

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

        # Step 4: Run the task_specific_caching script
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

        stderr, stdout = run_command(
            "meds-tab-cache-task",
            [],
            cache_config,
            "task_specific_caching",
        )
        # Check the files are not empty

        # Step 5: Run the xgboost script

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
        stderr, stdout = run_command(
            "meds-tab-xgboost",
            [],
            xgboost_config_kwargs,
            "xgboost",
        )
        output_files = list(Path(cfg.output_dir).parent.glob("**/*.json"))
        assert len(output_files) == 1
        assert output_files[0].stem == "model"
