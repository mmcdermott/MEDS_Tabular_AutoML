import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import glob
import json
import shutil
import subprocess
from io import StringIO
from pathlib import Path

import polars as pl
from hydra import compose, initialize
from meds_evaluation.schema import validate_binary_classification_schema

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
from tests.test_tabularize import (
    CODE_COLS,
    EXPECTED_STATIC_FILES,
    MEDS_OUTPUTS,
    NUM_SHARDS,
    SPLITS_JSON,
    STATIC_FIRST_COLS,
    STATIC_PRESENT_COLS,
    VALUE_COLS,
)


def run_command(script: str, args: list[str], hydra_kwargs: dict[str, str], test_name: str):
    command_parts = [script] + args + [f"{k}={v}" for k, v in hydra_kwargs.items()]
    command_out = subprocess.run(" ".join(command_parts), shell=True, capture_output=True)
    stderr = command_out.stderr.decode()
    stdout = command_out.stdout.decode()
    if command_out.returncode != 0:
        raise AssertionError(f"{test_name} failed!\nstdout:\n{stdout}\nstderr:\n{stderr}")
    return stderr, stdout


def test_integration(tmp_path):
    # Step 0: Setup Environment
    input_dir = Path(tmp_path) / "input_dir"
    output_dir = Path(tmp_path) / "output_dir"
    input_label_dir = Path(tmp_path) / "label_dir"
    output_model_dir = Path(tmp_path) / "output_model_dir"

    shared_config = {
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "do_overwrite": False,
        "seed": 1,
        "hydra.verbose": True,
        "tqdm": False,
        "loguru_init": True,
    }

    describe_codes_config = {**shared_config}

    with initialize(version_base=None, config_path="../src/MEDS_tabular_automl/configs/"):
        overrides = [f"{k}={v}" for k, v in describe_codes_config.items()]
        cfg = compose(config_name="describe_codes", overrides=overrides)

    # Create the directories
    (output_dir).mkdir(parents=True, exist_ok=True)

    # Store MEDS outputs
    all_data = []
    for split, data in MEDS_OUTPUTS.items():
        file_path = input_dir / f"{split}.parquet"
        file_path.parent.mkdir(exist_ok=True, parents=True)
        df = pl.read_csv(StringIO(data)).with_columns(pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f"))
        df.write_parquet(file_path)
        all_data.append(df)
        assert file_path.exists()

    all_data = pl.concat(all_data, how="diagonal_relaxed").sort(by=["subject_id", "time"])

    # Check the files are not empty
    meds_files = list_subdir_files(Path(cfg.input_dir), "parquet")
    assert (
        len(list_subdir_files(Path(cfg.input_dir), "parquet")) == 4
    ), "MEDS train split Data Files Should be 4!"
    for f in meds_files:
        assert pl.read_parquet(f).shape[0] > 0, "MEDS Data Tabular Dataframe Should not be Empty!"
    split_json = json.load(StringIO(SPLITS_JSON))
    splits_fp = input_dir / ".shards.json"
    json.dump(split_json, splits_fp.open("w"))

    # Step 1: Run the describe_codes script
    stderr, stdout = run_command(
        "meds-tab-describe",
        [],
        describe_codes_config,
        "describe_codes",
    )

    assert Path(cfg.output_filepath).is_file()

    feature_columns = get_feature_columns(cfg.output_filepath)
    assert get_feature_names("code/count", feature_columns) == sorted(CODE_COLS)
    assert get_feature_names("static/present", feature_columns) == sorted(STATIC_PRESENT_COLS)
    assert get_feature_names("static/first", feature_columns) == sorted(STATIC_FIRST_COLS)
    for value_agg in VALUE_AGGREGATIONS:
        assert get_feature_names(value_agg, feature_columns) == sorted(VALUE_COLS)

    # Step 2: Run the static data tabularization script
    tabularize_static_config = {
        **shared_config,
        "tabularization.min_code_inclusion_count": 1,
        "tabularization.window_sizes": "[30d,365d,full]",
    }
    stderr, stdout = run_command(
        "meds-tab-tabularize-static",
        [],
        tabularize_static_config,
        "tabularization",
    )
    with initialize(version_base=None, config_path="../src/MEDS_tabular_automl/configs/"):
        overrides = [f"{k}={v}" for k, v in tabularize_static_config.items()]
        cfg = compose(config_name="tabularization", overrides=overrides)

    output_dir = Path(cfg.output_dir) / "tabularize"

    output_files = list(output_dir.glob("**/static/**/*.npz"))
    actual_files = [get_shard_prefix(output_dir, each) + ".npz" for each in output_files]
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
            f"Static Data matrix Should have {expected_num_rows}" f" rows but has {static_matrix.shape[0]}!"
        )
    allowed_codes = cfg.tabularization._resolved_codes
    num_allowed_codes = len(allowed_codes)
    feature_columns = get_feature_columns(cfg.tabularization.filtered_code_metadata_fp)
    assert num_allowed_codes == len(
        feature_columns
    ), f"Should have {len(feature_columns)} codes but has {num_allowed_codes}"

    # Step 3: Run the time series tabularization script
    tabularize_config = {
        **shared_config,
        "tabularization.min_code_inclusion_count": 1,
        "tabularization.window_sizes": "[30d,365d,full]",
    }

    stderr, stdout = run_command(
        "meds-tab-tabularize-time-series",
        ["--multirun", 'worker="range(0,1)"', "hydra/launcher=joblib"],
        tabularize_config,
        "tabularization",
    )

    # confirm summary files exist:
    output_files = list_subdir_files(str(output_dir.resolve()), "npz")
    actual_files = [
        get_shard_prefix(output_dir, each) + ".npz" for each in output_files if "none/static" not in str(each)
    ]
    assert len(actual_files) > 0
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
            f"Time-Series Data matrix Should have {expected_num_rows}" f" rows but has {ts_matrix.shape[0]}!"
        )
    output_files = list_subdir_files(str(Path(cfg.output_tabularized_dir).resolve()), "npz")
    for split in split_json.keys():
        for window in cfg.tabularization.window_sizes:
            for agg in cfg.tabularization.aggs:
                if agg.startswith("static"):
                    if window != cfg.tabularization.window_sizes[0]:
                        continue
                    expected_fp = Path(cfg.output_tabularized_dir) / split / "none" / f"{agg}.npz"
                else:
                    expected_fp = Path(cfg.output_tabularized_dir) / split / window / f"{agg}.npz"
                assert expected_fp in output_files, f"Missing {expected_fp}"
    expected_num_time_tabs = (
        NUM_SHARDS * len(cfg.tabularization.window_sizes) * (len(cfg.tabularization.aggs) - 2)
    )
    expected_num_static_tabs = NUM_SHARDS * 2
    assert len(list_subdir_files(cfg.output_dir, "npz")) == expected_num_time_tabs + expected_num_static_tabs

    # Step 4: Run the task_specific_caching script
    cache_config = {
        **shared_config,
        "tabularization.min_code_inclusion_count": 1,
        "tabularization.window_sizes": "[30d,365d,full]",
        "task_name": "test_task",
        "input_label_dir": str(input_label_dir.resolve()),
    }
    with initialize(version_base=None, config_path="../src/MEDS_tabular_automl/configs/"):
        overrides = [f"{k}={v}" for k, v in cache_config.items()]
        cfg = compose(config_name="task_specific_caching", overrides=overrides)

    # Create fake labels
    df = get_unique_time_events_df(get_events_df(all_data.lazy(), feature_columns)).collect()
    pseudo_labels = pl.Series(([0, 1] * df.shape[0])[: df.shape[0]])
    df = df.with_columns(pl.Series(name="boolean_value", values=pseudo_labels))
    df = df.select("subject_id", pl.col("time").alias("prediction_time"), "boolean_value")

    out_fp = Path(cfg.input_label_dir) / "0.parquet"
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_fp)

    stderr, stdout_ws = run_command("generate-subsets", ["[30d]"], {}, "generate-subsets window_sizes")
    stderr, stdout_agg = run_command(
        "generate-subsets", ["[static/present,static/first]"], {}, "generate-subsets aggs"
    )

    stderr, stdout = run_command(
        "meds-tab-cache-task",
        [
            "--multirun",
            f"tabularization.aggs={stdout_agg.strip()}",
        ],
        cache_config,
        "task_specific_caching",
    )
    for split in split_json.keys():
        for window in cfg.tabularization.window_sizes:
            for agg in cfg.tabularization.aggs:
                if agg.startswith("static"):
                    if window != cfg.tabularization.window_sizes[0]:
                        continue
                    expected_fp = Path(cfg.output_tabularized_cache_dir) / split / "none" / f"{agg}.npz"
                else:
                    expected_fp = Path(cfg.output_tabularized_cache_dir) / split / window / f"{agg}.npz"
                output_files = list_subdir_files(str(Path(cfg.output_tabularized_cache_dir).resolve()), "npz")
                assert expected_fp in output_files, f"Missing {expected_fp}"
    [each for each in output_files if "0/30d" in str(each) and "code/count" in str(each)]
    assert (
        len(list_subdir_files(cfg.output_tabularized_cache_dir, "npz"))
        == expected_num_time_tabs + expected_num_static_tabs
    )

    stderr, stdout = run_command(
        "meds-tab-cache-task",
        [
            "--multirun",
            f"tabularization.aggs={stdout_agg.strip()}",
        ],
        cache_config,
        "task_specific_caching",
    )

    for model in [
        "xgboost",
        "knn_classifier",
        "logistic_regression",
        "random_forest_classifier",
        "sgd_classifier",
    ]:
        model_config = {
            **shared_config,
            "tabularization.min_code_inclusion_count": 1,
            "tabularization.window_sizes": "[30d,365d,full]",
            "task_name": "test_task",
            "output_model_dir": str(output_model_dir.resolve()),
            "model_launcher": model,
            "path.model_file_stem": model,
            "hydra.sweeper.n_trials": 3,
            "delete_below_top_k": 2,
            "data_loading_params.keep_data_in_memory": True,
        }
        overrides = [f"tabularization.aggs={stdout_agg.strip()}"]
        if model == "autogluon":
            script = "meds-tab-autogluon"
        else:
            script = "meds-tab-model"
            overrides = ["--multirun"] + overrides

        stderr, stdout = run_command(script, overrides, model_config, f"launch_model_{model}")
        assert "Performance of best model:" in stderr
        if model == "xgboost":
            assert len(glob.glob(str(output_model_dir / "*/sweep_results/**/*.json"))) == 2
            assert len(glob.glob(str(output_model_dir / "*/best_trial/*.json"))) == 1

            time_output_dir = next(output_model_dir.iterdir())
            assert (time_output_dir / "best_trial/held_out_predictions.parquet").exists()
            assert (time_output_dir / "best_trial/tuning_predictions.parquet").exists()
            assert (time_output_dir / "sweep_results_summary.parquet").exists()
            validate_binary_classification_schema(
                pl.read_parquet(time_output_dir / "best_trial/held_out_predictions.parquet")
            )
        else:
            assert len(glob.glob(str(output_model_dir / "*/sweep_results/**/*.pkl"))) == 2
            assert len(glob.glob(str(output_model_dir / "*/best_trial/*.pkl"))) == 1
        shutil.rmtree(output_model_dir)

    for model in [
        "xgboost",
        "sgd_classifier",
    ]:
        model_config = {
            **shared_config,
            "tabularization.min_code_inclusion_count": 1,
            "tabularization.window_sizes": "[30d,365d,full]",
            "task_name": "test_task",
            "output_model_dir": str(output_model_dir.resolve()),
            "model_launcher": model,
            "path.model_file_stem": model,
            "hydra.sweeper.n_trials": 3,
            "delete_below_top_k": 2,
            "data_loading_params.keep_data_in_memory": False,
        }
        overrides = [f"tabularization.aggs={stdout_agg.strip()}"]
        if model == "autogluon":
            script = "meds-tab-autogluon"
        else:
            script = "meds-tab-model"
            overrides = ["--multirun"] + overrides

        stderr, stdout = run_command(script, overrides, model_config, f"launch_model_{model}")
        assert "Performance of best model:" in stderr
        if model == "xgboost":
            assert len(glob.glob(str(output_model_dir / "*/sweep_results/**/*.json"))) == 2
            assert len(glob.glob(str(output_model_dir / "*/best_trial/*.json"))) == 1

            time_output_dir = next(output_model_dir.iterdir())
            assert (time_output_dir / "best_trial/held_out_predictions.parquet").exists()
            assert (time_output_dir / "best_trial/tuning_predictions.parquet").exists()
            assert (time_output_dir / "sweep_results_summary.parquet").exists()
            validate_binary_classification_schema(
                pl.read_parquet(time_output_dir / "best_trial/held_out_predictions.parquet")
            )
        else:
            assert len(glob.glob(str(output_model_dir / "*/sweep_results/**/*.pkl"))) == 2
            assert len(glob.glob(str(output_model_dir / "*/best_trial/*.pkl"))) == 1

        shutil.rmtree(output_model_dir)
