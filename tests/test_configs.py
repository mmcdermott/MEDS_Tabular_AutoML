import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)


import hydra
import polars as pl
import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

from MEDS_tabular_automl.sklearn_model import SklearnModel
from MEDS_tabular_automl.xgboost_model import XGBoostModel
from tests.test_integration import run_command


@pytest.mark.parametrize(
    "model_launcher_override",
    [
        "xgboost",
        "sgd_classifier",
        "knn_classifier",
        "logistic_regression",
        "random_forest_classifier",
        "autogluon",
    ],
)
@pytest.mark.parametrize("imputer", ["default", "mean_imputer", "mode_imputer", "median_imputer"])
@pytest.mark.parametrize("normalization", ["standard_scaler", "max_abs_scaler"])
def test_model_config(model_launcher_override, imputer, normalization, tmp_path):
    input_dir = "/foo/"
    code_metadata_fp = f"/{str(tmp_path)}/codes.parquet"
    model_launcher_config_kwargs = {
        "input_dir": input_dir,
        "output_dir": "/bar/",
        "output_model_dir": "/baz/",
        "++tabularization.filtered_code_metadata_fp": code_metadata_fp,
        "++tabularization.min_code_inclusion_count": "0",
        "task_name": "foo_bar",
    }
    pl.DataFrame({"code": ["E", "D", "A"], "count": [4, 3, 2]}).write_parquet(code_metadata_fp)

    with initialize(version_base=None, config_path="../src/MEDS_tabular_automl/configs/"):
        overrides = [
            f"model_launcher={model_launcher_override}",
            f"data_processing_params.imputer={imputer}",
            f"data_processing_params.normalization={normalization}",
        ] + [f"{k}={v}" for k, v in model_launcher_config_kwargs.items()]
        cfg = compose(config_name="launch_model", overrides=overrides, return_hydra_config=True)

    model_launcher = hydra.utils.instantiate(cfg.model_launcher)
    match model_launcher_override:
        case "xgboost":
            assert isinstance(
                model_launcher, XGBoostModel
            ), "model_launcher should be an instance of XGBoostModel"
        case "autogluon":
            assert isinstance(
                model_launcher, DictConfig
            ), "model_launcher should not be a DictConfig for autogluon"
        case _:
            assert isinstance(
                model_launcher, SklearnModel
            ), "model_launcher should be an instance of SklearnModel"
    assert cfg.tabularization.window_sizes


def test_generate_subsets_configs():
    input_dir = "blah"
    stderr, stdout_ws = run_command("generate-subsets", ["[30d]"], {}, "generate-subsets window_sizes")
    stderr, stdout_agg = run_command("generate-subsets", ["[static/present]"], {}, "generate-subsets aggs")
    xgboost_config_kwargs = {
        "input_dir": input_dir,
        "output_dir": "blah",
        "do_overwrite": False,
        "seed": 1,
        "hydra.verbose": True,
        "tqdm": False,
        "loguru_init": True,
        "tabularization.min_code_inclusion_count": 1,
        "tabularization.window_sizes": f"{stdout_ws.strip()}",
    }

    with initialize(version_base=None, config_path="../src/MEDS_tabular_automl/configs/"):
        overrides = [f"{k}={v}" for k, v in xgboost_config_kwargs.items()]
        cfg = compose(config_name="launch_model", overrides=overrides)
    assert cfg.tabularization.window_sizes
