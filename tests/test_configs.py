import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import subprocess

import hydra
import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from loguru import logger

from MEDS_tabular_automl.sklearn_model import SklearnModel
from MEDS_tabular_automl.xgboost_model import XGBoostModel

logger.disable("MEDS_tabular_automl")
from omegaconf import OmegaConf


def run_command(script: str, args: list[str], hydra_kwargs: dict[str, str], test_name: str):
    command_parts = [script] + args + [f"{k}={v}" for k, v in hydra_kwargs.items()]
    command_out = subprocess.run(" ".join(command_parts), shell=True, capture_output=True)
    stderr = command_out.stderr.decode()
    stdout = command_out.stdout.decode()
    if command_out.returncode != 0:
        raise AssertionError(f"{test_name} failed!\nstdout:\n{stdout}\nstderr:\n{stderr}")
    return stderr, stdout


def make_config_mutable(cfg):
    OmegaConf.set_readonly(cfg, False)
    for key in cfg:
        if isinstance(cfg[key], OmegaConf):
            make_config_mutable(cfg[key])


@pytest.mark.parametrize("model", ["xgboost", "sgd_classifier"])
def test_model_config(model):
    MEDS_cohort_dir = "blah"
    xgboost_config_kwargs = {
        "MEDS_cohort_dir": MEDS_cohort_dir,
        "output_cohort_dir": "blah",
        "do_overwrite": False,
        "seed": 1,
        "hydra.verbose": True,
        "tqdm": False,
        "loguru_init": True,
        "tabularization.min_code_inclusion_count": 1,
        "tabularization.window_sizes": "[30d,365d,full]",
        "tabularization._resolved_codes": "[test,test2]",
    }

    with initialize(
        version_base=None, config_path="../src/MEDS_tabular_automl/configs/"
    ):  # path to config.yaml
        overrides = [f"model={model}"] + [f"{k}={v}" for k, v in xgboost_config_kwargs.items()]
        cfg = compose(
            config_name="launch_model", overrides=overrides, return_hydra_config=True
        )  # config.yaml

    HydraConfig().set_config(cfg)
    # make_config_mutable(cfg)
    expected_model_class = XGBoostModel if model == "xgboost" else SklearnModel
    model = hydra.utils.instantiate(cfg.model_target)
    assert isinstance(model, expected_model_class)
    # assert cfg.tabularization.window_sizes
