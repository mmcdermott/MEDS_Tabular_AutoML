import json
from importlib.resources import files
from pathlib import Path

import hydra
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf

try:
    import autogluon.tabular as ag
except ImportError:
    ag = None

from MEDS_tabular_automl.tabular_dataset import TabularDataset as DenseIterator

from ..utils import hydra_loguru_init, stage_init

config_yaml = files("MEDS_tabular_automl").joinpath("configs/launch_model.yaml")
if not config_yaml.is_file():
    raise FileNotFoundError("Core configuration not successfully installed!")


def check_autogluon():
    if ag is None:
        raise ImportError(
            "AutoGluon could not be imported. Please try installing it using: `pip install autogluon`"
        )


@hydra.main(version_base=None, config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig) -> float:
    """Launches AutoGluon after collecting data based on the provided configuration.

    Args:
        cfg: The configuration dictionary specifying model and training parameters.
    """
    check_autogluon()
    stage_init(
        cfg, ["input_dir", "input_label_cache_dir", "output_dir", "tabularization.filtered_code_metadata_fp"]
    )
    if not cfg.loguru_init:
        hydra_loguru_init()

    # collect data based on the configuration
    itrain = DenseIterator(cfg, "train")
    ituning = DenseIterator(cfg, "tuning")
    iheld_out = DenseIterator(cfg, "held_out")

    # collect data for AutoGluon
    train_data, train_labels = itrain.densify()
    tuning_data, tuning_labels = ituning.densify()
    held_out_data, held_out_labels = iheld_out.densify()

    # construct dfs for AutoGluon
    train_df = pd.DataFrame(train_data.todense())
    train_df[cfg.task_name] = train_labels
    tuning_df = pd.DataFrame(
        tuning_data.todense(),
    )
    tuning_df[cfg.task_name] = tuning_labels
    held_out_df = pd.DataFrame(held_out_data.todense())
    held_out_df[cfg.task_name] = held_out_labels

    train_dataset = ag.TabularDataset(train_df)
    tuning_dataset = ag.TabularDataset(tuning_df)
    held_out_dataset = ag.TabularDataset(held_out_df)

    # train model with AutoGluon
    log_filepath = Path(cfg.path.sweep_results_dir) / f"{cfg.path.config_log_stem}_log.txt"

    predictor = ag.TabularPredictor(
        label=cfg.task_name,
        log_to_file=True,
        log_file_path=str(log_filepath.resolve()),
        path=cfg.time_output_model_dir,
    ).fit(train_data=train_dataset, tuning_data=tuning_dataset)

    # predict
    predictions = predictor.predict(held_out_dataset.drop(columns=[cfg.task_name]))
    logger.info("Predictions:", predictions)
    # evaluate
    score = predictor.evaluate(held_out_dataset)
    logger.info("Test score:", score)

    model_performance_log_filepath = (
        Path(cfg.path.sweep_results_dir) / f"{cfg.path.performance_log_stem}.json"
    )
    model_performance_log_filepath.parent.mkdir(parents=True, exist_ok=True)
    # store results
    performance_dict = {
        "output_model_dir": cfg.path.time_output_model_dir,
        "tabularization": OmegaConf.to_container(cfg.tabularization),
        "model_launcher": OmegaConf.to_container(cfg.model_launcher),
        "score": score,
    }
    with open(model_performance_log_filepath, "w") as f:
        json.dump(performance_dict, f)


if __name__ == "__main__":
    main()
