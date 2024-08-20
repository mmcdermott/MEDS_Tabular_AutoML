from importlib.resources import files

import hydra
import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from MEDS_tabular_automl.dense_iterator import DenseIterator

from ..utils import hydra_loguru_init

config_yaml = files("MEDS_tabular_automl").joinpath("configs/launch_xgboost.yaml")
if not config_yaml.is_file():
    raise FileNotFoundError("Core configuration not successfully installed!")


@hydra.main(version_base=None, config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig) -> float:
    """Launches AutoGluon after collecting data based on the provided configuration.

    Args:
        cfg: The configuration dictionary specifying model and training parameters.
    """

    # print(OmegaConf.to_yaml(cfg))
    if not cfg.loguru_init:
        hydra_loguru_init()

    # check that autogluon is installed
    try:
        import autogluon as ag
    except ImportError:
        logger.error("AutoGluon is not installed. Please install AutoGluon.")

    # collect data based on the configuration
    itrain = DenseIterator(cfg, "train")
    ituning = DenseIterator(cfg, "tuning")
    iheld_out = DenseIterator(cfg, "held_out")

    # collect data for AutoGluon
    train_data, train_labels, cols = itrain.densify()
    tuning_data, tuning_labels, _ = ituning.densify()
    held_out_data, held_out_labels, _ = iheld_out.densify()

    # construct dfs for AutoGluon
    train_df = pd.DataFrame(train_data.todense(), columns=cols)
    train_df[cfg.task_name] = train_labels
    tuning_df = pd.DataFrame(tuning_data.todense(), columns=cols)
    tuning_df[cfg.task_name] = tuning_labels
    held_out_df = pd.DataFrame(held_out_data.todense(), columns=cols)
    held_out_df[cfg.task_name] = held_out_labels

    # launch AutoGluon
    predictor = ag.TabularPredictor(label=cfg.task_name).fit(train_data=train_df, tuning_data=tuning_df)
    # TODO: fix logging, etc.
    auc = predictor.evaluate(held_out_df)
    logger.info(f"AUC: {auc}")


if __name__ == "__main__":
    main()
