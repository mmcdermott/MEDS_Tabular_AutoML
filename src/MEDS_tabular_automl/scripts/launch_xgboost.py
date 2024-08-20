from importlib.resources import files
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

from ..utils import hydra_loguru_init
from ..xgboost_model import XGBoostModel

config_yaml = files("MEDS_tabular_automl").joinpath("configs/launch_xgboost.yaml")
if not config_yaml.is_file():
    raise FileNotFoundError("Core configuration not successfully installed!")


@hydra.main(version_base=None, config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig) -> float:
    """Optimizes the model based on the provided configuration.

    Args:
        cfg: The configuration dictionary specifying model and training parameters.

    Returns:
        The evaluation result as the ROC AUC score on the held-out test set.
    """

    # print(OmegaConf.to_yaml(cfg))
    if not cfg.loguru_init:
        hydra_loguru_init()
    try:
        model = XGBoostModel(cfg)
        model.train()
        auc = model.evaluate()
        logger.info(f"AUC: {auc}")

        # print(
        #     "Time Profiling for window sizes ",
        #     f"{cfg.tabularization.window_sizes} and min ",
        #     f"code frequency of {cfg.tabularization.min_code_inclusion_frequency}:",
        # )
        # print("Train Time: \n", model._profile_durations())
        # print("Train Iterator Time: \n", model.itrain._profile_durations())
        # print("Tuning Iterator Time: \n", model.ituning._profile_durations())
        # print("Held Out Iterator Time: \n", model.iheld_out._profile_durations())

        # save model
        output_fp = Path(cfg.output_filepath)
        output_fp.parent.mkdir(parents=True, exist_ok=True)

        model.save_model(output_fp)
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        auc = 0.0
    return auc


if __name__ == "__main__":
    main()
