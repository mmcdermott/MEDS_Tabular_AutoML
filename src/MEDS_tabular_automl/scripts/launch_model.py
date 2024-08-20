from importlib.resources import files
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

from MEDS_tabular_automl.base_model import BaseModel
from MEDS_tabular_automl.sklearn_model import SklearnModel
from MEDS_tabular_automl.xgboost_model import XGBoostModel

MODEL_CLASSES: dict[str, type[BaseModel]] = {"xgboost": XGBoostModel, "sklearn": SklearnModel}

from ..utils import hydra_loguru_init

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
        model_type = cfg.model.type
        ModelClass = MODEL_CLASSES.get(model_type)
        if ModelClass is None:
            raise ValueError(f"Model type {model_type} not supported.")

        model = ModelClass(cfg)
        model.train()
        auc = model.evaluate()
        logger.info(f"AUC: {auc}")

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
