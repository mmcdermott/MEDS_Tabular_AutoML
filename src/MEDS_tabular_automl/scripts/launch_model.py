from importlib.resources import files
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, open_dict

from MEDS_tabular_automl.base_model import BaseModel

from ..utils import hydra_loguru_init

config_yaml = files("MEDS_tabular_automl").joinpath("configs/launch_model.yaml")
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

    model: BaseModel = hydra.utils.instantiate(cfg.model_target)
    # TODO - make tabularuzation be copied in the yaml instead of here
    with open_dict(cfg):
        model.cfg.tabularization = hydra.utils.instantiate(cfg.tabularization)

    model.train()
    auc = model.evaluate()
    logger.info(f"AUC: {auc}")

    # save model
    output_fp = Path(cfg.output_filepath)
    output_fp.parent.mkdir(parents=True, exist_ok=True)

    model.save_model(output_fp)
    return auc


if __name__ == "__main__":
    main()
