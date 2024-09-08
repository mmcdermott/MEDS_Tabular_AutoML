import time
from importlib.resources import files
from pathlib import Path

import hydra
from omegaconf import DictConfig

from MEDS_tabular_automl.base_model import BaseModel

from ..utils import hydra_loguru_init, log_to_logfile, stage_init

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
    stage_init(
        cfg, ["input_dir", "input_label_dir", "output_dir", "tabularization.filtered_code_metadata_fp"]
    )

    if not cfg.loguru_init:
        hydra_loguru_init()

    model: BaseModel = hydra.utils.instantiate(cfg.model_target)

    model.train()
    auc = model.evaluate()

    # save model
    output_fp = Path(cfg.model_saving.model_dir)
    output_fp = (
        output_fp.parent
        / f"{cfg.model_saving.model_file_stem}_{auc:.4f}_{time.time()}{cfg.model_target.model_file_extension}"
    )
    output_fp.parent.mkdir(parents=True, exist_ok=True)

    # log to logfile
    log_to_logfile(model, cfg, output_fp.stem)

    model.save_model(output_fp)
    return auc


if __name__ == "__main__":
    main()
