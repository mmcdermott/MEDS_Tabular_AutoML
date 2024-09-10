import json
from importlib.resources import files
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_tabular_automl.base_model import BaseModel

from ..utils import hydra_loguru_init, stage_init

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
        cfg, ["input_dir", "input_label_cache_dir", "output_dir", "tabularization.filtered_code_metadata_fp"]
    )

    if not cfg.loguru_init:
        hydra_loguru_init()

    try:
        cfg.tabularization._resolved_codes
    except ValueError as e:
        logger.warning(f"No codes meet loading criteria, trial returning 0 AUC: {str(e)}")
        return 0.0

    model_launcher: BaseModel = hydra.utils.instantiate(cfg.model_launcher)

    model_launcher.train()
    auc = model_launcher.evaluate()

    # Make output model directory
    path_cfg = model_launcher.cfg.path
    model_filename = f"{path_cfg.model_file_stem}{path_cfg.model_file_extension}"
    model_config_hash = abs(hash(json.dumps(OmegaConf.to_container(cfg), sort_keys=True)))
    trial_output_dir = Path(path_cfg.sweep_results_dir) / str(model_config_hash)
    trial_output_dir.mkdir(parents=True, exist_ok=True)

    # save model
    model_launcher.save_model(trial_output_dir / model_filename)

    # save model config
    config_fp = trial_output_dir / f"{cfg.path.config_log_stem}.log"
    with open(config_fp, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # save model performance
    model_performance_fp = trial_output_dir / f"{cfg.path.performance_log_stem}.log"
    with open(model_performance_fp, "w") as f:
        f.write("trial_name,tuning_auc,test_auc\n")
        f.write(
            f"{trial_output_dir.stem},{model_launcher.evaluate()},"
            f"{model_launcher.evaluate(split='held_out')}\n"
        )

    logger.debug(f"Model config and performance logged to {config_fp} and {model_performance_fp}")
    return auc


if __name__ == "__main__":
    main()
