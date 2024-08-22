import time
from importlib.resources import files
from pathlib import Path

import hydra
from omegaconf import DictConfig, open_dict

from MEDS_tabular_automl.base_model import BaseModel
from MEDS_tabular_automl.mapper import wrap as rwlock_wrap

from ..utils import hydra_loguru_init

config_yaml = files("MEDS_tabular_automl").joinpath("configs/launch_model.yaml")
if not config_yaml.is_file():
    raise FileNotFoundError("Core configuration not successfully installed!")


def log_to_logfile(model, cfg, output_fp):
    """Log model hyperparameters and performance to two log files."""
    log_fp = Path(cfg.model_log_dir)
    log_fp.mkdir(parents=True, exist_ok=True)
    # log hyperparameters
    out_fp = log_fp / "trial_performance_results.log"

    def write_fn(_, out_fp):
        with open(out_fp, "a") as f:
            f.write(
                f"{output_fp}\t{cfg.tabularization}\t{cfg.model_params}"
                f"\t{model.evaluate()}\t{model.evaluate(split='held_out')}\n"
            )

    rwlock_wrap(
        None,
        out_fp,
        lambda _: None,  # read_fn is ignored
        write_fn,
        cache_intermediate=True,
        clear_cache_on_completion=True,
        do_overwrite=True,
        do_return=False,
    )


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
    # logger.info(f"AUC: {auc}")

    # save model
    output_fp = Path(cfg.output_filepath)
    output_fp = output_fp.parent / f"{output_fp.stem}_{auc:.4f}_{time.time()}{output_fp.suffix}"
    output_fp.parent.mkdir(parents=True, exist_ok=True)

    # log to logfile
    log_to_logfile(model, cfg, output_fp)

    model.save_model(output_fp)
    return auc


if __name__ == "__main__":
    main()
