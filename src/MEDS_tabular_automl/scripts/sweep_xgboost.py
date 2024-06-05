import warnings
from copy import deepcopy
from importlib.resources import files
from itertools import combinations

import hydra
import optuna
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict

from MEDS_tabular_automl.scripts import launch_xgboost

warnings.filterwarnings("ignore", category=UserWarning)

config_yaml = files("MEDS_tabular_automl").joinpath("configs/launch_xgboost.yaml")
if not config_yaml.is_file():
    raise FileNotFoundError("Core configuration not successfully installed!")


def generate_permutations(list_of_options):
    """Generate all possible permutations of a list of options.

    Args:
    - list_of_options (list): List of options.

    Returns:
    - list: List of all possible permutations of length > 1
    """
    permutations = []
    for i in range(1, len(list_of_options) + 1):
        permutations.extend(list(combinations(list_of_options, r=i)))
    return permutations


OmegaConf.register_new_resolver("generate_permutations", generate_permutations)


def xgboost_singleton(trial: optuna.Trial, config: DictConfig) -> float:
    for key, value in config.optuna.params.suggest_categorical.items():
        logger.info(f"Optimizing {key} with {value}")
        config.tabularization[key] = trial.suggest_categorical(key, value)
    for key, value in config.optuna.params.suggest_float.items():
        with open_dict(config):
            config[key] = trial.suggest_float(key, **value)
    for key, value in config.optuna.params.suggest_int.items():
        with open_dict(config):
            config[key] = trial.suggest_int(key, **value)
    return launch_xgboost.main(config)


@hydra.main(version_base=None, config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig) -> None:
    study = optuna.create_study(
        study_name=cfg.optuna.study_name,
        storage=cfg.optuna.storage,
        load_if_exists=cfg.optuna.load_if_exists,
        direction=cfg.optuna.direction,
        sampler=cfg.optuna.sampler,
        pruner=cfg.optuna.pruner,
    )
    study.optimize(
        lambda trial: xgboost_singleton(trial, deepcopy(cfg)),
        n_trials=cfg.optuna.n_trials,
        n_jobs=cfg.optuna.n_jobs,
        show_progress_bar=cfg.optuna.show_progress_bar,
    )
    print(
        "Number of finished trials: ",
        len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
    )
    print(
        "Number of pruned trials: ",
        len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
    )
    print("Sampler:", study.sampler)
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
