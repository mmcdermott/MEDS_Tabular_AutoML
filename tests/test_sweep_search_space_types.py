"""Every sweep parameter of every shipped model_launcher config must sample the values its config declares.

This builds each config's whole search space the way Hydra's optuna sweeper does and samples it with optuna:

- a `tag(log, ...)` parameter must be sampled log-uniformly: the median draw is near the geometric mean of
  its bounds, not their arithmetic mean;
- an integer parameter must be sampled as integers. A parameter is an integer one if its default in the
  composed `launch_model` config is an int, or if it is swept with `range(...)`. The syntax of the bounds is
  not enough: `interval(0, 1)` is a float `l1_ratio`, and Hydra parses `interval(10, 1000000)` as floats.
"""

import math
from pathlib import Path

import numpy as np
import optuna
import pytest
import yaml
from hydra import compose, initialize
from hydra_plugins.hydra_optuna_sweeper._impl import create_params_from_overrides
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).parent.parent / "src" / "MEDS_tabular_automl" / "configs"
LAUNCHERS = sorted(p.stem for p in (CONFIG_DIR / "model_launcher").glob("*.yaml"))
NUM_DRAWS = 2000


def sweep_params(launcher: str) -> dict[str, str]:
    config = yaml.safe_load((CONFIG_DIR / "model_launcher" / f"{launcher}.yaml").read_text()) or {}
    return ((config.get("hydra") or {}).get("sweeper") or {}).get("params") or {}


def defaults(launcher: str):
    with initialize(version_base=None, config_path="../src/MEDS_tabular_automl/configs/"):
        return compose(config_name="launch_model", overrides=[f"model_launcher={launcher}"])


def draws(distribution, key: str) -> list:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    values = []
    for _ in range(NUM_DRAWS):
        trial = study.ask({key: distribution})
        values.append(trial.params[key])
        study.tell(trial, 0.0)
    return values


def parameters():
    found = []
    for launcher in LAUNCHERS:
        cfg = defaults(launcher)
        params = sweep_params(launcher)
        space, _ = create_params_from_overrides([f"{k}={v}" for k, v in params.items()])
        for key, sweep in params.items():
            default = OmegaConf.select(cfg, key.lstrip("+"), default=None, throw_on_missing=False)
            int_default = isinstance(default, int) and not isinstance(default, bool)
            is_integer = int_default or sweep.strip().startswith("range(")
            distribution = space[key] if key in space else space[key.lstrip("+")]
            found.append(pytest.param(distribution, key, sweep, is_integer, id=f"{launcher}-{key}"))
    return found


def test_there_are_log_tagged_and_integer_parameters():
    """Guard the guard: without both kinds of parameter, the tests below would skip everything."""
    found = [p.values for p in parameters()]
    assert any("tag(log" in sweep for _, _, sweep, _ in found)
    assert any(is_integer for _, _, _, is_integer in found)


@pytest.mark.parametrize("distribution, key, sweep, is_integer", parameters())
def test_log_tagged_parameters_sample_log_uniformly(distribution, key, sweep, is_integer):
    if "tag(log" not in sweep:
        pytest.skip("not log-tagged")
    values = np.array(draws(distribution, key), dtype=float)
    geometric, arithmetic = (
        math.sqrt(distribution.low * distribution.high),
        (distribution.low + distribution.high) / 2,
    )
    median = float(np.median(values))
    # Log-uniform: the median draw is the geometric mean of the bounds, far from the arithmetic one.
    assert abs(math.log(median / geometric)) < 0.25, (
        f"`{key}: {sweep}` builds {type(distribution).__name__}; median draw {median:.4g}, "
        f"geometric mean of the bounds {geometric:.4g}, arithmetic mean {arithmetic:.4g}"
    )


@pytest.mark.parametrize("distribution, key, sweep, is_integer", parameters())
def test_integer_parameters_sample_integers(distribution, key, sweep, is_integer):
    if not is_integer:
        pytest.skip("not an integer parameter")
    values = draws(distribution, key)
    non_integers = [v for v in values if not isinstance(v, int)]
    assert not non_integers, (
        f"`{key}: {sweep}` is an integer parameter, but the sweeper builds {type(distribution).__name__} and "
        f"draws values like {non_integers[:3]}. Wrap an interval in int(...), e.g. "
        "tag(log, int(interval(a, b)))."
    )
