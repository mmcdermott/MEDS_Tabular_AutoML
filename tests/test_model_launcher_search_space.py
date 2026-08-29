"""A `tag(log, ...)` sweep parameter must actually be searched log-uniformly.

Hydra's optuna sweeper reads the `log` tag only for an `interval(...)` sweep. For a `range(...)` sweep
it returns an `IntUniformDistribution` and the tag is silently discarded -- see
`hydra_plugins.hydra_optuna_sweeper._impl.create_optuna_distribution_from_override`, where `log` is
inspected inside the `is_interval_sweep()` branch and nowhere else.

That silence is the whole problem: the config says log, the sweep runs uniform, and nothing warns. For
`tabularization.min_code_inclusion_count` over [10, 1000000] the difference decides whether the search
is usable at all -- a uniform draw has a median around 500,000, so on most datasets nearly every trial
selects no codes, `launch_model` catches the resulting error and returns 0.0 AUC, no model is written,
and the evaluation callback's failure is downgraded by hydra to a warning, leaving the run to exit 0
with nothing to show for it.

This test is written over *every* log-tagged parameter in *every* shipped model_launcher config rather
than over the one that was wrong, so the same mistake cannot reappear elsewhere unnoticed.
"""

from pathlib import Path

import pytest
import yaml

from hydra_plugins.hydra_optuna_sweeper._impl import create_params_from_overrides

CONFIG_DIR = Path(__file__).parent.parent / "src" / "MEDS_tabular_automl" / "configs" / "model_launcher"

# optuna 2 expresses a logarithmic search as a dedicated class; optuna 3 folds it into a `log` flag on
# Int/FloatDistribution. Checking for either keeps this test meaningful across both rather than
# failing on an import and looking like the config is at fault.
LOG_DISTRIBUTION_NAMES = {"IntLogUniformDistribution", "LogUniformDistribution"}


def is_logarithmic(distribution) -> bool:
    return bool(getattr(distribution, "log", False)) or (
        type(distribution).__name__ in LOG_DISTRIBUTION_NAMES
    )


def log_tagged_parameters():
    """Every `key: tag(log, ...)` sweeper parameter shipped in a model_launcher config."""
    found = []
    for config in sorted(CONFIG_DIR.glob("*.yaml")):
        params = ((yaml.safe_load(config.read_text()) or {}).get("hydra") or {}).get("sweeper") or {}
        for key, value in (params.get("params") or {}).items():
            if isinstance(value, str) and "tag(log" in value:
                found.append(pytest.param(config.name, key, value, id=f"{config.stem}-{key}"))
    return found


def test_at_least_one_log_tagged_parameter_exists():
    """Guard the guard: an empty parameter list would make every test below vacuous."""
    assert log_tagged_parameters()


@pytest.mark.parametrize("config_name, key, value", log_tagged_parameters())
def test_log_tagged_parameters_are_searched_log_uniformly(config_name, key, value):
    space, _ = create_params_from_overrides([f"{key}={value}"])
    distribution = space[key]
    assert is_logarithmic(distribution), (
        f"{config_name} declares `{key}: {value}` but the sweeper builds "
        f"{type(distribution).__name__}, so the search is not logarithmic. Hydra's optuna sweeper "
        "honours the log tag only for `interval(...)`, never for `range(...)`."
    )
