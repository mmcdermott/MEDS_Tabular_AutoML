from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files

from omegaconf import OmegaConf

from .utils import filter_to_codes

__package_name__ = "MEDS_tabular_automl"
try:
    __version__ = version(__package_name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

CACHE_TASK_CFG = files(__package_name__).joinpath("configs/task_specific_caching.yaml")
TABULARIZATION_CFG = files(__package_name__).joinpath("configs/tabularization.yaml")
DESCRIBE_CODES_CFG = files(__package_name__).joinpath("configs/describe_codes.yaml")
LAUNCH_MODEL_CFG = files(__package_name__).joinpath("configs/launch_model.yaml")

OmegaConf.register_new_resolver("filter_to_codes", filter_to_codes, replace=True)
