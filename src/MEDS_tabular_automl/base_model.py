from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar

from mixins import TimeableMixin
from omegaconf import DictConfig

T = TypeVar("T")


class BaseModel(ABC, TimeableMixin):
    """Defines the interface for a model that can be trained and evaluated via the launch_model script."""

    @abstractmethod
    def __init__(self):  # pragma: no cover
        pass

    @abstractmethod
    def train(self):  # pragma: no cover
        pass

    @abstractmethod
    def evaluate(self) -> float:  # pragma: no cover
        pass

    @abstractmethod
    def save_model(self, output_fp: Path):  # pragma: no cover
        pass

    @classmethod
    def initialize(cls: T, **kwargs) -> T:
        return cls(DictConfig(kwargs, flags={"allow_objects": True}))
