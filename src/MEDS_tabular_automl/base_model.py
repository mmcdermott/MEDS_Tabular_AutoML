from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar

from mixins import TimeableMixin
from omegaconf import DictConfig

T = TypeVar("T")


class BaseModel(ABC, TimeableMixin):
    """Defines the interface for a model that can be trained and evaluated via the launch_model script."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self) -> float:
        pass

    @abstractmethod
    def save_model(self, output_fp: Path):
        pass

    @classmethod
    def initialize(cls: T, **kwargs) -> T:
        return cls(DictConfig(kwargs, flags={"allow_objects": True}))
