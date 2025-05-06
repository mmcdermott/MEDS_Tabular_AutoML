from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from mixins import TimeableMixin
from omegaconf import DictConfig

if TYPE_CHECKING:
    from pathlib import Path


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
    def initialize(cls: BaseModel, **kwargs) -> BaseModel:
        return cls(DictConfig(kwargs, flags={"allow_objects": True}))
