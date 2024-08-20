from typing import Dict, Type
from abc import ABC, abstractmethod
from pathlib import Path
from omegaconf import DictConfig
from mixins import TimeableMixin


class BaseModel(ABC, TimeableMixin):
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
