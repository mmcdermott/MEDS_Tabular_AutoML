"""Test set-up and fixtures code."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import polars as pl
import pytest
import scipy.sparse as sp
from omegaconf import DictConfig, OmegaConf


@pytest.fixture(autouse=True)
def _setup_doctest_namespace(doctest_namespace: dict[str, Any]):
    doctest_namespace.update(
        {
            "Path": Path,
            "sp": sp,
            "DictConfig": DictConfig,
            "OmegaConf": OmegaConf,
            "MagicMock": MagicMock,
            "Mock": Mock,
            "patch": patch,
            "json": json,
            "pl": pl,
            "datetime": datetime,
            "tempfile": tempfile,
        }
    )
