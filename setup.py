"""Builds the native engine (csrc/) into the package as MEDS_tabular_automl.native._engine.

The engine is a C shared library that ``native/engine.py`` loads with ctypes. It is declared
as an optional extension module, so installing MEDS-Tab compiles it when a C++17 compiler and
the zlib headers are available and installs the package without it otherwise. Nothing loads it
unless ``backend=native`` is selected. All other metadata is in pyproject.toml.

MEDS_TAB_BUILD_ENGINE=0 builds the package without the engine. The PyPI workflow uses it for the
published wheel, which stays pure Python; the published source distribution includes the engine.
"""

import os
from pathlib import Path

from setuptools import Extension, setup

engine = Extension(
    "MEDS_tabular_automl.native._engine",
    sources=sorted(str(path) for path in Path("csrc").glob("*.cc")),
    include_dirs=["csrc"],
    language="c++",
    # -ffp-contract=off: no fused multiply-adds, so sums and squares round exactly as in NumPy.
    extra_compile_args=["-std=c++17", "-O2", "-ffp-contract=off", "-pthread"],
    extra_link_args=["-pthread"],
    libraries=["z"],
    optional=True,
)

setup(ext_modules=[] if os.environ.get("MEDS_TAB_BUILD_ENGINE") == "0" else [engine])
