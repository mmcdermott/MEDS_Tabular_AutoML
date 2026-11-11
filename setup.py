"""Builds the native engine (csrc/) into the package as MEDS_tabular_automl.native._engine.

The engine is a C shared library that ``native/engine.py`` loads with ctypes. It is declared
as an optional extension module, so installing MEDS-Tab compiles it when a C++17 compiler and
the zlib headers are available and installs the package without it otherwise. Nothing loads it
unless ``backend=native`` is selected. All other metadata is in pyproject.toml.

MEDS_TAB_BUILD_ENGINE=0 builds the package without the engine. The PyPI workflow uses it for the
published wheel, which stays pure Python; the published source distribution includes the engine.
"""

import os
import shutil
import subprocess
import sys
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


def libomp_prefix() -> Path | None:
    """LLVM's OpenMP runtime for Apple clang: $LIBOMP_PREFIX, else Homebrew's libomp."""
    candidates = [os.environ.get("LIBOMP_PREFIX")]
    if shutil.which("brew"):
        try:
            out = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True, text=True, timeout=30)
            candidates.append(out.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    candidates += ["/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"]
    for prefix in candidates:
        if prefix and (Path(prefix) / "include" / "omp.h").exists():
            return Path(prefix)
    return None


def sparsegb_extension() -> Extension | None:
    """sparsegb (model_launcher=sparsegb): a separate optional library, since it needs OpenMP.

    Linux: GCC, or clang with LLVM's libomp. macOS: Apple clang with libomp (`brew install libomp`, or set
    LIBOMP_PREFIX, e.g. to a conda environment with llvm-openmp), or GCC (CC=gcc-14 ...). Windows is not
    supported yet (MSVC lacks the GCC builtins and flags it uses).
    """
    if sys.platform == "win32":
        print("sparsegb: Windows is not supported yet; building MEDS-Tab without it.")
        return None
    compile_args, link_args = ["-std=c++17", "-O3"], []
    include_dirs, library_dirs = [], []
    apple_clang = sys.platform == "darwin" and "gcc" not in os.environ.get("CC", "")
    if apple_clang:  # Apple clang's driver has no -fopenmp; hand it to the preprocessor and link libomp
        prefix = libomp_prefix()
        if prefix is None:
            print("sparsegb: no libomp found (brew install libomp, or set LIBOMP_PREFIX); skipping it.")
            return None
        compile_args += ["-Xpreprocessor", "-fopenmp"]
        link_args += ["-lomp", f"-Wl,-rpath,{prefix / 'lib'}"]
        include_dirs, library_dirs = [str(prefix / "include")], [str(prefix / "lib")]
    else:
        compile_args.append("-fopenmp")
        link_args.append("-fopenmp")
    return Extension(
        "MEDS_tabular_automl.sparsegb._sparsegb",
        sources=sorted(str(path) for path in Path("csrc/sparsegb").glob("*.cc")),
        language="c++",
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        optional=True,
    )


if os.environ.get("MEDS_TAB_BUILD_ENGINE") == "0":
    ext_modules = []
else:
    ext_modules = [ext for ext in (engine, sparsegb_extension()) if ext is not None]
setup(ext_modules=ext_modules)
