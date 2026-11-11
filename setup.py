"""Builds the compiled libraries: the native engine (csrc/, MEDS_tabular_automl.native._engine) and
sparsegb (csrc/sparsegb/, MEDS_tabular_automl.sparsegb._sparsegb).

Both are C shared libraries loaded with ctypes and declared as optional extension modules, so
installing MEDS-Tab compiles them when a C++17 compiler (plus the zlib headers for the engine and
OpenMP for sparsegb) is available and installs the package without them otherwise. All other
metadata is in pyproject.toml.

MEDS_TAB_BUILD_ENGINE=0 builds the package without either. The PyPI workflow uses it for the
pure-Python fallback wheel; it publishes compiled wheels (cibuildwheel, see pyproject.toml) and a
source distribution that include both.
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
    """Builds sparsegb (model_launcher=sparsegb), a separate optional library since it needs OpenMP.

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
