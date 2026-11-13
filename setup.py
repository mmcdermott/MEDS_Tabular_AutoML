"""Builds the compiled libraries: the native engine (csrc/, MEDS_tabular_automl.native._engine) and
medsboost (csrc/medsboost/, MEDS_tabular_automl.medsboost._medsboost).

Both are C shared libraries loaded with ctypes and declared as optional extension modules, so
installing MEDS-Tab compiles them when a C++17 compiler (plus OpenMP for medsboost) is available and
installs the package without them otherwise: GCC or clang on Linux and macOS, MSVC on Windows. All
other metadata is in pyproject.toml.

MEDS_TAB_BUILD_ENGINE=0 builds the package without either. The PyPI workflow uses it for the
pure-Python fallback wheel; it publishes compiled wheels (cibuildwheel, see pyproject.toml) and a
source distribution that include both.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup

MSVC = sys.platform == "win32"  # setuptools builds extensions with MSVC on Windows


def c_api(source: Path, prefix: str) -> list[str]:
    """Names of the C functions starting with `prefix` in `source`, which ctypes calls.

    MSVC exports only listed symbols from a DLL (GCC and clang export every symbol), so the extensions pass
    them as export_symbols; setuptools ignores the list elsewhere.
    """
    return sorted(set(re.findall(rf"\b({prefix}\w+)\s*\(", source.read_text())))


engine = Extension(
    "MEDS_tabular_automl.native._engine",
    sources=sorted(str(path) for path in Path("csrc").glob("*.cc")),
    include_dirs=["csrc"],
    language="c++",
    # No fused multiply-adds, so sums and squares round exactly as in NumPy (MSVC's default
    # /fp:precise does not contract).
    extra_compile_args=["/std:c++17", "/O2", "/EHsc"]
    if MSVC
    else ["-std=c++17", "-O2", "-ffp-contract=off", "-pthread"],
    extra_link_args=[] if MSVC else ["-pthread"],
    export_symbols=c_api(Path("csrc/meds_tab.h"), "meds_tab_"),
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


def medsboost_extension() -> Extension | None:
    """Builds medsboost (model_launcher=medsboost), a separate optional library since it needs OpenMP.

    Linux: GCC, or clang with LLVM's libomp. macOS: Apple clang with libomp (`brew install libomp`, or set
    LIBOMP_PREFIX, e.g. to a conda environment with llvm-openmp), or GCC (CC=gcc-14 ...). Windows: MSVC
    with its OpenMP runtime (vcomp140.dll, part of the Visual C++ redistributable).
    """
    sources = sorted(str(path) for path in Path("csrc/medsboost").glob("*.cc"))
    exports = c_api(Path("csrc/medsboost/medsboost.cc"), "medsboost_")
    if MSVC:
        return Extension(
            "MEDS_tabular_automl.medsboost._medsboost",
            sources=sources,
            language="c++",
            extra_compile_args=["/std:c++17", "/O2", "/EHsc", "/openmp"],
            export_symbols=exports,
            optional=True,
        )
    compile_args, link_args = ["-std=c++17", "-O3"], []
    include_dirs, library_dirs = [], []
    apple_clang = sys.platform == "darwin" and "gcc" not in os.environ.get("CC", "")
    if apple_clang:  # Apple clang's driver has no -fopenmp; hand it to the preprocessor and link libomp
        prefix = libomp_prefix()
        if prefix is None:
            print("medsboost: no libomp found (brew install libomp, or set LIBOMP_PREFIX); skipping it.")
            return None
        compile_args += ["-Xpreprocessor", "-fopenmp"]
        link_args += ["-lomp", f"-Wl,-rpath,{prefix / 'lib'}"]
        include_dirs, library_dirs = [str(prefix / "include")], [str(prefix / "lib")]
    else:
        compile_args.append("-fopenmp")
        link_args.append("-fopenmp")
    return Extension(
        "MEDS_tabular_automl.medsboost._medsboost",
        sources=sources,
        language="c++",
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        export_symbols=exports,
        optional=True,
    )


if os.environ.get("MEDS_TAB_BUILD_ENGINE") == "0":
    ext_modules = []
else:
    ext_modules = [ext for ext in (engine, medsboost_extension()) if ext is not None]
setup(ext_modules=ext_modules)
