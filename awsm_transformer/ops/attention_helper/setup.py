"""
setup.py for awsm_attention

Builds the C/CUDA shared libraries (libflash2.so, libflash3.so,
libattentionwrapper.so) and installs them into the Python package so that
`pip install .` or `pip install -e .` just works.

Directory layout expected:
    .
    ├── awsm_attention/          # Python package
    │   ├── __init__.py
    │   ├── attention.py
    │   └── lib/                 # created at build time; .so files copied here
    ├── flash2/                  # flash2 sub-project with its own Makefile
    ├── flash3/                  # flash3 sub-project with its own Makefile
    ├── include/                 # attention_helper.h etc.
    ├── src/
    │   └── attention_helper.c
    ├── lib/                     # build output of the top-level Makefile
    ├── objs/
    ├── Makefile                 # existing top-level Makefile
    ├── setup.py                 # this file
    └── pyproject.toml
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop


ROOT = Path(__file__).resolve().parent
PKG_LIB_DIR = ROOT / "awsm_attention" / "lib"


def _build_native_libs():
    """Invoke the top-level Makefile that builds all shared libraries."""
    # Ensure output dirs exist
    (ROOT / "objs").mkdir(exist_ok=True)
    (ROOT / "lib").mkdir(exist_ok=True)
    PKG_LIB_DIR.mkdir(exist_ok=True)

    env = os.environ.copy()
    # Allow user to override build flags via env, default to optimized build
    env.setdefault("CFLAGS", "-O3 -fPIC")
    env.setdefault("NVCC_FLAGS", "-O4 --use_fast_math")

    # Parallelism: default to 8, override with AWSM_BUILD_JOBS env var
    n_jobs = env.get("ATTENTION_BUILD_JOBS", "8")

    print("=" * 60, flush=True)
    print(f"Building native flash attention libraries (-j{n_jobs}) ...", flush=True)
    print("=" * 60, flush=True)

    subprocess.check_call(
        ["make", f"-j{n_jobs}", "lib/libattentionwrapper.so"],
        cwd=str(ROOT),
        env=env,
        # Stream build output directly to terminal
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    # Copy all required .so files into awsm_attention/lib/ so they ship
    # with the wheel / editable install.
    so_files = [
        ROOT / "lib" / "libattentionwrapper.so",
        ROOT / "flash3" / "lib" / "libflash3.so",
        ROOT / "flash2" / "lib" / "libflash2.so",
    ]
    for src in so_files:
        if src.is_file():
            dst = PKG_LIB_DIR / src.name
            shutil.copy2(str(src), str(dst))
            print(f"  copied {src} -> {dst}")
        else:
            print(f"  WARNING: expected {src} but not found", file=sys.stderr)


class BuildWithNative(build_py):
    """Custom build_py that compiles native libs first."""

    def run(self):
        _build_native_libs()
        super().run()


class DevelopWithNative(develop):
    """Custom develop (pip install -e .) that compiles native libs first."""

    def run(self):
        _build_native_libs()
        super().run()


setup(
    name="awsm_attention",
    version="0.0.1",
    description="Python wrapper for flash attention C/CUDA library",
    packages=find_packages(),
    package_data={
        "awsm_attention.lib": ["*.so"],
    },
    install_requires=[
        "torch",
    ],
    python_requires=">=3.8",
    cmdclass={
        "build_py": BuildWithNative,
        "develop": DevelopWithNative,
    },
)