"""
Setup script for GPU 100 Days Challenge
"""
import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the directory containing this setup.py
ROOT_DIR = Path(__file__).parent.absolute()
CSRC_DIR = ROOT_DIR / "csrc"

# CUDA source files
cuda_sources = [
    str(CSRC_DIR / "bindings.cu"),
    str(CSRC_DIR / "day01_printAdd.cu"),
    str(CSRC_DIR / "day02_function.cu"),
    str(CSRC_DIR / "day03_vectorAdd.cu"),
    str(CSRC_DIR / "day04_matmul.cu"),
    str(CSRC_DIR / "day05_matrixAdd.cu"),
    str(CSRC_DIR / "day06_countElement.cu"),
    str(CSRC_DIR / "day07_matrixCopy.cu"),
    str(CSRC_DIR / "day08_relu.cu"),
    str(CSRC_DIR / "day09_silu.cu"),
    str(CSRC_DIR / "day10_conv1d.cu"),
]

# CUDA extension module
ext_modules = [
    CUDAExtension(
        name="cuda_ops",
        sources=cuda_sources,
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": [
                "-O3",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "-lineinfo",
            ],
        },
    )
]

setup(
    name="gpu_20days",  # Package name for pip
    version="0.1.0",
    author="GPU Study",
    description="CUDA and Triton kernels for GPU 20 Days Challenge",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "triton>=2.0.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
