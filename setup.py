"""
Setup script for GPU 100 Days Challenge
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# CUDA source files (relative paths from setup.py)
cuda_sources = [
    "csrc/bindings.cu",
    "csrc/day01_printAdd.cu",
    "csrc/day02_function.cu",
    "csrc/day03_vectorAdd.cu",
    "csrc/day04_matmul.cu",
    "csrc/day05_matrixAdd.cu",
    "csrc/day06_countElement.cu",
    "csrc/day07_matrixCopy.cu",
    "csrc/day08_relu.cu",
    "csrc/day09_silu.cu",
    "csrc/day10_conv1d.cu",
    "csrc/day11_softmax.cu",
    "csrc/day12_layernorm.cu",
    "csrc/day13_rmsnorm.cu",
    "csrc/day14_fused_softmax.cu",
    "csrc/day15_fused_attention.cu",
    "csrc/day16_group_gemm.cu",
    "csrc/day17_persistent_matmul.cu",
    "csrc/day18_block_scaled_matmul.cu",
    "csrc/day19_rope.cu",
    "csrc/day20_conv2d.cu",
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
