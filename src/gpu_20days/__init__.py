"""
GPU 20 Days - CUDA and Triton kernels for GPU programming study
"""

from .day01_printAdd import day01_printAdd
from .day02_function import day02_function
from .day03_vectorAdd import day03_vectorAdd
from .day04_matmul import day04_matmul
from .day05_matrixAdd import day05_matrixAdd
from .day06_countElement import day06_countElement
from .day07_matrixCopy import day07_matrixCopy
from .day08_relu import day08_relu
from .day09_silu import day09_silu
from .day10_conv1d import day10_conv1d
from .day11_softmax import day11_softmax

# Try to import CUDA kernels (may not be available if not built)
try:
    from .cuda_kernels import day01_printAdd as cuda_day01_printAdd
    from .cuda_kernels import day02_function as cuda_day02_function
    from .cuda_kernels import day03_vectorAdd as cuda_day03_vectorAdd
    from .cuda_kernels import day04_matmul as cuda_day04_matmul
    from .cuda_kernels import day05_matrixAdd as cuda_day05_matrixAdd
    from .cuda_kernels import day06_countElement as cuda_day06_countElement
    from .cuda_kernels import day07_matrixCopy as cuda_day07_matrixCopy
    from .cuda_kernels import day08_relu as cuda_day08_relu
    from .cuda_kernels import day09_silu as cuda_day09_silu
    from .cuda_kernels import day10_conv1d as cuda_day10_conv1d
    from .cuda_kernels import day11_softmax as cuda_day11_softmax

    __cuda_available__ = True
except ImportError:
    __cuda_available__ = False

__version__ = "0.1.0"

__all__ = [
    # Triton kernels (default)
    "day01_printAdd",
    "day02_function",
    "day03_vectorAdd",
    "day04_matmul",
    "day05_matrixAdd",
    "day06_countElement",
    "day07_matrixCopy",
    "day08_relu",
    "day09_silu",
    "day10_conv1d",
    "day11_softmax",
]
