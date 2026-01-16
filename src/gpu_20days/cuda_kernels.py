"""
CUDA kernels Python wrapper for GPU 20 Days Challenge
"""
import torch

try:
    import cuda_ops
    __cuda_available__ = True
except ImportError:
    __cuda_available__ = False
    cuda_ops = None


def _check_cuda():
    if not __cuda_available__:
        raise ImportError("CUDA kernels not available. Please build the package first.")


def day01_printAdd(N: int) -> None:
    """Day 01: Print global indices"""
    _check_cuda()
    cuda_ops.day01_printAdd(N)


def day02_function(input: torch.Tensor) -> torch.Tensor:
    """Day 02: Device function example (doubles input)"""
    _check_cuda()
    return cuda_ops.day02_function(input)


def day03_vectorAdd(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Day 03: Vector addition"""
    _check_cuda()
    return cuda_ops.day03_vectorAdd(A, B)


def day04_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Day 04: Matrix multiplication"""
    _check_cuda()
    return cuda_ops.day04_matmul(A, B)


def day05_matrixAdd(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Day 05: Matrix addition"""
    _check_cuda()
    return cuda_ops.day05_matrixAdd(A, B)


def day06_countElement(input: torch.Tensor, K: int) -> int:
    """Day 06: Count elements equal to K"""
    _check_cuda()
    return cuda_ops.day06_countElement(input, K)


def day07_matrixCopy(A: torch.Tensor) -> torch.Tensor:
    """Day 07: Matrix copy"""
    _check_cuda()
    return cuda_ops.day07_matrixCopy(A)


def day08_relu(input: torch.Tensor) -> torch.Tensor:
    """Day 08: ReLU activation"""
    _check_cuda()
    return cuda_ops.day08_relu(input)


def day09_silu(input: torch.Tensor) -> torch.Tensor:
    """Day 09: SiLU activation"""
    _check_cuda()
    return cuda_ops.day09_silu(input)


def day10_conv1d(input: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Day 10: 1D Convolution"""
    _check_cuda()
    return cuda_ops.day10_conv1d(input, kernel)

