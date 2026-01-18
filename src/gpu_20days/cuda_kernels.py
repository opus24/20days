"""
CUDA kernels Python wrapper for GPU 20 Days Challenge
"""

from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from types import ModuleType

    _cuda_ops_module: Optional[ModuleType] = None
    __cuda_available__: bool = True
else:
    try:
        import cuda_ops as _cuda_ops_module  # type: ignore[no-redef]

        __cuda_available__ = True
    except ImportError:
        __cuda_available__ = False
        _cuda_ops_module = None  # type: ignore[assignment]

cuda_ops: Optional[Any] = _cuda_ops_module


def _check_cuda() -> None:
    if not __cuda_available__ or cuda_ops is None:
        raise ImportError("CUDA kernels not available. Please build the package first.")


def day01_printAdd(N: int) -> None:
    """Day 01: Print global indices"""
    _check_cuda()
    assert cuda_ops is not None  # Type guard for mypy
    cuda_ops.day01_printAdd(N)


def day02_function(input: torch.Tensor) -> torch.Tensor:
    """Day 02: Device function example (doubles input)"""
    _check_cuda()
    assert cuda_ops is not None  # Type guard for mypy
    return cuda_ops.day02_function(input)


def day03_vectorAdd(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Day 03: Vector addition"""
    _check_cuda()
    assert cuda_ops is not None  # Type guard for mypy
    return cuda_ops.day03_vectorAdd(A, B)


def day04_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Day 04: Matrix multiplication"""
    _check_cuda()
    assert cuda_ops is not None  # Type guard for mypy
    return cuda_ops.day04_matmul(A, B)


def day05_matrixAdd(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Day 05: Matrix addition"""
    _check_cuda()
    assert cuda_ops is not None  # Type guard for mypy
    return cuda_ops.day05_matrixAdd(A, B)


def day06_countElement(input: torch.Tensor, K: int) -> int:
    """Day 06: Count elements equal to K"""
    _check_cuda()
    assert cuda_ops is not None  # Type guard for mypy
    return cuda_ops.day06_countElement(input, K)


def day07_matrixCopy(A: torch.Tensor) -> torch.Tensor:
    """Day 07: Matrix copy"""
    _check_cuda()
    assert cuda_ops is not None  # Type guard for mypy
    return cuda_ops.day07_matrixCopy(A)


def day08_relu(input: torch.Tensor) -> torch.Tensor:
    """Day 08: ReLU activation"""
    _check_cuda()
    assert cuda_ops is not None  # Type guard for mypy
    return cuda_ops.day08_relu(input)


def day09_silu(input: torch.Tensor) -> torch.Tensor:
    """Day 09: SiLU activation"""
    _check_cuda()
    assert cuda_ops is not None  # Type guard for mypy
    return cuda_ops.day09_silu(input)


def day10_conv1d(input: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Day 10: 1D Convolution"""
    _check_cuda()
    assert cuda_ops is not None  # Type guard for mypy
    return cuda_ops.day10_conv1d(input, kernel)


def day11_softmax(input: torch.Tensor) -> torch.Tensor:
    """Day 11: Softmax activation"""
    _check_cuda()
    assert cuda_ops is not None  # Type guard for mypy
    return cuda_ops.day11_softmax(input)


def day12_layernorm(
    input: torch.Tensor, gamma: torch.Tensor = None, beta: torch.Tensor = None, eps: float = 1e-5
) -> torch.Tensor:
    """Day 12: Layer normalization"""
    _check_cuda()
    assert cuda_ops is not None  # Type guard for mypy
    if gamma is None:
        gamma = torch.empty(0, device=input.device, dtype=input.dtype)
    if beta is None:
        beta = torch.empty(0, device=input.device, dtype=input.dtype)
    return cuda_ops.day12_layernorm(input, gamma, beta, eps)


def day13_rmsnorm(
    input: torch.Tensor, weight: torch.Tensor = None, eps: float = 1e-5
) -> torch.Tensor:
    """Day 13: RMS normalization"""
    _check_cuda()
    assert cuda_ops is not None  # Type guard for mypy
    if weight is None:
        weight = torch.empty(0, device=input.device, dtype=input.dtype)
    return cuda_ops.day13_rmsnorm(input, weight, eps)


def day14_fused_softmax(
    input: torch.Tensor, mask: torch.Tensor = None, scale: float = 1.0
) -> torch.Tensor:
    """Day 14: Fused softmax operation"""
    _check_cuda()
    assert cuda_ops is not None  # Type guard for mypy
    if mask is None:
        mask = torch.empty(0, device=input.device, dtype=input.dtype)
    return cuda_ops.day14_fused_softmax(input, mask, scale)


def day15_fused_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None, scale: float = 0.0
) -> torch.Tensor:
    """Day 15: Fused attention mechanism"""
    _check_cuda()
    assert cuda_ops is not None  # Type guard for mypy
    if mask is None:
        mask = torch.empty(0, device=Q.device, dtype=Q.dtype)
    return cuda_ops.day15_fused_attention(Q, K, V, mask, scale)


def day16_group_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Day 16: Grouped general matrix multiplication"""
    _check_cuda()
    assert cuda_ops is not None  # Type guard for mypy
    return cuda_ops.day16_group_gemm(A, B)


def day17_persistent_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Day 17: Persistent matrix multiplication"""
    _check_cuda()
    assert cuda_ops is not None  # Type guard for mypy
    return cuda_ops.day17_persistent_matmul(A, B)


def day18_block_scaled_matmul(A: torch.Tensor, B: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Day 18: Block-scaled matrix multiplication"""
    _check_cuda()
    assert cuda_ops is not None  # Type guard for mypy
    return cuda_ops.day18_block_scaled_matmul(A, B, scale)


def day19_rope(
    query: torch.Tensor, key: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Day 19: Rotary position embedding"""
    _check_cuda()
    assert cuda_ops is not None  # Type guard for mypy
    return cuda_ops.day19_rope(query, key, cos_cache, sin_cache)


def day20_conv2d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    padding: tuple[int, int] = (0, 0),
    stride: tuple[int, int] = (1, 1),
) -> torch.Tensor:
    """Day 20: Two-dimensional convolution"""
    _check_cuda()
    assert cuda_ops is not None  # Type guard for mypy
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    return cuda_ops.day20_conv2d(input, kernel, pad_h, pad_w, stride_h, stride_w)
