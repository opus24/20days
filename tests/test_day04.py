"""
Tests for Day 04: Matrix Multiplication (CUDA + Triton)
"""

import sys
from pathlib import Path

import pytest
import torch

# Add tests directory to path to import conftest
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

from conftest import benchmark_kernel_vs_pytorch, compare_kernel_with_pytorch, ensure_cuda_device

# Test cases: ((M, N, K), description)
MATMUL_TEST_CASES = [
    ((4, 4, 4), "tiny_4x4"),
    ((10, 20, 30), "small"),
    ((64, 128, 256), "medium"),
    ((32, 32, 32), "square_small"),
    ((128, 128, 128), "square_medium"),
]


@pytest.mark.parametrize("shape,description", MATMUL_TEST_CASES)
def test_matmul_triton(shape, description):
    """Test Triton matrix multiplication"""
    try:
        from gpu_20days.day04_matmul import day04_matmul
    except ImportError:
        pytest.skip("gpu_20days package not available")

    device = ensure_cuda_device()
    M, N, K = shape

    print(f"Testing Triton matmul with shape {shape} ({description})...")
    A = torch.randn(M, N, device=device, dtype=torch.float32)
    B = torch.randn(N, K, device=device, dtype=torch.float32)

    output = day04_matmul(A, B)
    expected = A @ B

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape,description", MATMUL_TEST_CASES)
def test_matmul_cuda(shape, description):
    """Test CUDA matrix multiplication"""
    try:
        from gpu_20days.cuda_kernels import day04_matmul
    except ImportError:
        pytest.skip("CUDA kernels not built")

    device = ensure_cuda_device()
    M, N, K = shape

    print(f"Testing CUDA matmul with shape {shape} ({description})...")
    A = torch.randn(M, N, device=device, dtype=torch.float32)
    B = torch.randn(N, K, device=device, dtype=torch.float32)

    output = day04_matmul(A, B)
    expected = A @ B

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)
