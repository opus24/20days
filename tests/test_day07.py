"""
Tests for Day 07: Matrix Copy (CUDA + Triton)
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

# Test cases: (N, description) where N is the matrix size (N x N)
MATRIX_COPY_TEST_CASES = [
    (1, "single_element"),
    (10, "small_10"),
    (64, "medium_64"),
    (128, "medium_128"),
    (256, "large_256"),
    (32, "power2_32"),
]


@pytest.mark.parametrize("n,description", MATRIX_COPY_TEST_CASES)
def test_matrixCopy_triton(n, description):
    """Test Triton matrix copy"""
    try:
        from gpu_20days import day07_matrixCopy
    except ImportError:
        pytest.skip("gpu_20days package not available")

    device = ensure_cuda_device()

    print(f"Testing Triton matrixCopy with size {n}x{n} ({description})...")
    A = torch.randn(n, n, device=device, dtype=torch.float32)

    output = day07_matrixCopy(A)

    torch.testing.assert_close(output, A, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("n,description", MATRIX_COPY_TEST_CASES)
def test_matrixCopy_cuda(n, description):
    """Test CUDA matrix copy"""
    try:
        from gpu_20days.cuda_kernels import day07_matrixCopy
    except ImportError:
        pytest.skip("CUDA kernels not built")

    device = ensure_cuda_device()

    print(f"Testing CUDA matrixCopy with size {n}x{n} ({description})...")
    A = torch.randn(n, n, device=device, dtype=torch.float32)

    output = day07_matrixCopy(A)

    torch.testing.assert_close(output, A, rtol=1e-5, atol=1e-8)
