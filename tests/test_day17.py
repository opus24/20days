"""
Tests for Day 17: Persistent Matmul (CUDA + Triton)
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

# Test cases: (M, N, K, description)
PERSISTENT_MATMUL_TEST_CASES = [
    (8, 8, 16, "small_8x8x16"),
    (32, 32, 64, "medium_32x32x64"),
    (64, 64, 128, "medium_64x64x128"),
]


@pytest.mark.parametrize("M,N,K,description", PERSISTENT_MATMUL_TEST_CASES)
def test_persistent_matmul_triton(M, N, K, description):
    """Test Triton Persistent Matmul"""
    try:
        from gpu_20days import day17_persistent_matmul
    except ImportError:
        pytest.skip("gpu_20days package not available")

    device = ensure_cuda_device()

    print(f"Testing Triton Persistent Matmul with shape ({M}, {N}) x ({N}, {K}) ({description})...")
    A = torch.randn(M, N, device=device, dtype=torch.float32)
    B = torch.randn(N, K, device=device, dtype=torch.float32)

    output = day17_persistent_matmul(A, B)
    expected = torch.matmul(A, B)

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("M,N,K,description", PERSISTENT_MATMUL_TEST_CASES)
def test_persistent_matmul_cuda(M, N, K, description):
    """Test CUDA Persistent Matmul"""
    try:
        from gpu_20days.cuda_kernels import day17_persistent_matmul
    except ImportError:
        pytest.skip("CUDA kernels not built")

    device = ensure_cuda_device()

    print(f"Testing CUDA Persistent Matmul with shape ({M}, {N}) x ({N}, {K}) ({description})...")
    A = torch.randn(M, N, device=device, dtype=torch.float32)
    B = torch.randn(N, K, device=device, dtype=torch.float32)

    output = day17_persistent_matmul(A, B)
    expected = torch.matmul(A, B)

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
