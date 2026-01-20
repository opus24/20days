"""
Tests for Day 16: Group GEMM (CUDA + Triton)
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

# Test cases: (num_groups, M, N, K, description)
GROUP_GEMM_TEST_CASES = [
    (2, 8, 8, 16, "small_2x8x8x16"),
    (4, 16, 16, 32, "medium_4x16x16x32"),
    (8, 32, 32, 64, "medium_8x32x32x64"),
]


@pytest.mark.parametrize("num_groups,M,N,K,description", GROUP_GEMM_TEST_CASES)
def test_group_gemm_triton(num_groups, M, N, K, description):
    """Test Triton Group GEMM"""
    try:
        from gpu_20days.day16_group_gemm import day16_group_gemm
    except ImportError:
        pytest.skip("gpu_20days package not available")

    device = ensure_cuda_device()

    print(
        f"Testing Triton Group GEMM with shape ({num_groups}, {M}, {K}) x ({num_groups}, {K}, {N}) ({description})..."
    )
    A = torch.randn(num_groups, M, K, device=device, dtype=torch.float32)
    B = torch.randn(num_groups, K, N, device=device, dtype=torch.float32)

    output = day16_group_gemm(A, B)
    expected = torch.stack([torch.matmul(A[i], B[i]) for i in range(num_groups)])

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("num_groups,M,N,K,description", GROUP_GEMM_TEST_CASES)
def test_group_gemm_cuda(num_groups, M, N, K, description):
    """Test CUDA Group GEMM"""
    try:
        from gpu_20days.cuda_kernels import day16_group_gemm
    except ImportError:
        pytest.skip("CUDA kernels not built")

    device = ensure_cuda_device()

    print(
        f"Testing CUDA Group GEMM with shape ({num_groups}, {M}, {K}) x ({num_groups}, {K}, {N}) ({description})..."
    )
    A = torch.randn(num_groups, M, K, device=device, dtype=torch.float32)
    B = torch.randn(num_groups, K, N, device=device, dtype=torch.float32)

    output = day16_group_gemm(A, B)
    expected = torch.stack([torch.matmul(A[i], B[i]) for i in range(num_groups)])

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
