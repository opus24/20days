"""
Tests for Day 18: Block Scaled Matrix Multiplication (CUDA + Triton)
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

# Test cases: (M, N, K, scale, description)
BLOCK_SCALED_MATMUL_TEST_CASES = [
    (8, 8, 16, 0.5, "small_8x8x16_scale0.5"),
    (32, 32, 64, 2.0, "medium_32x32x64_scale2.0"),
    (64, 64, 128, 1.0, "medium_64x64x128_scale1.0"),
]


@pytest.mark.parametrize("M,N,K,scale,description", BLOCK_SCALED_MATMUL_TEST_CASES)
def test_block_scaled_matmul_triton(M, N, K, scale, description):
    """Test Triton Block Scaled Matmul"""
    try:
        from gpu_20days import day18_block_scaled_matmul
    except ImportError:
        pytest.skip("gpu_20days package not available")

    device = ensure_cuda_device()

    print(
        f"Testing Triton Block Scaled Matmul with shape ({M}, {N}) x ({N}, {K}), scale={scale} ({description})..."
    )
    A = torch.randn(M, N, device=device, dtype=torch.float32)
    B = torch.randn(N, K, device=device, dtype=torch.float32)

    output = day18_block_scaled_matmul(A, B, scale)
    expected = torch.matmul(A, B) * scale

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("M,N,K,scale,description", BLOCK_SCALED_MATMUL_TEST_CASES)
def test_block_scaled_matmul_cuda(M, N, K, scale, description):
    """Test CUDA Block Scaled Matmul"""
    try:
        from gpu_20days.cuda_kernels import day18_block_scaled_matmul
    except ImportError:
        pytest.skip("CUDA kernels not built")

    device = ensure_cuda_device()

    print(
        f"Testing CUDA Block Scaled Matmul with shape ({M}, {N}) x ({N}, {K}), scale={scale} ({description})..."
    )
    A = torch.randn(M, N, device=device, dtype=torch.float32)
    B = torch.randn(N, K, device=device, dtype=torch.float32)

    output = day18_block_scaled_matmul(A, B, scale)
    expected = torch.matmul(A, B) * scale

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
