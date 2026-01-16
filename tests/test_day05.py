"""
Tests for Day 05: Matrix Addition (CUDA + Triton)
"""
import pytest
import torch
from conftest import (
    ensure_cuda_device,
    compare_kernel_with_pytorch,
    benchmark_kernel_vs_pytorch,
)


# Test cases: (N, description) where N is the matrix size (N x N)
MATRIX_ADD_TEST_CASES = [
    (1, "single_element"),
    (10, "small_10"),
    (64, "medium_64"),
    (128, "medium_128"),
    (256, "large_256"),
    (32, "power2_32"),
]


@pytest.mark.parametrize("n,description", MATRIX_ADD_TEST_CASES)
def test_matrixAdd_triton(n, description):
    """Test Triton matrix addition"""
    try:
        from gpu_20days import day05_matrixAdd
    except ImportError:
        pytest.skip("gpu_20days package not available")
    
    device = ensure_cuda_device()
    
    print(f"Testing Triton matrixAdd with size {n}x{n} ({description})...")
    A = torch.randn(n, n, device=device, dtype=torch.float32)
    B = torch.randn(n, n, device=device, dtype=torch.float32)
    
    output = day05_matrixAdd(A, B)
    expected = A + B
    
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("n,description", MATRIX_ADD_TEST_CASES)
def test_matrixAdd_cuda(n, description):
    """Test CUDA matrix addition"""
    try:
        from gpu_20days.cuda_kernels import day05_matrixAdd
    except ImportError:
        pytest.skip("CUDA kernels not built")
    
    device = ensure_cuda_device()
    
    print(f"Testing CUDA matrixAdd with size {n}x{n} ({description})...")
    A = torch.randn(n, n, device=device, dtype=torch.float32)
    B = torch.randn(n, n, device=device, dtype=torch.float32)
    
    output = day05_matrixAdd(A, B)
    expected = A + B
    
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-8)
