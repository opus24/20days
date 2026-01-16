"""
Tests for Day 02: Function (CUDA + Triton)
"""
import pytest
import torch
from conftest import (
    ensure_cuda_device,
    compare_kernel_with_pytorch,
    benchmark_kernel_vs_pytorch,
)


# Test cases: (size, description)
FUNCTION_TEST_CASES = [
    (1, "single_element"),
    (100, "small_100"),
    (1000, "medium_1000"),
    (10000, "medium_10000"),
    (256, "power2_256"),
    (1024, "power2_1024"),
]


@pytest.mark.parametrize("n,description", FUNCTION_TEST_CASES)
def test_function_triton(n, description):
    """Test Triton function (doubles input)"""
    try:
        from gpu_20days import day02_function
    except ImportError:
        pytest.skip("gpu_20days package not available")
    
    device = ensure_cuda_device()
    
    print(f"Testing Triton function with size {n} ({description})...")
    input_tensor = torch.randint(1, 100, (n,), device=device, dtype=torch.int32)
    
    output = day02_function(input_tensor)
    expected = input_tensor * 2
    
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("n,description", FUNCTION_TEST_CASES)
def test_function_cuda(n, description):
    """Test CUDA function (doubles input)"""
    try:
        from gpu_20days.cuda_kernels import day02_function
    except ImportError:
        pytest.skip("CUDA kernels not built")
    
    device = ensure_cuda_device()
    
    print(f"Testing CUDA function with size {n} ({description})...")
    input_tensor = torch.randint(1, 100, (n,), device=device, dtype=torch.int32)
    
    output = day02_function(input_tensor)
    expected = input_tensor * 2
    
    torch.testing.assert_close(output, expected)
