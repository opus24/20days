"""
Tests for Day 10: 1D Convolution (CUDA + Triton)
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

# Test cases: ((input_size, kernel_size), description)
CONV1D_TEST_CASES = [
    ((10, 3), "small"),
    ((100, 5), "medium"),
    ((1000, 7), "large"),
    ((10000, 11), "very_large"),
    ((256, 5), "power2"),
]


def conv1d_pytorch(input, kernel):
    """PyTorch reference 1D convolution (valid mode)"""
    kernel_2d = kernel.view(1, 1, -1)
    input_2d = input.view(1, 1, -1)
    result = torch.nn.functional.conv1d(input_2d, kernel_2d, padding=0)
    return result.view(-1)


@pytest.mark.parametrize("params,description", CONV1D_TEST_CASES)
def test_conv1d_triton(params, description):
    """Test Triton 1D convolution"""
    try:
        from gpu_20days import day10_conv1d
    except ImportError:
        pytest.skip("gpu_20days package not available")

    device = ensure_cuda_device()
    input_size, kernel_size = params

    print(
        f"Testing Triton conv1d with input_size={input_size}, kernel_size={kernel_size} ({description})..."
    )
    input_arr = torch.randn(input_size, device=device, dtype=torch.float32)
    kernel = torch.randn(kernel_size, device=device, dtype=torch.float32)

    output = day10_conv1d(input_arr, kernel)
    expected = conv1d_pytorch(input_arr, kernel)

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("params,description", CONV1D_TEST_CASES)
def test_conv1d_cuda(params, description):
    """Test CUDA 1D convolution"""
    try:
        from gpu_20days.cuda_kernels import day10_conv1d
    except ImportError:
        pytest.skip("CUDA kernels not built")

    device = ensure_cuda_device()
    input_size, kernel_size = params

    print(
        f"Testing CUDA conv1d with input_size={input_size}, kernel_size={kernel_size} ({description})..."
    )
    input_arr = torch.randn(input_size, device=device, dtype=torch.float32)
    kernel = torch.randn(kernel_size, device=device, dtype=torch.float32)

    output = day10_conv1d(input_arr, kernel)
    expected = conv1d_pytorch(input_arr, kernel)

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)
