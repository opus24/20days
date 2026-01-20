"""
Tests for Day 20: 2D Convolution (CUDA + Triton)
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

# Test cases: (in_channels, out_channels, height, width, kernel_size, description) - batch_size is always 1
CONV2D_TEST_CASES = [
    (1, 1, 8, 8, 3, "small_1x1_8x8_k3"),
    (3, 4, 16, 16, 3, "medium_3x4_16x16_k3"),
    (8, 16, 32, 32, 5, "medium_8x16_32x32_k5"),
]


@pytest.mark.parametrize(
    "in_channels,out_channels,height,width,kernel_size,description", CONV2D_TEST_CASES
)
def test_conv2d_triton(in_channels, out_channels, height, width, kernel_size, description):
    """Test Triton 2D Convolution"""
    try:
        from gpu_20days.day20_conv2d import day20_conv2d
    except ImportError:
        pytest.skip("gpu_20days package not available")

    device = ensure_cuda_device()

    print(
        f"Testing Triton 2D Conv with shape ({in_channels}, {height}, {width}), kernel={kernel_size} ({description})..."
    )
    # batch_size is always 1, so input is 3D
    input_tensor = torch.randn(in_channels, height, width, device=device, dtype=torch.float32)
    kernel = torch.randn(
        out_channels, in_channels, kernel_size, kernel_size, device=device, dtype=torch.float32
    )

    output = day20_conv2d(input_tensor, kernel, padding=(1, 1), stride=(1, 1))
    # Add batch dimension for PyTorch reference
    input_with_batch = input_tensor.unsqueeze(0)
    expected = torch.nn.functional.conv2d(input_with_batch, kernel, padding=(1, 1), stride=(1, 1))
    expected = expected.squeeze(0)  # Remove batch dimension

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "in_channels,out_channels,height,width,kernel_size,description", CONV2D_TEST_CASES
)
def test_conv2d_cuda(in_channels, out_channels, height, width, kernel_size, description):
    """Test CUDA 2D Convolution"""
    try:
        from gpu_20days.cuda_kernels import day20_conv2d
    except ImportError:
        pytest.skip("CUDA kernels not built")

    device = ensure_cuda_device()

    print(
        f"Testing CUDA 2D Conv with shape ({in_channels}, {height}, {width}), kernel={kernel_size} ({description})..."
    )
    # batch_size is always 1, so input is 3D
    input_tensor = torch.randn(in_channels, height, width, device=device, dtype=torch.float32)
    kernel = torch.randn(
        out_channels, in_channels, kernel_size, kernel_size, device=device, dtype=torch.float32
    )

    output = day20_conv2d(input_tensor, kernel, padding=(1, 1), stride=(1, 1))
    # Add batch dimension for PyTorch reference
    input_with_batch = input_tensor.unsqueeze(0)
    expected = torch.nn.functional.conv2d(input_with_batch, kernel, padding=(1, 1), stride=(1, 1))
    expected = expected.squeeze(0)  # Remove batch dimension

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
