"""
Tests for Day 11: Softmax (CUDA + Triton)
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

# Test cases: (batch_size, feature_size, description)
SOFTMAX_TEST_CASES = [
    (1, 10, "small_1x10"),
    (10, 100, "medium_10x100"),
    (32, 128, "medium_32x128"),
    (100, 1000, "large_100x1000"),
    (256, 256, "power2_256x256"),
]


@pytest.mark.parametrize("batch_size,feature_size,description", SOFTMAX_TEST_CASES)
def test_softmax_triton(batch_size, feature_size, description):
    """Test Triton Softmax"""
    try:
        from gpu_20days import day11_softmax
    except ImportError:
        pytest.skip("gpu_20days package not available")

    device = ensure_cuda_device()

    print(f"Testing Triton Softmax with shape ({batch_size}, {feature_size}) ({description})...")
    input_tensor = torch.randn(batch_size, feature_size, device=device, dtype=torch.float32)

    output = day11_softmax(input_tensor)
    expected = torch.softmax(input_tensor, dim=1)

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("batch_size,feature_size,description", SOFTMAX_TEST_CASES)
def test_softmax_cuda(batch_size, feature_size, description):
    """Test CUDA Softmax"""
    try:
        from gpu_20days.cuda_kernels import day11_softmax
    except ImportError:
        pytest.skip("CUDA kernels not built")

    device = ensure_cuda_device()

    print(f"Testing CUDA Softmax with shape ({batch_size}, {feature_size}) ({description})...")
    input_tensor = torch.randn(batch_size, feature_size, device=device, dtype=torch.float32)

    output = day11_softmax(input_tensor)
    expected = torch.softmax(input_tensor, dim=1)

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
