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

# Test cases: (feature_size, description)
SOFTMAX_TEST_CASES = [
    (10, "small_10"),
    (100, "medium_100"),
    (128, "medium_128"),
    (1000, "large_1000"),
    (256, "power2_256"),
]


@pytest.mark.parametrize("feature_size,description", SOFTMAX_TEST_CASES)
def test_softmax_triton(feature_size, description):
    """Test Triton Softmax"""
    try:
        from gpu_20days import day11_softmax
    except ImportError:
        pytest.skip("gpu_20days package not available")

    device = ensure_cuda_device()

    print(f"Testing Triton Softmax with shape ({feature_size},) ({description})...")
    input_tensor = torch.randn(feature_size, device=device, dtype=torch.float32)

    output = day11_softmax(input_tensor)
    expected = torch.softmax(input_tensor, dim=0)

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("feature_size,description", SOFTMAX_TEST_CASES)
def test_softmax_cuda(feature_size, description):
    """Test CUDA Softmax"""
    try:
        from gpu_20days.cuda_kernels import day11_softmax
    except ImportError:
        pytest.skip("CUDA kernels not built")

    device = ensure_cuda_device()

    print(f"Testing CUDA Softmax with shape ({feature_size},) ({description})...")
    input_tensor = torch.randn(feature_size, device=device, dtype=torch.float32)

    output = day11_softmax(input_tensor)
    expected = torch.softmax(input_tensor, dim=0)

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
