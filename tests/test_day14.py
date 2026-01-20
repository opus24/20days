"""
Tests for Day 14: Fused Softmax (CUDA + Triton)
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

# Test cases: (seq_len, feature_size, description) - batch_size is always 1
FUSED_SOFTMAX_TEST_CASES = [
    (10, 32, "small_10x32"),
    (32, 64, "medium_32x64"),
    (128, 128, "medium_128x128"),
]


@pytest.mark.parametrize("seq_len,feature_size,description", FUSED_SOFTMAX_TEST_CASES)
def test_fused_softmax_triton(seq_len, feature_size, description):
    """Test Triton Fused Softmax"""
    try:
        from gpu_20days.day14_fused_softmax import day14_fused_softmax
    except ImportError:
        pytest.skip("gpu_20days package not available")

    device = ensure_cuda_device()

    print(f"Testing Triton Fused Softmax with shape ({seq_len}, {feature_size}) ({description})...")
    # batch_size is always 1, so input is 2D
    input_tensor = torch.randn(seq_len, feature_size, device=device, dtype=torch.float32)

    output = day14_fused_softmax(input_tensor)
    expected = torch.softmax(input_tensor, dim=-1)

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("seq_len,feature_size,description", FUSED_SOFTMAX_TEST_CASES)
def test_fused_softmax_cuda(seq_len, feature_size, description):
    """Test CUDA Fused Softmax"""
    try:
        from gpu_20days.cuda_kernels import day14_fused_softmax
    except ImportError:
        pytest.skip("CUDA kernels not built")

    device = ensure_cuda_device()

    print(f"Testing CUDA Fused Softmax with shape ({seq_len}, {feature_size}) ({description})...")
    # batch_size is always 1, so input is 2D
    input_tensor = torch.randn(seq_len, feature_size, device=device, dtype=torch.float32)

    output = day14_fused_softmax(input_tensor)
    expected = torch.softmax(input_tensor, dim=-1)

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
