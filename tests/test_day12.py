"""
Tests for Day 12: LayerNorm (CUDA + Triton)
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
LAYERNORM_TEST_CASES = [
    (1, 10, "small_1x10"),
    (10, 100, "medium_10x100"),
    (32, 128, "medium_32x128"),
    (100, 1000, "large_100x1000"),
]


@pytest.mark.parametrize("batch_size,feature_size,description", LAYERNORM_TEST_CASES)
def test_layernorm_triton(batch_size, feature_size, description):
    """Test Triton LayerNorm"""
    try:
        from gpu_20days import day12_layernorm
    except ImportError:
        pytest.skip("gpu_20days package not available")

    device = ensure_cuda_device()

    print(f"Testing Triton LayerNorm with shape ({batch_size}, {feature_size}) ({description})...")
    input_tensor = torch.randn(batch_size, feature_size, device=device, dtype=torch.float32)
    gamma = torch.ones(feature_size, device=device, dtype=torch.float32)
    beta = torch.zeros(feature_size, device=device, dtype=torch.float32)

    output = day12_layernorm(input_tensor, gamma, beta)
    expected = torch.nn.functional.layer_norm(input_tensor, (feature_size,), gamma, beta)

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("batch_size,feature_size,description", LAYERNORM_TEST_CASES)
def test_layernorm_cuda(batch_size, feature_size, description):
    """Test CUDA LayerNorm"""
    try:
        from gpu_20days.cuda_kernels import day12_layernorm
    except ImportError:
        pytest.skip("CUDA kernels not built")

    device = ensure_cuda_device()

    print(f"Testing CUDA LayerNorm with shape ({batch_size}, {feature_size}) ({description})...")
    input_tensor = torch.randn(batch_size, feature_size, device=device, dtype=torch.float32)
    gamma = torch.ones(feature_size, device=device, dtype=torch.float32)
    beta = torch.zeros(feature_size, device=device, dtype=torch.float32)

    output = day12_layernorm(input_tensor, gamma, beta)
    expected = torch.nn.functional.layer_norm(input_tensor, (feature_size,), gamma, beta)

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
