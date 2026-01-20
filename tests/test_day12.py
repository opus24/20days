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

# Test cases: (feature_size, description) - batch_size is always 1
LAYERNORM_TEST_CASES = [
    (10, "small_10"),
    (100, "medium_100"),
    (128, "medium_128"),
    (1000, "large_1000"),
]


@pytest.mark.parametrize("feature_size,description", LAYERNORM_TEST_CASES)
def test_layernorm_triton(feature_size, description):
    """Test Triton LayerNorm"""
    try:
        from gpu_20days.day12_layernorm import day12_layernorm
    except ImportError:
        pytest.skip("gpu_20days package not available")

    device = ensure_cuda_device()

    print(f"Testing Triton LayerNorm with shape ({feature_size},) ({description})...")
    input_tensor = torch.randn(feature_size, device=device, dtype=torch.float32)
    gamma = torch.ones(feature_size, device=device, dtype=torch.float32)
    beta = torch.zeros(feature_size, device=device, dtype=torch.float32)

    output = day12_layernorm(input_tensor, gamma, beta)
    # batch_size=1이므로 입력을 2D로 변환하여 비교
    input_2d = input_tensor.unsqueeze(0)
    expected_2d = torch.nn.functional.layer_norm(input_2d, (feature_size,), gamma, beta)
    expected = expected_2d.squeeze(0)

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("feature_size,description", LAYERNORM_TEST_CASES)
def test_layernorm_cuda(feature_size, description):
    """Test CUDA LayerNorm"""
    try:
        from gpu_20days.cuda_kernels import day12_layernorm
    except ImportError:
        pytest.skip("CUDA kernels not built")

    device = ensure_cuda_device()

    print(f"Testing CUDA LayerNorm with shape ({feature_size},) ({description})...")
    input_tensor = torch.randn(feature_size, device=device, dtype=torch.float32)
    gamma = torch.ones(feature_size, device=device, dtype=torch.float32)
    beta = torch.zeros(feature_size, device=device, dtype=torch.float32)

    output = day12_layernorm(input_tensor, gamma, beta)
    # batch_size=1이므로 입력을 2D로 변환하여 비교
    input_2d = input_tensor.unsqueeze(0)
    expected_2d = torch.nn.functional.layer_norm(input_2d, (feature_size,), gamma, beta)
    expected = expected_2d.squeeze(0)

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
