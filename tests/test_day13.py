"""
Tests for Day 13: RMS Normalization (CUDA + Triton)
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
RMSNORM_TEST_CASES = [
    (10, "small_10"),
    (100, "medium_100"),
    (128, "medium_128"),
    (1000, "large_1000"),
]


@pytest.mark.parametrize("feature_size,description", RMSNORM_TEST_CASES)
def test_rmsnorm_triton(feature_size, description):
    """Test Triton RMS Normalization"""
    try:
        from gpu_20days.day13_rmsnorm import day13_rmsnorm
    except ImportError:
        pytest.skip("gpu_20days package not available")

    device = ensure_cuda_device()

    print(f"Testing Triton RMSNorm with shape ({feature_size},) ({description})...")
    input_tensor = torch.randn(feature_size, device=device, dtype=torch.float32)
    weight = torch.ones(feature_size, device=device, dtype=torch.float32)

    output = day13_rmsnorm(input_tensor, weight)
    # RMSNorm reference: (x / sqrt(mean(x^2) + eps)) * weight
    # batch_size=1이므로 입력을 2D로 변환하여 비교
    input_2d = input_tensor.unsqueeze(0)
    rms = torch.sqrt(torch.mean(input_2d**2, dim=-1, keepdim=True) + 1e-5)
    expected_2d = (input_2d / rms) * weight
    expected = expected_2d.squeeze(0)

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("feature_size,description", RMSNORM_TEST_CASES)
def test_rmsnorm_cuda(feature_size, description):
    """Test CUDA RMS Normalization"""
    try:
        from gpu_20days.cuda_kernels import day13_rmsnorm
    except ImportError:
        pytest.skip("CUDA kernels not built")

    device = ensure_cuda_device()

    print(f"Testing CUDA RMSNorm with shape ({feature_size},) ({description})...")
    input_tensor = torch.randn(feature_size, device=device, dtype=torch.float32)
    weight = torch.ones(feature_size, device=device, dtype=torch.float32)

    output = day13_rmsnorm(input_tensor, weight)
    # batch_size=1이므로 입력을 2D로 변환하여 비교
    input_2d = input_tensor.unsqueeze(0)
    rms = torch.sqrt(torch.mean(input_2d**2, dim=-1, keepdim=True) + 1e-5)
    expected_2d = (input_2d / rms) * weight
    expected = expected_2d.squeeze(0)

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
