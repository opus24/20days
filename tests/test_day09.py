"""
Tests for Day 09: SiLU (CUDA + Triton)
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

# Test cases: (size, description)
SILU_TEST_CASES = [
    (1, "single_element"),
    (100, "small_100"),
    (1000, "medium_1000"),
    (10000, "medium_10000"),
    (100000, "large_100000"),
    (256, "power2_256"),
    (1024, "power2_1024"),
]


def silu_pytorch(x):
    """SiLU reference implementation"""
    return x * torch.sigmoid(x)


@pytest.mark.parametrize("n,description", SILU_TEST_CASES)
def test_silu_triton(n, description):
    """Test Triton SiLU"""
    try:
        from gpu_20days.day09_silu import day09_silu
    except ImportError:
        pytest.skip("gpu_20days package not available")

    device = ensure_cuda_device()

    print(f"Testing Triton SiLU with size {n} ({description})...")
    input_arr = torch.randn(n, device=device, dtype=torch.float32) * 2.0 - 1.0

    output = day09_silu(input_arr)
    expected = silu_pytorch(input_arr)

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("n,description", SILU_TEST_CASES)
def test_silu_cuda(n, description):
    """Test CUDA SiLU"""
    try:
        from gpu_20days.cuda_kernels import day09_silu
    except ImportError:
        pytest.skip("CUDA kernels not built")

    device = ensure_cuda_device()

    print(f"Testing CUDA SiLU with size {n} ({description})...")
    input_arr = torch.randn(n, device=device, dtype=torch.float32) * 2.0 - 1.0

    output = day09_silu(input_arr)
    expected = silu_pytorch(input_arr)

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)
