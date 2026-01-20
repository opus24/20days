"""
Tests for Day 03: Vector Addition (CUDA + Triton)
"""

import sys
from pathlib import Path

import pytest
import torch

# Add tests directory to path to import conftest
# pytest automatically loads conftest.py, but we need explicit import for functions
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

from conftest import benchmark_kernel_vs_pytorch, compare_kernel_with_pytorch, ensure_cuda_device

# Test cases: (size, description)
VECTOR_ADD_TEST_CASES = [
    (1, "single_element"),
    (10, "small_10"),
    (100, "small_100"),
    (1000, "medium_1000"),
    (10000, "medium_10000"),
    (100000, "large_100000"),
    (1000000, "large_1000000"),
    (999, "odd_999"),
    (10001, "odd_10001"),
    (256, "power2_256"),
    (1024, "power2_1024"),
    (4096, "power2_4096"),
]


@pytest.mark.parametrize("n,description", VECTOR_ADD_TEST_CASES)
def test_vectorAdd_triton(n, description):
    """Test Triton vector addition"""
    try:
        from gpu_20days.day03_vector import day03_vectorAdd
    except ImportError:
        pytest.skip("gpu_20days package not available")

    device = ensure_cuda_device()

    print(f"Testing Triton vectorAdd with size {n} ({description})...")
    a = torch.randn(n, device=device, dtype=torch.float32)
    b = torch.randn(n, device=device, dtype=torch.float32)

    output = day03_vectorAdd(a, b)
    expected = a + b

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("n,description", VECTOR_ADD_TEST_CASES)
def test_vectorAdd_cuda(n, description):
    """Test CUDA vector addition"""
    try:
        from gpu_20days.cuda_kernels import day03_vectorAdd
    except ImportError:
        pytest.skip("CUDA kernels not built")

    device = ensure_cuda_device()

    print(f"Testing CUDA vectorAdd with size {n} ({description})...")
    a = torch.randn(n, device=device, dtype=torch.float32)
    b = torch.randn(n, device=device, dtype=torch.float32)

    output = day03_vectorAdd(a, b)
    expected = a + b

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-8)


def test_vectorAdd_benchmark():
    """Benchmark vector addition"""
    try:
        from gpu_20days.day03_vector import day03_vectorAdd
    except ImportError:
        pytest.skip("gpu_20days package not available")

    device = ensure_cuda_device()
    n = 1000000

    a = torch.randn(n, device=device, dtype=torch.float32)
    b = torch.randn(n, device=device, dtype=torch.float32)

    benchmark_kernel_vs_pytorch(day03_vectorAdd, lambda x, y: x + y, a, b)
