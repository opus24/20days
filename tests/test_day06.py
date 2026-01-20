"""
Tests for Day 06: Count Array Element (CUDA + Triton)
"""

import sys
from pathlib import Path

import pytest
import torch

# Add tests directory to path to import conftest
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

from conftest import ensure_cuda_device

# Test cases: ((N, K), description)
COUNT_ELEMENT_TEST_CASES = [
    ((10, 5), "small"),
    ((100, 42), "medium"),
    ((1000, 100), "large"),
    ((10000, 999), "very_large"),
    ((256, 128), "power2"),
]


@pytest.mark.parametrize("params,description", COUNT_ELEMENT_TEST_CASES)
def test_countElement_triton(params, description):
    """Test Triton count element"""
    try:
        from gpu_20days.day06_count import day06_countElement
    except ImportError:
        pytest.skip("gpu_20days package not available")

    device = ensure_cuda_device()
    N, K = params

    print(f"Testing Triton countElement with N={N}, K={K} ({description})...")
    input_arr = torch.randint(0, 200, (N,), device=device, dtype=torch.int32)
    # Set some elements to K
    num_k = N // 10
    input_arr[:num_k] = K

    count = day06_countElement(input_arr, K)
    expected = (input_arr == K).sum().item()

    assert count == expected, f"Expected {expected}, got {count}"


@pytest.mark.parametrize("params,description", COUNT_ELEMENT_TEST_CASES)
def test_countElement_cuda(params, description):
    """Test CUDA count element"""
    try:
        from gpu_20days.cuda_kernels import day06_countElement
    except ImportError:
        pytest.skip("CUDA kernels not built")

    device = ensure_cuda_device()
    N, K = params

    print(f"Testing CUDA countElement with N={N}, K={K} ({description})...")
    input_arr = torch.randint(0, 200, (N,), device=device, dtype=torch.int32)
    # Set some elements to K
    num_k = N // 10
    input_arr[:num_k] = K

    count = day06_countElement(input_arr, K)
    expected = (input_arr == K).sum().item()

    assert count == expected, f"Expected {expected}, got {count}"
