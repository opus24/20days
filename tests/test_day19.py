"""
Tests for Day 19: RoPE (Rotary Position Embedding) (CUDA + Triton)
"""

import math
import sys
from pathlib import Path

import pytest
import torch

# Add tests directory to path to import conftest
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

from conftest import benchmark_kernel_vs_pytorch, compare_kernel_with_pytorch, ensure_cuda_device

# Test cases: (num_heads, seq_len, head_dim, description) - batch_size is always 1
ROPE_TEST_CASES = [
    (2, 8, 32, "small_2x8x32"),
    (4, 16, 64, "medium_4x16x64"),
]


def create_rope_cache(seq_len, head_dim, device):
    """Create cos and sin cache for RoPE"""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    cos_cache = torch.cos(freqs)
    sin_cache = torch.sin(freqs)
    return cos_cache, sin_cache


@pytest.mark.parametrize("num_heads,seq_len,head_dim,description", ROPE_TEST_CASES)
def test_rope_triton(num_heads, seq_len, head_dim, description):
    """Test Triton RoPE"""
    try:
        from gpu_20days.day19_rope import day19_rope
    except ImportError:
        pytest.skip("gpu_20days package not available")

    device = ensure_cuda_device()

    print(f"Testing Triton RoPE with shape ({num_heads}, {seq_len}, {head_dim}) ({description})...")
    # batch_size is always 1, so input is 3D
    query = torch.randn(num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    key = torch.randn(num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    cos_cache, sin_cache = create_rope_cache(seq_len, head_dim // 2, device)

    rotated_query, rotated_key = day19_rope(query, key, cos_cache, sin_cache)
    # Note: Full RoPE implementation would require complex rotation logic
    # This is a placeholder test - actual implementation needed
    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape


@pytest.mark.parametrize("num_heads,seq_len,head_dim,description", ROPE_TEST_CASES)
def test_rope_cuda(num_heads, seq_len, head_dim, description):
    """Test CUDA RoPE"""
    try:
        from gpu_20days.cuda_kernels import day19_rope
    except ImportError:
        pytest.skip("CUDA kernels not built")

    device = ensure_cuda_device()

    print(f"Testing CUDA RoPE with shape ({num_heads}, {seq_len}, {head_dim}) ({description})...")
    # batch_size is always 1, so input is 3D
    query = torch.randn(num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    key = torch.randn(num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    cos_cache, sin_cache = create_rope_cache(seq_len, head_dim // 2, device)

    rotated_query, rotated_key = day19_rope(query, key, cos_cache, sin_cache)
    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape
