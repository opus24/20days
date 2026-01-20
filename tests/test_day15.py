"""
Tests for Day 15: Fused Attention (CUDA + Triton)
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

# Test cases: (num_heads, seq_len, head_dim, description) - batch_size is always 1
ATTENTION_TEST_CASES = [
    (2, 8, 32, "small_2x8x32"),
    (4, 16, 64, "medium_4x16x64"),
    (8, 32, 128, "medium_8x32x128"),
]


@pytest.mark.parametrize("num_heads,seq_len,head_dim,description", ATTENTION_TEST_CASES)
def test_fused_attention_triton(num_heads, seq_len, head_dim, description):
    """Test Triton Fused Attention"""
    try:
        from gpu_20days.day15_fused_attention import day15_fused_attention
    except ImportError:
        pytest.skip("gpu_20days package not available")

    device = ensure_cuda_device()

    print(
        f"Testing Triton Fused Attention with shape ({num_heads}, {seq_len}, {head_dim}) ({description})..."
    )
    # batch_size is always 1, so input is 3D
    Q = torch.randn(num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    K = torch.randn(num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    V = torch.randn(num_heads, seq_len, head_dim, device=device, dtype=torch.float32)

    output = day15_fused_attention(Q, K, V)
    # Reference: scaled dot-product attention
    scale = 1.0 / (head_dim**0.5)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    expected = torch.matmul(attn, V)

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("num_heads,seq_len,head_dim,description", ATTENTION_TEST_CASES)
def test_fused_attention_cuda(num_heads, seq_len, head_dim, description):
    """Test CUDA Fused Attention"""
    try:
        from gpu_20days.cuda_kernels import day15_fused_attention
    except ImportError:
        pytest.skip("CUDA kernels not built")

    device = ensure_cuda_device()

    print(
        f"Testing CUDA Fused Attention with shape ({num_heads}, {seq_len}, {head_dim}) ({description})..."
    )
    # batch_size is always 1, so input is 3D
    Q = torch.randn(num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    K = torch.randn(num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    V = torch.randn(num_heads, seq_len, head_dim, device=device, dtype=torch.float32)

    output = day15_fused_attention(Q, K, V)
    scale = 1.0 / (head_dim**0.5)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    expected = torch.matmul(attn, V)

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
