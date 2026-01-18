"""
Day 15: Fused Attention
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def day15_fused_attention_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    output_ptr,
    mask_ptr,
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    """
    TODO: Fused Attention kernel 구현

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    힌트:
    1. Q @ K^T 계산 (matrix multiplication)
    2. Scale by sqrt(d_k)
    3. Apply mask (optional)
    4. Softmax
    5. @ V (matrix multiplication)

    모든 연산을 하나의 커널로 융합하여 성능 최적화
    """
    # TODO: 구현하세요
    # batch_idx = tl.program_id(0)
    # head_idx = tl.program_id(1)
    # seq_idx = tl.program_id(2)
    # feature_idx = tl.arange(0, BLOCK_SIZE)
    # mask = feature_idx < head_dim
    pass


def day15_fused_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Day 15: Fused attention mechanism"""
    # TODO: 구현하세요
    BLOCK_SIZE = 256
    batch_size, num_heads, seq_len, head_dim = Q.shape

    if scale is None:
        scale = 1.0 / (head_dim**0.5)

    output = torch.zeros_like(Q)

    def grid(meta):
        return (batch_size, num_heads, seq_len)

    # day15_fused_attention_kernel[grid](
    #     Q, K, V, output, mask, batch_size, num_heads, seq_len, head_dim, scale, BLOCK_SIZE=BLOCK_SIZE
    # )
    return output
