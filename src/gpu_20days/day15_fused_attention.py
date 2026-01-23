"""
Day 15: Fused Attention
Attention(Q, K, V) = softmax(Q @ K^T * scale) @ V
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
    seq_len,
    head_dim,
    scale,
    BLOCK_D: tl.constexpr,
):
    head = tl.program_id(0)
    row = tl.program_id(1)

    head_offset = head * seq_len * head_dim
    q_base = head_offset + row * head_dim
    d_idx = tl.arange(0, BLOCK_D)

    q = tl.load(Q_ptr + q_base + d_idx, mask=d_idx < head_dim, other=0.0)

    scores = tl.zeros([1], dtype=tl.float32)
    max_score = tl.zeros([1], dtype=tl.float32) - float("inf")
    output = tl.zeros([BLOCK_D], dtype=tl.float32)
    sum_exp = tl.zeros([1], dtype=tl.float32)

    for s in range(seq_len):
        k = tl.load(K_ptr + head_offset + s * head_dim + d_idx, mask=d_idx < head_dim, other=0.0)
        score = tl.sum(q * k) * scale

        new_max = tl.maximum(max_score, score)
        correction = tl.exp(max_score - new_max)
        exp_score = tl.exp(score - new_max)

        output = output * correction
        sum_exp = sum_exp * correction + exp_score

        v = tl.load(V_ptr + head_offset + s * head_dim + d_idx, mask=d_idx < head_dim, other=0.0)
        output += exp_score * v
        max_score = new_max

    output = output / sum_exp
    tl.store(output_ptr + q_base + d_idx, output, mask=d_idx < head_dim)


def day15_fused_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Day 15: Fused attention (batch_size is always 1)"""
    assert Q.dim() == 3, "Expected 3D tensor (num_heads, seq_len, head_dim)"
    num_heads, seq_len, head_dim = Q.shape

    if scale is None:
        scale = 1.0 / (head_dim**0.5)

    output = torch.empty_like(Q)
    BLOCK_D = triton.next_power_of_2(head_dim)

    day15_fused_attention_kernel[(num_heads, seq_len)](
        Q, K, V, output, seq_len, head_dim, scale, BLOCK_D=BLOCK_D
    )
    return output
