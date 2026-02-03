"""
Day 19: RoPE (Rotary Position Embedding)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def day19_rope_kernel(
    query_ptr,
    key_ptr,
    rotated_query_ptr,
    rotated_key_ptr,
    cos_cache_ptr,
    sin_cache_ptr,
    num_heads,
    seq_len,
    head_dim,
):
    """RoPE: 한 프로그램이 (head, seq, pair) 하나만 처리. x'_2i = x_2i*cos - x_2i+1*sin 등"""
    half_dim = head_dim // 2

    # 그리드: (num_heads, seq_len, half_dim) → 프로그램 하나당 pair 하나
    head_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    pair_idx = tl.program_id(2)

    base = head_idx * seq_len * head_dim + seq_idx * head_dim
    d0 = base + 2 * pair_idx
    d1 = base + 2 * pair_idx + 1

    cache_idx = seq_idx * half_dim + pair_idx
    cos_val = tl.load(cos_cache_ptr + cache_idx)
    sin_val = tl.load(sin_cache_ptr + cache_idx)

    q0 = tl.load(query_ptr + d0)
    q1 = tl.load(query_ptr + d1)
    tl.store(rotated_query_ptr + d0, q0 * cos_val - q1 * sin_val)
    tl.store(rotated_query_ptr + d1, q0 * sin_val + q1 * cos_val)

    k0 = tl.load(key_ptr + d0)
    k1 = tl.load(key_ptr + d1)
    tl.store(rotated_key_ptr + d0, k0 * cos_val - k1 * sin_val)
    tl.store(rotated_key_ptr + d1, k0 * sin_val + k1 * cos_val)


def day19_rope(
    query: torch.Tensor, key: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Day 19: Rotary position embedding (batch_size는 항상 1)"""
    if query.dim() != 3:
        raise ValueError(
            "day19_rope expects 3D tensor (num_heads, seq_len, head_dim), batch_size is always 1"
        )

    num_heads, seq_len, head_dim = query.shape
    half_dim = head_dim // 2

    rotated_query = torch.empty_like(query)
    rotated_key = torch.empty_like(key)

    # 그리드: (num_heads, seq_len, half_dim) → 프로그램 하나당 pair 하나
    def grid(meta):
        return (num_heads, seq_len, half_dim)

    day19_rope_kernel[grid](
        query,
        key,
        rotated_query,
        rotated_key,
        cos_cache,
        sin_cache,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
    )
    return rotated_query, rotated_key
