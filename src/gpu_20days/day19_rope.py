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
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    TODO: RoPE kernel 구현

    회전 위치 임베딩을 적용합니다

    힌트:
    1. 삼각함수 연산 (sin, cos) 사용
    2. 위치 정보를 이용한 회전 변환
    3. Query와 Key에 각각 적용
    """
    # TODO: 구현하세요
    # batch_idx = tl.program_id(0)
    # head_idx = tl.program_id(1)
    # seq_idx = tl.program_id(2)
    # dim_idx = tl.arange(0, BLOCK_SIZE)
    # mask = dim_idx < head_dim
    pass


def day19_rope(
    query: torch.Tensor, key: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Day 19: Rotary position embedding"""
    # TODO: 구현하세요
    BLOCK_SIZE = 256
    batch_size, num_heads, seq_len, head_dim = query.shape

    rotated_query = torch.zeros_like(query)
    rotated_key = torch.zeros_like(key)

    def grid(meta):
        return (batch_size, num_heads, seq_len)

    # day19_rope_kernel[grid](
    #     query, key, rotated_query, rotated_key,
    #     cos_cache, sin_cache,
    #     batch_size, num_heads, seq_len, head_dim,
    #     BLOCK_SIZE=BLOCK_SIZE
    # )
    return rotated_query, rotated_key
