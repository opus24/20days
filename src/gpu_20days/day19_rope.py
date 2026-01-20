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
    BLOCK_SIZE: tl.constexpr,
):
    """
    TODO: RoPE kernel 구현

    회전 위치 임베딩을 적용합니다

    힌트:
    1. 삼각함수 연산 (sin, cos) 사용
    2. 위치 정보를 이용한 회전 변환
    3. Query와 Key에 각각 적용

    batch_size는 항상 1입니다.
    """
    # TODO: 구현하세요
    pass


def day19_rope(
    query: torch.Tensor, key: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Day 19: Rotary position embedding (batch_size is always 1)"""
    # TODO: 구현하세요
    BLOCK_SIZE = 256
    # batch_size는 항상 1이므로 입력은 3D
    if query.dim() != 3:
        raise ValueError(
            "day19_rope expects 3D tensor (num_heads, seq_len, head_dim), batch_size is always 1"
        )

    num_heads, seq_len, head_dim = query.shape

    rotated_query = torch.zeros_like(query)
    rotated_key = torch.zeros_like(key)

    def grid(meta):
        return (num_heads, triton.cdiv(seq_len, BLOCK_SIZE))

    # day19_rope_kernel[grid](
    #     query, key, rotated_query, rotated_key,
    #     cos_cache, sin_cache,
    #     num_heads, seq_len, head_dim,
    #     BLOCK_SIZE=BLOCK_SIZE
    # )
    return rotated_query, rotated_key
