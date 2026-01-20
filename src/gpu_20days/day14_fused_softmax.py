"""
Day 14: Fused Softmax
"""

import torch
import triton
import triton.language as tl


@triton.jit
def day14_fused_softmax_kernel(
    input_ptr,
    output_ptr,
    mask_ptr,
    seq_len,
    feature_size,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    """
    TODO: Fused Softmax kernel 구현

    여러 연산을 하나의 커널로 융합하여 성능을 향상시킵니다

    예시: Softmax + Scale + Mask 등을 하나의 커널로 융합

    힌트:
    1. 기본 Softmax 구현
    2. 추가 연산 (scale, mask 등)을 같은 커널에서 처리
    3. 메모리 접근 최소화

    batch_size는 항상 1입니다.
    """
    # TODO: 구현하세요
    pass


def day14_fused_softmax(
    input: torch.Tensor, mask: torch.Tensor = None, scale: float = 1.0
) -> torch.Tensor:
    """Day 14: Fused softmax operation (batch_size is always 1)"""
    # TODO: 구현하세요
    BLOCK_SIZE = 256
    # batch_size는 항상 1이므로 입력은 2D
    if input.dim() != 2:
        raise ValueError(
            "day14_fused_softmax expects 2D tensor (seq_len, feature_size), batch_size is always 1"
        )

    seq_len, feature_size = input.shape

    output = torch.zeros_like(input)

    def grid(meta):
        return (triton.cdiv(seq_len, BLOCK_SIZE),)

    # day14_fused_softmax_kernel[grid](
    #     input, output, mask, seq_len, feature_size, scale, BLOCK_SIZE=BLOCK_SIZE
    # )
    return output
