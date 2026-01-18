"""
Day 13: RMS Normalization
"""

import torch
import triton
import triton.language as tl


@triton.jit
def day13_rmsnorm_kernel(
    input_ptr, output_ptr, weight_ptr, batch_size, feature_size, eps, BLOCK_SIZE: tl.constexpr
):
    """
    TODO: RMS Normalization kernel 구현

    RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * weight

    힌트:
    1. 각 row의 x^2의 평균을 계산합니다 (RMS: Root Mean Square)
    2. x / sqrt(mean(x^2) + eps) 계산
    3. weight를 곱합니다
    """
    # TODO: 구현하세요
    # batch_idx = tl.program_id(0)
    # feature_idx = tl.arange(0, BLOCK_SIZE)
    # mask = feature_idx < feature_size
    pass


def day13_rmsnorm(
    input: torch.Tensor, weight: torch.Tensor = None, eps: float = 1e-5
) -> torch.Tensor:
    """Day 13: RMS normalization"""
    # TODO: 구현하세요
    BLOCK_SIZE = 256
    batch_size, feature_size = input.shape

    if weight is None:
        weight = torch.ones(feature_size, device=input.device, dtype=input.dtype)

    output = torch.zeros_like(input)

    def grid(meta):
        return (batch_size,)

    # day13_rmsnorm_kernel[grid](
    #     input, output, weight, batch_size, feature_size, eps, BLOCK_SIZE=BLOCK_SIZE
    # )
    return output
