"""
Day 12: Layer Normalization
"""

import torch
import triton
import triton.language as tl


@triton.jit
def day12_layernorm_kernel(
    input_ptr,
    output_ptr,
    gamma_ptr,
    beta_ptr,
    batch_size,
    feature_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    TODO: LayerNorm kernel 구현

    LayerNorm(x) = gamma * (x - mean) / sqrt(variance + eps) + beta

    힌트:
    1. 각 row의 평균을 계산합니다 (reduction)
    2. 각 row의 분산을 계산합니다 (reduction)
    3. 정규화: (x - mean) / sqrt(variance + eps)
    4. affine transformation: gamma * normalized + beta
    """
    # TODO: 구현하세요
    # batch_idx = tl.program_id(0)
    # feature_idx = tl.arange(0, BLOCK_SIZE)
    # mask = feature_idx < feature_size
    pass


def day12_layernorm(
    input: torch.Tensor, gamma: torch.Tensor = None, beta: torch.Tensor = None, eps: float = 1e-5
) -> torch.Tensor:
    """Day 12: Layer normalization"""
    # TODO: 구현하세요
    BLOCK_SIZE = 256
    batch_size, feature_size = input.shape

    if gamma is None:
        gamma = torch.ones(feature_size, device=input.device, dtype=input.dtype)
    if beta is None:
        beta = torch.zeros(feature_size, device=input.device, dtype=input.dtype)

    output = torch.zeros_like(input)

    def grid(meta):
        return (batch_size,)

    # day12_layernorm_kernel[grid](
    #     input, output, gamma, beta, batch_size, feature_size, eps, BLOCK_SIZE=BLOCK_SIZE
    # )
    return output
