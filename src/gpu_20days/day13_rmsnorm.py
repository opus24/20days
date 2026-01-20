"""
Day 13: RMS Normalization
"""

import torch
import triton
import triton.language as tl


@triton.jit
def day13_rmsnorm_kernel(
    input_ptr, output_ptr, weight_ptr, feature_size, eps, BLOCK_SIZE: tl.constexpr
):
    """

    RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * weight

    """
    pid = tl.program_id(0)
    feature_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = feature_idx < feature_size

    _mean_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, feature_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask_cols = cols < feature_size
        a = tl.load(input_ptr + cols, mask=mask_cols, other=0.0).to(tl.float32)
        _mean_sq += tl.where(mask_cols, a * a, 0.0)
    mean_sq = tl.sum(_mean_sq, axis=0) / feature_size
    input = tl.load(input_ptr + feature_idx, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + feature_idx, mask=mask, other=1.0)
    normalized = input / tl.sqrt(mean_sq + eps)
    output = normalized * weight
    tl.store(output_ptr + feature_idx, output, mask=mask)


def day13_rmsnorm(
    input: torch.Tensor, weight: torch.Tensor = None, eps: float = 1e-5
) -> torch.Tensor:
    """Day 13: RMS normalization (batch_size is always 1)"""
    BLOCK_SIZE = 256
    if input.dim() != 1:
        raise ValueError("day13_rmsnorm expects 1D tensor (feature_size), batch_size is always 1")

    feature_size = input.size(0)

    if weight is None:
        weight = torch.ones(feature_size, device=input.device, dtype=input.dtype)

    output = torch.zeros_like(input)

    def grid(meta):
        return (triton.cdiv(feature_size, BLOCK_SIZE),)

    day13_rmsnorm_kernel[grid](input, output, weight, feature_size, eps, BLOCK_SIZE=BLOCK_SIZE)
    return output
