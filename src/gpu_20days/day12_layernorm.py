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
    feature_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    TODO: LayerNorm kernel 구현

    LayerNorm(x) = gamma * (x - mean) / sqrt(variance + eps) + beta

    """

    pid = tl.program_id(0)
    feature_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = feature_idx < feature_size

    # Compute mean
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, feature_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask_cols = cols < feature_size
        a = tl.load(input_ptr + cols, mask=mask_cols, other=0.0).to(tl.float32)
        _mean += tl.where(mask_cols, a, 0.0)
    mean = tl.sum(_mean, axis=0) / feature_size

    # Compute variance
    _variance = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, feature_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask_cols = cols < feature_size
        a = tl.load(input_ptr + cols, mask=mask_cols, other=0.0).to(tl.float32)
        _variance += tl.where(mask_cols, (a - mean) * (a - mean), 0.0)
    variance = tl.sum(_variance, axis=0) / feature_size

    input_vals = tl.load(input_ptr + feature_idx, mask=mask, other=0.0)
    gamma_vals = tl.load(gamma_ptr + feature_idx, mask=mask, other=1.0)
    beta_vals = tl.load(beta_ptr + feature_idx, mask=mask, other=0.0)
    normalized = (input_vals - mean) / tl.sqrt(variance + eps)
    output = gamma_vals * normalized + beta_vals
    tl.store(output_ptr + feature_idx, output, mask=mask)


def day12_layernorm(
    input: torch.Tensor, gamma: torch.Tensor = None, beta: torch.Tensor = None, eps: float = 1e-5
) -> torch.Tensor:
    """Day 12: Layer normalization (batch_size is always 1)"""
    BLOCK_SIZE = 256
    if input.dim() != 1:
        raise ValueError("day12_layernorm expects 1D tensor (feature_size), batch_size is always 1")

    feature_size = input.size(0)

    if gamma is None:
        gamma = torch.ones(feature_size, device=input.device, dtype=input.dtype)
    if beta is None:
        beta = torch.zeros(feature_size, device=input.device, dtype=input.dtype)

    output = torch.zeros_like(input)

    def grid(meta):
        return (triton.cdiv(feature_size, BLOCK_SIZE),)

    day12_layernorm_kernel[grid](
        input, output, gamma, beta, feature_size, eps, BLOCK_SIZE=BLOCK_SIZE
    )
    return output
