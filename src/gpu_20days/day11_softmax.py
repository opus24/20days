"""
Day 11: Softmax Activation
"""

import torch
import triton
import triton.language as tl


@triton.jit
def day11_softmax_kernel(input_ptr, output_ptr, feature_size, BLOCK_SIZE: tl.constexpr):
    """
    Softmax kernel 구현

    Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    """
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < feature_size

    max_val = tl.load(input_ptr)
    for i in range(1, feature_size):
        max_val = tl.maximum(max_val, tl.load(input_ptr + i))

    sum_val = 0.0
    for i in range(feature_size):
        sum_val += tl.exp(tl.load(input_ptr + i) - max_val)

    output = tl.exp(tl.load(input_ptr + idx) - max_val) / sum_val
    tl.store(output_ptr + idx, output, mask=mask)


def day11_softmax(input: torch.Tensor) -> torch.Tensor:
    """Day 11: Softmax activation"""
    # Ensure 1D input
    if input.dim() != 1:
        raise ValueError("day11_softmax expects 1D tensor")

    feature_size = input.size(0)
    output = torch.zeros_like(input)

    BLOCK_SIZE = 256

    def grid(meta):
        return (triton.cdiv(feature_size, BLOCK_SIZE),)

    day11_softmax_kernel[grid](input, output, feature_size, BLOCK_SIZE=BLOCK_SIZE)

    return output
