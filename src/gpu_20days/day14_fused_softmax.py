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
    stride,
    feature_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Calculate softmax for each row"""
    row_idx = tl.program_id(0)
    row_start = row_idx * stride
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < feature_size

    # load data
    x = tl.load(input_ptr + row_start + offsets, mask=mask, other=-float("inf"))

    # softmax: exp(x - max) / sum(exp(x - max))
    x = tl.exp(x - tl.max(x, axis=0))
    x = x / tl.sum(x, axis=0)

    # save result
    tl.store(output_ptr + row_start + offsets, x, mask=mask)


def day14_fused_softmax(
    input: torch.Tensor, mask: torch.Tensor = None, scale: float = 1.0
) -> torch.Tensor:
    """Day 14: Fused softmax operation (batch_size is always 1)"""
    assert input.dim() == 2, "Expected 2D tensor"
    seq_len, feature_size = input.shape
    output = torch.empty_like(input)

    # BLOCK_SIZE must be greater than or equal to feature_size
    BLOCK_SIZE = triton.next_power_of_2(feature_size)

    day14_fused_softmax_kernel[(seq_len,)](
        input, output, input.stride(0), feature_size, BLOCK_SIZE=BLOCK_SIZE
    )
    return output
