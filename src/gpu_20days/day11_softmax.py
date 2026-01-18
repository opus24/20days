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

    각 row에 대해 독립적으로 softmax를 적용합니다
    """
    row_idx = tl.program_id(0)
    col_idx = tl.arange(0, BLOCK_SIZE)
    mask = col_idx < feature_size

    # Load input
    input_offset = row_idx * feature_size + col_idx
    input_vals = tl.load(input_ptr + input_offset, mask=mask, other=-float("inf"))

    # Step 1: Find max
    row_max = tl.max(input_vals, axis=0)

    # Step 2: Compute exp(x - max)
    exp_vals = tl.exp(input_vals - row_max)

    # Step 3: Compute sum
    row_sum = tl.sum(exp_vals, axis=0)

    # Step 4: Normalize
    output_vals = exp_vals / row_sum

    # Store output
    tl.store(output_ptr + input_offset, output_vals, mask=mask)


def day11_softmax(input: torch.Tensor) -> torch.Tensor:
    """Day 11: Softmax activation"""
    BLOCK_SIZE = 256

    # Handle different input shapes
    if input.dim() == 1:
        input = input.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    original_shape = input.shape
    batch_size = original_shape[0]
    feature_size = original_shape[-1]

    # Flatten to 2D if needed
    if input.dim() > 2:
        input = input.view(-1, feature_size)
        batch_size = input.size(0)

    output = torch.zeros_like(input)

    def grid(meta):
        return (batch_size,)

    day11_softmax_kernel[grid](input, output, feature_size, BLOCK_SIZE=BLOCK_SIZE)

    # Reshape to original shape
    output = output.view(original_shape)
    if squeeze_output:
        output = output.squeeze(0)

    return output
