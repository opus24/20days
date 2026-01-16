"""
Day 02: Device Function Example
"""
import torch
import triton
import triton.language as tl


@triton.jit
def add(a, b):
    return a + b


@triton.jit
def day02_function_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    input = tl.load(input_ptr + idx, mask=mask)
    # Apply function: output = add(input, input) = input * 2
    output = add(input, input)
    tl.store(output_ptr + idx, output, mask=mask)


def day02_function(input: torch.Tensor) -> torch.Tensor:
    """Day 02: Device function example (doubles input)"""
    BLOCK_SIZE = 256
    N = input.numel()
    output = torch.zeros_like(input)

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    day02_function_kernel[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)
    return output

