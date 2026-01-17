"""
Day 09: SiLU Activation
"""
import torch
import triton
import triton.language as tl


@triton.jit
def day09_silu_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    input = tl.load(input_ptr + idx, mask=mask)
    # SiLU: output = input / (1 + exp(-input))
    output = input / (1.0 + tl.exp(-input))
    tl.store(output_ptr + idx, output, mask=mask)


def day09_silu(input: torch.Tensor) -> torch.Tensor:
    """Day 09: SiLU activation"""
    BLOCK_SIZE = 256
    N = input.numel()
    output = torch.zeros_like(input)

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    day09_silu_kernel[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)
    return output



