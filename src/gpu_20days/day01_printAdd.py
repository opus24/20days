"""
Day 01: Print Add (Identity operation for demonstration)
"""
import torch
import triton
import triton.language as tl


@triton.jit
def day01_printAdd_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    input = tl.load(input_ptr + idx, mask=mask)
    # Basic element-wise operation (identity for demonstration)
    tl.store(output_ptr + idx, input, mask=mask)


def day01_printAdd(input: torch.Tensor) -> torch.Tensor:
    """Day 01: Print global indices (identity operation)"""
    BLOCK_SIZE = 256
    N = input.numel()
    output = torch.zeros_like(input)

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    day01_printAdd_kernel[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)
    return output



