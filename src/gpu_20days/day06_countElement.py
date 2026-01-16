"""
Day 06: Count Array Elements
"""
import torch
import triton
import triton.language as tl


@triton.jit
def day06_countElement_kernel(input_ptr, output_ptr, N, K, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    input = tl.load(input_ptr + idx, mask=mask)
    # Count how many elements equal K in this block
    count = tl.sum(tl.where(input == K, 1, 0))
    # Atomically add to output (output is a single value)
    tl.atomic_add(output_ptr, count)


def day06_countElement(input: torch.Tensor, K: int) -> int:
    """Day 06: Count elements equal to K. Returns count as integer."""
    BLOCK_SIZE = 256
    N = input.numel()
    # Output is a single value (count)
    output = torch.zeros(1, dtype=torch.int32, device=input.device)

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    day06_countElement_kernel[grid](input, output, N, K, BLOCK_SIZE=BLOCK_SIZE)
    
    return output.item()

