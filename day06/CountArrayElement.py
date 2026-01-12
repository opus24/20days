import torch
import triton
import triton.language as tl


@triton.jit
def count_equal_kernel(input_ptr, output_ptr, N, K, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    input = tl.load(input_ptr + idx, mask=mask)
    # Count how many elements equal K in this block
    count = tl.sum(tl.where(input == K, 1, 0))
    # Atomically add to output (output is a single value)
    tl.atomic_add(output_ptr, count)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, K: int):
    BLOCK_SIZE = 256

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    count_equal_kernel[grid](input, output, N, K, BLOCK_SIZE=BLOCK_SIZE)
