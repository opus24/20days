"""
Day 07: Matrix Copy
"""
import torch
import triton
import triton.language as tl


@triton.jit
def day07_matrixCopy_kernel(A_ptr, B_ptr, N, BLOCK_SIZE: tl.constexpr):
    total = N * N
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total
    A = tl.load(A_ptr + idx, mask=mask)
    tl.store(B_ptr + idx, A, mask=mask)


def day07_matrixCopy(A: torch.Tensor) -> torch.Tensor:
    """Day 07: Matrix copy"""
    BLOCK_SIZE = 256
    N = A.size(0)
    total = N * N
    B = torch.zeros_like(A)

    def grid(meta):
        return (triton.cdiv(total, meta["BLOCK_SIZE"]),)

    day07_matrixCopy_kernel[grid](A, B, N, BLOCK_SIZE=BLOCK_SIZE)
    return B

