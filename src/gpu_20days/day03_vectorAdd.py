"""
Day 03: Vector Addition
"""
import torch
import triton
import triton.language as tl


@triton.jit
def day03_vectorAdd_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    A = tl.load(A_ptr + idx, mask=mask)
    B = tl.load(B_ptr + idx, mask=mask)
    C = A + B
    tl.store(C_ptr + idx, C, mask=mask)


def day03_vectorAdd(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Day 03: Vector addition"""
    BLOCK_SIZE = 256
    N = A.numel()
    C = torch.zeros_like(A)

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    day03_vectorAdd_kernel[grid](A, B, C, N, BLOCK_SIZE=BLOCK_SIZE)
    return C



