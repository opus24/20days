import torch
import triton
import triton.language as tl


@triton.jit
def copy_matrix_kernel(A_ptr, B_ptr, N, BLOCK_SIZE: tl.constexpr):
    total = N * N
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total
    A = tl.load(A_ptr + idx, mask=mask)
    tl.store(B_ptr + idx, A, mask=mask)


# A, B are tensors on the GPU (N x N matrices)
def solve(A: torch.Tensor, B: torch.Tensor, N: int):
    BLOCK_SIZE = 256
    total = N * N

    def grid(meta):
        return (triton.cdiv(total, meta["BLOCK_SIZE"]),)

    copy_matrix_kernel[grid](A, B, N, BLOCK_SIZE=BLOCK_SIZE)

