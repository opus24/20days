"""
Day 05: Matrix Addition
"""
import torch
import triton
import triton.language as tl


@triton.jit
def day05_matrixAdd_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid_x = tl.program_id(0)  # row
    pid_y = tl.program_id(1)  # column block
    
    row = pid_x
    col = pid_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = (row < N) & (col < N)
    
    # Calculate index: idx = col + row * N
    idx = col + row * N
    
    A = tl.load(A_ptr + idx, mask=mask)
    B = tl.load(B_ptr + idx, mask=mask)
    C = A + B
    tl.store(C_ptr + idx, C, mask=mask)


def day05_matrixAdd(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Day 05: Matrix addition"""
    BLOCK_SIZE = 256
    N = A.size(0)
    C = torch.zeros_like(A)

    def grid(meta):
        return (N, triton.cdiv(N, meta["BLOCK_SIZE"]))

    day05_matrixAdd_kernel[grid](A, B, C, N, BLOCK_SIZE=BLOCK_SIZE)
    return C



