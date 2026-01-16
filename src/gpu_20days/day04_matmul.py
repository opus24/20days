"""
Day 04: Matrix Multiplication
"""
import torch
import triton
import triton.language as tl


@triton.jit
def day04_matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, BLOCK_SIZE: tl.constexpr):
    # Simple 1D block per output element
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Calculate row and column for each element
    row = idx // K
    col = idx % K
    
    mask = (row < M) & (col < K)
    
    # Compute dot product for each output element
    accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for n in range(0, N):
        a_mask = (row < M) & (n < N)
        b_mask = (n < N) & (col < K)
        a = tl.load(A_ptr + row * N + n, mask=a_mask, other=0.0)
        b = tl.load(B_ptr + n * K + col, mask=b_mask, other=0.0)
        accumulator += a * b
    
    # Store result
    tl.store(C_ptr + idx, accumulator, mask=mask)


def day04_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Day 04: Matrix multiplication"""
    BLOCK_SIZE = 256
    M, N = A.shape
    _, K = B.shape
    total_elements = M * K
    C = torch.zeros((M, K), device=A.device, dtype=A.dtype)

    def grid(meta):
        return (triton.cdiv(total_elements, meta["BLOCK_SIZE"]),)

    day04_matmul_kernel[grid](A, B, C, M, N, K, BLOCK_SIZE=BLOCK_SIZE)
    return C

