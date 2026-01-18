"""
Day 18: Block Scaled Matrix Multiplication
"""

import torch
import triton
import triton.language as tl


@triton.jit
def day18_block_scaled_matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, scale, BLOCK_SIZE: tl.constexpr):
    """
    TODO: Block Scaled Matrix Multiplication kernel 구현

    블록 단위로 스케일링을 적용한 행렬 곱셈

    힌트:
    1. 행렬 곱셈 수행
    2. 블록 단위로 스케일링 적용
    3. 수치 정밀도 고려
    """
    # TODO: 구현하세요
    # row_idx = tl.program_id(0)
    # col_idx = tl.program_id(1)
    pass


def day18_block_scaled_matmul(A: torch.Tensor, B: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Day 18: Block-scaled matrix multiplication"""
    # TODO: 구현하세요
    BLOCK_SIZE = 256
    M, K = A.shape
    _, N = B.shape

    C = torch.zeros(M, N, device=A.device, dtype=A.dtype)

    def grid(meta):
        return (M, N)

    # day18_block_scaled_matmul_kernel[grid](
    #     A, B, C, M, N, K, scale, BLOCK_SIZE=BLOCK_SIZE
    # )
    return C
