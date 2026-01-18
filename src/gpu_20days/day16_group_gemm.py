"""
Day 16: Group GEMM
"""

import torch
import triton
import triton.language as tl


@triton.jit
def day16_group_gemm_kernel(A_ptr, B_ptr, C_ptr, num_groups, M, N, K, BLOCK_SIZE: tl.constexpr):
    """
    TODO: Group GEMM kernel 구현

    여러 행렬 곱셈을 배치로 처리합니다

    힌트:
    1. 여러 (A, B) 행렬 쌍을 배치로 처리
    2. 각 그룹의 행렬 곱셈을 병렬로 수행
    3. 메모리 레이아웃 최적화 (interleaved 또는 contiguous)
    """
    # TODO: 구현하세요
    # group_idx = tl.program_id(0)
    # row_idx = tl.program_id(1)
    # col_idx = tl.arange(0, BLOCK_SIZE)
    # mask = col_idx < N
    pass


def day16_group_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Day 16: Grouped general matrix multiplication"""
    # TODO: 구현하세요
    BLOCK_SIZE = 256
    num_groups, M, K = A.shape
    _, _, N = B.shape

    C = torch.zeros(num_groups, M, N, device=A.device, dtype=A.dtype)

    def grid(meta):
        return (num_groups, M)

    # day16_group_gemm_kernel[grid](
    #     A, B, C, num_groups, M, N, K, BLOCK_SIZE=BLOCK_SIZE
    # )
    return C
