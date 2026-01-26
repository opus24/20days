"""
Day 16: Group GEMM
"""

import torch
import triton
import triton.language as tl


@triton.jit
def day16_group_gemm_kernel(A_ptr, B_ptr, C_ptr, num_groups, M, N, K, BLOCK_SIZE_N: tl.constexpr):
    """
    TODO: Group GEMM kernel 구현

    여러 행렬 곱셈을 배치로 처리합니다

    힌트:
    1. 여러 (A, B) 행렬 쌍을 배치로 처리
    2. 각 그룹의 행렬 곱셈을 병렬로 수행
    3. 메모리 레이아웃 최적화 (interleaved 또는 contiguous)
    """
    group_idx = tl.program_id(0)
    row_idx = tl.program_id(1)
    col_idx = tl.arange(0, BLOCK_SIZE_N)
    mask = col_idx < N

    A_group_offset = group_idx * M * K
    B_group_offset = group_idx * K * N
    C_group_offset = group_idx * M * N

    A_row_offset = row_idx * K
    C_row_offset = row_idx * N

    accumulator = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    for k in range(K):
        a_ptr = A_ptr + A_group_offset + A_row_offset + k
        b_ptr = B_ptr + B_group_offset + k * N + col_idx

        a_val = tl.load(a_ptr)
        b_vals = tl.load(b_ptr, mask=mask, other=0.0)
        accumulator += a_val * b_vals

    c_ptr = C_ptr + C_group_offset + C_row_offset + col_idx
    tl.store(c_ptr, accumulator, mask=mask)


def day16_group_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Day 16: Grouped general matrix multiplication"""

    num_groups, M, K = A.shape
    _, _, N = B.shape

    C = torch.zeros(num_groups, M, N, device=A.device, dtype=A.dtype)
    BLOCK_SIZE_N = triton.next_power_of_2(N)

    def grid(meta):
        return (num_groups, M)

    day16_group_gemm_kernel[grid](A, B, C, num_groups, M, N, K, BLOCK_SIZE_N=BLOCK_SIZE_N)
    return C
