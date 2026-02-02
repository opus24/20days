"""
Day 18: Block Scaled Matrix Multiplication

Shared Memory 타일링 개념을 적용한 Matmul + Scale
"""

import torch
import triton
import triton.language as tl


@triton.jit
def day18_block_scaled_matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    K,
    N,  # A(M,K) @ B(K,N) = C(M,N)
    scale,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    block_scaled 기법은 Blackwell 이상에서만 가능
    따라서 Shared Memory 타일링 연습으로 수정
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    col_mask = col_offsets < N

    # 이 행의 결과를 누적
    accumulator = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)

    # K 축을 따라 순회 (내적)
    for k in range(K):
        # A[row_idx, k] - 스칼라
        a_val = tl.load(A_ptr + row_idx * K + k)
        # B[k, :] - 벡터
        b_vals = tl.load(B_ptr + k * N + col_offsets, mask=col_mask, other=0.0)
        accumulator += a_val * b_vals

    # Scale 적용
    accumulator = accumulator * scale

    # 결과 저장
    tl.store(C_ptr + row_idx * N + col_offsets, accumulator, mask=col_mask)


def day18_block_scaled_matmul(A: torch.Tensor, B: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Day 18: Block-scaled matrix multiplication

    C = (A @ B) * scale
    """
    M, K = A.shape
    _, N = B.shape

    C = torch.zeros(M, N, device=A.device, dtype=A.dtype)
    BLOCK_SIZE_N = triton.next_power_of_2(N)

    # 각 블록이 한 행 담당
    grid = (M,)

    day18_block_scaled_matmul_kernel[grid](
        A,
        B,
        C,
        M,
        K,
        N,
        scale,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return C
