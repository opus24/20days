"""
Day 17: Persistent Matmul
"""

import torch
import triton
import triton.language as tl


@triton.jit
def day17_persistent_matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, BLOCK_SIZE: tl.constexpr):
    """
    TODO: Persistent Matmul kernel 구현

    Persistent 커널 패턴을 사용한 행렬 곱셈

    힌트:
    1. Persistent 커널: 커널이 여러 작업을 반복 처리
    2. 메모리 관리 최적화
    3. 워프 레벨 최적화
    """
    # TODO: 구현하세요
    # row_idx = tl.program_id(0)
    # col_idx = tl.program_id(1)
    pass


def day17_persistent_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Day 17: Persistent matrix multiplication"""
    # TODO: 구현하세요
    BLOCK_SIZE = 256
    M, K = A.shape
    _, N = B.shape

    C = torch.zeros(M, N, device=A.device, dtype=A.dtype)

    def grid(meta):
        return (M, N)

    # day17_persistent_matmul_kernel[grid](
    #     A, B, C, M, N, K, BLOCK_SIZE=BLOCK_SIZE
    # )
    return C
