#include <cuda_runtime.h>
#define ceil(x, y) (((x) + (y) - 1) / (y))

// TODO: Persistent Matmul kernel 구현
// Persistent 커널 패턴을 사용한 행렬 곱셈
//
// 힌트:
// 1. Persistent 커널: 커널이 여러 작업을 반복 처리
// 2. 메모리 관리 최적화
// 3. 워프 레벨 최적화
//
// 입력: A (M, K), B (K, N)
// 출력: C (M, N)

__global__ void persistent_matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    // TODO: 구현하세요
    // Persistent 커널 패턴 사용
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (row_idx < M && col_idx < N) {
        // TODO: Persistent Matmul 계산
        int c_idx = row_idx * N + col_idx;
        C[c_idx] = 0.0f;
    }
}

extern "C" void day17_persistent_matmul(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    // TODO: kernel launch configuration 설정
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(ceil(M, 16), ceil(N, 16));

    persistent_matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A, B, C, M, N, K
    );
    cudaDeviceSynchronize();
}
