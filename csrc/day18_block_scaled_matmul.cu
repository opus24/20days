#include <cuda_runtime.h>
#include <cmath>
#define ceil(x, y) (((x) + (y) - 1) / (y))

// TODO: Block Scaled Matrix Multiplication kernel 구현
// 블록 단위로 스케일링을 적용한 행렬 곱셈
//
// 힌트:
// 1. 행렬 곱셈 수행
// 2. 블록 단위로 스케일링 적용
// 3. 수치 정밀도 고려
//
// 입력: A (M, K), B (K, N), scale (block_size)
// 출력: C (M, N) = scaled(A @ B)

__global__ void block_scaled_matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    float scale
) {
    // TODO: 구현하세요
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (row_idx < M && col_idx < N) {
        // TODO: Block Scaled Matmul 계산
        int c_idx = row_idx * N + col_idx;
        C[c_idx] = 0.0f;
    }
}

extern "C" void day18_block_scaled_matmul(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    float scale
) {
    // TODO: kernel launch configuration 설정
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(ceil(M, 16), ceil(N, 16));

    block_scaled_matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A, B, C, M, N, K, scale
    );
    cudaDeviceSynchronize();
}
