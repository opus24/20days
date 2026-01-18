#include <cuda_runtime.h>
#define ceil(x, y) (((x) + (y) - 1) / (y))

// TODO: Group GEMM kernel 구현
// 여러 행렬 곱셈을 배치로 처리합니다
//
// 힌트:
// 1. 여러 (A, B) 행렬 쌍을 배치로 처리
// 2. 각 그룹의 행렬 곱셈을 병렬로 수행
// 3. 메모리 레이아웃 최적화 (interleaved 또는 contiguous)
//
// 입력: A (num_groups, M, K), B (num_groups, K, N)
// 출력: C (num_groups, M, N)

__global__ void group_gemm_kernel(
    const float* A,
    const float* B,
    float* C,
    int num_groups,
    int M,
    int N,
    int K
) {
    // TODO: 구현하세요
    // 각 thread는 하나의 그룹의 행렬 곱셈을 처리할 수 있습니다
    int group_idx = blockIdx.x;
    int row_idx = blockIdx.y;
    int col_idx = threadIdx.x;

    if (group_idx < num_groups && row_idx < M && col_idx < N) {
        // TODO: Group GEMM 계산
        int c_idx = group_idx * M * N + row_idx * N + col_idx;
        C[c_idx] = 0.0f;
    }
}

extern "C" void day16_group_gemm(
    const float* A,
    const float* B,
    float* C,
    int num_groups,
    int M,
    int N,
    int K
) {
    // TODO: kernel launch configuration 설정
    dim3 threadsPerBlock(N);
    dim3 blocksPerGrid(num_groups, M);

    group_gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A, B, C, num_groups, M, N, K
    );
    cudaDeviceSynchronize();
}
