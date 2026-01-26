#include <cuda_runtime.h>
#define ceil(x, y) (((x) + (y) - 1) / (y))

__global__ void group_gemm_kernel(
    const float* A,
    const float* B,
    float* C,
    int num_groups,
    int M,
    int N,
    int K
) {
    int group_idx = blockIdx.x;
    int row_idx = blockIdx.y;
    int col_idx = threadIdx.x;

    if (group_idx < num_groups && row_idx < M && col_idx < N) {
        int A_group_offset = group_idx * M * K;
        int B_group_offset = group_idx * K * N;
        int C_group_offset = group_idx * M * N;

        int A_row_offset = row_idx * K;
        int C_row_offset = row_idx * N;

        float accumulator = 0.0f;
        for (int k = 0; k < K; k++) {
            int a_idx = A_group_offset + A_row_offset + k;
            int b_idx = B_group_offset + k * N + col_idx;

            float a_val = A[a_idx];
            float b_val = B[b_idx];

            accumulator += a_val * b_val;
        }

        int c_idx = C_group_offset + C_row_offset + col_idx;
        C[c_idx] = accumulator;
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
    dim3 threadsPerBlock(N);
    dim3 blocksPerGrid(num_groups, M);

    group_gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A, B, C, num_groups, M, N, K
    );
    cudaDeviceSynchronize();
}
