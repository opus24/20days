#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int inner_K = blockDim.x * blockIdx.x + threadIdx.x;
    int inner_M = blockDim.y * blockIdx.y + threadIdx.y;
    if(inner_M < M && inner_K < K){
        float sum = 0;
        for (int inner_N = 0; inner_N < N; inner_N++){
            sum += A[inner_M * N + inner_N] * B[inner_N * K + inner_K];
        }
        C[inner_M * K + inner_K] = sum;
    }
}


#define ceil(x, y) (((x) + (y) - 1) / (y))
// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void day04_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(ceil(K,threadsPerBlock.x),ceil(M,threadsPerBlock.y));
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

