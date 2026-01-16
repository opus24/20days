#include <cuda_runtime.h>

__global__ void matrix_add(const float* A, const float* B, float* C, int N) {
    int col = blockDim.x * blockIdx.y
            + threadIdx.x;
    int row = blockIdx.x;

    if (col < N && row < N){
        int idx = col + row * N;
        C[idx] = A[idx] + B[idx];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void day05_matrixAdd(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    dim3 GridSize((N*N+N-1)/N, (N+threadsPerBlock-1)/threadsPerBlock);

    matrix_add<<<GridSize, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

