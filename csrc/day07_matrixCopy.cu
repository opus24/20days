#include <cuda_runtime.h>
#define ceil(x, y) (((x) + (y) - 1) / (y))

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int total = N * N;
    int idx = blockIdx.x * blockDim.x
            + threadIdx.x;
    if (idx < total) {
        B[idx] = A[idx];
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void day07_matrixCopy(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    dim3 blocksPerGrid(ceil(total, threadsPerBlock));
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    cudaDeviceSynchronize();
}

