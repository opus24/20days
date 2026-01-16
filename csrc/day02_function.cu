#include <cuda_runtime.h>

#define ceil(x,y) (((x) + (y) - 1) / (y))
#define BLOCK_SIZE 256

__device__ int add(int a, int b) {
    return a + b;
}

__global__ void functionKernel(int *h_A, int *h_B, int N) {
    int idx = blockIdx.x * blockDim.x
            + threadIdx.x;

    if (idx < N) {
        h_B[idx] = add(h_A[idx], h_A[idx]);
    }
}

// h_A, h_B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void day02_function(int *h_A, int *h_B, int N) {
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid(ceil(N, BLOCK_SIZE));
    
    functionKernel<<<blocksPerGrid, threadsPerBlock>>>(h_A, h_B, N);
    cudaDeviceSynchronize();
}

