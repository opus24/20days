#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int tid = blockIdx.x * blockDim.x // 0 <= blockIdx.x < blockPerGrid, blockDim.x = threadsPerBlock = 256
             + threadIdx.x; // 0 <= threadIdx.x < threadsPerBlock = 256
    if (tid < N){
        C[tid] = A[tid] + B[tid];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void day03_vectorAdd(const float* A, const float* B, float* C, int N) {
    dim3 threadsPerBlock(256); // = BlockDim
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x); // = GridDim

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

