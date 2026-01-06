#include <iostream>
#include <cuda_runtime.h>

#define ceil(x,y) (((x) + (y) - 1) / (y))
#define BLOCK_SIZE 256

__global__ void printThreadIndex(int N) {
    int idx = blockIdx.x * blockDim.x
            + threadIdx.x;

    if (idx < N) {
        printf("Block Index: %d, Thread Index: %d, Global Index: %d\n", blockIdx.x, threadIdx.x, idx);
    }
}

int main() {
    const int N = 1024;
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid(ceil(N, BLOCK_SIZE));

    printThreadIndex<<<blocksPerGrid, threadsPerBlock>>>(N);

    cudaDeviceSynchronize();

    return 0;
}
