#include <cuda_runtime.h>
#define BLOCKSIZE 256
#define ceil(x, y) (((x) + (y) - 1) / (y))

__global__ void silu_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x
            + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] / (1 + exp(-input[idx]));
    }
}

// input, output are device pointers
extern "C" void day09_silu(const float* input, float* output, int N) {
    int threadsPerBlock = BLOCKSIZE;
    dim3 blocksPerGrid(ceil(N, threadsPerBlock));

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

