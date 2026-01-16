#include <cuda_runtime.h>
#define ceil(x, y) (((x) + (y) - 1) / (y))

__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    int idx = blockIdx.x * blockDim.x
            + threadIdx.x;
    if (idx < N && input[idx] == K) {
        atomicAdd(&output[0], 1);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void day06_countElement(const int* input, int* output, int N, int K) {
    int threadsPerBlock = 256;
    dim3 blocksPerGrid(ceil(N, threadsPerBlock));

    count_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, K);
    cudaDeviceSynchronize();
}

