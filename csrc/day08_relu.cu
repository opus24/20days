#include <cuda_runtime.h>
#define ceil(x, y) (((x) + (y) - 1) / (y))

__global__ void relu_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x
            + threadIdx.x;
    if (idx < N) {
        if (input[idx] > 0) {
            output[idx] = input[idx];
        } else {
            output[idx] = 0;
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void day08_relu(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

