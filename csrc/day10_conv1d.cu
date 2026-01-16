#include <cuda_runtime.h>
#define BLOCKSIZE 256
#define ceil(x, y) (((x) + (y) - 1) / (y))

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int idx = blockIdx.x * blockDim.x
            + threadIdx.x;
    if (idx < output_size) {
        float sum = 0;
        for (int i = 0; i < kernel_size; i++) {
            sum += input[idx + i] * kernel[i];
        }
        output[idx] = sum;
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void day10_conv1d(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = BLOCKSIZE;
    dim3 blocksPerGrid(ceil(output_size, threadsPerBlock));

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size,
                                                              kernel_size);
    cudaDeviceSynchronize();
}

