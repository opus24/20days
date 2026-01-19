#include <cuda_runtime.h>
#include <cmath>
#define BLOCKSIZE 256
#define ceil(x, y) (((x) + (y) - 1) / (y))

// Softmax kernel 구현
// Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

__global__ void softmax_kernel(const float* input, float* output, int feature_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < feature_size) {
        float max_val = input[0];
        for (int i = 1; i < feature_size; i++) {
            if (input[i] > max_val) {
                max_val = input[i];
            }
        }

        float sum = 0.0f;
        for (int i = 0; i < feature_size; i++) {
            sum += expf(input[i] - max_val);
        }

        output[idx] = expf(input[idx] - max_val) / sum;
    }
}

extern "C" void day11_softmax(const float* input, float* output, int feature_size) {
    dim3 threadsPerBlock(BLOCKSIZE);
    dim3 blocksPerGrid(ceil(feature_size, BLOCKSIZE));

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, feature_size);
    cudaDeviceSynchronize();
}
