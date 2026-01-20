#include <cuda_runtime.h>
#define ceil(x, y) (((x) + (y) - 1) / (y))

// LayerNorm(x) = gamma * (x - mean) / sqrt(variance + eps) + beta

__global__ void layernorm_kernel(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int feature_size,
    float eps
) {
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (feature_idx < feature_size) {
        float mean = 0.0f;
        float variance = 0.0f;
        for (int i = 0; i < feature_size; i++){
            mean += input[i];
        }
        mean /= feature_size;
        for (int i = 0; i < feature_size; i++){
            variance += (input[i] - mean) * (input[i] - mean);
        }
        variance /= feature_size;
        output[feature_idx] = gamma[feature_idx] * (input[feature_idx] - mean) / sqrt(variance + eps) + beta[feature_idx];
    }
}
extern "C" void day12_layernorm(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int feature_size,
    float eps
) {
    const int BLOCKSIZE = 256;
    dim3 threadsPerBlock(BLOCKSIZE);
    dim3 blocksPerGrid(ceil(feature_size, BLOCKSIZE));

    layernorm_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, output, gamma, beta, feature_size, eps
    );
    cudaDeviceSynchronize();
}
