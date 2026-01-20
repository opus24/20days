#include <cuda_runtime.h>
#include <cmath>
#define BLOCKSIZE 256
#define ceil(x, y) (((x) + (y) - 1) / (y))

// RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * weight

__global__ void rmsnorm_kernel(
    const float* input,
    float* output,
    const float* weight,
    int feature_size,
    float eps
) {
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (feature_idx < feature_size) {
        float mean_sq = 0.0f;
        for (int i = 0; i < feature_size; i++){
            mean_sq += input[i] * input[i];
        }
        mean_sq /= feature_size;
        output[feature_idx] = input[feature_idx] / sqrt(mean_sq + eps) * weight[feature_idx];
    }
}

extern "C" void day13_rmsnorm(
    const float* input,
    float* output,
    const float* weight,
    int feature_size,
    float eps
) {
    // TODO: kernel launch configuration 설정
    // batch_size는 항상 1이므로 제거
    dim3 threadsPerBlock(BLOCKSIZE);
    dim3 blocksPerGrid(ceil(feature_size, BLOCKSIZE));

    rmsnorm_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, output, weight, feature_size, eps
    );
    cudaDeviceSynchronize();
}
