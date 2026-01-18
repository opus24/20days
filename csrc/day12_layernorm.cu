#include <cuda_runtime.h>
#define ceil(x, y) (((x) + (y) - 1) / (y))

// TODO: LayerNorm kernel 구현
// LayerNorm(x) = gamma * (x - mean) / sqrt(variance + eps) + beta
//
// 힌트:
// 1. 각 row의 평균을 계산합니다 (reduction)
// 2. 각 row의 분산을 계산합니다 (reduction)
// 3. 정규화: (x - mean) / sqrt(variance + eps)
// 4. affine transformation: gamma * normalized + beta
//
// 입력: input (batch_size, feature_size)
//      gamma (feature_size) - optional, 없으면 1.0 사용
//      beta (feature_size) - optional, 없으면 0.0 사용
//      eps - 작은 상수 (기본값 1e-5)
// 출력: output (batch_size, feature_size)

__global__ void layernorm_kernel(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int batch_size,
    int feature_size,
    float eps
) {
    // TODO: 구현하세요
    int batch_idx = blockIdx.x;
    int feature_idx = threadIdx.x;

    if (batch_idx < batch_size && feature_idx < feature_size) {
        // TODO: LayerNorm 계산
        output[batch_idx * feature_size + feature_idx] = input[batch_idx * feature_size + feature_idx];
    }
}

extern "C" void day12_layernorm(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int batch_size,
    int feature_size,
    float eps
) {
    // TODO: kernel launch configuration 설정
    dim3 threadsPerBlock(feature_size);
    dim3 blocksPerGrid(batch_size);

    layernorm_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, output, gamma, beta, batch_size, feature_size, eps
    );
    cudaDeviceSynchronize();
}
