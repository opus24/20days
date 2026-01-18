#include <cuda_runtime.h>
#include <cmath>
#define ceil(x, y) (((x) + (y) - 1) / (y))

// TODO: RMS Normalization kernel 구현
// RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * weight
//
// 힌트:
// 1. 각 row의 x^2의 평균을 계산합니다 (RMS: Root Mean Square)
// 2. x / sqrt(mean(x^2) + eps) 계산
// 3. weight를 곱합니다
//
// 입력: input (batch_size, feature_size)
//      weight (feature_size) - optional, 없으면 1.0 사용
//      eps - 작은 상수 (기본값 1e-5)
// 출력: output (batch_size, feature_size)

__global__ void rmsnorm_kernel(
    const float* input,
    float* output,
    const float* weight,
    int batch_size,
    int feature_size,
    float eps
) {
    // TODO: 구현하세요
    int batch_idx = blockIdx.x;
    int feature_idx = threadIdx.x;

    if (batch_idx < batch_size && feature_idx < feature_size) {
        // TODO: RMS Normalization 계산
        output[batch_idx * feature_size + feature_idx] = input[batch_idx * feature_size + feature_idx];
    }
}

extern "C" void day13_rmsnorm(
    const float* input,
    float* output,
    const float* weight,
    int batch_size,
    int feature_size,
    float eps
) {
    // TODO: kernel launch configuration 설정
    dim3 threadsPerBlock(feature_size);
    dim3 blocksPerGrid(batch_size);

    rmsnorm_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, output, weight, batch_size, feature_size, eps
    );
    cudaDeviceSynchronize();
}
