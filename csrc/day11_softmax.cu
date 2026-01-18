#include <cuda_runtime.h>
#define BLOCKSIZE 256
#define ceil(x, y) (((x) + (y) - 1) / (y))

// TODO: Softmax kernel 구현
// Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
//
// 힌트:
// 1. 먼저 각 row의 최대값을 찾아야 합니다 (reduction)
// 2. exp(x_i - max)를 계산합니다
// 3. exp 값들의 합을 계산합니다 (reduction)
// 4. 각 값을 합으로 나눕니다
//
// 입력: input (batch_size, feature_size)
// 출력: output (batch_size, feature_size)

__global__ void softmax_kernel(const float* input, float* output, int feature_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < feature_size) {
        float sum = atomicAdd(&sum, exp(input[idx]));
        output[idx] = input[idx];
    }
}

extern "C" void day11_softmax(const float* input, float* output, int feature_size) {
    // TODO: kernel launch configuration 설정
    dim3 threadsPerBlock(BLOCKSIZE));
    dim3 blocksPerGrid(ceil(feature_size, BLOCKSIZE));

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, feature_size);
    cudaDeviceSynchronize();
}
