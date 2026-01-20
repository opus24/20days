#include <cuda_runtime.h>
#define ceil(x, y) (((x) + (y) - 1) / (y))

// TODO: Fused Softmax kernel 구현
// 여러 연산을 하나의 커널로 융합하여 성능을 향상시킵니다
//
// 예시: Softmax + Scale + Mask 등을 하나의 커널로 융합
//
// 힌트:
// 1. 기본 Softmax 구현
// 2. 추가 연산 (scale, mask 등)을 같은 커널에서 처리
// 3. 메모리 접근 최소화
//
// 입력: input (seq_len, feature_size) - batch_size는 항상 1
//      scale (optional) - scaling factor
//      mask (optional) - attention mask
// 출력: output (seq_len, feature_size)

__global__ void fused_softmax_kernel(
    const float* input,
    float* output,
    const float* mask,
    int seq_len,
    int feature_size,
    float scale
) {
    // TODO: 구현하세요
    // Fused operations: scale -> mask -> softmax
    int seq_idx = blockIdx.x;
    int feature_idx = threadIdx.x;

    if (seq_idx < seq_len && feature_idx < feature_size) {
        int idx = seq_idx * feature_size + feature_idx;
        // TODO: Fused Softmax 계산
        output[idx] = input[idx];
    }
}

extern "C" void day14_fused_softmax(
    const float* input,
    float* output,
    const float* mask,
    int seq_len,
    int feature_size,
    float scale
) {
    // TODO: kernel launch configuration 설정
    // batch_size는 항상 1이므로 제거
    dim3 threadsPerBlock(feature_size);
    dim3 blocksPerGrid(seq_len);

    fused_softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, output, mask, seq_len, feature_size, scale
    );
    cudaDeviceSynchronize();
}
