#include <cuda_runtime.h>
#define ceil(x, y) (((x) + (y) - 1) / (y))

// TODO: Fused Softmax kernel 구현
// 예시: Softmax + Scale + Mask 등을 하나의 커널로 융합


__global__ void fused_softmax_kernel(
    const float* input,
    float* output,
    int seq_len,
    int feature_size
) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row >= seq_len || col >= feature_size) return;

    int base = row * feature_size;

    // find max value
    float max_val = -INFINITY;
    for (int i = 0; i < feature_size; i++) {
        float v = input[base + i];
        if (v > max_val) max_val = v;
    }

    // sum of exp values
    float sum = 0.0f;
    for (int i = 0; i < feature_size; i++) {
        sum += expf(input[base + i] - max_val);
    }

    // save result
    output[base + col] = expf(input[base + col] - max_val) / sum;
}

extern "C" void day14_fused_softmax(
    const float* input,
    float* output,
    const float* mask,
    int seq_len,
    int feature_size,
    float scale
) {
    fused_softmax_kernel<<<seq_len, feature_size>>>(
        input, output, seq_len, feature_size
    );
    cudaDeviceSynchronize();
}
