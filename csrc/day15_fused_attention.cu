#include <cuda_runtime.h>
#define ceil(x, y) (((x) + (y) - 1) / (y))

// TODO: Fused Attention kernel 구현
// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
//
// 힌트:
// 1. Q @ K^T 계산 (matrix multiplication)
// 2. Scale by sqrt(d_k)
// 3. Apply mask (optional)
// 4. Softmax
// 5. @ V (matrix multiplication)
//
// 모든 연산을 하나의 커널로 융합하여 성능 최적화
//
// 입력: Q (num_heads, seq_len, head_dim) - batch_size는 항상 1
//      K (num_heads, seq_len, head_dim)
//      V (num_heads, seq_len, head_dim)
//      mask (optional) - attention mask
// 출력: output (num_heads, seq_len, head_dim)

__global__ void fused_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    const float* mask,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    // TODO: 구현하세요
    // Fused Attention: QK^T -> scale -> mask -> softmax -> @V
    int head_idx = blockIdx.y;
    int seq_idx = blockIdx.x;

    // TODO: Attention 계산
    // 각 thread는 하나의 head_dim element를 처리할 수 있습니다
    int feature_idx = threadIdx.x;

    if (head_idx < num_heads && seq_idx < seq_len && feature_idx < head_dim) {
        int idx = head_idx * seq_len * head_dim +
                  seq_idx * head_dim +
                  feature_idx;
        // TODO: Fused Attention 계산
        output[idx] = Q[idx];
    }
}

extern "C" void day15_fused_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    const float* mask,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    // TODO: kernel launch configuration 설정
    // batch_size는 항상 1이므로 제거
    dim3 threadsPerBlock(head_dim);
    dim3 blocksPerGrid(seq_len, num_heads);

    fused_attention_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        Q, K, V, output, mask, num_heads, seq_len, head_dim, scale
    );
    cudaDeviceSynchronize();
}
