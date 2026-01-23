#include <cuda_runtime.h>

// Fused Attention: softmax(Q @ K^T * scale) @ V
// Each block handles one (head, query_row) pair
__global__ void fused_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int seq_len,
    int head_dim,
    float scale
) {
    int head = blockIdx.y;
    int row = blockIdx.x;
    int d = threadIdx.x;

    if (d >= head_dim) return;

    int head_offset = head * seq_len * head_dim;
    int q_base = head_offset + row * head_dim;

    float q_val = Q[q_base + d];

    // Online softmax variables
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float out_val = 0.0f;

    // Process each key-value pair
    for (int s = 0; s < seq_len; s++) {
        int kv_base = head_offset + s * head_dim;

        // Compute Q · K (need reduction across threads)
        __shared__ float dot_buffer[256];
        dot_buffer[d] = q_val * K[kv_base + d];
        __syncthreads();

        // Parallel reduction for dot product
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (d < stride && d + stride < head_dim) {
                dot_buffer[d] += dot_buffer[d + stride];
            }
            __syncthreads();
        }

        float score = dot_buffer[0] * scale;

        // Online softmax update
        float new_max = fmaxf(max_score, score);
        float correction = expf(max_score - new_max);
        float exp_score = expf(score - new_max);

        out_val = out_val * correction + exp_score * V[kv_base + d];
        sum_exp = sum_exp * correction + exp_score;
        max_score = new_max;
    }

    // Normalize and store
    output[q_base + d] = out_val / sum_exp;
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
    int threads = head_dim;
    if (threads > 256) threads = 256;

    dim3 blocks(seq_len, num_heads);

    fused_attention_kernel<<<blocks, threads>>>(
        Q, K, V, output, seq_len, head_dim, scale
    );
    cudaDeviceSynchronize();
}
