#include <cuda_runtime.h>
#define ceil(x, y) (((x) + (y) - 1) / (y))

// RoPE: (x_2i, x_2i+1) 쌍을 cos/sin으로 회전
// x'_2i = x_2i*cos - x_2i+1*sin,  x'_2i+1 = x_2i*sin + x_2i+1*cos

__global__ void rope_kernel(
    const float* query,
    const float* key,
    float* rotated_query,
    float* rotated_key,
    const float* cos_cache,
    const float* sin_cache,
    int num_heads,
    int seq_len,
    int head_dim
) {
    // 1차원 그리드: 스레드 하나가 한 (head, seq, pair) 담당
    int total_pairs = num_heads * seq_len * (head_dim / 2);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_pairs)
        return;

    int half_dim = head_dim / 2;
    int pairs_per_head = seq_len * half_dim;

    int head_idx = idx / pairs_per_head;
    int rest = idx % pairs_per_head;
    int seq_idx = rest / half_dim;
    int pair_idx = rest % half_dim;

    // query/key에서 이 쌍의 인덱스
    int base = head_idx * seq_len * head_dim + seq_idx * head_dim;
    int d0 = base + 2 * pair_idx;
    int d1 = base + 2 * pair_idx + 1;

    // cos, sin 캐시에서 읽기 (캐시 shape: seq_len, half_dim)
    int cache_idx = seq_idx * half_dim + pair_idx;
    float cos_val = cos_cache[cache_idx];
    float sin_val = sin_cache[cache_idx];

    // query 회전
    float q0 = query[d0];
    float q1 = query[d1];
    rotated_query[d0] = q0 * cos_val - q1 * sin_val;
    rotated_query[d1] = q0 * sin_val + q1 * cos_val;

    // key 회전
    float k0 = key[d0];
    float k1 = key[d1];
    rotated_key[d0] = k0 * cos_val - k1 * sin_val;
    rotated_key[d1] = k0 * sin_val + k1 * cos_val;
}

extern "C" void day19_rope(
    const float* query,
    const float* key,
    float* rotated_query,
    float* rotated_key,
    const float* cos_cache,
    const float* sin_cache,
    int num_heads,
    int seq_len,
    int head_dim
) {
    int half_dim = head_dim / 2;
    int total_pairs = num_heads * seq_len * half_dim;

    int threads = 256;
    int blocks = ceil(total_pairs, threads);

    rope_kernel<<<blocks, threads>>>(
        query, key, rotated_query, rotated_key,
        cos_cache, sin_cache,
        num_heads, seq_len, head_dim
    );
    cudaDeviceSynchronize();
}
