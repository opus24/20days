#include <cuda_runtime.h>
#include <cmath>
#define ceil(x, y) (((x) + (y) - 1) / (y))

// TODO: RoPE (Rotary Position Embedding) kernel 구현
// 회전 위치 임베딩을 적용합니다
//
// 힌트:
// 1. 삼각함수 연산 (sin, cos) 사용
// 2. 위치 정보를 이용한 회전 변환
// 3. Query와 Key에 각각 적용
//
// 입력: query (batch_size, num_heads, seq_len, head_dim)
//      key (batch_size, num_heads, seq_len, head_dim)
//      cos_cache (seq_len, head_dim / 2)
//      sin_cache (seq_len, head_dim / 2)
// 출력: rotated_query, rotated_key

__global__ void rope_kernel(
    const float* query,
    const float* key,
    float* rotated_query,
    float* rotated_key,
    const float* cos_cache,
    const float* sin_cache,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    // TODO: 구현하세요
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int seq_idx = blockIdx.x;
    int dim_idx = threadIdx.x;

    if (batch_idx < batch_size && head_idx < num_heads &&
        seq_idx < seq_len && dim_idx < head_dim) {
        // TODO: RoPE 계산
        int q_idx = batch_idx * num_heads * seq_len * head_dim +
                    head_idx * seq_len * head_dim +
                    seq_idx * head_dim +
                    dim_idx;
        rotated_query[q_idx] = query[q_idx];
        rotated_key[q_idx] = key[q_idx];
    }
}

extern "C" void day19_rope(
    const float* query,
    const float* key,
    float* rotated_query,
    float* rotated_key,
    const float* cos_cache,
    const float* sin_cache,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    // TODO: kernel launch configuration 설정
    dim3 threadsPerBlock(head_dim);
    dim3 blocksPerGrid(seq_len, num_heads, batch_size);

    rope_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        query, key, rotated_query, rotated_key,
        cos_cache, sin_cache,
        batch_size, num_heads, seq_len, head_dim
    );
    cudaDeviceSynchronize();
}
