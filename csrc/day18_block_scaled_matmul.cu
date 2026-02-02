#include <cuda_runtime.h>

#define TILE_SIZE 32
#define ceil_div(x, y) (((x) + (y) - 1) / (y))

// block_scaled 기법은 Blackwell 이상에서만 가능
// 따라서 Shared Memory 타일링 연습으로 수정
// A(M,K) @ B(K,N) * scale = C(M,N)

__global__ void block_scaled_matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N,
    float scale
) {
    // Shared Memory 선언
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // 블록/스레드 인덱스
    int bx = blockIdx.x;   // M 방향 블록
    int by = blockIdx.y;   // N 방향 블록
    int tx = threadIdx.x;  // 타일 내 행
    int ty = threadIdx.y;  // 타일 내 열

    // 이 스레드가 계산할 C의 위치
    int row = bx * TILE_SIZE + tx;
    int col = by * TILE_SIZE + ty;

    float accumulator = 0.0f;

    // K 축을 따라 타일 단위로 순회
    int num_tiles = ceil_div(K, TILE_SIZE);
    for (int t = 0; t < num_tiles; t++) {
        // A 타일 로드: A[row, t*TILE_SIZE + ty]
        int a_col = t * TILE_SIZE + ty;
        if (row < M && a_col < K) {
            As[tx][ty] = A[row * K + a_col];
        } else {
            As[tx][ty] = 0.0f;
        }

        // B 타일 로드: B[t*TILE_SIZE + tx, col]
        int b_row = t * TILE_SIZE + tx;
        if (b_row < K && col < N) {
            Bs[tx][ty] = B[b_row * N + col];
        } else {
            Bs[tx][ty] = 0.0f;
        }

        // 모든 스레드가 로드 완료할 때까지 대기
        __syncthreads();

        // 타일 내 내적 계산
        for (int k = 0; k < TILE_SIZE; k++) {
            accumulator += As[tx][k] * Bs[k][ty];
        }

        // 다음 타일 로드 전 대기
        __syncthreads();
    }

    // Scale 적용 후 저장
    if (row < M && col < N) {
        C[row * N + col] = accumulator * scale;
    }
}

extern "C" void day18_block_scaled_matmul(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    float scale
) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(ceil_div(M, TILE_SIZE), ceil_div(N, TILE_SIZE));

    block_scaled_matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A, B, C, M, K, N, scale
    );
    cudaDeviceSynchronize();
}
