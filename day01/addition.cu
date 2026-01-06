#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define ceil(x,y) (((x) + (y) - 1) / (y))
#define BLOCK_SIZE 256

__global__ void vectorAdd(int *A, int *B, int *C, int N) {
    int idx = blockIdx.x * blockDim.x
            + threadIdx.x;

    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1024;
    const int size = N * sizeof(int);
    
    // Allocate arrays in host memory
    int *h_A = (int*)malloc(size);
    int *h_B = (int*)malloc(size);
    int *h_C = (int*)malloc(size);
    
    // Allocate device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // A: ascending order from 1 to 1024, B: descending order from 1024 to 1
    for (int i = 0; i < N; i++) h_A[i] = i + 1;
    for (int i = 0; i < N; i++) h_B[i] = N - i;

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid(ceil(N, BLOCK_SIZE));
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    cudaDeviceSynchronize();
    
    // Copy results from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Verify results (print first 10 and last 10)
    printf("First 10 results:\n");
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %d + %d = %d\n", i, h_A[i], h_B[i], h_C[i]);
    }
    printf("\nLast 10 results:\n");
    for (int i = N - 10; i < N; i++) {
        printf("C[%d] = %d + %d = %d\n", i, h_A[i], h_B[i], h_C[i]);
    }

    return 0;
}
