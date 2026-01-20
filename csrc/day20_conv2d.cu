#include <cuda_runtime.h>
#define ceil(x, y) (((x) + (y) - 1) / (y))

// TODO: 2D Convolution kernel 구현
// 2D 컨볼루션 연산을 수행합니다
//
// 힌트:
// 1. 2D 슬라이딩 윈도우 패턴
// 2. 메모리 타일링 최적화
// 3. Shared memory 활용
//
// 입력: input (in_channels, height, width) - batch_size는 항상 1
//      kernel (out_channels, in_channels, kernel_h, kernel_w)
//      padding, stride
// 출력: output (out_channels, out_height, out_width)

__global__ void conv2d_kernel(
    const float* input,
    const float* kernel,
    float* output,
    int in_channels,
    int out_channels,
    int input_h,
    int input_w,
    int kernel_h,
    int kernel_w,
    int output_h,
    int output_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w
) {
    // TODO: 구현하세요
    // 2D 컨볼루션 계산
    int out_channel_idx = blockIdx.y;
    int out_row = blockIdx.x / output_w;
    int out_col = blockIdx.x % output_w;

    int thread_idx = threadIdx.x;

    if (out_channel_idx < out_channels && out_row < output_h && out_col < output_w) {
        // TODO: 2D Convolution 계산
        int out_idx = out_channel_idx * output_h * output_w +
                      out_row * output_w +
                      out_col;
        output[out_idx] = 0.0f;
    }
}

extern "C" void day20_conv2d(
    const float* input,
    const float* kernel,
    float* output,
    int in_channels,
    int out_channels,
    int input_h,
    int input_w,
    int kernel_h,
    int kernel_w,
    int output_h,
    int output_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w
) {
    // TODO: kernel launch configuration 설정
    // batch_size는 항상 1이므로 제거
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(output_h * output_w, out_channels);

    conv2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, kernel, output,
        in_channels, out_channels,
        input_h, input_w, kernel_h, kernel_w,
        output_h, output_w,
        pad_h, pad_w, stride_h, stride_w
    );
    cudaDeviceSynchronize();
}
