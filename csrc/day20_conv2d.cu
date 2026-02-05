#include <cuda_runtime.h>
#define ceil(x, y) (((x) + (y) - 1) / (y))

// 2D Convolution: 출력 한 픽셀당 스레드 하나. 입력 채널 × 커널 높이 × 커널 너비만큼 합산.

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
    // 1차원 그리드: 스레드 하나가 출력 한 칸 담당
    int total_out = out_channels * output_h * output_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_out)
        return;

    // 출력 인덱스 (oc, oh, ow) 계산
    int out_spatial = output_h * output_w;
    int out_channel_idx = idx / out_spatial;
    int out_linear = idx % out_spatial;
    int out_row = out_linear / output_w;
    int out_col = out_linear % output_w;

    // 이 출력 칸에 대해서 입력 × 커널 합산
    float sum = 0.0f;
    for (int c = 0; c < in_channels; c++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int in_h = out_row * stride_h + kh - pad_h;
                int in_w = out_col * stride_w + kw - pad_w;
                if (in_h >= 0 && in_h < input_h && in_w >= 0 && in_w < input_w) {
                    int in_idx = c * input_h * input_w + in_h * input_w + in_w;
                    int k_idx = out_channel_idx * in_channels * kernel_h * kernel_w
                                + c * kernel_h * kernel_w + kh * kernel_w + kw;
                    sum += input[in_idx] * kernel[k_idx];
                }
            }
        }
    }

    int out_idx = out_channel_idx * output_h * output_w + out_row * output_w + out_col;
    output[out_idx] = sum;
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
    int total_out = output_h * output_w * out_channels;
    int threads = 256;
    int blocks = ceil(total_out, threads);

    conv2d_kernel<<<blocks, threads>>>(
        input, kernel, output,
        in_channels, out_channels,
        input_h, input_w, kernel_h, kernel_w,
        output_h, output_w,
        pad_h, pad_w, stride_h, stride_w
    );
    cudaDeviceSynchronize();
}
