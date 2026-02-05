"""
Day 20: 2D Convolution
"""

import torch
import triton
import triton.language as tl


@triton.jit
def day20_conv2d_kernel(
    input_ptr,
    kernel_ptr,
    output_ptr,
    in_channels,
    out_channels,
    input_h,
    input_w,
    kernel_h,
    kernel_w,
    output_h,
    output_w,
    pad_h,
    pad_w,
    stride_h,
    stride_w,
):
    """2D Convolution: 한 프로그램이 출력 한 칸만 계산 (채널×커널H×커널W 합산)"""
    idx = tl.program_id(0)
    out_spatial = output_h * output_w
    total_out = out_channels * out_spatial

    if idx >= total_out:
        return

    out_channel_idx = idx // out_spatial
    out_linear = idx % out_spatial
    out_row = out_linear // output_w
    out_col = out_linear % output_w

    acc = 0.0
    for c in range(in_channels):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                in_row = out_row * stride_h + kh - pad_h
                in_col = out_col * stride_w + kw - pad_w
                in_bounds = (in_row >= 0) & (in_row < input_h) & (in_col >= 0) & (in_col < input_w)
                if in_bounds:
                    in_idx = c * input_h * input_w + in_row * input_w + in_col
                    k_idx = (
                        out_channel_idx * in_channels * kernel_h * kernel_w
                        + c * kernel_h * kernel_w
                        + kh * kernel_w
                        + kw
                    )
                    acc += tl.load(input_ptr + in_idx) * tl.load(kernel_ptr + k_idx)

    out_idx = out_channel_idx * output_h * output_w + out_row * output_w + out_col
    tl.store(output_ptr + out_idx, acc)


def day20_conv2d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    padding: tuple[int, int] = (0, 0),
    stride: tuple[int, int] = (1, 1),
) -> torch.Tensor:
    """Day 20: 2D Convolution (batch_size는 항상 1)"""
    if input.dim() != 3:
        raise ValueError(
            "day20_conv2d expects 3D tensor (in_channels, height, width), batch_size is always 1"
        )

    in_channels, input_h, input_w = input.shape
    out_channels, _, kernel_h, kernel_w = kernel.shape

    pad_h, pad_w = padding
    stride_h, stride_w = stride

    output_h = (input_h + 2 * pad_h - kernel_h) // stride_h + 1
    output_w = (input_w + 2 * pad_w - kernel_w) // stride_w + 1

    output = torch.zeros(out_channels, output_h, output_w, device=input.device, dtype=input.dtype)
    total_out = out_channels * output_h * output_w

    # 그리드: 출력 원소 개수만큼 프로그램 실행 (하나당 한 픽셀)
    def grid(meta):
        return (total_out,)

    day20_conv2d_kernel[grid](
        input,
        kernel,
        output,
        in_channels=in_channels,
        out_channels=out_channels,
        input_h=input_h,
        input_w=input_w,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        output_h=output_h,
        output_w=output_w,
        pad_h=pad_h,
        pad_w=pad_w,
        stride_h=stride_h,
        stride_w=stride_w,
    )
    return output
