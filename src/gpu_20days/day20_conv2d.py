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
    BLOCK_SIZE: tl.constexpr,
):
    """
    TODO: 2D Convolution kernel 구현

    2D 컨볼루션 연산을 수행합니다

    힌트:
    1. 2D 슬라이딩 윈도우 패턴
    2. 메모리 타일링 최적화
    3. Shared memory 활용

    batch_size는 항상 1입니다.
    """
    # TODO: 구현하세요
    pass


def day20_conv2d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    padding: tuple[int, int] = (0, 0),
    stride: tuple[int, int] = (1, 1),
) -> torch.Tensor:
    """Day 20: Two-dimensional convolution (batch_size is always 1)"""
    # TODO: 구현하세요
    BLOCK_SIZE = 256
    # batch_size는 항상 1이므로 입력은 3D
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

    def grid(meta):
        return (out_channels, triton.cdiv(output_h * output_w, BLOCK_SIZE))

    # day20_conv2d_kernel[grid](
    #     input, kernel, output,
    #     in_channels, out_channels,
    #     input_h, input_w, kernel_h, kernel_w,
    #     output_h, output_w,
    #     pad_h, pad_w, stride_h, stride_w,
    #     BLOCK_SIZE=BLOCK_SIZE
    # )
    return output
