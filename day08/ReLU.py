import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    input = tl.load(input_ptr + idx, mask=mask)
    # ReLU: output = max(0, input)
    output = tl.where(input > 0, input, 0.0)
    tl.store(output_ptr + idx, output, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 256

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    relu_kernel[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)

