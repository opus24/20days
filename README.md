# GPU_Kernel_Study
## Triton/Cuda Kernel Study

**CUDA & Triton 20-Day Challenge**

This challenge focuses on implementing high-performance kernels in both CUDA (C++) and Triton (Python). All implementations have been executed and verified for correctness on LeetGPU. The curriculum has been structured in order of difficulty by myself.

### Day 01: Print
- **Description**: Basic kernel launch and thread indexing
- **Topics**: Thread/Block indexing, kernel launch configuration

### Day 02: Function
- **Description**: Device functions and kernel organization
- **Topics**: `__device__` functions, function calls in kernels

### Day 03: Vector Addition
- **Description**: Element-wise vector addition
- **Topics**: Memory management, host-device data transfer

### Day 04: Matrix Multiplication (MatMul)
- **Description**: Matrix multiplication kernel implementation
- **Topics**: 2D thread indexing, shared memory basics

### Day 05: MAC (Multiply-Accumulate)
- **Description**: Multiply-accumulate operations
- **Topics**: Reduction patterns, atomic operations

### Day 06: Count Array Element
- **Description**: Counting elements in arrays
- **Topics**: Conditional operations, counting patterns

### Day 07: Matrix Copy
- **Description**: Efficient matrix copying
- **Topics**: Memory coalescing, copy patterns

### Day 08: ReLU
- **Description**: Rectified Linear Unit activation function
- **Topics**: Element-wise operations, conditional assignments

### Day 09: SiLU
- **Description**: Sigmoid Linear Unit activation function
- **Topics**: Mathematical functions, element-wise operations

### Day 10: 1D Convolution
- **Description**: One-dimensional convolution operation
- **Topics**: Sliding window patterns, memory access patterns

### Day 11: Softmax
- **Description**: Softmax activation function
- **Topics**: Reduction operations, numerical stability

### Day 12: LayerNorm
- **Description**: Layer normalization
- **Topics**: Mean and variance computation, normalization

### Day 13: RMS Normalization
- **Description**: Root Mean Square normalization
- **Topics**: RMS computation, normalization patterns

### Day 14: Fused Softmax
- **Description**: Fused softmax operation for improved performance
- **Topics**: Kernel fusion, memory optimization
- **Note**: Based on [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)

### Day 15: Fused Attention
- **Description**: Fused attention mechanism implementation
- **Topics**: Attention computation, kernel fusion
- **Note**: Based on [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)

### Day 16: Group GEMM
- **Description**: Grouped general matrix multiplication
- **Topics**: Batch matrix operations, optimization techniques
- **Note**: Based on [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)

### Day 17: Persistent Matmul
- **Description**: Persistent matrix multiplication kernel
- **Topics**: Persistent kernels, memory management
- **Note**: Based on [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)

### Day 18: Block Scaled Matrix Multiplication
- **Description**: Block-scaled matrix multiplication
- **Topics**: Block scaling, numerical precision
- **Note**: Based on [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)

### Day 19: RoPE (Rotary Position Embedding)
- **Description**: Rotary position embedding implementation
- **Topics**: Trigonometric operations, position encoding

### Day 20: 2D Convolution
- **Description**: Two-dimensional convolution operation
- **Topics**: 2D sliding window, memory tiling, shared memory

## Requirements

- LeetGPU account (for benchmarking)

## Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [Triton Documentation](https://triton-lang.org/)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [LeetGPU Platform](https://leetgpu.com/)
