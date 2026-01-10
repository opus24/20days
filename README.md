# GPU_Kernel_Study
## Triton/Cuda Kernel Study

**CUDA & Triton 100-Day Challenge**

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

### Day 11: Dot Product
- **Description**: Vector dot product computation
- **Topics**: Reduction operations, parallel reduction

### Day 12: GEMM (General Matrix Multiply)
- **Description**: Optimized general matrix multiplication
- **Topics**: Tiling, shared memory optimization, register blocking

### Day 13: MSE (Mean Squared Error)
- **Description**: Mean squared error loss computation
- **Topics**: Reduction operations, error computation

### Day 14: Top-K
- **Description**: Finding top-K elements
- **Topics**: Selection algorithms, sorting patterns

### Day 15: Top-P (Nucleus Sampling)
- **Description**: Top-P sampling for language models
- **Topics**: Cumulative sum, threshold-based selection

### Day 16: RoPE (Rotary Position Embedding)
- **Description**: Rotary position embedding implementation
- **Topics**: Trigonometric operations, position encoding

### Day 17: Softmax
- **Description**: Softmax activation function
- **Topics**: Reduction operations, numerical stability

### Day 18: LayerNorm
- **Description**: Layer normalization
- **Topics**: Mean and variance computation, normalization

### Day 19: RMS Normalization
- **Description**: Root Mean Square normalization
- **Topics**: RMS computation, normalization patterns

### Day 20: 2D Convolution
- **Description**: Two-dimensional convolution operation
- **Topics**: 2D sliding window, memory tiling, shared memory

## Requirements

- LeetGPU account (for benchmarking)

## Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [Triton Documentation](https://triton-lang.org/)
- [LeetGPU Platform](https://leetgpu.com/)
