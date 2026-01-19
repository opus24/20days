#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel declarations
extern "C" void day01_printAdd(int N);
extern "C" void day02_function(int *h_A, int *h_B, int N);
extern "C" void day03_vectorAdd(const float* A, const float* B, float* C, int N);
extern "C" void day04_matmul(const float* A, const float* B, float* C, int M, int N, int K);
extern "C" void day05_matrixAdd(const float* A, const float* B, float* C, int N);
extern "C" void day06_countElement(const int* input, int* output, int N, int K);
extern "C" void day07_matrixCopy(const float* A, float* B, int N);
extern "C" void day08_relu(const float* input, float* output, int N);
extern "C" void day09_silu(const float* input, float* output, int N);
extern "C" void day10_conv1d(const float* input, const float* kernel, float* output, int input_size, int kernel_size);
extern "C" void day11_softmax(const float* input, float* output, int feature_size);
extern "C" void day12_layernorm(const float* input, float* output, const float* gamma, const float* beta, int batch_size, int feature_size, float eps);
extern "C" void day13_rmsnorm(const float* input, float* output, const float* weight, int batch_size, int feature_size, float eps);
extern "C" void day14_fused_softmax(const float* input, float* output, const float* mask, int batch_size, int seq_len, int feature_size, float scale);
extern "C" void day15_fused_attention(const float* Q, const float* K, const float* V, float* output, const float* mask, int batch_size, int num_heads, int seq_len, int head_dim, float scale);
extern "C" void day16_group_gemm(const float* A, const float* B, float* C, int num_groups, int M, int N, int K);
extern "C" void day17_persistent_matmul(const float* A, const float* B, float* C, int M, int N, int K);
extern "C" void day18_block_scaled_matmul(const float* A, const float* B, float* C, int M, int N, int K, float scale);
extern "C" void day19_rope(const float* query, const float* key, float* rotated_query, float* rotated_key, const float* cos_cache, const float* sin_cache, int batch_size, int num_heads, int seq_len, int head_dim);
extern "C" void day20_conv2d(const float* input, const float* kernel, float* output, int batch_size, int in_channels, int out_channels, int input_h, int input_w, int kernel_h, int kernel_w, int output_h, int output_w, int pad_h, int pad_w, int stride_h, int stride_w);

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err)); \
    } while(0)

// Day 01: Print global indices
void day01_printAdd_wrapper(int N) {
    day01_printAdd(N);
}

// Day 02: Device function example
torch::Tensor day02_function_wrapper(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kInt32, "Input must be int32");

    int N = input.numel();
    torch::Tensor output = torch::zeros_like(input);

    day02_function(input.data_ptr<int>(), output.data_ptr<int>(), N);

    return output;
}

// Day 03: Vector addition
torch::Tensor day03_vectorAdd_wrapper(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "Inputs must be float32");
    TORCH_CHECK(A.numel() == B.numel(), "Inputs must have the same number of elements");

    int N = A.numel();
    torch::Tensor C = torch::zeros_like(A);

    day03_vectorAdd(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}

// Day 04: Matrix multiplication
torch::Tensor day04_matmul_wrapper(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "Inputs must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions must match for multiplication");

    int M = A.size(0);
    int N = A.size(1);
    int K = B.size(1);

    torch::Tensor C = torch::zeros({M, K}, A.options());

    day04_matmul(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}

// Day 05: Matrix addition
torch::Tensor day05_matrixAdd_wrapper(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "Inputs must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Inputs must be square matrices");
    TORCH_CHECK(A.size(0) == B.size(0), "Matrices must have the same size");

    int N = A.size(0);
    torch::Tensor C = torch::zeros_like(A);

    day05_matrixAdd(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}

// Day 06: Count array elements
int day06_countElement_wrapper(torch::Tensor input, int K) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kInt32, "Input must be int32");

    int N = input.numel();
    torch::Tensor output = torch::zeros({1}, input.options());

    day06_countElement(input.data_ptr<int>(), output.data_ptr<int>(), N, K);

    return output.item<int>();
}

// Day 07: Matrix copy
torch::Tensor day07_matrixCopy_wrapper(torch::Tensor A) {
    TORCH_CHECK(A.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(A.dim() == 2 && A.size(0) == A.size(1), "Input must be a square matrix");

    int N = A.size(0);
    torch::Tensor B = torch::zeros_like(A);

    day07_matrixCopy(A.data_ptr<float>(), B.data_ptr<float>(), N);

    return B;
}

// Day 08: ReLU activation
torch::Tensor day08_relu_wrapper(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    int N = input.numel();
    torch::Tensor output = torch::zeros_like(input);

    day08_relu(input.data_ptr<float>(), output.data_ptr<float>(), N);

    return output;
}

// Day 09: SiLU activation
torch::Tensor day09_silu_wrapper(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    int N = input.numel();
    torch::Tensor output = torch::zeros_like(input);

    day09_silu(input.data_ptr<float>(), output.data_ptr<float>(), N);

    return output;
}

// Day 10: 1D Convolution
torch::Tensor day10_conv1d_wrapper(torch::Tensor input, torch::Tensor kernel) {
    TORCH_CHECK(input.is_cuda() && kernel.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(input.dtype() == torch::kFloat32 && kernel.dtype() == torch::kFloat32, "Inputs must be float32");
    TORCH_CHECK(input.dim() == 1 && kernel.dim() == 1, "Inputs must be 1D tensors");

    int input_size = input.numel();
    int kernel_size = kernel.numel();
    int output_size = input_size - kernel_size + 1;

    TORCH_CHECK(output_size > 0, "Kernel size must be <= input size");

    torch::Tensor output = torch::zeros({output_size}, input.options());

    day10_conv1d(input.data_ptr<float>(), kernel.data_ptr<float>(), output.data_ptr<float>(), input_size, kernel_size);

    return output;
}

// Day 11: Softmax activation
torch::Tensor day11_softmax_wrapper(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 1, "Input must be 1D tensor");

    int feature_size = input.size(0);
    torch::Tensor output = torch::zeros_like(input);

    day11_softmax(input.data_ptr<float>(), output.data_ptr<float>(), feature_size);

    return output;
}

// Day 12: Layer normalization
torch::Tensor day12_layernorm_wrapper(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor (batch_size, feature_size)");

    int batch_size = input.size(0);
    int feature_size = input.size(1);

    // Default gamma and beta if not provided
    if (gamma.numel() == 0) {
        gamma = torch::ones({feature_size}, input.options());
    }
    if (beta.numel() == 0) {
        beta = torch::zeros({feature_size}, input.options());
    }

    torch::Tensor output = torch::zeros_like(input);

    day12_layernorm(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        batch_size,
        feature_size,
        eps
    );

    return output;
}

// Day 13: RMS normalization
torch::Tensor day13_rmsnorm_wrapper(torch::Tensor input, torch::Tensor weight, float eps) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor (batch_size, feature_size)");

    int batch_size = input.size(0);
    int feature_size = input.size(1);

    // Default weight if not provided
    if (weight.numel() == 0) {
        weight = torch::ones({feature_size}, input.options());
    }

    torch::Tensor output = torch::zeros_like(input);

    day13_rmsnorm(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        batch_size,
        feature_size,
        eps
    );

    return output;
}

// Day 14: Fused softmax
torch::Tensor day14_fused_softmax_wrapper(torch::Tensor input, torch::Tensor mask, float scale) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D tensor (batch_size, seq_len, feature_size)");

    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int feature_size = input.size(2);

    // Default mask if not provided (empty tensor)
    const float* mask_ptr = nullptr;
    if (mask.numel() > 0) {
        TORCH_CHECK(mask.is_cuda() && mask.dtype() == torch::kFloat32, "Mask must be CUDA float32 tensor");
        mask_ptr = mask.data_ptr<float>();
    }

    torch::Tensor output = torch::zeros_like(input);

    day14_fused_softmax(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        mask_ptr,
        batch_size,
        seq_len,
        feature_size,
        scale
    );

    return output;
}

// Day 15: Fused attention
torch::Tensor day15_fused_attention_wrapper(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor mask, float scale) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(Q.dtype() == torch::kFloat32 && K.dtype() == torch::kFloat32 && V.dtype() == torch::kFloat32, "Inputs must be float32");
    TORCH_CHECK(Q.dim() == 4, "Q, K, V must be 4D tensors (batch_size, num_heads, seq_len, head_dim)");

    int batch_size = Q.size(0);
    int num_heads = Q.size(1);
    int seq_len = Q.size(2);
    int head_dim = Q.size(3);

    // Default scale if not provided
    if (scale <= 0.0f) {
        scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    }

    // Default mask if not provided
    const float* mask_ptr = nullptr;
    if (mask.numel() > 0) {
        TORCH_CHECK(mask.is_cuda() && mask.dtype() == torch::kFloat32, "Mask must be CUDA float32 tensor");
        mask_ptr = mask.data_ptr<float>();
    }

    torch::Tensor output = torch::zeros_like(Q);

    day15_fused_attention(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        mask_ptr,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        scale
    );

    return output;
}

// Day 16: Group GEMM
torch::Tensor day16_group_gemm_wrapper(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "Inputs must be float32");
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3, "Inputs must be 3D tensors (num_groups, M, K) and (num_groups, K, N)");
    TORCH_CHECK(A.size(0) == B.size(0), "Number of groups must match");
    TORCH_CHECK(A.size(2) == B.size(1), "K dimension must match");

    int num_groups = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    torch::Tensor C = torch::zeros({num_groups, M, N}, A.options());

    day16_group_gemm(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        num_groups, M, N, K
    );

    return C;
}

// Day 17: Persistent Matmul
torch::Tensor day17_persistent_matmul_wrapper(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "Inputs must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions must match for multiplication");

    int M = A.size(0);
    int N = A.size(1);
    int K = B.size(1);

    torch::Tensor C = torch::zeros({M, K}, A.options());

    day17_persistent_matmul(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    return C;
}

// Day 18: Block Scaled Matrix Multiplication
torch::Tensor day18_block_scaled_matmul_wrapper(torch::Tensor A, torch::Tensor B, float scale) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "Inputs must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions must match for multiplication");

    int M = A.size(0);
    int N = A.size(1);
    int K = B.size(1);

    torch::Tensor C = torch::zeros({M, K}, A.options());

    day18_block_scaled_matmul(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N, scale
    );

    return C;
}

// Day 19: RoPE
std::tuple<torch::Tensor, torch::Tensor> day19_rope_wrapper(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor cos_cache,
    torch::Tensor sin_cache
) {
    TORCH_CHECK(query.is_cuda() && key.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(query.dtype() == torch::kFloat32 && key.dtype() == torch::kFloat32, "Inputs must be float32");
    TORCH_CHECK(query.dim() == 4 && key.dim() == 4, "Query and key must be 4D tensors (batch_size, num_heads, seq_len, head_dim)");

    int batch_size = query.size(0);
    int num_heads = query.size(1);
    int seq_len = query.size(2);
    int head_dim = query.size(3);

    torch::Tensor rotated_query = torch::zeros_like(query);
    torch::Tensor rotated_key = torch::zeros_like(key);

    day19_rope(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        rotated_query.data_ptr<float>(),
        rotated_key.data_ptr<float>(),
        cos_cache.data_ptr<float>(),
        sin_cache.data_ptr<float>(),
        batch_size, num_heads, seq_len, head_dim
    );

    return std::make_tuple(rotated_query, rotated_key);
}

// Day 20: 2D Convolution
torch::Tensor day20_conv2d_wrapper(
    torch::Tensor input,
    torch::Tensor kernel,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w
) {
    TORCH_CHECK(input.is_cuda() && kernel.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(input.dtype() == torch::kFloat32 && kernel.dtype() == torch::kFloat32, "Inputs must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D tensor (batch_size, in_channels, height, width)");
    TORCH_CHECK(kernel.dim() == 4, "Kernel must be 4D tensor (out_channels, in_channels, kernel_h, kernel_w)");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_h = input.size(2);
    int input_w = input.size(3);

    int out_channels = kernel.size(0);
    int kernel_h = kernel.size(2);
    int kernel_w = kernel.size(3);

    int output_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int output_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;

    TORCH_CHECK(output_h > 0 && output_w > 0, "Output dimensions must be positive");

    torch::Tensor output = torch::zeros({batch_size, out_channels, output_h, output_w}, input.options());

    day20_conv2d(
        input.data_ptr<float>(),
        kernel.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_h, input_w, kernel_h, kernel_w,
        output_h, output_w,
        pad_h, pad_w, stride_h, stride_w
    );

    return output;
}

// Module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "GPU 20 Days - CUDA kernels";

    m.def("day01_printAdd", &day01_printAdd_wrapper, "Day 01: Print global indices");
    m.def("day02_function", &day02_function_wrapper, "Day 02: Device function example");
    m.def("day03_vectorAdd", &day03_vectorAdd_wrapper, "Day 03: Vector addition");
    m.def("day04_matmul", &day04_matmul_wrapper, "Day 04: Matrix multiplication");
    m.def("day05_matrixAdd", &day05_matrixAdd_wrapper, "Day 05: Matrix addition");
    m.def("day06_countElement", &day06_countElement_wrapper, "Day 06: Count array elements");
    m.def("day07_matrixCopy", &day07_matrixCopy_wrapper, "Day 07: Matrix copy");
    m.def("day08_relu", &day08_relu_wrapper, "Day 08: ReLU activation");
    m.def("day09_silu", &day09_silu_wrapper, "Day 09: SiLU activation");
    m.def("day10_conv1d", &day10_conv1d_wrapper, "Day 10: 1D Convolution");
    m.def("day11_softmax", &day11_softmax_wrapper, "Day 11: Softmax activation");
    m.def("day12_layernorm", &day12_layernorm_wrapper, "Day 12: Layer normalization");
    m.def("day13_rmsnorm", &day13_rmsnorm_wrapper, "Day 13: RMS normalization");
    m.def("day14_fused_softmax", &day14_fused_softmax_wrapper, "Day 14: Fused softmax");
    m.def("day15_fused_attention", &day15_fused_attention_wrapper, "Day 15: Fused attention");
    m.def("day16_group_gemm", &day16_group_gemm_wrapper, "Day 16: Group GEMM");
    m.def("day17_persistent_matmul", &day17_persistent_matmul_wrapper, "Day 17: Persistent Matmul");
    m.def("day18_block_scaled_matmul", &day18_block_scaled_matmul_wrapper, "Day 18: Block Scaled Matmul");
    m.def("day19_rope", &day19_rope_wrapper, "Day 19: RoPE");
    m.def("day20_conv2d", &day20_conv2d_wrapper, "Day 20: 2D Convolution");
}
