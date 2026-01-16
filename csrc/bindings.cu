#include <torch/extension.h>
#include <cuda_runtime.h>

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

// Module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "GPU 100 Days - CUDA kernels";
    
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
}

