cmake_minimum_required(VERSION 3.18)
project(cuda_ops LANGUAGES CXX CUDA)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find packages
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Find PyTorch - try to find it in the Python environment
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE TORCH_FOUND_RESULT
)

if(TORCH_FOUND_RESULT EQUAL 0 AND TORCH_PREFIX)
    message(STATUS "Found PyTorch at: ${TORCH_PREFIX}")
    # Try to find TorchConfig.cmake in common locations
    set(TORCH_CMAKE_DIR "${TORCH_PREFIX}/share/cmake/Torch")
    if(EXISTS "${TORCH_CMAKE_DIR}/TorchConfig.cmake")
        set(Torch_DIR "${TORCH_CMAKE_DIR}")
        message(STATUS "Found TorchConfig.cmake at: ${Torch_DIR}")
    else()
        # Fallback: add to CMAKE_PREFIX_PATH
        list(APPEND CMAKE_PREFIX_PATH "${TORCH_PREFIX}")
        message(STATUS "Added PyTorch to CMAKE_PREFIX_PATH: ${TORCH_PREFIX}")
    endif()
else()
    message(WARNING "Could not find PyTorch via Python. Make sure PyTorch is installed in the Python environment.")
endif()

find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

# CUDA source files
set(CUDA_SOURCES
    bindings.cu
    day01_printAdd.cu
    day02_function.cu
    day03_vectorAdd.cu
    day04_matmul.cu
    day05_matrixAdd.cu
    day06_countElement.cu
    day07_matrixCopy.cu
    day08_relu.cu
    day09_silu.cu
    day10_conv1d.cu
    day11_softmax.cu
    day12_layernorm.cu
    day13_rmsnorm.cu
    day14_fused_softmax.cu
    day15_fused_attention.cu
    day16_group_gemm.cu
    day17_persistent_matmul.cu
    day18_block_scaled_matmul.cu
    day19_rope.cu
    day20_conv2d.cu
)

# Create the shared library
add_library(cuda_ops SHARED ${CUDA_SOURCES})

# Include directories
target_include_directories(cuda_ops PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(cuda_ops PRIVATE
    ${TORCH_LIBRARIES}
    ${Python_LIBRARIES}
    CUDA::cudart
)

# CUDA compilation flags
target_compile_options(cuda_ops PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        --expt-relaxed-constexpr
        -lineinfo
    >
)

# Set CUDA architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86")
endif()

# Generate compile_commands.json for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Output directory
set_target_properties(cuda_ops PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src
    PREFIX ""
)

# Python module suffix
if(WIN32)
    set_target_properties(cuda_ops PROPERTIES SUFFIX ".pyd")
else()
    set_target_properties(cuda_ops PROPERTIES SUFFIX ".so")
endif()
