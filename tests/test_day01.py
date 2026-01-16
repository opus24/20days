"""
Tests for Day 01: Print Add (CUDA + Triton)
"""
import pytest
import torch
from conftest import ensure_cuda_device


def test_printAdd_triton():
    """Test Triton printAdd (identity operation)"""
    try:
        from gpu_20days import day01_printAdd
    except ImportError:
        pytest.skip("gpu_20days package not available")
    
    device = ensure_cuda_device()
    
    print("Testing Triton printAdd...")
    input_tensor = torch.randn(100, device=device, dtype=torch.float32)
    
    output = day01_printAdd(input_tensor)
    
    # Identity operation should return the same values
    torch.testing.assert_close(output, input_tensor)


def test_printAdd_cuda():
    """Test CUDA printAdd (just prints, no return value check)"""
    try:
        from gpu_20days.cuda_kernels import day01_printAdd
    except ImportError:
        pytest.skip("CUDA kernels not built")
    
    device = ensure_cuda_device()
    
    print("Testing CUDA printAdd...")
    # Just check it runs without error
    day01_printAdd(10)
