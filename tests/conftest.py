"""
Pytest configuration and fixtures for GPU kernel tests
"""
import pytest
import torch
import numpy as np
import time
from typing import Callable


def ensure_cuda_device():
    """CUDA 디바이스 확인 및 반환"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.fixture(scope="session")
def cuda_device():
    """Fixture to check CUDA device availability"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests"""
    np.random.seed(42)
    torch.manual_seed(42)
    return 42


def assert_allclose(actual, expected, rtol=1e-5, atol=1e-8):
    """Helper function to assert arrays are close"""
    if isinstance(actual, torch.Tensor):
        actual = actual.cpu().numpy()
    if isinstance(expected, torch.Tensor):
        expected = expected.cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def compare_kernel_with_pytorch(
    kernel_func: Callable,
    pytorch_func: Callable,
    *args,
    rtol=1e-5,
    atol=1e-8
):
    """
    커널 결과와 PyTorch 결과 비교
    
    Args:
        kernel_func: 테스트할 커널 함수 (CUDA 또는 Triton)
        pytorch_func: PyTorch reference 함수
        *args: 함수에 전달할 인자들
        rtol: 상대 허용 오차
        atol: 절대 허용 오차
    """
    # PyTorch reference 결과 계산
    pytorch_result = pytorch_func(*args)
    
    # 커널 함수 실행
    kernel_result = kernel_func(*args)
    
    # 결과 비교
    assert_allclose(kernel_result, pytorch_result, rtol=rtol, atol=atol)


def benchmark_kernel_vs_pytorch(
    kernel_func: Callable,
    pytorch_func: Callable,
    *args,
    warmup=10,
    repeat=100
):
    """
    커널과 PyTorch 성능 비교
    
    Args:
        kernel_func: 테스트할 커널 함수
        pytorch_func: PyTorch reference 함수
        *args: 함수에 전달할 인자들
        warmup: 워밍업 반복 횟수
        repeat: 측정 반복 횟수
    """
    # 워밍업
    for _ in range(warmup):
        _ = pytorch_func(*args)
        _ = kernel_func(*args)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # PyTorch 벤치마크
    start = time.perf_counter()
    for _ in range(repeat):
        _ = pytorch_func(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / repeat
    
    # 커널 벤치마크
    start = time.perf_counter()
    for _ in range(repeat):
        _ = kernel_func(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    kernel_time = (time.perf_counter() - start) / repeat
    
    speedup = pytorch_time / kernel_time if kernel_time > 0 else 0.0
    
    print(f"\nBenchmark results:")
    print(f"  PyTorch: {pytorch_time*1000:.3f} ms")
    print(f"  Kernel:  {kernel_time*1000:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")
