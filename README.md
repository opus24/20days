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

- CUDA Toolkit (11.0 이상)
- Python 3.8 이상
- [uv](https://github.com/astral-sh/uv) (패키지 관리 및 가상환경 관리)
- PyTorch (1.12.0 이상)
- CMake (3.18 이상, CMake 빌드 사용 시)
- LeetGPU account (for benchmarking)

## uv 가상환경 설정

이 프로젝트는 [uv](https://github.com/astral-sh/uv)를 사용하여 의존성과 가상환경을 관리합니다.

### uv 설치

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# 또는 pip를 통해 설치
pip install uv
```

### 프로젝트 설정

```bash
# 저장소 클론
git clone <repository-url>
cd 20days

# uv를 사용하여 의존성 설치 및 가상환경 생성
# 이 명령은 자동으로 가상환경을 생성하고 모든 의존성을 설치합니다
uv sync

# 개발 의존성 포함하여 설치 (pytest 등)
uv sync --extra dev
```

### uv 명령어

```bash
# 자동으로 가상환경 사용
uv run python script.py
uv run pytest tests/

# 의존성 관리
uv sync --upgrade          # 의존성 업데이트
uv add package-name        # 패키지 추가
uv add --dev package-name  # 개발 의존성 추가

# 수동 활성화 (필요시)
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

## CUDA 코드 컴파일 및 실행

### 빌드

`uv sync` 명령으로 의존성 설치와 CUDA 확장 모듈 빌드가 자동으로 수행됩니다:

```bash
# 기본 빌드
uv sync

# 개발 의존성 포함
uv sync --extra dev
```

빌드가 완료되면 Python에서 사용할 수 있습니다:

```python
from gpu_20days.cuda_kernels import day03_vectorAdd
import torch

a = torch.randn(1024, device='cuda')
b = torch.randn(1024, device='cuda')
c = day03_vectorAdd(a, b)
```

### 대안 빌드 방법

**수동 빌드 (setuptools):**
```bash
source .venv/bin/activate  # 또는 .venv\Scripts\activate (Windows)
python setup.py build_ext --inplace
```

**CMake 빌드:**
```bash
# 가상환경 활성화 필수 (PyTorch 경로를 찾기 위해)
source .venv/bin/activate  # 또는 .venv\Scripts\activate (Windows)

mkdir -p build && cd build
cmake ..
cmake --build . --config Release

# 빌드된 라이브러리는 src/ 디렉토리에 생성됩니다
```

> **참고**: CMake 빌드는 가상환경이 활성화된 상태에서 실행해야 PyTorch를 자동으로 찾을 수 있습니다. CMakeLists.txt가 Python 환경에서 PyTorch 경로를 자동으로 감지합니다.

### 테스트 실행

```bash
# uv run 사용 (권장)
uv run pytest tests/
uv run pytest tests/test_day03.py -v

# 또는 가상환경 활성화 후
pytest tests/
```

### 문제 해결

- **CUDA를 찾을 수 없는 경우**
  ```bash
  export CUDA_HOME=/usr/local/cuda
  ```

- **PyTorch CUDA 버전 불일치**
  ```python
  import torch
  print(torch.version.cuda)
  ```

- **빌드 오류**
  ```bash
  uv sync --verbose
  # 또는
  python setup.py build_ext --inplace --verbose
  ```

- **uv 설치 확인**
  ```bash
  uv --version
  ```

## Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [Triton Documentation](https://triton-lang.org/)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [LeetGPU Platform](https://leetgpu.com/)
