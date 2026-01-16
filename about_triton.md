# Triton Language API Reference

Triton에서 자주 사용하는 함수 및 기능 정리

## 기본 설정

```python
import triton
import triton.language as tl
```

## 메모리 로드/저장

### `tl.load(ptr, mask=None, other=0.0)`
메모리에서 데이터를 로드합니다.

```python
# 기본 사용
input = tl.load(input_ptr + idx, mask=mask)

# mask와 함께 사용 (범위 체크)
input = tl.load(input_ptr + idx, mask=idx < N, other=0.0)
```

### `tl.store(ptr, value, mask=None)`
메모리에 데이터를 저장합니다.

```python
# 기본 사용
tl.store(output_ptr + idx, value, mask=mask)

# mask와 함께 사용
tl.store(output_ptr + idx, result, mask=idx < N)
```

## 인덱싱 및 범위

### `tl.program_id(axis)`
현재 프로그램(블록)의 ID를 반환합니다.

```python
pid = tl.program_id(0)  # 첫 번째 차원의 블록 ID
pid_x = tl.program_id(0)
pid_y = tl.program_id(1)  # 두 번째 차원의 블록 ID
```

### `tl.num_programs(axis)`
특정 차원의 총 프로그램(블록) 개수를 반환합니다.

```python
num_blocks = tl.num_programs(0)
```

### `tl.arange(start, end)`
시작부터 끝까지의 범위를 생성합니다.

```python
idx = tl.arange(0, BLOCK_SIZE)  # [0, 1, 2, ..., BLOCK_SIZE-1]
```

## 수학 연산

### 기본 연산
```python
a + b  # 덧셈
a - b  # 뺄셈
a * b  # 곱셈
a / b  # 나눗셈
a % b  # 나머지
```

### 수학 함수

#### `tl.exp(x)`
지수 함수: e^x

```python
exp_val = tl.exp(x)
```

#### `tl.log(x)`
자연 로그: ln(x)

```python
log_val = tl.log(x)
```

#### `tl.sqrt(x)`
제곱근

```python
sqrt_val = tl.sqrt(x)
```

#### `tl.sin(x)`, `tl.cos(x)`
삼각함수

```python
sin_val = tl.sin(x)
cos_val = tl.cos(x)
```

## 조건부 연산

### `tl.where(condition, x, y)`
조건에 따라 값을 선택합니다.

```python
# condition이 True면 x, False면 y
result = tl.where(x > 0, x, 0.0)  # ReLU
result = tl.where(mask, value, 0.0)
```

### `tl.max(x, y)`, `tl.min(x, y)`
최대값/최소값

```python
max_val = tl.max(a, b)
min_val = tl.min(a, b)
```

## 리덕션 연산

### `tl.sum(x, axis=None)`
합계 계산

```python
# 전체 합계
total = tl.sum(array)

# 특정 축에 대한 합계
sum_along_axis = tl.sum(matrix, axis=0)
```

### `tl.max(x, axis=None)`, `tl.min(x, axis=None)`
최대값/최소값 계산

```python
max_val = tl.max(array)
min_val = tl.min(array)
```

## 원자적 연산

### `tl.atomic_add(ptr, value)`
원자적으로 값을 더합니다.

```python
# 카운터 업데이트
tl.atomic_add(counter_ptr, count)
```

### `tl.atomic_max(ptr, value)`, `tl.atomic_min(ptr, value)`
원자적으로 최대값/최소값 업데이트

```python
tl.atomic_max(max_ptr, value)
tl.atomic_min(min_ptr, value)
```

## 유틸리티 함수

### `tl.zeros(shape, dtype)`
0으로 초기화된 배열 생성

```python
# 1D 배열
zeros = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

# 2D 배열
matrix = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)
```

### `tl.full(shape, value, dtype)`
특정 값으로 채워진 배열 생성

```python
ones = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32)
```

## 데이터 타입

### 기본 타입
```python
tl.int8, tl.int16, tl.int32, tl.int64
tl.float16, tl.float32, tl.float64
```

### 타입 변환
```python
# 명시적 캐스팅
float_val = tl.float32(int_val)
int_val = tl.int32(float_val)
```

## 커널 작성 패턴

### 기본 패턴
```python
@triton.jit
def kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    # 인덱스 계산
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 마스크 생성
    mask = idx < N
    
    # 데이터 로드
    input = tl.load(input_ptr + idx, mask=mask)
    
    # 연산 수행
    output = input * 2
    
    # 결과 저장
    tl.store(output_ptr + idx, output, mask=mask)
```

### 2D 인덱싱 패턴
```python
@triton.jit
def matrix_kernel(A_ptr, B_ptr, C_ptr, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_idx = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_idx = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = m_idx < M
    mask_n = n_idx < N
    
    # 2D 로드/저장
    a = tl.load(A_ptr + m_idx[:, None] * N + n_idx[None, :], 
                mask=mask_m[:, None] & mask_n[None, :])
```

### 리덕션 패턴
```python
@triton.jit
def reduction_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    
    input = tl.load(input_ptr + idx, mask=mask, other=0.0)
    
    # 블록 내 합계 계산
    local_sum = tl.sum(input)
    
    # 원자적 연산으로 전역 합계에 추가
    tl.atomic_add(output_ptr, local_sum)
```

## 커널 실행

### Grid 함수 정의
```python
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 256
    
    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    
    kernel[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)
```

### 2D Grid
```python
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int):
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    
    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_SIZE_M"]),
            triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        )
    
    kernel[grid](A, B, C, M, N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N)
```

## Auto-tuning (자동 튜닝)

Triton의 `autotune` 기능을 사용하면 커널의 성능을 자동으로 최적화할 수 있습니다.

### 기본 사용법

```python
import triton
from triton import autotune, Config

@triton.autotune(
    configs=[
        Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4),
        Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=4),
        Config({'BLOCK_SIZE': 1024}, num_stages=2, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def tuned_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    
    input = tl.load(input_ptr + idx, mask=mask)
    output = input * 2
    tl.store(output_ptr + idx, output, mask=mask)
```

### Config 클래스

`Config`는 커널의 설정을 정의합니다.

```python
Config(
    {'BLOCK_SIZE': 256, 'BLOCK_SIZE_M': 16},  # 커널 파라미터
    num_stages=3,      # 파이프라인 스테이지 수 (1-5)
    num_warps=4,       # 워프 수 (1, 2, 4, 8, 16, 32)
)
```

**파라미터 설명:**
- **커널 파라미터**: 딕셔너리로 전달되는 `tl.constexpr` 값들
- **num_stages**: 파이프라인 스테이지 수 (1-5, 기본값 3)
  - 높을수록 더 많은 레지스터 사용, 더 나은 성능 가능
- **num_warps**: 워프 수 (1, 2, 4, 8, 16, 32)
  - 워프는 32개의 스레드 그룹
  - 더 많은 워프 = 더 많은 병렬성

### key 파라미터

`key`는 어떤 입력 크기에 따라 다른 최적화를 선택할지 결정합니다.

```python
@triton.autotune(
    configs=[...],
    key=['N', 'M'],  # N과 M의 값에 따라 다른 최적화 선택
)
```

### 실제 예제: 행렬 곱셈 최적화

```python
@triton.autotune(
    configs=[
        Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
        Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
        Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, 
                  BLOCK_SIZE_M: tl.constexpr, 
                  BLOCK_SIZE_N: tl.constexpr, 
                  BLOCK_SIZE_K: tl.constexpr):
    # ... 커널 구현
    pass
```

### 여러 파라미터 조합

```python
@triton.autotune(
    configs=[
        Config({'BLOCK_SIZE': 128, 'USE_SHARED': True}, num_stages=2, num_warps=4),
        Config({'BLOCK_SIZE': 256, 'USE_SHARED': True}, num_stages=2, num_warps=4),
        Config({'BLOCK_SIZE': 128, 'USE_SHARED': False}, num_stages=3, num_warps=8),
        Config({'BLOCK_SIZE': 256, 'USE_SHARED': False}, num_stages=3, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def complex_kernel(input_ptr, output_ptr, N, 
                  BLOCK_SIZE: tl.constexpr, 
                  USE_SHARED: tl.constexpr):
    # USE_SHARED에 따라 다른 로직 구현 가능
    if USE_SHARED:
        # Shared memory 사용 로직
        pass
    else:
        # 일반 메모리 사용 로직
        pass
```

### Autotune 실행

Autotune은 첫 실행 시 자동으로 최적의 설정을 찾습니다.

```python
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    # 첫 실행: autotune이 최적 설정을 찾음
    # 이후 실행: 캐시된 최적 설정 사용
    tuned_kernel[(triton.cdiv(N, 256),)](input, output, N)
```

### Autotune 캐시

최적화된 설정은 자동으로 캐시되어 다음 실행 시 재사용됩니다.

```python
# 첫 실행: 모든 config 테스트 (느림)
solve(input1, output1, 1024)

# 이후 실행: 캐시된 최적 설정 사용 (빠름)
solve(input2, output2, 1024)
solve(input3, output3, 2048)  # 다른 크기는 다시 튜닝
```

### Autotune 팁

1. **적절한 Config 범위**: 너무 많은 config는 튜닝 시간이 오래 걸립니다.
2. **key 선택**: 입력 크기에 따라 성능이 달라지는 경우에만 `key`를 사용하세요.
3. **num_stages와 num_warps**: 
   - 작은 블록: 낮은 num_stages, 적은 num_warps
   - 큰 블록: 높은 num_stages, 많은 num_warps
4. **첫 실행 시간**: Autotune은 첫 실행 시 느릴 수 있지만, 이후 실행은 빠릅니다.

### Autotune 비활성화

특정 설정을 강제로 사용하려면:

```python
# Autotune 없이 직접 설정
@triton.jit
def simple_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    # ... 구현
    pass

def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    # 직접 BLOCK_SIZE 지정
    simple_kernel[(triton.cdiv(N, 256),)](input, output, N, BLOCK_SIZE=256)
```

## 유용한 팁

1. **마스크 사용**: 항상 범위 체크를 위해 `mask`를 사용하세요.
2. **타입 명시**: `dtype`을 명시적으로 지정하면 성능이 향상될 수 있습니다.
3. **블록 크기**: `BLOCK_SIZE`는 보통 128, 256, 512 등 2의 거듭제곱을 사용합니다.
4. **메모리 접근**: 연속된 메모리 접근이 성능에 유리합니다.
5. **원자적 연산**: 여러 스레드가 같은 메모리에 쓰는 경우 `atomic_*` 함수를 사용하세요.
6. **Autotune 활용**: 성능이 중요한 커널은 `autotune`을 사용하여 자동 최적화하세요.

## 참고 자료

- [Triton Documentation](https://triton-lang.org/)
- [Triton Tutorial](https://triton-lang.org/python-api/triton.language.html)

