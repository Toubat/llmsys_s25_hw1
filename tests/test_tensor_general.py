import random
from typing import Callable, Dict, Iterable, List, Tuple
import numpy as np
import numba
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis.strategies import DataObject, data, integers, lists, permutations

import minitorch
from minitorch import MathTestVariable, Tensor, TensorBackend, grad_check

from .strategies import assert_close, small_floats
from .tensor_strategies import assert_close_tensor, shaped_tensors, tensors

one_arg, two_arg, red_arg = MathTestVariable._comp_testing()

shared: Dict[str, TensorBackend] = {}
from minitorch.cuda_kernel_ops import CudaKernelOps


if numba.cuda.is_available():
    backend_tests = [pytest.param("cuda")]
    matmul_tests = [pytest.param("cuda")]
    shared["cuda"] = minitorch.TensorBackend(CudaKernelOps)


@given(lists(small_floats, min_size=1))
@pytest.mark.parametrize("backend", backend_tests)
def test_create(backend: str, t1: List[float]) -> None:
    "Create different tensors."
    t2 = minitorch.tensor(t1, backend=shared[backend])
    for i in range(len(t1)):
        np.testing.assert_allclose(t1[i], t2[i], atol=1e-5, rtol=1e-5)


@given(data())
@settings(max_examples=100)
@pytest.mark.parametrize("fn", one_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_cuda_one_args(
    fn: Tuple[str, Callable[[float], float], Callable[[Tensor], Tensor]],
    backend: str,
    data: DataObject,
) -> None:
    "Run forward for all one arg functions above."
    t1 = data.draw(tensors(backend=shared[backend]))
    name, base_fn, tensor_fn = fn
    t2 = tensor_fn(t1)
    for ind in t2._tensor.indices():
      assert_close(t2[ind], base_fn(t1[ind]))


@given(data())
@settings(max_examples=100)
@pytest.mark.parametrize("fn", two_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_cuda_two_args(
    fn: Tuple[str, Callable[[float, float], float], Callable[[Tensor, Tensor], Tensor]],
    backend: str,
    data: DataObject,
) -> None:
    "Run forward for all two arg functions above."
    t1, t2 = data.draw(shaped_tensors(2, backend=shared[backend]))
    name, base_fn, tensor_fn = fn
    t3 = tensor_fn(t1, t2)
    for ind in t3._tensor.indices():
        assert_close(t3[ind], base_fn(t1[ind], t2[ind]))


@given(data())
@pytest.mark.parametrize("fn", one_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_cuda_one_derivative(
    fn: Tuple[str, Callable[[float], float], Callable[[Tensor], Tensor]],
    backend: str,
    data: DataObject,
) -> None:
    "Run backward for all one arg functions above."
    t1 = data.draw(tensors(backend=shared[backend]))
    name, _, tensor_fn = fn
    grad_check(tensor_fn, t1)


@given(data())
@settings(max_examples=50)
@pytest.mark.parametrize("fn", two_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_cuda_two_grad(
    fn: Tuple[str, Callable[[float, float], float], Callable[[Tensor, Tensor], Tensor]],
    backend: str,
    data: DataObject,
) -> None:
    "Run backward for all two arg functions above."
    t1, t2 = data.draw(shaped_tensors(2, backend=shared[backend]))
    name, _, tensor_fn = fn
    grad_check(tensor_fn, t1, t2)


@given(data())
@settings(max_examples=25, suppress_health_check=(HealthCheck.data_too_large,))
@pytest.mark.parametrize("fn", two_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_cuda_two_grad_broadcast(
    fn: Tuple[str, Callable[[float, float], float], Callable[[Tensor, Tensor], Tensor]],
    backend: str,
    data: DataObject,
) -> None:
    "Run backward for all two arg functions above with broadcast."
    t1, t2 = data.draw(shaped_tensors(2, backend=shared[backend]))
    name, base_fn, tensor_fn = fn

    grad_check(tensor_fn, t1, t2)

    # broadcast check
    grad_check(tensor_fn, t1.sum(0), t2)
    grad_check(tensor_fn, t1, t2.sum(0))


@given(data())
@settings(max_examples=100)
@pytest.mark.parametrize("fn", red_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_cuda_reduce(
    fn: Tuple[str, Callable[[Iterable[float]], float], Callable[[Tensor], Tensor]],
    backend: str,
    data: DataObject,
) -> None:
    "Run backward for all reduce functions above."
    t1 = data.draw(tensors(backend=shared[backend]))
    name, _, tensor_fn = fn
    grad_check(tensor_fn, t1)


@pytest.mark.parametrize("backend", backend_tests)
def test_cuda_reduce_sum_practice1(
    backend: str,
) -> None:
    x = [random.random() for i in range(32)]
    b = minitorch.tensor(x)
    s = b.sum()[0]
    b2 = minitorch.tensor(x, backend=shared[backend])
    out = b2.sum(0)
    assert_close(s, out[0])


@pytest.mark.parametrize("backend", backend_tests)
def test_cuda_reduce_sum_practice2(
    backend: str,
) -> None:
    x = [random.random() for i in range(500)]
    b = minitorch.tensor(x)
    s = b.sum()[0]
    b2 = minitorch.tensor(x, backend=shared[backend])
    out = b2.sum(0)
    assert_close(s, out[0])


reduce_2d_dims = [
    (1, 2),
    (1, 4),
    (2, 1),
    (4, 1),
    (2, 2),
    (4, 8),
    (8, 4),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
]


@pytest.mark.parametrize("m,n", reduce_2d_dims)
@pytest.mark.parametrize("backend", backend_tests)
def test_cuda_reduce_sum_practice3(
    m, n, backend: str  
) -> None:
    x = [[random.random() for _ in range(n)] for _ in range(m)]
    b = minitorch.tensor(x)
    s = b.sum(1)
    b2 = minitorch.tensor(x, backend=shared[backend])
    out = b2.sum(1)
    for i in range(m):
        assert_close(s[i, 0], out[i, 0])


reduce_3d_dims = [
    (1, 2, 3),
    (2, 2, 2),
    (4, 8, 16),
    (16, 16, 16),
    (32, 32, 32),
    (64, 64, 64),
    (33, 128, 48),
    (21, 127, 43),
]

@pytest.mark.parametrize("m,n,p", reduce_3d_dims)
@pytest.mark.parametrize("backend", backend_tests)
def test_cuda_reduce_sum_3d(m: int, n: int, p: int, backend: str) -> None:
    """
    Test a sum reduction along dimension 1 (i.e. reduce the n dimension) for a 3D tensor.
    Given an input of shape (m, n, p), the result should have shape (m, p).
    """
    # Create a 3D tensor (as nested lists) of shape (m, n, p) with random floats.
    x = [[[random.random() for _ in range(p)] for _ in range(n)] for _ in range(m)]
    
    # Create a tensor using the default backend.
    b = minitorch.tensor(x)
    # Perform reduction along dimension 1.
    s = b.sum(1)
    
    # Create a tensor using the specified CUDA backend.
    b2 = minitorch.tensor(x, backend=shared[backend])
    # Perform the same reduction on the CUDA tensor.
    out = b2.sum(1)
    
    np.testing.assert_allclose(
        s.to_numpy()[:, 0, :],
        out.to_numpy()[:, 0, :],
        rtol=1e-5,
        atol=1e-5
    )


reduce_4d_dims = [
    (1, 2, 3, 4),
    (2, 2, 2, 2),
    (4, 8, 16, 8),
    (16, 16, 16, 16),
    (32, 32, 32, 32),
    (21, 64, 43, 5),
]

@pytest.mark.parametrize("a,b,c,d", reduce_4d_dims)
@pytest.mark.parametrize("backend", backend_tests)
def test_cuda_reduce_sum_4d(a: int, b: int, c: int, d: int, backend: str) -> None:
    """
    Test a sum reduction along dimension 2 for a 4D tensor.
    Given an input of shape (a, b, c, d), reducing along dim 2 produces an output of shape (a, b, d).
    """
    # Create a 4D tensor (as nested lists) with random floats.
    x = [[[[random.random() for _ in range(d)]
           for _ in range(c)]
          for _ in range(b)]
         for _ in range(a)]
    
    # Create the tensor using the default backend.
    b_default = minitorch.tensor(x)
    expected = b_default.sum(2)  # reduce dimension 2
    
    # Create the tensor using the specified CUDA backend.
    b_cuda = minitorch.tensor(x, backend=shared[backend])
    result = b_cuda.sum(2)
    
    # Convert both results to NumPy arrays and compare.
    np.testing.assert_allclose(
        expected.to_numpy(),
        result.to_numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"4D reduction failed for shape ({a},{b},{c},{d}) on backend {backend}"
    )

# Example 5D shapes: (p, q, r, s, t)
reduce_5d_dims = [
    (1, 2, 3, 4, 5),
    (2, 2, 2, 2, 2),
    (3, 4, 5, 6, 7),
    (4, 4, 4, 4, 4),
    (8, 8, 8, 8, 8),
    (16, 4, 16, 8, 3),
]

@pytest.mark.parametrize("p,q,r,s,t", reduce_5d_dims)
@pytest.mark.parametrize("backend", backend_tests)
def test_cuda_reduce_sum_5d(p: int, q: int, r: int, s: int, t: int, backend: str) -> None:
    """
    Test a sum reduction along dimension 3 for a 5D tensor.
    Given an input of shape (p, q, r, s, t), reducing along dim 3 produces an output of shape (p, q, r, t).
    """
    # Create a 5D tensor (as nested lists) with random floats.
    x = [[[[[random.random() for _ in range(t)]
             for _ in range(s)]
            for _ in range(r)]
           for _ in range(q)]
         for _ in range(p)]
    
    # Create the tensor using the default backend.
    b_default = minitorch.tensor(x)
    expected = b_default.sum(3)  # reduce dimension 3
    
    # Create the tensor using the specified CUDA backend.
    b_cuda = minitorch.tensor(x, backend=shared[backend])
    result = b_cuda.sum(3)
    
    # Convert both results to NumPy arrays and compare.
    np.testing.assert_allclose(
        expected.to_numpy(),
        result.to_numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"5D reduction failed for shape ({p},{q},{r},{s},{t}) on backend {backend}"
    )
    
matmul_dims = [
    (2, 2, 2),
    (33, 33, 33),
    (16, 16, 16),
    (8, 8, 8),
    (1, 2, 3),
    (3, 4, 5),
    (5, 4, 3),
    (64, 64, 64),
    (72, 72, 72),
    (72, 73, 74),
    (74, 73, 72),
    (128, 128, 128),
    (256, 128, 512), 
    (512, 1024, 256),
    (2048, 1024, 512),
    (256, 1, 512),
    (1, 512, 256),
    (512, 256, 2),
]

batch_sizes = [
    1,
    2,
    4,
    8,
]

batch_matmul_dims = [
    (2, 2, 2),
    (33, 33, 33),
    (16, 16, 16),
    (8, 8, 8),
    (1, 2, 3),
    (3, 4, 5),
    (5, 4, 3),
    (64, 64, 64),
    (72, 72, 72),
    (72, 73, 74),
    (74, 73, 72),
    (128, 128, 128),
    (256, 128, 512), 
    (256, 1, 512),
    (1, 512, 256),
    (512, 256, 2),
]


@pytest.mark.parametrize("m,n,p", matmul_dims)
@pytest.mark.parametrize("backend", matmul_tests)
def test_cuda_matmul_numpy_eq(m, n, p, backend):
    _a = [[random.random() for j in range(n)] for i in range(m)]
    _b = [[random.random() for j in range(p)] for i in range(n)]
    c = minitorch.tensor(_a, backend=shared[backend]) @ minitorch.tensor(
        _b, backend=shared[backend])
    _c = np.array(_a) @ np.array(_b)
    np.testing.assert_allclose(
      c.to_numpy(), _c,
      atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("batch_size", batch_sizes)
@pytest.mark.parametrize("m,n,p", batch_matmul_dims)
@pytest.mark.parametrize("backend", matmul_tests)
def test_cuda_matmul_batch_numpy_eq(batch_size, m, n, p, backend):
    _a = [[random.random() for j in range(n)] for i in range(m)]
    _b = [[random.random() for j in range(p)] for i in range(n)]

    _a = [_a for _ in range(batch_size)]
    _b = [_b for _ in range(batch_size)]
    
    _c = np.array(_a) @ np.array(_b)
    c = minitorch.tensor(_a, backend=shared[backend]) @ minitorch.tensor(_b, backend=shared[backend])

    np.testing.assert_allclose(
      c.to_numpy(), _c,
      atol=1e-5, rtol=1e-5,
    )


@pytest.mark.parametrize("backend", matmul_tests)
def test_cuda_matmul_transpose(
    backend: str
) -> None:
    "non-square matrix multiplication"
    y1 = [[1.060827],[1.088836]]  # (2, 1)
    x1 = [
        [0.078633, -0.014161, -0.004430, -0.026322, 0.025413, 0.053841, -0.019814, -0.030969, 0.030927, -0.017623, 0.043520, 0.017807, 0.027121, -0.014492, -0.002200, 0.023008, -0.051772, -0.015968, -0.026916, -0.025462, 0.022397, 0.006308, 0.041307, -0.017634 ,-0.063493, -0.138394, -0.016439, -0.001476 ,0.039018, -0.028352, 0.360597, 0.021912, -0.014997 ,-0.060067, 0.015498 ,0.037198, -0.004001, -0.056578, 0.017538, -0.020992, 0.002291, -0.011406, 0.025177, 0.044035, -0.011892, -0.032769, 0.007307, -0.010899, -0.025381, -0.029157],
        [0.025094, 0.021550,-0.038259 ,-0.021197 ,0.030559, 0.059986, 0.015934, -0.028781, -0.033896, 0.048600, -0.042933, -0.001561, -0.002027, -0.024751, 0.034852, -0.006681, 0.018085, 0.005446, -0.026051, -0.047808, -0.012565, 0.063337, 0.059277, -0.013247 ,0.041941, -0.105160 ,-0.059339, 0.045373, 0.057688, -0.033190, 0.324249 ,0.006693 ,0.018170 ,-0.019590, 0.017880 ,0.051460, 0.018164 ,0.027205, -0.000471 ,0.000983 ,0.039456 ,0.049255, -0.001225, 0.038359, -0.015454, 0.025885, 0.005716, 0.017060 ,-0.009751 ,0.007455]
    ]  # (2, 50)

    def transpose(a: Tensor) -> Tensor:
        order = list(range(a.dims))
        order[-2], order[-1] = order[-1], order[-2]
        return a._new(a._tensor.permute(*order))

    x = minitorch.tensor(x1, backend=shared[backend])
    y = minitorch.tensor(y1, backend=shared[backend])
    
    x_t = transpose(x) # (50, 2)
    np.testing.assert_equal(x_t.shape, np.array(x1).T.shape)
    np.testing.assert_allclose(
        x_t.to_numpy(), np.array(x1, dtype=np.float32).T, 
        atol=1e-5, rtol=1e-5,
    )
    
    z = x_t @ y # (50, 2) x (2, 1) = (50, 1)
    np.testing.assert_allclose(
      z.to_numpy(), np.array(x1).T @ np.array(y1),
      atol=1e-5, rtol=1e-5,
    )


@given(data())
@settings(max_examples=100)
@pytest.mark.parametrize("backend", backend_tests)
def test_cuda_permute(backend: str, data: DataObject) -> None:
    "Check permutations for all backends."
    t1 = data.draw(tensors(backend=shared[backend]))
    permutation = data.draw(permutations(range(len(t1.shape))))

    def permute(a: Tensor) -> Tensor:
        return a.permute(*permutation)

    minitorch.grad_check(permute, t1)