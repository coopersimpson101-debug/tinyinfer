"""Matmul kernels (all weight formats, batched, multithreaded) vs NumPy."""
import ctypes as C

import numpy as np
import pytest
from conftest import fptr
from tinyinfer_ref import quant


def c_matmul(lib, t, wbytes, x, rows, cols):
    x = np.ascontiguousarray(x, dtype=np.float32)
    n_tok = x.shape[0]
    y = np.zeros((n_tok, rows), dtype=np.float32)
    buf = C.create_string_buffer(wbytes, len(wbytes))
    lib.ti_matmul(t, buf, fptr(x), fptr(y), rows, cols, n_tok)
    return y


@pytest.mark.parametrize("t", [quant.F32, quant.F16, quant.BF16, quant.Q8_0, quant.Q4_0])
@pytest.mark.parametrize("rows,cols,n_tok", [(1, 32, 1), (67, 256, 1), (130, 512, 7), (33, 96, 64)])
def test_matmul_vs_numpy(lib, t, rows, cols, n_tok):
    rng = np.random.default_rng(rows * 1000 + cols + n_tok)
    w = rng.standard_normal((rows, cols)).astype(np.float32)
    x = rng.standard_normal((n_tok, cols)).astype(np.float32)
    wb = quant.quantize(w, t)
    wd = quant.dequantize(wb, t, w.size).reshape(rows, cols).astype(np.float64)
    xr = quant.fake_quant_q8_0(x) if t in (quant.Q8_0, quant.Q4_0) else x
    ref = xr.astype(np.float64) @ wd.T
    got = c_matmul(lib, t, wb, x, rows, cols)
    scale = np.sqrt(cols) * np.abs(wd).max() * np.abs(xr).max()
    assert np.abs(got - ref).max() <= 1e-6 * scale + 1e-6


@pytest.mark.parametrize("t", [quant.F32, quant.Q8_0, quant.Q4_0])
def test_matmul_quantised_close_to_float(lib, t):
    """End-to-end accuracy of a quantised matmul against the exact float product."""
    rng = np.random.default_rng(7)
    rows, cols = 256, 1024
    w = rng.standard_normal((rows, cols)).astype(np.float32)
    x = rng.standard_normal((3, cols)).astype(np.float32)
    got = c_matmul(lib, t, quant.quantize(w, t), x, rows, cols)
    ref = x.astype(np.float64) @ w.T.astype(np.float64)
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel < {quant.F32: 1e-6, quant.Q8_0: 0.01, quant.Q4_0: 0.15}[t], rel


@pytest.mark.parametrize("t", [quant.F32, quant.F16, quant.Q8_0, quant.Q4_0])
def test_threads_bitwise_identical(lib, t):
    rng = np.random.default_rng(3)
    rows, cols = 300, 512
    w = rng.standard_normal((rows, cols)).astype(np.float32)
    x = rng.standard_normal((4, cols)).astype(np.float32)
    wb = quant.quantize(w, t)
    before = lib.ti_get_threads()
    try:
        lib.ti_set_threads(1)
        y1 = c_matmul(lib, t, wb, x, rows, cols)
        lib.ti_set_threads(4)
        y4 = c_matmul(lib, t, wb, x, rows, cols)
    finally:
        lib.ti_set_threads(before)
    np.testing.assert_array_equal(y1, y4)


def test_thread_pool_stress(lib):
    """Many tiny back-to-back jobs: exercises the pool's publish/handshake path."""
    rng = np.random.default_rng(5)
    w = rng.standard_normal((8, 32)).astype(np.float32)
    wb = quant.quantize(w, quant.F32)
    x = rng.standard_normal((1, 32)).astype(np.float32)
    ref = c_matmul(lib, quant.F32, wb, x, 8, 32)
    for _ in range(3000):
        np.testing.assert_array_equal(c_matmul(lib, quant.F32, wb, x, 8, 32), ref)


def test_simd_path_reported(lib):
    assert lib.ti_simd_path().decode() in {"neon-dotprod", "neon", "avx2+f16c", "avx2", "scalar"}
