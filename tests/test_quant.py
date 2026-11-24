"""Quantisation: C vs NumPy bit-exactness and error bounds."""
import ctypes as C

import numpy as np
import pytest
from conftest import fptr
from tinyinfer_ref import quant


def c_quantize(lib, x, t):
    x = np.ascontiguousarray(x, dtype=np.float32)
    buf = C.create_string_buffer(lib.ti_type_row_size(t, x.size))
    lib.ti_quantize_row(t, fptr(x), buf, x.size)
    return buf.raw


def c_dequantize(lib, raw, t, n):
    out = np.zeros(n, dtype=np.float32)
    buf = C.create_string_buffer(raw, len(raw))
    lib.ti_dequantize_row(t, buf, fptr(out), n)
    return out


def test_fp16_all_halves_roundtrip(lib):
    h = np.arange(65536, dtype=np.uint16)
    f = h.view(np.float16).astype(np.float32)
    for i in range(0, 65536, 97):  # sample across the whole range, incl. subnormals
        if np.isnan(f[i]):
            continue
        assert lib.ti_fp16_to_fp32(int(h[i])) == f[i]
        assert lib.ti_fp32_to_fp16(float(f[i])) == h[i]


def test_fp32_to_fp16_rounding_matches_numpy(lib):
    rng = np.random.default_rng(0)
    vals = np.concatenate([
        (rng.standard_normal(3000) * 10.0 ** rng.integers(-9, 6, 3000)).astype(np.float32),
        np.array([65504, 65519, 65520, 1e6, -1e6, 2.0 ** -24, 2.0 ** -25, 2.0 ** -26, 3 * 2.0 ** -26,
                  6.0e-8, 1.0 + 2.0 ** -11, 1.0 + 3 * 2.0 ** -11, 0.0, -0.0], dtype=np.float32)])
    with np.errstate(over="ignore"):
        ref = vals.astype(np.float16).view(np.uint16)
    got = np.array([lib.ti_fp32_to_fp16(float(v)) for v in vals], dtype=np.uint16)
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("t", [quant.Q8_0, quant.Q4_0, quant.F16])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 300.0])
def test_c_matches_numpy_bit_exact(lib, t, scale):
    rng = np.random.default_rng(42)
    x = (rng.standard_normal(32 * 64) * scale).astype(np.float32)
    x[:32] = 0.0                     # all-zero block (d = 0)
    x[32:64] = np.linspace(-1, 1, 32)  # ties / symmetric extremes
    x[64] = 50 * scale               # outlier
    assert c_quantize(lib, x, t) == quant.quantize(x, t)
    raw = quant.quantize(x, t)
    np.testing.assert_array_equal(c_dequantize(lib, raw, t, x.size), quant.dequantize(raw, t, x.size))


def test_q8_0_error_bound(lib):
    rng = np.random.default_rng(1)
    x = rng.standard_normal(32 * 512).astype(np.float32)
    y = c_dequantize(lib, c_quantize(lib, x, quant.Q8_0), quant.Q8_0, x.size)
    d = np.abs(x.reshape(-1, 32)).max(1) / 127
    # |x - q*d16| <= d/2 + 127*|d - d16|, |d - d16| <= d * 2^-11
    bound = d / 2 + 127 * d * 2.0 ** -11 + 1e-12
    err = np.abs(x - y).reshape(-1, 32).max(1)
    assert np.all(err <= bound)
    rel = np.sqrt(np.mean((x - y) ** 2) / np.mean(x ** 2))
    assert rel < 0.01, rel


def test_q4_0_error_bound(lib):
    rng = np.random.default_rng(2)
    x = rng.standard_normal(32 * 512).astype(np.float32)
    y = c_dequantize(lib, c_quantize(lib, x, quant.Q4_0), quant.Q4_0, x.size)
    xb = x.reshape(-1, 32)
    idx = np.abs(xb).argmax(1)
    d = np.abs(xb[np.arange(len(xb)), idx]) / 8
    bound = d + 15 * d * 2.0 ** -11 + 1e-12          # clamp at q=15 can cost a full step
    err = np.abs(x - y).reshape(-1, 32)
    assert np.all(err.max(1) <= bound)
    # elements not hitting the clamp are within half a step
    q = np.abs(xb) / d[:, None]
    unclamped = q < 7.5
    half = np.broadcast_to((d / 2 + 15 * d * 2.0 ** -11)[:, None], err.shape)
    assert np.all(err[unclamped] <= half[unclamped] + 1e-12)
    # the extreme value of each block is reproduced (it maps exactly to -8 * d)
    yb = y.reshape(-1, 32)
    np.testing.assert_allclose(yb[np.arange(len(xb)), idx], xb[np.arange(len(xb)), idx], rtol=2e-3)
    rel = np.sqrt(np.mean((x - y) ** 2) / np.mean(x ** 2))
    assert rel < 0.12, rel


def test_row_sizes(lib):
    assert lib.ti_type_row_size(quant.Q8_0, 4096) == 4096 // 32 * 34
    assert lib.ti_type_row_size(quant.Q4_0, 4096) == 4096 // 32 * 18
    assert lib.ti_type_row_size(quant.F16, 10) == 20
    assert lib.ti_type_row_size(quant.F32, 10) == 40
