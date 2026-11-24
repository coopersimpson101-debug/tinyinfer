"""NumPy reference for the Q8_0 / Q4_0 block formats (bit-exact with src/quant.c)."""
import numpy as np

QK = 32
F32, F16, Q4_0, Q8_0, BF16 = 0, 1, 2, 8, 30
TYPE_NAMES = {F32: "f32", F16: "f16", Q4_0: "q4_0", Q8_0: "q8_0", BF16: "bf16"}
NAME_TO_TYPE = {v: k for k, v in TYPE_NAMES.items()}

Q8_DTYPE = np.dtype([("d", "<f2"), ("qs", "i1", QK)])
Q4_DTYPE = np.dtype([("d", "<f2"), ("qs", "u1", QK // 2)])
assert Q8_DTYPE.itemsize == 34 and Q4_DTYPE.itemsize == 18


def row_size(ggml_type: int, n: int) -> int:
    if ggml_type == F32:
        return 4 * n
    if ggml_type in (F16, BF16):
        return 2 * n
    if ggml_type == Q8_0:
        return n // QK * 34
    if ggml_type == Q4_0:
        return n // QK * 18
    raise ValueError(f"unsupported type {ggml_type}")


def _roundf(x: np.ndarray) -> np.ndarray:
    """C roundf: half away from zero. Done in float64 so |x|+0.5 is exact."""
    x = x.astype(np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def quantize_q8_0(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, QK)
    amax = np.abs(x).max(axis=1)
    d = (amax / np.float32(127.0)).astype(np.float32)
    with np.errstate(divide="ignore"):
        idv = np.where(d != 0, np.float32(1.0) / d, np.float32(0.0)).astype(np.float32)
    q = _roundf((x * idv[:, None]).astype(np.float32)).astype(np.int8)
    out = np.empty(x.shape[0], dtype=Q8_DTYPE)
    out["d"] = d.astype(np.float16)
    out["qs"] = q
    return out


def dequantize_q8_0(b: np.ndarray) -> np.ndarray:
    d = b["d"].astype(np.float32)
    return (d[:, None] * b["qs"].astype(np.float32)).reshape(-1)


def quantize_q4_0(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, QK)
    idx = np.abs(x).argmax(axis=1)
    maxv = x[np.arange(x.shape[0]), idx]
    d = (maxv / np.float32(-8.0)).astype(np.float32)
    with np.errstate(divide="ignore"):
        idv = np.where(d != 0, np.float32(1.0) / d, np.float32(0.0)).astype(np.float32)
    xs = (x * idv[:, None]).astype(np.float32)
    q = np.trunc((xs + np.float32(8.5)).astype(np.float32)).astype(np.int32)
    q = np.clip(q, 0, 15).astype(np.uint8)
    out = np.empty(x.shape[0], dtype=Q4_DTYPE)
    out["d"] = d.astype(np.float16)
    out["qs"] = q[:, : QK // 2] | (q[:, QK // 2:] << 4)
    return out


def dequantize_q4_0(b: np.ndarray) -> np.ndarray:
    d = b["d"].astype(np.float32)
    lo = (b["qs"] & 0x0F).astype(np.int32) - 8
    hi = (b["qs"] >> 4).astype(np.int32) - 8
    q = np.concatenate([lo, hi], axis=1).astype(np.float32)
    return (d[:, None] * q).reshape(-1)


def quantize(x: np.ndarray, ggml_type: int) -> bytes:
    """Quantise a (rows, cols) float array row-major into GGUF tensor bytes."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if ggml_type == F32:
        return x.tobytes()
    if ggml_type == F16:
        return x.astype("<f2").tobytes()
    if ggml_type == BF16:
        u = x.view(np.uint32)
        # round to nearest even
        r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")
        return r.tobytes()
    if ggml_type == Q8_0:
        return quantize_q8_0(x.reshape(-1)).tobytes()
    if ggml_type == Q4_0:
        return quantize_q4_0(x.reshape(-1)).tobytes()
    raise ValueError(ggml_type)


def dequantize(buf: bytes, ggml_type: int, n: int) -> np.ndarray:
    if ggml_type == F32:
        return np.frombuffer(buf, dtype="<f4", count=n).astype(np.float32)
    if ggml_type == F16:
        return np.frombuffer(buf, dtype="<f2", count=n).astype(np.float32)
    if ggml_type == BF16:
        u = np.frombuffer(buf, dtype="<u2", count=n).astype(np.uint32) << 16
        return u.view(np.float32)
    if ggml_type == Q8_0:
        return dequantize_q8_0(np.frombuffer(buf, dtype=Q8_DTYPE, count=n // QK))
    if ggml_type == Q4_0:
        return dequantize_q4_0(np.frombuffer(buf, dtype=Q4_DTYPE, count=n // QK))
    raise ValueError(ggml_type)


def fake_quant_q8_0(x: np.ndarray) -> np.ndarray:
    """What the C matmul does to activations before an int8 dot product."""
    shp = x.shape
    return dequantize_q8_0(quantize_q8_0(x.astype(np.float32).reshape(-1))).reshape(shp)
