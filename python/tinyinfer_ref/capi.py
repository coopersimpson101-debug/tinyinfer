"""ctypes bindings for the tinyinfer shared library (used by tests and examples)."""
import ctypes as C
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUILD = Path(os.environ.get("TI_BUILD_DIR", ROOT / "build"))
LIBNAME = {"win32": "tinyinfer.dll", "darwin": "libtinyinfer.dylib"}.get(sys.platform, "libtinyinfer.so")
EXE = ".exe" if sys.platform == "win32" else ""


class SamplerParams(C.Structure):
    _fields_ = [("temperature", C.c_float), ("top_k", C.c_int), ("top_p", C.c_float),
                ("repeat_penalty", C.c_float), ("repeat_last_n", C.c_int), ("seed", C.c_uint64)]


class Config(C.Structure):
    _fields_ = [(n, C.c_int) for n in ("n_layers", "dim", "ffn_dim", "n_heads", "n_kv_heads",
                                       "head_dim", "vocab_size", "max_ctx", "train_ctx")] + \
               [("rope_theta", C.c_float), ("rms_eps", C.c_float),
                ("tied_embeddings", C.c_int), ("has_qkv_bias", C.c_int)]


def load_lib():
    path = BUILD / LIBNAME
    if not path.exists() and not os.environ.get("TI_NO_AUTOBUILD"):
        subprocess.check_call([sys.executable, str(ROOT / "build.py")])
    lib = C.CDLL(str(path))
    P, I, F = C.c_void_p, C.c_int, C.c_float
    fp = C.POINTER(C.c_float)
    ip = C.POINTER(C.c_int32)
    sig = {
        "ti_model_load": (P, [C.c_char_p, I, I]),
        "ti_model_free": (None, [P]),
        "ti_model_config": (C.POINTER(Config), [P]),
        "ti_last_error": (C.c_char_p, []),
        "ti_forward": (I, [P, ip, I, I, fp, I]),
        "ti_tokenize": (I, [P, C.c_char_p, ip, I, I]),
        "ti_detokenize": (I, [P, ip, I, C.c_char_p, I]),
        "ti_token_id": (I, [P, C.c_char_p]),
        "ti_eos_id": (I, [P]),
        "ti_bos_id": (I, [P]),
        "ti_pretokenize": (I, [C.c_char_p, I, C.POINTER(C.c_int), I]),
        "ti_type_row_size": (C.c_size_t, [I, C.c_int64]),
        "ti_quantize_row": (None, [I, fp, P, C.c_int64]),
        "ti_dequantize_row": (None, [I, P, fp, C.c_int64]),
        "ti_fp32_to_fp16": (C.c_uint16, [F]),
        "ti_fp16_to_fp32": (F, [C.c_uint16]),
        "ti_matmul": (None, [I, P, fp, fp, I, I, I]),
        "ti_set_threads": (None, [I]),
        "ti_get_threads": (I, []),
        "ti_simd_path": (C.c_char_p, []),
        "ti_sampler_default": (SamplerParams, []),
        "ti_sample": (I, [fp, I, C.POINTER(SamplerParams), ip, I, C.POINTER(C.c_uint64)]),
    }
    for name, (res, args) in sig.items():
        f = getattr(lib, name)
        f.restype = res
        f.argtypes = args
    return lib




def fptr(a):
    return a.ctypes.data_as(C.POINTER(C.c_float))


def iptr(a):
    return a.ctypes.data_as(C.POINTER(C.c_int32))


class Model:
    """Thin ctypes wrapper around a loaded ti_model."""

    def __init__(self, lib, path, ctx=0, threads=0):
        self.lib = lib
        self.h = lib.ti_model_load(str(path).encode(), ctx, threads)
        if not self.h:
            raise RuntimeError(lib.ti_last_error().decode())
        self.cfg = lib.ti_model_config(self.h).contents

    def forward(self, tokens, pos=0, all_logits=True):
        t = np.ascontiguousarray(tokens, dtype=np.int32)
        V = self.cfg.vocab_size
        out = np.zeros((len(t) if all_logits else 1, V), dtype=np.float32)
        rc = self.lib.ti_forward(self.h, iptr(t), len(t), pos, fptr(out), int(all_logits))
        if rc != 0:
            raise RuntimeError(self.lib.ti_last_error().decode())
        return out

    def tokenize(self, text, parse_special=True):
        b = text.encode("utf-8")
        cap = len(b) + 16
        out = np.zeros(cap, dtype=np.int32)
        n = self.lib.ti_tokenize(self.h, b, iptr(out), cap, int(parse_special))
        assert 0 <= n <= cap
        return out[:n].tolist()

    def detokenize_bytes(self, ids):
        t = np.ascontiguousarray(ids, dtype=np.int32)
        need = self.lib.ti_detokenize(self.h, iptr(t), len(t), None, 0)
        buf = C.create_string_buffer(need + 1)
        self.lib.ti_detokenize(self.h, iptr(t), len(t), buf, need + 1)
        return buf.raw[:need]

    def close(self):
        if self.h:
            self.lib.ti_model_free(self.h)
            self.h = None
