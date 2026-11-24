"""Shared fixtures: the compiled C library (via ctypes) and tiny GGUF models."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from tinyinfer_ref.capi import (BUILD, EXE, Config, Model, SamplerParams, fptr, iptr,  # noqa: E402,F401
                                load_lib)

_load_lib = load_lib


@pytest.fixture(scope="session")
def lib():
    return load_lib()


@pytest.fixture(scope="session")
def models(tmp_path_factory):
    """Paths of tiny random models in several weight formats."""
    from make_tiny_model import build
    from tinyinfer_ref import quant
    d = tmp_path_factory.mktemp("models")
    out = {}
    for name in ("f32", "f16", "q8_0", "q4_0"):
        out[name] = build(str(d / f"tiny-{name}.gguf"), quant.NAME_TO_TYPE[name], seed=1)
    out["f32_untied"] = build(str(d / "tiny-f32-untied.gguf"), quant.F32, seed=2, tied=False,
                              theta=10000.0, n_layers=3)
    out["q8_0_untied"] = build(str(d / "tiny-q8-untied.gguf"), quant.Q8_0, seed=2, tied=False,
                               theta=10000.0, n_layers=3)
    # MHA (n_kv == n_heads) variant and a wider head_dim
    out["f32_mha"] = build(str(d / "tiny-f32-mha.gguf"), quant.F32, seed=3, n_heads=2, n_kv=2,
                           dim=64)
    return out


@pytest.fixture
def open_model(lib):
    opened = []

    def _open(path, **kw):
        m = Model(lib, path, **kw)
        opened.append(m)
        return m
    yield _open
    for m in opened:
        m.close()
