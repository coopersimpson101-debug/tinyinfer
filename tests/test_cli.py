"""End-to-end: the CLI, the re-quantiser and the C unit-test binary."""
import re
import subprocess
import sys

import numpy as np
import pytest
from conftest import BUILD, EXE
from tinyinfer_ref.gguf import read_gguf
from tinyinfer_ref.model import QwenRef
from tinyinfer_ref.tokenizer import BPETokenizer

CLI = BUILD / f"tinyinfer{EXE}"
QUANT = BUILD / f"tinyinfer-quantize{EXE}"
CORE = BUILD / f"test_core{EXE}"


def need(p):
    if not p.exists():
        pytest.skip(f"{p.name} not built (python build.py)")


def run(args, **kw):
    return subprocess.run([str(a) for a in args], capture_output=True, timeout=120, **kw)


def test_cli_greedy_generation_matches_reference(models):
    need(CLI)
    path = models["f32"]
    prompt = "Once upon a time"
    r = run([CLI, "-m", path, "-p", prompt, "-n", "12", "--temp", "0", "--repeat-penalty", "1", "-t", "2"])
    assert r.returncode == 0, r.stderr.decode()
    err = r.stderr.decode()
    assert re.search(r"prefill: \d+ tokens in [\d.]+ ms \([\d.]+ tok/s\)", err)
    assert re.search(r"decode:  \d+ tokens in [\d.]+ ms \([\d.]+ tok/s\)", err)

    # independent greedy decode with the numpy reference + python tokenizer
    g = read_gguf(path)
    tok = BPETokenizer(g.kv["tokenizer.ggml.tokens"], g.kv["tokenizer.ggml.merges"],
                       g.kv["tokenizer.ggml.token_type"])
    ref = QwenRef(path)
    ids = tok.encode(prompt)
    stops = {g.kv["tokenizer.ggml.eos_token_id"], tok.vocab["<|im_end|>"], tok.vocab["<|endoftext|>"]}
    for _ in range(12):
        nxt = int(ref.forward(ids)[-1].argmax())
        ids.append(nxt)
        if nxt in stops:
            break
    expect = tok.decode_bytes([i for i in ids if i not in stops])
    assert r.stdout.endswith(b"\n")
    assert r.stdout[:-1] == expect


def test_cli_chat_and_sampling(models):
    need(CLI)
    r = run([CLI, "-m", models["q4_0"], "-p", "hi", "-n", "8", "--chat", "--temp", "0.8",
             "--top-k", "20", "--top-p", "0.95", "-s", "42"])
    assert r.returncode == 0, r.stderr.decode()
    r2 = run([CLI, "-m", models["q4_0"], "-p", "hi", "-n", "8", "--chat", "--temp", "0.8",
              "--top-k", "20", "--top-p", "0.95", "-s", "42"])
    assert r.stdout == r2.stdout  # seeded sampling is reproducible


def test_cli_bench_mode(models):
    need(CLI)
    r = run([CLI, "-m", models["q8_0"], "--bench", "32,8", "-r", "1"])
    assert r.returncode == 0, r.stderr.decode()
    out = r.stdout.decode()
    assert "| pp32 |" in out and "| tg8 |" in out


def test_cli_errors(tmp_path):
    need(CLI)
    r = run([CLI, "-m", tmp_path / "nope.gguf"])
    assert r.returncode != 0 and b"cannot open" in r.stderr
    assert run([CLI]).returncode != 0


@pytest.mark.parametrize("target", ["q8_0", "q4_0", "f16"])
def test_quantize_tool_matches_python_quantiser(models, tmp_path, target):
    """f32 -> X with the C tool must be byte-identical to the Python writer's X."""
    need(QUANT)
    out = tmp_path / f"requant-{target}.gguf"
    r = run([QUANT, models["f32"], out, target])
    assert r.returncode == 0, r.stderr.decode()
    a, b = read_gguf(out), read_gguf(models[target])
    assert a.kv["general.file_type"] == b.kv["general.file_type"]
    assert a.kv["tokenizer.ggml.merges"] == b.kv["tokenizer.ggml.merges"]
    assert a.tensors.keys() == b.tensors.keys()
    for k in a.tensors:
        assert a.tensors[k].type == b.tensors[k].type, k
        assert a.tensors[k].data == b.tensors[k].data, k


def test_requantised_model_runs(models, tmp_path, open_model):
    need(QUANT)
    out = tmp_path / "q8-to-q4.gguf"
    assert run([QUANT, models["q8_0"], out, "q4_0"]).returncode == 0
    logits = open_model(out).forward([1, 2, 3])
    assert np.isfinite(logits).all()


def test_c_unit_tests(models):
    need(CORE)
    ref = QwenRef(models["q8_0"]).forward([5, 77, 300, 12]).astype(np.float32)
    exp = BUILD / "test_core_expected.bin"
    ref.tofile(exp)
    try:
        r = run([CORE, models["q8_0"], exp])
    finally:
        exp.unlink()
    sys.stdout.write(r.stdout.decode())
    assert r.returncode == 0, r.stdout.decode() + r.stderr.decode()
    assert b"0 failed" in r.stdout


def _dump_logits(exe, model, prompt, out, extra=()):
    r = run([exe, "-m", model, "-p", prompt, "-n", "1", "--temp", "0", "--dump-logits", out, *extra])
    assert r.returncode == 0, r.stderr.decode()
    return np.fromfile(out, dtype=np.float32)


def test_cli_dump_logits_matches_reference(models, tmp_path):
    need(CLI)
    g = read_gguf(models["f16"])
    tok = BPETokenizer(g.kv["tokenizer.ggml.tokens"], g.kv["tokenizer.ggml.merges"],
                       g.kv["tokenizer.ggml.token_type"])
    prompt = "The quick brown fox"
    got = _dump_logits(CLI, models["f16"], prompt, tmp_path / "l.bin")
    ref = QwenRef(models["f16"]).forward(tok.encode(prompt))[-1]
    assert np.abs(got - ref).max() / ref.std() < 1e-4


METAL = BUILD / "tinyinfer-metal"


@pytest.mark.parametrize("name", ["f32", "f16", "q8_0", "q4_0", "f32_untied"])
def test_metal_backend_matches_cpu(models, tmp_path, name):
    """Only runs on macOS after `make metal`. GPU uses f32 activations (no Q8_0
    activation quantisation), so it is compared with the float reference."""
    need(METAL)
    g = read_gguf(models[name])
    tok = BPETokenizer(g.kv["tokenizer.ggml.tokens"], g.kv["tokenizer.ggml.merges"],
                       g.kv["tokenizer.ggml.token_type"])
    prompt = "Once upon a time there was"
    got = _dump_logits(METAL, models[name], prompt, tmp_path / "m.bin", ["--metal"])
    ref = QwenRef(models[name], emulate_q8=False).forward(tok.encode(prompt))[-1]
    assert np.abs(got - ref).max() / ref.std() < 1e-3
