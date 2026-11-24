#!/usr/bin/env python3
"""Write a tiny, randomly initialised Qwen2 model to GGUF v3.

The model has the real Qwen2 structure (GQA, QKV bias, SwiGLU, RMSNorm, NeoX
RoPE with a configurable theta, tied or untied embeddings) and a real
byte-level BPE tokenizer trained on a small built-in corpus, so the whole C
pipeline can be tested without downloading anything.

  python scripts/make_tiny_model.py -o build/tiny-q8_0.gguf --type q8_0
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from tinyinfer_ref import quant  # noqa: E402
from tinyinfer_ref.gguf import GGUFWriter  # noqa: E402
from tinyinfer_ref.tokenizer import train_bpe  # noqa: E402

CORPUS = """
Once upon a time there was a tiny inference engine. It loaded a model, read the
tokenizer, and generated text one token at a time. The engine's kernels were
written by hand: matrix multiplication, RMS normalisation, rotary embeddings,
softmax and attention. We'll test it, they're sure it works, and you've seen it.
def forward(x, w):
    return x @ w.T  # numpy reference
for (int i = 0; i < n; i++) { y[i] += a * x[i]; }
The quick brown fox jumps over the lazy dog. 1234567890 3.14159 2026-11-24
Numbers like 42, 1000 and 65536 appear often; so do words such as the, and, of.
Café naïve résumé über straße добрый день καλημέρα
你好，世界。今天天气很好。 こんにちは世界 한국어 \U0001f600 \U0001f680
  indented line\n\ttabbed line\r\nwindows line
""" * 3


# Real Qwen2.5 hyper-parameters, for speed measurements with random weights.
PRESETS = {
    "qwen2.5-0.5b": dict(n_layers=24, dim=896, ffn=4864, n_heads=14, n_kv=2, vocab=151936),
    "qwen2.5-1.5b": dict(n_layers=28, dim=1536, ffn=8960, n_heads=12, n_kv=2, vocab=151936),
}


def build(path, wtype, seed=0, n_layers=2, dim=64, ffn=128, n_heads=4, n_kv=2, ctx=256,
          theta=1e6, tied=True, n_merges=300, embd_type=None, vocab=None):
    rng = np.random.default_rng(seed)
    tokens, merges, types = train_bpe(CORPUS, n_merges)
    if vocab is not None and vocab > len(tokens):  # pad with unused (type 5) tokens
        pad = vocab - len(tokens)
        tokens = tokens + [f"[unused{i}]" for i in range(pad)]
        types = types + [5] * pad
    vocab = len(tokens)
    hd = dim // n_heads
    w = GGUFWriter(path)
    w.add_string("general.architecture", "qwen2")
    w.add_string("general.name", "tinyinfer-random-qwen2")
    w.add_u32("general.file_type", {quant.F32: 0, quant.F16: 1, quant.Q4_0: 2, quant.Q8_0: 7}[wtype])
    w.add_u32("qwen2.block_count", n_layers)
    w.add_u32("qwen2.context_length", ctx)
    w.add_u32("qwen2.embedding_length", dim)
    w.add_u32("qwen2.feed_forward_length", ffn)
    w.add_u32("qwen2.attention.head_count", n_heads)
    w.add_u32("qwen2.attention.head_count_kv", n_kv)
    w.add_f32("qwen2.rope.freq_base", theta)
    w.add_f32("qwen2.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "qwen2")
    w.add_array_str("tokenizer.ggml.tokens", tokens)
    w.add_array_i32("tokenizer.ggml.token_type", types)
    w.add_array_str("tokenizer.ggml.merges", merges)
    w.add_u32("tokenizer.ggml.eos_token_id", tokens.index("<|endoftext|>"))
    w.add_u32("tokenizer.ggml.bos_token_id", tokens.index("<|endoftext|>"))

    def mat(rows, cols, scale=1.0):
        return rng.standard_normal((rows, cols)).astype(np.float32) * scale / np.sqrt(cols)

    et = wtype if embd_type is None else embd_type
    w.add_tensor("token_embd.weight", rng.standard_normal((vocab, dim)).astype(np.float32), et)
    for l in range(n_layers):
        p = f"blk.{l}."
        w.add_tensor(p + "attn_norm.weight", 1 + 0.1 * rng.standard_normal(dim))
        w.add_tensor(p + "attn_q.weight", mat(n_heads * hd, dim, 2.0), wtype)
        w.add_tensor(p + "attn_q.bias", 0.1 * rng.standard_normal(n_heads * hd))
        w.add_tensor(p + "attn_k.weight", mat(n_kv * hd, dim, 2.0), wtype)
        w.add_tensor(p + "attn_k.bias", 0.1 * rng.standard_normal(n_kv * hd))
        w.add_tensor(p + "attn_v.weight", mat(n_kv * hd, dim), wtype)
        w.add_tensor(p + "attn_v.bias", 0.1 * rng.standard_normal(n_kv * hd))
        w.add_tensor(p + "attn_output.weight", mat(dim, n_heads * hd), wtype)
        w.add_tensor(p + "ffn_norm.weight", 1 + 0.1 * rng.standard_normal(dim))
        w.add_tensor(p + "ffn_gate.weight", mat(ffn, dim), wtype)
        w.add_tensor(p + "ffn_up.weight", mat(ffn, dim), wtype)
        w.add_tensor(p + "ffn_down.weight", mat(dim, ffn), wtype)
    w.add_tensor("output_norm.weight", 1 + 0.1 * rng.standard_normal(dim))
    if not tied:
        w.add_tensor("output.weight", mat(vocab, dim, 4.0), wtype)
    w.write()
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--type", default="f32", choices=["f32", "f16", "q8_0", "q4_0"])
    ap.add_argument("--untied", action="store_true", help="separate output.weight")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--ffn", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--theta", type=float, default=1e6)
    ap.add_argument("--preset", choices=sorted(PRESETS),
                    help="real Qwen2.5 shapes with random weights (speed tests only; large file)")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    kw = dict(n_layers=a.layers, dim=a.dim, ffn=a.ffn, n_heads=a.heads, n_kv=a.kv_heads)
    if a.preset:
        kw = dict(PRESETS[a.preset])
        a.ctx = max(a.ctx, 4096)
    build(a.out, quant.NAME_TO_TYPE[a.type], seed=a.seed, ctx=a.ctx, theta=a.theta,
          tied=not a.untied, **kw)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
