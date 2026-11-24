"""NumPy reference implementation of the Qwen2 forward pass.

Deliberately written in the most direct way (full attention matrix, float64)
so it can serve as an oracle for the C engine. With `emulate_q8=True` the
activations entering a Q8_0/Q4_0 matmul are fake-quantised exactly like the C
kernels do, which isolates kernel bugs from quantisation noise.
"""
from dataclasses import dataclass

import numpy as np

from . import quant
from .gguf import read_gguf


@dataclass
class Config:
    n_layers: int
    dim: int
    ffn_dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    rope_theta: float
    rms_eps: float


class QwenRef:
    def __init__(self, path, emulate_q8=True):
        g = read_gguf(path)
        kv = g.kv
        assert kv["general.architecture"] == "qwen2"
        dim = kv["qwen2.embedding_length"]
        nh = kv["qwen2.attention.head_count"]
        self.cfg = Config(
            n_layers=kv["qwen2.block_count"], dim=dim, ffn_dim=kv["qwen2.feed_forward_length"],
            n_heads=nh, n_kv_heads=kv.get("qwen2.attention.head_count_kv", nh),
            head_dim=kv.get("qwen2.attention.key_length", dim // nh),
            vocab_size=g.tensors["token_embd.weight"].ne[1],
            rope_theta=kv.get("qwen2.rope.freq_base", 10000.0),
            rms_eps=kv.get("qwen2.attention.layer_norm_rms_epsilon", 1e-6))
        self.emulate_q8 = emulate_q8
        self.w = {}
        self.types = {}
        for name, t in g.tensors.items():
            self.w[name] = t.to_numpy().astype(np.float64)
            self.types[name] = t.type
        if "output.weight" not in self.w:
            self.w["output.weight"] = self.w["token_embd.weight"]
            self.types["output.weight"] = self.types["token_embd.weight"]

    def _mm(self, name, x):
        """x: (T, cols) -> (T, rows)"""
        if self.emulate_q8 and self.types[name] in (quant.Q8_0, quant.Q4_0):
            x = quant.fake_quant_q8_0(x.astype(np.float32)).astype(np.float64)
        return x @ self.w[name].T

    def _rmsnorm(self, x, w):
        x32 = x.astype(np.float32).astype(np.float64)
        ms = np.mean(x32 * x32, axis=-1, keepdims=True)
        return x32 / np.sqrt(ms + self.cfg.rms_eps) * w

    def _rope(self, x, pos):
        """NeoX rotation of (T, H, hd) at absolute positions pos (T,)"""
        hd = x.shape[-1]
        half = hd // 2
        inv = np.array([np.float32(self.cfg.rope_theta ** (-2.0 * i / hd)) for i in range(half)],
                       dtype=np.float64)
        ang = pos[:, None].astype(np.float64) * inv[None, :]
        c = np.cos(ang)[:, None, :]
        s = np.sin(ang)[:, None, :]
        a, b = x[..., :half], x[..., half:]
        return np.concatenate([a * c - b * s, a * s + b * c], axis=-1)

    def forward(self, tokens):
        """Logits for every position of `tokens` (processed from position 0)."""
        c = self.cfg
        T = len(tokens)
        hd, nh, nkv = c.head_dim, c.n_heads, c.n_kv_heads
        x = self.w["token_embd.weight"][np.asarray(tokens)]
        pos = np.arange(T)
        mask = np.triu(np.full((T, T), -np.inf), k=1)
        for l in range(c.n_layers):
            p = f"blk.{l}."
            h = self._rmsnorm(x, self.w[p + "attn_norm.weight"])
            q = self._mm(p + "attn_q.weight", h)
            k = self._mm(p + "attn_k.weight", h)
            v = self._mm(p + "attn_v.weight", h)
            if p + "attn_q.bias" in self.w:
                q = q + self.w[p + "attn_q.bias"]
                k = k + self.w[p + "attn_k.bias"]
                v = v + self.w[p + "attn_v.bias"]
            q = self._rope(q.reshape(T, nh, hd), pos)
            k = self._rope(k.reshape(T, nkv, hd), pos)
            v = v.reshape(T, nkv, hd)
            rep = nh // nkv
            k = np.repeat(k, rep, axis=1)  # GQA: head h uses kv head h // rep
            v = np.repeat(v, rep, axis=1)
            att = np.einsum("thd,shd->hts", q, k) / np.sqrt(hd) + mask[None]
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            o = np.einsum("hts,shd->thd", att, v).reshape(T, nh * hd)
            x = x + self._mm(p + "attn_output.weight", o)
            h = self._rmsnorm(x, self.w[p + "ffn_norm.weight"])
            g = self._mm(p + "ffn_gate.weight", h)
            u = self._mm(p + "ffn_up.weight", h)
            act = g / (1.0 + np.exp(-g)) * u
            x = x + self._mm(p + "ffn_down.weight", act)
        x = self._rmsnorm(x, self.w["output_norm.weight"])
        return self._mm("output.weight", x)
