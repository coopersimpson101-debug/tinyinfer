#!/usr/bin/env python3
"""Validate tinyinfer on a REAL Qwen2.5 checkpoint against Hugging Face transformers.

Needs: pip install torch transformers  (+ ~3 GB download of the HF weights).
This is for the author's machine; CI only uses random tiny models.

  python scripts/validate_hf.py models/qwen2.5-1.5b-instruct-q8_0.gguf \
      --hf Qwen/Qwen2.5-1.5B-Instruct

Checks
  1. tokenizer: tinyinfer token ids == HF tokenizer ids on a set of strings
  2. logits: last-position logits of a prompt, tinyinfer (GGUF, quantised) vs
     HF (float32). Reports max |diff|, cosine similarity and top-k overlap.
     Expect cosine > 0.99 for Q8_0; Q4_0 is noisier.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from tinyinfer_ref.capi import Model, load_lib  # noqa: E402

TEXTS = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog.",
    "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n",
    "Numbers: 1234567890, 3.14159, 2026-11-24",
    "It's we'll they're I'M",
    "Café naïve résumé, добрый день, 你好，世界。こんにちは 한국어 😀",
    "<|im_start|>user\nWhat is RoPE?<|im_end|>\n<|im_start|>assistant\n",
    "   multiple   spaces\n\n\ttabs\r\n",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--hf", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--prompt", default="<|im_start|>user\nExplain the KV cache in one sentence.<|im_end|>\n<|im_start|>assistant\n")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lib = load_lib()
    m = Model(lib, a.gguf, ctx=512)
    tok = AutoTokenizer.from_pretrained(a.hf)

    ok = 0
    for t in TEXTS:
        ours = m.tokenize(t)
        theirs = tok.encode(t, add_special_tokens=False)
        same = ours == theirs
        ok += same
        print(f"[{'OK ' if same else 'DIFF'}] {t[:40]!r}")
        if not same:
            print("   tinyinfer:", ours)
            print("   hf       :", theirs)
    print(f"tokenizer: {ok}/{len(TEXTS)} identical")

    ids = tok.encode(a.prompt, add_special_tokens=False)
    ours = m.forward(ids, all_logits=False)[0].astype(np.float64)
    hf = AutoModelForCausalLM.from_pretrained(a.hf, torch_dtype=torch.float32)
    with torch.no_grad():
        theirs = hf(torch.tensor([ids])).logits[0, -1].double().numpy()
    theirs = theirs[: len(ours)]
    cos = float(ours @ theirs / np.linalg.norm(ours) / np.linalg.norm(theirs))
    top = lambda x, k: set(np.argsort(-x)[:k].tolist())  # noqa: E731
    print(f"logits: max|diff|={np.abs(ours - theirs).max():.4f} cosine={cos:.5f} "
          f"top1 {'match' if ours.argmax() == theirs.argmax() else 'MISMATCH'} "
          f"top10 overlap={len(top(ours, 10) & top(theirs, 10))}/10")
    m.close()


if __name__ == "__main__":
    main()
