#!/usr/bin/env python3
"""Drive the C engine from Python via ctypes: tokenize, prefill, sample, stream.

  python build.py
  python scripts/make_tiny_model.py -o build/tiny.gguf
  python examples/generate.py build/tiny.gguf "Once upon a time" -n 32

(With a random tiny model the text is gibberish - the point is the API.)
"""
import argparse
import ctypes as C
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from tinyinfer_ref.capi import Model, SamplerParams, fptr, iptr, load_lib  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("prompt")
    ap.add_argument("-n", type=int, default=32)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    lib = load_lib()
    m = Model(lib, a.model, ctx=512)
    params = SamplerParams(a.temp, 40, 0.9, 1.1, 64, a.seed)
    rng = C.c_uint64(a.seed)
    toks = m.tokenize(a.prompt)
    t0 = time.perf_counter()
    logits = m.forward(toks, pos=0, all_logits=False)[0]
    t1 = time.perf_counter()
    sys.stdout.buffer.write(m.detokenize_bytes(toks))
    n_prompt, n_gen = len(toks), 0
    for _ in range(a.n):
        hist = np.array(toks, dtype=np.int32)
        nxt = lib.ti_sample(fptr(logits), len(logits), C.byref(params), iptr(hist), len(hist), C.byref(rng))
        if nxt == lib.ti_eos_id(m.h):
            break
        toks.append(nxt)
        n_gen += 1
        sys.stdout.buffer.write(m.detokenize_bytes([nxt]))
        sys.stdout.flush()
        logits = m.forward([nxt], pos=len(toks) - 1, all_logits=False)[0]
    t2 = time.perf_counter()
    print(f"\n\nprefill {n_prompt} tok in {1e3 * (t1 - t0):.1f} ms, "
          f"decode {n_gen} tok in {1e3 * (t2 - t1):.1f} ms", file=sys.stderr)
    m.close()


if __name__ == "__main__":
    main()
