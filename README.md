<p align="center">
  <img src="docs/assets/banner.svg" alt="tinyinfer: an LLM inference engine from scratch in C11" width="100%">
</p>

<p align="center">
  <a href="https://github.com/coopersimpson101-debug/tinyinfer/actions/workflows/ci.yml"><img src="https://github.com/coopersimpson101-debug/tinyinfer/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license"></a>
  <img src="https://img.shields.io/badge/language-C11-F97316.svg" alt="C11">
  <img src="https://img.shields.io/badge/dependencies-none-success.svg" alt="no dependencies">
  <img src="https://img.shields.io/badge/SIMD-NEON%20SDOT%20%C2%B7%20AVX2-informational.svg" alt="NEON SDOT and AVX2">
  <img src="https://img.shields.io/badge/GPU-Metal-lightgrey.svg" alt="Metal backend">
</p>

**Run Qwen2.5 GGUF models with an engine you can read in an afternoon.** tinyinfer
is a from-scratch C11 LLM runtime - GGUF loader, BPE tokenizer, Qwen2
transformer, Q8_0/Q4_0 kernels, thread pool, sampler and a Metal backend - with
no ggml, no BLAS and no tokenizer library, every piece checked against an
independent NumPy oracle.

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#results">Results</a> ·
  <a href="#roadmap">Roadmap</a>
</p>

<p align="center">
  <img src="docs/assets/terminal.svg" alt="Real tinyinfer run on the random-weight 2-layer test model" width="92%">
</p>
<p align="center"><sub>Real output on the 2-layer <b>random-weight</b> test model that the test-suite generates (hence the gibberish).
Regenerate with <code>python docs/assets/make_figures.py</code>.</sub></p>

## Highlights

- **Zero dependencies:** ~3.8k lines of C11 + Metal plus libc/pthreads (Win32 threads on Windows); Python + NumPy only for tests.
- **Quantised int8 matmuls:** Q8_0 / Q4_0 weights x Q8_0 activations, two `SDOT`s per 32-weight block on Apple Silicon, `maddubs` on AVX2, bit-exact with ggml's formats.
- **Verified, not eyeballed:** 168 pytest cases + 112 C tests compare logits, tokens and bytes with a NumPy reference and the real Qwen2 regex.
- **Runs on 4 CI targets:** Linux AVX2 / scalar / ASan+UBSan, Windows (zig cc), aarch64 NEON under qemu, and a GitHub macOS arm64 runner (NEON natively + Metal GPU).
- **Metal backend:** matvec for f32/f16/q8_0/q4_0, RMSNorm, RoPE, online-softmax attention, SwiGLU; GPU logits match NumPy to < 1e-3 x std on the CI runner.
- **Real-size speed on a desktop CPU:** 68.5 tok/s decode for Qwen2.5-0.5B-shaped Q4_0, 22.5 tok/s for 1.5B (Ryzen 5 3600, random weights).

## Status

| | what | where it was checked |
|---|---|---|
| ✅ | CPU engine, scalar + AVX2: full test-suite (168 passed, 112 C tests) | Windows 10 / Ryzen 5 3600 here; Linux CI (+ ASan/UBSan) |
| ✅ | NEON + SDOT kernels and full test-suite, natively | GitHub macOS arm64 runner (CI `macos-arm64`) |
| ✅ | Metal backend builds; 5 GPU-vs-NumPy logit tests pass | GitHub macOS arm64 runner (toy models only) |
| 🧪 | NEON kernels on `cortex-a76` + forward pass vs NumPy | qemu-aarch64 in CI (emulated) |
| ⏳ | Real Qwen2.5-1.5B weights vs Hugging Face logits | needs the 3 GB download + `scripts/validate_hf.py` |
| ⏳ | Throughput on an M1 MacBook, CPU and Metal, vs llama.cpp | needs an M1: `scripts/bench.sh` |

## Quickstart

```bash
git clone https://github.com/coopersimpson101-debug/tinyinfer && cd tinyinfer
make                                   # or: pip install ziglang && python build.py   (Windows)
python scripts/make_tiny_model.py -o build/tiny.gguf --type q4_0   # no download needed
build/tinyinfer -m build/tiny.gguf -p "Hello" -n 32                # random weights -> gibberish
scripts/download_qwen.sh && build/tinyinfer -m models/qwen2.5-1.5b-instruct-q8_0.gguf --chat -p "What is a KV cache?"
```

## Contents

- [What's implemented](#whats-implemented)
- [Architecture](#architecture)
- [How it works](#how-it-works)
- [Build and run](#build-and-run)
- [Tests](#tests)
- [Results](#results)
- [Hardware validation / TODO for the author](#hardware-validation--todo-for-the-author)
- [Roadmap](#roadmap)

## What's implemented

| component | file | notes |
|---|---|---|
| GGUF v3 reader (mmap, zero-copy) | `src/gguf.c` | all KV value types, arrays, alignment, bounds checks |
| GGUF writer / re-quantiser | `src/quantize_main.c` | `tinyinfer-quantize in.gguf out.gguf q4_0` ("pure" Q4_0, readable by llama.cpp too) |
| byte-level BPE tokenizer | `src/tokenizer.c` | Qwen2 regex pre-tokeniser hand-written in C, merge ranks, special tokens, streaming-safe decode |
| Qwen2 forward pass | `src/model.c` | RMSNorm, NeoX RoPE (theta from file), GQA, QKV bias, SwiGLU, tied/untied embeddings, batched prefill |
| KV cache | `src/model.c` | preallocated f32 `[layer][pos][kv_dim]` |
| quantisation | `src/quant.c` | Q8_0 / Q4_0 quantise + dequantise, bit-exact with ggml; fp16 <-> fp32 by hand |
| matmul kernels | `src/kernels.c` | F32, F16, BF16, Q8_0, Q4_0 weights; int8 dot products; NEON (+SDOT), AVX2 (+F16C), scalar; 4-token blocked prefill kernel |
| thread pool | `src/platform.c` | pthreads or native Win32, spin-then-sleep, dynamic chunking |
| sampling | `src/sampler.c` | greedy, temperature, top-k (heap), top-p (with pre-filter), repetition penalty, seeded xorshift |
| CLI | `src/main.c` | prompt / chat template / streaming / prefill + decode tok/s / `--bench pp,tg` |
| Metal backend | `metal/ti_metal.m`, `metal/kernels.metal` | matvec f32/f16/q8_0/q4_0, RMSNorm, RoPE, softmax-attention, SwiGLU; one command buffer per token. **Unverified.** |
| NumPy reference | `python/tinyinfer_ref/` | GGUF reader/writer, quant formats, BPE (+ trainer), Qwen2 forward |
| scripts | `scripts/` | random-model generator, Qwen download, llama.cpp benchmark, HF validation |

## Architecture

```mermaid
flowchart LR
    subgraph file["model.gguf (mmap)"]
        KV["metadata KV<br/>hparams + tokenizer"]
        W["tensors<br/>F16 / Q8_0 / Q4_0"]
    end
    KV --> G[gguf.c parser]
    W --> G
    G --> T[tokenizer.c<br/>specials → pre-tokenise → BPE merges]
    G --> M[model.c<br/>Qwen2 weights + KV cache]
    P([prompt]) --> T
    T -- token ids --> M
    M --> L{backend}
    L -- CPU --> K[kernels.c<br/>Q8_0 act. × Q8_0/Q4_0 weights<br/>NEON SDOT · AVX2 · scalar]
    K <--> TP[platform.c<br/>thread pool]
    L -- "--metal" --> MT[ti_metal.m + kernels.metal<br/>one command buffer / token]
    K --> LG[logits]
    MT --> LG
    LG --> S[sampler.c<br/>penalty → temp → top-k → top-p]
    S -- next token --> M
    S --> D[detokenise → stdout]
```

One decoder layer, as computed in `forward_chunk` (`src/model.c`):

```
x ──┬─► RMSNorm ─► Wq·+bq ─► RoPE ─┐
    │            ─► Wk·+bk ─► RoPE ─┼─► KV cache[l][pos] ─► softmax(qKᵀ/√d)·V ─► Wo ─┐
    │            ─► Wv·+bv ─────────┘          (GQA: head h reads kv head h/g)    │
    └───────────────────────────────────────────────────────────────────── (+) ◄──┘
                                                                           │
h ──┬─► RMSNorm ─► Wgate ─► SiLU ─┐                                         ▼
    │            ─► Wup   ────────(×)─► Wdown ─────────────────────────────(+) ─► x'
    └──────────────────────────────────────────────────────────────────────┘
```

## How it works

**RMSNorm.** `y = x / sqrt(mean(x²) + ε) · g` (sum of squares accumulated in
double).

**RoPE (NeoX layout).** Qwen2 GGUFs use the "rotate halves" convention: for
`i < d/2` the pair `(x[i], x[i + d/2])` is rotated by `θ_i = pos · base^(-2i/d)`,

```
x[i]       ← x[i]·cos θ_i − x[i+d/2]·sin θ_i
x[i + d/2] ← x[i]·sin θ_i + x[i+d/2]·cos θ_i
```

with `base = qwen2.rope.freq_base` (1,000,000 for Qwen2.5). Applying it to
`q` and `k` makes `q·k` depend only on the relative distance between positions.

**Grouped-query attention.** 12 query heads share 2 KV heads in
Qwen2.5-1.5B: query head `h` uses KV head `h / (n_heads / n_kv_heads)`, which
cuts the KV cache 6x (56 KiB per token instead of 336 KiB). Scores are
`q·k / √d` with a causal mask; the softmax is computed in one pass
("online softmax", the same rescaling trick FlashAttention uses):
`m' = max(m, s)`, `l' = l·e^(m−m') + e^(s−m')`, `acc' = acc·e^(m−m') + e^(s−m')·v`.

**SwiGLU MLP.** `down(SiLU(gate·x) ⊙ up·x)`, `SiLU(z) = z / (1 + e^−z)`.

**Block quantisation.** 32 weights share one fp16 scale. Q8_0 stores
`round(x/d)` in int8 (8.5 bits/weight); Q4_0 stores 4-bit levels `q` with
`x ≈ d·(q − 8)` (4.5 bits/weight). For a quantised matmul the activation
vector is itself quantised to Q8_0, so each 32-element block becomes one
integer dot product and one float multiply:

```
y[r] = Σ_b  d_w[r,b] · d_x[b] · Σ_{j<32} q_w[r,b,j] · q_x[b,j]
```

On M1 the inner sum is two `SDOT` instructions per block; on x86 it is
`maddubs`+`madd` with the sign trick. Decoding is memory-bound - every
weight is read once per token - so the format's bytes/weight sets the speed
limit: for Qwen2.5-1.5B (~1.5 B weights) that is ~1.65 GB/token in Q8_0 and
~0.87 GB/token in Q4_0. At the M1's ~68 GB/s that is a ceiling of roughly
40 tok/s (Q8_0) / 75 tok/s (Q4_0) - arithmetic, not a measurement.

**Tokenizer.** Byte-level BPE: specials split out first, then a hand-written
matcher for Qwen2's pre-tokenisation regex, GPT-2's byte→unicode map, and
repeated lowest-rank merges. Details, including the known approximation of
Unicode categories, are in [docs/internals.md](docs/internals.md).

**Sampling.** Repetition penalty (CTRL-style, over the last N tokens) →
temperature → top-k (O(V log k) heap) → top-p (nucleus; without top-k,
tokens with `p < (1−top_p)/(V−1)` are dropped before sorting since they can
never be in the nucleus) → draw with a seeded xorshift64* RNG.

**Threading.** Matmul rows are split over a persistent thread pool
(pthreads, or Win32 threads on Windows), and prompt tokens are processed in
chunks of 64 so weights are reused while they are in cache. See
[docs/internals.md](docs/internals.md) for the pool's synchronisation design.

## Build and run

Requirements: a C11 compiler. Python 3 + NumPy only for tests/scripts.

**macOS (Apple Silicon)**

```bash
make                 # build/tinyinfer, tinyinfer-quantize, test_core, bench_kernels, libtinyinfer.dylib
make metal           # build/tinyinfer-metal (needs Xcode command line tools; shaders compile at start-up)
make test            # C tests + Python tests
```

**Linux**: same `make` / `make test` (uses `-march=native`; `make ARCH=` for a
portable scalar build).

**Windows / anywhere without make**: `build.py` finds `$CC`, `zig cc`
(`pip install ziglang`) or `cc/gcc/clang`:

```bash
pip install -r requirements-dev.txt ziglang
python build.py                 # native SIMD
python build.py --portable      # scalar baseline
python build.py --check-neon    # compile the core for apple_m1 / cortex-a72 with zig (no link)
python build.py --target aarch64-linux-musl --mcpu cortex_a76 --only test_core   # cross build
```

**Get a model and run**

```bash
scripts/download_qwen.sh            # Qwen2.5-1.5B-Instruct Q8_0 -> models/ (+ pure Q4_0 copy)
build/tinyinfer -m models/qwen2.5-1.5b-instruct-q8_0.gguf -p "The capital of France is" -n 64 --temp 0
build/tinyinfer -m models/qwen2.5-1.5b-instruct-q8_0.gguf --chat -p "Explain RoPE briefly." -n 256
build/tinyinfer-metal -m models/qwen2.5-1.5b-instruct-q8_0.gguf --metal --chat -p "Hi" -n 64
```

The CLI prints the generated text on stdout and timing on stderr:

```
[tinyinfer] prefill: N tokens in X ms (Y tok/s)
[tinyinfer] decode:  M tokens in X ms (Y tok/s)
```

Options: `-n` tokens, `-t` threads, `-c` context, `--temp`, `--top-k`,
`--top-p`, `--repeat-penalty`, `--repeat-last-n`, `-s` seed, `--chat`,
`--system`, `--ignore-eos`, `--bench PP,TG`, `-v`.

No model? `python scripts/make_tiny_model.py -o build/tiny.gguf --type q4_0`
writes a random 2-layer Qwen2 with a real trained BPE tokenizer
(gibberish output, but it exercises everything). `--preset qwen2.5-1.5b`
writes random weights with the *real* shapes for speed tests (~1.6 GB file).

## Tests

```bash
python build.py && python -m pytest -q tests     # or: make test
build/test_core                                   # dependency-free C tests
```

What is tested (all against independent implementations, not against itself):

| area | how |
|---|---|
| fp16 conversion | all 65,536 half values round-trip; float→half rounding (ties, subnormals, overflow) equals NumPy |
| Q8_0 / Q4_0 | C bytes == NumPy reference bytes (bit-exact); per-element error bounds; RMS error |
| matmul | every weight type × several shapes × 1-64 tokens vs NumPy; 1 vs 4 threads bit-identical; 3,000 back-to-back tiny jobs (pool stress); SIMD vs scalar in C |
| tokenizer | C pre-tokeniser == Python port == **real Qwen2 regex** (via `regex`) on 21 samples + 400 random strings; C BPE == Python BPE; encode→decode round-trip on 200 random Unicode strings; special tokens; UTF-8 split across tokens |
| forward | full logits, C vs NumPy, for F32/F16/Q8_0/Q4_0, tied/untied, GQA/MHA, rope theta 1e4/1e6; KV-cache incremental decode == batch; >64 tokens (multiple prefill chunks); error paths |
| sampler | greedy; temperature distributions vs softmax (20k draws); top-k / top-p support sets; repetition penalty; seed determinism |
| end to end | CLI greedy output == NumPy greedy decode + Python detokeniser; `--bench`; chat + seeded sampling reproducible; `tinyinfer-quantize` output byte-identical to the Python quantiser |
| Metal | GPU logits vs float reference (skipped unless `build/tinyinfer-metal` exists) |

Last local run (Windows 10, Ryzen 5 3600, zig cc 0.17.1 (`pip install ziglang`) via `build.py`):
`168 passed, 5 skipped` (the 5 skipped are the Metal tests); `test_core`:
`112 passed, 0 failed`; the same suite also passes with `--portable` (scalar
kernels). CI (`.github/workflows/ci.yml`) runs the same suites on Linux AVX2 + scalar +
ASan/UBSan, Windows (zig), aarch64 under qemu (NEON/SDOT kernels and a full
forward pass vs NumPy) and a GitHub macOS arm64 runner (NEON natively:
`test_core` reports `simd path: neon-dotprod`, `112 passed`; pytest `168 passed`;
then `make metal` and the 5 Metal tests: `5 passed`).

## Results

### Correctness on the toy models (measured here)

Relative L2 error of the logits vs the F32 model with the same weights
(dim = 64, so each dot product averages very little noise; real models are
better): F16 ≈ 0.1 %, Q8_0 ≈ 3 %, Q4_0 ≈ 29 %. C vs NumPy reference with the
same format: max |Δ| / std(logits) < 1e-5 (F32/F16) and < 2e-2 (Q8_0/Q4_0,
the residual being rare Q8_0 activation-rounding flips).

### CPU speed on this dev machine (NOT an M1, NOT real weights)

Machine: AMD Ryzen 5 3600 (6 cores / 12 threads, Zen 2, AVX2), DDR4,
Windows 10, zig cc `-O2 -march=native`. Models: **random weights with the real
Qwen2.5 shapes** (`make_tiny_model.py --preset ...`; speed does not depend on
the weight values). `tinyinfer --bench PP,TG -t 6 -r 3` (mean of 3):

<p align="center"><img src="docs/assets/throughput.svg" alt="Measured CPU throughput on a Ryzen 5 3600, random-weight models with real Qwen2.5 shapes" width="88%"></p>

| shapes (random weights) | weights | file size | pp128 t/s | tg64 t/s |
|---|---|---|---|---|
| Qwen2.5-0.5B | Q8_0 | 529 MB | 154.0 | 40.9 |
| Qwen2.5-0.5B | Q4_0 | 282 MB | 156.6 | 68.5 |
| Qwen2.5-1.5B | Q8_0 | 1.64 GB | 58.2 | 13.5 |
| Qwen2.5-1.5B | Q4_0 | 0.87 GB | 51.8 | 22.5 |

(tg is measured from an empty KV cache, like llama-bench. For Q8_0 1.5B,
13.5 tok/s x 1.64 GB = ~22 GB/s of weight streaming on dual-channel DDR4.)

<details>
<summary><b>Kernel micro-benchmark</b> (<code>build/bench_kernels 6 50</code>, Qwen2.5-1.5B layer shapes)</summary>


| format | shape | decode ms | decode GB/s | prefill-64 GFLOP/s |
|---|---|---|---|---|
| F32 | 1536x1536 (q, o) | 0.053 | 176.8 | 111.5 |
| F16 | 1536x1536 (q, o) | 0.073 | 64.6 | 64.1 |
| Q8_0 | 1536x1536 (q, o) | 0.040 | 62.7 | 173.9 |
| Q4_0 | 1536x1536 (q, o) | 0.041 | 32.1 | 164.0 |
| F32 | 8960x1536 (gate, up) | 2.380 | 23.1 | 120.9 |
| F16 | 8960x1536 (gate, up) | 0.973 | 28.3 | 69.7 |
| Q8_0 | 8960x1536 (gate, up) | 0.221 | 66.2 | 189.3 |
| Q4_0 | 8960x1536 (gate, up) | 0.209 | 37.1 | 181.5 |
| Q8_0 | 1536x8960 (down) | 0.424 | 34.5 | 154.8 |
| Q4_0 | 1536x8960 (down) | 0.184 | 42.1 | 174.6 |

Caveat: a single matrix up to ~14 MB stays in the Ryzen's 32 MB L3 across
iterations, so these GB/s figures exceed DRAM bandwidth; the end-to-end
table above is the realistic one. Q4_0 moves about half the bytes of Q8_0
but spends extra instructions unpacking nibbles, so on cache-resident shapes
its latency is similar and its GB/s lower.

</details>

These are only here to show the engine works at real sizes; they say nothing
about the M1 or about llama.cpp.

### tinyinfer vs llama.cpp on M1 (⏳ pending an M1 run of `scripts/bench.sh`)

Qwen2.5-1.5B-Instruct, same GGUF file for both engines, pp512 / tg128,
M1 performance cores. Cells marked ⏳ have not been measured; nothing here is estimated.

| model | backend | test | tinyinfer t/s | llama.cpp t/s | ratio |
|---|---|---|---|---|---|
| Q8_0 | CPU (NEON) | pp512 | ⏳ | ⏳ | |
| Q8_0 | CPU (NEON) | tg128 | ⏳ | ⏳ | |
| Q4_0 (pure) | CPU (NEON) | pp512 | ⏳ | ⏳ | |
| Q4_0 (pure) | CPU (NEON) | tg128 | ⏳ | ⏳ | |
| Q8_0 | Metal | pp512 | ⏳ | ⏳ | |
| Q8_0 | Metal | tg128 | ⏳ | ⏳ | |
| Q4_0 (pure) | Metal | pp512 | ⏳ | ⏳ | |
| Q4_0 (pure) | Metal | tg128 | ⏳ | ⏳ | |

Expectation, so the numbers can be judged when they arrive: llama.cpp has
years of kernel tuning (register-blocked GEMM for prefill, simdgroup-matrix
Metal kernels, flash attention), so tinyinfer is expected to be competitive
on CPU decode (memory-bound, see the bandwidth ceiling above) and clearly
behind on prefill and on Metal.

## Hardware validation / TODO for the author

Items 1 and 2 now run on every push on a GitHub-hosted macOS arm64 runner
(an Apple-Silicon VM, toy models only); they still need a run on a real M1.
Items 3-5 have not been run. Commands are exact; paste the output into the tables.

1. **Build and test on the M1** (first native run of the NEON + SDOT kernels):

   ```bash
   make && make test
   build/test_core            # expect "... 0 failed", simd path: neon-dotprod
   ```

   | check | result |
   |---|---|
   | `make test`, GitHub macOS arm64 runner | ✅ 112 C tests + 168 pytest passed |
   | `test_core` simd path, GitHub macOS arm64 runner | ✅ `neon-dotprod` |
   | `make test` on M1 | ⏳ |

2. **Metal backend**:

   ```bash
   make metal
   TI_NO_AUTOBUILD=1 python3 -m pytest -q tests/test_cli.py -k metal   # GPU vs NumPy logits
   ```

   | check | result |
   |---|---|
   | `make metal` compiles (GitHub macOS arm64 runner) | ✅ |
   | shaders compile at start-up (runner) | ✅ |
   | 5 Metal tests pass (runner, paravirtual GPU) | ✅ `5 passed` |
   | 5 Metal tests pass on M1 | ⏳ |

3. **Real model correctness** vs Hugging Face transformers (needs
   `pip install torch transformers`, downloads ~3 GB of HF weights):

   ```bash
   scripts/download_qwen.sh                       # Q8_0 + pure Q4_0
   python3 scripts/validate_hf.py models/qwen2.5-1.5b-instruct-q8_0.gguf
   python3 scripts/validate_hf.py models/qwen2.5-1.5b-instruct-pure-q4_0.gguf
   ```

   | model | tokenizer identical | logits cosine | top-1 match | top-10 overlap |
   |---|---|---|---|---|
   | Q8_0 | | | | |
   | Q4_0 (pure) | | | | |

4. **Sample output** (greedy, for the README):

   ```bash
   build/tinyinfer -m models/qwen2.5-1.5b-instruct-q8_0.gguf --chat --temp 0 -n 128 \
       -p "Explain what a KV cache is in two sentences."
   ```

   <!-- paste output + the prefill/decode tok/s lines here -->

5. **Benchmarks vs llama.cpp** (build llama.cpp with Metal, put `llama-bench`
   on PATH or set `LLAMA_BENCH`):

   ```bash
   scripts/bench.sh models/qwen2.5-1.5b-instruct-q8_0.gguf models/qwen2.5-1.5b-instruct-pure-q4_0.gguf
   # writes bench_results.md -> copy into the table above
   ```

   <!-- screen recording of a chat session here -->

## Roadmap

- [x] NEON + Metal build and toy-model tests on Apple Silicon (GitHub macOS arm64 runner).
- [ ] Run everything in the hardware section on a real M1 with real weights.
- [ ] Prefill: register-blocked GEMM tiles (e.g. 4 rows × 4 tokens) instead of
      per-row dot products; Metal batched prefill with `simdgroup_matrix`.
- [ ] f16 KV cache (halves cache bandwidth at long context); KV cache on GPU in f16.
- [ ] Fused kernels on Metal (RMSNorm+matvec, QKV in one dispatch).
- [ ] More formats: Q4_K / Q6_K so the official Q4_K_M files load directly.
- [ ] Qwen3 (QK-norm, no QKV bias) and Llama (interleaved RoPE) architectures.
- [ ] Exact Unicode categories in the pre-tokeniser (generated tables).
- [ ] Speculative decoding with Qwen2.5-0.5B as the draft model.

## Repository layout

```
include/tinyinfer.h        public C API (also used from Python via ctypes)
src/                       engine: gguf, tokenizer, model, kernels, quant, sampler, platform, CLI
metal/                     Metal host (.m) + shaders (.metal)
python/tinyinfer_ref/      NumPy reference: GGUF I/O, quant formats, BPE, Qwen2 forward, ctypes API
scripts/                   make_tiny_model.py, download_qwen.sh, bench.sh, validate_hf.py
tests/                     pytest suite + test_core.c
tools/bench_kernels.c      matmul micro-benchmark
examples/                  generate.py (ctypes API), chat.sh
docs/internals.md          formats and design notes
```

## License

MIT - see [LICENSE](LICENSE).
