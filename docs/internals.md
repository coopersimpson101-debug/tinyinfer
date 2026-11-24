# tinyinfer internals

Notes on the file formats and design decisions, at the level of detail needed
to re-implement them. Source references are to `src/`.

## 1. GGUF v3 (`gguf.c`)

Why GGUF rather than safetensors: one self-contained file holds the
hyper-parameters, the full tokenizer (vocab, merges, token types, special ids)
and the weights already in block-quantised formats, and the same file can be
handed to `llama-bench`, which makes the benchmark comparison exact.

```
offset 0   u32 magic = 0x46554747 ("GGUF")   u32 version = 3
           u64 n_tensors                      u64 n_kv
           n_kv   x  { str key, u32 type, value }
           n_tensors x { str name, u32 n_dims, u64 ne[n_dims], u32 ggml_type, u64 offset }
           zero padding to general.alignment (default 32)
data:      tensor bytes, each tensor at data + offset (offsets are aligned)

str   = u64 length + UTF-8 bytes (no terminator)
array = u32 elem_type + u64 count + elements   (strings inside arrays are str)
value types: 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32, 7 bool, 8 str,
             9 array, 10 u64, 11 i64, 12 f64
```

`ne[0]` is the contiguous dimension, so a PyTorch `Linear(in, out)` weight of
shape `(out, in)` is stored with `ne = [in, out]`: `out` rows of `in` values.
The file is memory-mapped; tensor structs point into the mapping, so loading is
O(header) and pages fault in on first use (`load=` in `-v` output).

Metadata keys read for `general.architecture = "qwen2"`:

| key | meaning |
|---|---|
| `qwen2.block_count` | layers |
| `qwen2.embedding_length` | model dim |
| `qwen2.feed_forward_length` | SwiGLU hidden dim |
| `qwen2.attention.head_count` / `head_count_kv` | query heads / KV heads (GQA) |
| `qwen2.rope.freq_base` | RoPE theta (1e6 for Qwen2.5) |
| `qwen2.attention.layer_norm_rms_epsilon` | RMSNorm eps |
| `qwen2.context_length` | training context |
| `tokenizer.ggml.{tokens,merges,token_type,eos_token_id,bos_token_id}` | tokenizer |

Tensor names: `token_embd.weight`, `output_norm.weight`, optional
`output.weight` (absent = tied embeddings), and per layer `blk.N.` +
`attn_norm`, `attn_{q,k,v}.{weight,bias}`, `attn_output.weight`, `ffn_norm`,
`ffn_{gate,up,down}.weight`.

## 2. Block quantisation (`quant.c`, `kernels.c`)

Both formats use blocks of 32 consecutive weights of one row with one fp16
scale, bit-compatible with ggml:

| format | layout (bytes) | bits/weight | dequant |
|---|---|---|---|
| Q8_0 | `f16 d` + `i8 q[32]` (34) | 8.5 | `x = d * q` |
| Q4_0 | `f16 d` + `u8 nib[16]` (18) | 4.5 | `x = d * (q - 8)`, element j in the low nibble of byte j, element j+16 in the high nibble |

Quantisation:

* Q8_0: `d = max|x| / 127`, `q = round(x / d)`.
* Q4_0: let `m` be the element with the largest magnitude *with its sign*;
  `d = m / -8`, `q = min(15, trunc(x / d + 8.5))`. The extreme element lands
  exactly on level -8 (no error); the opposite extreme can be clipped to +7,
  costing at most one step. Measured error bounds are asserted in
  `tests/test_quant.py` (`|x - x'| <= d/2` for Q8_0 and for unclipped Q4_0
  elements, `<= |d|` for clipped ones, plus the fp16 rounding of `d`).

Quantised matmul. The activation vector is quantised to Q8_0 once per matmul
(per token), then each output is

```
y[r] = sum_b  d_w[r,b] * d_x[b] * sum_{j<32} q_w[r,b,j] * q_x[b,j]
```

The inner sum is pure integer arithmetic:

* NEON with `__ARM_FEATURE_DOTPROD` (M1): two `SDOT` (`vdotq_s32`) per block.
  Without dot-product: `vmull_s8` / `vmlal_s8` + pairwise adds.
* AVX2: `maddubs` needs one unsigned operand, so use `|w|` and `sign(x, w)`
  (`_mm256_sign_epi8`), then `madd` with ones to widen to int32.
* scalar fallback, always compiled and used as the oracle in `test_core`.

For prefill, `ti_dot_q_x4` loads (and for Q4_0 unpacks) a weight block once
and multiplies it with the same block of 4 tokens, keeping 4 accumulators.

## 3. Threading (`platform.c`)

A persistent pool (pthreads, or native Win32 threads + `CONDITION_VARIABLE`)
with the calling thread participating. Each `ti_parallel_for` publishes a job
by bumping an atomic generation counter; work is handed out in chunks from an
atomic counter and the caller waits for *items completed*, not for every
worker to check in, so a worker that the OS has descheduled cannot stall a
matmul. Workers spin ~20k pause iterations before sleeping on the condition
variable (decode issues ~200 small jobs per token, so wake-up latency matters).
The odd/even generation + `active` counter handshake prevents a late worker
from reading a half-written job descriptor. Matmul rows are split across
threads, so results are bit-identical for any thread count (tested).

## 4. Forward pass and KV cache (`model.c`)

KV cache: two f32 arrays `[n_layers][max_ctx][n_kv_heads * head_dim]`,
allocated once at load (`-c` sets `max_ctx`). For Qwen2.5-1.5B that is
`2 * 28 * 2 * 128 * 4 B = 56 KiB` per token, 112 MiB at 2048 tokens; GQA
(2 KV heads for 12 query heads) is what keeps it this small.

Prompt tokens are processed in chunks of up to 64 (`TI_MAX_BATCH`) so the
matmuls see a matrix instead of a vector. Attention uses a single-pass online
softmax per (token, head) pair - no score buffer, causal by construction
(token t only scans positions `<= pos + t`).

## 5. Tokenizer (`tokenizer.c`)

Byte-level BPE as in GPT-2 / Qwen2:

1. Special tokens (`token_type` 3 = control or 4 = user-defined, e.g.
   `<|im_start|>`) are matched greedily, longest first, and emitted verbatim.
2. The rest is pre-tokenised with a hand-written matcher for the Qwen2 regex
   ``(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+``
   including its backtracking behaviour (`\s+(?!\S)` leaves the last space for
   the next word; `\s*[\r\n]+` ends at the last newline of a whitespace run).
3. Each byte maps to a printable code point (GPT-2 `bytes_to_unicode`).
4. Merge loop: repeatedly merge the adjacent pair with the lowest rank in
   `tokenizer.ggml.merges` (hash map keyed by `"left right"`).

Unicode `\p{L}` / `\p{N}` are approximated with range tables (no ICU). The
test-suite checks the approximation against the real regex (Python `regex`
module) on English, code, whitespace edge cases, Latin, Greek, Cyrillic, CJK,
kana, Hangul, emoji and fullwidth forms, plus 400 random strings. Known
divergence: scripts that use combining marks (Devanagari, Thai, ...) - marks
are `\p{M}` in the regex but treated as letters here, so such words may be
split differently (decoding is still lossless).

## 6. Metal backend (`metal/`)

* Weights: the GGUF mapping is wrapped with `newBufferWithBytesNoCopy`
  (unified memory, zero copies); each tensor is `(buffer, offset)`.
* One command buffer and one serial compute encoder per token containing all
  ~20 dispatches per layer; one CPU wait per generated token.
* `matvec_*`: one SIMD-group (32 lanes) per output row, 8 rows per
  threadgroup, `simd_sum` reduction. Q4_0/Q8_0 are dequantised in registers.
* `rmsnorm`: one threadgroup, two-level `simd_sum` reduction.
* `rope_neox`, `kv_store`, `silu_mul`, `add_inplace`: element-wise.
* `attention`: one threadgroup per query head; scores into a device scratch
  row, `simd_max` / `simd_sum` reductions for the softmax, then the weighted
  sum of V.
* Activations stay f32 on the GPU (no Q8_0 activation quantisation), so the
  GPU result is compared with the float reference in `tests/test_cli.py`.
* Not done yet: batched prefill kernels (prompt tokens are run one by one),
  f16 KV cache, simdgroup-matrix kernels.
