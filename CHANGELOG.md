# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Paged KV cache for the CPU backend (`src/kvcache.c`): fixed-size blocks, free-list allocator, per-sequence
  block tables (16 sequences), rollback by position, fork with reference counts and copy-on-write;
  attention reads through the block table and is bit-identical to the contiguous cache
- `ti_forward_seq`, `ti_kv_enable_paged`, `ti_kv_seq_trim`, `ti_kv_seq_copy`, `ti_kv_seq_len`,
  `ti_kv_blocks_free/total` in the public API
- Speculative decoding (`src/speculative.c`): draft model proposes k tokens, target verifies them in one
  batched forward, exact rejection-sampling acceptance rule; greedy output identical to the target alone
- `ti_sampler_probs`, `ti_sample_from`, `ti_spec_accept`, `ti_generate`, `ti_spec_generate`
- CLI flags `--draft`, `--spec-k`, `--spec-baseline`, `--kv-paged`, `--kv-block`, `--kv-blocks`; acceptance-rate
  and tokens-per-round reporting
- Tests: allocator unit tests in `test_core`, `tests/test_paged_kv.py`, `tests/test_speculative.py`
- Measured with the from-scratch tinylm 25.7M target + 3.15M draft on a Ryzen 5 3600 (see README)

## [0.1.0] - 2026-11-25

First public release of a dependency-free C11 LLM inference engine for Qwen2-family models.

### Added
- GGUF v3 loader, Qwen2 byte-level BPE tokenizer, full Qwen2 forward pass with a preallocated KV cache
- Q8_0 / Q4_0 quantisation, bit-compatible with the llama.cpp block format
- Matmul kernels for NEON (dot-product), AVX2 and scalar C; native thread pool
- Metal backend: builds and passes its GPU-vs-reference tests on the GitHub macOS arm64 runner (toy models)
- Measured on a Ryzen 5 3600 with random Qwen2.5-1.5B-shaped weights: 22.5 tok/s decode (Q4_0)

[0.1.0]: https://github.com/coopersimpson101-debug/tinyinfer/releases/tag/v0.1.0
