# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-11-25

First public release of a dependency-free C11 LLM inference engine for Qwen2-family models.

### Added
- GGUF v3 loader, Qwen2 byte-level BPE tokenizer, full Qwen2 forward pass with a preallocated KV cache
- Q8_0 / Q4_0 quantisation, bit-compatible with the llama.cpp block format
- Matmul kernels for NEON (dot-product), AVX2 and scalar C; native thread pool
- Metal backend: builds and passes its GPU-vs-reference tests on the GitHub macOS arm64 runner (toy models)
- Measured on a Ryzen 5 3600 with random Qwen2.5-1.5B-shaped weights: 22.5 tok/s decode (Q4_0)

[0.1.0]: https://github.com/coopersimpson101-debug/tinyinfer/releases/tag/v0.1.0
