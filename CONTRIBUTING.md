# Contributing to tinyinfer

Thanks for your interest! Bug reports, fixes and hardware test results are all welcome.

## Setup

```bash
pip install -r requirements-dev.txt ziglang   # zig only needed on Windows
python build.py            # or: make
python -m pytest -q tests  # + build/test_core
```

## Ground rules

- Numerical changes need a test against the NumPy reference in `python/tinyinfer_ref/` (not against the C code itself).
- Kernel changes: run `build/bench_kernels` before/after and paste the numbers in the PR.
- Keep it dependency-free C11; no new libraries in the engine.
- Never commit model files (`*.gguf`).
- One logical change per PR; describe what you tested and where.
- Follow the style of the surrounding code (`.editorconfig` covers whitespace).

## Reporting results

Numbers and boot logs from real hardware are especially valuable. Open an issue
with the exact command, the full output, and the machine details.
