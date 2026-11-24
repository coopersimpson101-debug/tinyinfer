#!/usr/bin/env bash
# Download a Qwen2.5 Instruct GGUF from Hugging Face and (optionally) make a
# "pure" Q4_0 copy with tinyinfer-quantize.
#
#   scripts/download_qwen.sh               # Qwen2.5-1.5B-Instruct, Q8_0 (~1.7 GB)
#   scripts/download_qwen.sh fp16          # F16 (~3.1 GB) - best source for re-quantising
#   SIZE=0.5b scripts/download_qwen.sh     # Qwen2.5-0.5B-Instruct (quick tests)
#
# Why 1.5B: the Qwen2 / Qwen2.5 family has 0.5B, 1.5B, 3B, 7B... there is no
# 2B Qwen2 text model, so 1.5B is the "Qwen ~2B" target. (Qwen3-1.7B needs
# QK-norm support, see the roadmap.)
#
# Note: the official *_q4_0.gguf files keep some tensors in other formats
# (e.g. Q6_K/Q4_1 for the embedding/output), which tinyinfer does not
# implement. Use the pure Q4_0 produced below; feed the SAME file to
# llama-bench so the comparison is apples to apples.
set -euo pipefail

QUANT="${1:-q8_0}"                 # q8_0 | fp16
SIZE="${SIZE:-1.5b}"               # 0.5b | 1.5b | 3b
DEST="${MODELS_DIR:-models}"
SIZE_UP="$(echo "$SIZE" | tr '[:lower:]' '[:upper:]')"   # bash 3.2 (macOS) safe
REPO="Qwen/Qwen2.5-${SIZE_UP}-Instruct-GGUF"
FILE="qwen2.5-${SIZE}-instruct-${QUANT}.gguf"
URL="https://huggingface.co/${REPO}/resolve/main/${FILE}"

mkdir -p "$DEST"
if [[ -f "$DEST/$FILE" ]]; then
  echo "already present: $DEST/$FILE"
else
  echo "downloading $URL"
  # -C - resumes partial downloads; large files may be split on HF for bigger
  # models (e.g. *-00001-of-00002.gguf) - 0.5B/1.5B/3B are single files.
  curl -L --fail -C - -o "$DEST/$FILE.part" "$URL"
  mv "$DEST/$FILE.part" "$DEST/$FILE"
fi

QTOOL="${QTOOL:-build/tinyinfer-quantize}"
OUT="$DEST/qwen2.5-${SIZE}-instruct-pure-q4_0.gguf"
if [[ -x "$QTOOL" && ! -f "$OUT" ]]; then
  echo "re-quantising $FILE -> $(basename "$OUT")"
  "$QTOOL" "$DEST/$FILE" "$OUT" q4_0
fi
echo "done:"
ls -lh "$DEST"/*.gguf
