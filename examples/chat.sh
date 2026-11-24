#!/usr/bin/env bash
# Chat with Qwen2.5-1.5B-Instruct using tinyinfer (after scripts/download_qwen.sh).
#   examples/chat.sh "Explain rotary position embeddings in two sentences."
#   METAL=1 examples/chat.sh "..."      # GPU build (make metal)
set -euo pipefail
MODEL="${MODEL:-models/qwen2.5-1.5b-instruct-q8_0.gguf}"
PROMPT="${1:-Write a haiku about matrix multiplication.}"
if [[ "${METAL:-0}" == 1 ]]; then
  exec build/tinyinfer-metal -m "$MODEL" --metal --chat -p "$PROMPT" -n 256 --temp 0.7 -c 4096
fi
exec build/tinyinfer -m "$MODEL" --chat -p "$PROMPT" -n 256 --temp 0.7 -c 4096
