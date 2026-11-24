#!/usr/bin/env bash
# Benchmark tinyinfer against llama.cpp's llama-bench with matching settings
# and write a markdown table (default: bench_results.md).
#
#   scripts/bench.sh models/qwen2.5-1.5b-instruct-q8_0.gguf models/qwen2.5-1.5b-instruct-pure-q4_0.gguf
#
# Environment:
#   PP=512 TG=128 REPS=5    prompt / generation lengths and repetitions
#   THREADS                 CPU threads (default: performance cores on macOS)
#   LLAMA_BENCH             path to llama-bench (default: from PATH)
#   TINYINFER               default build/tinyinfer
#   TINYINFER_METAL         default build/tinyinfer-metal (skipped if missing)
#   OUT                     output markdown file
#
# Both engines run the same GGUF file. Semantics match llama-bench:
#   ppN = N-token prompt processed from an empty KV cache
#   tgN = N sequential single-token decode steps from an empty KV cache
set -euo pipefail

PP="${PP:-512}"; TG="${TG:-128}"; REPS="${REPS:-5}"
TINYINFER="${TINYINFER:-build/tinyinfer}"
TINYINFER_METAL="${TINYINFER_METAL:-build/tinyinfer-metal}"
LLAMA_BENCH="${LLAMA_BENCH:-$(command -v llama-bench || true)}"
OUT="${OUT:-bench_results.md}"
if [[ -z "${THREADS:-}" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    THREADS="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu)"
  else
    THREADS="$(nproc)"
  fi
fi
[[ $# -ge 1 ]] || { echo "usage: $0 model.gguf [model2.gguf ...]"; exit 1; }
[[ -x "$TINYINFER" ]] || { echo "build tinyinfer first (make)"; exit 1; }

# mean t/s for a test name (ppN / tgN) from tinyinfer's markdown output
ti_rate() { grep "| $2 |" <<<"$1" | awk -F'|' '{gsub(/ /,"",$5); print $5}'; }

# mean t/s from llama-bench JSON output (keys n_prompt, n_gen, avg_ts; checked
# against llama.cpp builds from 2024-2025 - adjust if your version differs)
lb_rate() {
  python3 - "$1" "$2" <<'EOF'
import json, sys
data = json.loads(open(sys.argv[1]).read())
want = sys.argv[2]
for r in data:
    name = f"pp{r['n_prompt']}" if r.get("n_gen", 0) == 0 else f"tg{r['n_gen']}"
    if name == want:
        print(f"{r['avg_ts']:.2f} ± {r.get('stddev_ts', 0):.2f}")
EOF
}

CPU="$( (sysctl -n machdep.cpu.brand_string 2>/dev/null || grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) | xargs)"
{
  echo "## Benchmark: tinyinfer vs llama.cpp"
  echo
  echo "- machine: ${CPU:-unknown} ($(uname -sm)), threads=$THREADS"
  echo "- settings: pp$PP / tg$TG, $REPS repetitions, date $(date -u +%Y-%m-%d)"
  if [[ -n "$LLAMA_BENCH" ]]; then echo "- llama.cpp: $("$LLAMA_BENCH" --version 2>&1 | head -1 || echo unknown)"; fi
  echo
  echo "| model | backend | test | tinyinfer t/s | llama.cpp t/s | ratio |"
  echo "|---|---|---|---|---|---|"
} > "$OUT"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
for MODEL in "$@"; do
  NAME="$(basename "$MODEL" .gguf)"
  for BACKEND in cpu metal; do
    if [[ "$BACKEND" == metal ]]; then
      [[ -x "$TINYINFER_METAL" ]] || { echo "skip metal (no $TINYINFER_METAL)"; continue; }
      TI_OUT="$("$TINYINFER_METAL" -m "$MODEL" --metal --bench "$PP,$TG" -r "$REPS" -t "$THREADS")"
      NGL=99
    else
      TI_OUT="$("$TINYINFER" -m "$MODEL" --bench "$PP,$TG" -r "$REPS" -t "$THREADS")"
      NGL=0
    fi
    LB_JSON="$TMP/lb.json"; echo "[]" > "$LB_JSON"
    if [[ -n "$LLAMA_BENCH" ]]; then
      "$LLAMA_BENCH" -m "$MODEL" -p "$PP" -n "$TG" -r "$REPS" -t "$THREADS" -ngl "$NGL" -o json > "$LB_JSON" || echo "[]" > "$LB_JSON"
    fi
    for TEST in "pp$PP" "tg$TG"; do
      A="$(ti_rate "$TI_OUT" "$TEST")"
      B=""
      if [[ -n "$LLAMA_BENCH" ]]; then B="$(lb_rate "$LB_JSON" "$TEST" 2>/dev/null || true)"; fi
      R="-"
      if [[ -n "$A" && -n "$B" ]]; then R="$(python3 -c "print(f'{$A/${B%% *}:.2f}x')")"; fi
      echo "| $NAME | $BACKEND | $TEST | ${A:-n/a} | ${B:-n/a} | $R |" | tee -a "$OUT"
    done
  done
done
echo
echo "wrote $OUT"
