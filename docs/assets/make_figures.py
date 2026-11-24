#!/usr/bin/env python3
"""Regenerate the README images in docs/assets/ from real tinyinfer output.

  python build.py                      # the CLI must exist in build/
  python docs/assets/make_figures.py   # writes banner.svg, terminal.svg, throughput.svg

* banner.svg     - the right-hand motif is the actual Q4_0 4-bit levels of the
                   first 16 blocks of blk.0.attn_q.weight in the tiny test model
                   (scripts/make_tiny_model.py --seed 0), read back from the GGUF.
* terminal.svg   - stdout/stderr of real `tinyinfer` runs on that random-weight
                   model, captured when this script runs.
* throughput.svg - the Ryzen 5 3600 end-to-end numbers from the README table
                   (`tinyinfer --bench 128,64 -t 6 -r 3` on random-weight models
                   with the real Qwen2.5 shapes). Pass --measure to re-run those
                   benchmarks on your machine first (writes ~2.5 GB of models).

Only NumPy is needed (for the reference GGUF reader); the SVGs are written by hand.
banner.png is rasterised from banner.svg with docs/assets/render_png.mjs
(`npm i @resvg/resvg-js` somewhere, then `node render_png.mjs banner.svg banner.png`).
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python"))
from tinyinfer_ref.gguf import read_gguf  # noqa: E402

EXE = ".exe" if os.name == "nt" else ""
CLI = ROOT / "build" / f"tinyinfer{EXE}"
ACCENT, ACCENT2 = "#F97316", "#FDBA74"
BG, FG, MUTED, GRID = "#0d1117", "#e6edf3", "#8b949e", "#161b22"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

# Measured on the dev machine (AMD Ryzen 5 3600, Windows 10, zig cc -O2 -march=native),
# see README "CPU speed on this dev machine". (shape, weights) -> (pp128, tg64) tok/s.
RYZEN = [("Qwen2.5-0.5B", "Q8_0", 154.0, 40.9), ("Qwen2.5-0.5B", "Q4_0", 156.6, 68.5),
         ("Qwen2.5-1.5B", "Q8_0", 58.2, 13.5), ("Qwen2.5-1.5B", "Q4_0", 51.8, 22.5)]


def run(cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        sys.exit(f"command failed: {cmd}\n{r.stderr}")
    return r.stdout, r.stderr


def tiny_model(path, wtype="q4_0"):
    run([sys.executable, "scripts/make_tiny_model.py", "-o", str(path), "--type", wtype, "--seed", "0"])


# --------------------------------------------------------------------------- banner
def banner(model):
    t = read_gguf(model).tensors["blk.0.attn_q.weight"]
    blocks = np.frombuffer(t.data, dtype=np.uint8).reshape(-1, 18)[:16]  # fp16 d + 16 nibble bytes
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32).ravel()
    qs = blocks[:, 2:]
    levels = np.concatenate([qs & 0x0F, qs >> 4], axis=1).astype(int)  # elements 0..15, 16..31

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="400" viewBox="0 0 1280 400">',
         '<defs><pattern id="g" width="32" height="32" patternUnits="userSpaceOnUse">'
         f'<path d="M32 0H0V32" fill="none" stroke="{GRID}" stroke-width="1"/></pattern>'
         f'<linearGradient id="fade" x1="0" x2="1"><stop offset="0" stop-color="{BG}"/>'
         f'<stop offset="1" stop-color="{BG}" stop-opacity="0"/></linearGradient></defs>',
         f'<rect width="1280" height="400" fill="{BG}"/><rect width="1280" height="400" fill="url(#g)"/>',
         '<rect width="700" height="400" fill="url(#fade)"/>',
         f'<rect x="72" y="118" width="56" height="6" rx="3" fill="{ACCENT}"/>',
         f'<text x="68" y="210" font-family="{MONO}" font-size="92" font-weight="700" fill="{FG}">tinyinfer</text>',
         f'<text x="72" y="258" font-family="{MONO}" font-size="24" fill="{MUTED}">'
         'An LLM inference engine from scratch in C11</text>',
         f'<text x="72" y="306" font-family="{MONO}" font-size="17" fill="{MUTED}">'
         f'GGUF <tspan fill="{ACCENT}">·</tspan> BPE <tspan fill="{ACCENT}">·</tspan> Qwen2 '
         f'<tspan fill="{ACCENT}">·</tspan> Q8_0/Q4_0 <tspan fill="{ACCENT}">·</tspan> NEON/AVX2 '
         f'<tspan fill="{ACCENT}">·</tspan> Metal</text>']
    x0, y0, c, gap = 728, 70, 13, 2
    dmax = float(np.abs(d).max())
    for r in range(16):
        y = y0 + r * (c + gap)
        w = 4 + 18 * abs(float(d[r])) / dmax  # fp16 scale of the block
        s.append(f'<rect x="{x0 - 8 - w:.1f}" y="{y}" width="{w:.1f}" height="{c}" rx="2" fill="{ACCENT2}" opacity="0.55"/>')
        for k in range(32):
            a = 0.08 + 0.92 * abs(levels[r, k] - 8) / 8.0
            s.append(f'<rect x="{x0 + k * (c + gap)}" y="{y}" width="{c}" height="{c}" rx="2" '
                     f'fill="{ACCENT}" opacity="{a:.2f}"/>')
    s.append(f'<text x="{x0}" y="{y0 + 16 * (c + gap) + 26}" font-family="{MONO}" font-size="14" fill="{MUTED}">'
             'blk.0.attn_q.weight · Q4_0 · 16 blocks × 32 levels</text>')
    s.append("</svg>")
    (OUT / "banner.svg").write_text("\n".join(s) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- terminal
def terminal_svg(lines, title, caption, path, cols=104):
    """lines: list of (text, colour). Long lines are wrapped at `cols` characters."""
    wrapped = []
    for text, col in lines:
        text = text.rstrip()
        while len(text) > cols:
            cut = text.rfind(" ", 0, cols)
            cut = cut if cut > cols // 2 else cols
            wrapped.append((text[:cut], col))
            text = "  " + text[cut:].lstrip()
        wrapped.append((text, col))
    lh, pad, top = 20, 22, 44
    w = int(cols * 8.45) + 2 * pad
    h = top + len(wrapped) * lh + 44
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
         f'<rect width="{w}" height="{h}" rx="10" fill="{BG}" stroke="#30363d"/>',
         f'<path d="M0 10a10 10 0 0 1 10-10h{w - 20}a10 10 0 0 1 10 10v20H0z" fill="{GRID}"/>',
         '<circle cx="20" cy="15" r="6" fill="#ff5f57"/><circle cx="40" cy="15" r="6" fill="#febc2e"/>'
         '<circle cx="60" cy="15" r="6" fill="#28c840"/>',
         f'<text x="{w / 2}" y="20" text-anchor="middle" font-family="{MONO}" font-size="13" fill="{MUTED}">{escape(title)}</text>',
         f'<g font-family="{MONO}" font-size="14" xml:space="preserve">']
    for i, (text, col) in enumerate(wrapped):
        y = top + 14 + i * lh
        if text.startswith("$ "):
            s.append(f'<text x="{pad}" y="{y}" fill="{col}"><tspan fill="{ACCENT}">$</tspan> {escape(text[2:])}</text>')
        else:
            s.append(f'<text x="{pad}" y="{y}" fill="{col}">{escape(text)}</text>')
    s.append("</g>")
    s.append(f'<text x="{pad}" y="{h - 16}" font-family="{MONO}" font-size="12" fill="{MUTED}">{escape(caption)}</text>')
    s.append("</svg>")
    path.write_text("\n".join(s) + "\n", encoding="utf-8")


def terminal(model):
    rel = "build/tiny.gguf"
    lines = [("$ python scripts/make_tiny_model.py -o build/tiny.gguf --type q4_0 --seed 0", FG),
             (f"wrote {rel}", MUTED)]
    cmd = ["-m", rel, "-p", "The capital of France is", "-n", "32", "--temp", "0", "-t", "6", "-v"]
    out, err = run([str(CLI), *cmd])
    lines.append(("$ build/tinyinfer " + " ".join(f'"{a}"' if " " in a else a for a in cmd), FG))
    for ln in err.splitlines():
        if ln.startswith("[tinyinfer] qwen2"):
            lines.append((ln, MUTED))
    lines.append((out.strip(), FG))
    for ln in err.splitlines():
        if "prefill" in ln or "decode" in ln:
            lines.append((ln, ACCENT2))
    cmd = ["-m", rel, "--bench", "128,64", "-t", "6", "-r", "3"]
    out, err = run([str(CLI), *cmd])
    lines.append(("$ build/tinyinfer " + " ".join(cmd), FG))
    for ln in (out + err).splitlines():
        if ln.startswith("|") and not ln.startswith("|---"):
            lines.append((ln, FG if "pp128" in ln or "tg64" in ln else MUTED))
    terminal_svg(lines, "tinyinfer - random-weight 2-layer Qwen2 test model",
                 "Real output, captured by docs/assets/make_figures.py. Random weights: gibberish text, toy-size tok/s.", OUT / "terminal.svg")


# --------------------------------------------------------------------------- throughput chart
def measure():
    rows = []
    for preset in ("qwen2.5-0.5b", "qwen2.5-1.5b"):
        for wt in ("q8_0", "q4_0"):
            p = ROOT / "models" / f"random-{preset}-{wt}.gguf"
            p.parent.mkdir(exist_ok=True)
            if not p.exists():
                run([sys.executable, "scripts/make_tiny_model.py", "-o", str(p), "--type", wt, "--preset", preset])
            out, err = run([str(CLI), "-m", str(p), "--bench", "128,64", "-t", "6", "-r", "3"])
            m = dict(re.findall(r"\| (pp128|tg64) \| ([\d.]+) \|", out + err))
            rows.append(("Qwen2.5-" + preset.split("-")[1].upper(), wt.upper(), float(m["pp128"]), float(m["tg64"])))
    return rows


def throughput(rows, machine):
    w, h = 900, 330
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
         f'<rect width="{w}" height="{h}" rx="10" fill="{BG}" stroke="#30363d"/>',
         f'<g font-family="{MONO}">',
         f'<text x="28" y="38" font-size="18" font-weight="700" fill="{FG}">End-to-end CPU throughput (tok/s, higher is better)</text>',
         f'<text x="28" y="62" font-size="13" fill="{MUTED}">{escape(machine)}</text>']
    panels = [("decode  tg64", 3, 80.0, 330), ("prefill  pp128", 2, 175.0, 640)]
    for title, idx, vmax, px in panels:
        s.append(f'<text x="{px - 130}" y="100" font-size="14" fill="{FG}">{title}</text>')
        for i, r in enumerate(rows):
            y = 116 + i * 44
            if px == 330:
                s.append(f'<text x="28" y="{y + 20}" font-size="13" fill="{FG}">{r[0]}</text>')
                s.append(f'<text x="150" y="{y + 20}" font-size="13" fill="{ACCENT if r[1] == "Q8_0" else ACCENT2}">{r[1]}</text>')
            bw = 230 * r[idx] / vmax
            col = ACCENT if r[1] == "Q8_0" else ACCENT2
            s.append(f'<rect x="{px - 130}" y="{y + 4}" width="{bw:.1f}" height="22" rx="3" fill="{col}"/>')
            s.append(f'<text x="{px - 124 + bw:.1f}" y="{y + 20}" font-size="13" fill="{FG}">{r[idx]:.1f}</text>')
    s.append(f'<text x="28" y="{h - 22}" font-size="12" fill="{MUTED}">Random weights with the real Qwen2.5 shapes '
             '(speed does not depend on weight values). Not an M1, not a llama.cpp comparison.</text>')
    s.append("</g></svg>")
    (OUT / "throughput.svg").write_text("\n".join(s) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true", help="re-run the throughput benchmarks here")
    a = ap.parse_args()
    if not CLI.exists():
        sys.exit("build the CLI first: python build.py")
    model = ROOT / "build" / "tiny.gguf"
    tiny_model(model)
    banner(model)
    terminal(model)
    if a.measure:
        throughput(measure(), "Measured by make_figures.py --measure, 6 threads, mean of 3")
    else:
        throughput(RYZEN, "AMD Ryzen 5 3600 (Zen 2, AVX2, DDR4), Windows 10, 6 threads, mean of 3 runs")
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.svg"))))


if __name__ == "__main__":
    main()
