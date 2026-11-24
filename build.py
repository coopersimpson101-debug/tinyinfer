#!/usr/bin/env python3
"""Portable build script for tinyinfer (no make/cmake needed).

Finds a C compiler in this order: $CC, `python -m ziglang cc` (pip install
ziglang), then cc / gcc / clang on PATH. Builds into ./build:

  tinyinfer[.exe]            CLI
  tinyinfer-quantize[.exe]   GGUF re-quantiser
  test_core[.exe]            C unit tests
  bench_kernels[.exe]        matmul micro-benchmark
  libtinyinfer.{so,dll,dylib} shared library used by the Python tests

Examples:
  python build.py                      # native, -O2, SIMD for the host CPU
  python build.py --portable           # no -march=native (CI-safe binaries)
  python build.py --target aarch64-linux-musl --only test_core   # zig cross build
  python build.py --check-neon         # compile-only check of the NEON path
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
LIB_SRCS = ["platform.c", "quant.c", "kernels.c", "gguf.c", "tokenizer.c", "model.c", "sampler.c"]


def find_zig():
    candidates = [sys.executable]
    for venv in (ROOT / ".venv" / "Scripts" / "python.exe", ROOT / ".venv" / "bin" / "python"):
        if venv.exists():
            candidates.append(str(venv))
    for py in candidates:
        try:
            r = subprocess.run([py, "-m", "ziglang", "version"], capture_output=True, text=True)
        except OSError:
            continue
        if r.returncode == 0:
            return [py, "-m", "ziglang", "cc"]
    return None


def find_cc(prefer_zig=False):
    if os.environ.get("CC") and not prefer_zig:
        return os.environ["CC"].split()
    zig = find_zig()
    if prefer_zig or (zig and platform.system() == "Windows"):
        if zig:
            return zig
        sys.exit("ziglang not found: pip install ziglang")
    for name in ("cc", "gcc", "clang"):
        if shutil.which(name):
            return [name]
    if zig:
        return zig
    sys.exit("no C compiler found: set CC or `pip install ziglang`")


def run(cmd, retries=2):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    env = dict(os.environ)
    if "ziglang" in " ".join(str(c) for c in cmd):
        # private zig cache: avoids contention with other zig builds on the machine
        env.setdefault("ZIG_GLOBAL_CACHE_DIR", str(ROOT / "build" / ".zig-cache"))
        env.setdefault("ZIG_LOCAL_CACHE_DIR", str(ROOT / "build" / ".zig-cache"))
    for attempt in range(retries + 1):
        r = subprocess.run([str(c) for c in cmd], env=env)
        if r.returncode == 0:
            return
        if attempt < retries:
            print(f"(compiler failed, retrying {attempt + 1}/{retries})", flush=True)
    sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--portable", action="store_true", help="do not use -march=native")
    ap.add_argument("--target", help="zig target triple, e.g. aarch64-linux-musl")
    ap.add_argument("--mcpu", help="zig/clang -mcpu value, e.g. apple_m1")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--sanitize", action="store_true", help="-fsanitize=address,undefined (clang/gcc)")
    ap.add_argument("--only", nargs="*", help="subset of: tinyinfer quantize test_core bench lib")
    ap.add_argument("--out", default="build")
    ap.add_argument("--check-neon", action="store_true",
                    help="compile the core for aarch64 (apple_m1 cpu) with zig, no link")
    args = ap.parse_args()

    if args.check_neon:
        cc = find_cc(prefer_zig=True)
        tmp = ROOT / args.out / "neon_check"
        tmp.mkdir(parents=True, exist_ok=True)
        common = ["-std=c11", "-O2", "-Wall", "-Wextra", "-Wno-unused-parameter", "-Werror", "-c"]
        for s in LIB_SRCS + ["main.c"]:
            run(cc + ["-target", "aarch64-macos", "-mcpu=apple_m1"] + common +
                ["-o", tmp / (s + ".m1.o"), SRC / s])
        run(cc + ["-target", "aarch64-linux-gnu", "-mcpu=cortex_a72"] + common +
            ["-o", tmp / "kernels.a72.o", SRC / "kernels.c"])
        shutil.rmtree(tmp)
        print("NEON path compiles (dotprod via apple_m1, plain NEON via cortex_a72)")
        return

    cross = args.target is not None
    cc = find_cc(prefer_zig=cross)
    is_zig = "ziglang" in " ".join(cc)
    target_os = (args.target or sys.platform).lower()
    windows = "windows" in target_os or target_os.startswith("win")
    macos = "macos" in target_os or target_os == "darwin"

    out = ROOT / args.out
    out.mkdir(exist_ok=True)
    flags = ["-std=c11", "-Wall", "-Wextra", "-Wno-unused-parameter"]
    flags += ["-O0", "-g"] if args.debug else ["-O2"]
    if args.target:
        flags += ["-target", args.target]
    if args.mcpu:
        flags += [f"-mcpu={args.mcpu}"]
    elif not args.portable and not cross:
        flags += ["-march=native"]
    elif args.portable and is_zig:
        flags += ["-mcpu=baseline"]  # zig cc otherwise targets the host CPU
    if args.sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    libs = [] if windows else ["-lm", "-lpthread"]
    exe = ".exe" if windows else ""
    shlib = "tinyinfer.dll" if windows else ("libtinyinfer.dylib" if macos else "libtinyinfer.so")

    lib = [SRC / s for s in LIB_SRCS]
    want = set(args.only or ["tinyinfer", "quantize", "test_core", "bench", "lib"])
    if "lib" in want:
        pic = [] if windows else ["-fPIC"]
        run(cc + flags + pic + ["-DTI_BUILD_DLL", "-shared", "-o", out / shlib] + lib + libs)
    if "tinyinfer" in want:
        run(cc + flags + ["-o", out / f"tinyinfer{exe}"] + lib + [SRC / "main.c"] + libs)
    if "quantize" in want:
        run(cc + flags + ["-o", out / f"tinyinfer-quantize{exe}", SRC / "platform.c", SRC / "quant.c",
                          SRC / "gguf.c", SRC / "quantize_main.c"] + libs)
    if "test_core" in want:
        static = ["-static"] if (cross and "musl" in target_os and is_zig) else []
        run(cc + flags + static + ["-o", out / f"test_core{exe}"] + lib +
            [ROOT / "tests" / "test_core.c"] + libs)
    if "bench" in want:
        run(cc + flags + ["-o", out / f"bench_kernels{exe}"] + lib + [ROOT / "tools" / "bench_kernels.c"] + libs)
    for junk in out.glob("*.pdb"):
        junk.unlink()
    for junk in out.glob("*.lib"):
        junk.unlink()
    print(f"built into {out}")


if __name__ == "__main__":
    main()
