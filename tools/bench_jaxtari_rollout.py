#!/usr/bin/env python3
"""Benchmark + diagnose jaxtari's HARD-mode rollout speed.

WHY THIS TOOL EXISTS
====================
The HARD conformance path (`StellaEnvironment.step` → `console.run_until_frame`
→ `cpu.step`) is the ~205x bottleneck the project keeps hitting: a 30-frame RAM
dump or 60-frame screen dump for ONE ROM takes minutes. The original brief was
"wrap the HARD `cpu_step` in `jax.lax.scan` / `lax.while_loop` like the SOFT
path already does, and it'll get dramatically faster."

This tool was written to test that premise empirically. It does NOT add a
compiled HARD rollout, because **the HARD `cpu_step` is not jit-traceable as
written** (see the DIAGNOSIS section at the bottom of this docstring and the
dated entry in `bug_fix_log.md`). Instead it:

  1. Measures the eager HARD baseline (instr/s, frames/s, ms/frame) so any
     future speedup has a hard number to beat.
  2. Micro-benchmarks the two per-instruction cost sources the HARD interpreter
     pays — `int(jax_scalar)` host-syncs and eager small-array JAX op dispatch —
     against their NumPy equivalents, to show WHERE the time goes and quantify
     the headroom a NumPy-backend (or fully-traceable) rewrite would unlock.
  3. Measures the *compiled* instruction-stepping ceiling using the SOFT path's
     `soft_run_scan` (which IS `jax.jit` + `lax.scan`): one-time compile cost vs
     steady-state warm throughput, plus the JAX persistent-compilation-cache
     cold-vs-warm cross-process win.

Run:
    jaxtari/.venv/bin/python tools/bench_jaxtari_rollout.py \
        --rom xitari/roms/pong.bin --frames 10

    # persistent-cache cross-process demo (run twice; 2nd is warm):
    JAX_COMPILATION_CACHE_DIR=/tmp/jaxtari_cache \
        jaxtari/.venv/bin/python tools/bench_jaxtari_rollout.py \
        --rom xitari/roms/pong.bin --soft-only --soft-steps 1200

DIAGNOSIS (measured 2026-06-19, see bug_fix_log.md):
  * eager `console_step` ~= 463 instr/s (2.16 ms/instr); `run_until_frame`
    ~= 0.07 frames/s (14.7 s/frame on pong) — this is the bottleneck.
  * `cpu.step` is a CONCRETE-VALUE PYTHON INTERPRETER: it does `int(state.PC)`,
    `int(opcode)`, `OPCODES.get(opcode)` dict lookups, and `if mnemonic == ...`
    dispatch chains — ~98 `int(jax_array)` host-syncs per instruction. None of
    that survives `jax.jit` tracing (it branches on traced values). And
    `Console.bus.cart` is a MUTABLE Python `Cart` object (not a pytree), so a
    `Console` cannot even be passed through `jit`/`scan`. Verified: jitting
    `console_step` raises `TypeError: ... Cart ... not an abstract array`.
  * The SOFT path (`soft_run_scan`) shows the compiled ceiling, but it models
    only CPU + 128B RAM + a minimal RIOT timer — NO TIA rendering (no screen),
    NO frame-end detection, NO bank-switching. So it is NOT a drop-in for the
    HARD conformance rollout; it is used here only to measure the compiled
    instruction-stepping ceiling and the persistent-cache behaviour.
"""
from __future__ import annotations

import os

# Pin single-threaded BEFORE importing jax, mirroring tools/jaxtari_dump.py
# (the sweep worker) so numbers are apples-to-apples with the real sweep.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
_xla = os.environ.get("XLA_FLAGS", "")
if "xla_cpu_multi_thread_eigen" not in _xla:
    os.environ["XLA_FLAGS"] = (_xla + " --xla_cpu_multi_thread_eigen=false").strip()

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "jaxtari"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


# --------------------------------------------------------------------------- #
# Eager HARD baseline
# --------------------------------------------------------------------------- #
def bench_eager(rom_bytes: np.ndarray, n_instr: int, n_frames: int) -> dict:
    from jaxtari.console import (initial_console, console_reset,
                                 console_step, run_until_frame)
    c = initial_console(jnp.asarray(rom_bytes))
    c = console_reset(c)
    # warm a couple frames so we're past boot self-test sliver frames
    for _ in range(2):
        c = run_until_frame(c)

    # instruction-level throughput
    t0 = time.perf_counter()
    cc = c
    for _ in range(n_instr):
        cc = console_step(cc)
    _ = int(cc.cpu.PC)  # force the last value
    dt_instr = time.perf_counter() - t0

    # frame-level throughput
    t0 = time.perf_counter()
    for _ in range(n_frames):
        c = run_until_frame(c)
    _ = np.asarray(c.bus.ram)
    dt_frame = time.perf_counter() - t0

    return {
        "instr_per_s": n_instr / dt_instr,
        "us_per_instr": dt_instr / n_instr * 1e6,
        "frames_per_s": n_frames / dt_frame,
        "ms_per_frame": dt_frame / n_frames * 1000,
    }


# --------------------------------------------------------------------------- #
# Per-instruction cost decomposition (why eager is slow)
# --------------------------------------------------------------------------- #
def bench_micro(n: int = 20000) -> dict:
    xj = jnp.uint16(1234)
    xn = np.uint16(1234)
    aj, bj = jnp.uint8(5), jnp.uint8(7)
    an, bn = np.uint8(5), np.uint8(7)

    def timed(fn):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n * 1e6  # us/call

    return {
        "int_jax_us": timed(lambda: int(xj)),
        "int_numpy_us": timed(lambda: int(xn)),
        "add_jax_us": timed(lambda: aj + bj),
        "add_numpy_us": timed(lambda: an + bn),
    }


# --------------------------------------------------------------------------- #
# Compiled instruction-stepping ceiling via the SOFT path (jit + lax.scan)
# --------------------------------------------------------------------------- #
def bench_soft_scan(rom_bytes: np.ndarray, n_steps: int) -> dict:
    from jaxtari.diff.soft_state import initial_soft_cpu_state, initial_soft_bus
    from jaxtari.diff.soft_step import soft_run_scan

    n = len(rom_bytes)
    lo = int(rom_bytes[n - 4])
    hi = int(rom_bytes[n - 3])
    pc = lo | (hi << 8)
    s = initial_soft_cpu_state(pc=float(pc))
    b = initial_soft_bus(jnp.asarray(rom_bytes))

    # cold = trace + compile + run (the per-process cost a sweep worker pays)
    t0 = time.perf_counter()
    s2, b2 = soft_run_scan(s, b, n_steps)
    _ = int(s2.PC)
    _ = np.asarray(b2.ram)
    t_cold = time.perf_counter() - t0

    # warm = steady-state run (cached compilation reused in-process)
    t0 = time.perf_counter()
    s3, b3 = soft_run_scan(s, b, n_steps)
    _ = int(s3.PC)
    _ = np.asarray(b3.ram)
    t_warm = time.perf_counter() - t0

    return {
        "cold_s": t_cold,
        "warm_ms": t_warm * 1000,
        "warm_instr_per_s": n_steps / t_warm,
        "compile_s_est": t_cold - t_warm,
    }


def _fmt(d, keys):
    return "  ".join(f"{k}={d[k]:,.3f}" if isinstance(d[k], float) else f"{k}={d[k]}"
                     for k in keys)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rom", type=Path, required=True)
    p.add_argument("--instr", type=int, default=1000,
                   help="instructions for the eager instr/s probe")
    p.add_argument("--frames", type=int, default=5,
                   help="frames for the eager frames/s probe (each can take seconds!)")
    p.add_argument("--soft-steps", type=int, default=1200,
                   help="instructions for the compiled SOFT-scan ceiling probe")
    p.add_argument("--soft-only", action="store_true",
                   help="skip the (slow) eager probe; only run SOFT-scan "
                        "(use with JAX_COMPILATION_CACHE_DIR for the cache demo)")
    p.add_argument("--no-micro", action="store_true")
    args = p.parse_args(argv)

    rom = np.frombuffer(args.rom.read_bytes(), dtype=np.uint8)
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    print(f"# jaxtari rollout benchmark")
    print(f"# rom={args.rom.name} size={len(rom)} jax={jax.__version__} "
          f"x64={jax.config.read('jax_enable_x64')} "
          f"persistent_cache={cache_dir or '(off)'}")

    if not args.no_micro:
        m = bench_micro()
        print("\n[micro] per-call costs (us):")
        print(f"  int(jax)={m['int_jax_us']:.2f}  int(numpy)={m['int_numpy_us']:.2f}"
              f"  -> {m['int_jax_us']/m['int_numpy_us']:.0f}x")
        print(f"  add(jax)={m['add_jax_us']:.2f}  add(numpy)={m['add_numpy_us']:.2f}"
              f"  -> {m['add_jax_us']/m['add_numpy_us']:.0f}x")

    if not args.soft_only:
        print("\n[eager HARD] (the conformance path)")
        e = bench_eager(rom, args.instr, args.frames)
        print(f"  console_step:    {e['instr_per_s']:,.0f} instr/s "
              f"({e['us_per_instr']:.1f} us/instr)")
        print(f"  run_until_frame: {e['frames_per_s']:.3f} frames/s "
              f"({e['ms_per_frame']:.0f} ms/frame)")

    print("\n[compiled ceiling] SOFT-scan (jit + lax.scan; CPU+RAM only, "
          "NO screen/banks)")
    sft = bench_soft_scan(rom, args.soft_steps)
    print(f"  cold (compile+run): {sft['cold_s']:.2f} s  "
          f"(compile ~= {sft['compile_s_est']:.2f} s)")
    print(f"  warm (run only):    {sft['warm_ms']:.1f} ms  "
          f"-> {sft['warm_instr_per_s']:,.0f} instr/s")
    if not args.soft_only:
        ratio = sft['warm_instr_per_s'] / e['instr_per_s']
        print(f"  warm speedup vs eager instr/s: {ratio:,.0f}x")
    if cache_dir:
        print(f"\n  NOTE: with JAX_COMPILATION_CACHE_DIR set, the 'cold' number "
              f"above\n  drops to ~warm on the SECOND process invocation "
              f"(cross-process\n  kernel reuse). Run this command twice to see it.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
