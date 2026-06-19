#!/usr/bin/env python3
"""Batched (vmapped) jaxtari SOFT throughput — N parallel envs via jax.vmap.

Companion to bench_soft_run.py (unbatched). Vectorises `soft_run_scan` over a
batch axis and reports aggregate throughput (env-instructions/s) and per-env
throughput, for N = 128 and 1024.

Caveat: jaxtari's opcode dispatch is a data-dependent 256-way `lax.switch`.
Under `vmap`, a batched switch index can force XLA to evaluate all branches and
select, so batching may not accelerate on the CPU backend (this is the GPU/SIMD
codesign CuLE does and we do not). The number reported here is the honest
measured result either way.

Usage:
  python tools/bench_soft_run_batched.py --rom xitari/roms/pong.bin \
      --batch 128 --steps 2000 --repeats 5
  python tools/bench_soft_run_batched.py --rom xitari/roms/pong.bin --verify
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

from jaxtari.diff import SoftBus, initial_soft_cpu_state, soft_run_scan


def _make_batch(rom, n):
    """N identical envs (same ROM/zeroed RAM) — throughput is data-independent."""
    st = initial_soft_cpu_state()
    bus = SoftBus(ram=jnp.zeros((128,), jnp.float32), rom=rom)
    rep = lambda x: jnp.broadcast_to(jnp.asarray(x), (n,) + jnp.shape(x))
    return tree_util.tree_map(rep, st), tree_util.tree_map(rep, bus)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rom", required=True, type=Path)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--verify", action="store_true",
                   help="correctness check: batched[i] PC == unbatched PC.")
    a = p.parse_args(argv)

    rom = jnp.asarray(np.fromfile(a.rom, dtype=np.uint8), dtype=jnp.float32)
    steps = a.steps if not a.verify else 50
    batched = jax.jit(jax.vmap(lambda s, b: soft_run_scan(s, b, steps),
                               in_axes=(0, 0)))

    if a.verify:
        n = 4
        sb, bb = _make_batch(rom, n)
        sB, _ = batched(sb, bb)
        sB.PC.block_until_ready()
        st = initial_soft_cpu_state()
        bus = SoftBus(ram=jnp.zeros((128,), jnp.float32), rom=rom)
        sU, _ = soft_run_scan(st, bus, steps)
        ok = bool(jnp.all(sB.PC == sU.PC))
        print(f"[verify] N={n}, steps={steps}: all envs' PC == unbatched PC "
              f"-> {ok}  (PC=${int(sU.PC):04X})")
        return 0 if ok else 1

    n = a.batch
    sb, bb = _make_batch(rom, n)
    walls = []
    for r in range(a.repeats):
        t0 = time.perf_counter()
        sB, _ = batched(sb, bb)
        sB.PC.block_until_ready()
        w = time.perf_counter() - t0
        walls.append(w)
        print(f"  repeat {r}: {w:.3f}s -> {n * steps / w:,.0f} env-instr/s")

    cached = walls[1:] if a.repeats > 1 else walls
    med = statistics.median(cached)
    print(f"\n[jaxtari batched N={n}] {n}x{steps} env-instr, median {med:.3f}s "
          f"-> {n * steps / med:,.0f} env-instr/s "
          f"(per-env {steps / med:,.0f}/s; compile+first {walls[0]:.2f}s)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
