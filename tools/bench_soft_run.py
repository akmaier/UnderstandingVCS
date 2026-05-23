#!/usr/bin/env python3
"""P9 — jaxtari throughput benchmark for SOFT-mode execution.

Loads a ROM, runs N SOFT-mode CPU instructions via `soft_run_scan`,
and reports throughput. Companion to `bench_soft_run.jl` (jutari side).

Usage
-----

    python tools/bench_soft_run.py \\
        --rom xitari/roms/pong.bin --steps 50000 --repeats 5

Prints one JSONL line per repeat to stdout plus a summary on stderr:

    {"port": "jaxtari", "rom": "pong.bin", "steps": 50000,
     "wall_s": 31.27, "steps_per_s": 1599.1, "repeat": 0}
    ...

Methodology
-----------
The first call to `soft_run_scan(N)` includes JIT compilation; the
reported `repeat=0` line therefore measures compile+run together and
`repeat=1..N-1` measure cached run. The summary takes the median over
the cached repeats — that's the throughput an XAI workflow actually
sees after warm-up.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jaxtari.diff import SoftBus, initial_soft_cpu_state, soft_run_scan


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rom", required=True, type=Path)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args(argv)

    rom_bytes = np.fromfile(args.rom, dtype=np.uint8)
    rom = jnp.asarray(rom_bytes, dtype=jnp.float32)

    rom_label = args.rom.name
    wall_times = []
    for repeat in range(args.repeats):
        bus = SoftBus(ram=jnp.zeros((128,), dtype=jnp.float32), rom=rom)
        state = initial_soft_cpu_state()
        t0 = time.perf_counter()
        state, bus = soft_run_scan(state, bus, args.steps)
        state.PC.block_until_ready()
        wall = time.perf_counter() - t0
        wall_times.append(wall)
        print(json.dumps({
            "port":         "jaxtari",
            "rom":          rom_label,
            "steps":        args.steps,
            "wall_s":       round(wall, 4),
            "steps_per_s":  round(args.steps / wall, 1),
            "repeat":       repeat,
        }))

    # Summary on stderr (cached runs only — repeat 0 includes JIT compile).
    if args.repeats > 1:
        cached = wall_times[1:]
        median_wall = statistics.median(cached)
        median_sps = args.steps / median_wall
        print(
            f"\n[jaxtari summary] {rom_label} × {args.steps} steps × "
            f"{len(cached)} cached repeats: "
            f"median {median_wall:.3f} s ({median_sps:,.0f} steps/s; "
            f"compile+first-run {wall_times[0]:.3f} s)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
