#!/usr/bin/env python3
"""bus_cycle_trace_jaxtari.py — per-bus-op trace dumper for jaxtari.

The jaxtari twin of `tools/cpu_tia_cycle_trace.jl`. Emits the SAME CSV
schema so `tools/cycle_trace_inspect.py diff` can pin the first bus op
whose (kind, scanline, scanline_cycle, color_clock, addr, value)
diverges between jutari and jaxtari — the beam-phase root-cause scalpel.

CSV columns (identical to the jutari tool):
  global_idx,frame,kind,scanline,scanline_cycle,color_clock,addr,value

`kind ∈ {peek, poke, tick, frame_boundary}`. Beam (scanline /
scanline_cycle / color_clock) is sampled from `bus.tia` at the moment
of the op — i.e. the instruction-start beam, BEFORE pending_tia_cycles
is folded in — exactly matching jutari's `_TRACE_LIVE_TIA[] = bus.tia`
convention. `addr` is the 13-bit-masked CPU address (addr & 0x1FFF),
again matching jutari. `value` is the byte read (peek) or written (poke).

Tracing spans the boot-burn (and the construction probe, if enabled),
so boot-phase beam drift is captured — jutari does the same.

Boot is IDENTICAL to the sweep (`tools/jaxtari_dump.py`):
StellaEnvironment(rom, settings).reset(boot_noop_steps, boot_reset_steps,
construction_probe).

Usage:
    jaxtari/.venv/bin/python tools/bus_cycle_trace_jaxtari.py \\
        --rom tools/rom_sweep/roms/demon_attack.bin \\
        --actions tools/breakout_video/output/breakout_random_actions.txt \\
        --max-frames 3 --out /tmp/demon_jaxtari_3.csv
"""
from __future__ import annotations

import os

# pin single-threaded BEFORE importing jax (mirror jaxtari_dump.py)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
_xla = os.environ.get("XLA_FLAGS", "")
if "xla_cpu_multi_thread_eigen" not in _xla:
    os.environ["XLA_FLAGS"] = (_xla + " --xla_cpu_multi_thread_eigen=false").strip()

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "jaxtari"))
sys.path.insert(0, str(_REPO / "tools"))

import numpy as np  # noqa: E402

from jaxtari.bus.system import Bus  # noqa: E402
import jaxtari.cpu.m6502 as m6502  # noqa: E402
from jaxtari.env.stella_environment import StellaEnvironment  # noqa: E402
from jaxtari_dump import _settings_for_rom, _load_actions  # noqa: E402

# Global trace buffer — list of (kind, sl, sc, cc, addr, value) tuples.
# Process-global, like jutari's _TRACE_BUFFER.
_BUF: list[tuple] = []
_TRACING = False

# Capture the originals m6502 imported.
_orig_peek = m6502.peek
_orig_poke = m6502.poke
_orig_tick = m6502.pending_tick


def _beam(world):
    """Sample instruction-start beam from bus.tia (matches jutari)."""
    tia = world.tia
    return int(tia.scanline), int(tia.scanline_cycle), int(tia.color_clock)


def _traced_peek(world, addr):
    if _TRACING and isinstance(world, Bus):
        sl, sc, cc = _beam(world)
        value, new_world = _orig_peek(world, addr)
        _BUF.append(("peek", sl, sc, cc, addr & 0x1FFF, value & 0xFF))
        return value, new_world
    return _orig_peek(world, addr)


def _traced_poke(world, addr, value):
    if _TRACING and isinstance(world, Bus):
        sl, sc, cc = _beam(world)
        _BUF.append(("poke", sl, sc, cc, addr & 0x1FFF, value & 0xFF))
    return _orig_poke(world, addr, value)


def _traced_tick(world):
    if _TRACING and isinstance(world, Bus):
        sl, sc, cc = _beam(world)
        _BUF.append(("tick", sl, sc, cc, 0x0000, 0x00))
    return _orig_tick(world)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rom", type=Path, required=True)
    p.add_argument("--actions", type=Path, required=True)
    p.add_argument("--max-frames", type=int, default=3)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--construction-probe", action="store_true",
                   help="run xitari's double-boot construction probe")
    args = p.parse_args(argv)

    global _TRACING

    # install the trace hooks on the names m6502 actually calls
    m6502.peek = _traced_peek
    m6502.poke = _traced_poke
    m6502.pending_tick = _traced_tick

    rom = np.frombuffer(args.rom.read_bytes(), dtype=np.uint8)
    settings = _settings_for_rom(args.rom)
    env = StellaEnvironment(rom, settings)

    actions = _load_actions(args.actions)
    n = min(args.max_frames, len(actions))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as io:
        io.write("global_idx,frame,kind,scanline,scanline_cycle,color_clock,addr,value\n")
        idx = 0

        def flush(frame: int):
            nonlocal idx
            for (kind, sl, sc, cc, addr, val) in _BUF:
                idx += 1
                io.write(f"{idx},{frame},{kind},{sl},{sc},{cc},{addr:x},{val}\n")
            _BUF.clear()
            idx += 1
            io.write(f"{idx},{frame},frame_boundary,0,0,0,0,0\n")

        # Trace across the boot-burn (frame 0), like jutari.
        _TRACING = True
        env.reset(boot_noop_steps=60, boot_reset_steps=4,
                  construction_probe=args.construction_probe)
        flush(0)

        for f in range(1, n + 1):
            a = actions[f - 1] if (f - 1) < len(actions) else 0
            env.step(int(a))
            flush(f)
        _TRACING = False

    sz = args.out.stat().st_size
    print(f"wrote {n} frame(s) of trace to {args.out} ({sz} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
