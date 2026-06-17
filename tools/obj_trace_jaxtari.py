#!/usr/bin/env python3
"""obj_trace_jaxtari.py — per-scanline object-state trace for jaxtari.

The jaxtari twin of tools/obj_trace.jl. Boots a ROM exactly like the sweep
(60 NOOP + 4 RESET), steps to a target frame with the trace OFF, then traces
the ONE target frame and prints one CSV row per completed scanline:

  scanline,p0_x,p1_x,m0_x,m1_x,bl_x,grp0_old,grp1_old,m0_cosmic_line,p0_skip,p1_skip,grp0_live,grp1_live

Columns 0..8 match jutari's obj_trace.jl CSV exactly (diff those cross-port);
9..12 (skip-first + live GRP) are jaxtari-side extras for insight.

Usage:
    jaxtari/.venv/bin/python tools/obj_trace_jaxtari.py \\
        --rom tools/rom_sweep/roms/pitfall.bin \\
        --actions tools/breakout_video/output/breakout_random_actions.txt --frame 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[0].parent
sys.path.insert(0, str(_REPO / "jaxtari"))
sys.path.insert(0, str(_REPO / "tools"))

from jaxtari_dump import _settings_for_rom, _load_actions  # noqa: E402
from jaxtari.env.stella_environment import StellaEnvironment  # noqa: E402
from jaxtari.tia import system as tia_sys  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rom", type=Path, required=True)
    p.add_argument("--actions", type=Path, required=True)
    p.add_argument("--frame", type=int, default=0)
    args = p.parse_args(argv)

    rom = np.frombuffer(args.rom.read_bytes(), dtype=np.uint8)
    env = StellaEnvironment(rom, _settings_for_rom(args.rom))
    env.reset(boot_noop_steps=60, boot_reset_steps=4)
    actions = _load_actions(args.actions)

    def act(k):  # 0-based action index, clamped
        if not actions:
            return 0
        return actions[k] if k < len(actions) else actions[-1]

    # Run up to (but not into) the target frame with the trace OFF.
    for k in range(args.frame):
        env.step(int(act(k)))
    # Trace exactly the target frame.
    tia_sys._OBJ_TRACE_LOG.clear()
    tia_sys._OBJ_TRACE_ENABLED = True
    env.step(int(act(args.frame)))
    tia_sys._OBJ_TRACE_ENABLED = False

    print("scanline,p0_x,p1_x,m0_x,m1_x,bl_x,grp0_old,grp1_old,m0_cosmic_line,"
          "p0_skip,p1_skip,grp0_live,grp1_live")
    for r in tia_sys._OBJ_TRACE_LOG:
        print(",".join(str(v) for v in r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
