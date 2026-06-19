#!/usr/bin/env python3
"""Probe the agent's joystick controller (P0/high nibble vs P1/low nibble) that
xitari actually drives, per game. jutari currently routes every agent action to
P0 (SWCHA high nibble); for games where xitari drives P1 (low nibble) the joystick
read diverges the instant gameplay reads it (past the attract window). For each
game we run xitari --bus-trace over a window around its known first-divergence
frame and report the first SWCHA ($280) read where a direction bit is cleared, and
in which nibble.

  jaxtari/.venv/bin/python tools/probe_agent_player.py
"""
from __future__ import annotations
import csv, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TD = REPO / "tools/trace_dump"
ROMS = REPO / "tools/rom_sweep/roms"
OUTDIR = REPO / "tools/comparison_videos/output"

# game -> first screen-divergence frame (from results_longhorizon.md)
GAMES = {
    "wizard_of_wor": 219, "road_runner": 765, "riverraid": 958,
    "montezuma_revenge": 867, "berzerk": 581, "asterix": 1160,
    "pooyan": 1605, "kangaroo": 1720, "phoenix": 1743,
    "pacman": 1771, "ms_pacman": 1786,
}


def probe(game: str, fdiv: int) -> str:
    rom = ROMS / f"{game}.bin"
    acts = OUTDIR / f"{game}_breakout_random_actions.txt"
    if not rom.exists() or not acts.exists():
        return "MISSING rom/stream"
    lo, hi = max(1, fdiv - 40), fdiv + 2
    bt = Path(tempfile.mktemp(suffix=f"_{game}_bt.csv"))
    try:
        subprocess.run([str(TD), "--rom", str(rom), "--actions", str(acts),
                        "--max-frames", str(hi + 2), "--bus-trace", str(bt),
                        "--bus-trace-frames", f"{lo},{hi}"],
                       capture_output=True, timeout=600, check=True)
        first = None
        with open(bt) as f:
            r = csv.reader(f); next(r, None)
            for x in r:
                # global_idx,frame,kind,sl,sc,cc,addr,value
                if x[2] != "peek":
                    continue
                a = int(x[6], 16) & 0x1FFF
                if (a & 0x1000) == 0 and (a & 0x80) and (a & 0x280) == 0x280 and (a & 0x7) == 0:
                    v = int(x[7])
                    if v != 0xFF:              # a direction is pressed
                        first = (int(x[1]), v)
                        break
        if first is None:
            return "no directional SWCHA read in window (non-joystick cause?)"
        fr, v = first
        hi_n, lo_n = (v >> 4) & 0xF, v & 0xF
        if hi_n != 0xF and lo_n == 0xF:
            who = "P0 (high nibble)"
        elif lo_n != 0xF and hi_n == 0xF:
            who = "P1 (low nibble)"
        else:
            who = "BOTH/odd"
        return f"frame {fr}: SWCHA=0x{v:02X} -> agent on {who}"
    except subprocess.CalledProcessError as e:
        return f"xitari error: {(e.stderr or b'')[-120:]!r}"
    finally:
        bt.unlink(missing_ok=True)


def main() -> int:
    for g, f in GAMES.items():
        print(f"{g:<20} {probe(g, f)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
