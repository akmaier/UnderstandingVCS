#!/usr/bin/env python3
"""50+ ROM conformance sweep — jutari RAM bit-exactness vs xitari (NOOP from boot).

Same paradigm as PXC1/PXC2 (RIOT-RAM bit-exactness), applied to all 64
xitari-supported ALE games. jutari-first: this script runs ONLY the jutari arm
(xitari trace_dump reference + jutari, per-frame 128 B RAM diff). The jaxtari arm
+ screen pixel-exactness + videos are deferred to follow-on background runs so
the slow jaxtari path never blocks the jutari deliverable.

Output: tools/rom_sweep/results_jutari_ram.md (rewritten after every ROM, so the
partial table is always readable). Each ROM runs with a wall-clock timeout; a
hang/crash is recorded as ERROR/TIMEOUT and the sweep continues.

Run from anywhere; paths are repo-relative.
"""
from __future__ import annotations
import re, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROMS = REPO / "tools/rom_sweep/roms"
VENPY = REPO / "jaxtari/.venv/bin/python"
DIFF = REPO / "tools/jutari_xitari_ram_diff.py"
OUT = REPO / "tools/rom_sweep/results_jutari_ram.md"
NAMES = (REPO / "tools/rom_sweep/rom_names.txt").read_text().split()
FRAMES = 30
TIMEOUT = 420  # seconds per ROM

# jutari ports with real RomSettings (basename auto-detect in jutari_trace_dump.jl);
# the rest run generic — flagged so divergence can be triaged settings-vs-emulation.
JUTARI_SETTINGS = {"breakout", "pong", "pitfall", "enduro"}

rx_max = re.compile(r"max diff in any frame:\s*(\d+)")
rx_first = re.compile(r"first .*diverge.*?frame\s*(\d+)", re.I)

def run_one(game: str) -> dict:
    rom = ROMS / f"{game}.bin"
    if not rom.is_file():
        return {"game": game, "status": "NO_ROM", "maxdiff": None}
    t0 = time.time()
    try:
        r = subprocess.run(
            [str(VENPY), str(DIFF), "--rom", str(rom), "--max-frames", str(FRAMES)],
            cwd=str(REPO / "jaxtari"), capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"game": game, "status": "TIMEOUT", "maxdiff": None, "secs": time.time() - t0}
    out = r.stdout + "\n" + r.stderr
    m = rx_max.search(out)
    secs = time.time() - t0
    if m is None:
        tail = "; ".join(l.strip() for l in out.strip().splitlines()[-3:])[:200]
        return {"game": game, "status": "ERROR", "maxdiff": None, "secs": secs, "note": tail}
    maxd = int(m.group(1))
    fm = rx_first.search(out)
    return {"game": game, "status": "OK", "maxdiff": maxd,
            "first": (int(fm.group(1)) if fm else None), "secs": secs}

def write_table(rows: list[dict], done: int):
    bit_exact = sum(1 for r in rows if r.get("status") == "OK" and r.get("maxdiff") == 0)
    with open(OUT, "w") as f:
        f.write("# ROM sweep — jutari RAM bit-exactness vs xitari (NOOP, %d frames)\n\n" % FRAMES)
        f.write("Same paradigm as PXC1/PXC2: per-frame 128 B RIOT-RAM diff, jutari vs "
                "xitari trace_dump, NOOP from the standard ALE boot (60 NOOP + 4 RESET). "
                "`generic` = jutari ran with GenericRomSettings (no game-specific starting "
                "actions; divergence may be settings- not emulation-driven).\n\n")
        f.write(f"**Progress: {done}/{len(NAMES)} ROMs. Bit-exact (0 b/f): "
                f"{bit_exact}/{len([r for r in rows if r.get('status')=='OK'])} completed.**\n\n")
        f.write("| game | settings | max RAM diff (b/f) | first div frame | status | secs |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in rows:
            s = "real" if r["game"] in JUTARI_SETTINGS else "generic"
            md = "—" if r.get("maxdiff") is None else (
                "**0 ✅**" if r["maxdiff"] == 0 else str(r["maxdiff"]))
            fd = r.get("first")
            f.write(f"| {r['game']} | {s} | {md} | {fd if fd is not None else '—'} "
                    f"| {r['status']} | {r.get('secs', 0):.0f} |\n")

def main():
    rows = []
    for i, g in enumerate(NAMES):
        res = run_one(g)
        rows.append(res)
        write_table(rows, i + 1)
        print(f"[{i+1}/{len(NAMES)}] {g}: {res['status']} "
              f"maxdiff={res.get('maxdiff')} ({res.get('secs',0):.0f}s)", flush=True)
    print("DONE. Results -> %s" % OUT)

if __name__ == "__main__":
    main()
