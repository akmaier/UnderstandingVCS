#!/usr/bin/env python3
"""64-ROM jutari SCREEN-conformance sweep (per-frame framebuffer diff vs xitari).

The render-side companion to `sweep_jutari_ram.py`. For each ROM it runs the
SAME boot + action stream through both emulators and diffs the rendered
210x160 palette-index framebuffer frame-by-frame:

  - xitari:  tools/trace_dump --screen          (ground truth; ALE auto-applies
             each game's getStartingActions)
  - jutari:  tools/jutari_screen_dump.jl        (full per-game RomSettings map —
             MUST match jutari_trace_dump.jl so the boot is identical)

Reports, per ROM: max px diff in any frame, total px over the run, the first
diverging frame, and the scanline-row band of the worst frame (to localize).
Writes a markdown scoreboard to tools/rom_sweep/results_jutari_screen.md.

Action stream + frame count mirror the RAM sweep paradigm (breakout_random_
actions, a fixed N) so a divergence here is a genuine RENDER delta, not a
boot/settings mismatch. This is a FIRST-PASS depth (N frames); games that
diverge get deeper per-game runs (see PROBING_PLAN_RENDER.md).

Usage:
    jaxtari/.venv/bin/python tools/rom_sweep/sweep_jutari_screen.py [--jobs N] [--frames N] [--games ...]
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROMS = REPO / "tools" / "rom_sweep" / "roms"
TRACE_DUMP = REPO / "tools" / "trace_dump"
SCREEN_DUMP = REPO / "tools" / "jutari_screen_dump.jl"
ACTIONS = REPO / "tools" / "breakout_video" / "output" / "breakout_random_actions.txt"
RESULTS = REPO / "tools" / "rom_sweep" / "results_jutari_screen.md"
H, W = 210, 160
FRAMES = 60


def _tmp(name: str, rom: Path) -> Path:
    return Path(f"/tmp/_screensweep_{os.getpid()}_{rom.stem}_{name}")


def _xitari_screens(rom: Path, acts: Path, n: int):
    r = subprocess.run(
        [str(TRACE_DUMP), "--rom", str(rom), "--actions", str(acts),
         "--max-frames", str(n), "--screen"],
        capture_output=True, text=True, check=True, timeout=600)
    out = []
    for line in r.stdout.strip().splitlines():
        o = json.loads(line)
        if o.get("boot_end"):
            continue
        out.append(np.frombuffer(bytes.fromhex(o["screen"]), dtype=np.uint8)
                   .reshape(o["h"], o["w"]))
    return out


def _jutari_screens(rom: Path, acts: Path, n: int):
    outp = _tmp("ju.bin", rom)
    subprocess.run(
        ["julia", "--project=" + str(REPO / "jutari"), str(SCREEN_DUMP),
         "--rom", str(rom), "--actions", str(acts), "--out", str(outp),
         "--max-frames", str(n)],
        check=True, capture_output=True, timeout=900)
    b = np.fromfile(outp, dtype=np.uint8)
    outp.unlink(missing_ok=True)
    nf = len(b) // (H * W)
    return [b[i * H * W:(i + 1) * H * W].reshape(H, W) for i in range(nf)]


def _diff_one(rom: Path, n: int):
    """Return (max_px, total_px, first_div_frame_or_-1, worst_rows_str)."""
    xi = _xitari_screens(rom, ACTIONS, n)
    ju = _jutari_screens(rom, ACTIONS, n)
    m = min(len(xi), len(ju))
    if m == 0:
        return (-1, -1, -1, "NO FRAMES")
    max_px = total = 0
    first_div = -1
    worst_rows = "—"
    for i in range(m):
        if xi[i].shape != ju[i].shape:
            # PAL games: xitari renders the PAL screen height (e.g. 250) while
            # jutari is NTSC-only (210). Not comparable pixel-wise — flag it.
            return (-2, -2, -1, f"PAL? xitari {xi[i].shape[0]}h vs jutari {ju[i].shape[0]}h")
        d = xi[i] != ju[i]
        nd = int(d.sum())
        total += nd
        if nd and first_div < 0:
            first_div = i + 1
        if nd > max_px:
            max_px = nd
            rows = np.where(d.any(axis=1))[0]
            worst_rows = f"{rows.min()}-{rows.max()}"
    return (max_px, total, first_div, worst_rows)


def main():
    import argparse
    from concurrent.futures import ThreadPoolExecutor, as_completed
    p = argparse.ArgumentParser(description="64-ROM jutari SCREEN-conformance sweep.")
    default_jobs = max(1, (os.cpu_count() or 4) // 2)
    p.add_argument("--jobs", type=int, default=default_jobs)
    p.add_argument("--frames", type=int, default=FRAMES)
    p.add_argument("--games", nargs="*", default=None)
    args = p.parse_args()

    roms = sorted(ROMS.glob("*.bin"))
    if args.games:
        want = set(args.games)
        roms = [r for r in roms if r.stem in want]
    print(f"screen sweep: {len(roms)} ROMs, {args.jobs} workers, {args.frames} frames",
          flush=True)
    results = {}

    def work(rom):
        t0 = time.time()
        try:
            res = _diff_one(rom, args.frames)
        except subprocess.TimeoutExpired:
            res = (-3, -3, -3, "TIMEOUT")
        except Exception as e:  # noqa: BLE001
            res = (-4, -4, -4, f"ERR {type(e).__name__}: {str(e)[:60]}")
        return rom.stem, res, time.time() - t0

    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(work, r) for r in roms]
        for fut in as_completed(futs):
            name, res, dt = fut.result()
            results[name] = res
            done += 1
            mx, tot, fd, rows = res
            tag = "OK 0px ✅" if mx == 0 else f"maxpx={mx}"
            print(f"[{done}/{len(roms)}] {name}: {tag} total={tot} "
                  f"first_div={fd} rows={rows} ({dt:.0f}s)", flush=True)

    n_exact = sum(1 for v in results.values() if v[0] == 0)
    lines = [
        "# ROM sweep — jutari SCREEN (framebuffer) bit-exactness vs xitari",
        "",
        f"Per-frame 210x160 palette-index diff, jutari `jutari_screen_dump.jl` vs "
        f"xitari `trace_dump --screen`, breakout_random_actions stream, "
        f"first **{args.frames}** frames after the standard 60-NOOP+4-RESET boot. "
        f"Same per-game RomSettings as the RAM sweep (so a divergence is a genuine "
        f"render delta, not a settings/boot mismatch).",
        "",
        f"**Pixel-exact (0 px) over {args.frames} frames: {n_exact}/{len(results)}.**",
        "",
        "| game | max px/frame | total px | first div frame | worst-frame rows |",
        "|---|---|---|---|---|",
    ]
    for name in sorted(results):
        mx, tot, fd, rows = results[name]
        if mx < 0:
            # non-comparable (PAL height) or error — show the explanation.
            lines.append(f"| {name} | n/a | — | — | {rows} |")
            continue
        cell = "**0 ✅**" if mx == 0 else f"{mx}"
        fdc = "—" if fd < 0 else str(fd)
        lines.append(f"| {name} | {cell} | {tot} | {fdc} | {rows} |")
    RESULTS.write_text("\n".join(lines) + "\n")
    print(f"DONE. {n_exact}/{len(results)} pixel-exact. Results -> {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
