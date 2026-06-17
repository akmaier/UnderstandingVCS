#!/usr/bin/env python3
"""64-ROM jaxtari conformance sweep — RAM + SCREEN, vs xitari, fully parallel.

The jaxtari analogue of `sweep_jutari_ram.py` + `sweep_jutari_screen.py`, fused
into ONE driver so the RAM jobs and SCREEN jobs share a single worker pool and
keep every core busy regardless of the job mix. jaxtari is the slow engine
(~3.5 s/eager-step), so parallelism across ROMs is the whole game here.

PARALLELISM / HARDWARE SATURATION
  - Each (rom, kind) job runs jaxtari in its OWN subprocess (`tools/jaxtari_dump.py`),
    pinned SINGLE-THREADED (OMP/BLAS=1, XLA Eigen multi-thread off — set both in
    the child env here AND inside jaxtari_dump.py). jaxtari's arrays are tiny, so
    one-core-per-job loses nothing per job and lets us run `--jobs` jobs at once.
  - Default `--jobs = os.cpu_count()` (one job per core) → N cores fully saturated
    by N single-threaded jaxtari processes, no oversubscription/thrash. The brief
    xitari `trace_dump` reference bursts (fast C++) interleave harmlessly.
  - 64 ROMs × {ram, screen} = up to 128 jobs streamed through the pool.

THREAD SAFETY
  - The pool is a ThreadPoolExecutor, but the parent threads only spawn/await
    subprocesses (GIL released during `subprocess.run`) and parse their output —
    no shared JAX state, since every jaxtari run is a separate process.
  - All shared mutable state (the two results dicts + the markdown writes) is
    guarded by a single lock; tables are rewritten after every completed job so a
    partial run is always readable.
  - Every subprocess writes to a UNIQUE temp file (`tempfile.mkstemp`), so
    concurrent jobs never clobber each other.

Usage:
    jaxtari/.venv/bin/python tools/rom_sweep/sweep_jaxtari.py \\
        [--jobs N] [--mode both|ram|screen] \\
        [--ram-frames 30] [--screen-frames 60] [--games pong breakout ...]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROMS_DIR = REPO / "tools" / "rom_sweep" / "roms"
NAMES = (REPO / "tools" / "rom_sweep" / "rom_names.txt").read_text().split()
TRACE_DUMP = REPO / "tools" / "trace_dump"
JAXTARI_DUMP = REPO / "tools" / "jaxtari_dump.py"
VENPY = REPO / "jaxtari" / ".venv" / "bin" / "python"
ACTIONS = REPO / "tools" / "breakout_video" / "output" / "breakout_random_actions.txt"
RAM_OUT = REPO / "tools" / "rom_sweep" / "results_jaxtari_ram.md"
SCREEN_OUT = REPO / "tools" / "rom_sweep" / "results_jaxtari_screen.md"
W = 160

# Per-job subprocess timeouts (s). jaxtari is slow; be generous so a healthy-but-
# slow job isn't killed, but a true hang is still bounded.
XITARI_TIMEOUT = 600
JAXTARI_TIMEOUT = 1800

# Force each jaxtari child single-threaded (override any inherited thread vars).
_CHILD_ENV = {
    **os.environ,
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    "XLA_FLAGS": (os.environ.get("XLA_FLAGS", "")
                  + " --xla_cpu_multi_thread_eigen=false").strip(),
}


def _xitari_frames(rom: Path, n: int, screen: bool):
    """xitari reference: per-frame RAM (128 B) or screen (h x 160) arrays."""
    cmd = [str(TRACE_DUMP), "--rom", str(rom), "--actions", str(ACTIONS),
           "--max-frames", str(n)]
    if screen:
        cmd.append("--screen")
    r = subprocess.run(cmd, capture_output=True, text=True, check=True,
                       timeout=XITARI_TIMEOUT)
    out = []
    for line in r.stdout.strip().splitlines():
        o = json.loads(line)
        if o.get("boot_end"):
            continue
        if screen:
            out.append(np.frombuffer(bytes.fromhex(o["screen"]), dtype=np.uint8)
                       .reshape(o["h"], o["w"]))
        else:
            out.append(np.frombuffer(bytes.fromhex(o["ram"]), dtype=np.uint8))
    return out


def _jaxtari_frames(rom: Path, n: int, screen: bool, h: int):
    """jaxtari (isolated subprocess): per-frame RAM (128 B) or screen (h x 160)."""
    fd, tmp = tempfile.mkstemp(prefix=f"jaxdump_{rom.stem}_", suffix=".bin")
    os.close(fd)
    tmp = Path(tmp)
    try:
        subprocess.run(
            [str(VENPY), str(JAXTARI_DUMP), "--rom", str(rom),
             "--actions", str(ACTIONS), "--max-frames", str(n),
             "--mode", "screen" if screen else "ram", "--out", str(tmp)],
            check=True, capture_output=True, text=True,
            timeout=JAXTARI_TIMEOUT, env=_CHILD_ENV)
        b = np.fromfile(tmp, dtype=np.uint8)
    finally:
        tmp.unlink(missing_ok=True)
    unit = (h * W) if screen else 128
    if len(b) == 0 or len(b) % unit != 0:
        return None
    nf = len(b) // unit
    if screen:
        return [b[i * unit:(i + 1) * unit].reshape(h, W) for i in range(nf)]
    return [b[i * unit:(i + 1) * unit] for i in range(nf)]


def _diff(rom: Path, n: int, screen: bool):
    """Return (max_diff, total, first_div_frame_or_-1, worst_band_str)."""
    xi = _xitari_frames(rom, n, screen)
    if not xi:
        return (-1, -1, -1, "NO XITARI FRAMES")
    h = xi[0].shape[0] if screen else 128
    jx = _jaxtari_frames(rom, n, screen, h)
    if jx is None:
        return (-2, -2, -1, "shape/height mismatch (PAL?)")
    m = min(len(xi), len(jx))
    if m == 0:
        return (-1, -1, -1, "NO FRAMES")
    max_d = total = 0
    first_div = -1
    worst = "—"
    for i in range(m):
        if xi[i].shape != jx[i].shape:
            return (-2, -2, -1, f"shape {xi[i].shape} vs {jx[i].shape}")
        d = xi[i] != jx[i]
        nd = int(d.sum())
        total += nd
        if nd and first_div < 0:
            first_div = i + 1
        if nd > max_d:
            max_d = nd
            if screen:
                rows = np.where(d.any(axis=1))[0]
                worst = f"rows {rows.min()}-{rows.max()}"
            else:
                offs = np.where(d)[0]
                worst = "$" + ",$".join(f"{o:02x}" for o in offs[:8].tolist())
    return (max_d, total, first_div, worst)


def _write_ram_table(results: dict, total_jobs: int):
    done = len(results)
    exact = sum(1 for v in results.values() if v[0] == 0)
    lines = [
        "# ROM sweep — jaxtari RAM bit-exactness vs xitari",
        "",
        "Per-frame 128 B RIOT-RAM diff, jaxtari (live `StellaEnvironment`, isolated "
        "subprocess) vs xitari `trace_dump`, breakout_random_actions stream from the "
        "standard 60-NOOP+4-RESET boot. Full 25-game RomSettings map (matches "
        "jutari). Generated by `tools/rom_sweep/sweep_jaxtari.py` (parallel).",
        "",
        f"**Bit-exact (0 b/f): {exact}/{done} completed.**",
        "",
        "| game | max RAM diff (b/f) | first div frame | worst bytes |",
        "|---|---|---|---|",
    ]
    for name in NAMES:
        if name not in results:
            continue
        mx, tot, fd, band = results[name]
        cell = "**0 ✅**" if mx == 0 else (band if mx < 0 else str(mx))
        fdc = "—" if fd < 0 else str(fd)
        lines.append(f"| {name} | {cell} | {fdc} | {band if mx > 0 else '—'} |")
    RAM_OUT.write_text("\n".join(lines) + "\n")


def _write_screen_table(results: dict, total_jobs: int):
    done = len(results)
    exact = sum(1 for v in results.values() if v[0] == 0)
    lines = [
        "# ROM sweep — jaxtari SCREEN (framebuffer) bit-exactness vs xitari",
        "",
        "Per-frame h×160 palette-index diff, jaxtari (live, isolated subprocess) vs "
        "xitari `trace_dump --screen`, breakout_random_actions stream after the "
        "standard 60-NOOP+4-RESET boot. Generated by "
        "`tools/rom_sweep/sweep_jaxtari.py` (parallel).",
        "",
        f"**Pixel-exact (0 px): {exact}/{done} completed.**",
        "",
        "| game | max px/frame | total px | first div frame | worst-frame rows |",
        "|---|---|---|---|---|",
    ]
    for name in NAMES:
        if name not in results:
            continue
        mx, tot, fd, band = results[name]
        cell = "**0 ✅**" if mx == 0 else (band if mx < 0 else str(mx))
        fdc = "—" if fd < 0 else str(fd)
        lines.append(f"| {name} | {cell} | {tot if mx >= 0 else '—'} | {fdc} | {band if mx > 0 else '—'} |")
    SCREEN_OUT.write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Parallel 64-ROM jaxtari RAM+SCREEN conformance sweep vs xitari.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs", type=int, default=(os.cpu_count() or 4),
                   help="parallel jobs (default = all cores; each jaxtari child "
                        "is single-threaded so this saturates the hardware)")
    p.add_argument("--mode", choices=("both", "ram", "screen"), default="both")
    p.add_argument("--ram-frames", type=int, default=30)
    p.add_argument("--screen-frames", type=int, default=60)
    p.add_argument("--games", nargs="*", default=None,
                   help="subset of game names (default: all in rom_names.txt)")
    args = p.parse_args(argv)

    if not VENPY.exists():
        print(f"FATAL: jaxtari venv python not found at {VENPY}", file=sys.stderr)
        return 2
    names = args.games if args.games else NAMES
    roms = [ROMS_DIR / f"{n}.bin" for n in names]
    missing = [n for n, r in zip(names, roms) if not r.is_file()]
    if missing:
        print(f"WARNING: {len(missing)} ROM(s) not resolved in {ROMS_DIR} "
              f"(run tools/rom_sweep/resolve_roms.py): {missing[:8]}"
              f"{'...' if len(missing) > 8 else ''}", file=sys.stderr)
    roms = [r for r in roms if r.is_file()]

    kinds = []
    if args.mode in ("both", "ram"):
        kinds.append(("ram", args.ram_frames, False))
    if args.mode in ("both", "screen"):
        kinds.append(("screen", args.screen_frames, True))
    jobs = [(r, kind, frames, screen) for r in roms for (kind, frames, screen) in kinds]

    ram_results: dict[str, tuple] = {}
    screen_results: dict[str, tuple] = {}
    lock = threading.Lock()
    print(f"jaxtari sweep: {len(roms)} ROMs, {len(jobs)} jobs ({args.mode}), "
          f"{args.jobs} parallel workers, {os.cpu_count()} cores", flush=True)

    def work(job):
        rom, kind, frames, screen = job
        t0 = time.time()
        try:
            res = _diff(rom, frames, screen)
        except subprocess.TimeoutExpired:
            res = (-3, -3, -3, "TIMEOUT")
        except subprocess.CalledProcessError as e:
            tail = (e.stderr or "")[-120:].replace("\n", " ")
            res = (-4, -4, -4, f"SUBPROC {tail}")
        except Exception as e:  # noqa: BLE001
            res = (-5, -5, -5, f"ERR {type(e).__name__}: {str(e)[:80]}")
        return rom.stem, kind, res, time.time() - t0

    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for fut in as_completed(futs):
            name, kind, res, dt = fut.result()
            with lock:
                if kind == "ram":
                    ram_results[name] = res
                    _write_ram_table(ram_results, len(jobs))
                else:
                    screen_results[name] = res
                    _write_screen_table(screen_results, len(jobs))
                done += 1
                mx = res[0]
                tag = "0 ✅" if mx == 0 else (res[3] if mx < 0 else f"diff={mx}")
                print(f"[{done}/{len(jobs)}] {name} {kind}: {tag} ({dt:.0f}s)",
                      flush=True)

    with lock:
        if ram_results:
            _write_ram_table(ram_results, len(jobs))
        if screen_results:
            _write_screen_table(screen_results, len(jobs))
    re = sum(1 for v in ram_results.values() if v[0] == 0)
    se = sum(1 for v in screen_results.values() if v[0] == 0)
    print(f"DONE. RAM bit-exact {re}/{len(ram_results)} -> {RAM_OUT.name}; "
          f"SCREEN pixel-exact {se}/{len(screen_results)} -> {SCREEN_OUT.name}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
