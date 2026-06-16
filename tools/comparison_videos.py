#!/usr/bin/env python3
"""Batch-render xitari-vs-port comparison MP4s for the full 64-ROM ALE set.

For every ROM in `tools/rom_sweep/roms/` this drives the per-ROM renderer
`tools/breakout_video/render_breakout_compare.py`, which produces a 3-panel
side-by-side video:

    | xitari (reference) | <port> | DIFFERENCE (magenta = differing pixels) |

Because jutari is bit-for-bit identical to xitari on all 64 games, the DIFFERENCE
panel is solid black for a correct port — that is the point of the videos.

The default port is **jutari** (xitari_vs_jutari.mp4 per game). Pass
`--port jaxtari` (or `--port both`) to also render jaxtari — but note jaxtari is
~200× slower per frame, so the jaxtari pass is much longer and is normally run
later / in the background.

Both emulators are driven by the SAME deterministic action stream and the SAME
per-game RomSettings + boot the renderer already uses for the conformance sweep,
so the comparison is apples-to-apples.

Outputs: `<out-dir>/<game>_xitari_vs_<port>.mp4` (default out-dir
`tools/comparison_videos/output/`). Large intermediate `*.raw` frame dumps are
deleted after each game unless `--keep-raw` is given.

Examples (run from the repo root):

    # all 64 games, xitari vs jutari, 10-second (600-frame) clips:
    jaxtari/.venv/bin/python tools/comparison_videos.py

    # longer 30 s clips, 4 games at a time:
    jaxtari/.venv/bin/python tools/comparison_videos.py --frames 1800 --jobs 4

    # just a few games:
    jaxtari/.venv/bin/python tools/comparison_videos.py --games elevator_action pong qbert

    # also (or only) jaxtari — slow:
    jaxtari/.venv/bin/python tools/comparison_videos.py --port jaxtari
    jaxtari/.venv/bin/python tools/comparison_videos.py --port both

`python3` works too; the jaxtari venv is only needed for `--port jaxtari/both`
(the renderer invokes `jaxtari/.venv/bin/python3` itself for the jaxtari dump).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROMS_DIR = REPO / "tools" / "rom_sweep" / "roms"
RENDERER = REPO / "tools" / "breakout_video" / "render_breakout_compare.py"
TRACE_DUMP = REPO / "tools" / "trace_dump"
DEFAULT_OUT = REPO / "tools" / "comparison_videos" / "output"


def all_roms() -> list[Path]:
    return sorted(ROMS_DIR.glob("*.bin"))


def _raw_prefix(stem: str) -> str:
    # Mirror render_breakout_compare._rom_prefix (breakout uses no prefix).
    return "" if stem == "breakout" else f"{stem}_"


def render_one(rom: Path, port: str, out_dir: Path, n_frames: int, seed: int,
               keep_raw: bool) -> tuple[str, bool, str]:
    """Render one game's comparison video(s). Returns (stem, ok, errtail)."""
    skip: list[str] = []
    if port == "jutari":
        skip = ["--skip-jaxtari"]
    elif port == "jaxtari":
        skip = ["--skip-jutari"]
    # port == "both" → render both (no skip)

    cmd = [
        sys.executable, str(RENDERER),
        "--rom", str(rom),
        "--out-dir", str(out_dir),
        "--n-frames", str(n_frames),
        "--seed", str(seed),
        *skip,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ok = proc.returncode == 0

    if not keep_raw:
        prefix = _raw_prefix(rom.stem)
        for raw in out_dir.glob(f"{prefix}*.raw"):
            raw.unlink(missing_ok=True)

    errtail = "" if ok else (proc.stderr or proc.stdout or "")[-1500:]
    return rom.stem, ok, errtail


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", choices=["jutari", "jaxtari", "both"],
                   default="jutari",
                   help="which port to compare against xitari (default jutari).")
    p.add_argument("--frames", type=int, default=600,
                   help="frames per video (60 fps; default 600 = 10 s).")
    p.add_argument("--games", nargs="*", default=None,
                   help="game stems to render (default: all 64). "
                        "e.g. --games pong elevator_action")
    p.add_argument("--jobs", type=int, default=1,
                   help="number of games to render in parallel (default 1).")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=42,
                   help="action-stream seed (shared by all engines).")
    p.add_argument("--keep-raw", action="store_true",
                   help="keep the intermediate *.raw frame dumps (large).")
    args = p.parse_args(argv)

    # Preflight.
    if not RENDERER.exists():
        print(f"error: renderer not found: {RENDERER}", file=sys.stderr)
        return 2
    if not TRACE_DUMP.exists():
        print(f"error: xitari trace_dump not built: {TRACE_DUMP}\n"
              f"       build it: (cd tools && make)", file=sys.stderr)
        return 2
    if shutil.which("ffmpeg") is None:
        print("error: ffmpeg not on PATH (needed to encode MP4s).", file=sys.stderr)
        return 2

    roms = all_roms()
    if not roms:
        print(f"error: no ROMs under {ROMS_DIR}", file=sys.stderr)
        return 2
    if args.games:
        want = set(args.games)
        roms = [r for r in roms if r.stem in want]
        missing = want - {r.stem for r in roms}
        if missing:
            print(f"error: unknown game(s): {sorted(missing)}\n"
                  f"       available: {[r.stem for r in all_roms()]}",
                  file=sys.stderr)
            return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.port in ("jaxtari", "both"):
        print("note: jaxtari is ~200x slower than jutari per frame — the jaxtari "
              "pass will take a long time.", file=sys.stderr)

    print(f"rendering {len(roms)} game(s) | port={args.port} | "
          f"{args.frames} frames | jobs={args.jobs} | out={args.out_dir}",
          file=sys.stderr)

    results: list[tuple[str, bool, str]] = []

    def _go(rom: Path) -> tuple[str, bool, str]:
        print(f"  [start] {rom.stem}", file=sys.stderr)
        res = render_one(rom, args.port, args.out_dir, args.frames,
                         args.seed, args.keep_raw)
        print(f"  [{'ok ' if res[1] else 'FAIL'}] {rom.stem}", file=sys.stderr)
        return res

    if args.jobs <= 1:
        for rom in roms:
            results.append(_go(rom))
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_go, rom): rom for rom in roms}
            for fut in as_completed(futs):
                results.append(fut.result())

    ok = sorted(s for s, good, _ in results if good)
    fail = sorted((s, e) for s, good, e in results if not good)
    print(f"\n==== done: {len(ok)}/{len(results)} rendered → {args.out_dir} ====")
    if fail:
        print(f"FAILED ({len(fail)}):", file=sys.stderr)
        for stem, err in fail:
            print(f"  - {stem}", file=sys.stderr)
            for line in err.strip().splitlines()[-4:]:
                print(f"      {line}", file=sys.stderr)
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
