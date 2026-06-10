"""Per-frame jutari↔xitari RAM diff for any ROM + action stream.

Generalised from `pong_3way_ram_diff.py`. Drops the jaxtari arm so
this stays jutari-focused (per the "jutari first" sequencing decision
in `P3I_G_THREADING_PLAN.md`).

Usage:
    cd jaxtari && .venv/bin/python ../tools/jutari_xitari_ram_diff.py \\
        --rom xitari/roms/breakout.bin \\
        --actions tools/breakout_video/output/breakout_random_actions.txt \\
        --rom-settings breakout \\
        --max-frames 280

Why it lives at the repo root (not under jaxtari/): we need the
jaxtari .venv's Python to run trace_dump as a subprocess + invoke
julia, but the script itself doesn't import jaxtari. Run it from
`jaxtari/` so relative paths work.

What it does
============

  1. Invokes `tools/trace_dump` (xitari binary) with the action stream
     to capture xitari's per-frame RAM.
  2. Invokes `julia tools/jutari_trace_dump.jl` with a synthetic
     trace fixture (frame indices + the same actions) to capture
     jutari's per-frame RAM.
  3. Diffs the two RAM streams per frame.
  4. Prints:
       - first frame where jutari diverges from xitari + which bytes
       - per-frame byte-count histogram (helps see if divergence
         GROWS over time vs stays constant)

Findings format: prints to stderr; one summary block at the end.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ACTIONS = REPO / "tools/breakout_video/output/breakout_random_actions.txt"


def _load_actions(path: Path) -> list[int]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(int(line))
    return out


def _run_xitari(rom: Path, actions: list[int], n: int) -> list[np.ndarray]:
    """Invoke tools/trace_dump and parse per-frame RAM."""
    acts_path = Path("/tmp/_xitari_actions.txt")
    acts_path.write_text("\n".join(str(a) for a in actions[:n]))
    r = subprocess.run(
        [str(REPO / "tools/trace_dump"),
         "--rom", str(rom),
         "--actions", str(acts_path),
         "--max-frames", str(n)],
        capture_output=True, text=True, check=True)
    rams = []
    for line in r.stdout.strip().splitlines():
        obj = json.loads(line)
        # Task #80 / 2026-06-09: trace_dump emits a synthetic "frame 0"
        # record with `boot_end=true` BEFORE any user-action frame, to
        # let `boot_state_diff.jl` compare post-boot RAM directly. Skip
        # it here so this tool's frame-N comparison stays aligned with
        # jutari's user-action frames (jutari only emits user-action
        # frames, no boot_end record).
        if obj.get("boot_end"):
            continue
        rams.append(np.frombuffer(bytes.fromhex(obj["ram"]), dtype=np.uint8))
    return rams


def _run_jutari(rom: Path, actions: list[int], n: int) -> list[np.ndarray]:
    """Invoke tools/jutari_trace_dump.jl via the synthetic trace
    fixture pattern that `pong_3way_ram_diff.py` uses."""
    fixture_path = Path("/tmp/_jutari_actions_trace.jsonl")
    with open(fixture_path, "w") as f:
        for i, a in enumerate(actions[:n]):
            f.write(f'{{"frame": {i+1}, "action": {a}, "ram": ""}}\n')
    out_path = Path("/tmp/_jutari_rams.jsonl")
    subprocess.run(
        ["julia", "--project=" + str(REPO / "jutari"),
         str(REPO / "tools/jutari_trace_dump.jl"),
         "--rom", str(rom),
         "--trace", str(fixture_path),
         "--out", str(out_path)],
        check=True)
    rams = []
    with open(out_path) as f:
        for line in f:
            obj = json.loads(line)
            rams.append(np.frombuffer(bytes.fromhex(obj["ram"]),
                                       dtype=np.uint8))
    return rams


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rom", type=Path, required=True)
    p.add_argument("--actions", type=Path, default=DEFAULT_ACTIONS)
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--rom-settings", type=str, default="generic",
                   help="(carried for future per-game RAM-byte naming; "
                        "currently unused for the diff itself)")
    args = p.parse_args(argv)

    actions = _load_actions(args.actions)
    n = min(args.max_frames, len(actions))
    print(f"  using {n} actions from {args.actions}", file=sys.stderr)
    print(f"  ROM: {args.rom}", file=sys.stderr)

    print(f"\n  running xitari (subprocess to trace_dump)...", file=sys.stderr)
    xi = _run_xitari(args.rom, actions, n)
    print(f"  xitari produced {len(xi)} frames", file=sys.stderr)

    print(f"  running jutari (subprocess to jutari_trace_dump.jl)...",
          file=sys.stderr)
    ju = _run_jutari(args.rom, actions, n)
    print(f"  jutari produced {len(ju)} frames", file=sys.stderr)

    n = min(len(xi), len(ju), n)
    diff_counts = []
    first_divergence = None
    for i in range(n):
        differing = int((xi[i] != ju[i]).sum())
        diff_counts.append(differing)
        if first_divergence is None and differing > 0:
            first_divergence = i

    # Summary
    print(f"\n  === SUMMARY ===")
    print(f"  frames compared:  {n}")
    print(f"  first divergence: "
          f"{'NONE — bit-exact!' if first_divergence is None else f'frame {first_divergence}'}")
    if first_divergence is not None:
        # Find the bytes
        i = first_divergence
        diffs = np.where(xi[i] != ju[i])[0]
        print(f"    frame {i} action={actions[i]}: {len(diffs)} bytes differ at: "
              f"{[f'${o:02x}' for o in diffs.tolist()]}")
        for off in diffs[:15]:
            print(f"      RAM[${off:02x}]: xitari=${int(xi[i][off]):02x}  "
                  f"jutari=${int(ju[i][off]):02x}")

    # Per-frame histogram (bucketed)
    print(f"\n  per-frame divergence count (frame: bytes_differ):")
    for i in range(0, n, max(1, n // 30)):
        marker = " <-- first" if i == first_divergence else ""
        print(f"    {i:4d}: {diff_counts[i]:>3d}{marker}")
    if diff_counts:
        print(f"  max diff in any frame: {max(diff_counts)} bytes")
        post = diff_counts[first_divergence:] if first_divergence is not None else diff_counts
        if post:
            print(f"  mean diff post-first-divergence: "
                  f"{sum(post)/len(post):.1f} bytes/frame")
    return 0


if __name__ == "__main__":
    sys.exit(main())
