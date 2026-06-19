#!/usr/bin/env python3
"""Randomized-action conformance validation (reviewer pt 11).

The headline 64/64 conformance is measured on a NOOP stream (RAM sweep) and one
fixed pseudo-random stream (screen sweep). This driver stresses the ports with
*several independent* random action streams per game and checks that jutari stays
bit-identical to xitari on every frame, reporting the first divergence (or
"exact"). It reuses tools/jutari_xitari_ram_diff.py, so per-game RomSettings come
from the same basename auto-detect as the conformance sweeps.

Both emulators receive the SAME action integers, so even actions outside a game's
minimal set are a fair test: the two must agree regardless. Actions are drawn
uniformly from 0..17 (the full ALE action interface).

Run ALONE (it spawns xitari + jutari subprocesses per run; keep off a busy box):
  jaxtari/.venv/bin/python tools/random_action_validation.py \
      --seeds 1 2 3 --frames 300 --jobs 6
  # subset for a quick check:
  ... --games pong breakout space_invaders qbert montezuma_revenge
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
VENPY = REPO / "jaxtari/.venv/bin/python"
DIFF = REPO / "tools/jutari_xitari_ram_diff.py"
ROMDIR = REPO / "tools/rom_sweep/roms"
MANIFEST = REPO / "tools/rom_sweep/manifest.txt"

ALL_GAMES = [l.split("\t")[0] for l in MANIFEST.read_text().splitlines() if l.strip()]


def _first_divergence(stdout: str) -> int | None:
    """Return the first frame index that differs, or None if fully exact."""
    m = re.search(r"frame (\d+) action=\d+: \d+ bytes differ", stdout)
    return int(m.group(1)) if m else None


def run_one(game: str, seed: int, frames: int, max_action: int) -> dict:
    rom = ROMDIR / f"{game}.bin"
    if not rom.exists():
        return {"game": game, "seed": seed, "status": "MISSING"}
    acts = np.random.default_rng(seed).integers(0, max_action + 1, size=frames)
    with tempfile.NamedTemporaryFile("w", suffix=f"_{game}_{seed}.txt",
                                     delete=False) as fh:
        fh.write("\n".join(str(int(a)) for a in acts))
        apath = fh.name
    try:
        r = subprocess.run(
            [str(VENPY), str(DIFF), "--rom", str(rom),
             "--actions", apath, "--max-frames", str(frames)],
            capture_output=True, text=True, timeout=900)
        out = r.stdout + r.stderr
        div = _first_divergence(out)
        status = "exact" if div is None else f"div@{div}"
        if r.returncode != 0 and div is None:
            status = "ERROR"
        return {"game": game, "seed": seed, "status": status}
    except subprocess.TimeoutExpired:
        return {"game": game, "seed": seed, "status": "TIMEOUT"}
    finally:
        os.unlink(apath)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--games", nargs="*", default=ALL_GAMES)
    p.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--max-action", type=int, default=17)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    p.add_argument("--out", type=Path,
                   default=REPO / "tools/rom_sweep/results_random_actions.md")
    a = p.parse_args(argv)

    jobs = [(g, s) for g in a.games for s in a.seeds]
    results: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(run_one, g, s, a.frames, a.max_action): (g, s)
                for g, s in jobs}
        for fut in cf.as_completed(futs):
            res = fut.result()
            results.append(res)
            print(f"  {res['game']:<22} seed {res['seed']}: {res['status']}",
                  file=sys.stderr)

    results.sort(key=lambda r: (r["game"], r["seed"]))
    n_exact = sum(1 for r in results if r["status"] == "exact")
    # per-game: exact iff exact on every seed
    per_game = {}
    for r in results:
        per_game.setdefault(r["game"], []).append(r["status"])
    games_all_exact = sum(1 for g, ss in per_game.items()
                          if all(s == "exact" for s in ss))

    with open(a.out, "w") as f:
        f.write("# Randomized-action conformance — jutari vs xitari\n\n")
        f.write(f"Per-frame 128 B RIOT-RAM diff over **{a.frames}** frames after "
                f"the standard boot, with actions drawn uniformly from "
                f"0..{a.max_action} under seeds {a.seeds}. Both emulators receive "
                f"the same stream; `exact` = byte-identical on every frame.\n\n")
        f.write(f"**{games_all_exact}/{len(per_game)} games exact on all "
                f"{len(a.seeds)} seeds; {n_exact}/{len(results)} (game,seed) runs "
                f"exact.**\n\n")
        f.write("| game | " + " | ".join(f"seed {s}" for s in a.seeds) + " |\n")
        f.write("|---|" + "---|" * len(a.seeds) + "\n")
        for g in sorted(per_game):
            row = {r["seed"]: r["status"] for r in results if r["game"] == g}
            cells = [row.get(s, "—") for s in a.seeds]
            f.write(f"| {g} | " + " | ".join(cells) + " |\n")

    print(f"\n[random-action] {games_all_exact}/{len(per_game)} games exact on all "
          f"seeds; {n_exact}/{len(results)} runs exact -> {a.out}", file=sys.stderr)
    return 0 if n_exact == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
