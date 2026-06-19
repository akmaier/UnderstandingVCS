#!/usr/bin/env python3
"""Long-horizon conformance sweep for the games KNOWN to diverge past the
30-frame-RAM / 60-frame-screen in-window sweep. The standard sweeps
(sweep_jutari_ram.py / sweep_jutari_screen.py) confirm bit-exactness only inside
that window; this sweep drives each affected game to a full 60s/3600-frame
episode (or a per-game cap past its known first divergence) and reports the first
frame where jutari's rendered screen differs from xitari, via the 60fps
comparison-video pipeline (tools/longhorizon_diff.py).

Scope is deliberately the PROBLEM games only — the rest were verified by hand.
Re-run after a fix to confirm a game's first-divergence frame moved (ideally to
"none").

  jaxtari/.venv/bin/python tools/rom_sweep/sweep_longhorizon.py [--jobs N]
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
PY = REPO / "jaxtari/.venv/bin/python"
TOOL = REPO / "tools/longhorizon_diff.py"
OUT = REPO / "tools/rom_sweep/results_longhorizon.md"

# (game, frames) — frames ≈ reported_seconds*60 + margin (video is 60 fps).
PROBLEM_GAMES = [
    ("asteroids", 360), ("wizard_of_wor", 320), ("berzerk", 760),
    ("road_runner", 900), ("montezuma_revenge", 1040), ("riverraid", 1100),
    ("space_invaders", 1260), ("asterix", 1320), ("pooyan", 1700),
    ("kangaroo", 1880), ("phoenix", 1860), ("pacman", 1900), ("ms_pacman", 1900),
]


def run_one(game: str, n: int) -> dict:
    try:
        r = subprocess.run([str(PY), str(TOOL), game, "--frames", str(n)],
                           capture_output=True, text=True, timeout=3600)
        line = [l for l in r.stdout.strip().splitlines() if l.startswith("{")]
        return json.loads(line[-1]) if line else {"game": game, "error": "no output"}
    except Exception as e:  # noqa: BLE001
        return {"game": game, "error": str(e)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=3)
    a = p.parse_args(argv)
    res = {}
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(run_one, g, n): g for g, n in PROBLEM_GAMES}
        for f in cf.as_completed(futs):
            d = f.result(); res[d["game"]] = d
            print(f"  {d.get('game'):<18} first_div={d.get('first_div_frame')} "
                  f"({d.get('first_div_sec')}s) ju_frozen={d.get('ju_frozen_tail')}",
                  file=sys.stderr)
    with open(OUT, "w") as fh:
        fh.write("# Long-horizon conformance sweep (problem games only)\n\n")
        fh.write("Screen first-divergence jutari vs xitari via the 60fps comparison-video "
                 "pipeline. The in-window sweeps (30f RAM / 60f screen) pass for all of "
                 "these; divergence is post-window. `ju_frozen`>0 with `xi_frozen`=0 means "
                 "jutari is stuck at game-over while xitari continues.\n\n")
        fh.write("| game | first div frame | sec | ju_frozen | xi_frozen | diverging frames |\n")
        fh.write("|---|---|---|---|---|---|\n")
        for g, _ in PROBLEM_GAMES:
            d = res.get(g, {})
            fh.write(f"| {g} | {d.get('first_div_frame')} | {d.get('first_div_sec')} "
                     f"| {d.get('ju_frozen_tail')} | {d.get('xi_frozen_tail')} "
                     f"| {d.get('diverging_frames')} |\n")
    print(f"\nwrote {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
