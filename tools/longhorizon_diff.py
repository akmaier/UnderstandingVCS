#!/usr/bin/env python3
"""Long-horizon screen-divergence localizer: jutari vs xitari through the
comparison-video pipeline (dump_xitari_frames.py + dump_jutari_frames.jl, both
auto-resetting on game-over), so it reproduces exactly what the rendered videos
show. Emits one JSON line with the first diverging frame, its symptom, and a
frozen-tail signal (flags the "jutari stops at game-over while xitari continues"
class). Run ONE game at a time; meant to be fanned out across games.

  python tools/longhorizon_diff.py <game> [--frames N] [--rom NAME] [--stream PATH]
"""
import argparse, collections, json, subprocess, sys
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENPY = REPO / "jaxtari/.venv/bin/python"
XI = REPO / "tools/breakout_video/dump_xitari_frames.py"
JU = REPO / "tools/breakout_video/dump_jutari_frames.jl"
OUTDIR = REPO / "tools/comparison_videos/output"
H, W = 210, 160   # NTSC fallback; real per-game (h,w) comes from the .shape sidecar


def _load_raw(path):
    """Load a raw screen dump as (n, h, w). The per-game height is read from the
    `<path>.shape` sidecar the dumpers write — NTSC is 210 but pooyan renders 220
    and PAL games 250 (task #110). Hardcoding 210 mis-reshaped those and
    fabricated false divergences (pooyan's bogus 'render bug', #127b). Falls back
    to 210x160 only if no sidecar is present."""
    arr = np.fromfile(path, np.uint8)
    h, w = H, W
    sc = Path(str(path) + ".shape")
    if sc.exists():
        parts = sc.read_text().split()
        if len(parts) >= 3:
            h, w = int(parts[1]), int(parts[2])
    n = len(arr) // (h * w)
    return arr[:n * h * w].reshape(n, h, w)


def _last_static_run(arr):
    run = 0
    for i in range(len(arr) - 1, 0, -1):
        if (arr[i] == arr[i - 1]).all():
            run += 1
        else:
            break
    return run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("game")
    p.add_argument("--frames", type=int, default=900)
    p.add_argument("--rom", default=None)
    p.add_argument("--stream", default=None)
    a = p.parse_args()
    rom = REPO / f"tools/rom_sweep/roms/{a.rom or a.game}.bin"
    stream = Path(a.stream) if a.stream else OUTDIR / f"{a.game}_breakout_random_actions.txt"
    xr = Path(f"/tmp/lh_{a.game}_xi.raw")
    jr = Path(f"/tmp/lh_{a.game}_ju.raw")

    err = None
    try:
        subprocess.run([str(VENPY), str(XI), "--rom", str(rom), "--actions",
                        str(stream), "--out", str(xr), "--max-frames", str(a.frames)],
                       check=True, capture_output=True, timeout=2400, text=True)
        subprocess.run(["julia", "--project=" + str(REPO / "jutari"), str(JU),
                        "--rom", str(rom), "--actions", str(stream), "--out",
                        str(jr), "--max-frames", str(a.frames)],
                       check=True, capture_output=True, timeout=2400, text=True)
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "")[-400:]
        print(json.dumps({"game": a.game, "error": "dump failed", "detail": err}))
        return 1

    xi = _load_raw(xr); ju = _load_raw(jr)
    # crop to a common (h,w); xitari and the port should agree, but guard PAL/NTSC edges
    hh = min(xi.shape[1], ju.shape[1]); ww = min(xi.shape[2], ju.shape[2])
    xi = xi[:, :hh, :ww]; ju = ju[:, :hh, :ww]
    nxi, nju = xi.shape[0], ju.shape[0]
    m = min(nxi, nju)

    first, sym = None, {}
    for i in range(m):
        d = xi[i] != ju[i]
        if d.any():
            first = i + 1
            rows = np.where(d.any(1))[0]; cols = np.where(d.any(0))[0]
            pr = collections.Counter(zip(xi[i][d].tolist(), ju[i][d].tolist()))
            sym = {"px": int(d.sum()),
                   "rows": [int(rows.min()), int(rows.max())],
                   "cols": [int(cols.min()), int(cols.max())],
                   "xi_to_ju_swaps": [[int(x), int(y), int(c)]
                                      for (x, y), c in pr.most_common(6)]}
            break

    # peak divergence + total diverging frames (post-onset severity)
    div_frames = sum(1 for i in range(m) if (xi[i] != ju[i]).any())
    res = {"game": a.game, "n": a.frames, "xi_frames": nxi, "ju_frames": nju,
           "first_div_frame": first, "first_div_sec": None if first is None else round(first / 60, 1),
           "diverging_frames": div_frames, "symptom": sym,
           "ju_frozen_tail": _last_static_run(ju), "xi_frozen_tail": _last_static_run(xi)}
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
