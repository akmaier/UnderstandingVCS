#!/usr/bin/env python3
"""Frame-offset signature probe for the long-horizon Cluster-A render games.

Berzerk's f581 divergence was diagnosed (commit 14d60e5) as a VSYNC
frame-boundary / CPU<->TIA clock-phase lag: jutari renders ~1 extra animation
frame, so jutari frame k re-aligns with xitari frame k-1 (a whole-frame OFFSET),
not a localized per-pixel artifact.

This probe loads the raw screen dumps that tools/longhorizon_diff.py leaves at
/tmp/lh_<game>_{xi,ju}.raw and, for each diverging frame k in a window, reports
the per-pixel mismatch of ju[k] against xi[k-1], xi[k], xi[k+1]. If the minimum
is consistently at offset -1 (or +1) AND much smaller than offset 0, that is the
shared frame-boundary-lag signature. If offset 0 stays the minimum (no better
alignment at +/-1), the divergence is a distinct localized render bug.

Usage:  frame_offset_probe.py <game> [--around N] [--window W]
"""
import argparse, json, sys
import numpy as np
from pathlib import Path

H, W = 210, 160


def _load_raw(path):
    """Load a raw dump as (n, h, w), reading per-game (h,w) from the
    `<path>.shape` sidecar (pooyan=220, PAL=250 — task #110). 210 is only a
    fallback; hardcoding it fabricated false offsets for non-NTSC-210 games."""
    arr = np.fromfile(path, np.uint8)
    h, w = H, W
    sc = Path(str(path) + ".shape")
    if sc.exists():
        parts = sc.read_text().split()
        if len(parts) >= 3:
            h, w = int(parts[1]), int(parts[2])
    n = len(arr) // (h * w)
    return arr[:n * h * w].reshape(n, h, w)


def load(game):
    xi = _load_raw(f"/tmp/lh_{game}_xi.raw")
    ju = _load_raw(f"/tmp/lh_{game}_ju.raw")
    hh = min(xi.shape[1], ju.shape[1]); ww = min(xi.shape[2], ju.shape[2])
    return xi[:, :hh, :ww], ju[:, :hh, :ww]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("game")
    p.add_argument("--around", type=int, default=None, help="center frame (raw idx); default = first diverging")
    p.add_argument("--window", type=int, default=12)
    p.add_argument("--maxoff", type=int, default=2)
    a = p.parse_args()
    xi, ju = load(a.game)
    m = min(len(xi), len(ju))

    # first diverging raw index
    first = None
    for i in range(m):
        if (xi[i] != ju[i]).any():
            first = i
            break
    if first is None:
        print(json.dumps({"game": a.game, "result": "no divergence"}))
        return 0

    center = a.around if a.around is not None else first
    lo = max(a.maxoff, center - a.window)
    hi = min(m - a.maxoff, center + a.window)

    print(f"# game={a.game} first_div_rawidx={first} (action-frame {first+1}) window=[{lo},{hi})")
    print(f"# columns: rawk  div@0   | " + "  ".join(f"off{o:+d}" for o in range(-a.maxoff, a.maxoff + 1)) + "   best")
    offsets = list(range(-a.maxoff, a.maxoff + 1))
    best_off_counter = {}
    for k in range(lo, hi):
        d0 = int((xi[k] != ju[k]).sum())
        if d0 == 0:
            continue
        diffs = {}
        for o in offsets:
            j = k + o
            if 0 <= j < len(xi):
                diffs[o] = int((xi[j] != ju[k]).sum())
            else:
                diffs[o] = -1
        valid = {o: v for o, v in diffs.items() if v >= 0}
        best = min(valid, key=valid.get)
        best_off_counter[best] = best_off_counter.get(best, 0) + 1
        row = f"  {k:4d}  {d0:6d}  | " + "  ".join(
            (f"{diffs[o]:6d}" if diffs[o] >= 0 else "   n/a") for o in offsets
        ) + f"   {best:+d}"
        print(row)
    print(f"# best-offset histogram over diverging frames in window: {best_off_counter}")
    # verdict
    nonzero = sum(v for k, v in best_off_counter.items())
    at0 = best_off_counter.get(0, 0)
    shifted = nonzero - at0
    if nonzero == 0:
        verdict = "no diverging frames in window"
    elif shifted > at0 and shifted >= 2:
        dom = max((o for o in best_off_counter if o != 0), key=lambda o: best_off_counter[o], default=0)
        verdict = f"FRAME-OFFSET signature (berzerk-like): ju re-aligns to xi at offset {dom:+d} on {shifted}/{nonzero} frames"
    else:
        verdict = f"LOCALIZED render bug: offset 0 stays best on {at0}/{nonzero} frames (no whole-frame re-alignment)"
    print(f"# VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
