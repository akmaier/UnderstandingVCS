#!/usr/bin/env python3
"""Diagnostic: localize a jaxtari↔xitari screen-conformance divergence.

Runs jaxtari live for a ROM, diffs each frame against the xitari fixture
(and the jutari fixture, the golden 0-px reference), and characterises the
diff geometry: per-row diff counts, plus best vertical/horizontal shift
alignment (a pure shift ⇒ some offset drives the diff to ~0 ⇒ a crop /
row-indexing bug rather than a per-pixel render bug).

Usage:
    jaxtari/.venv/bin/python tools/diag_screen_diff.py pitfall_noop_10 [n_frames]
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "jaxtari"))

from jaxtari.env.stella_environment import StellaEnvironment  # noqa: E402
from jaxtari.games.breakout import BreakoutRomSettings  # noqa: E402
from jaxtari.games.pong import PongRomSettings  # noqa: E402
from jaxtari.games.atari_classics import (  # noqa: E402
    PitfallRomSettings,
    EnduroRomSettings,
)

_ROMS = _REPO / "xitari" / "roms"
_SCREENS = _REPO / "tools" / "fixtures" / "screens"
_ACTIONS = _REPO / "tools" / "fixtures" / "actions"
_H, _W = 210, 160

_SETTINGS = {
    "pong.bin": PongRomSettings,
    "breakout.bin": BreakoutRomSettings,
    "pitfall.bin": PitfallRomSettings,
    "enduro.bin": EnduroRomSettings,
}
# case-name -> rom file
_ROMOF = {
    "pitfall_noop_10": "pitfall.bin",
    "enduro_noop_10": "enduro.bin",
    "pong_noop_10": "pong.bin",
    "breakout_noop_10": "breakout.bin",
    "space_invaders_noop_10": "space_invaders.bin",
    "seaquest_noop_10": "seaquest.bin",
}


def _load(path: Path):
    if not path.exists():
        return None
    raw = gzip.open(path, "rb").read()
    n = len(raw) // (_H * _W)
    return np.frombuffer(raw, dtype=np.uint8)[: n * _H * _W].reshape(n, _H, _W)


def _actions(case):
    out = []
    for line in (_ACTIONS / f"{case}.txt").read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(int(s))
    return out


def _run_jaxtari(case, rom_file, n):
    rom = np.frombuffer((_ROMS / rom_file).read_bytes(), dtype=np.uint8)
    factory = _SETTINGS.get(rom_file)
    env = StellaEnvironment(rom, factory()) if factory else StellaEnvironment(rom)
    env.reset(boot_noop_steps=60, boot_reset_steps=4)
    acts = _actions(case)[:n]
    fr = []
    full = []
    for a in acts:
        env.step(int(a))
        fr.append(np.asarray(env.get_screen(), dtype=np.uint8))
        full.append(np.asarray(env._console.bus.tia.framebuffer, dtype=np.uint8))
    return np.stack(fr), np.stack(full)


def _shift_align(jx, xi):
    """Best vertical / horizontal integer shift minimising the diff."""
    best = (0, 0, int((jx != xi).sum()))
    for dy in range(-6, 7):
        for dx in range(-6, 7):
            shifted = np.roll(np.roll(jx, dy, axis=0), dx, axis=1)
            # only score the overlap region (ignore wrapped edges)
            d = int((shifted != xi).sum())
            if d < best[2]:
                best = (dy, dx, d)
    return best


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "pitfall_noop_10"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    rom_file = _ROMOF[case]

    xi = _load(_SCREENS / f"{case}.screen.gz")
    jt = _load(_SCREENS / f"{case}_jutari.screen.gz")
    print(f"== {case} (rom={rom_file}, n={n}) ==")
    print(f"xitari fixture frames: {None if xi is None else len(xi)}")
    print(f"jutari fixture frames: {None if jt is None else len(jt)}")

    jx, full = _run_jaxtari(case, rom_file, n)
    print(f"jaxtari live frames:   {len(jx)}  full-fb shape={full.shape}")

    m = min(len(xi), len(jx))
    for i in range(m):
        d_xi = int((jx[i] != xi[i]).sum())
        d_jt = None if jt is None else int((jx[i] != jt[i]).sum())
        print(f"\n-- frame {i}: jaxtari↔xitari={d_xi} px   jaxtari↔jutari={d_jt} px")
        if d_xi == 0:
            continue
        # per-row diff
        rows = (jx[i] != xi[i]).sum(axis=1)
        nz = np.nonzero(rows)[0]
        print(f"   differing rows: {len(nz)} rows, "
              f"range [{nz.min()}..{nz.max()}], "
              f"top diffs: {sorted(zip(rows[nz].tolist(), nz.tolist()), reverse=True)[:8]}")
        # vertical/horizontal shift alignment
        dy, dx, dbest = _shift_align(jx[i], xi[i])
        print(f"   best shift align: dy={dy} dx={dx} -> {dbest} px "
              f"(vs {d_xi} unshifted)")

    # Cache the run so follow-up analysis doesn't re-run slow jaxtari.
    cache = _SCREENS.parent / "cache"
    cache.mkdir(exist_ok=True)
    np.save(cache / f"{case}_jaxtari_screen.npy", jx)
    np.save(cache / f"{case}_jaxtari_full.npy", full)

    # PIXEL-VALUE DUMP — for frame 0's worst rows, show the differing columns
    # with (col: jaxtari -> xitari [jutari]) so we can see WHAT differs
    # (wrong colour? shifted object? missing/extra pixels?).
    if m and int((jx[0] != xi[0]).sum()):
        rows = (jx[0] != xi[0]).sum(axis=1)
        worst = sorted(np.nonzero(rows)[0].tolist(),
                       key=lambda r: -rows[r])[:6]
        print(f"\n== frame 0 worst-row pixel values (out_row = fb_row-34) ==")
        for r in sorted(worst):
            cols = np.nonzero(jx[0][r] != xi[0][r])[0]
            jtr = jt[0][r] if jt is not None else None
            parts = []
            for c in cols[:20]:
                jtv = f",{int(jtr[c]):02x}" if jtr is not None else ""
                parts.append(f"x{c}:{int(jx[0][r][c]):02x}->{int(xi[0][r][c]):02x}{jtv}")
            print(f"   row {r:3d} (fb {r+34}, {len(cols)} diff): {' '.join(parts)}"
                  + (" ..." if len(cols) > 20 else ""))
        # column span of all differences (is it a contiguous band?)
        allcols = np.nonzero((jx[0] != xi[0]).any(axis=0))[0]
        print(f"   all differing columns span: [{allcols.min()}..{allcols.max()}] "
              f"({len(allcols)} distinct cols)")


if __name__ == "__main__":
    main()
