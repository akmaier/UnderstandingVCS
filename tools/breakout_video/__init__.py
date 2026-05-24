"""Breakout side-by-side video pipeline (xitari + jaxtari + jutari).

Shared helpers for the comparison-video deliverable:

  generate_actions(n, seed)   — deterministic random paddle motion.
  load_ntsc_palette()         — Stella NTSC palette (256 RGB entries).
  decode_palette(idx_frame)   — uint8 palette indices → (H, W, 3) RGB.

The three dump scripts (`dump_xitari_frames.py`, `dump_jaxtari_frames.py`,
`dump_jutari_frames.jl`) all read the same `actions.txt` so the runs
are byte-identical at the input side. The `render_breakout_compare.py`
top-level entry point orchestrates them and stitches the videos with
ffmpeg.
"""
from __future__ import annotations

import random


# Action enum aligned with ALE / xitari.
A_NOOP, A_FIRE, A_UP, A_RIGHT, A_LEFT, A_DOWN = 0, 1, 2, 3, 4, 5


def generate_actions(n: int, *, seed: int = 42) -> list[int]:
    """Generate a deterministic random Breakout action sequence.

    The motion model is a slightly-persistent random walk: pick a
    direction (NOOP / LEFT / RIGHT) and hold it for a few frames
    before resampling. FIRE is sprinkled in once per ~120 frames to
    re-launch the ball if it goes off-screen.

    Identical seed → identical actions across all three emulator
    runs, which is the prerequisite for a meaningful side-by-side
    comparison.
    """
    rng = random.Random(seed)
    actions: list[int] = []
    direction = A_NOOP
    hold = 0
    for i in range(n):
        if i < 20:
            # First 20 frames: send NOOP to let the title settle.
            actions.append(A_NOOP)
            continue
        if i % 120 == 20:
            actions.append(A_FIRE)
            continue
        if hold == 0:
            direction = rng.choice([A_NOOP, A_LEFT, A_RIGHT, A_LEFT, A_RIGHT])
            hold = rng.randint(3, 12)
        actions.append(direction)
        hold -= 1
    return actions


# --------------------------------------------------------------------------- #
# Atari NTSC palette — Stella convention, lifted from
# xitari/emucore/Console.cxx::ourNTSCPalette. 128 distinct colours,
# repeated to fill 256 entries (Stella stores them at even indices and
# leaves the odd entries 0; we expand to 256 by carrying the previous
# even entry through, matching how `ALEScreen::getArray()` indices
# them).
# --------------------------------------------------------------------------- #

_PALETTE_PAIRS = [
    0x000000, 0x4a4a4a, 0x6f6f6f, 0x8e8e8e, 0xaaaaaa, 0xc0c0c0, 0xd6d6d6, 0xececec,
    0x484800, 0x69690f, 0x86861d, 0xa2a22a, 0xbbbb35, 0xd2d240, 0xe8e84a, 0xfcfc54,
    0x7c2c00, 0x904811, 0xa26221, 0xb47a30, 0xc3903d, 0xd2a44a, 0xdfb755, 0xecc860,
    0x901c00, 0xa33915, 0xb55328, 0xc66c3a, 0xd5824a, 0xe39759, 0xf0aa67, 0xfcbc74,
    0x940000, 0xa71a1a, 0xb83232, 0xc84848, 0xd65c5c, 0xe46f6f, 0xf08080, 0xfc9090,
    0x840064, 0x97197a, 0xa8308f, 0xb846a2, 0xc659b3, 0xd46cc3, 0xe07cd2, 0xec8ce0,
    0x500084, 0x68199a, 0x7d30ad, 0x9246c0, 0xa459d0, 0xb56ce0, 0xc57cee, 0xd48cfc,
    0x140090, 0x331aa3, 0x4e32b5, 0x6848c6, 0x7f5cd5, 0x956fe3, 0xa980f0, 0xbc90fc,
    0x000094, 0x181aa7, 0x2d32b8, 0x4248c8, 0x545cd6, 0x656fe4, 0x7580f0, 0x8490fc,
    0x001c88, 0x183b9d, 0x2d57b0, 0x4272c2, 0x548ad2, 0x65a0e1, 0x75b5ef, 0x84c8fc,
    0x003064, 0x185080, 0x2d6d98, 0x4288b0, 0x54a0c5, 0x65b7d9, 0x75cceb, 0x84e0fc,
    0x004030, 0x18624e, 0x2d8169, 0x429e82, 0x54b899, 0x65d1ae, 0x75e7c2, 0x84fcd4,
    0x004400, 0x1a661a, 0x328432, 0x48a048, 0x5cba5c, 0x6fd26f, 0x80e880, 0x90fc90,
    0x143c00, 0x355f18, 0x527e2d, 0x6e9c42, 0x87b754, 0x9ed065, 0xb4e775, 0xc8fc84,
    0x303800, 0x505916, 0x6d762b, 0x88923e, 0xa0ab4f, 0xb7c25f, 0xccd86e, 0xe0ec7c,
    0x482c00, 0x694d14, 0x866a26, 0xa28638, 0xbb9f47, 0xd2b656, 0xe8cc63, 0xfce070,
]


def load_ntsc_palette():
    """Return a `(256, 3)` uint8 array of RGB entries indexed by the
    Stella NTSC palette byte. Even palette indices are populated
    directly from `ourNTSCPalette`; odd indices carry the previous
    even entry (xitari leaves them 0 in its raw table because real
    cart writes always have the low bit clear)."""
    import numpy as np
    pal = np.zeros((256, 3), dtype=np.uint8)
    for i, rgb in enumerate(_PALETTE_PAIRS):
        idx = i * 2
        pal[idx, 0] = (rgb >> 16) & 0xFF
        pal[idx, 1] = (rgb >> 8) & 0xFF
        pal[idx, 2] = rgb & 0xFF
        # Carry through to the odd entry so jaxtari / jutari renderers
        # that mask only the high nibble + low nibble's high 3 bits
        # still produce a sensible RGB.
        if idx + 1 < 256:
            pal[idx + 1] = pal[idx]
    return pal


def decode_palette(idx_frame, palette):
    """`idx_frame` is a `(H, W)` uint8 palette-index array. Returns
    `(H, W, 3)` uint8 RGB."""
    return palette[idx_frame]
