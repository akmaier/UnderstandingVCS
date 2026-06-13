"""PXC-S — screen (framebuffer) conformance: xitari ↔ jaxtari and
xitari ↔ jutari, per rendered frame.

The PXC1/PXC2 RAM tests compare the 128-byte RIOT RAM and so are blind to
*rendering* bugs — the TIA register state (PF*, COLU*, sprite positions)
and the resulting framebuffer can be wrong while RAM stays bit-exact. The
Breakout "red columns" regression (a deferred PF2=$00 clear getting
clobbered, leaking the brick pattern into the lower screen) was exactly
that: RAM bit-exact, screen badly wrong. This test closes that gap.

For each ROM it loads xitari's per-frame screen (`tools/fixtures/screens/
<rom>_noop_10.screen.gz`, 210×160 palette indices, captured from
`tools/trace_dump --screen`), then:

  * runs jaxtari live and diffs each frame's `get_screen()` against xitari;
  * loads the jutari screen fixture (`*_jutari.screen.gz`) and diffs it
    against xitari.

It asserts the per-frame differing-pixel count stays ≤ the pinned
`max_screen_diff`. **Tighten the pin when you improve a ROM's rendering**
(regenerate the jutari fixture in the same commit so both ports stay
lock-step). The pins record the current state:

  breakout  16   — near-perfect (only a 1 px paddle-X offset, 7 rows)
  pong      —    — large rendering gap (sprite/net/score positioning)
  others    —    — pre-existing rendering-accuracy gaps; RAM is much
                   closer than the screen (see bug_fix_log.md)
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from jaxtari.env.stella_environment import StellaEnvironment
from jaxtari.games.breakout import BreakoutRomSettings as _BreakoutRomSettings
from jaxtari.games.pong import PongRomSettings as _PongRomSettings
from jaxtari.games.atari_classics import (
    PitfallRomSettings as _PitfallRomSettings,
    EnduroRomSettings as _EnduroRomSettings,
)

_REPO = Path(__file__).resolve().parents[2]
_ROMS = _REPO / "xitari" / "roms"
_SCREENS = _REPO / "tools" / "fixtures" / "screens"
_ACTIONS = _REPO / "tools" / "fixtures" / "actions"

# Task #95 (2026-06-13): pitfall + enduro were MISSING here, so the
# jaxtari live arm rendered them with bare StellaEnvironment (generic
# settings) — omitting their getStartingActions — while the xitari
# fixtures used the real per-game settings. That settings mismatch
# inflated the enduro screen diff (516→249 px once corrected). The
# jutari fixture tools had the identical gap; all now carry pitfall +
# enduro. (Seaquest has a jaxtari RomSettings but no jutari one yet, so
# it stays generic on both sides for now — still apples-to-apples.)
_SETTINGS = {
    "pong.bin": _PongRomSettings,
    "breakout.bin": _BreakoutRomSettings,
    "pitfall.bin": _PitfallRomSettings,
    "enduro.bin": _EnduroRomSettings,
}

_H, _W = 210, 160


class _Case(NamedTuple):
    name: str
    rom: str
    # Max per-frame differing-pixel count tolerated (jaxtari-live AND
    # jutari-fixture, each vs the xitari screen). breakout is tight (the
    # fix); the rest document pre-existing rendering gaps.
    max_screen_diff: int


_CASES = [
    # P3i-g pt3 NUSIZ-wide +1 quirk fix: breakout 16->8, seaquest 3946->3940,
    # enduro 1988->1972. pt4 COLU& 0xFE mask: pong 29760->920 (the LSB of
    # the colour-luminance registers is unused on real hardware; ROMs that
    # write odd values produce visually-identical frames but byte-level
    # diffs against xitari's masked store). pt5 CTRLPF.D1 SCOREMODE:
    # pong 920->568. pt6 defer ENAM0/ENAM1/ENABL to activation cc:
    # breakout 8->0 BIT-EXACT. pt7 extend defer to NUSIZ/COLU/CTRLPF/REFP:
    # space_invaders 2145->2079, enduro 1972->1954.
    # pt8 RIOT INTIM `-1` offset (xitari `M6532::peek` case 0x04 has
    # `myTimer - (delta>>shift) - 1` in its formula — the off-by-one
    # was making every INTIM-polling loop in early VBLANK exit 1+
    # iterations LATE in jaxtari, accumulating ~76 CPU cycles of TIA
    # scanline drift per loop). MASSIVE cross-ROM win:
    #   pong         568 -> 32  (-536, 94%)
    #   space_inv    2079 -> 12  (-2067, 99.4%)
    #   pitfall      1786 -> 322 (-1464, 82%)
    #   seaquest     3941 -> 1104 (-2837, 72%)
    #   enduro       1954 -> 1197 (-757, 39%)
    # Task #84 (2026-06-10): GRP0/GRP1 defer to activation_clock so
    # pong's paddle stops painting 1 row early.
    # Task #85 (2026-06-11): RESMP* gate in `_missile_set`/`_overlay_missile`
    # — RESMP*=$02 hides the missile (locked to player center, invisible).
    # Pong's color-stripe-below-score 16 px phantom GONE.
    # Cross-ROM screen deltas after task #84 + #85 (fixture regen):
    #   pong        32 -> 8     (-24, only HMOVE row-0 comb [#83] left)
    #   space_inv   12 -> 42    (post-#84 GRP defer exposed other timing)
    #   pitfall    322 -> 553   (same — task #84 exposed a render-order
    #                            mismatch in pitfall's PF/sprite layering)
    #   seaquest  1104 -> 1043  (improved)
    #   enduro    1197 -> 774   (improved, PXC1 bit-exact via #84)
    # PXC1+PXC2 RAM stay bit-exact across all 21 cases.
    # Task #83 round 3 (2026-06-11): Y_START gate closes pong's last 8 px
    # row-0 HMOVE comb residual — **pong BIT-EXACT**.
    # Task #91 (2026-06-12): regenerated stale xitari fixtures for pitfall
    # (553→184, fixture was off by 5570 px) and enduro (774→660, off by
    # 1140 px). The previous "pitfall HUD bug" was largely a stale-fixture
    # artifact.
    # Task #91 round 2 / #94 (2026-06-12): VDELP/VDELBL shadow latches now
    # fire at ACTIVATION time inside the pending-writes drain (xitari's
    # `updateFrame(clock+delay)`-then-mutate ordering) instead of a single
    # poke-time snapshot per scanline. Pitfall's 6-digit kernel rewrites
    # GRP0/GRP1 4× per HUD row — each digit value lives in the shadow for
    # ~2 copy slots, so whole-row snapshots rendered wrong digit values
    # ("wrong format") + phantom copies. Cross-ROM (jutari == jaxtari):
    #   pitfall   166 -> 0     (BIT-EXACT — HUD time/score fully correct)
    #   seaquest 1087 -> 904   (best ever; was 1043 pre-#91)
    #   enduro   1133 -> 516   (best ever; was 660 pre-#91)
    #   pong / breakout / space_invaders stay BIT-EXACT.
    # Task #95 (2026-06-13): the screen tooling (this test's _SETTINGS,
    # jutari_screen_dump.jl, dump_jutari_frames.jl) was missing pitfall +
    # enduro RomSettings, so the jutari fixture and jaxtari live arm
    # rendered them with GENERIC settings (no getStartingActions) while
    # the xitari fixtures used real settings. Correcting the settings:
    #   enduro    516 -> 249  (settings-mismatch artifact removed; both ports)
    #   pitfall     0 ->   0  (noop-10 unaffected by the UP-start)
    # This also fixed the user-reported pitfall VIDEO bug (Harry not
    # jumping) — pitfall is now BIT-EXACT vs xitari across 60 frames of
    # random-action gameplay incl. the jump. The emulator was always
    # correct; only the screen tooling's settings map was wrong.
    _Case("breakout_noop_10",       "breakout.bin",          0),
    _Case("pong_noop_10",           "pong.bin",              0),
    _Case("space_invaders_noop_10", "space_invaders.bin",    0),
    _Case("pitfall_noop_10",        "pitfall.bin",           0),
    # Task #97 (2026-06-13): HMOVE-blank wrap fix (beam_sc>=76 wraps to the
    # next-line cycle instead of returning false + clobbering the current
    # line's comb). enduro free-running lines strobe HMOVE at the next
    # line's cyc~3 (beam_sc 79); the fix recovers ~half the missed combs:
    #   enduro  249 -> 137  (both ports; PXC2 preserved)
    # Remaining 137 = the NEXT-line comb attribution (the line whose HMOVE
    # was recorded early on the previous line) — needs a deferred/next
    # hmove-blank flag (task #97 follow-up). 4 bit-exact ROMs unaffected.
    _Case("seaquest_noop_10",       "seaquest.bin",        904),
    _Case("enduro_noop_10",         "enduro.bin",          137),
]


def _load_screens(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    raw = gzip.open(path, "rb").read()
    n = len(raw) // (_H * _W)
    return np.frombuffer(raw, dtype=np.uint8)[: n * _H * _W].reshape(n, _H, _W)


def _load_actions(path: Path) -> list[int]:
    out = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(int(s))
    return out


def _xitari(case: _Case):
    return _load_screens(_SCREENS / f"{case.name}.screen.gz")


def _jutari(case: _Case):
    return _load_screens(_SCREENS / f"{case.name}_jutari.screen.gz")


def _jaxtari_screens(case: _Case, n: int) -> np.ndarray:
    rom = np.frombuffer((_ROMS / case.rom).read_bytes(), dtype=np.uint8)
    factory = _SETTINGS.get(case.rom)
    env = StellaEnvironment(rom, factory()) if factory else StellaEnvironment(rom)
    env.reset(boot_noop_steps=60, boot_reset_steps=4)
    actions = _load_actions(_ACTIONS / f"{case.name}.txt")
    frames = []
    for a in actions[:n]:
        env.step(int(a))
        frames.append(np.asarray(env.get_screen(), dtype=np.uint8))
    return np.stack(frames)


def _per_frame_diffs(a: np.ndarray, b: np.ndarray) -> list[int]:
    n = min(len(a), len(b))
    return [int((a[i] != b[i]).sum()) for i in range(n)]


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_jutari_screen_matches_xitari(case: _Case):
    """xitari ↔ jutari per-frame screen diff (fixture vs fixture)."""
    xi, jt = _xitari(case), _jutari(case)
    if xi is None or jt is None:
        pytest.skip(f"[{case.name}] missing screen fixture(s) "
                    f"(xitari/jutari) — regenerate with tools/jutari_screen_dump.jl")
    diffs = _per_frame_diffs(xi, jt)
    worst = max(diffs)
    assert worst <= case.max_screen_diff, (
        f"[{case.name}] jutari↔xitari screen diff regressed: worst frame "
        f"{worst} px > pinned {case.max_screen_diff}. Per-frame: {diffs}. "
        f"If this is a real rendering fix, tighten max_screen_diff (and "
        f"regenerate the jutari screen fixture in the same commit).")


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_jaxtari_screen_matches_xitari(case: _Case):
    """xitari ↔ jaxtari per-frame screen diff (jaxtari live vs fixture)."""
    xi = _xitari(case)
    if xi is None:
        pytest.skip(f"[{case.name}] missing xitari screen fixture")
    jx = _jaxtari_screens(case, len(xi))
    diffs = _per_frame_diffs(xi, jx)
    worst = max(diffs)
    assert worst <= case.max_screen_diff, (
        f"[{case.name}] jaxtari↔xitari screen diff regressed: worst frame "
        f"{worst} px > pinned {case.max_screen_diff}. Per-frame: {diffs}. "
        f"If this is a real rendering fix, tighten max_screen_diff.")
