"""Regression test for the breakout-frame-92 ball-doesn't-die bug.

Root cause was the VDELBL shadow latch capturing a STALE ENABL value
when a deferred ENABL write hadn't yet activated at GRP1-write time
— see jutari commit `20b5de0` and `a418e4c` (jaxtari mirror).

This test runs jaxtari's breakout with the canonical seed-42
random-action stream and asserts the lives counter (RAM[$39])
decrements 5 → 4 → 3 → 2 → 1 → 0 at the expected frames matching
xitari (±2 frames tolerance). Future regressions of the shadow-latch
fix will fail this test immediately.

Mirror of the `@testset "breakout ball-death"` block in
`jutari/test/runtests.jl`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jaxtari.env.stella_environment import StellaEnvironment
from jaxtari.games.breakout import BreakoutRomSettings


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BREAKOUT_ROM = _REPO_ROOT / "xitari" / "roms" / "breakout.bin"
_ACTIONS_PATH = (_REPO_ROOT / "tools" / "breakout_video" / "output"
                 / "breakout_random_actions.txt")

# xitari-measured lives transitions for the canonical seed-42 action
# stream. ±2 frame tolerance covers timing slop without masking real
# regressions.
_EXPECTED = [(1, 5), (117, 4), (237, 3), (357, 2), (477, 1), (597, 0)]


@pytest.mark.skipif(not _BREAKOUT_ROM.exists() or not _ACTIONS_PATH.exists(),
                    reason="breakout ROM or action stream not present")
def test_breakout_lives_decrement_matches_xitari():
    """Lives counter (RAM[$39]) decrements every ~120 frames matching xitari.

    Regression for the VDELBL shadow / deferred-ENABL bug fixed in
    jaxtari commit `a418e4c`. Before the fix, jaxtari's `enabl_old`
    could retain a stale "ball enabled" bit across an ENABL clear that
    hadn't activated by GRP1-write time, causing spurious BL-PF
    collisions, a wrong CPU branch, and ball-doesn't-die behavior.
    """
    rom = np.fromfile(_BREAKOUT_ROM, dtype=np.uint8)
    env = StellaEnvironment(rom, BreakoutRomSettings())
    env.reset(boot_noop_steps=60, boot_reset_steps=4)
    actions = [int(l.strip()) for l in _ACTIONS_PATH.read_text().splitlines()
               if l.strip() and not l.startswith("#")]

    transitions: list[tuple[int, int]] = []
    prev = -1
    for i, act in enumerate(actions[:600], start=1):
        env.step(int(act))
        lives = int(np.asarray(env.get_ram())[0x39])
        if lives != prev:
            transitions.append((i, lives))
            prev = lives

    assert len(transitions) == len(_EXPECTED), (
        f"unexpected number of lives transitions: got {transitions}, "
        f"expected {_EXPECTED}"
    )
    for got, exp in zip(transitions, _EXPECTED):
        assert got[1] == exp[1], (
            f"lives counter went to {got[1]} at frame {got[0]}, "
            f"expected {exp[1]} at frame ~{exp[0]}"
        )
        assert abs(got[0] - exp[0]) <= 2, (
            f"lives reached {got[1]} at frame {got[0]}, expected ~{exp[0]} "
            f"(off by {abs(got[0] - exp[0])} frames; tolerance ±2)"
        )
