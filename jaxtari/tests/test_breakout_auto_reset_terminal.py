"""Regression test for jaxtari BreakoutRomSettings auto-reset terminal latch.

Asserts:
  - After env.reset(), env.game_over() == False (we have 5 lives).
  - After 600 frames of the canonical seed-42 action stream,
    env.game_over() flips True at frame ~597, matching xitari (the
    moment the last ball is lost).

Pinning this means a future regression of BreakoutRomSettings's
started+terminal latch (commit 69a53a2 — xitari-faithful semantics)
would break the test immediately rather than silently breaking the
auto-reset path that the comparison-video tool depends on.

Mirror of the `@testset "breakout auto-reset terminal latch (task #76)"`
block in `jutari/test/runtests.jl`.
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


@pytest.mark.skipif(not _BREAKOUT_ROM.exists() or not _ACTIONS_PATH.exists(),
                    reason="breakout ROM or action stream not present")
def test_breakout_auto_reset_terminal_latch():
    rom = np.fromfile(_BREAKOUT_ROM, dtype=np.uint8)
    env = StellaEnvironment(rom, BreakoutRomSettings())
    env.reset(boot_noop_steps=60, boot_reset_steps=4)
    # Post-reset: terminal must be False (we have 5 lives).
    assert env.game_over() is False

    with open(_ACTIONS_PATH) as f:
        actions = [int(l.strip()) for l in f
                   if l.strip() and not l.startswith('#')][:600]

    # Walk for 600 frames; terminal must flip True at frame ~597
    # (xitari measured: lives → 0 at frame 597).
    flipped_at = -1
    for i, a in enumerate(actions, start=1):
        env.step(int(a))
        if env.game_over() and flipped_at < 0:
            flipped_at = i
            break
    assert flipped_at > 0, "env.game_over() never flipped True over 600 frames"
    assert abs(flipped_at - 597) <= 2, \
        f"Terminal flipped at frame {flipped_at}; expected 597 ±2 (xitari ground-truth)"
