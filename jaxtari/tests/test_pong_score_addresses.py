"""Regression test for task #78 — pong score addresses must be $0D/$0E.

xitari/games/supported/Pong.cpp::step() reads:
    int x = readRam(&system, 13);  // cpu score, = $0D
    int y = readRam(&system, 14);  // player score, = $0E

Before this fix, jaxtari + jutari pong settings used $14/$15 (the
addresses are a docstring leftover from an earlier guess). Those
cells hold sprite-pattern bytes that briefly hit 0x82=130 within ~60
frames of FIRE, making `max(0, 130) >= 21` return True. The env then
returned early on `terminal=True` and the user paddle froze.

This test pins the constants and the live-RAM read path so regressing
the addresses (or accidentally peeking $14/$15 again) is caught at
unit-test time, not after a costly side-by-side video re-render.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path('/Users/maier/Documents/code/UnderstandingVCS') / 'jaxtari'))

import numpy as np

from jaxtari.games.pong import (
    PongRomSettings,
    PONG_P0_SCORE_ADDR,
    PONG_P1_SCORE_ADDR,
    PONG_TARGET_SCORE,
    _scores,
)


def test_score_address_constants_match_xitari():
    """Sanity-check the module-level address constants."""
    assert PONG_P0_SCORE_ADDR == 0x0D, (
        f"PONG_P0_SCORE_ADDR = {PONG_P0_SCORE_ADDR:#x}; xitari uses 0x0D "
        f"(readRam(&system, 13)). $14 was the WRONG legacy value."
    )
    assert PONG_P1_SCORE_ADDR == 0x0E, (
        f"PONG_P1_SCORE_ADDR = {PONG_P1_SCORE_ADDR:#x}; xitari uses 0x0E "
        f"(readRam(&system, 14)). $15 was the WRONG legacy value."
    )
    assert PONG_TARGET_SCORE == 21


def test_pong_terminal_does_not_falsely_trigger_during_play():
    """The original bug: env.terminal became True after ~60 frames of
    FIRE+LEFT because $15 reads back >= 21 from sprite-pattern data.
    Post-fix the scores stay at 0 and terminal remains False.
    """
    from jaxtari.env.stella_environment import StellaEnvironment

    rom = np.fromfile(
        '/Users/maier/Documents/code/UnderstandingVCS/xitari/roms/pong.bin',
        dtype=np.uint8)
    env = StellaEnvironment(rom, PongRomSettings())
    env.reset(boot_noop_steps=60, boot_reset_steps=4)
    env.step(1)              # FIRE (start the ball)
    for _ in range(80):
        env.step(4)          # PLAYER_A_LEFTFIRE
        assert not env.game_over(), (
            "pong env.terminal fired during normal play — score-address "
            "regression? Check RAM[$14]/[$15] are not being read as scores."
        )
    # Direct settings hook: read the actual scores via the public path.
    p0, p1 = _scores(env._console)
    assert 0 <= p0 < PONG_TARGET_SCORE, f"P0 score out of range: {p0}"
    assert 0 <= p1 < PONG_TARGET_SCORE, f"P1 score out of range: {p1}"


if __name__ == '__main__':
    test_score_address_constants_match_xitari()
    test_pong_terminal_does_not_falsely_trigger_during_play()
    print('PASS — pong score addresses correctly map to xitari $0D/$0E.')
