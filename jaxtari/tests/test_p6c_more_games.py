"""P6c — Breakout + Space Invaders RomSettings (P6c follow-ons to Pong).

Same shape as the Pong tests: synthetic-RAM Console + write the
canonical score / lives bytes, assert the per-step reward, lives, and
termination match expectations.
"""

from __future__ import annotations

import jax.numpy as jnp

from jaxtari.console import Console, initial_console
from jaxtari.games import (
    BREAKOUT_LIVES_ADDR,
    BREAKOUT_SCORE_HI_ADDR,
    BREAKOUT_SCORE_LO_ADDR,
    BreakoutRomSettings,
    KANGAROO_GAME_OVER,
    KANGAROO_LIVES_ADDR,
    KangarooRomSettings,
    ROADRUNNER_LIVES_ADDR,
    ROADRUNNER_XVEL_DEATH_ADDR,
    ROADRUNNER_YVEL_ADDR,
    RoadRunnerRomSettings,
    SI_LIVES_ADDR,
    SI_SCORE_HI_ADDR,
    SI_SCORE_LO_ADDR,
    SpaceInvadersRomSettings,
)


def _empty_rom() -> jnp.ndarray:
    rom = jnp.zeros((4096,), dtype=jnp.uint8)
    rom = rom.at[0x0FFC].set(jnp.uint8(0x00))
    rom = rom.at[0x0FFD].set(jnp.uint8(0xF0))
    return rom


def _console_with(addr_value_pairs: dict[int, int]) -> Console:
    """Build a console whose RAM cells (mod 0x7F) hold the requested
    bytes. The high-mirror addresses xitari uses (e.g. Space Invaders
    at $E8) wrap into the 128-byte RAM the same way the bus dispatch
    does."""
    console = initial_console(_empty_rom())
    new_ram = console.bus.ram
    for addr, val in addr_value_pairs.items():
        new_ram = new_ram.at[addr & 0x7F].set(jnp.uint8(val & 0xFF))
    return console._replace(bus=console.bus._replace(ram=new_ram))


# --------------------------------------------------------------------------- #
# Breakout
# --------------------------------------------------------------------------- #

def test_breakout_score_zero_at_start():
    s = BreakoutRomSettings()
    s.reset()
    c = _console_with({})
    # Fresh RAM → score = 0 → no delta.
    assert s.get_reward(c) == 0


def test_breakout_not_terminal_before_started():
    """Fresh RAM has lives=0 but the game hasn't STARTED yet (lives never
    observed at 5). The sticky `_started` latch (BreakoutRomSettings
    docstring) deliberately suppresses the boot-time false-terminal, so
    a fresh console must report NOT terminal — that's the latch's whole
    purpose. (Previously this test asserted the pre-latch behaviour.)"""
    s = BreakoutRomSettings()
    assert not s.is_terminal(_console_with({}))
    assert not s.is_terminal(_console_with({BREAKOUT_LIVES_ADDR: 0}))


def test_breakout_terminal_after_started_then_lives_zero():
    """Terminal = started && lives == 0. Observe lives==5 first (latches
    `_started`), then lives==0 → game over."""
    s = BreakoutRomSettings()
    assert not s.is_terminal(_console_with({BREAKOUT_LIVES_ADDR: 5}))  # start
    assert s.is_terminal(_console_with({BREAKOUT_LIVES_ADDR: 0}))      # then 0


def test_breakout_not_terminal_with_lives_remaining():
    s = BreakoutRomSettings()
    c = _console_with({BREAKOUT_LIVES_ADDR: 3})
    assert not s.is_terminal(c)


def test_breakout_lives_count():
    s = BreakoutRomSettings()
    c = _console_with({BREAKOUT_LIVES_ADDR: 5})
    assert s.lives(c) == 5


def test_breakout_score_decoding_simple():
    """Score = $76's high*100 + low*10 + $77's low nibble."""
    s = BreakoutRomSettings()
    s.reset()
    # First frame: score == 0
    s.get_reward(_console_with({}))
    # Second frame: $76 = 0x12 (hundreds=1, tens=2), $77 = 0x05 → 125
    r = s.get_reward(_console_with({
        BREAKOUT_SCORE_HI_ADDR: 0x12,
        BREAKOUT_SCORE_LO_ADDR: 0x05,
    }))
    assert r == 125


def test_breakout_reward_is_delta():
    s = BreakoutRomSettings()
    s.reset()
    # 50 → 60 = +10
    s.get_reward(_console_with({BREAKOUT_SCORE_HI_ADDR: 0x05, BREAKOUT_SCORE_LO_ADDR: 0x00}))
    r = s.get_reward(_console_with({BREAKOUT_SCORE_HI_ADDR: 0x06, BREAKOUT_SCORE_LO_ADDR: 0x00}))
    assert r == 10


def test_breakout_reset_clears_previous_score():
    s = BreakoutRomSettings()
    s.reset()
    s.get_reward(_console_with({BREAKOUT_SCORE_HI_ADDR: 0x99, BREAKOUT_SCORE_LO_ADDR: 0x09}))
    s.reset()
    # After reset, the next get_reward starts from score 0.
    assert s.get_reward(_console_with({})) == 0


# --------------------------------------------------------------------------- #
# Space Invaders
# --------------------------------------------------------------------------- #

def test_space_invaders_terminal_when_lives_zero():
    s = SpaceInvadersRomSettings()
    c = _console_with({SI_LIVES_ADDR: 0})
    assert s.is_terminal(c)


def test_space_invaders_lives_count():
    s = SpaceInvadersRomSettings()
    c = _console_with({SI_LIVES_ADDR: 3})
    assert s.lives(c) == 3


def test_space_invaders_score_decoding():
    """Score = (high BCD) * 100 + (low BCD). $E8=0x12, $E9=0x34 → 1234."""
    s = SpaceInvadersRomSettings()
    s.reset()
    s.get_reward(_console_with({}))
    r = s.get_reward(_console_with({
        SI_SCORE_HI_ADDR: 0x12,
        SI_SCORE_LO_ADDR: 0x34,
    }))
    assert r == 1234


def test_space_invaders_reward_is_delta():
    s = SpaceInvadersRomSettings()
    s.reset()
    s.get_reward(_console_with({SI_SCORE_HI_ADDR: 0x00, SI_SCORE_LO_ADDR: 0x50}))
    r = s.get_reward(_console_with({SI_SCORE_HI_ADDR: 0x01, SI_SCORE_LO_ADDR: 0x00}))
    # 0050 → 0100 = +50
    assert r == 50


def test_space_invaders_constants():
    assert SI_SCORE_HI_ADDR == 0xE8
    assert SI_SCORE_LO_ADDR == 0xE9
    assert SI_LIVES_ADDR    == 0xC9


def test_breakout_constants():
    assert BREAKOUT_SCORE_LO_ADDR == 0x77
    assert BREAKOUT_SCORE_HI_ADDR == 0x76
    assert BREAKOUT_LIVES_ADDR    == 0x39


def test_both_settings_satisfy_RomSettings_protocol():
    from jaxtari.games import RomSettings
    assert isinstance(BreakoutRomSettings(), RomSettings)
    assert isinstance(SpaceInvadersRomSettings(), RomSettings)


# --------------------------------------------------------------------------- #
# Road Runner — Cluster B (#127b). Mirror xitari RoadRunner.cpp::step.
# --------------------------------------------------------------------------- #

def test_road_runner_score_single_nibble_digits_low_first():
    # $C9..$CC are single-nibble decimal digits, LOW digit first; 0xA is the
    # "blank zero" sentinel. C9=2 CA=1 CB=0xA CC=0 → 2+10+0+0 = 12 → *100.
    s = RoadRunnerRomSettings()
    c = _console_with({0xC9: 2, 0xCA: 1, 0xCB: 0xA, 0xCC: 0})
    assert s.get_reward(c) == 1200


def test_road_runner_lives_low3_plus_one():
    s = RoadRunnerRomSettings()
    c = _console_with({ROADRUNNER_LIVES_ADDR: 2})
    assert s.lives(c) == 3


def test_road_runner_terminal_only_when_dead_and_moving():
    s = RoadRunnerRomSettings()
    # lives 0 but velocities 0 → not yet terminal (death anim hasn't started)
    c = _console_with({ROADRUNNER_LIVES_ADDR: 0,
                       ROADRUNNER_YVEL_ADDR: 0,
                       ROADRUNNER_XVEL_DEATH_ADDR: 0})
    assert s.is_terminal(c) is False
    # lives 0 and y-velocity nonzero → terminal
    c = _console_with({ROADRUNNER_LIVES_ADDR: 0, ROADRUNNER_YVEL_ADDR: 5})
    assert s.is_terminal(c) is True
    # lives 0 and death x-velocity nonzero → terminal
    c = _console_with({ROADRUNNER_LIVES_ADDR: 0, ROADRUNNER_XVEL_DEATH_ADDR: 7})
    assert s.is_terminal(c) is True
    # lives remaining → never terminal regardless of velocity
    c = _console_with({ROADRUNNER_LIVES_ADDR: 1, ROADRUNNER_YVEL_ADDR: 5})
    assert s.is_terminal(c) is False


# --------------------------------------------------------------------------- #
# Kangaroo — Cluster B (#127b). Mirror xitari Kangaroo.cpp::step.
# --------------------------------------------------------------------------- #

def test_kangaroo_score_bcd_high_low_times_100():
    # getDecimalScore(0xA8, 0xA7): A7 high (thousands+hundreds), A8 low
    # (tens+ones); *100. A7=0x12 A8=0x34 → 1234 → *100.
    s = KangarooRomSettings()
    c = _console_with({0xA7: 0x12, 0xA8: 0x34})
    assert s.get_reward(c) == 123400


def test_kangaroo_lives_low3_plus_one():
    s = KangarooRomSettings()
    c = _console_with({KANGAROO_LIVES_ADDR: 2})
    assert s.lives(c) == 3


def test_kangaroo_terminal_on_game_over_sentinel():
    s = KangarooRomSettings()
    c = _console_with({KANGAROO_LIVES_ADDR: KANGAROO_GAME_OVER})
    assert s.is_terminal(c) is True
    c = _console_with({KANGAROO_LIVES_ADDR: 0x00})
    assert s.is_terminal(c) is False


def test_cluster_b_settings_satisfy_RomSettings_protocol():
    from jaxtari.games import RomSettings
    assert isinstance(RoadRunnerRomSettings(), RomSettings)
    assert isinstance(KangarooRomSettings(), RomSettings)
