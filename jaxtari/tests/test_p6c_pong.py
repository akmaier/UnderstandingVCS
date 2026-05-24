"""P6c (Pong) tests — game-specific score detection + termination.

PongRomSettings reads the per-frame P0 / P1 score bytes from RAM at
the canonical $14 / $15 addresses and computes
`reward = ΔP0 − ΔP1`, `terminal = max(P0, P1) ≥ 21`.

We test against a tiny synthetic-RAM `Console` (no real Pong ROM
needed) so the tests stay fast and the scoring contract is verified
in isolation from xitari emulation.
"""

from __future__ import annotations

import jax.numpy as jnp

from jaxtari.bus.system import Bus, initial_bus
from jaxtari.console import Console, initial_console
from jaxtari.games import (
    PONG_P0_SCORE_ADDR,
    PONG_P1_SCORE_ADDR,
    PONG_TARGET_SCORE,
    PongRomSettings,
)


# A 4 KB cart that just jumps to itself — enough to give `initial_console`
# something to chew on. We mutate the resulting console's RAM directly to
# simulate Pong-cart score writes.
def _empty_rom() -> jnp.ndarray:
    rom = jnp.zeros((4096,), dtype=jnp.uint8)
    rom = rom.at[0x0FFC].set(jnp.uint8(0x00))      # reset vector lo
    rom = rom.at[0x0FFD].set(jnp.uint8(0xF0))      # reset vector hi
    return rom


def _console_with_scores(p0: int, p1: int) -> Console:
    """Build a console whose RAM cells $14 / $15 hold the requested
    scores. Everything else stays at the default."""
    console = initial_console(_empty_rom())
    new_ram = (
        console.bus.ram
        .at[PONG_P0_SCORE_ADDR].set(jnp.uint8(p0 & 0xFF))
        .at[PONG_P1_SCORE_ADDR].set(jnp.uint8(p1 & 0xFF))
    )
    return console._replace(bus=console.bus._replace(ram=new_ram))


# --------------------------------------------------------------------------- #
# Reward — deltas from frame to frame
# --------------------------------------------------------------------------- #

def test_reward_zero_at_episode_start_with_zero_score():
    s = PongRomSettings()
    s.reset()
    c = _console_with_scores(0, 0)
    assert s.get_reward(c) == 0


def test_reward_plus_one_when_p0_scores():
    s = PongRomSettings()
    s.reset()
    # Frame 1: still 0-0
    s.get_reward(_console_with_scores(0, 0))
    # Frame 2: P0 scored → reward = +1
    assert s.get_reward(_console_with_scores(1, 0)) == 1


def test_reward_minus_one_when_p1_scores():
    s = PongRomSettings()
    s.reset()
    s.get_reward(_console_with_scores(0, 0))
    assert s.get_reward(_console_with_scores(0, 1)) == -1


def test_reward_zero_when_both_score_same_frame():
    """Unlikely in practice but the delta math should still cancel."""
    s = PongRomSettings()
    s.reset()
    s.get_reward(_console_with_scores(3, 5))
    assert s.get_reward(_console_with_scores(4, 6)) == 0


def test_reward_only_counts_delta_not_absolute_score():
    """A jump from 10-5 to 12-5 is +2, not +12 — the absolute score
    must not leak into the reward."""
    s = PongRomSettings()
    s.reset()
    s.get_reward(_console_with_scores(10, 5))
    assert s.get_reward(_console_with_scores(12, 5)) == 2


def test_reset_clears_previous_score_state():
    """After reset, the next get_reward must compare against 0-0, not
    against whatever was cached at the end of the previous episode."""
    s = PongRomSettings()
    s.reset()
    s.get_reward(_console_with_scores(15, 12))
    s.reset()
    # Next call should see (0 - 0) for both — reward = 0.
    assert s.get_reward(_console_with_scores(0, 0)) == 0


# --------------------------------------------------------------------------- #
# Termination — max score hits the target
# --------------------------------------------------------------------------- #

def test_not_terminal_below_target():
    s = PongRomSettings()
    s.reset()
    assert not s.is_terminal(_console_with_scores(20, 20))


def test_terminal_when_p0_reaches_target():
    s = PongRomSettings()
    s.reset()
    assert s.is_terminal(_console_with_scores(PONG_TARGET_SCORE, 0))


def test_terminal_when_p1_reaches_target():
    s = PongRomSettings()
    s.reset()
    assert s.is_terminal(_console_with_scores(0, PONG_TARGET_SCORE))


def test_terminal_remains_after_overshoot():
    """If the cart for some reason writes past 21, terminal stays True
    (the comparison is ≥, not ==)."""
    s = PongRomSettings()
    s.reset()
    assert s.is_terminal(_console_with_scores(25, 0))


# --------------------------------------------------------------------------- #
# Lives + protocol conformance
# --------------------------------------------------------------------------- #

def test_pong_has_no_lives():
    s = PongRomSettings()
    s.reset()
    assert s.lives(_console_with_scores(5, 5)) == 0


def test_pong_settings_satisfies_RomSettings_protocol():
    from jaxtari.games import RomSettings
    s = PongRomSettings()
    assert isinstance(s, RomSettings)


# --------------------------------------------------------------------------- #
# Constants exposed at module top-level
# --------------------------------------------------------------------------- #

def test_constants_match_xitari_convention():
    assert PONG_P0_SCORE_ADDR == 0x14
    assert PONG_P1_SCORE_ADDR == 0x15
    assert PONG_TARGET_SCORE  == 21
