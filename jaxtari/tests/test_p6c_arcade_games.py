"""P6c — Asteroids, Q*Bert, Ms Pac-Man `RomSettings` tests.

Same shape as the Pong / Breakout / Space Invaders tests: a tiny
synthetic-RAM console + write the canonical RAM cells, assert the
per-step reward, lives, terminal flag.
"""

from __future__ import annotations

import jax.numpy as jnp

from jaxtari.console import Console, initial_console
from jaxtari.games import (
    ASTEROIDS_LIVES_ADDR,
    ASTEROIDS_SCORE_HI_ADDR,
    ASTEROIDS_SCORE_LO_ADDR,
    AsteroidsRomSettings,
    MSPACMAN_DEATH_TIMER,
    MSPACMAN_DEATH_VALUE,
    MSPACMAN_LIVES_ADDR,
    MSPACMAN_SCORE_ADDRS,
    MsPacmanRomSettings,
    QBERT_LIVES_ADDR,
    QBERT_LIVES_END,
    QBERT_SCORE_ADDRS,
    QbertRomSettings,
)


def _empty_rom() -> jnp.ndarray:
    rom = jnp.zeros((4096,), dtype=jnp.uint8)
    rom = rom.at[0x0FFC].set(jnp.uint8(0x00))
    rom = rom.at[0x0FFD].set(jnp.uint8(0xF0))
    return rom


def _console_with(addr_value_pairs: dict[int, int]) -> Console:
    console = initial_console(_empty_rom())
    new_ram = console.bus.ram
    for addr, val in addr_value_pairs.items():
        new_ram = new_ram.at[addr & 0x7F].set(jnp.uint8(val & 0xFF))
    return console._replace(bus=console.bus._replace(ram=new_ram))


# --------------------------------------------------------------------------- #
# Asteroids
# --------------------------------------------------------------------------- #

def test_asteroids_score_zero_at_start():
    s = AsteroidsRomSettings()
    s.reset()
    assert s.get_reward(_console_with({})) == 0


def test_asteroids_score_decoding_with_x10_multiplier():
    """$3E=0x01, $3D=0x23 → 123, multiplied by 10 → 1230."""
    s = AsteroidsRomSettings()
    s.reset()
    s.get_reward(_console_with({}))
    r = s.get_reward(_console_with({
        ASTEROIDS_SCORE_HI_ADDR: 0x01,
        ASTEROIDS_SCORE_LO_ADDR: 0x23,
    }))
    assert r == 1230


def test_asteroids_score_wrap_handles_overflow():
    """If the BCD chain wraps from a high score back to near 0, the
    reward should be `+ASTEROIDS_WRAP` instead of a huge negative."""
    s = AsteroidsRomSettings()
    s.reset()
    s.get_reward(_console_with({
        ASTEROIDS_SCORE_HI_ADDR: 0x99,
        ASTEROIDS_SCORE_LO_ADDR: 0x99,
    }))                                              # 9999 * 10 = 99990
    r = s.get_reward(_console_with({
        ASTEROIDS_SCORE_HI_ADDR: 0x00,
        ASTEROIDS_SCORE_LO_ADDR: 0x10,
    }))                                              # 10 * 10 = 100
    # Naive delta = 100 - 99990 = -99890. With wrap: +100000 → 110.
    assert r == 110


def test_asteroids_lives_from_high_nibble():
    s = AsteroidsRomSettings()
    # $3C = 0x40 → 4 lives
    assert s.lives(_console_with({ASTEROIDS_LIVES_ADDR: 0x40})) == 4


def test_asteroids_terminal_when_lives_zero():
    s = AsteroidsRomSettings()
    assert s.is_terminal(_console_with({ASTEROIDS_LIVES_ADDR: 0x05}))    # high nibble = 0
    assert not s.is_terminal(_console_with({ASTEROIDS_LIVES_ADDR: 0x10}))  # 1 life


# --------------------------------------------------------------------------- #
# Q*Bert
# --------------------------------------------------------------------------- #

def test_qbert_terminal_when_lives_is_death_sentinel():
    s = QbertRomSettings()
    assert s.is_terminal(_console_with({QBERT_LIVES_ADDR: QBERT_LIVES_END}))


def test_qbert_not_terminal_with_default_lives():
    s = QbertRomSettings()
    assert not s.is_terminal(_console_with({QBERT_LIVES_ADDR: 0x02}))


def test_qbert_score_decoding_three_bytes():
    """Score = bcd($DB) * 10000 + bcd($DA) * 100 + bcd($D9). $DB=0x01,
    $DA=0x23, $D9=0x45 → 12345."""
    s = QbertRomSettings()
    s.reset()
    s.get_reward(_console_with({}))
    r = s.get_reward(_console_with({
        QBERT_SCORE_ADDRS[0]: 0x01,
        QBERT_SCORE_ADDRS[1]: 0x23,
        QBERT_SCORE_ADDRS[2]: 0x45,
        QBERT_LIVES_ADDR:     0x02,             # not terminal
    }))
    assert r == 12345


def test_qbert_no_reward_on_terminal_frame():
    """xitari skips score updates on the death frame so the agent
    doesn't see a huge negative on its last step."""
    s = QbertRomSettings()
    s.reset()
    s.get_reward(_console_with({}))
    r = s.get_reward(_console_with({
        QBERT_SCORE_ADDRS[0]: 0xFF,             # nonsense
        QBERT_LIVES_ADDR:     QBERT_LIVES_END,  # terminal
    }))
    assert r == 0


def test_qbert_lives_count_after_signed_cast():
    """Lives = signed(byte) + 2, clamped to 0."""
    s = QbertRomSettings()
    assert s.lives(_console_with({QBERT_LIVES_ADDR: 0x02})) == 4
    assert s.lives(_console_with({QBERT_LIVES_ADDR: 0x01})) == 3
    assert s.lives(_console_with({QBERT_LIVES_ADDR: 0xFF})) == 1     # -1 + 2


# --------------------------------------------------------------------------- #
# Ms Pac-Man
# --------------------------------------------------------------------------- #

def test_mspacman_terminal_requires_zero_lives_and_death_sentinel():
    s = MsPacmanRomSettings()
    assert s.is_terminal(_console_with({
        MSPACMAN_LIVES_ADDR:  0x00,
        MSPACMAN_DEATH_TIMER: MSPACMAN_DEATH_VALUE,
    }))


def test_mspacman_not_terminal_with_lives_remaining():
    s = MsPacmanRomSettings()
    assert not s.is_terminal(_console_with({
        MSPACMAN_LIVES_ADDR:  0x03,
        MSPACMAN_DEATH_TIMER: MSPACMAN_DEATH_VALUE,
    }))


def test_mspacman_not_terminal_when_lives_zero_but_no_death_sentinel():
    """Lives is 0 BUT death timer doesn't have the sentinel — still
    not terminal (animation hasn't fully played out)."""
    s = MsPacmanRomSettings()
    assert not s.is_terminal(_console_with({
        MSPACMAN_LIVES_ADDR:  0x00,
        MSPACMAN_DEATH_TIMER: 0x00,
    }))


def test_mspacman_score_decoding_three_bytes():
    """$F8=0x12, $F9=0x34, $FA=0x56 → 123456."""
    s = MsPacmanRomSettings()
    s.reset()
    s.get_reward(_console_with({}))
    r = s.get_reward(_console_with({
        MSPACMAN_SCORE_ADDRS[0]: 0x12,
        MSPACMAN_SCORE_ADDRS[1]: 0x34,
        MSPACMAN_SCORE_ADDRS[2]: 0x56,
    }))
    assert r == 123456


def test_mspacman_lives_from_low_nibble():
    s = MsPacmanRomSettings()
    assert s.lives(_console_with({MSPACMAN_LIVES_ADDR: 0x03})) == 3
    # High-nibble noise must not bleed in.
    assert s.lives(_console_with({MSPACMAN_LIVES_ADDR: 0xF3})) == 3


# --------------------------------------------------------------------------- #
# Protocol conformance
# --------------------------------------------------------------------------- #

def test_all_three_satisfy_RomSettings_protocol():
    from jaxtari.games import RomSettings
    for cls in (AsteroidsRomSettings, QbertRomSettings, MsPacmanRomSettings):
        assert isinstance(cls(), RomSettings)
