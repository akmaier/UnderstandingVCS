"""P6c — Pitfall / BeamRider / Enduro / Seaquest tests."""

from __future__ import annotations

import jax.numpy as jnp

from jaxtari.console import Console, initial_console
from jaxtari.games import (
    BEAMRIDER_LIVES_ADDR,
    BEAMRIDER_SCORE_ADDRS,
    BEAMRIDER_TERM_ADDR,
    BeamriderRomSettings,
    ENDURO_DEATH_ADDR,
    ENDURO_DEATH_VALUE,
    ENDURO_LEVEL_ADDR,
    ENDURO_SCORE_HI_ADDR,
    ENDURO_SCORE_LO_ADDR,
    EnduroRomSettings,
    PITFALL_INITIAL_SCORE,
    PITFALL_LIVES_ADDR,
    PITFALL_LOGO_ADDR,
    PITFALL_SCORE_ADDRS,
    PitfallRomSettings,
    SEAQUEST_SCORE_ADDRS,
    SEAQUEST_TERM_ADDR,
    SeaquestRomSettings,
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
# Pitfall
# --------------------------------------------------------------------------- #

def test_pitfall_initial_score_is_2000():
    """Pitfall starts at 2000; the first reset() locks that in so the
    very first frame's reward is `current - 2000`."""
    s = PitfallRomSettings()
    s.reset()
    # Fresh RAM ($D7/$D6/$D5 = 0) → score 0 → reward = 0 - 2000.
    assert s.get_reward(_console_with({})) == -2000


def test_pitfall_score_decoding_three_bytes():
    """$D7=0x01, $D6=0x23, $D5=0x45 → 1·10000 + 23·100 + 45 = 12345.
    First call burns the 2000-initial-score baseline; the second's
    reward is just the delta within a single in-game run."""
    s = PitfallRomSettings()
    s.reset()
    s.get_reward(_console_with({}))               # consume the -2000 from the baseline
    r = s.get_reward(_console_with({
        PITFALL_SCORE_ADDRS[0]: 0x01,
        PITFALL_SCORE_ADDRS[1]: 0x23,
        PITFALL_SCORE_ADDRS[2]: 0x45,
    }))
    assert r == 12345


def test_pitfall_terminal_requires_lives_zero_and_logo_timer_nonzero():
    s = PitfallRomSettings()
    assert s.is_terminal(_console_with({PITFALL_LIVES_ADDR: 0x00, PITFALL_LOGO_ADDR: 0x42}))
    assert not s.is_terminal(_console_with({PITFALL_LIVES_ADDR: 0x00, PITFALL_LOGO_ADDR: 0x00}))


def test_pitfall_not_terminal_with_lives_remaining():
    s = PitfallRomSettings()
    assert not s.is_terminal(_console_with({PITFALL_LIVES_ADDR: 0xA0, PITFALL_LOGO_ADDR: 0xFF}))


def test_pitfall_lives_decoding():
    s = PitfallRomSettings()
    assert s.lives(_console_with({PITFALL_LIVES_ADDR: 0xA0})) == 3
    assert s.lives(_console_with({PITFALL_LIVES_ADDR: 0x80})) == 2
    assert s.lives(_console_with({PITFALL_LIVES_ADDR: 0x40})) == 1


# --------------------------------------------------------------------------- #
# BeamRider
# --------------------------------------------------------------------------- #

def test_beamrider_score_decoding():
    s = BeamriderRomSettings()
    s.reset()
    s.get_reward(_console_with({}))
    r = s.get_reward(_console_with({
        BEAMRIDER_SCORE_ADDRS[0]: 0x01,
        BEAMRIDER_SCORE_ADDRS[1]: 0x23,
        BEAMRIDER_SCORE_ADDRS[2]: 0x45,
    }))
    assert r == 12345


def test_beamrider_terminal_at_ff_sentinel():
    s = BeamriderRomSettings()
    assert s.is_terminal(_console_with({BEAMRIDER_TERM_ADDR: 0xFF}))
    assert not s.is_terminal(_console_with({BEAMRIDER_TERM_ADDR: 0x00}))


def test_beamrider_lives_is_byte_plus_one():
    s = BeamriderRomSettings()
    assert s.lives(_console_with({BEAMRIDER_LIVES_ADDR: 0x02})) == 3


# --------------------------------------------------------------------------- #
# Enduro
# --------------------------------------------------------------------------- #

def test_enduro_level_zero_yields_zero_reward():
    """At level 0 (waiting to start), score is constant 0 — the first
    real-frame reward is 0."""
    s = EnduroRomSettings()
    s.reset()
    r = s.get_reward(_console_with({
        ENDURO_LEVEL_ADDR: 0x00,
        ENDURO_SCORE_HI_ADDR: 0x99,           # nonsense values shouldn't matter
        ENDURO_SCORE_LO_ADDR: 0x99,
    }))
    assert r == 0


def test_enduro_level_one_baseline_200():
    """Level 1: score = 200 - cars_passed. With $AB=$00, $AC=$50 →
    cars=50 → score = 200 - 50 = 150."""
    s = EnduroRomSettings()
    s.reset()
    s.get_reward(_console_with({}))           # prev = 0
    r = s.get_reward(_console_with({
        ENDURO_LEVEL_ADDR: 0x01,
        ENDURO_SCORE_HI_ADDR: 0x00,
        ENDURO_SCORE_LO_ADDR: 0x50,
    }))
    assert r == 150


def test_enduro_level_two_baseline_300():
    s = EnduroRomSettings()
    s.reset()
    s.get_reward(_console_with({}))
    r = s.get_reward(_console_with({
        ENDURO_LEVEL_ADDR: 0x02,
        ENDURO_SCORE_HI_ADDR: 0x00,
        ENDURO_SCORE_LO_ADDR: 0x50,
    }))
    assert r == 250                                  # 300 - 50


def test_enduro_terminal_at_death_value():
    s = EnduroRomSettings()
    assert s.is_terminal(_console_with({ENDURO_DEATH_ADDR: ENDURO_DEATH_VALUE}))
    assert not s.is_terminal(_console_with({ENDURO_DEATH_ADDR: 0x00}))


# --------------------------------------------------------------------------- #
# Seaquest
# --------------------------------------------------------------------------- #

def test_seaquest_score_decoding():
    s = SeaquestRomSettings()
    s.reset()
    s.get_reward(_console_with({}))
    r = s.get_reward(_console_with({
        SEAQUEST_SCORE_ADDRS[0]: 0x12,
        SEAQUEST_SCORE_ADDRS[1]: 0x34,
        SEAQUEST_SCORE_ADDRS[2]: 0x56,
    }))
    assert r == 123456


def test_seaquest_terminal_when_a3_nonzero():
    s = SeaquestRomSettings()
    assert s.is_terminal(_console_with({SEAQUEST_TERM_ADDR: 0x42}))
    assert not s.is_terminal(_console_with({SEAQUEST_TERM_ADDR: 0x00}))


# --------------------------------------------------------------------------- #
# Protocol conformance
# --------------------------------------------------------------------------- #

def test_all_four_satisfy_RomSettings_protocol():
    from jaxtari.games import RomSettings
    for cls in (PitfallRomSettings, BeamriderRomSettings,
                EnduroRomSettings, SeaquestRomSettings):
        assert isinstance(cls(), RomSettings)
