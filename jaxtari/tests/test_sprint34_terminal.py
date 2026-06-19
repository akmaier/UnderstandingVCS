"""#127b sprint 3+4 — terminal/auto-reset readers for the remaining 8
long-horizon games (berzerk, montezuma_revenge, riverraid, asterix, pooyan,
phoenix, pacman, ms_pacman).

Synthetic-RAM Console + write the canonical terminal/score bytes, assert the
per-game predicate matches xitari's `<Game>.cpp::step()` (mirrored from jutari
TerminalGames.jl / JoystickGames.jl). Same shape as
tests/test_p6c_more_games.py's Cluster B cases.

Each game's terminal must be FALSE at its documented boot RAM, so the
non-resetting in-window 64-ROM sweep cannot freeze (the deaths all land far
past the 60-frame window).
"""

from __future__ import annotations

import jax.numpy as jnp

from jaxtari.console import Console, initial_console
from jaxtari.games import (
    ASTERIX_DEATH_ADDR,
    ASTERIX_LIVES_ADDR,
    AsterixRomSettings,
    BERZERK_GAME_OVER,
    BERZERK_LIVES_ADDR,
    BerzerkRomSettings,
    MONTEZUMA_DEATH_STATE,
    MONTEZUMA_LIVES_ADDR,
    MONTEZUMA_STATE_ADDR,
    MontezumaRevengeRomSettings,
    MsPacmanRomSettings,
    PACMAN_ANIM_ADDR,
    PACMAN_DEATH_ANIM,
    PACMAN_LIVES_ADDR,
    PacmanRomSettings,
    PHOENIX_DEATH_STATE,
    PHOENIX_LIVES_ADDR,
    PHOENIX_STATE_ADDR,
    PhoenixRomSettings,
    POOYAN_DEATH_STATE,
    POOYAN_LIVES_ADDR,
    POOYAN_STATE_ADDR,
    PooyanRomSettings,
    RIVERRAID_DEATH_BYTE,
    RIVERRAID_LIVES_ADDR,
    RIVERRAID_PREDEATH,
    RiverRaidRomSettings,
    RomSettings,
)
from jaxtari.games.more_games import (
    BERZERK_SCORE_ADDRS,
    MONTEZUMA_SCORE_ADDRS,
    RIVERRAID_SCORE_ADDRS,
    _riverraid_digit,
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
# Berzerk — terminal RAM[$DA]==0xFF; score 3 BCD bytes $5D/$5E/$5F; lives byte+1
# --------------------------------------------------------------------------- #

def test_berzerk_terminal_on_ff_sentinel():
    s = BerzerkRomSettings()
    assert s.is_terminal(_console_with({BERZERK_LIVES_ADDR: BERZERK_GAME_OVER})) is True
    assert s.is_terminal(_console_with({BERZERK_LIVES_ADDR: 0x00})) is False
    assert s.is_terminal(_console_with({BERZERK_LIVES_ADDR: 0x02})) is False


def test_berzerk_lives_byte_plus_one():
    s = BerzerkRomSettings()
    assert s.lives(_console_with({BERZERK_LIVES_ADDR: 0x02})) == 3


def test_berzerk_score_3byte_bcd_high_first():
    # (0x5D high, 0x5E mid, 0x5F low): 0x01 0x23 0x45 → 12345.
    hi, mid, lo = BERZERK_SCORE_ADDRS
    s = BerzerkRomSettings()
    s.reset()
    s.get_reward(_console_with({}))
    r = s.get_reward(_console_with({hi: 0x01, mid: 0x23, lo: 0x45}))
    assert r == 12345


def test_berzerk_boot_not_terminal():
    # Boot livesByte != 0xFF → not terminal (the death is at f581, far past the
    # 60-frame in-window sweep).
    s = BerzerkRomSettings()
    assert s.is_terminal(_console_with({BERZERK_LIVES_ADDR: 0x06})) is False


# --------------------------------------------------------------------------- #
# Montezuma's Revenge — terminal RAM[$BA]==0 && RAM[$B6]==0x60; lives (&7)+1
# --------------------------------------------------------------------------- #

def test_montezuma_terminal_needs_both_bytes():
    s = MontezumaRevengeRomSettings()
    # lives 0 AND state 0x60 → terminal
    assert s.is_terminal(_console_with({MONTEZUMA_LIVES_ADDR: 0,
                                        MONTEZUMA_STATE_ADDR: MONTEZUMA_DEATH_STATE})) is True
    # lives 0 but wrong state → not terminal
    assert s.is_terminal(_console_with({MONTEZUMA_LIVES_ADDR: 0,
                                        MONTEZUMA_STATE_ADDR: 0x00})) is False
    # state 0x60 but lives remaining → not terminal
    assert s.is_terminal(_console_with({MONTEZUMA_LIVES_ADDR: 3,
                                        MONTEZUMA_STATE_ADDR: MONTEZUMA_DEATH_STATE})) is False


def test_montezuma_lives_low3_plus_one():
    s = MontezumaRevengeRomSettings()
    assert s.lives(_console_with({MONTEZUMA_LIVES_ADDR: 5})) == 6  # (5&7)+1


def test_montezuma_score_3byte_bcd():
    lo_addr, mid_addr, hi_addr = MONTEZUMA_SCORE_ADDRS  # (0x93, 0x94, 0x95) high-first
    s = MontezumaRevengeRomSettings()
    s.reset()
    s.get_reward(_console_with({}))
    r = s.get_reward(_console_with({lo_addr: 0x00, mid_addr: 0x02, hi_addr: 0x50}))
    assert r == 250   # 0*10000 + 02*100 + 50


def test_montezuma_boot_not_terminal():
    # Starts with 6 lives; death is at f867.
    s = MontezumaRevengeRomSettings()
    assert s.is_terminal(_console_with({MONTEZUMA_LIVES_ADDR: 6,
                                        MONTEZUMA_STATE_ADDR: 0x00})) is False


# --------------------------------------------------------------------------- #
# River Raid — STATEFUL terminal: RAM[$C0] transitions 0x59 → 0x58.
# --------------------------------------------------------------------------- #

def test_riverraid_terminal_only_on_59_to_58_transition():
    s = RiverRaidRomSettings()
    s.reset()   # seeds prev=0x58
    # First frame at 0x59 (predeath): prev was 0x58 → not terminal, prev←0x59.
    assert s.is_terminal(_console_with({RIVERRAID_LIVES_ADDR: RIVERRAID_PREDEATH})) is False
    # Next frame at 0x58: prev was 0x59 → TERMINAL (the dying transition).
    assert s.is_terminal(_console_with({RIVERRAID_LIVES_ADDR: RIVERRAID_DEATH_BYTE})) is True


def test_riverraid_no_terminal_at_steady_58():
    s = RiverRaidRomSettings()
    s.reset()   # prev=0x58
    # Staying at 0x58 (boot/steady) never fires: prev stays 0x58, not 0x59.
    assert s.is_terminal(_console_with({RIVERRAID_LIVES_ADDR: 0x58})) is False
    assert s.is_terminal(_console_with({RIVERRAID_LIVES_ADDR: 0x58})) is False


def test_riverraid_reset_reseeds_prev_byte():
    s = RiverRaidRomSettings()
    s.reset()
    s.is_terminal(_console_with({RIVERRAID_LIVES_ADDR: RIVERRAID_PREDEATH}))  # prev←0x59
    s.reset()   # prev back to 0x58
    # After reset, a 0x58 read does NOT fire (prev is 0x58, not 0x59).
    assert s.is_terminal(_console_with({RIVERRAID_LIVES_ADDR: 0x58})) is False


def test_riverraid_digit_lut():
    assert _riverraid_digit(0) == 0
    assert _riverraid_digit(8) == 1
    assert _riverraid_digit(72) == 9
    assert _riverraid_digit(80) == 0   # 8*10 out of [0,72] → 0
    assert _riverraid_digit(7) == 0    # not a multiple of 8 → 0


def test_riverraid_score_low_digit_first():
    # addrs LOW-first; place digit 1 (=val 8) at the lowest addr → score 1.
    lo = RIVERRAID_SCORE_ADDRS[0]
    s = RiverRaidRomSettings()
    s.reset()
    s.get_reward(_console_with({}))
    r = s.get_reward(_console_with({lo: 8}))   # digit 1 in the ones place
    assert r == 1


def test_riverraid_lives_numeric():
    s = RiverRaidRomSettings()
    assert s.lives(_console_with({RIVERRAID_LIVES_ADDR: 0x58})) == 4
    assert s.lives(_console_with({RIVERRAID_LIVES_ADDR: 0x59})) == 1
    assert s.lives(_console_with({RIVERRAID_LIVES_ADDR: 16})) == 3   # 16/8 + 1


# --------------------------------------------------------------------------- #
# Asterix — terminal RAM[$C7]==0x01 && (RAM[$D3]&0xF)==1
# --------------------------------------------------------------------------- #

def test_asterix_terminal_needs_death_counter_and_one_life():
    s = AsterixRomSettings()
    assert s.is_terminal(_console_with({ASTERIX_DEATH_ADDR: 0x01,
                                        ASTERIX_LIVES_ADDR: 0x01})) is True
    # death counter not 1
    assert s.is_terminal(_console_with({ASTERIX_DEATH_ADDR: 0x00,
                                        ASTERIX_LIVES_ADDR: 0x01})) is False
    # more than 1 life
    assert s.is_terminal(_console_with({ASTERIX_DEATH_ADDR: 0x01,
                                        ASTERIX_LIVES_ADDR: 0x03})) is False


def test_asterix_keeps_fire_starting_action():
    # Extending the class must NOT drop its starting action.
    assert AsterixRomSettings().starting_actions() == [1]


# --------------------------------------------------------------------------- #
# Pooyan — terminal RAM[$96]==0 && RAM[$98]==0x05; keeps render props.
# --------------------------------------------------------------------------- #

def test_pooyan_terminal_needs_both_bytes():
    s = PooyanRomSettings()
    assert s.is_terminal(_console_with({POOYAN_LIVES_ADDR: 0,
                                        POOYAN_STATE_ADDR: POOYAN_DEATH_STATE})) is True
    assert s.is_terminal(_console_with({POOYAN_LIVES_ADDR: 0,
                                        POOYAN_STATE_ADDR: 0x00})) is False
    assert s.is_terminal(_console_with({POOYAN_LIVES_ADDR: 2,
                                        POOYAN_STATE_ADDR: POOYAN_DEATH_STATE})) is False


def test_pooyan_preserves_render_props():
    s = PooyanRomSettings()
    assert s.screen_height() == 220
    assert s.screen_y_start() == 26


def test_pooyan_boot_not_terminal():
    # Boot RAM[$96]=2 → not terminal (death at f1532).
    s = PooyanRomSettings()
    assert s.is_terminal(_console_with({POOYAN_LIVES_ADDR: 2})) is False


# --------------------------------------------------------------------------- #
# Pac-Man — terminal RAM[$98]==0 && RAM[$E4]==0x3F; keeps y_start=33.
# --------------------------------------------------------------------------- #

def test_pacman_terminal_needs_both_bytes():
    s = PacmanRomSettings()
    assert s.is_terminal(_console_with({PACMAN_LIVES_ADDR: 0,
                                        PACMAN_ANIM_ADDR: PACMAN_DEATH_ANIM})) is True
    assert s.is_terminal(_console_with({PACMAN_LIVES_ADDR: 0,
                                        PACMAN_ANIM_ADDR: 0x00})) is False
    assert s.is_terminal(_console_with({PACMAN_LIVES_ADDR: 3,
                                        PACMAN_ANIM_ADDR: PACMAN_DEATH_ANIM})) is False


def test_pacman_preserves_y_start():
    assert PacmanRomSettings().screen_y_start() == 33


def test_pacman_boot_not_terminal():
    # Boot RAM[$98]=3 → not terminal (death at f1771).
    s = PacmanRomSettings()
    assert s.is_terminal(_console_with({PACMAN_LIVES_ADDR: 3})) is False


# --------------------------------------------------------------------------- #
# Ms Pac-Man — already had a terminal reader; reconfirm (lives==0 && $A7==0x53).
# --------------------------------------------------------------------------- #

def test_mspacman_terminal_and_hmove_blanks_preserved():
    s = MsPacmanRomSettings()
    # lives low-nibble 0 AND death timer 0x53 → terminal
    assert s.is_terminal(_console_with({0xFB: 0x00, 0xA7: 0x53})) is True
    assert s.is_terminal(_console_with({0xFB: 0x02, 0xA7: 0x53})) is False
    # the render prop must survive (carries hmove_blanks=False)
    assert s.hmove_blanks() is False


# --------------------------------------------------------------------------- #
# Phoenix — terminal RAM[$CC]==0x80; lives (&7); new class.
# --------------------------------------------------------------------------- #

def test_phoenix_terminal_on_state_byte():
    s = PhoenixRomSettings()
    assert s.is_terminal(_console_with({PHOENIX_STATE_ADDR: PHOENIX_DEATH_STATE})) is True
    assert s.is_terminal(_console_with({PHOENIX_STATE_ADDR: 0x00})) is False


def test_phoenix_lives_low3():
    s = PhoenixRomSettings()
    assert s.lives(_console_with({PHOENIX_LIVES_ADDR: 5})) == 5


def test_phoenix_boot_not_terminal():
    # Boot RAM[$CC]=0 → not terminal (death at f1743).
    s = PhoenixRomSettings()
    assert s.is_terminal(_console_with({PHOENIX_STATE_ADDR: 0x00})) is False


# --------------------------------------------------------------------------- #
# Protocol conformance for all eight.
# --------------------------------------------------------------------------- #

def test_sprint34_settings_satisfy_RomSettings_protocol():
    for cls in (BerzerkRomSettings, MontezumaRevengeRomSettings,
                RiverRaidRomSettings, AsterixRomSettings, PooyanRomSettings,
                PhoenixRomSettings, PacmanRomSettings, MsPacmanRomSettings):
        assert isinstance(cls(), RomSettings), cls.__name__
