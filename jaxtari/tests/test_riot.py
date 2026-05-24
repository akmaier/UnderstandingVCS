"""P4 tests — RIOT M6532 timer + I/O ports.

The 128 B RAM is in P2 (bus); this file covers the timer (INTIM/INSTAT/
TIM*T) and the two I/O ports (SWCHA/SWCHB + their DDRs).
"""

import jax.numpy as jnp

from jaxtari.bus import initial_bus, peek, poke
from jaxtari.cpu.m6502 import step
from jaxtari.riot.system import (
    initial_riot_state,
    riot_advance,
    riot_peek,
    riot_poke,
    set_swcha_input,
    set_swchb_input,
)
from jaxtari.types import CPUState, initial_cpu_state


def _state(**fields) -> CPUState:
    s = initial_cpu_state()
    return s._replace(**{
        k: jnp.asarray(v).astype(getattr(s, k).dtype) for k, v in fields.items()
    })


# --------------------------------------------------------------------------- #
# Initial state
# --------------------------------------------------------------------------- #

def test_initial_riot_state():
    r = initial_riot_state()
    assert r.intim == 0
    assert r.prescaler_shift == 0
    assert r.cycles_since_tick == 0
    assert r.timer_expired is False
    assert r.swcha_in == 0xFF
    assert r.swchb_in == 0xFF
    assert r.swacnt == 0x00 and r.swbcnt == 0x00


def test_initial_port_reads_return_input_lines():
    """Default DDR=0 → ports report the input lines (0xFF on power-on)."""
    r = initial_riot_state()
    assert riot_peek(r, 0x0280)[0] == 0xFF     # SWCHA  (peek now returns (value, riot))
    assert riot_peek(r, 0x0282)[0] == 0xFF     # SWCHB


# --------------------------------------------------------------------------- #
# Timer write addresses + prescaler decode
# --------------------------------------------------------------------------- #

def test_tim1t_load():
    r = riot_poke(initial_riot_state(), 0x0294, 100)
    assert r.intim == 100
    assert r.prescaler_shift == 0


def test_tim8t_load():
    r = riot_poke(initial_riot_state(), 0x0295, 50)
    assert r.intim == 50
    assert r.prescaler_shift == 3            # 2**3 = 8


def test_tim64t_load():
    r = riot_poke(initial_riot_state(), 0x0296, 20)
    assert r.prescaler_shift == 6            # 2**6 = 64


def test_t1024t_load():
    r = riot_poke(initial_riot_state(), 0x0297, 5)
    assert r.prescaler_shift == 10           # 2**10 = 1024


def test_timer_load_clears_expired_flag():
    r = initial_riot_state()._replace(timer_expired=True)
    r = riot_poke(r, 0x0294, 1)
    assert r.timer_expired is False


# --------------------------------------------------------------------------- #
# riot_advance — pre-expiration
# --------------------------------------------------------------------------- #

def test_advance_under_one_tick_does_not_change_intim():
    r = riot_poke(initial_riot_state(), 0x0295, 50)        # TIM8T 50
    r = riot_advance(r, 5)                                  # 5 < 8
    assert r.intim == 50
    assert r.cycles_since_tick == 5


def test_advance_one_tick_decrements_intim():
    r = riot_poke(initial_riot_state(), 0x0295, 50)
    r = riot_advance(r, 8)
    assert r.intim == 49
    assert r.cycles_since_tick == 0


def test_advance_partial_then_full_tick():
    r = riot_poke(initial_riot_state(), 0x0295, 50)
    r = riot_advance(r, 5)
    r = riot_advance(r, 3)
    assert r.intim == 49


def test_advance_multiple_ticks_tim1t():
    r = riot_poke(initial_riot_state(), 0x0294, 100)
    r = riot_advance(r, 30)
    assert r.intim == 70


# --------------------------------------------------------------------------- #
# riot_advance — expiration boundary + post-expiration behaviour
# --------------------------------------------------------------------------- #

def test_advance_to_exact_expiration():
    """TIM1T value 10: 11 cycles → INTIM wraps to 0xFF and expired set."""
    r = riot_poke(initial_riot_state(), 0x0294, 10)
    r = riot_advance(r, 11)
    assert r.intim == 0xFF
    assert r.timer_expired is True


def test_post_expiration_ticks_once_per_cycle():
    r = riot_poke(initial_riot_state(), 0x0294, 10)
    r = riot_advance(r, 11)                       # → expired, intim=0xFF
    r = riot_advance(r, 5)                        # 5 cycles, 5 ticks
    assert r.intim == 0xFA
    assert r.timer_expired is True


def test_expiration_during_tim8t_advance():
    """TIM8T value 5 (cycles_to_expire = 6×8 = 48). Advance 50 cycles →
    expired at cycle 48; 2 extra cycles at post-expiration rate (1/cycle).
    Result: INTIM = (0xFF - 2) & 0xFF = 0xFD."""
    r = riot_poke(initial_riot_state(), 0x0295, 5)
    r = riot_advance(r, 50)
    assert r.timer_expired is True
    assert r.intim == 0xFD


def test_writing_timer_clears_expired_and_resets_prescaler():
    r = riot_poke(initial_riot_state(), 0x0294, 5)
    r = riot_advance(r, 100)
    assert r.timer_expired is True
    r = riot_poke(r, 0x0295, 100)
    assert r.timer_expired is False
    assert r.intim == 100
    assert r.cycles_since_tick == 0


# --------------------------------------------------------------------------- #
# INSTAT
# --------------------------------------------------------------------------- #

def test_instat_zero_before_expiration():
    r = riot_poke(initial_riot_state(), 0x0294, 5)
    assert riot_peek(r, 0x0285)[0] == 0x00


def test_instat_d7_set_after_expiration():
    r = riot_poke(initial_riot_state(), 0x0294, 5)
    r = riot_advance(r, 6)
    assert riot_peek(r, 0x0285)[0] & 0x80 != 0


def test_intim_readable_via_peek():
    r = riot_poke(initial_riot_state(), 0x0294, 42)
    assert riot_peek(r, 0x0284)[0] == 42


# --------------------------------------------------------------------------- #
# P4d — INTIM read clears the timer-expired latch (real MOS 6532 semantic).
# --------------------------------------------------------------------------- #

def test_p4d_intim_read_clears_timer_expired():
    r = riot_poke(initial_riot_state(), 0x0294, 1)
    r = riot_advance(r, 2)                          # expire
    assert r.timer_expired is True
    # INTIM read returns the value AND clears the latch.
    value, r_after = riot_peek(r, 0x0284)
    assert r_after.timer_expired is False
    # Sanity: subsequent INSTAT read sees the cleared latch.
    assert riot_peek(r_after, 0x0285)[0] & 0x80 == 0


def test_p4d_instat_read_does_NOT_clear_timer_expired():
    r = riot_poke(initial_riot_state(), 0x0294, 1)
    r = riot_advance(r, 2)
    assert r.timer_expired is True
    # INSTAT read returns value AND leaves the latch alone (PA7 clear
    # only — not modelled yet).
    _, r_after = riot_peek(r, 0x0285)
    assert r_after.timer_expired is True


# --------------------------------------------------------------------------- #
# I/O ports
# --------------------------------------------------------------------------- #

def test_swcha_input_reflected_when_ddr_zero():
    r = initial_riot_state()
    r = set_swcha_input(r, 0b10101010)
    assert riot_peek(r, 0x0280)[0] == 0b10101010


def test_swcha_output_reflected_when_ddr_one():
    r = initial_riot_state()
    r = riot_poke(r, 0x0281, 0xFF)               # SWACNT = all output
    r = riot_poke(r, 0x0280, 0x5A)               # SWCHA out = $5A
    assert riot_peek(r, 0x0280)[0] == 0x5A
    # Inputs are now ignored because all bits are configured as outputs.
    r = set_swcha_input(r, 0x00)
    assert riot_peek(r, 0x0280)[0] == 0x5A


def test_swcha_mixed_ddr_combines_input_and_output():
    """High nibble = output, low nibble = input."""
    r = initial_riot_state()
    r = riot_poke(r, 0x0281, 0xF0)               # SWACNT = $F0
    r = riot_poke(r, 0x0280, 0xA5)               # SWCHA out: high nibble used
    r = set_swcha_input(r, 0x33)                 # SWCHA in: low nibble used
    # Result: high nibble from out ($A0), low nibble from in ($03) → $A3.
    assert riot_peek(r, 0x0280)[0] == 0xA3


def test_swchb_input_reflected_when_ddr_zero():
    r = initial_riot_state()
    r = set_swchb_input(r, 0b01010101)
    assert riot_peek(r, 0x0282)[0] == 0b01010101


def test_ddr_registers_readable():
    r = initial_riot_state()
    r = riot_poke(r, 0x0281, 0xC3)
    r = riot_poke(r, 0x0283, 0x3C)
    assert riot_peek(r, 0x0281)[0] == 0xC3
    assert riot_peek(r, 0x0283)[0] == 0x3C


# --------------------------------------------------------------------------- #
# Bus integration
# --------------------------------------------------------------------------- #

def test_bus_riot_timer_load_and_read_via_cpu():
    """LDA #$50 / STA $295 / LDA $284 (INTIM) → after the STA + load, A=$50."""
    rom = jnp.zeros((4096,), dtype=jnp.uint8)
    program = [
        0xA9, 0x50,        # LDA #$50            (2 cyc)
        0x8D, 0x95, 0x02,  # STA $0295 (TIM8T)    (4 cyc)
        0xAD, 0x84, 0x02,  # LDA $0284 (INTIM)    (4 cyc)
    ]
    for i, b in enumerate(program):
        rom = rom.at[i].set(jnp.uint8(b))
    bus = initial_bus(rom)
    s = _state(PC=0xF000)
    s, bus = step(s, bus)                # LDA #$50
    assert int(s.A) == 0x50
    s, bus = step(s, bus)                # STA TIM8T
    assert bus.riot.intim == 0x50
    s, bus = step(s, bus)                # LDA INTIM
    # By the time LDA INTIM completes the timer has been ticked by ~10 cycles
    # (LDA + STA + LDA), but TIM8T's prescaler is 8 → just 1 tick so far.
    assert int(s.A) in (0x4F, 0x50)


def test_step_advances_riot_timer():
    """Run a few NOPs through the bus and check INTIM decreases under TIM1T."""
    rom = jnp.full((4096,), 0xEA, dtype=jnp.uint8)   # all NOPs (2 cycles each)
    bus = initial_bus(rom)
    bus = bus._replace(riot=riot_poke(bus.riot, 0x0294, 50))  # TIM1T = 50
    s = _state(PC=0xF000)
    for _ in range(10):                  # 20 cycles
        s, bus = step(s, bus)
    assert bus.riot.intim == 30          # 50 - 20
    assert not bus.riot.timer_expired


def test_step_wsync_stall_also_advances_riot():
    """STA WSYNC's stall cycles should also tick the RIOT timer."""
    rom = jnp.zeros((4096,), dtype=jnp.uint8)
    rom = rom.at[0].set(jnp.uint8(0x85))             # STA $02 = WSYNC
    rom = rom.at[1].set(jnp.uint8(0x02))
    bus = initial_bus(rom)
    bus = bus._replace(riot=riot_poke(bus.riot, 0x0294, 200))  # TIM1T = 200
    s = _state(PC=0xF000)
    s, bus = step(s, bus)
    # STA WSYNC takes 3 cyc + stall 73 = 76 cycles total. INTIM = 200 - 76 = 124.
    assert bus.riot.intim == 124
