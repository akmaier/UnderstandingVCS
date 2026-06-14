"""P4 tests — RIOT M6532 timer + I/O ports.

The 128 B RAM is in P2 (bus); this file covers the timer (INTIM/INSTAT/
TIM*T) and the two I/O ports (SWCHA/SWCHB + their DDRs).

Task #80: the timer is now a **faithful lazy port of xitari's `M6532`** —
INTIM/INSTAT are computed on read from a monotonic cycle counter rather
than decremented eagerly. `riot_advance` is a no-op. The read formula is
`myTimer - (delta >> shift) - 1` where `delta = (cur_cycles - 1) -
cycles_when_set`, so every read here passes an explicit `cur_cycles`
(the monotonic system cycle count INCLUDING the read's bus op — xitari
`mySystem->cycles()`).
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
    # xitari M6532::reset: myTimer=25, myIntervalShift=6 (64-cycle prescaler).
    assert r.my_timer == 25
    assert r.interval_shift == 6
    assert r.cycles_when_set == 0
    assert r.read_after_int is False
    assert r.cycles_when_int_reset == 0
    assert r.swcha_in == 0xFF
    # Task #64: SWCHB defaults to 0x3F to match xitari's
    # `Switches::Switches` (B/B difficulty + COLOR + Select/Reset
    # released). Atari Breakout uses bit 7 to toggle paddle size — A
    # (1) = small 4-bit paddle, B (0) = large 8-bit paddle — so the
    # old 0xFF default rendered the harder small-paddle variant.
    assert r.swchb_in == 0x3F
    assert r.swacnt == 0x00 and r.swbcnt == 0x00


def test_initial_port_reads_return_input_lines():
    """Default DDR=0 → ports report the input lines (SWCHA=0xFF idle
    joystick, SWCHB=0x3F = xitari console-switch default — see task #64
    in the SWCHB docstring)."""
    r = initial_riot_state()
    assert riot_peek(r, 0x0280)[0] == 0xFF     # SWCHA  (peek returns (value, riot))
    assert riot_peek(r, 0x0282)[0] == 0x3F     # SWCHB — task #64


# --------------------------------------------------------------------------- #
# Timer write addresses + prescaler decode
# --------------------------------------------------------------------------- #

def test_tim1t_load():
    r = riot_poke(initial_riot_state(), 0x0294, 100)
    assert r.my_timer == 100
    assert r.interval_shift == 0


def test_tim8t_load():
    r = riot_poke(initial_riot_state(), 0x0295, 50)
    assert r.my_timer == 50
    assert r.interval_shift == 3             # 2**3 = 8


def test_tim64t_load():
    r = riot_poke(initial_riot_state(), 0x0296, 20)
    assert r.interval_shift == 6             # 2**6 = 64


def test_t1024t_load():
    r = riot_poke(initial_riot_state(), 0x0297, 5)
    assert r.interval_shift == 10            # 2**10 = 1024


def test_timer_load_stamps_cycles_when_set():
    """The TIM*T write records the monotonic cycle count and clears the
    read-after-interrupt latch (xitari `myCyclesWhenTimerSet` +
    `myTimerReadAfterInterrupt = false`)."""
    r = initial_riot_state()._replace(read_after_int=True)
    r = riot_poke(r, 0x0294, 1, cur_cycles=500)
    assert r.read_after_int is False
    assert r.cycles_when_set == 500


# --------------------------------------------------------------------------- #
# riot_advance is a no-op in the lazy model
# --------------------------------------------------------------------------- #

def test_riot_advance_is_noop():
    r = riot_poke(initial_riot_state(), 0x0294, 50)
    assert riot_advance(r, 100) == r         # nothing accumulates


# --------------------------------------------------------------------------- #
# INTIM lazy read formula — pre-expiration countdown
# --------------------------------------------------------------------------- #

def test_intim_readable_via_peek():
    # xitari's INTIM read formula has an extra `- 1`
    # (`myTimer - (delta>>shift) - 1`), so reading INTIM one cycle after a
    # TIM1T write of 42 returns 41. `delta = (cur-1) - set = (1-1) - 0 = 0`.
    r = riot_poke(initial_riot_state(), 0x0294, 42)   # set=0
    assert riot_peek(r, 0x0284, 1)[0] == 41


def test_intim_tim1t_countdown():
    """TIM1T (shift 0): INTIM decrements once per cycle. set=0."""
    r = riot_poke(initial_riot_state(), 0x0294, 100)
    assert riot_peek(r, 0x0284, 1)[0]  == 99      # delta = 0
    assert riot_peek(r, 0x0284, 11)[0] == 89      # delta = 10
    assert riot_peek(r, 0x0284, 31)[0] == 69      # delta = 30


def test_intim_tim8t_countdown():
    """TIM8T (shift 3): INTIM decrements once per 8 cycles. set=0."""
    r = riot_poke(initial_riot_state(), 0x0295, 50)
    assert riot_peek(r, 0x0284, 1)[0]  == 49      # delta = 0  → 0>>3 = 0
    assert riot_peek(r, 0x0284, 9)[0]  == 48      # delta = 8  → 8>>3 = 1
    assert riot_peek(r, 0x0284, 17)[0] == 47      # delta = 16 → 16>>3 = 2


# --------------------------------------------------------------------------- #
# INSTAT (D7 = "timer not yet expired-and-read")
# --------------------------------------------------------------------------- #

def test_instat_zero_before_expiration():
    r = riot_poke(initial_riot_state(), 0x0294, 5)
    # cur=1 → delta=0 → timer = 5-0-1 = 4 >= 0 → D7 clear.
    assert riot_peek(r, 0x0285, 1)[0] == 0x00


def test_instat_d7_set_after_expiration():
    r = riot_poke(initial_riot_state(), 0x0294, 5)
    # cur=7 → delta=6 → timer = 5-6-1 = -2 < 0, not yet read → D7 set.
    assert riot_peek(r, 0x0285, 7)[0] & 0x80 != 0


def test_instat_read_does_not_latch_read_after_int():
    """INSTAT (xitari case 0x05) is a pure read — it never sets
    `myTimerReadAfterInterrupt` (only the timer-output read does)."""
    r = riot_poke(initial_riot_state(), 0x0294, 1)
    _, r_after = riot_peek(r, 0x0285, 7)
    assert r_after.read_after_int is False


# --------------------------------------------------------------------------- #
# Post-expiry: INTIM read latches read_after_int (xitari slow-count branch)
# --------------------------------------------------------------------------- #

def test_post_expiry_intim_read_latches_read_after_int():
    r = riot_poke(initial_riot_state(), 0x0294, 1)   # my_timer=1, shift=0, set=0
    assert r.read_after_int is False
    # cur=3 → delta=2: pre-timer = 1-2-1 = -2 < 0 → expired branch.
    #   timer = (1<<0) - 2 - 1 = -2 ≤ -2 → latch read_after_int, int_reset=3.
    #   offset = 3 - (0 + (1<<0)) = 2 → timer = 1 - (2>>0) - 2 = -3 → 0xFD.
    value, r2 = riot_peek(r, 0x0284, 3)
    assert value == 0xFD
    assert r2.read_after_int is True
    assert r2.cycles_when_int_reset == 3
    # Once latched, INSTAT reads D7=0 (xitari: timer>=0 || rai → 0x00).
    assert riot_peek(r2, 0x0285, 4)[0] & 0x80 == 0


def test_writing_timer_clears_read_after_int():
    r = riot_poke(initial_riot_state(), 0x0294, 1)
    _, r = riot_peek(r, 0x0284, 5)          # post-expiry read latches it
    assert r.read_after_int is True
    r = riot_poke(r, 0x0295, 100)           # new TIM8T load clears it
    assert r.read_after_int is False
    assert r.my_timer == 100
    assert r.interval_shift == 3


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
    """LDA #$50 / STA $295 (TIM8T) / LDA $284 (INTIM)."""
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
    assert bus.riot.my_timer == 0x50
    s, bus = step(s, bus)                # LDA INTIM
    # Lazy read: by the LDA INTIM read the timer has barely moved under
    # the 8-cycle prescaler, so A is $50 or one tick below.
    assert int(s.A) in (0x4F, 0x50)


def test_step_advances_riot_timer():
    """Run a few NOPs through the bus and check INTIM (read lazily)
    decreases under TIM1T."""
    rom = jnp.full((4096,), 0xEA, dtype=jnp.uint8)   # all NOPs (2 cycles each)
    bus = initial_bus(rom)
    bus = bus._replace(riot=riot_poke(bus.riot, 0x0294, 50))  # TIM1T = 50, set=0
    s = _state(PC=0xF000)
    for _ in range(10):                  # 20 cycles → total_cycles = 20
        s, bus = step(s, bus)
    # peek's own bus op makes cur = 20 + 1 = 21 → delta = 20 → 50-20-1 = 29.
    assert peek(bus, 0x0284)[0] == 29


def test_step_wsync_stall_also_advances_riot():
    """STA WSYNC's stall cycles tick the monotonic counter too, so a
    lazy INTIM read after a WSYNC reflects the full scanline of cycles."""
    rom = jnp.zeros((4096,), dtype=jnp.uint8)
    rom = rom.at[0].set(jnp.uint8(0x85))             # STA $02 = WSYNC
    rom = rom.at[1].set(jnp.uint8(0x02))
    bus = initial_bus(rom)
    bus = bus._replace(riot=riot_poke(bus.riot, 0x0294, 200))  # TIM1T = 200, set=0
    s = _state(PC=0xF000)
    s, bus = step(s, bus)
    # STA WSYNC = 3 cyc + 73 stall = 76 cycles → total_cycles = 76.
    # peek: cur = 76 + 1 = 77 → delta = 76 → 200-76-1 = 123.
    assert peek(bus, 0x0284)[0] == 123
