"""RIOT — MOS 6532 timer + I/O ports.

The 6532 has three independent functions: 128 B of RAM (wired through
the Bus in P2), two 8-bit I/O ports (A and B), and a programmable
interval timer. This module covers the latter two.

Register map within \$0280–\$029F (also mirrored throughout the RIOT
I/O bank; the chip decodes only A0–A4):

  Read / Write:
    \$0280  SWCHA   port A data        (input bits read; output bits reflect SWCHA-out)
    \$0281  SWACNT  port A direction   (1 = output bit, 0 = input bit)
    \$0282  SWCHB   port B data
    \$0283  SWBCNT  port B direction
  Read only:
    \$0284  INTIM   current timer value (8-bit)
    \$0285  INSTAT  timer status — D7 = expired flag
  Write only:
    \$0294  TIM1T   load timer with 1-cycle prescaler
    \$0295  TIM8T            8-cycle prescaler
    \$0296  TIM64T          64-cycle prescaler
    \$0297  T1024T        1024-cycle prescaler

Timer model — **faithful lazy port of xitari `M6532` (task #80).**

Rather than decrementing INTIM eagerly each CPU cycle, xitari stores
`myTimer` / `myIntervalShift` / `myCyclesWhenTimerSet` at the TIM*T
write and *computes* INTIM / INSTAT on read from the monotonic system
cycle counter. We mirror that here bit-for-bit, including the
`myTimerReadAfterInterrupt` post-expiry slow-count branch.

The caller passes `cur_cycles = tia.total_cycles + pending_tia_cycles`
(jaxtari's equivalent of xitari `mySystem->cycles()`, incremented
*before* each bus access). The read formula subtracts 1 (`cycles - 1`)
exactly as xitari's `M6532::peek`. `riot_advance` is a no-op (lazy).

Power-on state mirrors `M6532::reset`: `myTimer = 25`,
`myIntervalShift = 6` (i.e. INTIM counts down from 25 at the 64-cycle
prescaler). This default — together with the lazy post-expiry formula
and the TIA's max-scanlines frame cutoff — is what makes seaquest's
INTIM-polling boot land bit-exact (the eager model + 262-only frame
counting ran one extra cart-loop iteration → RAM[$01] = $3f vs the
reference $3e).

Port semantics:

  * SWCHA / SWCHB return `(out & ddr) | (input & ~ddr)`, i.e. each bit
    is either the latched output (when configured as output) or the
    external input (when configured as input). On power-up DDR is
    all-zero so the ports are pure inputs.
  * `set_swcha_input` / `set_swchb_input` are helpers for tests and
    front-ends — they set the input lines the chip "sees" at the bus.
"""

from __future__ import annotations

from typing import NamedTuple

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Prescaler shift per timer-write address (addr & 0x03):
#   0 (TIM1T)   → 1     (2**0)
#   1 (TIM8T)   → 8     (2**3)
#   2 (TIM64T)  → 64    (2**6)
#   3 (T1024T)  → 1024  (2**10)
_PRESCALER_SHIFT = (0, 3, 6, 10)


# --------------------------------------------------------------------------- #
# 32-bit reinterpret helpers (match xitari's uInt32 / Int32 arithmetic)
# --------------------------------------------------------------------------- #

def _u32(x: int) -> int:
    """Truncate to unsigned 32-bit (xitari `uInt32`)."""
    return x & 0xFFFFFFFF


def _i32(x: int) -> int:
    """Reinterpret the low 32 bits as signed (xitari `(Int32)` cast)."""
    x &= 0xFFFFFFFF
    return x - 0x100000000 if x >= 0x80000000 else x


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

class RIOTState(NamedTuple):
    """Snapshot of the RIOT's timer + I/O state. RAM is handled by the Bus.

    Timer fields mirror xitari `M6532`'s members exactly:
      * my_timer              — the value last written to TIM*T (`myTimer`)
      * interval_shift        — prescaler shift 0/3/6/10 (`myIntervalShift`)
      * cycles_when_set       — `mySystem->cycles()` at the TIM*T write
      * read_after_int        — `myTimerReadAfterInterrupt` latch
      * cycles_when_int_reset — `myCyclesWhenInterruptReset`
    """
    my_timer: int
    interval_shift: int
    cycles_when_set: int
    read_after_int: bool
    cycles_when_int_reset: int
    swcha_in: int
    swchb_in: int
    swacnt: int
    swbcnt: int
    swcha_out: int
    swchb_out: int


def initial_riot_state() -> RIOTState:
    """Power-on state — mirrors xitari `M6532::reset`: `myTimer = 25`,
    `myIntervalShift = 6`, the cycle stamps zeroed and the
    read-after-interrupt latch cleared.

    Ports default to all-input (DDR = 0). No inputs asserted: joysticks
    pull-up active-low so "no buttons pressed" on SWCHA reads as 0xFF.

    SWCHB (console-switches port) defaults to **0x3F** — matches xitari's
    `Switches::Switches` initialization for an unmodified-properties
    cartridge:

      * bit 7 (Right-Difficulty): 0 = **B (amateur)**.
      * bit 6 (Left-Difficulty):  0 = **B (amateur)**.
      * bit 3 (TV-Type):          1 = **COLOR**.
      * bits 1,0 (Select/Reset):  1 = **released** (active-low).
      * bits 5,4,2: unused, idle high.

    Per-game overrides (Q*Bert wants A/A, etc.) belong on the per-game
    RomSettings — same pattern as `uses_paddles`.
    """
    return RIOTState(
        # xitari M6532::reset: myTimer=25, myIntervalShift=6.
        my_timer=25,
        interval_shift=6,
        cycles_when_set=0,
        read_after_int=False,
        cycles_when_int_reset=0,
        swcha_in=0xFF,
        # 0x3F = 0011_1111. See docstring above for the per-bit meaning.
        swchb_in=0x3F,
        swacnt=0x00,
        swbcnt=0x00,
        swcha_out=0xFF,
        swchb_out=0xFF,
    )


# --------------------------------------------------------------------------- #
# Helpers exposed for tests / front-ends
# --------------------------------------------------------------------------- #

def set_swcha_input(riot: RIOTState, value: int) -> RIOTState:
    """Set the external input lines feeding port A. Each cleared bit
    represents a pressed direction / button (active-low)."""
    return riot._replace(swcha_in=value & 0xFF)


def set_swchb_input(riot: RIOTState, value: int) -> RIOTState:
    """Set the external input lines feeding port B (console switches)."""
    return riot._replace(swchb_in=value & 0xFF)


# --------------------------------------------------------------------------- #
# Timer "output" — xitari M6532::peek case 0x04 (value before & 0xFF)
# --------------------------------------------------------------------------- #

def _timer_output(riot: RIOTState, cur_cycles: int):
    """Compute the timer-output byte exactly as xitari `M6532::peek`
    case 0x04. Returns `(timer_full_int, new_riot)` — the post-expiry
    branch latches `read_after_int` / `cycles_when_int_reset`, so a read
    can mutate the (immutable) RIOT state.
    """
    cycles = _u32(int(cur_cycles) - 1)
    delta = _u32(cycles - _u32(riot.cycles_when_set))
    shift = riot.interval_shift
    timer = _i32(riot.my_timer - _i32(delta >> shift) - 1)
    if timer >= 0:
        return timer, riot

    # Expired branch (xitari M6532.cxx:172-189).
    timer = _i32((riot.my_timer << shift) - _i32(delta) - 1)
    new_riot = riot
    read_after_int = riot.read_after_int
    cycles_when_int_reset = riot.cycles_when_int_reset
    if timer <= -2 and not read_after_int:
        # Indicate that the timer has been read after the interrupt occurred.
        read_after_int = True
        cycles_when_int_reset = int(cur_cycles)
        new_riot = riot._replace(read_after_int=True,
                                 cycles_when_int_reset=int(cur_cycles))
    if read_after_int:
        offset = _i32(cycles_when_int_reset -
                      (riot.cycles_when_set + (riot.my_timer << shift)))
        timer = _i32(riot.my_timer - _i32(delta >> shift) - offset)
    return timer, new_riot


# --------------------------------------------------------------------------- #
# Register access
# --------------------------------------------------------------------------- #

def riot_peek(riot: RIOTState, addr: int, cur_cycles: int = 0):
    """Read a RIOT register. Returns `(value, new_riot)`.

    `addr` is the CPU bus address (anywhere in the RIOT mirror band).
    `cur_cycles` is the monotonic system cycle count *including* the
    current bus op — xitari `mySystem->cycles()` (the read formula then
    uses `cycles - 1`). Most reads return the SAME `riot`; only the
    timer-output read (reg 4/6) can mutate it (latching the
    post-interrupt slow-count state — xitari `myTimerReadAfterInterrupt`).
    """
    reg = addr & 0x07
    if reg == 0:
        return (((riot.swcha_out & riot.swacnt) |
                 (riot.swcha_in & (~riot.swacnt & 0xFF))) & 0xFF, riot)
    if reg == 1:
        return riot.swacnt, riot
    if reg == 2:
        return (((riot.swchb_out & riot.swbcnt) |
                 (riot.swchb_in & (~riot.swbcnt & 0xFF))) & 0xFF, riot)
    if reg == 3:
        return riot.swbcnt, riot
    if reg == 4 or reg == 6:
        timer, new_riot = _timer_output(riot, cur_cycles)
        return timer & 0xFF, new_riot
    if reg == 5 or reg == 7:
        # INSTAT (xitari case 0x05): D7 = "not expired and not yet read
        # after interrupt". Pure read — does NOT latch read_after_int.
        cycles = _u32(int(cur_cycles) - 1)
        delta = _u32(cycles - _u32(riot.cycles_when_set))
        timer = _i32(riot.my_timer - _i32(delta >> riot.interval_shift) - 1)
        value = 0x00 if (timer >= 0 or riot.read_after_int) else 0x80
        return value, riot
    return 0, riot


def riot_poke(riot: RIOTState, addr: int, value: int,
              cur_cycles: int = 0) -> RIOTState:
    """Write a RIOT register. Timer-load addresses (TIM*T at \$0294–\$0297,
    detected by `(addr & 0x14) == 0x14`) store the value + prescaler shift
    and stamp `cycles_when_set` with the current monotonic cycle count,
    clearing the read-after-interrupt latch — exactly as xitari
    `M6532::poke` case TIM*T.

    Mirrors jutari `riot_poke!`. Unlike the previous eager model there is
    no `cycles_since_tick` pre-subtraction: xitari records
    `myCyclesWhenTimerSet = mySystem->cycles()` at the poke moment and the
    lazy read uses `cycles - 1`, so passing the monotonic
    `tia.total_cycles + pending_tia_cycles` here is sufficient and exact.
    """
    if (addr & 0x14) == 0x14:
        return riot._replace(
            my_timer=value & 0xFF,
            interval_shift=_PRESCALER_SHIFT[addr & 0x03],
            cycles_when_set=int(cur_cycles),
            read_after_int=False,
        )
    reg = addr & 0x07
    if reg == 0:
        return riot._replace(swcha_out=value & 0xFF)
    if reg == 1:
        return riot._replace(swacnt=value & 0xFF)
    if reg == 2:
        return riot._replace(swchb_out=value & 0xFF)
    if reg == 3:
        return riot._replace(swbcnt=value & 0xFF)
    return riot


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #

def riot_advance(riot: RIOTState, cpu_cycles: int) -> RIOTState:
    """No-op in the lazy timer model — INTIM / INSTAT are computed on read
    from the monotonic cycle counter, so nothing accumulates here. Kept
    for call-site compatibility (the CPU step still calls it). Mirrors
    jutari `riot_advance!`."""
    return riot
