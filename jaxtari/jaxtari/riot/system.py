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

Timer semantics:

  * Writing TIM*T loads INTIM with the value and resets the prescaler
    counter. The "expired" flag is cleared by the same write.
  * INTIM decrements by 1 every `prescaler` CPU cycles.
  * When INTIM decrements past zero it wraps to \$FF and the expired
    flag becomes set.
  * After expiration the timer continues to decrement once per CPU
    cycle, regardless of the originally-selected prescaler.

Port semantics:

  * SWCHA / SWCHB return `(out & ddr) | (input & ~ddr)`, i.e. each bit
    is either the latched output (when configured as output) or the
    external input (when configured as input). On power-up DDR is
    all-zero so the ports are pure inputs; on the Atari, SWCHA carries
    the two joysticks and SWCHB the console switches.
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
# State
# --------------------------------------------------------------------------- #

class RIOTState(NamedTuple):
    """Snapshot of the RIOT's timer + I/O state. RAM is handled by the Bus."""
    intim: int
    prescaler_shift: int
    cycles_since_tick: int
    timer_expired: bool
    swcha_in: int
    swchb_in: int
    swacnt: int
    swbcnt: int
    swcha_out: int
    swchb_out: int


def initial_riot_state() -> RIOTState:
    """Power-on state: timer at 0 with 1× prescaler, ports all-input, no
    inputs asserted (joysticks pull-up active-low so "no buttons pressed"
    on SWCHA reads as 0xFF).

    SWCHB (console-switches port) defaults to **0x3F** — matches xitari's
    `Switches::Switches` initialization for an unmodified-properties
    cartridge:

      * bit 7 (Right-Difficulty switch): 0 = **B (amateur)**. Properties
        default `Console_RightDifficulty = "B"` → bit cleared. Atari
        Breakout uses bit 7 as the "small paddle" toggle — A (=1) draws
        a 4-bit paddle, B (=0) draws an 8-bit paddle. (Task #64.)
      * bit 6 (Left-Difficulty switch): 0 = **B (amateur)**, same rule.
      * bit 3 (TV-Type): 1 = **COLOR**. Properties default
        `Console_TelevisionType = "COLOR"` → bit set.
      * bits 1,0 (Select / Reset): 1 = **released** (active-low).
      * bits 5,4,2: unused, idle high.

    Per-game overrides (e.g. Q*Bert wants A/A, some prototypes flip TV
    type) belong on the per-game RomSettings — same pattern as
    `uses_paddles`.
    """
    return RIOTState(
        intim=0,
        prescaler_shift=0,
        cycles_since_tick=0,
        timer_expired=False,
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
# Register access
# --------------------------------------------------------------------------- #

def riot_peek(riot: RIOTState, addr: int, pending_extra_cycles: int = 0):
    """Read a RIOT register. Returns `(value, new_riot)`.

    `addr` is the CPU bus address (anywhere in the RIOT mirror band).
    Most reads return the SAME `riot` object — only INTIM (reg=4) has
    a side effect: it clears the timer-expired latch, matching the
    real MOS 6532's read-clears-flag semantic (**P4d**).

    `pending_extra_cycles` = the CPU cycles consumed inside the current
    instruction up to and including this bus op. Used by Phase 5
    RIOT-read threading: xitari's `M6532::peek` case 0x04 uses
    `cycles = mySystem->cycles() - 1` so the effective extra is
    `max(0, pending_extra_cycles - 1)`.

    INSTAT (reg=5) returns D7 = timer_expired but does NOT clear it
    on read — that's an INTIM-only side effect. INSTAT clears the
    PA7 latch on real hardware, but PA7 isn't modelled yet (deferred
    as **P4b**), so for the moment INSTAT reads are pure.
    """
    reg = addr & 0x07
    if reg == 0:
        return (((riot.swcha_out & riot.swacnt) |
                 (riot.swcha_in  & (~riot.swacnt & 0xFF))) & 0xFF, riot)
    if reg == 1:
        return riot.swacnt, riot
    if reg == 2:
        return (((riot.swchb_out & riot.swbcnt) |
                 (riot.swchb_in  & (~riot.swbcnt & 0xFF))) & 0xFF, riot)
    if reg == 3:
        return riot.swbcnt, riot
    if reg == 4:
        # P4d: INTIM read clears the timer-expired latch.
        # P3i-g pt8: xitari's INTIM read formula has an extra `- 1` term
        # (`M6532::peek` case 0x04: `myTimer - (delta>>shift) - 1`).
        # Phase 5 (2026-06-03): xitari ALSO uses `cycles - 1` for the
        # delta numerator (line 161 of M6532.cxx), so the effective
        # mid-instruction extra is `max(0, pending_extra_cycles - 1)`.
        # The Phase 5 fix computes the intra-instruction timer
        # advancement so cross-prescaler-boundary INTIM reads return
        # the value xitari would. Read-only — does NOT mutate
        # riot.intim (the real advance happens in
        # `_step_inner`/post-step via `riot_advance`).
        intim_int = int(riot.intim)
        eff = max(0, int(pending_extra_cycles) - 1)
        if not riot.timer_expired:
            shift = int(riot.prescaler_shift)
            extra = (int(riot.cycles_since_tick) + eff) >> shift
            intim_int -= extra
        else:
            intim_int -= eff
        return (intim_int - 1) & 0xFF, riot._replace(timer_expired=False)
    if reg == 5:
        return (0x80 if riot.timer_expired else 0x00), riot
    return 0, riot


def riot_poke(riot: RIOTState, addr: int, value: int) -> RIOTState:
    """Write a RIOT register. Timer-load addresses (TIM*T at \$0294–\$0297,
    detected by (addr & 0x14) == 0x14) reset the prescaler and clear the
    expired flag; the prescaler is selected by the low two bits of the
    address (0=1×, 1=8×, 2=64×, 3=1024×)."""
    if (addr & 0x14) == 0x14:
        return riot._replace(
            intim=value & 0xFF,
            prescaler_shift=_PRESCALER_SHIFT[addr & 0x03],
            cycles_since_tick=0,
            timer_expired=False,
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
    """Advance the timer by `cpu_cycles` CPU cycles.

    Two regimes:
      - Pre-expiration: INTIM ticks every `prescaler` cycles. Expiration
        happens when INTIM would decrement past 0; INTIM wraps to \$FF
        and `timer_expired` becomes True.
      - Post-expiration: INTIM ticks once per CPU cycle (regardless of
        the originally-selected prescaler).

    The branch handles a transition during this advance by computing the
    cycle count needed to expire, then applying the leftover cycles at
    the post-expiration rate.
    """
    intim = riot.intim
    cycles = riot.cycles_since_tick + cpu_cycles
    expired = riot.timer_expired

    if not expired:
        prescaler = 1 << riot.prescaler_shift
        cycles_to_expire = (intim + 1) * prescaler
        if cycles >= cycles_to_expire:
            # Cross the expiration boundary during this advance.
            cycles_post = cycles - cycles_to_expire
            intim = (0xFF - cycles_post) & 0xFF
            expired = True
            cycles_left = 0
        else:
            ticks = cycles // prescaler
            intim -= ticks
            cycles_left = cycles % prescaler
    else:
        # Already expired: one tick per cycle.
        intim = (intim - cycles) & 0xFF
        cycles_left = 0

    return riot._replace(
        intim=intim,
        cycles_since_tick=cycles_left,
        timer_expired=expired,
    )
