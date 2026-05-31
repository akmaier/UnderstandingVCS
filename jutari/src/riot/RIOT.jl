"""
    RIOT

MOS 6532 timer + I/O ports. The 128 B RAM is wired through the Bus
directly (P2); this module covers the rest.

Register map within \$0280–\$029F (mirrored throughout the RIOT bank):

  Read / Write:
    \$0280 SWCHA   port A data
    \$0281 SWACNT  port A direction (1 = output bit)
    \$0282 SWCHB   port B data
    \$0283 SWBCNT  port B direction
  Read only:
    \$0284 INTIM   current timer value
    \$0285 INSTAT  D7 = timer expired flag
  Write only:
    \$0294 TIM1T   load timer (×1 prescaler)
    \$0295 TIM8T               (×8)
    \$0296 TIM64T              (×64)
    \$0297 T1024T              (×1024)

Timer semantics: pre-expiration ticks every `prescaler` cycles; once
INTIM wraps past 0 to \$FF, expired=true and subsequent ticks happen
every CPU cycle.
"""
module RIOT

export RIOTState, initial_riot_state,
       riot_peek, riot_peek!, riot_poke!, riot_advance!,
       set_swcha_input!, set_swchb_input!

const _PRESCALER_SHIFT = (0, 3, 6, 10)

"""
    RIOTState

Mutable timer + I/O state. RAM is handled by the Bus.
"""
mutable struct RIOTState
    intim::UInt8
    prescaler_shift::Int
    cycles_since_tick::Int
    timer_expired::Bool
    swcha_in::UInt8
    swchb_in::UInt8
    swacnt::UInt8
    swbcnt::UInt8
    swcha_out::UInt8
    swchb_out::UInt8
end

initial_riot_state() = RIOTState(
    UInt8(0), 0, 0, false,
    # SWCHA_in = 0xFF (no joystick directions / triggers pressed).
    # SWCHB_in = 0x3F: matches xitari `Switches::Switches` initial state
    # for an unmodified-properties cartridge — B/B difficulty (bits 7+6
    # cleared), COLOR TV (bit 3 set), Select/Reset released (bits 1+0
    # set). Bit 7 in particular controls Atari Breakout's paddle size:
    # A (1) = 4-bit small paddle, B (0) = 8-bit large paddle. (Task #64.)
    UInt8(0xFF), UInt8(0x3F),
    UInt8(0x00), UInt8(0x00),
    UInt8(0xFF), UInt8(0xFF),
)

"""
    set_swcha_input!(riot, value)

Set the external input lines at port A. Each cleared bit is an
"active" input (button held / joystick direction pressed).
"""
@inline function set_swcha_input!(riot::RIOTState, value::Integer)
    riot.swcha_in = UInt8(Int(value) & 0xFF)
    return nothing
end

@inline function set_swchb_input!(riot::RIOTState, value::Integer)
    riot.swchb_in = UInt8(Int(value) & 0xFF)
    return nothing
end

# --------------------------------------------------------------------------- #
# Register access
# --------------------------------------------------------------------------- #

"""
    riot_peek!(riot, addr) -> UInt8

Read a RIOT register. `addr` is the CPU bus address (anywhere in the
RIOT mirror band). **Mutating** because real-hardware reads of INTIM /
INSTAT have side-effects on the latches:

  * INTIM (reg=4, addresses `\$0284` / `\$0286`): returns the timer
    count, then **clears the timer-expired latch**. Pre-P4d the latch
    was only cleared by writing a fresh value to TIM*T; the canonical
    MOS 6532 datasheet says reading INTIM also clears it, which a few
    Atari games (and the cycle-accurate parts of xitari) depend on.

  * INSTAT (reg=5, addresses `\$0285` / `\$0287`): returns D7 = the
    timer-expired latch AND D6 = the PA7 latch. **Reading INSTAT
    clears ONLY the PA7 latch** (not the timer latch — that's an INTIM
    semantic). PA7 isn't modelled yet (deferred as P4b), so for the
    moment this is a no-op beyond returning D7.
"""
function riot_peek!(riot::RIOTState, addr::Integer)
    reg = Int(addr) & 0x07
    if reg == 0
        return UInt8(((riot.swcha_out & riot.swacnt) |
                      (riot.swcha_in  & (~riot.swacnt & 0xFF))) & 0xFF)
    elseif reg == 1
        return riot.swacnt
    elseif reg == 2
        return UInt8(((riot.swchb_out & riot.swbcnt) |
                      (riot.swchb_in  & (~riot.swbcnt & 0xFF))) & 0xFF)
    elseif reg == 3
        return riot.swbcnt
    elseif reg == 4
        # P4d: INTIM read clears the timer-expired latch.
        # P3i-g pt8: xitari's INTIM read formula has an extra `- 1` term
        # (M6532::peek case 0x04: `myTimer - (delta>>shift) - 1`). That
        # makes xitari's INTIM 1 less than the raw register. Without this
        # offset, INTIM-polling loops in early VBLANK exit 1+ iterations
        # later than xitari, accumulating ~76 CPU cycles of drift per loop
        # — the source of pong's cross-scanline 1-line residual + most of
        # the other ROMs' large PXC-S residuals (pong 568→32, SI 2079→12,
        # pitfall 1786→322, seaquest 3941→1104, enduro 1954→1197).
        value = UInt8((Int(riot.intim) - 1) & 0xFF)
        riot.timer_expired = false
        return value
    elseif reg == 5
        # INSTAT — D7 = timer latch (unchanged by this read), D6 =
        # PA7 latch (cleared by this read in real hardware; PA7
        # isn't modelled yet so this is a no-op).
        return riot.timer_expired ? UInt8(0x80) : UInt8(0x00)
    end
    return UInt8(0)
end

# Backwards-compatibility alias — existing callers that don't need
# the new mutating semantics (or that read the bus from a const
# reference) can still use the old name. New code should prefer
# the `!` variant since reads ARE state-changing on real hardware.
const riot_peek = riot_peek!

"""
    riot_poke!(riot, addr, value)

Write a RIOT register. Timer-load addresses (\$0294–\$0297) reset the
prescaler counter and clear the expired flag; the prescaler is selected
by the low two bits of `addr` (0=1×, 1=8×, 2=64×, 3=1024×).
"""
function riot_poke!(riot::RIOTState, addr::Integer, value::Integer)
    if (Int(addr) & 0x14) == 0x14
        riot.intim = UInt8(Int(value) & 0xFF)
        riot.prescaler_shift = _PRESCALER_SHIFT[(Int(addr) & 0x03) + 1]
        riot.cycles_since_tick = 0
        riot.timer_expired = false
        return nothing
    end
    reg = Int(addr) & 0x07
    if reg == 0
        riot.swcha_out = UInt8(Int(value) & 0xFF)
    elseif reg == 1
        riot.swacnt = UInt8(Int(value) & 0xFF)
    elseif reg == 2
        riot.swchb_out = UInt8(Int(value) & 0xFF)
    elseif reg == 3
        riot.swbcnt = UInt8(Int(value) & 0xFF)
    end
    return nothing
end

# --------------------------------------------------------------------------- #
# Timer
# --------------------------------------------------------------------------- #

"""
    riot_advance!(riot, cpu_cycles)

Advance the timer by `cpu_cycles` CPU cycles. Two regimes:
  - Pre-expiration: tick every `prescaler` cycles.
  - Post-expiration: tick once per CPU cycle.
The branch handles a transition during this advance.
"""
function riot_advance!(riot::RIOTState, cpu_cycles::Integer)
    intim = Int(riot.intim)
    cycles = riot.cycles_since_tick + Int(cpu_cycles)
    expired = riot.timer_expired

    if !expired
        prescaler = 1 << riot.prescaler_shift
        cycles_to_expire = (intim + 1) * prescaler
        if cycles >= cycles_to_expire
            cycles_post = cycles - cycles_to_expire
            intim = (0xFF - cycles_post) & 0xFF
            expired = true
            cycles_left = 0
        else
            ticks = cycles ÷ prescaler
            intim -= ticks
            cycles_left = cycles % prescaler
        end
    else
        intim = (intim - cycles) & 0xFF
        cycles_left = 0
    end

    riot.intim = UInt8(intim & 0xFF)
    riot.cycles_since_tick = cycles_left
    riot.timer_expired = expired
    return nothing
end

end # module
