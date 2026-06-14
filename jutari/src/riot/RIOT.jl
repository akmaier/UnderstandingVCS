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

Timer model — **faithful lazy port of xitari `M6532` (task #80).** Store
my_timer/interval_shift/cycles_when_set and compute INTIM/INSTAT on read
from the monotonic cycle counter, mirroring xitari's exact formulas incl.
the `myTimerReadAfterInterrupt` post-expiry slow-count. The caller passes
`cur_cycles = tia.total_cycles + pending_tia_cycles` (xitari
`mySystem->cycles()`); the read uses `cycles()-1`. `riot_advance!` is a
no-op (lazy).
"""
module RIOT

export RIOTState, initial_riot_state,
       riot_peek, riot_peek!, riot_poke!, riot_advance!,
       set_swcha_input!, set_swchb_input!

const _PRESCALER_SHIFT = (0, 3, 6, 10)

mutable struct RIOTState
    my_timer::Int
    interval_shift::Int
    cycles_when_set::Int
    read_after_int::Bool
    cycles_when_int_reset::Int
    swcha_in::UInt8
    swchb_in::UInt8
    swacnt::UInt8
    swbcnt::UInt8
    swcha_out::UInt8
    swchb_out::UInt8
end

initial_riot_state() = RIOTState(
    # xitari M6532::reset: myTimer=25, myIntervalShift=6, etc.
    25, 6, 0, false, 0,
    UInt8(0xFF), UInt8(0x3F),
    UInt8(0x00), UInt8(0x00),
    UInt8(0xFF), UInt8(0xFF),
)

@inline function set_swcha_input!(riot::RIOTState, value::Integer)
    riot.swcha_in = UInt8(Int(value) & 0xFF)
    return nothing
end

@inline function set_swchb_input!(riot::RIOTState, value::Integer)
    riot.swchb_in = UInt8(Int(value) & 0xFF)
    return nothing
end

# Timer "output" exactly as xitari M6532::peek case 0x04 (value before &0xFF).
# xitari uses uInt32 for cycles/delta and Int32 for timer; `(Int32)delta`
# is a bit-reinterpret, so we use `reinterpret(Int32, ...)` to match.
@inline function _timer_output!(riot::RIOTState, cur_cycles::Integer)
    cycles = UInt32((Int(cur_cycles) - 1) & 0xFFFFFFFF)
    delta  = cycles - UInt32(riot.cycles_when_set & 0xFFFFFFFF)
    shift  = riot.interval_shift
    timer  = Int32(riot.my_timer) - reinterpret(Int32, delta >> shift) - Int32(1)
    if timer >= 0
        return timer
    end
    # Expired branch (xitari M6532.cxx:172-189).
    timer = Int32(Int(riot.my_timer) << shift) - reinterpret(Int32, delta) - Int32(1)
    if timer <= -2 && !riot.read_after_int
        riot.read_after_int = true
        riot.cycles_when_int_reset = Int(cur_cycles)
    end
    if riot.read_after_int
        offset = Int32(riot.cycles_when_int_reset -
                       (riot.cycles_when_set + (Int(riot.my_timer) << shift)))
        timer = Int32(riot.my_timer) - reinterpret(Int32, delta >> shift) - offset
    end
    return timer
end

function riot_peek!(riot::RIOTState, addr::Integer,
                     cur_cycles::Integer = 0)
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
    elseif reg == 4 || reg == 6
        return UInt8(_timer_output!(riot, cur_cycles) & 0xFF)
    elseif reg == 5 || reg == 7
        cycles = UInt32((Int(cur_cycles) - 1) & 0xFFFFFFFF)
        delta  = cycles - UInt32(riot.cycles_when_set & 0xFFFFFFFF)
        timer  = Int32(riot.my_timer) - Int32((delta >> riot.interval_shift) & 0xFFFFFFFF) - Int32(1)
        return (timer >= 0 || riot.read_after_int) ? UInt8(0x00) : UInt8(0x80)
    end
    return UInt8(0)
end

const riot_peek = riot_peek!

function riot_poke!(riot::RIOTState, addr::Integer, value::Integer,
                     cur_cycles::Integer = 0)
    if (Int(addr) & 0x14) == 0x14
        riot.my_timer        = Int(value) & 0xFF
        riot.interval_shift  = _PRESCALER_SHIFT[(Int(addr) & 0x03) + 1]
        riot.cycles_when_set = Int(cur_cycles)
        riot.read_after_int  = false
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

@inline riot_advance!(riot::RIOTState, cpu_cycles::Integer) = nothing

end # module
