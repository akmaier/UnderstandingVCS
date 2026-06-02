"""
    Bus

6507 address bus and memory access for JuTari.

The Atari 2600's CPU is a 6507 — a 6502 variant with only 13 address pins
(A0–A12). Any 16-bit address from the CPU is masked to 13 bits before
decode. Within the 8K window the decode is:

    A12=1                              → Cartridge ROM (\$1000–\$1FFF)
    A12=0, A7=0                        → TIA registers (\$0000–\$007F, mirrored)
    A12=0, A7=1, A9=0                  → RIOT RAM (\$0080–\$00FF, 128 B, mirrored)
    A12=0, A7=1, A9=1                  → RIOT I/O (\$0280–\$029F, mirrored)

A real-world consequence: the 6502 stack lives at \$0100–\$01FF, but on
the 6507 that range is split — \$0100–\$017F is a TIA mirror (writes go
nowhere useful) while \$0180–\$01FF mirrors RIOT RAM. The effective
stack is therefore 128 B shared with zero-page-relative RAM; programs
keep SP in the upper half.

Status as of P5: address decode, RAM peek/poke, full TIA register file +
rendering (P3), RIOT timer + I/O ports (P4), and cartridge peek/poke
with 2K/4K/F8/F6/F4 bank-switching (P5). SC variants and the more
exotic cart formats (E0, FE, 3F, 3E, MB, MC, AR, DPC) are deferred.

Multiple dispatch handles two world types: a `BusState` (proper 6507
bus) and a flat `Vector{UInt8}` (used by the P1 unit tests). The same
`step` works against both.
"""
module Bus

using ..TIA: TIAState, initial_tia_state, tia_peek, tia_poke!, tia_advance!,
             _apply_pixel_collisions!, _object_pixel_sets,
             HBLANK_COLOR_CLOCKS, COLOR_CLOCKS_PER_SCANLINE
using ..RIOT: RIOTState, initial_riot_state, riot_peek!, riot_poke!
using ..Cart: CartState, make_cart, cart_peek, cart_poke!

export BusState, initial_bus, peek, poke!, pending_tick!,
       trace_enable!, trace_disable!, trace_take!

"""
    BusState

6507 system bus state. `ram` is the 128 B RAM; `cart` is the cartridge
(includes ROM + bank state, P5); `tia` and `riot` are the chips' states.

`data_bus_state` tracks the last byte that crossed the data bus —
updated on every `peek` / `poke!`. The real TIA only actively
drives 1-2 of the 8 data lines on a read; the rest "float" and
resolve to whatever was last on the bus. This is the PXC1-x
round 3 quirk that matches xitari's `System::peek/poke` + the
real 1A05 hardware. See `peek(bus::BusState, ...)` for how the
per-register driven mask is applied.
"""
mutable struct BusState
    ram::Vector{UInt8}
    cart::CartState
    tia::TIAState
    riot::RIOTState
    data_bus_state::UInt8
    # P3i-g (mid-instruction TIA-write CPU↔TIA cycle threading): CPU
    # cycles consumed since the last TIA sync. Every bus op (peek/poke!)
    # bumps this by 1 — on a real 6507 every cycle IS one bus op. When a
    # TIA-region *write* happens, the accumulator is flushed via
    # `tia_advance!(tia, pending)` BEFORE the poke, so PF*/RESP*/HMOVE/
    # COLU* land at the precise sub-instruction color clock (xitari
    # increments its cycle counter *before* each access, then `TIA::poke`
    # runs `updateFrame(cycles*3)`). Reads are NOT flushed — that keeps
    # read-driven game logic on the per-instruction model (render-only
    # change). `tia_advanced_this_instruction` records how much the
    # inline write-flushes already advanced so `_tia_post_step!` drains
    # exactly the remainder. Mirrors jaxtari `bus/system.py`.
    pending_tia_cycles::Int
    tia_advanced_this_instruction::Int
end

"""
    initial_bus(rom=nothing) -> BusState

Build a `BusState` with all-zero RAM, an auto-detected cart built from
`rom` (default: all-zero 4 KB), and fresh TIA / RIOT states.
"""
function initial_bus(rom=nothing)
    rom === nothing && (rom = zeros(UInt8, 4096))
    return BusState(zeros(UInt8, 128), make_cart(rom),
                    initial_tia_state(), initial_riot_state(), 0x00, 0, 0)
end

"""
    pending_tick!(world)

Account for one CPU cycle of internal bus activity not modelled as a real
`peek`/`poke!` — e.g. the cycle-3 "dummy read" of the indexed zero-page
addressing modes. The 6502 drives a bus access on that cycle, but its
data-bus value is always overwritten by the instruction's final access,
so reproducing the read has no observable effect; what matters for P3i-g
write-cycle threading is that the cycle is *counted*, so a following TIA
write flushes the right number of color-clock cycles. No-op for flat
test memory. Mirrors jaxtari `bus/system.py::pending_tick`.
"""
@inline pending_tick!(::Vector{UInt8}) = nothing
@inline function pending_tick!(bus::BusState)
    bus.pending_tia_cycles += 1
    # Phase 1 diagnostic — record internal-cycle ticks too so the
    # trace shows the full per-cycle picture (otherwise a (zp,X)
    # dummy-read cycle looks like missing time in the dump).
    _trace_record!(:tick, 0x0000, 0x00)
    return nothing
end

# --------------------------------------------------------------------------- #
# Phase 1 diagnostic tap — per-bus-op trace (P3I_G_THREADING_PLAN.md)
# --------------------------------------------------------------------------- #
#
# Zero-cost when disabled (a single `=== nothing` check per peek/poke).
# When enabled (via `trace_enable!()`), every peek/poke + internal-cycle
# tick + WSYNC release is recorded as a tuple in a module-level buffer
# along with the TIA's `(scanline, scanline_cycle, color_clock)` at the
# moment of the bus op. `tools/cpu_tia_cycle_trace.jl` flushes the buffer
# to CSV for cross-comparison against an xitari trace.
#
# The buffer is intentionally module-level (not on the Bus struct) so
# adding diagnostics doesn't disturb the BusState layout — which would
# force a rebuild of every test fixture. The cost of using a global is
# that the trace is process-global, not bus-instance-local; we only run
# one bus at a time so this is fine.
#
# Event tuple shape: (kind, scanline, scanline_cycle, color_clock,
#                     addr, value) — `kind ∈ {:peek, :poke, :tick,
#                     :wsync_release}`. `value` is the byte read/written
#                     (for peek/poke), the WSYNC stall count (for
#                     wsync_release), or 0 (for tick).
const _TraceTuple = Tuple{Symbol, Int, Int, Int, UInt16, Int}
const _TRACE_BUFFER = Ref{Union{Nothing, Vector{_TraceTuple}}}(nothing)

"""
    trace_enable!()

Begin recording bus-op events for diagnostic comparison against xitari.
See `tools/cpu_tia_cycle_trace.jl` for the canonical user.
"""
trace_enable!()  = (_TRACE_BUFFER[] = _TraceTuple[]; nothing)

"""
    trace_disable!()

Stop recording. The buffer is discarded.
"""
trace_disable!() = (_TRACE_BUFFER[] = nothing; nothing)

"""
    trace_take!() -> Vector{_TraceTuple}

Return the current trace buffer and reset to empty (still enabled).
Returns an empty vector if tracing is disabled.
"""
function trace_take!()
    buf = _TRACE_BUFFER[]
    buf === nothing && return _TraceTuple[]
    out = buf
    _TRACE_BUFFER[] = _TraceTuple[]
    return out
end

# Internal: per-event record. Uses a closure-free reference to keep the
# disabled-path branch as cheap as a single pointer load + nil-check.
# Calls `_trace_record!(:kind, addr, value)` with the *current* TIA timing
# (passed in via the global ref's last-bus pointer). For simplicity we
# look up the live BusState through a second ref written by peek/poke.
const _TRACE_LIVE_TIA = Ref{Union{Nothing, TIAState}}(nothing)

@inline function _trace_record!(kind::Symbol, addr::UInt16, value::Integer)
    buf = _TRACE_BUFFER[]
    buf === nothing && return
    tia = _TRACE_LIVE_TIA[]
    sl  = tia === nothing ? 0 : Int(tia.scanline)
    sc  = tia === nothing ? 0 : Int(tia.scanline_cycle)
    cc  = tia === nothing ? 0 : Int(tia.color_clock)
    push!(buf, (kind, sl, sc, cc, addr, Int(value)))
    return
end

# --------------------------------------------------------------------------- #
# peek / poke! — multiple dispatch on the world type
# --------------------------------------------------------------------------- #

# TIA-read driven-bits mask — which bits the TIA actively drives. The
# **noise** (floating-bus contribution) is always `data_bus_state &
# 0x3F` — only the low 6 bits float, never bits 6 or 7 (xitari does
# `noise = mySystem->getDataBusState() & 0x3F` unconditionally before
# the per-register OR). Bits the TIA neither drives nor lets float
# (e.g. D6 of CXBLPF / D6 of INPT) read as 0. 1-indexed for Julia.
#
#   CXM0P/CXM1P/CXP0FB/CXP1FB/CXM0FB/CXM1FB (regs 0..5): D7+D6 driven (0xC0)
#   CXBLPF (reg 6):                                       D7 only      (0x80)
#   CXPPMM (reg 7):                                       D7+D6 driven (0xC0)
#   INPT0..INPT5 (regs 8..13):                            D7 only      (0x80)
#   regs 14, 15:                                          nothing      (0x00)
const _TIA_PEEK_DRIVEN_MASK = (
    0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xC0,
    0x80, 0xC0,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x00, 0x00,
)

# Noise is always low 6 bits of the data bus, masked unconditionally.
const _TIA_NOISE_MASK = UInt8(0x3F)

"""
    peek(world, addr) -> UInt8

Read a byte from `world` at the 16-bit CPU address `addr`. For a `BusState`
the 6507 address decode is applied; for a `Vector{UInt8}` the byte at
`(addr & 0xFFFF) + 1` is returned (1-based Julia indexing).

PXC1-x round 3: `peek(bus::BusState, ...)` updates `bus.data_bus_state`
on every read. For TIA reads, the un-driven bits are taken from the
*previous* `data_bus_state` (the floating-bus quirk).
"""
@inline peek(memory::Vector{UInt8}, addr::Integer) =
    memory[(Int(addr) & 0xFFFF) + 1]

# P3i-d2: mid-scanline collision catch-up. Mutates `tia.collisions` in
# place, OR'ing per-pixel collision bits for color clocks
# `[HBLANK_COLOR_CLOCKS, effective_beam_cc]`. Idempotent — `tia_advance!`'s
# end-of-scanline render can re-cover the prefix without double-count.
# Mirrors `jaxtari/jaxtari/bus/system.py::_tia_catch_up_collisions`.
#
# Task #67 (Phase 1 follow-on): `effective_beam_cc` now includes the
# sub-instruction cycle drift (`pending_tia_cycles * 3`). Before this
# patch we caught up to `tia.color_clock` (= the LAST tia_advance!
# point, typically instruction start), missing 0..18 color clocks of
# progress. That under-counts collisions for any read in the middle
# or end of a multi-cycle instruction — which is exactly the breakout
# ball-doesn't-die failure mode (CXBLPF is read late in an instruction
# but our catch-up didn't include the cycles between instruction-start
# and the actual peek).
@inline function _tia_catch_up_collisions!(tia::TIAState, effective_cc::Int = Int(tia.color_clock))
    effective_cc <= HBLANK_COLOR_CLOCKS && return       # still in HBLANK
    # Clamp to scanline end — render-loop also stops at COLOR_CLOCKS_PER_SCANLINE.
    end_cc = min(effective_cc, COLOR_CLOCKS_PER_SCANLINE - 1)
    sets = _object_pixel_sets(tia)
    for c in HBLANK_COLOR_CLOCKS:end_cc
        _apply_pixel_collisions!(tia, c - HBLANK_COLOR_CLOCKS, sets)
    end
    return
end

@inline function peek(bus::BusState, addr::Integer)
    a = Int(addr) & 0x1FFF                       # 6507 13-bit mirror
    # P3i-g: every bus op is one CPU cycle. Reads only *count* the cycle
    # (no TIA flush) so a later TIA write threads the right cycle count.
    bus.pending_tia_cycles += 1
    _TRACE_LIVE_TIA[] = bus.tia                  # Phase 1 diagnostic
    if (a & 0x1000) != 0
        v = cart_peek(bus.cart, a)               # cartridge (may switch bank)
        bus.data_bus_state = v
        _trace_record!(:peek, UInt16(a), v)
        return v
    end
    if (a & 0x80) == 0
        # P3i-d2: for collision-register reads (regs $00-$07), run
        # partial-scanline per-pixel collision evaluation up to
        # current beam position before returning the latch. xitari
        # `TIA::peek` calls `updateFrame(mySystem->cycles() * 3)`
        # before reading any latched-bit register. Idempotent OR so
        # the eventual end-of-scanline render in `tia_advance!` can
        # re-cover the prefix without double-count.
        #
        # Task #67: effective_cc includes the sub-instruction cycle
        # drift (`pending_tia_cycles * 3`). This is the cycle the
        # peek would resolve to under full read-side TIA flushing,
        # WITHOUT actually advancing the TIA (which has the 6×
        # slowdown problem documented in bug_fix_log P3i-d2).
        if (a & 0x0F) < 8
            effective_cc = Int(bus.tia.color_clock) + bus.pending_tia_cycles * 3
            _tia_catch_up_collisions!(bus.tia, effective_cc)
        end
        raw = tia_peek(bus.tia, a)               # TIA register read
        mask = UInt8(_TIA_PEEK_DRIVEN_MASK[(a & 0x0F) + 1])
        noise = bus.data_bus_state & _TIA_NOISE_MASK
        v = UInt8(((raw & mask) | noise) & 0xFF)
        bus.data_bus_state = v
        _trace_record!(:peek, UInt16(a), v)
        return v
    end
    if (a & 0x200) != 0
        v = riot_peek!(bus.riot, a)              # RIOT timer + I/O ports (P4d)
        bus.data_bus_state = v
        _trace_record!(:peek, UInt16(a), v)
        return v
    end
    v = bus.ram[(a & 0x7F) + 1]                  # RIOT RAM
    bus.data_bus_state = v
    _trace_record!(:peek, UInt16(a), v)
    return v
end

"""
    poke!(world, addr, value)

Write `value` (low 8 bits) to `world` at `addr`. Mutates in place. For a
`BusState`, RAM writes land; ROM / TIA / RIOT writes are silently dropped
(TIA/RIOT proper behaviour lands in P3/P4). On a `BusState` the
`data_bus_state` is updated to the byte being written (xitari mirrors
this in `System::poke` — needed by the floating-bus quirk).
"""
@inline function poke!(memory::Vector{UInt8}, addr::Integer, value::Integer)
    memory[(Int(addr) & 0xFFFF) + 1] = UInt8(Int(value) & 0xFF)
    return nothing
end

@inline function poke!(bus::BusState, addr::Integer, value::Integer)
    a = Int(addr) & 0x1FFF
    v8 = UInt8(Int(value) & 0xFF)
    bus.data_bus_state = v8
    # P3i-g: this bus op is one CPU cycle. `pending_tia_cycles` counts
    # cycles since instruction start (reset in `_tia_post_step!`).
    bus.pending_tia_cycles += 1
    _TRACE_LIVE_TIA[] = bus.tia                  # Phase 1 diagnostic
    if (a & 0x1000) != 0
        cart_poke!(bus.cart, a, value)           # cart hotspot may switch bank
        _trace_record!(:poke, UInt16(a), v8)
        return nothing
    end
    if (a & 0x80) == 0
        # TIA write — P3i-g TIMING-ONLY threading: pass the *effective*
        # sub-instruction beam position (instruction-start beam + cycles
        # consumed so far) so PF*/RESP*/HMOVE land at the right color
        # clock, but DO NOT advance/render the TIA here. Advancing inline
        # rendered scanlines prematurely (mid-instruction), capturing the
        # pre-clear PF pattern and leaking it into later scanlines (the
        # Breakout "red columns"). The TIA is advanced exactly once per
        # instruction in `_tia_post_step!`, draining the deferred writes
        # at their (now-accurate) activation clocks.
        beam_cc = Int(bus.tia.color_clock) + bus.pending_tia_cycles * 3
        beam_sc = Int(bus.tia.scanline_cycle) + bus.pending_tia_cycles
        tia_poke!(bus.tia, a, value, beam_cc, beam_sc)
        _trace_record!(:poke, UInt16(a), v8)
        return nothing
    end
    if (a & 0x200) != 0
        riot_poke!(bus.riot, a, value)           # RIOT timer / ports write
        _trace_record!(:poke, UInt16(a), v8)
        return nothing
    end
    bus.ram[(a & 0x7F) + 1] = v8
    _trace_record!(:poke, UInt16(a), v8)
    return nothing
end

end # module
