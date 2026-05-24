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

using ..TIA: TIAState, initial_tia_state, tia_peek, tia_poke!
using ..RIOT: RIOTState, initial_riot_state, riot_peek!, riot_poke!
using ..Cart: CartState, make_cart, cart_peek, cart_poke!

export BusState, initial_bus, peek, poke!

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
end

"""
    initial_bus(rom=nothing) -> BusState

Build a `BusState` with all-zero RAM, an auto-detected cart built from
`rom` (default: all-zero 4 KB), and fresh TIA / RIOT states.
"""
function initial_bus(rom=nothing)
    rom === nothing && (rom = zeros(UInt8, 4096))
    return BusState(zeros(UInt8, 128), make_cart(rom),
                    initial_tia_state(), initial_riot_state(), 0x00)
end

# --------------------------------------------------------------------------- #
# peek / poke! — multiple dispatch on the world type
# --------------------------------------------------------------------------- #

# TIA-read driven-bits mask — which bits the TIA actively drives. The
# other bits float and resolve to the last data-bus byte. Matches
# xitari's TIA::peek (and the real 1A05 hardware). 1-indexed for Julia.
#
#   CXM0P/CXM1P/CXP0FB/CXP1FB/CXM0FB/CXM1FB (regs 0..5): D7+D6 driven (0xC0)
#   CXBLPF (reg 6):                                       D7 only      (0x80)
#   CXPPMM (reg 7):                                       D7+D6 driven (0xC0)
#   INPT0..INPT5 (regs 8..13):                            D7 only      (0x80)
#   regs 14, 15:                                          fully float  (0x00)
const _TIA_PEEK_DRIVEN_MASK = (
    0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xC0,
    0x80, 0xC0,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x00, 0x00,
)

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

@inline function peek(bus::BusState, addr::Integer)
    a = Int(addr) & 0x1FFF                       # 6507 13-bit mirror
    if (a & 0x1000) != 0
        v = cart_peek(bus.cart, a)               # cartridge (may switch bank)
        bus.data_bus_state = v
        return v
    end
    if (a & 0x80) == 0
        raw = tia_peek(bus.tia, a)               # TIA register read
        mask = UInt8(_TIA_PEEK_DRIVEN_MASK[(a & 0x0F) + 1])
        noise = bus.data_bus_state & UInt8(~mask & 0xFF)
        v = UInt8(((raw & mask) | noise) & 0xFF)
        bus.data_bus_state = v
        return v
    end
    if (a & 0x200) != 0
        v = riot_peek!(bus.riot, a)              # RIOT timer + I/O ports (P4d)
        bus.data_bus_state = v
        return v
    end
    v = bus.ram[(a & 0x7F) + 1]                  # RIOT RAM
    bus.data_bus_state = v
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
    if (a & 0x1000) != 0
        cart_poke!(bus.cart, a, value)           # cart hotspot may switch bank
        return nothing
    end
    if (a & 0x80) == 0
        tia_poke!(bus.tia, a, value)             # TIA write — records byte + WSYNC
        return nothing
    end
    if (a & 0x200) != 0
        riot_poke!(bus.riot, a, value)           # RIOT timer / ports write
        return nothing
    end
    bus.ram[(a & 0x7F) + 1] = v8
    return nothing
end

end # module
