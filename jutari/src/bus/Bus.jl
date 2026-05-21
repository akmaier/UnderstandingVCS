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

P2 status: address decode plus RAM/ROM peek/poke. TIA and RIOT regions
are stubs (read 0, ignore writes); proper behaviour lands in P3 (TIA),
P4 (RIOT) and P5 (bank-switching cartridges).

Multiple dispatch handles two world types: a `BusState` (proper 6507
bus) and a flat `Vector{UInt8}` (used by the P1 unit tests). The same
`step` works against both.
"""
module Bus

export BusState, initial_bus, peek, poke!

"""
    BusState

6507 system bus state. `ram` is the 128-byte RIOT internal RAM; `rom` is
the cartridge ROM image (4096 bytes in P2; bank-switched cartridges land
in P5).
"""
mutable struct BusState
    ram::Vector{UInt8}
    rom::Vector{UInt8}
end

"""
    initial_bus(rom=nothing) -> BusState

Build a `BusState` with all-zero RAM and the given 4 KB ROM (default: zeros).
"""
function initial_bus(rom=nothing)
    if rom === nothing
        rom = zeros(UInt8, 4096)
    else
        rom = Vector{UInt8}(rom)
        length(rom) == 4096 || throw(ArgumentError(
            "P2 expects a flat 4K ROM; got length $(length(rom)). " *
            "Bank-switched cartridges land in P5."))
    end
    return BusState(zeros(UInt8, 128), rom)
end

# --------------------------------------------------------------------------- #
# peek / poke! — multiple dispatch on the world type
# --------------------------------------------------------------------------- #

"""
    peek(world, addr) -> UInt8

Read a byte from `world` at the 16-bit CPU address `addr`. For a `BusState`
the 6507 address decode is applied; for a `Vector{UInt8}` the byte at
`(addr & 0xFFFF) + 1` is returned (1-based Julia indexing).
"""
@inline peek(memory::Vector{UInt8}, addr::Integer) =
    memory[(Int(addr) & 0xFFFF) + 1]

@inline function peek(bus::BusState, addr::Integer)
    a = Int(addr) & 0x1FFF                       # 6507 13-bit mirror
    if (a & 0x1000) != 0
        return bus.rom[(a & 0x0FFF) + 1]         # cartridge ROM
    end
    if (a & 0x80) == 0
        return UInt8(0)                          # TIA (P2 stub)
    end
    if (a & 0x200) != 0
        return UInt8(0)                          # RIOT I/O (P2 stub)
    end
    return bus.ram[(a & 0x7F) + 1]               # RIOT RAM
end

"""
    poke!(world, addr, value)

Write `value` (low 8 bits) to `world` at `addr`. Mutates in place. For a
`BusState`, RAM writes land; ROM / TIA / RIOT writes are silently dropped
(TIA/RIOT proper behaviour lands in P3/P4).
"""
@inline function poke!(memory::Vector{UInt8}, addr::Integer, value::Integer)
    memory[(Int(addr) & 0xFFFF) + 1] = UInt8(Int(value) & 0xFF)
    return nothing
end

@inline function poke!(bus::BusState, addr::Integer, value::Integer)
    a = Int(addr) & 0x1FFF
    (a & 0x1000) != 0 && return nothing          # ROM is read-only
    (a & 0x80) == 0   && return nothing          # TIA write stub
    (a & 0x200) != 0  && return nothing          # RIOT I/O write stub
    bus.ram[(a & 0x7F) + 1] = UInt8(Int(value) & 0xFF)
    return nothing
end

end # module
