"""
    ALU

6502 ALU flag helpers. P1a needs only `set_zn!` (loads and transfers update N
and Z). Arithmetic / logical / shift flag updates land in P1b–P1c.

`set_zn!` mutates `state.P` in place; the U (unused, bit 5) flag is always
forced high to match real hardware.
"""
module ALU

using ..CPUTables: FLAG_N, FLAG_Z, FLAG_U
using ..Types: CPUState

export set_zn!

function set_zn!(state::CPUState, value::Integer)
    p = state.P & ~(FLAG_Z | FLAG_N)
    if (Int(value) & 0xFF) == 0
        p |= FLAG_Z
    end
    if Int(value) & 0x80 != 0
        p |= FLAG_N
    end
    state.P = (p | FLAG_U) & UInt8(0xFF)
    return nothing
end

end # module
