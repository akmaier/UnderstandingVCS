"""
    ALU

6502 ALU flag helpers (mutating; mirror of `jaxtari/jaxtari/cpu/alu.py`).

- P1a:  `set_zn!`
- P1b1: `compare_flags!` (CMP/CPX/CPY), `bit_flags!` (BIT)
- P1b2: ADC/SBC arithmetic (incl. decimal mode) — pending
- P1c:  shift/rotate carry handling — pending

Each helper mutates `state.P` in place and force-asserts `FLAG_U` (bit 5)
which is always read as 1 on real hardware.
"""
module ALU

using ..CPUTables: FLAG_N, FLAG_Z, FLAG_C, FLAG_V, FLAG_U
# Three dots: see note in Addressing.jl on the JuTari.CPU.Addressing module
# vs. JuTari.Types path.
using ...Types: CPUState

export set_zn!, compare_flags!, bit_flags!

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

"""
    compare_flags!(state, reg, operand)

For CMP / CPX / CPY: set Z, N, C from `reg - operand` (8-bit unsigned).
C is set when `reg >= operand` (no borrow).
"""
function compare_flags!(state::CPUState, reg::Integer, operand::Integer)
    r = Int(reg) & 0xFF
    o = Int(operand) & 0xFF
    diff = (r - o) & 0xFF
    p = state.P & ~(FLAG_Z | FLAG_N | FLAG_C)
    if diff == 0
        p |= FLAG_Z
    end
    if diff & 0x80 != 0
        p |= FLAG_N
    end
    if r >= o
        p |= FLAG_C
    end
    state.P = (p | FLAG_U) & UInt8(0xFF)
    return nothing
end

"""
    bit_flags!(state, a, operand)

For BIT: Z from `A AND operand`; N from operand bit 7; V from operand bit 6.
"""
function bit_flags!(state::CPUState, a::Integer, operand::Integer)
    p = state.P & ~(FLAG_Z | FLAG_N | FLAG_V)
    if ((Int(a) & Int(operand)) & 0xFF) == 0
        p |= FLAG_Z
    end
    if Int(operand) & 0x80 != 0
        p |= FLAG_N
    end
    if Int(operand) & 0x40 != 0
        p |= FLAG_V
    end
    state.P = (p | FLAG_U) & UInt8(0xFF)
    return nothing
end

end # module
