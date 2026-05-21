"""
    ALU

6502 ALU flag helpers (mutating; mirror of `jaxtari/jaxtari/cpu/alu.py`).

- P1a:  `set_zn!`
- P1b1: `compare_flags!` (CMP/CPX/CPY), `bit_flags!` (BIT)
- P1b2: `adc!`, `sbc!` — binary and BCD (decimal mode), matching xitari's
        M6502Hi.ins NMOS semantics bit-for-bit (case 0x69 / 0xe9).
- P1c:  shift/rotate carry handling — pending.

Each helper mutates `state.P` (and for adc!/sbc!, `state.A`) in place and
force-asserts `FLAG_U` (bit 5) which is always read as 1 on real hardware.
"""
module ALU

using ..CPUTables: FLAG_N, FLAG_Z, FLAG_C, FLAG_V, FLAG_U, FLAG_D
# Three dots: see note in Addressing.jl on the JuTari.CPU.Addressing module
# vs. JuTari.Types path.
using ...Types: CPUState

export set_zn!, compare_flags!, bit_flags!, adc!, sbc!

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

# --------------------------------------------------------------------------- #
# P1b2 — ADC / SBC
# Reference: xitari/emucore/m6502/src/M6502Hi.ins case 0x69 and case 0xe9.
# --------------------------------------------------------------------------- #

"""BCD byte → integer 0..99 (high nibble * 10 + low nibble)."""
@inline _bcd_to_int(b::Integer) = ((Int(b) >> 4) & 0x0F) * 10 + (Int(b) & 0x0F)

"""Mirror of xitari's `ourBCDTable[1]`: BCD encoding of `n mod 100`."""
@inline function _int_to_bcd_byte(n::Integer)
    m = mod(Int(n), 100)
    return UInt8(((m ÷ 10) << 4) | (m % 10))
end

"""
    adc!(state, operand)

ADC: A = A + operand + C, dispatched on the D flag. Mutates `state.A` and
`state.P` to match xitari NMOS semantics.
"""
function adc!(state::CPUState, operand::Integer)
    a = Int(state.A) & 0xFF
    op = Int(operand) & 0xFF
    c_in = (state.P & FLAG_C) != 0 ? 1 : 0
    decimal = (state.P & FLAG_D) != 0
    old_a = a

    if !decimal
        # Binary
        a_signed = a >= 0x80 ? a - 0x100 : a
        op_signed = op >= 0x80 ? op - 0x100 : op
        sum_signed = a_signed + op_signed + c_in
        v_set = (sum_signed > 127) || (sum_signed < -128)

        sum_unsigned = a + op + c_in
        new_a = sum_unsigned & 0xFF
        c_set = sum_unsigned > 0xFF
    else
        # BCD
        bcd_sum = _bcd_to_int(a) + _bcd_to_int(op) + c_in
        new_a = Int(_int_to_bcd_byte(bcd_sum & 0xFF))
        c_set = bcd_sum > 99
        v_set = (((old_a ⊻ new_a) & 0x80) != 0) && (((new_a ⊻ op) & 0x80) != 0)
    end

    p = state.P & ~(FLAG_C | FLAG_Z | FLAG_N | FLAG_V)
    c_set && (p |= FLAG_C)
    new_a == 0 && (p |= FLAG_Z)
    (new_a & 0x80) != 0 && (p |= FLAG_N)
    v_set && (p |= FLAG_V)

    state.A = UInt8(new_a & 0xFF)
    state.P = (p | FLAG_U) & UInt8(0xFF)
    return nothing
end

"""
    sbc!(state, operand)

SBC: A = A - operand - (1 - C), dispatched on the D flag. Mutates `state.A`
and `state.P` to match xitari NMOS semantics (case 0xe9 / 0xeb).
"""
function sbc!(state::CPUState, operand::Integer)
    a = Int(state.A) & 0xFF
    op = Int(operand) & 0xFF
    c_in = (state.P & FLAG_C) != 0 ? 1 : 0
    decimal = (state.P & FLAG_D) != 0
    old_a = a

    if !decimal
        # Binary SBC ≡ ADC with ~operand
        op_inv = (~op) & 0xFF
        a_signed = a >= 0x80 ? a - 0x100 : a
        op_signed = op_inv >= 0x80 ? op_inv - 0x100 : op_inv
        diff_signed = a_signed + op_signed + c_in
        v_set = (diff_signed > 127) || (diff_signed < -128)

        diff_unsigned = a + op_inv + c_in
        new_a = diff_unsigned & 0xFF
        c_set = diff_unsigned > 0xFF
    else
        # BCD SBC
        diff = _bcd_to_int(a) - _bcd_to_int(op) - (1 - c_in)
        diff < 0 && (diff += 100)
        new_a = Int(_int_to_bcd_byte(diff))
        # xitari: C uses ORIGINAL bytes (not BCD-decoded)
        c_set = old_a >= (op + (1 - c_in))
        v_set = (((old_a ⊻ new_a) & 0x80) != 0) && (((new_a ⊻ op) & 0x80) != 0)
    end

    p = state.P & ~(FLAG_C | FLAG_Z | FLAG_N | FLAG_V)
    c_set && (p |= FLAG_C)
    new_a == 0 && (p |= FLAG_Z)
    (new_a & 0x80) != 0 && (p |= FLAG_N)
    v_set && (p |= FLAG_V)

    state.A = UInt8(new_a & 0xFF)
    state.P = (p | FLAG_U) & UInt8(0xFF)
    return nothing
end

end # module
