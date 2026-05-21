"""6502 ALU flag helpers.

- P1a:  `set_zn` (loads, transfers).
- P1b1: `compare_flags` (CMP/CPX/CPY), `bit_flags` (BIT).
- P1b2: `adc`, `sbc` — binary and BCD (decimal mode), matching xitari's
        M6502Hi.ins NMOS semantics bit-for-bit (see comments inline).
- P1c:  `asl_op`, `lsr_op`, `rol_op`, `ror_op` — shifts and rotates with
        carry-in / carry-out.

All helpers take and return a plain Python int for the status byte `P`. The
constant bit 5 (`FLAG_U`) is asserted at the end of each helper because the
real 6502 always reads bit 5 as 1.
"""

from typing import Tuple

from jaxtari.cpu.tables import FLAG_C, FLAG_D, FLAG_N, FLAG_U, FLAG_V, FLAG_Z


# --------------------------------------------------------------------------- #
# P1a / P1b1
# --------------------------------------------------------------------------- #

def set_zn(p: int, value: int) -> int:
    """Return `p` with Z and N derived from `value` (8-bit)."""
    p = p & (0xFF ^ FLAG_Z ^ FLAG_N)
    if (value & 0xFF) == 0:
        p |= FLAG_Z
    if value & 0x80:
        p |= FLAG_N
    return (p | FLAG_U) & 0xFF


def compare_flags(p: int, reg: int, operand: int) -> int:
    """For CMP / CPX / CPY: set Z, N, C from `reg - operand` (8-bit unsigned).

    - Z = 1 if reg == operand
    - N = 1 if (reg - operand) & 0x80
    - C = 1 if reg >= operand (no borrow)
    """
    reg &= 0xFF
    operand &= 0xFF
    diff = (reg - operand) & 0xFF
    p = p & (0xFF ^ FLAG_Z ^ FLAG_N ^ FLAG_C)
    if diff == 0:
        p |= FLAG_Z
    if diff & 0x80:
        p |= FLAG_N
    if reg >= operand:
        p |= FLAG_C
    return (p | FLAG_U) & 0xFF


def bit_flags(p: int, a: int, operand: int) -> int:
    """For BIT: Z from `A AND operand`; N from operand bit 7; V from operand bit 6."""
    p = p & (0xFF ^ FLAG_Z ^ FLAG_N ^ FLAG_V)
    if ((a & operand) & 0xFF) == 0:
        p |= FLAG_Z
    if operand & 0x80:
        p |= FLAG_N
    if operand & 0x40:
        p |= FLAG_V
    return (p | FLAG_U) & 0xFF


# --------------------------------------------------------------------------- #
# P1b2 — ADC / SBC
#
# Reference: xitari/emucore/m6502/src/M6502Hi.ins, case 0x69 (ADC) and case
# 0xe9 (SBC). NMOS 6502 semantics; the V flag in decimal mode follows the
# xitari/Stella convention, which differs from CMOS 65C02 behaviour.
# --------------------------------------------------------------------------- #

def _bcd_to_int(b: int) -> int:
    """Treat `b` as two BCD digits → integer 0..99 (high nibble * 10 + low)."""
    return ((b >> 4) & 0x0F) * 10 + (b & 0x0F)


def _int_to_bcd_byte(n: int) -> int:
    """Mirror of xitari's `ourBCDTable[1]`: BCD encoding of `n mod 100`.

    For `n` in 0..199 (the only range adc / sbc produce here) this is
    equivalent to `((n / 10 % 10) << 4) | (n % 10)`.
    """
    n %= 100
    return ((n // 10) << 4) | (n % 10)


def adc(p: int, a: int, operand: int) -> Tuple[int, int]:
    """ADC: A = A + operand + C, with D-flag (decimal) dispatch.

    Returns `(new_a, new_p)`. Matches xitari/M6502Hi.ins case 0x69.
    """
    a &= 0xFF
    operand &= 0xFF
    c_in = 1 if (p & FLAG_C) else 0
    decimal = (p & FLAG_D) != 0
    old_a = a

    if not decimal:
        # Binary mode (D=0)
        a_signed = a - 0x100 if a >= 0x80 else a
        op_signed = operand - 0x100 if operand >= 0x80 else operand
        sum_signed = a_signed + op_signed + c_in
        v_set = (sum_signed > 127) or (sum_signed < -128)

        sum_unsigned = a + operand + c_in
        new_a = sum_unsigned & 0xFF
        c_set = sum_unsigned > 0xFF
    else:
        # BCD mode (D=1) — xitari uses an integer detour via ourBCDTable
        bcd_sum = _bcd_to_int(a) + _bcd_to_int(operand) + c_in
        new_a = _int_to_bcd_byte(bcd_sum & 0xFF)
        c_set = bcd_sum > 99
        # V is computed against the BCD-encoded result, following xitari's
        # `((oldA ^ A) & 0x80) && ((A ^ operand) & 0x80)` formulation.
        v_set = (((old_a ^ new_a) & 0x80) != 0) and (((new_a ^ operand) & 0x80) != 0)

    p = p & (0xFF ^ FLAG_C ^ FLAG_Z ^ FLAG_N ^ FLAG_V)
    if c_set:
        p |= FLAG_C
    if new_a == 0:
        p |= FLAG_Z
    if new_a & 0x80:
        p |= FLAG_N
    if v_set:
        p |= FLAG_V
    return new_a, (p | FLAG_U) & 0xFF


def sbc(p: int, a: int, operand: int) -> Tuple[int, int]:
    """SBC: A = A - operand - (1 - C), with D-flag (decimal) dispatch.

    Returns `(new_a, new_p)`. Matches xitari/M6502Hi.ins case 0xe9 / 0xeb.
    """
    a &= 0xFF
    operand &= 0xFF
    c_in = 1 if (p & FLAG_C) else 0
    decimal = (p & FLAG_D) != 0
    old_a = a

    if not decimal:
        # Binary SBC ≡ ADC with one's-complement operand
        op_inv = (~operand) & 0xFF
        a_signed = a - 0x100 if a >= 0x80 else a
        op_signed = op_inv - 0x100 if op_inv >= 0x80 else op_inv
        diff_signed = a_signed + op_signed + c_in
        v_set = (diff_signed > 127) or (diff_signed < -128)

        diff_unsigned = a + op_inv + c_in
        new_a = diff_unsigned & 0xFF
        c_set = diff_unsigned > 0xFF
    else:
        # BCD SBC
        diff = _bcd_to_int(a) - _bcd_to_int(operand) - (1 - c_in)
        if diff < 0:
            diff += 100
        new_a = _int_to_bcd_byte(diff)
        # xitari: C uses the ORIGINAL bytes (not BCD-decoded), with the
        # pre-op carry inverted as in the standard SBC carry convention.
        c_set = old_a >= (operand + (1 - c_in))
        v_set = (((old_a ^ new_a) & 0x80) != 0) and (((new_a ^ operand) & 0x80) != 0)

    p = p & (0xFF ^ FLAG_C ^ FLAG_Z ^ FLAG_N ^ FLAG_V)
    if c_set:
        p |= FLAG_C
    if new_a == 0:
        p |= FLAG_Z
    if new_a & 0x80:
        p |= FLAG_N
    if v_set:
        p |= FLAG_V
    return new_a, (p | FLAG_U) & 0xFF


# --------------------------------------------------------------------------- #
# P1c — shifts and rotates
# Reference: xitari/emucore/m6502/src/M6502Hi.ins, the ASL/LSR/ROL/ROR cases.
# All four return (result_byte, new_p) and update C, Z, N. The dispatcher in
# `cpu.m6502` decides whether the result goes back into A (accumulator mode)
# or into memory (zp / zp,X / abs / abs,X).
# --------------------------------------------------------------------------- #

def asl_op(p: int, value: int) -> Tuple[int, int]:
    """Arithmetic shift left: bit 7 → C; bit 0 ← 0; N/Z from result."""
    value &= 0xFF
    new_c = (value >> 7) & 1
    result = (value << 1) & 0xFF
    p = p & (0xFF ^ FLAG_C ^ FLAG_Z ^ FLAG_N)
    if new_c:
        p |= FLAG_C
    if result == 0:
        p |= FLAG_Z
    if result & 0x80:
        p |= FLAG_N
    return result, (p | FLAG_U) & 0xFF


def lsr_op(p: int, value: int) -> Tuple[int, int]:
    """Logical shift right: bit 0 → C; bit 7 ← 0; N always 0; Z from result."""
    value &= 0xFF
    new_c = value & 1
    result = (value >> 1) & 0x7F  # bit 7 forced to 0 by the shift
    p = p & (0xFF ^ FLAG_C ^ FLAG_Z ^ FLAG_N)
    if new_c:
        p |= FLAG_C
    if result == 0:
        p |= FLAG_Z
    # N is always 0 after LSR — no need to set.
    return result, (p | FLAG_U) & 0xFF


def rol_op(p: int, value: int) -> Tuple[int, int]:
    """Rotate left through carry: old C → bit 0; bit 7 → new C; N/Z from result."""
    value &= 0xFF
    c_in = 1 if (p & FLAG_C) else 0
    new_c = (value >> 7) & 1
    result = ((value << 1) | c_in) & 0xFF
    p = p & (0xFF ^ FLAG_C ^ FLAG_Z ^ FLAG_N)
    if new_c:
        p |= FLAG_C
    if result == 0:
        p |= FLAG_Z
    if result & 0x80:
        p |= FLAG_N
    return result, (p | FLAG_U) & 0xFF


def ror_op(p: int, value: int) -> Tuple[int, int]:
    """Rotate right through carry: old C → bit 7; bit 0 → new C; N/Z from result."""
    value &= 0xFF
    c_in = 1 if (p & FLAG_C) else 0
    new_c = value & 1
    result = ((value >> 1) | (c_in << 7)) & 0xFF
    p = p & (0xFF ^ FLAG_C ^ FLAG_Z ^ FLAG_N)
    if new_c:
        p |= FLAG_C
    if result == 0:
        p |= FLAG_Z
    if result & 0x80:
        p |= FLAG_N
    return result, (p | FLAG_U) & 0xFF
