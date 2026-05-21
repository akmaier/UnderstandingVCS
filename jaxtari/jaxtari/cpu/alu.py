"""6502 ALU flag helpers.

- P1a: `set_zn` (loads, transfers).
- P1b1: `compare_flags` (CMP/CPX/CPY), `bit_flags` (BIT).
- P1b2: ADC/SBC arithmetic flag helpers (incl. decimal mode) land here.
- P1c: shift/rotate carry handling.

All helpers take and return a plain Python int for the status byte `P`. The
constant bit 5 (`FLAG_U`) is asserted at the end of each helper because the
real 6502 always reads bit 5 as 1.
"""

from jaxtari.cpu.tables import FLAG_C, FLAG_N, FLAG_U, FLAG_V, FLAG_Z


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
    - N = 1 if (reg - operand) & 0x80, i.e. the high bit of the 8-bit result
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
