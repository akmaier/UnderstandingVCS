"""6502 ALU flag helpers.

P1a only needs `set_zn` (loads and transfers update N and Z). Arithmetic /
logical / shift flag updates land in P1b–P1c.

All helpers take and return a plain Python int for the status byte `P`, with
the constant bit 5 (`FLAG_U`) preserved if it was set on input — the dispatcher
in `cpu.m6502` is responsible for keeping bit 5 == 1 across the whole step.
"""

from jaxtari.cpu.tables import FLAG_N, FLAG_Z


def set_zn(p: int, value: int) -> int:
    """Return `p` with the Z and N bits derived from `value` (8-bit)."""
    p = p & (0xFF ^ FLAG_Z ^ FLAG_N)
    if (value & 0xFF) == 0:
        p |= FLAG_Z
    if value & 0x80:
        p |= FLAG_N
    return p & 0xFF
