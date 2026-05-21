"""Effective-address resolution for the 6502's addressing modes.

Each `resolve_*` function takes the `CPUState` (for X / Y / PC) and a "world"
(either a `jaxtari.bus.Bus` for proper 6507 emulation, or a flat
`jnp.ndarray` for unit-test scratch memory), and returns
`(effective_addr, page_crossed)` as plain Python ints / bools.
`page_crossed` is only meaningful for absolute,X / absolute,Y / (indirect),Y
modes — read instructions add +1 cycle on a crossing, store instructions do not.

Memory reads are routed through `jaxtari.bus.peek` so the same code works for
both world types.

For implied / accumulator modes the address is meaningless and the dispatcher
in `cpu.m6502` short-circuits without calling these.

PC passed in is the address of the *opcode* itself (operand is at PC+1).
"""

from __future__ import annotations

from typing import Callable, Tuple

from jaxtari.bus.system import peek as _peek
from jaxtari.cpu.tables import (
    ADDR_ABSOLUTE,
    ADDR_ABSOLUTE_X,
    ADDR_ABSOLUTE_Y,
    ADDR_IMMEDIATE,
    ADDR_IMPLIED,
    ADDR_INDIRECT,
    ADDR_INDIRECT_X,
    ADDR_INDIRECT_Y,
    ADDR_RELATIVE,
    ADDR_ZERO,
    ADDR_ZERO_X,
    ADDR_ZERO_Y,
)
from jaxtari.types import CPUState


def _peek16(memory, addr: int) -> int:
    """Little-endian 16-bit read across two consecutive bytes (no page-wrap quirk)."""
    return _peek(memory, addr) | (_peek(memory, addr + 1) << 8)


def resolve_immediate(state: CPUState, memory) -> Tuple[int, bool]:
    return (int(state.PC) + 1) & 0xFFFF, False


def resolve_zero(state: CPUState, memory) -> Tuple[int, bool]:
    return _peek(memory, int(state.PC) + 1), False


def resolve_zero_x(state: CPUState, memory) -> Tuple[int, bool]:
    return (_peek(memory, int(state.PC) + 1) + int(state.X)) & 0xFF, False


def resolve_zero_y(state: CPUState, memory) -> Tuple[int, bool]:
    return (_peek(memory, int(state.PC) + 1) + int(state.Y)) & 0xFF, False


def resolve_absolute(state: CPUState, memory) -> Tuple[int, bool]:
    return _peek16(memory, int(state.PC) + 1), False


def resolve_absolute_x(state: CPUState, memory) -> Tuple[int, bool]:
    base = _peek16(memory, int(state.PC) + 1)
    eff = (base + int(state.X)) & 0xFFFF
    return eff, (base & 0xFF00) != (eff & 0xFF00)


def resolve_absolute_y(state: CPUState, memory) -> Tuple[int, bool]:
    base = _peek16(memory, int(state.PC) + 1)
    eff = (base + int(state.Y)) & 0xFFFF
    return eff, (base & 0xFF00) != (eff & 0xFF00)


def resolve_indirect_x(state: CPUState, memory) -> Tuple[int, bool]:
    """`(zp,X)`: pointer = (operand + X) & 0xFF, address = mem[pointer..+1] (zp-wrapped)."""
    zp = (_peek(memory, int(state.PC) + 1) + int(state.X)) & 0xFF
    lo = _peek(memory, zp)
    hi = _peek(memory, (zp + 1) & 0xFF)
    return lo | (hi << 8), False


def resolve_indirect_y(state: CPUState, memory) -> Tuple[int, bool]:
    """`(zp),Y`: address = mem[zp..+1] (zp-wrapped) + Y."""
    zp = _peek(memory, int(state.PC) + 1)
    lo = _peek(memory, zp)
    hi = _peek(memory, (zp + 1) & 0xFF)
    base = lo | (hi << 8)
    eff = (base + int(state.Y)) & 0xFFFF
    return eff, (base & 0xFF00) != (eff & 0xFF00)


def resolve_relative(state: CPUState, memory) -> Tuple[int, bool]:
    """Branch target = PC + 2 + signed(operand). `page_crossed` reports if it crosses."""
    offset = _peek(memory, int(state.PC) + 1)
    if offset >= 0x80:
        offset -= 0x100
    base = (int(state.PC) + 2) & 0xFFFF
    eff = (base + offset) & 0xFFFF
    return eff, (base & 0xFF00) != (eff & 0xFF00)


def resolve_indirect(state: CPUState, memory) -> Tuple[int, bool]:
    """JMP indirect, faithfully replicating the 6502 page-wrap bug at $xxFF."""
    ptr = _peek16(memory, int(state.PC) + 1)
    lo = _peek(memory, ptr)
    hi = _peek(memory, (ptr & 0xFF00) | ((ptr + 1) & 0xFF))
    return lo | (hi << 8), False


Resolver = Callable[[CPUState, object], Tuple[int, bool]]

RESOLVERS: dict[int, Resolver] = {
    ADDR_IMMEDIATE:  resolve_immediate,
    ADDR_ZERO:       resolve_zero,
    ADDR_ZERO_X:     resolve_zero_x,
    ADDR_ZERO_Y:     resolve_zero_y,
    ADDR_ABSOLUTE:   resolve_absolute,
    ADDR_ABSOLUTE_X: resolve_absolute_x,
    ADDR_ABSOLUTE_Y: resolve_absolute_y,
    ADDR_INDIRECT:   resolve_indirect,
    ADDR_INDIRECT_X: resolve_indirect_x,
    ADDR_INDIRECT_Y: resolve_indirect_y,
    ADDR_RELATIVE:   resolve_relative,
}


INSTRUCTION_LENGTH: dict[int, int] = {
    ADDR_IMPLIED:    1,
    ADDR_IMMEDIATE:  2,
    ADDR_ZERO:       2,
    ADDR_ZERO_X:     2,
    ADDR_ZERO_Y:     2,
    ADDR_ABSOLUTE:   3,
    ADDR_ABSOLUTE_X: 3,
    ADDR_ABSOLUTE_Y: 3,
    ADDR_INDIRECT:   3,
    ADDR_INDIRECT_X: 2,
    ADDR_INDIRECT_Y: 2,
    ADDR_RELATIVE:   2,
}
