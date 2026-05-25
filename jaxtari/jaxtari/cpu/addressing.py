"""Effective-address resolution for the 6502's addressing modes.

Each `resolve_*` function takes the `CPUState` (for X / Y / PC) and a "world"
(either a `jaxtari.bus.Bus` for proper 6507 emulation, or a flat
`jnp.ndarray` for unit-test scratch memory), and returns
`(effective_addr, page_crossed, new_memory)` as plain Python ints / bools
plus a possibly-updated world. **The new_memory threading is P4d** —
RIOT reads can clear latches, so the caller has to carry the new
world forward (`memory = new_memory` after each resolver call).

`page_crossed` is only meaningful for absolute,X / absolute,Y / (indirect),Y
modes — read instructions add +1 cycle on a crossing, store instructions do not.

Memory reads are routed through `jaxtari.bus.peek` (which now also
returns `(value, new_world)`) so the same code works for both world
types and a single resolver call correctly threads any RIOT side-
effects produced during address resolution.

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


def _peek16(memory, addr: int):
    """Little-endian 16-bit read across two consecutive bytes (no page-wrap
    quirk). Returns `(word, new_memory)` so any side-effect bus updates
    from either byte read carry forward."""
    lo, memory = _peek(memory, addr)
    hi, memory = _peek(memory, addr + 1)
    return lo | (hi << 8), memory


def resolve_immediate(state: CPUState, memory):
    return (int(state.PC) + 1) & 0xFFFF, False, memory


def resolve_zero(state: CPUState, memory):
    operand, memory = _peek(memory, int(state.PC) + 1)
    return operand, False, memory


def resolve_zero_x(state: CPUState, memory):
    operand, memory = _peek(memory, int(state.PC) + 1)
    return (operand + int(state.X)) & 0xFF, False, memory


def resolve_zero_y(state: CPUState, memory):
    operand, memory = _peek(memory, int(state.PC) + 1)
    return (operand + int(state.Y)) & 0xFF, False, memory


def resolve_absolute(state: CPUState, memory):
    addr, memory = _peek16(memory, int(state.PC) + 1)
    return addr, False, memory


def resolve_absolute_x(state: CPUState, memory):
    """`abs,X`: effective = base + X. Task #50: on a page cross the
    real 6502 does a "wrong-page" dummy peek at (base_hi, eff_lo)
    before the corrected fetch — visible on the data bus, so the
    floating-bus latch updates."""
    base, memory = _peek16(memory, int(state.PC) + 1)
    eff = (base + int(state.X)) & 0xFFFF
    page_crossed = (base & 0xFF00) != (eff & 0xFF00)
    if page_crossed:
        wrong_addr = (base & 0xFF00) | (eff & 0xFF)
        _, memory = _peek(memory, wrong_addr)              # wrong-page dummy
    return eff, page_crossed, memory


def resolve_absolute_y(state: CPUState, memory):
    """`abs,Y`: as `abs,X` — page-cross wrong-page dummy peek."""
    base, memory = _peek16(memory, int(state.PC) + 1)
    eff = (base + int(state.Y)) & 0xFFFF
    page_crossed = (base & 0xFF00) != (eff & 0xFF00)
    if page_crossed:
        wrong_addr = (base & 0xFF00) | (eff & 0xFF)
        _, memory = _peek(memory, wrong_addr)              # wrong-page dummy
    return eff, page_crossed, memory


def resolve_indirect_x(state: CPUState, memory):
    """`(zp,X)`: pointer = (operand + X) & 0xFF, address = mem[pointer..+1] (zp-wrapped).

    Task #50: real NMOS 6502 inserts a one-cycle "internal" peek of
    `zp` (the operand byte itself, BEFORE adding X) at cycle 3,
    discarding the result. This dummy peek is visible on the data
    bus, so it updates the floating-bus latch the TIA's INPT/
    collision reads OR into D5-D0. Mirrors xitari's M6502 (zp,X)
    handling.
    """
    operand, memory = _peek(memory, int(state.PC) + 1)
    _, memory       = _peek(memory, operand)               # internal-cycle peek
    zp = (operand + int(state.X)) & 0xFF
    lo, memory = _peek(memory, zp)
    hi, memory = _peek(memory, (zp + 1) & 0xFF)
    return lo | (hi << 8), False, memory


def resolve_indirect_y(state: CPUState, memory):
    """`(zp),Y`: address = mem[zp..+1] (zp-wrapped) + Y.

    Task #50: on a page cross the real 6502 does a "wrong-page"
    dummy peek at (base_hi, eff_lo) before the corrected fetch —
    visible on the data bus, so the floating-bus latch updates.
    """
    zp, memory = _peek(memory, int(state.PC) + 1)
    lo, memory = _peek(memory, zp)
    hi, memory = _peek(memory, (zp + 1) & 0xFF)
    base = lo | (hi << 8)
    eff = (base + int(state.Y)) & 0xFFFF
    page_crossed = (base & 0xFF00) != (eff & 0xFF00)
    if page_crossed:
        wrong_addr = (base & 0xFF00) | (eff & 0xFF)
        _, memory = _peek(memory, wrong_addr)              # wrong-page dummy
    return eff, page_crossed, memory


def resolve_relative(state: CPUState, memory):
    """Branch target = PC + 2 + signed(operand). `page_crossed` reports if it crosses."""
    offset, memory = _peek(memory, int(state.PC) + 1)
    if offset >= 0x80:
        offset -= 0x100
    base = (int(state.PC) + 2) & 0xFFFF
    eff = (base + offset) & 0xFFFF
    return eff, (base & 0xFF00) != (eff & 0xFF00), memory


def resolve_indirect(state: CPUState, memory):
    """JMP indirect, faithfully replicating the 6502 page-wrap bug at $xxFF."""
    ptr, memory = _peek16(memory, int(state.PC) + 1)
    lo, memory = _peek(memory, ptr)
    hi, memory = _peek(memory, (ptr & 0xFF00) | ((ptr + 1) & 0xFF))
    return lo | (hi << 8), False, memory


Resolver = Callable[[CPUState, object], Tuple[int, bool, object]]

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
