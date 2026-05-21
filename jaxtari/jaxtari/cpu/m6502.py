"""6502 fetch–decode–execute step.

Phase P1a status: load / store / transfer opcodes are implemented
(LDA, LDX, LDY, STA, STX, STY, TAX, TAY, TXA, TYA, TSX, TXS — 37 opcodes
total). All other opcodes fall through to the P0 stub that advances PC by 1
and bumps `cycles` by the base count. P1b–P1f will fill the rest.

Design:
- `step(state, memory)` is *one 6502 instruction* (not one machine cycle).
- Decode uses `cpu.tables.ADDRESSING_MODE_TABLE` and `cpu.tables.CYCLE_TABLE`.
- Effective-address resolution lives in `cpu.addressing`.
- Flag-setting helpers (currently just `set_zn`) live in `cpu.alu`.
- Dispatch is a Python dict lookup on opcode → mnemonic → handler. This is
  fine for HARD-mode correctness; jit-friendly `jax.lax.switch` dispatch is a
  P1-final optimisation (PORTING_PLAN.md §6.1).
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp

from jaxtari.cpu.addressing import INSTRUCTION_LENGTH, RESOLVERS
from jaxtari.cpu.alu import set_zn
from jaxtari.cpu.tables import ADDRESSING_MODE_TABLE, CYCLE_TABLE, FLAG_U
from jaxtari.types import CPUState

Memory = jnp.ndarray  # shape (1 << 16,), dtype uint8


# Opcode → P1a mnemonic. Anything not in this table falls through to the stub.
P1A_OPCODES: dict[int, str] = {
    # LDA — 8 modes
    0xA9: "LDA", 0xA5: "LDA", 0xB5: "LDA", 0xAD: "LDA",
    0xBD: "LDA", 0xB9: "LDA", 0xA1: "LDA", 0xB1: "LDA",
    # LDX — 5 modes (no zp,X — uses zp,Y instead)
    0xA2: "LDX", 0xA6: "LDX", 0xB6: "LDX", 0xAE: "LDX", 0xBE: "LDX",
    # LDY — 5 modes
    0xA0: "LDY", 0xA4: "LDY", 0xB4: "LDY", 0xAC: "LDY", 0xBC: "LDY",
    # STA — 7 modes (no immediate)
    0x85: "STA", 0x95: "STA", 0x8D: "STA",
    0x9D: "STA", 0x99: "STA", 0x81: "STA", 0x91: "STA",
    # STX — 3 modes
    0x86: "STX", 0x96: "STX", 0x8E: "STX",
    # STY — 3 modes
    0x84: "STY", 0x94: "STY", 0x8C: "STY",
    # Transfers — implied, 1 byte, 2 cycles each
    0xAA: "TAX", 0xA8: "TAY",
    0x8A: "TXA", 0x98: "TYA",
    0xBA: "TSX", 0x9A: "TXS",
}


def _stub_advance(state: CPUState, base_cycles: int) -> CPUState:
    """Unimplemented opcode: advance PC by 1, bump cycles by the base count."""
    return state._replace(
        PC=jnp.uint16((int(state.PC) + 1) & 0xFFFF),
        cycles=state.cycles + jnp.uint64(base_cycles),
    )


def _commit_load(
    state: CPUState,
    reg: str,
    value: int,
    mode: int,
    base_cycles: int,
    extra: int,
    update_flags: bool = True,
) -> CPUState:
    """Common path for LDA / LDX / LDY (and the flag-setting transfers)."""
    p = set_zn(int(state.P), value) if update_flags else int(state.P)
    p |= FLAG_U  # bit 5 is hard-wired high on real hardware
    fields = {
        "P": jnp.uint8(p & 0xFF),
        "PC": jnp.uint16((int(state.PC) + INSTRUCTION_LENGTH[mode]) & 0xFFFF),
        "cycles": state.cycles + jnp.uint64(base_cycles + extra),
    }
    fields[reg] = jnp.uint8(value & 0xFF)
    return state._replace(**fields)


def _commit_store(
    state: CPUState,
    memory: Memory,
    addr: int,
    value: int,
    mode: int,
    base_cycles: int,
) -> Tuple[CPUState, Memory]:
    """Common path for STA / STX / STY. Stores never apply the page-cross penalty."""
    new_memory = memory.at[addr & 0xFFFF].set(jnp.uint8(value & 0xFF))
    new_state = state._replace(
        PC=jnp.uint16((int(state.PC) + INSTRUCTION_LENGTH[mode]) & 0xFFFF),
        cycles=state.cycles + jnp.uint64(base_cycles),
    )
    return new_state, new_memory


def step(state: CPUState, memory: Memory) -> Tuple[CPUState, Memory]:
    """Execute one 6502 instruction. Returns the new `(state, memory)`."""
    pc = int(state.PC) & 0xFFFF
    opcode = int(memory[pc])
    mode = int(ADDRESSING_MODE_TABLE[opcode])
    base_cycles = int(CYCLE_TABLE[opcode])

    mnemonic = P1A_OPCODES.get(opcode)
    if mnemonic is None:
        return _stub_advance(state, base_cycles), memory

    # --- Transfers (implied, no addressing) ---------------------------------
    if mnemonic == "TAX":
        return _commit_load(state, "X", int(state.A), mode, base_cycles, 0), memory
    if mnemonic == "TAY":
        return _commit_load(state, "Y", int(state.A), mode, base_cycles, 0), memory
    if mnemonic == "TXA":
        return _commit_load(state, "A", int(state.X), mode, base_cycles, 0), memory
    if mnemonic == "TYA":
        return _commit_load(state, "A", int(state.Y), mode, base_cycles, 0), memory
    if mnemonic == "TSX":
        return _commit_load(state, "X", int(state.SP), mode, base_cycles, 0), memory
    if mnemonic == "TXS":
        # TXS is the only transfer that does NOT touch flags.
        return _commit_load(
            state, "SP", int(state.X), mode, base_cycles, 0, update_flags=False
        ), memory

    # --- Loads --------------------------------------------------------------
    if mnemonic in ("LDA", "LDX", "LDY"):
        addr, page_crossed = RESOLVERS[mode](state, memory)
        value = int(memory[addr & 0xFFFF])
        reg = {"LDA": "A", "LDX": "X", "LDY": "Y"}[mnemonic]
        extra = 1 if page_crossed else 0
        return _commit_load(state, reg, value, mode, base_cycles, extra), memory

    # --- Stores -------------------------------------------------------------
    if mnemonic in ("STA", "STX", "STY"):
        addr, _ = RESOLVERS[mode](state, memory)
        value = {"STA": int(state.A), "STX": int(state.X), "STY": int(state.Y)}[mnemonic]
        return _commit_store(state, memory, addr, value, mode, base_cycles)

    # Defensive — should be unreachable given P1A_OPCODES keys.
    return _stub_advance(state, base_cycles), memory
