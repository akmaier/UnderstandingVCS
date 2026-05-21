"""6502 fetch–decode–execute step.

Implemented so far (PORTING_PLAN.md §5):
- P1a: load (LDA, LDX, LDY), store (STA, STX, STY), transfer
  (TAX, TAY, TXA, TYA, TSX, TXS).
- P1b1: bitwise (AND, ORA, EOR), compare (CMP, CPX, CPY), BIT.

Pending: P1b2 (ADC, SBC + decimal mode), P1c (ASL, LSR, ROL, ROR), P1d
(branches + JMP/JSR/RTS), P1e (stack + status), P1f (BRK/RTI/IRQ/NMI/RESET
and the cycle-counting fine print). Unknown opcodes fall through to the
stub: PC += 1, cycles += base.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp

from jaxtari.cpu.addressing import INSTRUCTION_LENGTH, RESOLVERS
from jaxtari.cpu.alu import bit_flags, compare_flags, set_zn
from jaxtari.cpu.tables import ADDRESSING_MODE_TABLE, CYCLE_TABLE, FLAG_U
from jaxtari.types import CPUState

Memory = jnp.ndarray  # shape (1 << 16,), dtype uint8


# Opcode → mnemonic. Anything not in this dict falls through to the stub.
OPCODES: dict[int, str] = {
    # --- P1a -----------------------------------------------------------------
    # LDA — 8 modes
    0xA9: "LDA", 0xA5: "LDA", 0xB5: "LDA", 0xAD: "LDA",
    0xBD: "LDA", 0xB9: "LDA", 0xA1: "LDA", 0xB1: "LDA",
    # LDX — 5 modes
    0xA2: "LDX", 0xA6: "LDX", 0xB6: "LDX", 0xAE: "LDX", 0xBE: "LDX",
    # LDY — 5 modes
    0xA0: "LDY", 0xA4: "LDY", 0xB4: "LDY", 0xAC: "LDY", 0xBC: "LDY",
    # STA — 7 modes
    0x85: "STA", 0x95: "STA", 0x8D: "STA",
    0x9D: "STA", 0x99: "STA", 0x81: "STA", 0x91: "STA",
    # STX / STY — 3 modes each
    0x86: "STX", 0x96: "STX", 0x8E: "STX",
    0x84: "STY", 0x94: "STY", 0x8C: "STY",
    # Transfers — implied
    0xAA: "TAX", 0xA8: "TAY", 0x8A: "TXA", 0x98: "TYA", 0xBA: "TSX", 0x9A: "TXS",

    # --- P1b1 ----------------------------------------------------------------
    # AND — 8 modes
    0x29: "AND", 0x25: "AND", 0x35: "AND", 0x2D: "AND",
    0x3D: "AND", 0x39: "AND", 0x21: "AND", 0x31: "AND",
    # ORA — 8 modes
    0x09: "ORA", 0x05: "ORA", 0x15: "ORA", 0x0D: "ORA",
    0x1D: "ORA", 0x19: "ORA", 0x01: "ORA", 0x11: "ORA",
    # EOR — 8 modes
    0x49: "EOR", 0x45: "EOR", 0x55: "EOR", 0x4D: "EOR",
    0x5D: "EOR", 0x59: "EOR", 0x41: "EOR", 0x51: "EOR",
    # CMP — 8 modes
    0xC9: "CMP", 0xC5: "CMP", 0xD5: "CMP", 0xCD: "CMP",
    0xDD: "CMP", 0xD9: "CMP", 0xC1: "CMP", 0xD1: "CMP",
    # CPX / CPY — 3 modes each
    0xE0: "CPX", 0xE4: "CPX", 0xEC: "CPX",
    0xC0: "CPY", 0xC4: "CPY", 0xCC: "CPY",
    # BIT — 2 modes (zp, abs)
    0x24: "BIT", 0x2C: "BIT",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _stub_advance(state: CPUState, base_cycles: int) -> CPUState:
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
    p = set_zn(int(state.P), value) if update_flags else (int(state.P) | FLAG_U)
    fields = {
        "P": jnp.uint8(p & 0xFF),
        "PC": jnp.uint16((int(state.PC) + INSTRUCTION_LENGTH[mode]) & 0xFFFF),
        "cycles": state.cycles + jnp.uint64(base_cycles + extra),
    }
    fields[reg] = jnp.uint8(value & 0xFF)
    return state._replace(**fields)


def _commit_store(
    state: CPUState, memory: Memory, addr: int, value: int, mode: int, base_cycles: int
) -> Tuple[CPUState, Memory]:
    """Common path for STA / STX / STY. Stores never apply the page-cross penalty."""
    new_memory = memory.at[addr & 0xFFFF].set(jnp.uint8(value & 0xFF))
    new_state = state._replace(
        PC=jnp.uint16((int(state.PC) + INSTRUCTION_LENGTH[mode]) & 0xFFFF),
        cycles=state.cycles + jnp.uint64(base_cycles),
    )
    return new_state, new_memory


def _commit_a_with_zn(
    state: CPUState, new_a: int, mode: int, base_cycles: int, extra: int
) -> CPUState:
    """Common path for AND / ORA / EOR — A = new_a; set ZN; advance PC; bump cycles."""
    return state._replace(
        A=jnp.uint8(new_a & 0xFF),
        P=jnp.uint8(set_zn(int(state.P), new_a) & 0xFF),
        PC=jnp.uint16((int(state.PC) + INSTRUCTION_LENGTH[mode]) & 0xFFFF),
        cycles=state.cycles + jnp.uint64(base_cycles + extra),
    )


def _commit_flags_only(
    state: CPUState, new_p: int, mode: int, base_cycles: int, extra: int
) -> CPUState:
    """Common path for CMP / CPX / CPY / BIT — only P/PC/cycles change."""
    return state._replace(
        P=jnp.uint8(new_p & 0xFF),
        PC=jnp.uint16((int(state.PC) + INSTRUCTION_LENGTH[mode]) & 0xFFFF),
        cycles=state.cycles + jnp.uint64(base_cycles + extra),
    )


# --------------------------------------------------------------------------- #
# step
# --------------------------------------------------------------------------- #

def step(state: CPUState, memory: Memory) -> Tuple[CPUState, Memory]:
    """Execute one 6502 instruction. Returns the new `(state, memory)`."""
    pc = int(state.PC) & 0xFFFF
    opcode = int(memory[pc])
    mode = int(ADDRESSING_MODE_TABLE[opcode])
    base_cycles = int(CYCLE_TABLE[opcode])

    mnemonic = OPCODES.get(opcode)
    if mnemonic is None:
        return _stub_advance(state, base_cycles), memory

    # --- Transfers (implied) -----------------------------------------------
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
        return _commit_load(
            state, "SP", int(state.X), mode, base_cycles, 0, update_flags=False
        ), memory

    # --- Loads -------------------------------------------------------------
    if mnemonic in ("LDA", "LDX", "LDY"):
        addr, page_crossed = RESOLVERS[mode](state, memory)
        value = int(memory[addr & 0xFFFF])
        reg = {"LDA": "A", "LDX": "X", "LDY": "Y"}[mnemonic]
        extra = 1 if page_crossed else 0
        return _commit_load(state, reg, value, mode, base_cycles, extra), memory

    # --- Stores ------------------------------------------------------------
    if mnemonic in ("STA", "STX", "STY"):
        addr, _ = RESOLVERS[mode](state, memory)
        value = {"STA": int(state.A), "STX": int(state.X), "STY": int(state.Y)}[mnemonic]
        return _commit_store(state, memory, addr, value, mode, base_cycles)

    # --- Bitwise A-ops (AND / ORA / EOR) -----------------------------------
    if mnemonic in ("AND", "ORA", "EOR"):
        addr, page_crossed = RESOLVERS[mode](state, memory)
        value = int(memory[addr & 0xFFFF])
        a = int(state.A)
        if mnemonic == "AND":
            new_a = a & value
        elif mnemonic == "ORA":
            new_a = a | value
        else:  # EOR
            new_a = a ^ value
        extra = 1 if page_crossed else 0
        return _commit_a_with_zn(state, new_a, mode, base_cycles, extra), memory

    # --- Compares (CMP / CPX / CPY) ----------------------------------------
    if mnemonic in ("CMP", "CPX", "CPY"):
        addr, page_crossed = RESOLVERS[mode](state, memory)
        value = int(memory[addr & 0xFFFF])
        reg_val = {
            "CMP": int(state.A),
            "CPX": int(state.X),
            "CPY": int(state.Y),
        }[mnemonic]
        new_p = compare_flags(int(state.P), reg_val, value)
        # Only CMP has indexed/indirect modes that can page-cross; CPX/CPY use
        # only immediate / zp / abs, so `page_crossed` is False for them.
        extra = 1 if page_crossed else 0
        return _commit_flags_only(state, new_p, mode, base_cycles, extra), memory

    # --- BIT --------------------------------------------------------------
    if mnemonic == "BIT":
        addr, _ = RESOLVERS[mode](state, memory)
        value = int(memory[addr & 0xFFFF])
        new_p = bit_flags(int(state.P), int(state.A), value)
        return _commit_flags_only(state, new_p, mode, base_cycles, 0), memory

    # Defensive — should be unreachable.
    return _stub_advance(state, base_cycles), memory
