"""6502 fetch–decode–execute step.

Implemented so far (PORTING_PLAN.md §5):
- P1a: load (LDA, LDX, LDY), store (STA, STX, STY), transfer
  (TAX, TAY, TXA, TYA, TSX, TXS).
- P1b1: bitwise (AND, ORA, EOR), compare (CMP, CPX, CPY), BIT.
- P1b2: arithmetic (ADC, SBC including BCD/decimal mode) + the undocumented
  USBC (0xEB) alias for SBC immediate.
- P1c: shifts/rotates (ASL, LSR, ROL, ROR) in both accumulator and memory
  (zp / zp,X / abs / abs,X) addressing modes.
- P1d: conditional branches (BPL/BMI/BVC/BVS/BCC/BCS/BNE/BEQ), JMP
  (absolute + indirect with page-wrap bug), JSR, RTS. Introduces the
  push8 / pop8 stack helpers reused by P1e and P1f.
- P1e: stack push/pull (PHA/PHP/PLA/PLP), status-flag setters/clearers
  (SEC/CLC/SEI/CLI/SED/CLD/CLV), and NOP. PHP pushes P with B set; PLP
  forces both B and U on pull, following xitari's PSLockupTable convention
  ("the 6507's B flag is always true" — see M6502.hxx line 318).
- P1f: INC/DEC memory, INX/INY/DEX/DEY register, BRK (push PC+2, push
  P|B|U, set I, jump via $FFFE/$FFFF IRQ vector), RTI (pop P, pop PC).
  This completes the documented NMOS 6502 opcode set.

External hardware interrupts (IRQ / NMI lines, RESET pin) are not part
of step() — they require a wire-level integration with the bus and are
deferred to P3 (TIA) and P6 (Console wiring). Unknown opcodes still fall
through to the stub: PC += 1, cycles += base.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp

from jaxtari.bus.system import Bus, peek, poke
from jaxtari.cpu.addressing import INSTRUCTION_LENGTH, RESOLVERS
from jaxtari.cpu.alu import (
    adc,
    asl_op,
    bit_flags,
    compare_flags,
    lsr_op,
    rol_op,
    ror_op,
    sbc,
    set_zn,
)
from jaxtari.cpu.tables import (
    ADDR_ABSOLUTE_X,
    ADDR_IMPLIED,
    ADDR_INDIRECT,
    ADDRESSING_MODE_TABLE,
    CYCLE_TABLE,
    FLAG_B,
    FLAG_C,
    FLAG_D,
    FLAG_I,
    FLAG_N,
    FLAG_U,
    FLAG_V,
    FLAG_Z,
)
from jaxtari.riot.system import riot_advance
from jaxtari.tia.system import tia_advance, tia_apply_wsync
from jaxtari.types import CPUState

# A "world" is whatever the CPU steps against — see jaxtari.bus.system.
# Either a Bus (proper 6507 emulation) or a flat 65,536-byte jnp.ndarray
# (used by the P1 unit tests). All memory access goes through peek/poke.
Memory = jnp.ndarray


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

    # --- P1b2 ----------------------------------------------------------------
    # ADC — 8 modes
    0x69: "ADC", 0x65: "ADC", 0x75: "ADC", 0x6D: "ADC",
    0x7D: "ADC", 0x79: "ADC", 0x61: "ADC", 0x71: "ADC",
    # SBC — 8 modes + 1 undocumented alias (0xEB = USBC)
    0xE9: "SBC", 0xE5: "SBC", 0xF5: "SBC", 0xED: "SBC",
    0xFD: "SBC", 0xF9: "SBC", 0xE1: "SBC", 0xF1: "SBC",
    0xEB: "SBC",

    # --- P1c -----------------------------------------------------------------
    # ASL — accumulator + 4 memory modes
    0x0A: "ASL", 0x06: "ASL", 0x16: "ASL", 0x0E: "ASL", 0x1E: "ASL",
    # LSR — accumulator + 4 memory modes
    0x4A: "LSR", 0x46: "LSR", 0x56: "LSR", 0x4E: "LSR", 0x5E: "LSR",
    # ROL — accumulator + 4 memory modes
    0x2A: "ROL", 0x26: "ROL", 0x36: "ROL", 0x2E: "ROL", 0x3E: "ROL",
    # ROR — accumulator + 4 memory modes
    0x6A: "ROR", 0x66: "ROR", 0x76: "ROR", 0x6E: "ROR", 0x7E: "ROR",

    # --- P1d -----------------------------------------------------------------
    # Conditional branches (relative, 2 bytes; +1 cycle if taken, +1 more if
    # the branch crosses a page boundary)
    0x10: "BPL", 0x30: "BMI", 0x50: "BVC", 0x70: "BVS",
    0x90: "BCC", 0xB0: "BCS", 0xD0: "BNE", 0xF0: "BEQ",
    # JMP (absolute / indirect with the documented page-wrap bug)
    0x4C: "JMP", 0x6C: "JMP",
    # Subroutine call / return
    0x20: "JSR",
    0x60: "RTS",

    # --- P1e -----------------------------------------------------------------
    # Stack push / pull (all implied; PHA/PHP=3, PLA/PLP=4 cycles)
    0x48: "PHA", 0x08: "PHP",
    0x68: "PLA", 0x28: "PLP",
    # Status-flag setters / clearers (all implied, 2 cycles)
    0x18: "CLC", 0x38: "SEC",
    0x58: "CLI", 0x78: "SEI",
    0xB8: "CLV",
    0xD8: "CLD", 0xF8: "SED",
    # NOP (the only documented one: $EA — implied, 2 cycles)
    0xEA: "NOP",

    # --- P1f -----------------------------------------------------------------
    # INC memory (4 modes)
    0xE6: "INC", 0xF6: "INC", 0xEE: "INC", 0xFE: "INC",
    # DEC memory (4 modes)
    0xC6: "DEC", 0xD6: "DEC", 0xCE: "DEC", 0xDE: "DEC",
    # Register inc/dec (all implied)
    0xE8: "INX", 0xC8: "INY",
    0xCA: "DEX", 0x88: "DEY",
    # Interrupts
    0x00: "BRK",
    0x40: "RTI",

    # --- P1h — common undocumented NMOS 6502 opcodes ------------------------
    #
    # NMOS 6502 has a wide undocumented opcode set; some carts (and many
    # 6502 test ROMs) rely on the well-behaved subset:
    #
    #   NOP variants  — extra implied / imm / zp / zp,X / abs / abs,X
    #                   NOPs that just consume cycles and (for the
    #                   addressing-mode variants) dummy-read the operand.
    #   LAX           — load A AND X from the same operand (LDA + TAX, no
    #                   flag side-effects beyond N/Z from the load).
    #   SAX           — store (A AND X) at the operand address; never
    #                   touches flags.
    #
    # The famous "magic AND" LAX #imm ($AB) is intentionally left out —
    # it's unstable on real hardware. Similarly the combo RMW opcodes
    # (DCP / ISC / RLA / RRA / SLO / SRE) are kept deferred; the NOPs
    # are by far the most common XAI-relevant subset.

    # Undocumented one-byte 2-cycle NOPs (implied).
    0x1A: "NOP", 0x3A: "NOP", 0x5A: "NOP", 0x7A: "NOP",
    0xDA: "NOP", 0xFA: "NOP",
    # Undocumented two-byte 2-cycle NOPs (immediate — operand is read but
    # discarded).
    0x80: "NOP", 0x82: "NOP", 0x89: "NOP", 0xC2: "NOP", 0xE2: "NOP",
    # Undocumented zero-page NOPs (read operand byte, discard).
    0x04: "NOP", 0x44: "NOP", 0x64: "NOP",
    # Undocumented zero-page,X NOPs.
    0x14: "NOP", 0x34: "NOP", 0x54: "NOP", 0x74: "NOP",
    0xD4: "NOP", 0xF4: "NOP",
    # Undocumented absolute NOP.
    0x0C: "NOP",
    # Undocumented absolute,X NOPs (+1 cycle if page-crossed, same as a
    # documented absolute,X read).
    0x1C: "NOP", 0x3C: "NOP", 0x5C: "NOP", 0x7C: "NOP",
    0xDC: "NOP", 0xFC: "NOP",

    # LAX — load A and X from the same operand. 6 modes.
    0xA7: "LAX", 0xB7: "LAX", 0xAF: "LAX", 0xBF: "LAX",
    0xA3: "LAX", 0xB3: "LAX",

    # SAX — store A AND X. 4 modes.
    0x87: "SAX", 0x97: "SAX", 0x8F: "SAX", 0x83: "SAX",
}


# (opcode → flag bit) for SEC/CLC/SEI/CLI/SED/CLD/CLV. The setter / clearer
# direction is implicit in the opcode: 0x38/0x78/0xF8 set; 0x18/0x58/0xB8/0xD8
# clear. _STATUS_FLAG[opcode] gives the bit to flip.
_STATUS_FLAG: dict[int, int] = {
    0x18: FLAG_C, 0x38: FLAG_C,
    0x58: FLAG_I, 0x78: FLAG_I,
    0xB8: FLAG_V,
    0xD8: FLAG_D, 0xF8: FLAG_D,
}
_STATUS_SET_OPCODES = frozenset({0x38, 0x78, 0xF8})


# Branch opcode → (flag bit, take_when_set). Used by the branch dispatcher.
_BRANCH_INFO: dict[int, tuple[int, bool]] = {
    0x10: (FLAG_N, False),  # BPL — branch if N=0
    0x30: (FLAG_N, True),   # BMI — branch if N=1
    0x50: (FLAG_V, False),  # BVC
    0x70: (FLAG_V, True),   # BVS
    0x90: (FLAG_C, False),  # BCC
    0xB0: (FLAG_C, True),   # BCS
    0xD0: (FLAG_Z, False),  # BNE
    0xF0: (FLAG_Z, True),   # BEQ
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _stub_advance(state: CPUState, base_cycles: int) -> CPUState:
    return state._replace(
        PC=jnp.uint16((int(state.PC) + 1) & 0xFFFF),
        cycles=state.cycles + jnp.uint64(base_cycles),
    )


# --------------------------------------------------------------------------- #
# Stack helpers — shared by JSR/RTS (P1d), PHA/PLA/PHP/PLP (P1e), and
# BRK/RTI/IRQ/NMI (P1f). Stack lives at $0100 + SP; SP decrements on push.
# --------------------------------------------------------------------------- #

def push8(memory, sp: int, value: int):
    new_memory = poke(memory, 0x0100 + (sp & 0xFF), value)
    new_sp = (sp - 1) & 0xFF
    return new_memory, new_sp


def pop8(memory, sp: int):
    """Pull a byte from the stack. Returns `(value, new_sp, new_memory)`
    — the new_memory tuple member carries any side-effect bus update
    forward (P4d's read-clears-flag semantic; a stack pop never
    actually hits a RIOT register but the API stays uniform)."""
    new_sp = (sp + 1) & 0xFF
    value, new_memory = peek(memory, 0x0100 + new_sp)
    return value, new_sp, new_memory


def push16(memory: Memory, sp: int, value: int) -> Tuple[Memory, int]:
    """Push high byte first, then low byte (so RTS pops low then high)."""
    memory, sp = push8(memory, sp, (value >> 8) & 0xFF)
    memory, sp = push8(memory, sp, value & 0xFF)
    return memory, sp


def pop16(memory: Memory, sp: int):
    """Pull a 16-bit word. Returns `(value, new_sp, new_memory)`."""
    low,  sp, memory = pop8(memory, sp)
    high, sp, memory = pop8(memory, sp)
    return (high << 8) | low, sp, memory


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
    state: CPUState, memory, addr: int, value: int, mode: int, base_cycles: int
):
    """Common path for STA / STX / STY. Stores never apply the page-cross penalty."""
    new_memory = poke(memory, addr, value)
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

def step(state: CPUState, memory):
    """Execute one 6502 instruction. Returns the new `(state, memory)`.

    `memory` may be either a `jaxtari.bus.Bus` (proper 6507 bus) or a flat
    65,536-byte `jnp.ndarray` (P1-style scratch memory). All accesses go
    through `jaxtari.bus.peek` / `poke` which dispatch on the type.

    When `memory` is a `Bus`, the TIA's scanline / frame counters are
    advanced by the cycles this instruction consumed, and any WSYNC stall
    queued by a write to \$02 is resolved before returning.
    """
    new_state, new_memory = _step_inner(state, memory)
    if isinstance(new_memory, Bus):
        new_state, new_memory = _tia_post_step(state, new_state, new_memory)
    return new_state, new_memory


def _tia_post_step(old_state: CPUState, new_state: CPUState, bus: Bus):
    """Advance the TIA and RIOT by the cycles consumed and resolve a
    pending WSYNC. WSYNC's stall cycles also feed the RIOT timer (the
    RIOT continues to tick while the CPU is held)."""
    delta = int(new_state.cycles - old_state.cycles)
    new_tia = tia_advance(bus.tia, delta)
    new_riot = riot_advance(bus.riot, delta)
    stall, new_tia = tia_apply_wsync(new_tia)
    if stall:
        new_state = new_state._replace(
            cycles=new_state.cycles + jnp.uint64(stall),
        )
        new_riot = riot_advance(new_riot, stall)
    return new_state, bus._replace(tia=new_tia, riot=new_riot)


def _step_inner(state: CPUState, memory):
    """Inner dispatch — runs one instruction without any TIA post-processing.

    **P4d**: every `peek(memory, ...)` now returns `(value, new_memory)`
    — the new_memory carries any read-side-effect bus update forward
    (e.g. INTIM-read-clears-timer-expired-latch). Every resolver also
    returns `(addr, page_crossed, new_memory)` for the same reason.
    The previous return-only-value API was the architectural blocker
    for P4d; threading the new memory through every read site is the
    cost of getting bit-exact PXC1-x semantics.
    """
    pc = int(state.PC) & 0xFFFF
    opcode, memory = peek(memory, pc)
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
        addr, page_crossed, memory = RESOLVERS[mode](state, memory)
        value, memory = peek(memory, addr)
        reg = {"LDA": "A", "LDX": "X", "LDY": "Y"}[mnemonic]
        extra = 1 if page_crossed else 0
        return _commit_load(state, reg, value, mode, base_cycles, extra), memory

    # --- Stores ------------------------------------------------------------
    if mnemonic in ("STA", "STX", "STY"):
        addr, _, memory = RESOLVERS[mode](state, memory)
        value = {"STA": int(state.A), "STX": int(state.X), "STY": int(state.Y)}[mnemonic]
        return _commit_store(state, memory, addr, value, mode, base_cycles)

    # --- Bitwise A-ops (AND / ORA / EOR) -----------------------------------
    if mnemonic in ("AND", "ORA", "EOR"):
        addr, page_crossed, memory = RESOLVERS[mode](state, memory)
        value, memory = peek(memory, addr)
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
        addr, page_crossed, memory = RESOLVERS[mode](state, memory)
        value, memory = peek(memory, addr)
        reg_val = {
            "CMP": int(state.A),
            "CPX": int(state.X),
            "CPY": int(state.Y),
        }[mnemonic]
        new_p = compare_flags(int(state.P), reg_val, value)
        extra = 1 if page_crossed else 0
        return _commit_flags_only(state, new_p, mode, base_cycles, extra), memory

    # --- BIT --------------------------------------------------------------
    if mnemonic == "BIT":
        addr, _, memory = RESOLVERS[mode](state, memory)
        value, memory = peek(memory, addr)
        new_p = bit_flags(int(state.P), int(state.A), value)
        return _commit_flags_only(state, new_p, mode, base_cycles, 0), memory

    # --- ADC / SBC --------------------------------------------------------
    if mnemonic in ("ADC", "SBC"):
        addr, page_crossed, memory = RESOLVERS[mode](state, memory)
        value, memory = peek(memory, addr)
        new_a, new_p = (adc if mnemonic == "ADC" else sbc)(
            int(state.P), int(state.A), value
        )
        extra = 1 if page_crossed else 0
        return state._replace(
            A=jnp.uint8(new_a & 0xFF),
            P=jnp.uint8(new_p & 0xFF),
            PC=jnp.uint16((int(state.PC) + INSTRUCTION_LENGTH[mode]) & 0xFFFF),
            cycles=state.cycles + jnp.uint64(base_cycles + extra),
        ), memory

    # --- Shifts / rotates -------------------------------------------------
    if mnemonic in ("ASL", "LSR", "ROL", "ROR"):
        op = {"ASL": asl_op, "LSR": lsr_op, "ROL": rol_op, "ROR": ror_op}[mnemonic]
        if mode == ADDR_IMPLIED:
            # Accumulator-mode shift / rotate: operand and destination are A.
            new_a, new_p = op(int(state.P), int(state.A))
            return state._replace(
                A=jnp.uint8(new_a & 0xFF),
                P=jnp.uint8(new_p & 0xFF),
                PC=jnp.uint16((int(state.PC) + 1) & 0xFFFF),
                cycles=state.cycles + jnp.uint64(base_cycles),
            ), memory
        # Memory-mode RMW. Task #50: NMOS 6502 emits the famous
        # double-write — the cell is written *twice*: first with the
        # OLD value (dummy/internal cycle), then with the NEW value.
        # The dummy write is visible on the data bus, so it updates
        # the floating-bus latch that subsequent TIA reads OR into
        # their result. (Mirrors xitari's M6502 RMW emulation +
        # System::poke updating myDataBusState on every write.)
        addr, _, memory = RESOLVERS[mode](state, memory)
        value, memory = peek(memory, addr)
        memory = poke(memory, addr, value)              # RMW dummy write
        new_value, new_p = op(int(state.P), value)
        memory = poke(memory, addr, new_value)
        return state._replace(
            P=jnp.uint8(new_p & 0xFF),
            PC=jnp.uint16((int(state.PC) + INSTRUCTION_LENGTH[mode]) & 0xFFFF),
            cycles=state.cycles + jnp.uint64(base_cycles),
        ), memory

    # --- Conditional branches --------------------------------------------
    if mnemonic in ("BPL", "BMI", "BVC", "BVS", "BCC", "BCS", "BNE", "BEQ"):
        flag, take_when_set = _BRANCH_INFO[opcode]
        flag_set = (int(state.P) & flag) != 0
        take = flag_set if take_when_set else not flag_set
        if take:
            target, page_crossed, memory = RESOLVERS[mode](state, memory)
            # Task #50: real NMOS 6502 branch-taken has a "wasted
            # opcode prefetch" — at cycle 3 it peeks PC+2 (the
            # would-be next opcode before the branch redirects). If
            # the branch crosses a page, cycle 4 then peeks at the
            # wrong page's high byte before the actual fetch
            # corrects. Both are dummy peeks that update the
            # floating-bus latch.
            _, memory = peek(memory, (int(state.PC) + 2) & 0xFFFF)
            if page_crossed:
                # Wrong-page dummy peek: the target's low byte with
                # the PC+2 high byte (the un-corrected high byte).
                wrong_addr = ((int(state.PC) + 2) & 0xFF00) | (target & 0xFF)
                _, memory = peek(memory, wrong_addr)
            extra = 1 + (1 if page_crossed else 0)
            new_pc = target
        else:
            new_pc = (int(state.PC) + 2) & 0xFFFF
            extra = 0
        return state._replace(
            PC=jnp.uint16(new_pc),
            cycles=state.cycles + jnp.uint64(base_cycles + extra),
        ), memory

    # --- JMP (absolute and indirect) -------------------------------------
    if mnemonic == "JMP":
        target, _, memory = RESOLVERS[mode](state, memory)
        return state._replace(
            PC=jnp.uint16(target),
            cycles=state.cycles + jnp.uint64(base_cycles),
        ), memory

    # --- JSR ---------------------------------------------------------------
    if mnemonic == "JSR":
        # Task #50: real NMOS 6502 JSR has a "pre-push internal cycle"
        # at cycle 3 that reads from $0100+SP (the byte that's about
        # to be overwritten by the PCH push) and discards the result.
        # This pre-push read is visible on the data bus, so it updates
        # the floating-bus latch the TIA's INPT/collision reads OR
        # into D5-D0. Mirrors xitari's M6502 JSR implementation.
        target, _, memory = RESOLVERS[mode](state, memory)
        _, memory = peek(memory, 0x0100 + int(state.SP))   # pre-push discard
        return_addr = (int(state.PC) + 2) & 0xFFFF
        memory, new_sp = push16(memory, int(state.SP), return_addr)
        return state._replace(
            SP=jnp.uint8(new_sp),
            PC=jnp.uint16(target),
            cycles=state.cycles + jnp.uint64(base_cycles),
        ), memory

    # --- RTS ---------------------------------------------------------------
    if mnemonic == "RTS":
        return_addr, new_sp, memory = pop16(memory, int(state.SP))
        return state._replace(
            SP=jnp.uint8(new_sp),
            PC=jnp.uint16((return_addr + 1) & 0xFFFF),
            cycles=state.cycles + jnp.uint64(base_cycles),
        ), memory

    # --- Stack push / pull (P1e) ------------------------------------------
    if mnemonic == "PHA":
        memory, new_sp = push8(memory, int(state.SP), int(state.A))
        return state._replace(
            SP=jnp.uint8(new_sp),
            PC=jnp.uint16((int(state.PC) + 1) & 0xFFFF),
            cycles=state.cycles + jnp.uint64(base_cycles),
        ), memory
    if mnemonic == "PHP":
        memory, new_sp = push8(
            memory, int(state.SP), int(state.P) | FLAG_B | FLAG_U
        )
        return state._replace(
            SP=jnp.uint8(new_sp),
            PC=jnp.uint16((int(state.PC) + 1) & 0xFFFF),
            cycles=state.cycles + jnp.uint64(base_cycles),
        ), memory
    if mnemonic == "PLA":
        value, new_sp, memory = pop8(memory, int(state.SP))
        return state._replace(
            A=jnp.uint8(value & 0xFF),
            SP=jnp.uint8(new_sp),
            P=jnp.uint8(set_zn(int(state.P), value) & 0xFF),
            PC=jnp.uint16((int(state.PC) + 1) & 0xFFFF),
            cycles=state.cycles + jnp.uint64(base_cycles),
        ), memory
    if mnemonic == "PLP":
        popped, new_sp, memory = pop8(memory, int(state.SP))
        return state._replace(
            SP=jnp.uint8(new_sp),
            P=jnp.uint8((popped | FLAG_U | FLAG_B) & 0xFF),
            PC=jnp.uint16((int(state.PC) + 1) & 0xFFFF),
            cycles=state.cycles + jnp.uint64(base_cycles),
        ), memory

    # --- Status-flag setters / clearers + NOP -----------------------------
    if mnemonic in ("CLC", "SEC", "CLI", "SEI", "CLV", "CLD", "SED"):
        flag = _STATUS_FLAG[opcode]
        p = int(state.P)
        p = (p | flag) if opcode in _STATUS_SET_OPCODES else (p & ~flag)
        return state._replace(
            P=jnp.uint8((p | FLAG_U) & 0xFF),
            PC=jnp.uint16((int(state.PC) + 1) & 0xFFFF),
            cycles=state.cycles + jnp.uint64(base_cycles),
        ), memory
    if mnemonic == "NOP":
        extra = 0
        if mode == ADDR_ABSOLUTE_X:
            _, page_crossed, memory = RESOLVERS[mode](state, memory)
            extra = 1 if page_crossed else 0
        return state._replace(
            PC=jnp.uint16((int(state.PC) + INSTRUCTION_LENGTH[mode]) & 0xFFFF),
            cycles=state.cycles + jnp.uint64(base_cycles + extra),
        ), memory

    # --- LAX (P1h) — load A and X from the same operand. N/Z from value. ---
    if mnemonic == "LAX":
        addr, page_crossed, memory = RESOLVERS[mode](state, memory)
        value, memory = peek(memory, addr)
        extra = 1 if page_crossed else 0
        p = set_zn(int(state.P), value)
        return state._replace(
            A=jnp.uint8(value & 0xFF),
            X=jnp.uint8(value & 0xFF),
            P=jnp.uint8(p & 0xFF),
            PC=jnp.uint16((int(state.PC) + INSTRUCTION_LENGTH[mode]) & 0xFFFF),
            cycles=state.cycles + jnp.uint64(base_cycles + extra),
        ), memory

    # --- SAX (P1h) — store A AND X, no flag side-effects. ------------------
    if mnemonic == "SAX":
        addr, _, memory = RESOLVERS[mode](state, memory)
        memory = poke(memory, addr, int(state.A) & int(state.X) & 0xFF)
        return state._replace(
            PC=jnp.uint16((int(state.PC) + INSTRUCTION_LENGTH[mode]) & 0xFFFF),
            cycles=state.cycles + jnp.uint64(base_cycles),
        ), memory

    # --- INC / DEC memory (P1f) -------------------------------------------
    if mnemonic in ("INC", "DEC"):
        # Task #50: NMOS 6502 RMW double-write — INC/DEC write the
        # OLD value back as a dummy/internal cycle before writing the
        # NEW value. Same data-bus side effect as the shifts/rotates
        # block above.
        addr, _, memory = RESOLVERS[mode](state, memory)
        value, memory = peek(memory, addr)
        memory = poke(memory, addr, value)              # RMW dummy write
        delta = 1 if mnemonic == "INC" else -1
        new_value = (value + delta) & 0xFF
        new_p = set_zn(int(state.P), new_value)
        memory = poke(memory, addr, new_value)
        return state._replace(
            P=jnp.uint8(new_p & 0xFF),
            PC=jnp.uint16((int(state.PC) + INSTRUCTION_LENGTH[mode]) & 0xFFFF),
            cycles=state.cycles + jnp.uint64(base_cycles),
        ), memory

    # --- Register inc/dec (P1f) -------------------------------------------
    if mnemonic in ("INX", "INY", "DEX", "DEY"):
        reg = mnemonic[-1]                       # 'X' or 'Y'
        cur = int(state.X if reg == "X" else state.Y)
        delta = 1 if mnemonic[0] == "I" else -1
        new_val = (cur + delta) & 0xFF
        p = set_zn(int(state.P), new_val)
        fields = {
            "P": jnp.uint8(p & 0xFF),
            "PC": jnp.uint16((int(state.PC) + 1) & 0xFFFF),
            "cycles": state.cycles + jnp.uint64(base_cycles),
        }
        fields[reg] = jnp.uint8(new_val)
        return state._replace(**fields), memory

    # --- BRK (P1f) --------------------------------------------------------
    if mnemonic == "BRK":
        return_addr = (int(state.PC) + 2) & 0xFFFF
        memory, sp = push16(memory, int(state.SP), return_addr)
        memory, sp = push8(memory, sp, int(state.P) | FLAG_B | FLAG_U)
        new_p = (int(state.P) | FLAG_I | FLAG_U) & 0xFF
        irq_lo, memory = peek(memory, 0xFFFE)
        irq_hi, memory = peek(memory, 0xFFFF)
        new_pc = irq_lo | (irq_hi << 8)
        return state._replace(
            SP=jnp.uint8(sp),
            P=jnp.uint8(new_p),
            PC=jnp.uint16(new_pc),
            cycles=state.cycles + jnp.uint64(base_cycles),
        ), memory

    # --- RTI (P1f) --------------------------------------------------------
    if mnemonic == "RTI":
        popped_p, sp, memory = pop8(memory, int(state.SP))
        popped_pc, sp, memory = pop16(memory, sp)
        return state._replace(
            SP=jnp.uint8(sp),
            P=jnp.uint8((popped_p | FLAG_U | FLAG_B) & 0xFF),
            PC=jnp.uint16(popped_pc),
            cycles=state.cycles + jnp.uint64(base_cycles),
        ), memory

    # Defensive — should be unreachable.
    return _stub_advance(state, base_cycles), memory
