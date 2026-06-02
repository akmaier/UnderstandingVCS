"""Phase 2b jaxtari mirror — audit pending_tia_cycles vs CYCLE_TABLE for
every opcode in OPCODES. Same approach as jutari/test/scratch_cycle_audit.jl.

Run from `jaxtari/` with the venv active:
    .venv/bin/python tests/scratch_cycle_audit.py

Output: one line per opcode whose count diverges. Total at the end.
"""
from __future__ import annotations

import sys

import numpy as np
import jax.numpy as jnp

from jaxtari.cpu.m6502 import OPCODES, _step_inner
from jaxtari.cpu.tables import (
    CYCLE_TABLE, ADDRESSING_MODE_TABLE,
    ADDR_RELATIVE,
)
from jaxtari.bus.system import initial_bus


_BRANCH_OPS = {
    0x10: ("N", False), 0x30: ("N", True),    # BPL / BMI
    0x50: ("V", False), 0x70: ("V", True),    # BVC / BVS
    0x90: ("C", False), 0xB0: ("C", True),    # BCC / BCS
    0xD0: ("Z", False), 0xF0: ("Z", True),    # BNE / BEQ
}


def _make_bus(opcode: int):
    rom = np.zeros(4096, dtype=np.uint8)
    rom[0] = opcode                            # $F000
    rom[0xFFFC - 0xF000] = 0x00                # reset vector lo
    rom[0xFFFD - 0xF000] = 0xF0                # reset vector hi
    return initial_bus(jnp.asarray(rom))


def _make_cpu(opcode: int):
    """Return a CPUState with PC=$F000. Choose P so conditional branches
    do NOT take (branches taken add +1 cycle to base CYCLE_TABLE)."""
    from jaxtari.types import CPUState
    if opcode in _BRANCH_OPS:
        flag, take_when_set = _BRANCH_OPS[opcode]
        # If branch takes when flag SET — clear all flags (P=0x24).
        # Otherwise — set all flags (P=0xE7).
        P = 0x24 if take_when_set else 0xE7
    else:
        P = 0x24
    return CPUState(
        A=jnp.uint8(0), X=jnp.uint8(0), Y=jnp.uint8(0),
        SP=jnp.uint8(0xFD), PC=jnp.uint16(0xF000), P=jnp.uint8(P),
        cycles=jnp.uint64(0),
    )


def audit():
    mismatches = []
    for opcode, mnemonic in OPCODES.items():
        bus = _make_bus(opcode)
        s = _make_cpu(opcode)
        try:
            new_s, new_bus = _step_inner(s, bus)
        except Exception as e:
            mismatches.append((opcode, mnemonic,
                               int(CYCLE_TABLE[opcode]), -1,
                               int(ADDRESSING_MODE_TABLE[opcode]),
                               f"{type(e).__name__}: {e}"))
            continue
        consumed = int(new_bus.pending_tia_cycles)
        expected = int(CYCLE_TABLE[opcode])
        if consumed != expected:
            mismatches.append((opcode, mnemonic, expected, consumed,
                               int(ADDRESSING_MODE_TABLE[opcode]), ""))
    return mismatches


mm = audit()
if not mm:
    print(f"ALL OK: pending_tia_cycles == CYCLE_TABLE for all {len(OPCODES)} opcodes.")
else:
    print(f"MISMATCHES: {len(mm)} of {len(OPCODES)} opcodes diverge.\n")
    print("opcode  mnemonic  expected  actual  mode  notes")
    print("------  --------  --------  ------  ----  -----")
    for op, mn, exp, act, mode, note in sorted(mm):
        actstr = "EX" if act == -1 else str(act)
        print(f"  0x{op:02X}  {mn:<7s}  {exp:<7d}   {actstr:<6s}  {mode:<4d}  {note}")
