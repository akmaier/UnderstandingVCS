"""PXC4 — Klaus Dormann 6502 functional-test ROM regression.

Klaus Dormann's `6502_functional_test.bin` is the canonical
correctness test for a 6502 / 65C02 emulator. It exercises every
documented opcode in every addressing mode, with edge-cases for
flag handling, BCD, branches, page-crossing, and the
JMP-indirect-bug, then JMPs to a fixed "success" address when all
sub-tests pass. A correct emulator runs to that address; a
buggy one infinite-loops on the failing sub-test.

The test ROM is **not shipped** with the project (it's an external
Public-Domain artifact from
<https://github.com/Klaus2m5/6502_65C02_functional_tests>). To
enable PXC4 locally:

    git clone https://github.com/Klaus2m5/6502_65C02_functional_tests.git \\
        /tmp/klaus
    cp /tmp/klaus/bin_files/6502_functional_test.bin \\
        tools/fixtures/cpu/

This test skips gracefully when the ROM isn't present, so
contributors who don't care about CPU regression detail can
ignore it.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from jaxtari.bus.system import Bus, peek
from jaxtari.cpu.m6502 import _step_inner
from jaxtari.types import CPUState


REPO_ROOT = Path(__file__).resolve().parents[2]
KLAUS_ROM = REPO_ROOT / "tools" / "fixtures" / "cpu" / "6502_functional_test.bin"

# Klaus's load-and-start convention: the binary is a flat 64 KB image
# meant to be loaded at $0000. PC starts at $0400. The "success" trap
# in the standard build is a `JMP *` (infinite loop to the same
# address) at $3469 — when the test completes successfully it jumps
# to that PC and loops forever. (The build's `success` label is at
# `report_success`, which is at $3469 in the standard `bin_files/
# 6502_functional_test.bin`.)
KLAUS_START_PC     = 0x0400
KLAUS_SUCCESS_PC   = 0x3469
KLAUS_MAX_CYCLES   = 100_000_000          # plenty for the full test
KLAUS_TIMEOUT_TRAP = 100_000              # detect-stuck threshold


def _state(*, PC=KLAUS_START_PC, A=0, X=0, Y=0, SP=0xFF, P=0x34):
    return CPUState(
        A=jnp.uint8(A), X=jnp.uint8(X), Y=jnp.uint8(Y),
        SP=jnp.uint8(SP), P=jnp.uint8(P),
        PC=jnp.uint16(PC), cycles=jnp.uint64(0),
    )


def _flat_memory_from_rom(rom_bytes: bytes) -> jnp.ndarray:
    """Klaus's ROM is a 64K flat image loaded at $0000. We model it as
    a plain `jnp.ndarray[uint8]` of size 65536 — the CPU step's
    `peek` / `poke` accept either a Bus or a flat array, so this
    bypasses the cart / TIA / RIOT dispatch entirely (the test is
    CPU-only)."""
    mem = jnp.zeros((1 << 16,), dtype=jnp.uint8)
    for i, b in enumerate(rom_bytes):
        if i >= 1 << 16:
            break
        mem = mem.at[i].set(jnp.uint8(b))
    return mem


@pytest.mark.skipif(not KLAUS_ROM.is_file(),
                    reason=f"Klaus Dormann ROM not present at {KLAUS_ROM}. "
                           f"See `tests/test_pxc4_klaus_dormann.py` for "
                           f"download instructions.")
def test_klaus_dormann_functional_test_passes():
    """Run Klaus's full functional test against jaxtari's HARD CPU.

    Pass condition: PC reaches `KLAUS_SUCCESS_PC` (where the test
    binary loops on itself). Failure modes: PC stuck on a different
    JMP-self loop (some sub-test failed) — we detect that with a
    "no-progress for KLAUS_TIMEOUT_TRAP steps" guard. The cycle
    budget is generous; a passing run takes ~30 million cycles in
    practice.
    """
    rom_bytes = KLAUS_ROM.read_bytes()
    memory = _flat_memory_from_rom(rom_bytes)
    state = _state()

    last_pc = -1
    same_pc_count = 0
    step_count = 0
    max_steps = KLAUS_MAX_CYCLES // 2          # rough upper bound
    while step_count < max_steps:
        state, memory = _step_inner(state, memory)
        step_count += 1
        cur_pc = int(state.PC)
        if cur_pc == KLAUS_SUCCESS_PC:
            return                              # test passed
        if cur_pc == last_pc:
            same_pc_count += 1
            if same_pc_count >= KLAUS_TIMEOUT_TRAP:
                pytest.fail(
                    f"Klaus test stuck at PC=${cur_pc:04x} after "
                    f"{step_count} steps — sub-test failed. Run a "
                    f"disassembler on the ROM at that address to see "
                    f"which check is failing.")
        else:
            same_pc_count = 0
            last_pc = cur_pc

    pytest.fail(
        f"Klaus test did not reach success PC=${KLAUS_SUCCESS_PC:04x} "
        f"after {step_count} steps (last PC=${last_pc:04x}). "
        f"Increase KLAUS_MAX_CYCLES if you expect a slow path.")


def test_klaus_dormann_rom_path_documented():
    """Trivial test that always runs — exists to remind contributors
    that PXC4 is a wiring point, even when the ROM is missing."""
    assert "KLAUS_ROM" in globals()
    assert isinstance(KLAUS_ROM, Path)
