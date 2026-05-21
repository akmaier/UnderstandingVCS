"""P7c-d tests — SOFT-mode branches, JMP indirect, JSR / RTS.

Conditional branches use a HARD predicate (forward PC is exact). The
tests check that each branch takes / doesn't take based on the right
flag, that the signed displacement resolves correctly (forward and
backward), and that JSR / RTS round-trip the return address through the
SOFT stack.
"""

import jax.numpy as jnp

from jaxtari.diff import (
    SOFT_SUPPORTED_OPCODES,
    initial_soft_bus,
    initial_soft_cpu_state,
    soft_run,
    soft_step,
)


def _rom_with(bytes_: list[int], size: int = 4096) -> jnp.ndarray:
    rom = jnp.zeros((size,), dtype=jnp.float32)
    for i, b in enumerate(bytes_):
        rom = rom.at[i].set(jnp.float32(b))
    return rom


def test_p7c_d_opcodes_all_present():
    p7c_d = {0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0, 0x6C, 0x20, 0x60}
    assert p7c_d.issubset(SOFT_SUPPORTED_OPCODES)
    assert len(p7c_d) == 11


# --------------------------------------------------------------------------- #
# Conditional branches — taken / not-taken
# --------------------------------------------------------------------------- #

def test_bne_taken_when_z_clear():
    # BNE +4 with Z=0 → PC = F000 + 2 + 4 = F006
    bus = initial_soft_bus(_rom_with([0xD0, 0x04]))
    state = initial_soft_cpu_state()._replace(P=jnp.float32(0x34))   # Z=0
    state, _ = soft_step(state, bus)
    assert float(state.PC) == 0xF006
    assert float(state.cycles) == 3.0   # 2 + 1 taken


def test_bne_not_taken_when_z_set():
    bus = initial_soft_bus(_rom_with([0xD0, 0x04]))
    state = initial_soft_cpu_state()._replace(P=jnp.float32(0x36))   # Z=1
    state, _ = soft_step(state, bus)
    assert float(state.PC) == 0xF002
    assert float(state.cycles) == 2.0


def test_beq_taken_when_z_set():
    bus = initial_soft_bus(_rom_with([0xF0, 0x10]))
    state = initial_soft_cpu_state()._replace(P=jnp.float32(0x36))   # Z=1
    state, _ = soft_step(state, bus)
    assert float(state.PC) == 0xF012


def test_bcc_taken_when_carry_clear():
    bus = initial_soft_bus(_rom_with([0x90, 0x08]))
    state = initial_soft_cpu_state()._replace(P=jnp.float32(0x34))   # C=0
    state, _ = soft_step(state, bus)
    assert float(state.PC) == 0xF00A


def test_bcs_taken_when_carry_set():
    bus = initial_soft_bus(_rom_with([0xB0, 0x08]))
    state = initial_soft_cpu_state()._replace(P=jnp.float32(0x35))   # C=1
    state, _ = soft_step(state, bus)
    assert float(state.PC) == 0xF00A


def test_bmi_taken_when_negative():
    bus = initial_soft_bus(_rom_with([0x30, 0x02]))
    state = initial_soft_cpu_state()._replace(P=jnp.float32(0xB4))   # N=1
    state, _ = soft_step(state, bus)
    assert float(state.PC) == 0xF004


def test_bpl_taken_when_positive():
    bus = initial_soft_bus(_rom_with([0x10, 0x02]))
    state = initial_soft_cpu_state()._replace(P=jnp.float32(0x34))   # N=0
    state, _ = soft_step(state, bus)
    assert float(state.PC) == 0xF004


def test_bvc_taken_when_overflow_clear():
    bus = initial_soft_bus(_rom_with([0x50, 0x02]))
    state = initial_soft_cpu_state()._replace(P=jnp.float32(0x34))   # V=0
    state, _ = soft_step(state, bus)
    assert float(state.PC) == 0xF004


def test_bvs_taken_when_overflow_set():
    bus = initial_soft_bus(_rom_with([0x70, 0x02]))
    state = initial_soft_cpu_state()._replace(P=jnp.float32(0x74))   # V=1
    state, _ = soft_step(state, bus)
    assert float(state.PC) == 0xF004


def test_branch_backward_displacement():
    """A negative displacement subtracts. BNE -2 → infinite loop in
    place (PC back to the BNE)."""
    # 0xFE as a signed byte = -2. PC = F000 + 2 + (-2) = F000.
    bus = initial_soft_bus(_rom_with([0xD0, 0xFE]))
    state = initial_soft_cpu_state()._replace(P=jnp.float32(0x34))   # Z=0
    state, _ = soft_step(state, bus)
    assert float(state.PC) == 0xF000


# --------------------------------------------------------------------------- #
# JMP indirect
# --------------------------------------------------------------------------- #

def test_jmp_indirect_follows_pointer():
    """JMP ($0030) — pointer in RAM[$30..$31] holds the destination."""
    bus = initial_soft_bus(_rom_with([0x6C, 0x30, 0x00]))
    bus = bus._replace(ram=bus.ram
                       .at[0x30].set(jnp.float32(0x34))
                       .at[0x31].set(jnp.float32(0x12)))
    state, _ = soft_step(initial_soft_cpu_state(), bus)
    assert float(state.PC) == 0x1234


# --------------------------------------------------------------------------- #
# JSR / RTS round-trip
# --------------------------------------------------------------------------- #

def test_jsr_sets_pc_and_decrements_sp():
    # JSR $F010
    bus = initial_soft_bus(_rom_with([0x20, 0x10, 0xF0]))
    state, _ = soft_step(initial_soft_cpu_state(), bus)
    assert float(state.PC) == 0xF010
    # SP started at 0xFD; two pushes → 0xFB.
    assert float(state.SP) == 0xFB


def test_jsr_then_rts_returns_to_instruction_after_jsr():
    """JSR $F010 ... at $F010 put RTS. After RTS, PC should be the byte
    right after the 3-byte JSR — i.e. $F003."""
    rom = _rom_with([0x20, 0x10, 0xF0])     # JSR $F010 at $F000
    rom = rom.at[0x010].set(jnp.float32(0x60))   # RTS at $F010
    bus = initial_soft_bus(rom)
    state = initial_soft_cpu_state()
    state, bus = soft_step(state, bus)      # JSR
    assert float(state.PC) == 0xF010
    state, bus = soft_step(state, bus)      # RTS
    assert float(state.PC) == 0xF003
    # SP back to its starting value.
    assert float(state.SP) == 0xFD


def test_jsr_pushes_return_address_bytes_to_stack():
    bus = initial_soft_bus(_rom_with([0x20, 0x10, 0xF0]))
    state, bus = soft_step(initial_soft_cpu_state(), bus)
    # Return address pushed is PC+2 = $F002. High $F0 at $01FD, low $02 at $01FC.
    # In the SOFT bus model $01FD → RAM[$7D], $01FC → RAM[$7C].
    assert float(bus.ram[0x7D]) == 0xF0
    assert float(bus.ram[0x7C]) == 0x02
