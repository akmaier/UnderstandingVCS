"""P7c-c tests — SOFT-mode shifts and rotates (ASL/LSR/ROL/ROR).

Each opcode tests:
- Forward result (the byte transformation)
- Carry flag from the shifted-out bit
- N/Z flags from the result
- Memory modes confirm RMW behaviour — the byte at the target address
  gets updated in place
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


def test_p7c_c_opcodes_all_present():
    p7c_c = {
        0x0A, 0x06, 0x16, 0x0E, 0x1E,   # ASL
        0x4A, 0x46, 0x56, 0x4E, 0x5E,   # LSR
        0x2A, 0x26, 0x36, 0x2E, 0x3E,   # ROL
        0x6A, 0x66, 0x76, 0x6E, 0x7E,   # ROR
    }
    assert p7c_c.issubset(SOFT_SUPPORTED_OPCODES)
    assert len(p7c_c) == 20


# --------------------------------------------------------------------------- #
# ASL — bit 7 → C; left-shift
# --------------------------------------------------------------------------- #

def test_asl_accumulator_basic():
    # A = 0x40 → 0x80; C=0 (bit 7 was 0)
    state = initial_soft_cpu_state()._replace(A=jnp.float32(0x40))
    bus = initial_soft_bus(_rom_with([0x0A]))
    state, _ = soft_step(state, bus)
    assert float(state.A) == 0x80
    assert not (int(state.P) & 0x01)   # C clear
    assert int(state.P) & 0x80         # N set


def test_asl_accumulator_carries_high_bit():
    # A = 0x81 → 0x02; C=1
    state = initial_soft_cpu_state()._replace(A=jnp.float32(0x81))
    bus = initial_soft_bus(_rom_with([0x0A]))
    state, _ = soft_step(state, bus)
    assert float(state.A) == 0x02
    assert int(state.P) & 0x01


def test_asl_zp_writes_back_to_ram():
    # RAM[$20] = 0x03 → 0x06
    bus = initial_soft_bus(_rom_with([0x06, 0x20]))
    bus = bus._replace(ram=bus.ram.at[0x20].set(jnp.float32(0x03)))
    _, bus = soft_step(initial_soft_cpu_state(), bus)
    assert float(bus.ram[0x20]) == 0x06


# --------------------------------------------------------------------------- #
# LSR — bit 0 → C; right-shift; bit 7 always 0
# --------------------------------------------------------------------------- #

def test_lsr_accumulator():
    # A = 0x03 → 0x01; C=1 (bit 0 was 1)
    state = initial_soft_cpu_state()._replace(A=jnp.float32(0x03))
    bus = initial_soft_bus(_rom_with([0x4A]))
    state, _ = soft_step(state, bus)
    assert float(state.A) == 0x01
    assert int(state.P) & 0x01


def test_lsr_clears_n():
    # 0xFF → 0x7F. N must always be 0 after LSR.
    state = initial_soft_cpu_state()._replace(A=jnp.float32(0xFF))
    bus = initial_soft_bus(_rom_with([0x4A]))
    state, _ = soft_step(state, bus)
    assert float(state.A) == 0x7F
    assert not (int(state.P) & 0x80)


def test_lsr_zero_result_sets_z():
    state = initial_soft_cpu_state()._replace(A=jnp.float32(0x01))
    bus = initial_soft_bus(_rom_with([0x4A]))
    state, _ = soft_step(state, bus)
    assert float(state.A) == 0x00
    assert int(state.P) & 0x02   # Z set
    assert int(state.P) & 0x01   # C set (bit 0 was 1)


def test_lsr_abs_x_indexes_correctly():
    # Pre-set RAM[$25] = 0x04 → LSR $20,X with X=5 → RAM[$25] = 0x02
    bus = initial_soft_bus(_rom_with([0x5E, 0x20, 0x00]))
    bus = bus._replace(ram=bus.ram.at[0x25].set(jnp.float32(0x04)))
    state = initial_soft_cpu_state()._replace(X=jnp.float32(5))
    _, bus = soft_step(state, bus)
    assert float(bus.ram[0x25]) == 0x02


# --------------------------------------------------------------------------- #
# ROL — rotate left through carry
# --------------------------------------------------------------------------- #

def test_rol_acc_takes_carry_into_bit_0():
    # A = 0x01, C=1 → A = 0x03, C = old bit 7 = 0
    state = initial_soft_cpu_state()._replace(
        A=jnp.float32(0x01), P=jnp.float32(0x35))   # C = 1
    bus = initial_soft_bus(_rom_with([0x2A]))
    state, _ = soft_step(state, bus)
    assert float(state.A) == 0x03
    assert not (int(state.P) & 0x01)


def test_rol_acc_high_bit_to_carry():
    state = initial_soft_cpu_state()._replace(
        A=jnp.float32(0x80), P=jnp.float32(0x34))   # C = 0
    bus = initial_soft_bus(_rom_with([0x2A]))
    state, _ = soft_step(state, bus)
    assert float(state.A) == 0x00
    assert int(state.P) & 0x01   # C set (was bit 7 of A)
    assert int(state.P) & 0x02   # Z set


def test_rol_zp_writes_back():
    # RAM[$10] = 0x40; with C=0 → RAM[$10] = 0x80
    bus = initial_soft_bus(_rom_with([0x26, 0x10]))
    bus = bus._replace(ram=bus.ram.at[0x10].set(jnp.float32(0x40)))
    state = initial_soft_cpu_state()._replace(P=jnp.float32(0x34))
    _, bus = soft_step(state, bus)
    assert float(bus.ram[0x10]) == 0x80


# --------------------------------------------------------------------------- #
# ROR — rotate right through carry
# --------------------------------------------------------------------------- #

def test_ror_acc_takes_carry_into_bit_7():
    # A = 0x02, C=1 → A = 0x81 (bit 0 of A → C; old C → bit 7)
    state = initial_soft_cpu_state()._replace(
        A=jnp.float32(0x02), P=jnp.float32(0x35))   # C = 1
    bus = initial_soft_bus(_rom_with([0x6A]))
    state, _ = soft_step(state, bus)
    assert float(state.A) == 0x81
    assert not (int(state.P) & 0x01)   # bit 0 was 0


def test_ror_acc_drops_low_bit_to_carry():
    state = initial_soft_cpu_state()._replace(
        A=jnp.float32(0x03), P=jnp.float32(0x34))   # C = 0
    bus = initial_soft_bus(_rom_with([0x6A]))
    state, _ = soft_step(state, bus)
    assert float(state.A) == 0x01
    assert int(state.P) & 0x01   # bit 0 was 1


def test_ror_abs_writes_back():
    # RAM[$30] = 0x02, C=1 → RAM[$30] = 0x81
    bus = initial_soft_bus(_rom_with([0x6E, 0x30, 0x00]))
    bus = bus._replace(ram=bus.ram.at[0x30].set(jnp.float32(0x02)))
    state = initial_soft_cpu_state()._replace(P=jnp.float32(0x35))
    _, bus = soft_step(state, bus)
    assert float(bus.ram[0x30]) == 0x81


# --------------------------------------------------------------------------- #
# PC + cycle advance sanity (accumulator vs. memory mode)
# --------------------------------------------------------------------------- #

def test_asl_acc_uses_1_byte_2_cycles():
    state = initial_soft_cpu_state()._replace(A=jnp.float32(0x01))
    bus = initial_soft_bus(_rom_with([0x0A]))
    state, _ = soft_step(state, bus)
    assert float(state.PC) == 0xF001
    assert float(state.cycles) == 2.0


def test_asl_zp_uses_2_bytes_5_cycles():
    bus = initial_soft_bus(_rom_with([0x06, 0x20]))
    state, _ = soft_step(initial_soft_cpu_state(), bus)
    assert float(state.PC) == 0xF002
    assert float(state.cycles) == 5.0


def test_asl_abs_x_uses_3_bytes_7_cycles():
    bus = initial_soft_bus(_rom_with([0x1E, 0x00, 0x00]))
    state, _ = soft_step(initial_soft_cpu_state(), bus)
    assert float(state.PC) == 0xF003
    assert float(state.cycles) == 7.0
