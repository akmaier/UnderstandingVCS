"""P7c-e tests — SOFT-mode stack, status-flag opcodes, INC/DEC, register
increment/decrement.

Covers PHA/PHP/PLA/PLP (round-trip through the SOFT stack),
CLC/SEC/CLI/SEI/CLV/CLD/SED, INC/DEC on memory (RMW), and
INX/INY/DEX/DEY on registers — all with the N/Z flag updates the 6502
applies.
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


def test_p7c_e_opcodes_all_present():
    p7c_e = {
        0x48, 0x08, 0x68, 0x28,                       # PHA/PHP/PLA/PLP
        0x18, 0x38, 0x58, 0x78, 0xB8, 0xD8, 0xF8,     # flag ops
        0xE6, 0xF6, 0xEE, 0xFE,                       # INC
        0xC6, 0xD6, 0xCE, 0xDE,                       # DEC
        0xE8, 0xC8, 0xCA, 0x88,                       # INX/INY/DEX/DEY
    }
    assert p7c_e.issubset(SOFT_SUPPORTED_OPCODES)
    assert len(p7c_e) == 23


# --------------------------------------------------------------------------- #
# Stack — PHA / PLA
# --------------------------------------------------------------------------- #

def test_pha_then_pla_round_trips_a():
    """PHA pushes A; load a different A; PLA pulls the original back."""
    # PHA / LDA #$00 / PLA  — A should end up as the pushed value.
    rom = _rom_with([0x48, 0xA9, 0x00, 0x68])
    state = initial_soft_cpu_state()._replace(A=jnp.float32(0x5A))
    bus = initial_soft_bus(rom)
    state, bus = soft_step(state, bus)        # PHA  (A=0x5A pushed)
    assert float(state.SP) == 0xFC
    state, bus = soft_step(state, bus)        # LDA #$00
    assert float(state.A) == 0x00
    state, bus = soft_step(state, bus)        # PLA  (A back to 0x5A)
    assert float(state.A) == 0x5A
    assert float(state.SP) == 0xFD


def test_pla_sets_z_flag_on_zero():
    rom = _rom_with([0x48, 0x68])             # PHA / PLA
    state = initial_soft_cpu_state()._replace(A=jnp.float32(0x00))
    bus = initial_soft_bus(rom)
    state, bus = soft_run(state, bus, 2)
    assert int(state.P) & 0x02   # Z set


# --------------------------------------------------------------------------- #
# Stack — PHP / PLP
# --------------------------------------------------------------------------- #

def test_php_pushes_p_with_b_and_u_set():
    rom = _rom_with([0x08])
    state = initial_soft_cpu_state()._replace(P=jnp.float32(0x00))
    bus = initial_soft_bus(rom)
    state, bus = soft_step(state, bus)
    # Pushed P = 0x00 | 0x30. SP was 0xFD → byte at $01FD → RAM[$7D].
    assert float(bus.ram[0x7D]) == 0x30


def test_plp_forces_b_and_u_on_pull():
    """Push 0x00, pull it back — P should come back with B+U forced on."""
    rom = _rom_with([0x08, 0x28])             # PHP / PLP
    state = initial_soft_cpu_state()._replace(P=jnp.float32(0x00))
    bus = initial_soft_bus(rom)
    state, bus = soft_run(state, bus, 2)
    assert int(state.P) & 0x10   # B
    assert int(state.P) & 0x20   # U


# --------------------------------------------------------------------------- #
# Status-flag opcodes
# --------------------------------------------------------------------------- #

def test_sec_sets_carry():
    state, _ = soft_step(initial_soft_cpu_state(), initial_soft_bus(_rom_with([0x38])))
    assert int(state.P) & 0x01


def test_clc_clears_carry():
    state = initial_soft_cpu_state()._replace(P=jnp.float32(0x35))   # C=1
    state, _ = soft_step(state, initial_soft_bus(_rom_with([0x18])))
    assert not (int(state.P) & 0x01)


def test_sei_sets_interrupt_disable():
    state = initial_soft_cpu_state()._replace(P=jnp.float32(0x30))   # I=0
    state, _ = soft_step(state, initial_soft_bus(_rom_with([0x78])))
    assert int(state.P) & 0x04


def test_cli_clears_interrupt_disable():
    state = initial_soft_cpu_state()._replace(P=jnp.float32(0x34))   # I=1
    state, _ = soft_step(state, initial_soft_bus(_rom_with([0x58])))
    assert not (int(state.P) & 0x04)


def test_sed_sets_decimal():
    state, _ = soft_step(initial_soft_cpu_state(), initial_soft_bus(_rom_with([0xF8])))
    assert int(state.P) & 0x08


def test_cld_clears_decimal():
    state = initial_soft_cpu_state()._replace(P=jnp.float32(0x3C))   # D=1
    state, _ = soft_step(state, initial_soft_bus(_rom_with([0xD8])))
    assert not (int(state.P) & 0x08)


def test_clv_clears_overflow():
    state = initial_soft_cpu_state()._replace(P=jnp.float32(0x74))   # V=1
    state, _ = soft_step(state, initial_soft_bus(_rom_with([0xB8])))
    assert not (int(state.P) & 0x40)


# --------------------------------------------------------------------------- #
# INC / DEC memory
# --------------------------------------------------------------------------- #

def test_inc_zp_increments_ram_byte():
    bus = initial_soft_bus(_rom_with([0xE6, 0x20]))
    bus = bus._replace(ram=bus.ram.at[0x20].set(jnp.float32(0x41)))
    _, bus = soft_step(initial_soft_cpu_state(), bus)
    assert float(bus.ram[0x20]) == 0x42


def test_inc_wraps_ff_to_zero_and_sets_z():
    bus = initial_soft_bus(_rom_with([0xE6, 0x20]))
    bus = bus._replace(ram=bus.ram.at[0x20].set(jnp.float32(0xFF)))
    state, bus = soft_step(initial_soft_cpu_state(), bus)
    assert float(bus.ram[0x20]) == 0x00
    assert int(state.P) & 0x02   # Z set


def test_dec_zp_decrements_ram_byte():
    bus = initial_soft_bus(_rom_with([0xC6, 0x20]))
    bus = bus._replace(ram=bus.ram.at[0x20].set(jnp.float32(0x10)))
    _, bus = soft_step(initial_soft_cpu_state(), bus)
    assert float(bus.ram[0x20]) == 0x0F


def test_dec_wraps_zero_to_ff_and_sets_n():
    bus = initial_soft_bus(_rom_with([0xC6, 0x20]))
    bus = bus._replace(ram=bus.ram.at[0x20].set(jnp.float32(0x00)))
    state, bus = soft_step(initial_soft_cpu_state(), bus)
    assert float(bus.ram[0x20]) == 0xFF
    assert int(state.P) & 0x80   # N set


def test_inc_abs_x_indexed():
    bus = initial_soft_bus(_rom_with([0xFE, 0x20, 0x00]))
    bus = bus._replace(ram=bus.ram.at[0x24].set(jnp.float32(0x07)))
    state = initial_soft_cpu_state()._replace(X=jnp.float32(4))
    _, bus = soft_step(state, bus)
    assert float(bus.ram[0x24]) == 0x08


# --------------------------------------------------------------------------- #
# INX / INY / DEX / DEY
# --------------------------------------------------------------------------- #

def test_inx_increments_x():
    state = initial_soft_cpu_state()._replace(X=jnp.float32(0x10))
    state, _ = soft_step(state, initial_soft_bus(_rom_with([0xE8])))
    assert float(state.X) == 0x11


def test_iny_wraps_and_sets_z():
    state = initial_soft_cpu_state()._replace(Y=jnp.float32(0xFF))
    state, _ = soft_step(state, initial_soft_bus(_rom_with([0xC8])))
    assert float(state.Y) == 0x00
    assert int(state.P) & 0x02


def test_dex_decrements_x():
    state = initial_soft_cpu_state()._replace(X=jnp.float32(0x05))
    state, _ = soft_step(state, initial_soft_bus(_rom_with([0xCA])))
    assert float(state.X) == 0x04


def test_dey_wraps_zero_to_ff_sets_n():
    state = initial_soft_cpu_state()._replace(Y=jnp.float32(0x00))
    state, _ = soft_step(state, initial_soft_bus(_rom_with([0x88])))
    assert float(state.Y) == 0xFF
    assert int(state.P) & 0x80
