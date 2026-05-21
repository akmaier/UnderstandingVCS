"""P7c-b tests — arithmetic/logic opcodes in SOFT mode + N/Z/C/V flags.

Covers ADC / SBC (binary mode only; BCD is a P7c-bx deferral), AND / ORA /
EOR, CMP / CPX / CPY, and BIT. ADC/SBC keep float arithmetic on the value
so `jax.grad` flows through; AND/ORA/EOR cast to int for the bitwise op
(operand gradient breaks but the operand's *path* — `soft_rom_peek` —
still produces a clean gradient on the dispatched address). Compares and
BIT are pure flag updates with no register change.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxtari.diff import (
    SoftBus,
    SoftCPUState,
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


# --------------------------------------------------------------------------- #
# Coverage sanity
# --------------------------------------------------------------------------- #

def test_p7c_b_opcodes_all_present():
    p7c_b = {
        # ADC
        0x69, 0x65, 0x75, 0x6D, 0x7D, 0x79, 0x61, 0x71,
        # SBC + USBC
        0xE9, 0xE5, 0xF5, 0xED, 0xFD, 0xF9, 0xE1, 0xF1, 0xEB,
        # AND
        0x29, 0x25, 0x35, 0x2D, 0x3D, 0x39, 0x21, 0x31,
        # ORA
        0x09, 0x05, 0x15, 0x0D, 0x1D, 0x19, 0x01, 0x11,
        # EOR
        0x49, 0x45, 0x55, 0x4D, 0x5D, 0x59, 0x41, 0x51,
        # CMP
        0xC9, 0xC5, 0xD5, 0xCD, 0xDD, 0xD9, 0xC1, 0xD1,
        # CPX/CPY/BIT
        0xE0, 0xE4, 0xEC,
        0xC0, 0xC4, 0xCC,
        0x24, 0x2C,
    }
    assert p7c_b.issubset(SOFT_SUPPORTED_OPCODES)


# --------------------------------------------------------------------------- #
# ADC
# --------------------------------------------------------------------------- #

def test_adc_imm_simple_sum_no_carry():
    # LDA #$10 / ADC #$22
    bus = initial_soft_bus(_rom_with([0xA9, 0x10, 0x69, 0x22]))
    state, _ = soft_run(initial_soft_cpu_state(), bus, 2)
    assert float(state.A) == 0x32
    assert not (int(state.P) & 0x01)   # C clear
    assert not (int(state.P) & 0x02)   # Z clear
    assert not (int(state.P) & 0x40)   # V clear


def test_adc_imm_with_carry_in():
    # Pre-set C = 1; LDA #$01 / ADC #$01 → A = 3
    bus = initial_soft_bus(_rom_with([0xA9, 0x01, 0x69, 0x01]))
    state = initial_soft_cpu_state()._replace(P=jnp.float32(0x35))   # bit 0 = C set
    state, _ = soft_run(state, bus, 2)
    assert float(state.A) == 0x03


def test_adc_overflows_to_carry():
    # LDA #$FF / ADC #$01 → 0x100 → A = 0, C = 1, Z = 1
    bus = initial_soft_bus(_rom_with([0xA9, 0xFF, 0x69, 0x01]))
    state, _ = soft_run(initial_soft_cpu_state(), bus, 2)
    assert float(state.A) == 0x00
    assert int(state.P) & 0x01   # C set
    assert int(state.P) & 0x02   # Z set


def test_adc_signed_overflow_sets_v():
    # LDA #$50 / ADC #$50 → 0xA0; sign-bit flipped from positive operands → V=1
    bus = initial_soft_bus(_rom_with([0xA9, 0x50, 0x69, 0x50]))
    state, _ = soft_run(initial_soft_cpu_state(), bus, 2)
    assert float(state.A) == 0xA0
    assert int(state.P) & 0x40   # V set


def test_adc_zp_reads_ram_operand():
    # Pre-set RAM[$30] = 0x05; LDA #$03 / ADC $30 → A = 8
    bus = initial_soft_bus(_rom_with([0xA9, 0x03, 0x65, 0x30]))
    bus = bus._replace(ram=bus.ram.at[0x30].set(jnp.float32(0x05)))
    state, _ = soft_run(initial_soft_cpu_state(), bus, 2)
    assert float(state.A) == 0x08


# --------------------------------------------------------------------------- #
# SBC
# --------------------------------------------------------------------------- #

def test_sbc_imm_with_carry_set_simple_diff():
    # SBC requires C=1 for "no borrow". LDA #$50 / SEC trick: pre-set C=1.
    bus = initial_soft_bus(_rom_with([0xA9, 0x50, 0xE9, 0x30]))
    state = initial_soft_cpu_state()._replace(P=jnp.float32(0x35))   # C=1
    state, _ = soft_run(state, bus, 2)
    assert float(state.A) == 0x20
    assert int(state.P) & 0x01   # C remains set (no borrow needed)


def test_sbc_borrows_when_minuend_smaller():
    # LDA #$10 / SBC #$20 with C=1 → 0xF0, C=0 (borrow occurred)
    bus = initial_soft_bus(_rom_with([0xA9, 0x10, 0xE9, 0x20]))
    state = initial_soft_cpu_state()._replace(P=jnp.float32(0x35))   # C=1
    state, _ = soft_run(state, bus, 2)
    assert float(state.A) == 0xF0
    assert not (int(state.P) & 0x01)   # C cleared


def test_usbc_eb_aliases_sbc_imm():
    # 0xEB is USBC — undocumented, same forward behaviour as SBC #imm.
    bus = initial_soft_bus(_rom_with([0xA9, 0x80, 0xEB, 0x10]))
    state = initial_soft_cpu_state()._replace(P=jnp.float32(0x35))
    state, _ = soft_run(state, bus, 2)
    assert float(state.A) == 0x70


# --------------------------------------------------------------------------- #
# Bitwise — AND / ORA / EOR
# --------------------------------------------------------------------------- #

def test_and_imm_masks_bits():
    bus = initial_soft_bus(_rom_with([0xA9, 0xF0, 0x29, 0x0F]))
    state, _ = soft_run(initial_soft_cpu_state(), bus, 2)
    assert float(state.A) == 0x00
    assert int(state.P) & 0x02   # Z set


def test_ora_imm_or_bits():
    bus = initial_soft_bus(_rom_with([0xA9, 0x0F, 0x09, 0xF0]))
    state, _ = soft_run(initial_soft_cpu_state(), bus, 2)
    assert float(state.A) == 0xFF
    assert int(state.P) & 0x80   # N set


def test_eor_imm_flips_bits():
    bus = initial_soft_bus(_rom_with([0xA9, 0x55, 0x49, 0xFF]))
    state, _ = soft_run(initial_soft_cpu_state(), bus, 2)
    assert float(state.A) == 0xAA
    assert int(state.P) & 0x80   # N set


def test_and_zp_reads_from_ram():
    bus = initial_soft_bus(_rom_with([0xA9, 0x0F, 0x25, 0x20]))
    bus = bus._replace(ram=bus.ram.at[0x20].set(jnp.float32(0x06)))
    state, _ = soft_run(initial_soft_cpu_state(), bus, 2)
    assert float(state.A) == 0x06


# --------------------------------------------------------------------------- #
# CMP / CPX / CPY
# --------------------------------------------------------------------------- #

def test_cmp_equal_sets_z_and_c():
    # LDA #$42 / CMP #$42
    bus = initial_soft_bus(_rom_with([0xA9, 0x42, 0xC9, 0x42]))
    state, _ = soft_run(initial_soft_cpu_state(), bus, 2)
    assert float(state.A) == 0x42   # CMP doesn't change A
    assert int(state.P) & 0x02   # Z set (equal)
    assert int(state.P) & 0x01   # C set (>=)


def test_cmp_greater_sets_c_clears_z():
    bus = initial_soft_bus(_rom_with([0xA9, 0x50, 0xC9, 0x30]))
    state, _ = soft_run(initial_soft_cpu_state(), bus, 2)
    assert int(state.P) & 0x01    # C set
    assert not (int(state.P) & 0x02)   # Z clear


def test_cmp_less_clears_c():
    bus = initial_soft_bus(_rom_with([0xA9, 0x20, 0xC9, 0x50]))
    state, _ = soft_run(initial_soft_cpu_state(), bus, 2)
    assert not (int(state.P) & 0x01)   # C clear
    assert int(state.P) & 0x80    # N set (negative result)


def test_cpx_imm_compares_x():
    state = initial_soft_cpu_state()._replace(X=jnp.float32(0x10))
    bus = initial_soft_bus(_rom_with([0xE0, 0x10]))
    state, _ = soft_step(state, bus)
    assert int(state.P) & 0x02
    assert int(state.P) & 0x01


def test_cpy_imm_compares_y():
    state = initial_soft_cpu_state()._replace(Y=jnp.float32(0x05))
    bus = initial_soft_bus(_rom_with([0xC0, 0x07]))
    state, _ = soft_step(state, bus)
    assert not (int(state.P) & 0x01)   # Y < operand → C clear


# --------------------------------------------------------------------------- #
# BIT
# --------------------------------------------------------------------------- #

def test_bit_zp_z_set_when_no_overlap():
    state = initial_soft_cpu_state()._replace(A=jnp.float32(0x0F))
    bus = initial_soft_bus(_rom_with([0x24, 0x20]))
    bus = bus._replace(ram=bus.ram.at[0x20].set(jnp.float32(0xF0)))
    state, _ = soft_step(state, bus)
    assert int(state.P) & 0x02   # Z set (A & op == 0)
    assert int(state.P) & 0x80   # N from op bit 7 (0xF0 → bit7=1)
    assert int(state.P) & 0x40   # V from op bit 6 (0xF0 → bit6=1)


def test_bit_abs_n_v_from_operand_z_from_and():
    state = initial_soft_cpu_state()._replace(A=jnp.float32(0xFF))
    # BIT $0030 (abs). RAM[$30] = 0xC0 → N=1, V=1, A&op = 0xC0 → Z clear
    bus = initial_soft_bus(_rom_with([0x2C, 0x30, 0x00]))
    bus = bus._replace(ram=bus.ram.at[0x30].set(jnp.float32(0xC0)))
    state, _ = soft_step(state, bus)
    assert int(state.P) & 0x80
    assert int(state.P) & 0x40
    assert not (int(state.P) & 0x02)


# --------------------------------------------------------------------------- #
# Gradient tests — ADC keeps a clean gradient through float arithmetic.
# --------------------------------------------------------------------------- #

def test_grad_adc_imm_through_a():
    """LDA #$10 / ADC #$22; ∂A/∂rom[3] (the second immediate) = 1."""
    rom_init = _rom_with([0xA9, 0x10, 0x69, 0x22], size=256)

    def simulator(rom_arr):
        bus = SoftBus(ram=jnp.zeros((128,), dtype=jnp.float32), rom=rom_arr)
        state = initial_soft_cpu_state()
        state, _ = soft_run(state, bus, n_steps=2)
        return state.A

    forward = simulator(rom_init)
    assert float(forward) == 0x32

    grad = jax.grad(simulator)(rom_init)
    # The ADC operand (rom[3]) adds into A — gradient is 1.0.
    assert float(grad[3]) == pytest.approx(1.0)
    # The LDA immediate (rom[1]) also feeds into A — gradient is 1.0.
    assert float(grad[1]) == pytest.approx(1.0)
    other = (float(jnp.sum(jnp.abs(grad)))
             - float(jnp.abs(grad[1]))
             - float(jnp.abs(grad[3])))
    assert other == pytest.approx(0.0)


def test_grad_sbc_imm_through_a():
    """LDA #$50 / SBC #$30 (with C=1) → A=0x20; ∂A/∂rom[1] = 1, ∂A/∂rom[3] = -1."""
    rom_init = _rom_with([0xA9, 0x50, 0xE9, 0x30], size=256)

    def simulator(rom_arr):
        bus = SoftBus(ram=jnp.zeros((128,), dtype=jnp.float32), rom=rom_arr)
        state = initial_soft_cpu_state()._replace(P=jnp.float32(0x35))   # C=1
        state, _ = soft_run(state, bus, n_steps=2)
        return state.A

    assert float(simulator(rom_init)) == 0x20
    grad = jax.grad(simulator)(rom_init)
    assert float(grad[1]) == pytest.approx(1.0)
    # SBC subtracts the operand, so gradient is -1.
    assert float(grad[3]) == pytest.approx(-1.0)
