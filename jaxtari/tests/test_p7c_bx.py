"""P7c-bx tests — BCD (decimal-mode) ADC / SBC in SOFT mode.

When the D flag is set, ADC/SBC operate on packed binary-coded-decimal
operands. P7c-bx computes both the binary and the BCD result and
selects on the D flag. The binary path stays gradient-clean; the BCD
path is integer arithmetic. Forward behaviour matches xitari's
M6502Hi.ins decimal convention (the same one the HARD `cpu.alu` uses).
"""

import jax
import jax.numpy as jnp
import pytest

from jaxtari.diff import (
    SoftBus,
    initial_soft_bus,
    initial_soft_cpu_state,
    soft_run,
    soft_step,
)


def _rom_with(bytes_: list[int], size: int = 256) -> jnp.ndarray:
    rom = jnp.zeros((size,), dtype=jnp.float32)
    for i, b in enumerate(bytes_):
        rom = rom.at[i].set(jnp.float32(b))
    return rom


# P flag bytes: U=0x20 always; D=0x08; C=0x01.
_P_DECIMAL       = 0x20 | 0x08          # D set, C clear
_P_DECIMAL_CARRY = 0x20 | 0x08 | 0x01   # D set, C set
_P_BINARY        = 0x20                 # D clear, C clear


# --------------------------------------------------------------------------- #
# BCD ADC
# --------------------------------------------------------------------------- #

def test_adc_bcd_simple_sum():
    """0x25 + 0x12 in decimal mode = 0x37 (25 + 12 = 37)."""
    bus = initial_soft_bus(_rom_with([0xA9, 0x25, 0x69, 0x12]))
    state = initial_soft_cpu_state()._replace(P=jnp.float32(_P_DECIMAL))
    state, _ = soft_run(state, bus, 2)
    assert float(state.A) == 0x37


def test_adc_bcd_with_decimal_carry_out():
    """0x55 + 0x55 in decimal = 110 → 0x10 with carry set."""
    bus = initial_soft_bus(_rom_with([0xA9, 0x55, 0x69, 0x55]))
    state = initial_soft_cpu_state()._replace(P=jnp.float32(_P_DECIMAL))
    state, _ = soft_run(state, bus, 2)
    assert float(state.A) == 0x10
    assert int(state.P) & 0x01   # C set (sum exceeded 99)


def test_adc_bcd_uses_carry_in():
    """0x09 + 0x00 + C=1 in decimal = 0x10 (9 + 0 + 1 = 10)."""
    bus = initial_soft_bus(_rom_with([0xA9, 0x09, 0x69, 0x00]))
    state = initial_soft_cpu_state()._replace(P=jnp.float32(_P_DECIMAL_CARRY))
    state, _ = soft_run(state, bus, 2)
    assert float(state.A) == 0x10


def test_adc_binary_still_works_when_d_clear():
    """With D clear, ADC must still do binary arithmetic: 0x25 + 0x12 = 0x37
    here too (no decimal carries) — but 0x09 + 0x08 differs by mode."""
    bus = initial_soft_bus(_rom_with([0xA9, 0x09, 0x69, 0x08]))
    state = initial_soft_cpu_state()._replace(P=jnp.float32(_P_BINARY))
    state, _ = soft_run(state, bus, 2)
    # Binary: 0x09 + 0x08 = 0x11. (Decimal would give 0x17.)
    assert float(state.A) == 0x11


def test_adc_decimal_vs_binary_diverge():
    """The same bytes give different results in the two modes — proof the
    D-flag dispatch actually fires."""
    rom = _rom_with([0xA9, 0x09, 0x69, 0x08])
    dec, _ = soft_run(initial_soft_cpu_state()._replace(P=jnp.float32(_P_DECIMAL)),
                      initial_soft_bus(rom), 2)
    binv, _ = soft_run(initial_soft_cpu_state()._replace(P=jnp.float32(_P_BINARY)),
                       initial_soft_bus(rom), 2)
    assert float(dec.A) == 0x17    # 9 + 8 = 17 decimal
    assert float(binv.A) == 0x11   # 0x09 + 0x08 binary


# --------------------------------------------------------------------------- #
# BCD SBC
# --------------------------------------------------------------------------- #

def test_sbc_bcd_simple_diff():
    """0x50 - 0x25 in decimal (C=1, no borrow) = 0x25 (50 - 25 = 25)."""
    bus = initial_soft_bus(_rom_with([0xA9, 0x50, 0xE9, 0x25]))
    state = initial_soft_cpu_state()._replace(P=jnp.float32(_P_DECIMAL_CARRY))
    state, _ = soft_run(state, bus, 2)
    assert float(state.A) == 0x25


def test_sbc_bcd_borrow_wraps():
    """0x10 - 0x20 in decimal (C=1) = 90 with borrow (10 - 20 = -10 → 90)."""
    bus = initial_soft_bus(_rom_with([0xA9, 0x10, 0xE9, 0x20]))
    state = initial_soft_cpu_state()._replace(P=jnp.float32(_P_DECIMAL_CARRY))
    state, _ = soft_run(state, bus, 2)
    assert float(state.A) == 0x90
    assert not (int(state.P) & 0x01)   # C cleared — a borrow happened


def test_sbc_decimal_vs_binary_diverge():
    rom = _rom_with([0xA9, 0x30, 0xE9, 0x11])
    dec, _ = soft_run(
        initial_soft_cpu_state()._replace(P=jnp.float32(_P_DECIMAL_CARRY)),
        initial_soft_bus(rom), 2)
    binv, _ = soft_run(
        initial_soft_cpu_state()._replace(P=jnp.float32(_P_BINARY | 0x01)),
        initial_soft_bus(rom), 2)
    assert float(dec.A) == 0x19    # 30 - 11 = 19 decimal
    assert float(binv.A) == 0x1F   # 0x30 - 0x11 binary


# --------------------------------------------------------------------------- #
# Gradient — the binary path stays gradient-clean
# --------------------------------------------------------------------------- #

def test_grad_adc_still_flows_in_binary_mode():
    """With D clear, ADC gradient is unchanged from P7c-b: ∂A/∂operand = 1."""
    rom_init = _rom_with([0xA9, 0x10, 0x69, 0x22])

    def simulator(rom_arr):
        bus = SoftBus(ram=jnp.zeros((128,), dtype=jnp.float32), rom=rom_arr)
        state = initial_soft_cpu_state()._replace(P=jnp.float32(_P_BINARY))
        state, _ = soft_run(state, bus, n_steps=2)
        return state.A

    grad = jax.grad(simulator)(rom_init)
    assert float(grad[1]) == pytest.approx(1.0)   # LDA immediate
    assert float(grad[3]) == pytest.approx(1.0)   # ADC immediate
