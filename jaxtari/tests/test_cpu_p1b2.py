"""P1b2 tests — ADC and SBC, binary and BCD modes.

Reference algorithm: xitari/emucore/m6502/src/M6502Hi.ins case 0x69 (ADC)
and case 0xe9 (SBC). NMOS semantics with the xitari/Stella decimal-mode
V-flag convention.
"""

import jax.numpy as jnp

from jaxtari.cpu.m6502 import step
from jaxtari.cpu.tables import FLAG_C, FLAG_D, FLAG_N, FLAG_U, FLAG_V, FLAG_Z
from jaxtari.types import CPUState, initial_cpu_state


def _make_memory(image: dict[int, int]) -> jnp.ndarray:
    mem = jnp.zeros((1 << 16,), dtype=jnp.uint8)
    for addr, value in image.items():
        mem = mem.at[addr].set(jnp.uint8(value & 0xFF))
    return mem


def _state(**fields) -> CPUState:
    s = initial_cpu_state()
    return s._replace(**{
        k: jnp.asarray(v).astype(getattr(s, k).dtype) for k, v in fields.items()
    })


# --------------------------------------------------------------------------- #
# ADC binary
# --------------------------------------------------------------------------- #

def test_adc_binary_simple():
    s = _state(PC=0x8000, A=0x10, P=FLAG_U)  # C=0
    mem = _make_memory({0x8000: 0x69, 0x8001: 0x05})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x15
    assert int(s2.P) & FLAG_C == 0
    assert int(s2.P) & FLAG_V == 0
    assert int(s2.P) & FLAG_Z == 0
    assert int(s2.P) & FLAG_N == 0
    assert int(s2.cycles) == 2


def test_adc_binary_uses_carry_in():
    s = _state(PC=0x8000, A=0x10, P=FLAG_U | FLAG_C)
    mem = _make_memory({0x8000: 0x69, 0x8001: 0x05})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x16  # +carry-in


def test_adc_binary_carry_out_on_overflow():
    s = _state(PC=0x8000, A=0xFE, P=FLAG_U | FLAG_C)
    mem = _make_memory({0x8000: 0x69, 0x8001: 0x01})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x00
    assert int(s2.P) & FLAG_C != 0
    assert int(s2.P) & FLAG_Z != 0


def test_adc_binary_signed_overflow_positive_to_negative():
    s = _state(PC=0x8000, A=0x7F, P=FLAG_U)
    mem = _make_memory({0x8000: 0x69, 0x8001: 0x01})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x80
    assert int(s2.P) & FLAG_V != 0    # 127 + 1 → -128, overflow
    assert int(s2.P) & FLAG_N != 0
    assert int(s2.P) & FLAG_C == 0    # unsigned 0x7F + 0x01 = 0x80, no carry


def test_adc_binary_signed_overflow_negative_to_positive():
    s = _state(PC=0x8000, A=0x80, P=FLAG_U)
    mem = _make_memory({0x8000: 0x69, 0x8001: 0x80})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x00
    assert int(s2.P) & FLAG_V != 0    # -128 + -128 = -256, overflow
    assert int(s2.P) & FLAG_C != 0    # unsigned 0x80 + 0x80 = 0x100


def test_adc_binary_no_signed_overflow_different_signs():
    s = _state(PC=0x8000, A=0x7F, P=FLAG_U)
    mem = _make_memory({0x8000: 0x69, 0x8001: 0x80})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0xFF
    assert int(s2.P) & FLAG_V == 0    # different signs cannot overflow
    assert int(s2.P) & FLAG_N != 0


# --------------------------------------------------------------------------- #
# ADC decimal
# --------------------------------------------------------------------------- #

def test_adc_decimal_simple():
    s = _state(PC=0x8000, A=0x12, P=FLAG_U | FLAG_D)  # 12 + 34 = 46
    mem = _make_memory({0x8000: 0x69, 0x8001: 0x34})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x46
    assert int(s2.P) & FLAG_C == 0


def test_adc_decimal_carry_in():
    s = _state(PC=0x8000, A=0x12, P=FLAG_U | FLAG_D | FLAG_C)  # 12 + 34 + 1
    mem = _make_memory({0x8000: 0x69, 0x8001: 0x34})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x47


def test_adc_decimal_overflow_sets_carry():
    s = _state(PC=0x8000, A=0x55, P=FLAG_U | FLAG_D)  # 55 + 55 = 110 → 10, C
    mem = _make_memory({0x8000: 0x69, 0x8001: 0x55})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x10
    assert int(s2.P) & FLAG_C != 0


def test_adc_decimal_zero_when_99_plus_1_with_carry():
    s = _state(PC=0x8000, A=0x99, P=FLAG_U | FLAG_D | FLAG_C)  # 99 + 0 + 1 = 100
    mem = _make_memory({0x8000: 0x69, 0x8001: 0x00})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x00
    assert int(s2.P) & FLAG_C != 0
    assert int(s2.P) & FLAG_Z != 0


# --------------------------------------------------------------------------- #
# SBC binary
# --------------------------------------------------------------------------- #

def test_sbc_binary_simple_with_carry_set():
    s = _state(PC=0x8000, A=0x10, P=FLAG_U | FLAG_C)  # 0x10 - 0x05 = 0x0B
    mem = _make_memory({0x8000: 0xE9, 0x8001: 0x05})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x0B
    assert int(s2.P) & FLAG_C != 0  # no borrow


def test_sbc_binary_borrow_when_carry_clear():
    s = _state(PC=0x8000, A=0x10, P=FLAG_U)  # C=0 means an extra -1
    mem = _make_memory({0x8000: 0xE9, 0x8001: 0x05})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x0A
    assert int(s2.P) & FLAG_C != 0


def test_sbc_binary_borrow_clears_carry_on_underflow():
    s = _state(PC=0x8000, A=0x05, P=FLAG_U | FLAG_C)
    mem = _make_memory({0x8000: 0xE9, 0x8001: 0x10})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0xF5
    assert int(s2.P) & FLAG_C == 0  # borrow occurred


def test_sbc_binary_signed_overflow():
    # 0x50 (+80) - 0xB0 (-80) - 0 = 0xA0 (-96); signed result actually +160 → overflow
    s = _state(PC=0x8000, A=0x50, P=FLAG_U | FLAG_C)
    mem = _make_memory({0x8000: 0xE9, 0x8001: 0xB0})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0xA0
    assert int(s2.P) & FLAG_V != 0


def test_sbc_undocumented_0xEB_immediate_aliases_sbc():
    s = _state(PC=0x8000, A=0x42, P=FLAG_U | FLAG_C)
    mem = _make_memory({0x8000: 0xEB, 0x8001: 0x02})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x40


# --------------------------------------------------------------------------- #
# SBC decimal
# --------------------------------------------------------------------------- #

def test_sbc_decimal_simple_with_carry_set():
    s = _state(PC=0x8000, A=0x46, P=FLAG_U | FLAG_D | FLAG_C)  # 46 - 12 = 34
    mem = _make_memory({0x8000: 0xE9, 0x8001: 0x12})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x34
    assert int(s2.P) & FLAG_C != 0


def test_sbc_decimal_borrow_into_next_digit():
    s = _state(PC=0x8000, A=0x40, P=FLAG_U | FLAG_D | FLAG_C)  # 40 - 12 = 28
    mem = _make_memory({0x8000: 0xE9, 0x8001: 0x12})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x28


def test_sbc_decimal_underflow_wraps_and_clears_carry():
    s = _state(PC=0x8000, A=0x10, P=FLAG_U | FLAG_D | FLAG_C)  # 10 - 20 = -10 → 90, borrow
    mem = _make_memory({0x8000: 0xE9, 0x8001: 0x20})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x90
    assert int(s2.P) & FLAG_C == 0  # borrow taken


# --------------------------------------------------------------------------- #
# Addressing-mode coverage smoke test (binary mode only, to keep this short)
# --------------------------------------------------------------------------- #

def test_adc_absolute_x_page_cross_adds_cycle():
    s = _state(PC=0x8000, A=0x01, X=0x10, P=FLAG_U)
    mem = _make_memory({0x8000: 0x7D, 0x8001: 0xF5, 0x8002: 0x12, 0x1305: 0x02})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x03
    assert int(s2.cycles) == 5


def test_sbc_indirect_y_no_page_cross():
    s = _state(PC=0x8000, A=0x10, Y=0x01, P=FLAG_U | FLAG_C)
    mem = _make_memory({
        0x8000: 0xF1, 0x8001: 0x10,
        0x0010: 0x00, 0x0011: 0x12,
        0x1201: 0x05,
    })
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x0B
    assert int(s2.cycles) == 5  # 5 base, no page cross
