"""P1b1 tests — AND / ORA / EOR, CMP / CPX / CPY, BIT.

Same memory-image helpers as test_cpu_p1a.py; covers the flag side effects
that are unique to this phase (carry-on-compare, V-from-bit-6 on BIT).
"""

import jax.numpy as jnp

from jaxtari.cpu.m6502 import step
from jaxtari.cpu.tables import FLAG_C, FLAG_N, FLAG_U, FLAG_V, FLAG_Z
from jaxtari.types import CPUState, initial_cpu_state


def _make_memory(image: dict[int, int]) -> jnp.ndarray:
    mem = jnp.zeros((1 << 16,), dtype=jnp.uint8)
    for addr, value in image.items():
        mem = mem.at[addr].set(jnp.uint8(value & 0xFF))
    return mem


def _state(**fields) -> CPUState:
    """Build a CPUState with arbitrary scalar fields overridden.

    Uses `astype(getattr(s, k).dtype)` rather than `type(...)` because in JAX
    every field's concrete type is `jaxlib._jax.ArrayImpl` and that class is
    not callable as a constructor.
    """
    s = initial_cpu_state()
    return s._replace(**{
        k: jnp.asarray(v).astype(getattr(s, k).dtype) for k, v in fields.items()
    })


# --------------------------------------------------------------------------- #
# AND / ORA / EOR
# --------------------------------------------------------------------------- #

def test_and_immediate():
    s = _state(PC=0x8000, A=0xF0)
    mem = _make_memory({0x8000: 0x29, 0x8001: 0x0F})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x00
    assert int(s2.P) & FLAG_Z != 0
    assert int(s2.P) & FLAG_N == 0
    assert int(s2.cycles) == 2


def test_ora_immediate_sets_n():
    s = _state(PC=0x8000, A=0x01)
    mem = _make_memory({0x8000: 0x09, 0x8001: 0x80})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x81
    assert int(s2.P) & FLAG_N != 0
    assert int(s2.P) & FLAG_Z == 0


def test_eor_immediate_toggles_high_bit():
    s = _state(PC=0x8000, A=0xFF)
    mem = _make_memory({0x8000: 0x49, 0x8001: 0x80})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x7F
    assert int(s2.P) & FLAG_N == 0


def test_and_absolute_x_with_page_cross_adds_cycle():
    s = _state(PC=0x8000, A=0xFF, X=0x10)
    mem = _make_memory({0x8000: 0x3D, 0x8001: 0xF5, 0x8002: 0x12, 0x1305: 0x0F})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x0F
    assert int(s2.cycles) == 5  # 4 base + 1 page cross


def test_ora_indirect_y_no_page_cross():
    s = _state(PC=0x8000, A=0x00, Y=0x01)
    mem = _make_memory({
        0x8000: 0x11, 0x8001: 0x10,
        0x0010: 0x00, 0x0011: 0x12,
        0x1201: 0x42,
    })
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x42
    assert int(s2.cycles) == 5


# --------------------------------------------------------------------------- #
# CMP / CPX / CPY
# --------------------------------------------------------------------------- #

def test_cmp_immediate_equal_sets_z_and_c():
    s = _state(PC=0x8000, A=0x42)
    mem = _make_memory({0x8000: 0xC9, 0x8001: 0x42})
    s2, _ = step(s, mem)
    assert int(s2.P) & FLAG_Z != 0
    assert int(s2.P) & FLAG_C != 0
    assert int(s2.P) & FLAG_N == 0
    assert int(s2.A) == 0x42  # A unchanged
    assert int(s2.cycles) == 2


def test_cmp_immediate_a_greater_sets_c_clears_z():
    s = _state(PC=0x8000, A=0x80)
    mem = _make_memory({0x8000: 0xC9, 0x8001: 0x40})
    s2, _ = step(s, mem)
    assert int(s2.P) & FLAG_C != 0
    assert int(s2.P) & FLAG_Z == 0
    assert int(s2.P) & FLAG_N == 0  # diff=0x40 positive


def test_cmp_immediate_a_less_clears_c_sets_n():
    s = _state(PC=0x8000, A=0x10)
    mem = _make_memory({0x8000: 0xC9, 0x8001: 0x20})
    s2, _ = step(s, mem)
    assert int(s2.P) & FLAG_C == 0       # borrow
    assert int(s2.P) & FLAG_Z == 0
    assert int(s2.P) & FLAG_N != 0       # diff = 0xF0


def test_cmp_absolute_x_page_cross_adds_cycle():
    s = _state(PC=0x8000, A=0x10, X=0x10)
    mem = _make_memory({0x8000: 0xDD, 0x8001: 0xF5, 0x8002: 0x12, 0x1305: 0x10})
    s2, _ = step(s, mem)
    assert int(s2.P) & FLAG_Z != 0
    assert int(s2.cycles) == 5  # 4 base + 1 page cross


def test_cpx_zero_page():
    s = _state(PC=0x8000, X=0xFF)
    mem = _make_memory({0x8000: 0xE4, 0x8001: 0x10, 0x0010: 0xFF})
    s2, _ = step(s, mem)
    assert int(s2.P) & FLAG_Z != 0
    assert int(s2.P) & FLAG_C != 0


def test_cpy_immediate_y_less():
    s = _state(PC=0x8000, Y=0x05)
    mem = _make_memory({0x8000: 0xC0, 0x8001: 0x10})
    s2, _ = step(s, mem)
    assert int(s2.P) & FLAG_C == 0
    assert int(s2.P) & FLAG_N != 0


# --------------------------------------------------------------------------- #
# BIT
# --------------------------------------------------------------------------- #

def test_bit_zero_page_z_from_a_and_operand():
    s = _state(PC=0x8000, A=0x0F)
    mem = _make_memory({0x8000: 0x24, 0x8001: 0x10, 0x0010: 0xF0})
    s2, _ = step(s, mem)
    # A & operand = 0x0F & 0xF0 = 0x00 -> Z set
    assert int(s2.P) & FLAG_Z != 0
    # operand bits 7 and 6 -> N=1, V=1
    assert int(s2.P) & FLAG_N != 0
    assert int(s2.P) & FLAG_V != 0
    assert int(s2.A) == 0x0F  # A unchanged
    assert int(s2.cycles) == 3


def test_bit_absolute_clears_v_when_operand_bit6_clear():
    s = _state(PC=0x8000, A=0xFF, P=FLAG_U | FLAG_V)  # V preset
    mem = _make_memory({0x8000: 0x2C, 0x8001: 0x00, 0x8002: 0x12, 0x1200: 0x80})
    s2, _ = step(s, mem)
    # operand 0x80 -> bit 7 set (N=1), bit 6 clear (V cleared)
    assert int(s2.P) & FLAG_N != 0
    assert int(s2.P) & FLAG_V == 0
    # A & operand = 0xFF & 0x80 = 0x80 -> Z cleared
    assert int(s2.P) & FLAG_Z == 0
    assert int(s2.cycles) == 4
