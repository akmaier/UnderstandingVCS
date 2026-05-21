"""P1c tests — ASL, LSR, ROL, ROR in accumulator and memory addressing modes."""

import jax.numpy as jnp

from jaxtari.cpu.m6502 import step
from jaxtari.cpu.tables import FLAG_C, FLAG_N, FLAG_U, FLAG_Z
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
# ASL
# --------------------------------------------------------------------------- #

def test_asl_accumulator_shifts_bit_into_carry():
    s = _state(PC=0x8000, A=0x80, P=FLAG_U)
    mem = _make_memory({0x8000: 0x0A})  # ASL A
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x00
    assert int(s2.P) & FLAG_C != 0
    assert int(s2.P) & FLAG_Z != 0
    assert int(s2.P) & FLAG_N == 0
    assert int(s2.cycles) == 2


def test_asl_accumulator_normal_shift():
    s = _state(PC=0x8000, A=0x01, P=FLAG_U)
    mem = _make_memory({0x8000: 0x0A})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x02
    assert int(s2.P) & FLAG_C == 0
    assert int(s2.P) & FLAG_Z == 0


def test_asl_zero_page_writes_back_and_sets_n():
    s = _state(PC=0x8000)
    mem = _make_memory({0x8000: 0x06, 0x8001: 0x10, 0x0010: 0x41})  # 0x41 << 1 = 0x82
    s2, mem2 = step(s, mem)
    assert int(mem2[0x0010]) == 0x82
    assert int(s2.P) & FLAG_N != 0
    assert int(s2.P) & FLAG_C == 0
    assert int(s2.cycles) == 5


def test_asl_absolute_x_no_page_cross_penalty():
    s = _state(PC=0x8000, X=0x10)
    # Base $12F5, +X=$1305 — page crossed but ASL is always 7 cycles for abs,X.
    mem = _make_memory({0x8000: 0x1E, 0x8001: 0xF5, 0x8002: 0x12, 0x1305: 0x40})
    s2, mem2 = step(s, mem)
    assert int(mem2[0x1305]) == 0x80
    assert int(s2.cycles) == 7  # NOT 8


# --------------------------------------------------------------------------- #
# LSR
# --------------------------------------------------------------------------- #

def test_lsr_accumulator_bit0_into_carry_and_n_always_clear():
    s = _state(PC=0x8000, A=0x01, P=FLAG_U | FLAG_N)  # N preset
    mem = _make_memory({0x8000: 0x4A})  # LSR A
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x00
    assert int(s2.P) & FLAG_C != 0
    assert int(s2.P) & FLAG_Z != 0
    assert int(s2.P) & FLAG_N == 0   # LSR always clears N


def test_lsr_accumulator_high_bit_drops():
    """LSR of 0xFF → 0x7F, C=1; N must be 0 even though input had bit 7 set."""
    s = _state(PC=0x8000, A=0xFF, P=FLAG_U)
    mem = _make_memory({0x8000: 0x4A})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x7F
    assert int(s2.P) & FLAG_C != 0
    assert int(s2.P) & FLAG_N == 0


def test_lsr_zero_page_x_writes_back():
    s = _state(PC=0x8000, X=0x05)
    mem = _make_memory({0x8000: 0x56, 0x8001: 0x10, 0x0015: 0x08})
    s2, mem2 = step(s, mem)
    assert int(mem2[0x0015]) == 0x04
    assert int(s2.cycles) == 6


# --------------------------------------------------------------------------- #
# ROL
# --------------------------------------------------------------------------- #

def test_rol_accumulator_brings_in_carry():
    s = _state(PC=0x8000, A=0x40, P=FLAG_U | FLAG_C)  # 0x40<<1=0x80; C in → bit 0
    mem = _make_memory({0x8000: 0x2A})  # ROL A
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x81  # 0x80 | 0x01
    assert int(s2.P) & FLAG_C == 0     # bit 7 was 0
    assert int(s2.P) & FLAG_N != 0


def test_rol_accumulator_bit7_into_carry():
    s = _state(PC=0x8000, A=0x80, P=FLAG_U)
    mem = _make_memory({0x8000: 0x2A})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x00
    assert int(s2.P) & FLAG_C != 0
    assert int(s2.P) & FLAG_Z != 0


def test_rol_absolute_writes_back():
    s = _state(PC=0x8000, P=FLAG_U | FLAG_C)
    mem = _make_memory({0x8000: 0x2E, 0x8001: 0x00, 0x8002: 0x20, 0x2000: 0x55})
    s2, mem2 = step(s, mem)
    # 0x55 << 1 = 0xAA, | C=1 → 0xAB
    assert int(mem2[0x2000]) == 0xAB
    assert int(s2.P) & FLAG_C == 0  # bit 7 of 0x55 was 0
    assert int(s2.P) & FLAG_N != 0
    assert int(s2.cycles) == 6


# --------------------------------------------------------------------------- #
# ROR
# --------------------------------------------------------------------------- #

def test_ror_accumulator_brings_in_carry_to_bit7():
    s = _state(PC=0x8000, A=0x02, P=FLAG_U | FLAG_C)  # 0x02>>1=0x01; C in → bit 7
    mem = _make_memory({0x8000: 0x6A})  # ROR A
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x81  # 0x01 | 0x80
    assert int(s2.P) & FLAG_C == 0   # bit 0 of input was 0
    assert int(s2.P) & FLAG_N != 0


def test_ror_accumulator_bit0_into_carry():
    s = _state(PC=0x8000, A=0x01, P=FLAG_U)
    mem = _make_memory({0x8000: 0x6A})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x00
    assert int(s2.P) & FLAG_C != 0
    assert int(s2.P) & FLAG_Z != 0
    assert int(s2.P) & FLAG_N == 0


def test_ror_zero_page_writes_back():
    s = _state(PC=0x8000, P=FLAG_U | FLAG_C)
    mem = _make_memory({0x8000: 0x66, 0x8001: 0x10, 0x0010: 0x02})
    s2, mem2 = step(s, mem)
    # 0x02 >> 1 = 0x01, C → bit 7 → 0x81
    assert int(mem2[0x0010]) == 0x81
    assert int(s2.P) & FLAG_C == 0
    assert int(s2.cycles) == 5
