"""P1f tests — INC/DEC memory, INX/INY/DEX/DEY register inc/dec, BRK, RTI."""

import jax.numpy as jnp

from jaxtari.cpu.m6502 import step
from jaxtari.cpu.tables import (
    FLAG_B,
    FLAG_C,
    FLAG_I,
    FLAG_N,
    FLAG_U,
    FLAG_Z,
)
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
# INC / DEC memory
# --------------------------------------------------------------------------- #

def test_inc_zero_page_increments_and_writes_back():
    s = _state(PC=0x8000)
    mem = _make_memory({0x8000: 0xE6, 0x8001: 0x10, 0x0010: 0x41})
    s2, mem2 = step(s, mem)
    assert int(mem2[0x0010]) == 0x42
    assert int(s2.P) & FLAG_Z == 0
    assert int(s2.P) & FLAG_N == 0
    assert int(s2.cycles) == 5


def test_inc_wraps_from_ff_to_00_and_sets_z():
    s = _state(PC=0x8000)
    mem = _make_memory({0x8000: 0xE6, 0x8001: 0x10, 0x0010: 0xFF})
    s2, mem2 = step(s, mem)
    assert int(mem2[0x0010]) == 0x00
    assert int(s2.P) & FLAG_Z != 0
    assert int(s2.P) & FLAG_N == 0


def test_inc_absolute_x_sets_n():
    s = _state(PC=0x8000, X=0x01)
    # 0x7F → 0x80 → N set
    mem = _make_memory({0x8000: 0xFE, 0x8001: 0x00, 0x8002: 0x12, 0x1201: 0x7F})
    s2, mem2 = step(s, mem)
    assert int(mem2[0x1201]) == 0x80
    assert int(s2.P) & FLAG_N != 0
    assert int(s2.cycles) == 7  # abs,X for RMW is always 7 (no page-cross penalty needed)


def test_dec_zero_page():
    s = _state(PC=0x8000)
    mem = _make_memory({0x8000: 0xC6, 0x8001: 0x10, 0x0010: 0x01})
    s2, mem2 = step(s, mem)
    assert int(mem2[0x0010]) == 0x00
    assert int(s2.P) & FLAG_Z != 0


def test_dec_wraps_from_00_to_ff_and_sets_n():
    s = _state(PC=0x8000)
    mem = _make_memory({0x8000: 0xC6, 0x8001: 0x10, 0x0010: 0x00})
    s2, mem2 = step(s, mem)
    assert int(mem2[0x0010]) == 0xFF
    assert int(s2.P) & FLAG_N != 0


# --------------------------------------------------------------------------- #
# INX / INY / DEX / DEY
# --------------------------------------------------------------------------- #

def test_inx_normal():
    s = _state(PC=0x8000, X=0x05)
    mem = _make_memory({0x8000: 0xE8})
    s2, _ = step(s, mem)
    assert int(s2.X) == 0x06
    assert int(s2.cycles) == 2


def test_inx_wraps_and_sets_z():
    s = _state(PC=0x8000, X=0xFF)
    mem = _make_memory({0x8000: 0xE8})
    s2, _ = step(s, mem)
    assert int(s2.X) == 0x00
    assert int(s2.P) & FLAG_Z != 0


def test_iny_sets_n():
    s = _state(PC=0x8000, Y=0x7F)
    mem = _make_memory({0x8000: 0xC8})
    s2, _ = step(s, mem)
    assert int(s2.Y) == 0x80
    assert int(s2.P) & FLAG_N != 0


def test_dex_wraps_and_sets_n():
    s = _state(PC=0x8000, X=0x00)
    mem = _make_memory({0x8000: 0xCA})
    s2, _ = step(s, mem)
    assert int(s2.X) == 0xFF
    assert int(s2.P) & FLAG_N != 0


def test_dey_to_zero_sets_z():
    s = _state(PC=0x8000, Y=0x01)
    mem = _make_memory({0x8000: 0x88})
    s2, _ = step(s, mem)
    assert int(s2.Y) == 0x00
    assert int(s2.P) & FLAG_Z != 0


# --------------------------------------------------------------------------- #
# BRK / RTI
# --------------------------------------------------------------------------- #

def test_brk_pushes_pc_plus_2_and_jumps_via_vector():
    s = _state(PC=0x8000, SP=0xFD, P=FLAG_U)
    mem = _make_memory({
        0x8000: 0x00,           # BRK
        0xFFFE: 0x34, 0xFFFF: 0x12,  # IRQ vector → $1234
    })
    s2, mem2 = step(s, mem)
    # Return address = PC + 2 = 0x8002, pushed high then low
    assert int(mem2[0x01FD]) == 0x80
    assert int(mem2[0x01FC]) == 0x02
    # Pushed P has B and U set
    pushed_p = int(mem2[0x01FB])
    assert pushed_p & FLAG_B != 0
    assert pushed_p & FLAG_U != 0
    # I flag set in current P
    assert int(s2.P) & FLAG_I != 0
    # PC loaded from vector
    assert int(s2.PC) == 0x1234
    # SP decremented by 3
    assert int(s2.SP) == 0xFA
    assert int(s2.cycles) == 7


def test_rti_pops_p_then_pc_no_plus_1():
    """Unlike RTS, RTI pops the exact PC (no +1)."""
    s = _state(PC=0x1234, SP=0xFA)
    # Stack: byte at SP+1 is P (P|B|U|C); then PC low at SP+2; high at SP+3.
    mem = _make_memory({
        0x1234: 0x40,
        0x01FB: FLAG_B | FLAG_U | FLAG_C,   # popped P
        0x01FC: 0x02,                       # popped PC low
        0x01FD: 0x80,                       # popped PC high
    })
    s2, _ = step(s, mem)
    assert int(s2.PC) == 0x8002       # NOT 0x8003
    assert int(s2.P) & FLAG_C != 0
    assert int(s2.P) & FLAG_B != 0
    assert int(s2.P) & FLAG_U != 0
    assert int(s2.SP) == 0xFD
    assert int(s2.cycles) == 6


def test_brk_then_rti_round_trip():
    """BRK at $8000 with IRQ vector → $1234; RTI at $1234 lands at $8002."""
    s = _state(PC=0x8000, SP=0xFD, P=FLAG_U | FLAG_C)
    mem = _make_memory({
        0x8000: 0x00,
        0x1234: 0x40,
        0xFFFE: 0x34, 0xFFFF: 0x12,
    })
    s1, mem1 = step(s, mem)
    assert int(s1.PC) == 0x1234
    s2, _ = step(s1, mem1)
    assert int(s2.PC) == 0x8002
    assert int(s2.SP) == 0xFD
    assert int(s2.P) & FLAG_C != 0  # restored
