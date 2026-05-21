"""P1a tests — load / store / transfer opcodes against hand-built memory layouts.

Each test programs a tiny memory image, calls `step` once, and checks the
register file, the memory, the PC, the cycle counter, and the relevant flag
bits. The xitari-trace-based conformance harness comes in a later phase
(PORTING_PLAN.md §4) — these are unit-level sanity tests until then.
"""

import jax.numpy as jnp

from jaxtari.cpu.m6502 import step
from jaxtari.cpu.tables import FLAG_N, FLAG_U, FLAG_Z
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
# LDA addressing-mode coverage
# --------------------------------------------------------------------------- #

def test_lda_immediate_sets_a_and_clears_zn():
    s = _state(PC=0x8000)
    mem = _make_memory({0x8000: 0xA9, 0x8001: 0x42})  # LDA #$42
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x42
    assert int(s2.PC) == 0x8002
    assert int(s2.cycles) == 2
    assert int(s2.P) & FLAG_Z == 0
    assert int(s2.P) & FLAG_N == 0
    assert int(s2.P) & FLAG_U != 0  # bit 5 stays high


def test_lda_immediate_zero_sets_z_and_not_n():
    s = _state(PC=0x8000)
    mem = _make_memory({0x8000: 0xA9, 0x8001: 0x00})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0
    assert int(s2.P) & FLAG_Z != 0
    assert int(s2.P) & FLAG_N == 0


def test_lda_immediate_negative_sets_n_and_not_z():
    s = _state(PC=0x8000)
    mem = _make_memory({0x8000: 0xA9, 0x8001: 0x80})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x80
    assert int(s2.P) & FLAG_N != 0
    assert int(s2.P) & FLAG_Z == 0


def test_lda_zero_page():
    s = _state(PC=0x8000)
    mem = _make_memory({0x8000: 0xA5, 0x8001: 0x33, 0x0033: 0x77})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x77
    assert int(s2.PC) == 0x8002
    assert int(s2.cycles) == 3


def test_lda_zero_page_x_wraps_at_0xff():
    s = _state(PC=0x8000, X=0x10)
    mem = _make_memory({0x8000: 0xB5, 0x8001: 0xF5, 0x0005: 0xAB})  # 0xF5 + 0x10 wraps to 0x05
    s2, _ = step(s, mem)
    assert int(s2.A) == 0xAB
    assert int(s2.cycles) == 4


def test_lda_absolute_no_page_cross():
    s = _state(PC=0x8000)
    mem = _make_memory({0x8000: 0xAD, 0x8001: 0x34, 0x8002: 0x12, 0x1234: 0x99})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x99
    assert int(s2.PC) == 0x8003
    assert int(s2.cycles) == 4


def test_lda_absolute_x_no_page_cross():
    s = _state(PC=0x8000, X=0x05)
    mem = _make_memory({0x8000: 0xBD, 0x8001: 0x00, 0x8002: 0x12, 0x1205: 0x44})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x44
    assert int(s2.cycles) == 4


def test_lda_absolute_x_page_cross_adds_cycle():
    s = _state(PC=0x8000, X=0x10)
    mem = _make_memory({0x8000: 0xBD, 0x8001: 0xF5, 0x8002: 0x12, 0x1305: 0x55})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x55
    assert int(s2.cycles) == 5  # 4 + 1 page cross


def test_lda_absolute_y_page_cross_adds_cycle():
    s = _state(PC=0x8000, Y=0x80)
    mem = _make_memory({0x8000: 0xB9, 0x8001: 0xF0, 0x8002: 0x20, 0x2170: 0x33})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x33
    assert int(s2.cycles) == 5


def test_lda_indirect_x_with_zp_wrap():
    s = _state(PC=0x8000, X=0x04)
    # operand=0xFE, +X=0x02 (wraps). Pointer at $02..$03 = $34, $12 -> $1234 -> $66.
    mem = _make_memory({
        0x8000: 0xA1, 0x8001: 0xFE,
        0x0002: 0x34, 0x0003: 0x12,
        0x1234: 0x66,
    })
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x66
    assert int(s2.cycles) == 6


def test_lda_indirect_y_with_zp_pointer_wrap_and_page_cross():
    s = _state(PC=0x8000, Y=0x10)
    # zp=0xFF, lo=mem[$FF]=$F5, hi=mem[$00]=$12 (zp wrap). base=$12F5; +Y=$1305.
    mem = _make_memory({
        0x8000: 0xB1, 0x8001: 0xFF,
        0x00FF: 0xF5, 0x0000: 0x12,
        0x1305: 0x88,
    })
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x88
    assert int(s2.cycles) == 6  # 5 base + 1 page cross


# --------------------------------------------------------------------------- #
# LDX / LDY
# --------------------------------------------------------------------------- #

def test_ldx_immediate():
    s = _state(PC=0x8000)
    mem = _make_memory({0x8000: 0xA2, 0x8001: 0xFE})
    s2, _ = step(s, mem)
    assert int(s2.X) == 0xFE
    assert int(s2.P) & FLAG_N != 0


def test_ldx_zero_page_y():
    s = _state(PC=0x8000, Y=0x02)
    mem = _make_memory({0x8000: 0xB6, 0x8001: 0x10, 0x0012: 0x07})
    s2, _ = step(s, mem)
    assert int(s2.X) == 0x07
    assert int(s2.cycles) == 4


def test_ldy_absolute_x_no_page_cross():
    s = _state(PC=0x8000, X=0x01)
    mem = _make_memory({0x8000: 0xBC, 0x8001: 0x00, 0x8002: 0x12, 0x1201: 0x09})
    s2, _ = step(s, mem)
    assert int(s2.Y) == 0x09
    assert int(s2.cycles) == 4


# --------------------------------------------------------------------------- #
# STA / STX / STY — must NOT apply page-cross penalty even when one occurs.
# --------------------------------------------------------------------------- #

def test_sta_zero_page_writes_memory():
    s = _state(PC=0x8000, A=0x5A)
    mem = _make_memory({0x8000: 0x85, 0x8001: 0x42})
    s2, mem2 = step(s, mem)
    assert int(mem2[0x0042]) == 0x5A
    assert int(s2.cycles) == 3


def test_sta_absolute_x_no_page_cross_penalty():
    s = _state(PC=0x8000, A=0xCC, X=0x10)
    # Base $12F5, +X=$1305 — page crossed, but STA must still be exactly 5 cycles.
    mem = _make_memory({0x8000: 0x9D, 0x8001: 0xF5, 0x8002: 0x12})
    s2, mem2 = step(s, mem)
    assert int(mem2[0x1305]) == 0xCC
    assert int(s2.cycles) == 5  # NOT 6
    # STA does not touch flags.
    assert int(s2.P) == int(s.P)


def test_stx_zero_page_y():
    s = _state(PC=0x8000, X=0x77, Y=0x03)
    mem = _make_memory({0x8000: 0x96, 0x8001: 0x10})
    s2, mem2 = step(s, mem)
    assert int(mem2[0x0013]) == 0x77
    assert int(s2.cycles) == 4


def test_sty_absolute():
    s = _state(PC=0x8000, Y=0x42)
    mem = _make_memory({0x8000: 0x8C, 0x8001: 0x00, 0x8002: 0x20})
    s2, mem2 = step(s, mem)
    assert int(mem2[0x2000]) == 0x42
    assert int(s2.PC) == 0x8003


# --------------------------------------------------------------------------- #
# Transfers
# --------------------------------------------------------------------------- #

def test_tax_copies_a_and_sets_zn():
    s = _state(PC=0x8000, A=0x80)
    mem = _make_memory({0x8000: 0xAA})
    s2, _ = step(s, mem)
    assert int(s2.X) == 0x80
    assert int(s2.A) == 0x80
    assert int(s2.P) & FLAG_N != 0
    assert int(s2.cycles) == 2


def test_tay():
    s = _state(PC=0x8000, A=0x00)
    mem = _make_memory({0x8000: 0xA8})
    s2, _ = step(s, mem)
    assert int(s2.Y) == 0x00
    assert int(s2.P) & FLAG_Z != 0


def test_txa_tya_tsx():
    s = _state(PC=0x8000, X=0x01, Y=0x02, SP=0xF0)
    mem = _make_memory({0x8000: 0x8A, 0x8001: 0x98, 0x8002: 0xBA})
    s1, _ = step(s, mem)
    assert int(s1.A) == 0x01
    s2, _ = step(s1, mem)
    assert int(s2.A) == 0x02
    s3, _ = step(s2, mem)
    assert int(s3.X) == 0xF0
    assert int(s3.P) & FLAG_N != 0  # SP=0xF0 has high bit set


def test_txs_does_not_touch_flags():
    """TXS is the unique transfer that does not affect P."""
    s = _state(PC=0x8000, X=0x00, P=FLAG_U)  # everything cleared except hard-wired bit 5
    mem = _make_memory({0x8000: 0x9A})
    s2, _ = step(s, mem)
    assert int(s2.SP) == 0x00
    assert int(s2.P) == FLAG_U  # Z would have been set by any other transfer
