"""P1h tests — common undocumented NMOS 6502 opcodes.

Adds the well-behaved subset:
  - NOP variants ($1A/$3A/$5A/$7A/$DA/$FA implied; $80/$82/$89/$C2/$E2
    immediate; $04/$44/$64 zp; $14/$34/$54/$74/$D4/$F4 zp,X; $0C abs;
    $1C/$3C/$5C/$7C/$DC/$FC abs,X).
  - LAX: load A and X from the same operand.
  - SAX: store A AND X (no flag side-effects).

The "magic AND" LAX #imm ($AB) and the RMW combos (DCP / ISC / RLA /
RRA / SLO / SRE) stay deferred. They're rare in real ROMs and
$AB-style instability isn't worth modelling.
"""

import jax.numpy as jnp

from jaxtari.cpu.tables import FLAG_N, FLAG_U, FLAG_Z
from jaxtari.cpu.m6502 import _step_inner
from jaxtari.types import CPUState


def _state(*, PC=0x8000, A=0, X=0, Y=0, SP=0xFD, P=0x34):
    return CPUState(
        A=jnp.uint8(A), X=jnp.uint8(X), Y=jnp.uint8(Y),
        SP=jnp.uint8(SP), P=jnp.uint8(P),
        PC=jnp.uint16(PC), cycles=jnp.uint64(0),
    )


def _memory(image: dict[int, int]) -> jnp.ndarray:
    mem = jnp.zeros((1 << 16,), dtype=jnp.uint8)
    for addr, val in image.items():
        mem = mem.at[addr & 0xFFFF].set(jnp.uint8(val & 0xFF))
    return mem


# --------------------------------------------------------------------------- #
# Undocumented NOPs — bytes consumed, no other side effects
# --------------------------------------------------------------------------- #

def test_unofficial_nop_1a_implied():
    """$1A — implied, 1 byte, 2 cycles, no register / flag changes."""
    state = _state()
    mem = _memory({0x8000: 0x1A})
    new_state, _ = _step_inner(state, mem)
    assert int(new_state.PC) == 0x8001
    assert int(new_state.cycles) == 2
    assert int(new_state.A) == 0 and int(new_state.X) == 0


def test_unofficial_nop_3a_5a_7a_da_fa_implied():
    """Same shape as $1A — exercise the other one-byte NOPs."""
    for opcode in (0x3A, 0x5A, 0x7A, 0xDA, 0xFA):
        state = _state()
        mem = _memory({0x8000: opcode})
        new_state, _ = _step_inner(state, mem)
        assert int(new_state.PC) == 0x8001, f"opcode ${opcode:02x}"
        assert int(new_state.cycles) == 2


def test_unofficial_nop_80_imm_consumes_operand():
    """$80 #imm — 2 bytes, 2 cycles, operand discarded."""
    state = _state()
    mem = _memory({0x8000: 0x80, 0x8001: 0xFF})
    new_state, _ = _step_inner(state, mem)
    assert int(new_state.PC) == 0x8002
    assert int(new_state.cycles) == 2


def test_unofficial_nop_04_zp_consumes_operand():
    """$04 zp — 2 bytes, 3 cycles, operand byte dummy-read."""
    state = _state()
    mem = _memory({0x8000: 0x04, 0x8001: 0x42, 0x0042: 0x99})
    new_state, mem2 = _step_inner(state, mem)
    assert int(new_state.PC) == 0x8002
    assert int(new_state.cycles) == 3
    # Memory unchanged (NOP doesn't store).
    assert int(mem2[0x0042]) == 0x99


def test_unofficial_nop_14_zp_x():
    state = _state(X=0x10)
    mem = _memory({0x8000: 0x14, 0x8001: 0x05})    # NOP $05,X → reads $15
    new_state, _ = _step_inner(state, mem)
    assert int(new_state.PC) == 0x8002
    assert int(new_state.cycles) == 4


def test_unofficial_nop_0c_abs():
    state = _state()
    mem = _memory({0x8000: 0x0C, 0x8001: 0x34, 0x8002: 0x12})
    new_state, _ = _step_inner(state, mem)
    assert int(new_state.PC) == 0x8003
    assert int(new_state.cycles) == 4


def test_unofficial_nop_1c_abs_x_no_page_cross():
    state = _state(X=0x10)
    mem = _memory({0x8000: 0x1C, 0x8001: 0x00, 0x8002: 0x12})
    new_state, _ = _step_inner(state, mem)
    assert int(new_state.PC) == 0x8003
    assert int(new_state.cycles) == 4


def test_unofficial_nop_1c_abs_x_page_cross_adds_cycle():
    """$1C reading $12F5+X with X=0x10 → $1305 — crosses the page."""
    state = _state(X=0x10)
    mem = _memory({0x8000: 0x1C, 0x8001: 0xF5, 0x8002: 0x12})
    new_state, _ = _step_inner(state, mem)
    assert int(new_state.PC) == 0x8003
    assert int(new_state.cycles) == 5      # 4 + 1 page-cross penalty


def test_unofficial_nop_does_not_touch_flags():
    """The P register must come back unchanged."""
    state = _state(P=FLAG_U | FLAG_N)
    mem = _memory({0x8000: 0x1A})
    new_state, _ = _step_inner(state, mem)
    assert int(new_state.P) == int(state.P)


# --------------------------------------------------------------------------- #
# LAX — load A AND X from operand
# --------------------------------------------------------------------------- #

def test_lax_zp_loads_both_a_and_x():
    state = _state()
    mem = _memory({0x8000: 0xA7, 0x8001: 0x42, 0x0042: 0x77})
    new_state, _ = _step_inner(state, mem)
    assert int(new_state.A) == 0x77
    assert int(new_state.X) == 0x77
    assert int(new_state.PC) == 0x8002


def test_lax_sets_n_flag_on_negative_value():
    state = _state()
    mem = _memory({0x8000: 0xA7, 0x8001: 0x42, 0x0042: 0x80})
    new_state, _ = _step_inner(state, mem)
    assert int(new_state.A) == 0x80
    assert int(new_state.X) == 0x80
    assert (int(new_state.P) & FLAG_N) != 0


def test_lax_sets_z_flag_on_zero_value():
    state = _state(A=0xFF, X=0xFF)
    mem = _memory({0x8000: 0xA7, 0x8001: 0x42, 0x0042: 0x00})
    new_state, _ = _step_inner(state, mem)
    assert int(new_state.A) == 0x00
    assert int(new_state.X) == 0x00
    assert (int(new_state.P) & FLAG_Z) != 0


def test_lax_abs():
    state = _state()
    mem = _memory({0x8000: 0xAF, 0x8001: 0x00, 0x8002: 0x20, 0x2000: 0x55})
    new_state, _ = _step_inner(state, mem)
    assert int(new_state.A) == 0x55
    assert int(new_state.X) == 0x55
    assert int(new_state.PC) == 0x8003


def test_lax_abs_y_with_page_cross():
    """LAX $12F5,Y with Y=0x10 → $1305 (page cross). 4 + 1 cycles."""
    state = _state(Y=0x10)
    mem = _memory({
        0x8000: 0xBF, 0x8001: 0xF5, 0x8002: 0x12,
        0x1305: 0x66,
    })
    new_state, _ = _step_inner(state, mem)
    assert int(new_state.A) == 0x66
    assert int(new_state.X) == 0x66


def test_lax_ind_y():
    """LAX ($zp),Y."""
    state = _state(Y=0x01)
    mem = _memory({
        0x8000: 0xB3, 0x8001: 0x10,
        0x0010: 0x00, 0x0011: 0x12,
        0x1201: 0x42,
    })
    new_state, _ = _step_inner(state, mem)
    assert int(new_state.A) == 0x42
    assert int(new_state.X) == 0x42


# --------------------------------------------------------------------------- #
# SAX — store A AND X, no flag side-effects
# --------------------------------------------------------------------------- #

def test_sax_zp_stores_a_and_x():
    state = _state(A=0xF0, X=0x0F)
    mem = _memory({0x8000: 0x87, 0x8001: 0x42})
    new_state, mem2 = _step_inner(state, mem)
    # A AND X = 0xF0 & 0x0F = 0x00. SAX stores that.
    assert int(mem2[0x0042]) == 0x00


def test_sax_zp_actual_intersection():
    state = _state(A=0xFF, X=0x55)
    mem = _memory({0x8000: 0x87, 0x8001: 0x42})
    new_state, mem2 = _step_inner(state, mem)
    assert int(mem2[0x0042]) == 0x55             # 0xFF & 0x55


def test_sax_does_not_touch_flags():
    state = _state(A=0xFF, X=0x80, P=FLAG_U | FLAG_N | FLAG_Z)
    mem = _memory({0x8000: 0x87, 0x8001: 0x42})
    new_state, _ = _step_inner(state, mem)
    assert int(new_state.P) == int(state.P)


def test_sax_abs():
    state = _state(A=0xFF, X=0x42)
    mem = _memory({0x8000: 0x8F, 0x8001: 0x00, 0x8002: 0x20})
    new_state, mem2 = _step_inner(state, mem)
    assert int(mem2[0x2000]) == 0x42
    assert int(new_state.PC) == 0x8003


def test_sax_zp_y():
    state = _state(A=0xFF, X=0x33, Y=0x05)
    mem = _memory({0x8000: 0x97, 0x8001: 0x10})       # SAX $10,Y → $15
    new_state, mem2 = _step_inner(state, mem)
    assert int(mem2[0x0015]) == 0x33


def test_sax_ind_x():
    state = _state(A=0xFF, X=0x04)
    mem = _memory({
        0x8000: 0x83, 0x8001: 0xFE,
        0x0002: 0x34, 0x0003: 0x12,                  # X-indexed zp pointer
    })
    new_state, mem2 = _step_inner(state, mem)
    # A AND X = 0xFF & 0x04 = 0x04. Stored at $1234.
    assert int(mem2[0x1234]) == 0x04
