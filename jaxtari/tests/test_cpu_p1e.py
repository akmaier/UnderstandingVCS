"""P1e tests — stack push/pull (PHA/PHP/PLA/PLP), status-flag setters /
clearers (SEC/CLC/SEI/CLI/SED/CLD/CLV), and NOP."""

import jax.numpy as jnp

from jaxtari.cpu.m6502 import step
from jaxtari.cpu.tables import (
    FLAG_B,
    FLAG_C,
    FLAG_D,
    FLAG_I,
    FLAG_N,
    FLAG_U,
    FLAG_V,
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
# PHA / PLA
# --------------------------------------------------------------------------- #

def test_pha_pushes_a_and_decrements_sp():
    s = _state(PC=0x8000, A=0x42, SP=0xFD)
    mem = _make_memory({0x8000: 0x48})  # PHA
    s2, mem2 = step(s, mem)
    assert int(s2.SP) == 0xFC
    assert int(mem2[0x01FD]) == 0x42
    assert int(s2.PC) == 0x8001
    assert int(s2.cycles) == 3


def test_pla_sets_zn_and_increments_sp():
    s = _state(PC=0x8000, A=0x00, SP=0xFC)
    mem = _make_memory({0x8000: 0x68, 0x01FD: 0x80})  # PLA; stack has 0x80
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x80
    assert int(s2.SP) == 0xFD
    assert int(s2.P) & FLAG_N != 0
    assert int(s2.P) & FLAG_Z == 0
    assert int(s2.cycles) == 4


def test_pla_zero_sets_z():
    s = _state(PC=0x8000, A=0xFF, SP=0xFC)
    mem = _make_memory({0x8000: 0x68, 0x01FD: 0x00})
    s2, _ = step(s, mem)
    assert int(s2.A) == 0x00
    assert int(s2.P) & FLAG_Z != 0


def test_pha_pla_round_trip():
    s = _state(PC=0x8000, A=0x77, SP=0xFD)
    mem = _make_memory({0x8000: 0x48, 0x8001: 0xA9, 0x8002: 0x00, 0x8003: 0x68})
    # PHA, LDA #$00, PLA — should restore A=0x77
    s1, mem1 = step(s, mem)
    s2, _ = step(s1, mem1)
    s3, _ = step(s2, mem1)
    assert int(s3.A) == 0x77
    assert int(s3.SP) == 0xFD


# --------------------------------------------------------------------------- #
# PHP / PLP — B and U bits always set on push, always set on pull
# --------------------------------------------------------------------------- #

def test_php_pushes_p_with_b_and_u_set():
    s = _state(PC=0x8000, SP=0xFD, P=FLAG_U | FLAG_C)  # B not set in P
    mem = _make_memory({0x8000: 0x08})  # PHP
    s2, mem2 = step(s, mem)
    pushed = int(mem2[0x01FD])
    assert pushed & FLAG_B != 0
    assert pushed & FLAG_U != 0
    assert pushed & FLAG_C != 0
    assert int(s2.SP) == 0xFC
    assert int(s2.cycles) == 3


def test_plp_pulls_p_and_forces_b_and_u():
    s = _state(PC=0x8000, SP=0xFC, P=FLAG_U)
    # Stack has a byte with N and Z set, B and U cleared
    mem = _make_memory({0x8000: 0x28, 0x01FD: FLAG_N | FLAG_Z})
    s2, _ = step(s, mem)
    assert int(s2.P) & FLAG_N != 0
    assert int(s2.P) & FLAG_Z != 0
    assert int(s2.P) & FLAG_B != 0   # forced set
    assert int(s2.P) & FLAG_U != 0   # forced set
    assert int(s2.SP) == 0xFD
    assert int(s2.cycles) == 4


def test_php_plp_round_trip():
    """PHP then PLP should restore P (modulo the B/U force).

    Start with a P that has C, V, N set (and U as invariant). PHP pushes
    that; PLP pulls it back. The pulled P should equal the original | B | U.
    """
    original_p = FLAG_U | FLAG_C | FLAG_V | FLAG_N
    s = _state(PC=0x8000, SP=0xFD, P=original_p)
    mem = _make_memory({0x8000: 0x08, 0x8001: 0xA9, 0x8002: 0x00, 0x8003: 0x28})
    # PHP; LDA #$00 (which clobbers N and sets Z); PLP
    s1, mem1 = step(s, mem)
    s2, _ = step(s1, mem1)
    s3, _ = step(s2, mem1)
    assert int(s3.P) == original_p | FLAG_B  # B/U set; rest restored
    assert int(s3.SP) == 0xFD


# --------------------------------------------------------------------------- #
# Status-flag setters / clearers
# --------------------------------------------------------------------------- #

def test_sec_sets_carry():
    s = _state(PC=0x8000, P=FLAG_U)
    mem = _make_memory({0x8000: 0x38})  # SEC
    s2, _ = step(s, mem)
    assert int(s2.P) & FLAG_C != 0
    assert int(s2.cycles) == 2


def test_clc_clears_carry():
    s = _state(PC=0x8000, P=FLAG_U | FLAG_C)
    mem = _make_memory({0x8000: 0x18})  # CLC
    s2, _ = step(s, mem)
    assert int(s2.P) & FLAG_C == 0


def test_sei_sets_interrupt_disable():
    s = _state(PC=0x8000, P=FLAG_U)
    mem = _make_memory({0x8000: 0x78})
    s2, _ = step(s, mem)
    assert int(s2.P) & FLAG_I != 0


def test_cli_clears_interrupt_disable():
    s = _state(PC=0x8000, P=FLAG_U | FLAG_I)
    mem = _make_memory({0x8000: 0x58})
    s2, _ = step(s, mem)
    assert int(s2.P) & FLAG_I == 0


def test_sed_sets_decimal():
    s = _state(PC=0x8000, P=FLAG_U)
    mem = _make_memory({0x8000: 0xF8})
    s2, _ = step(s, mem)
    assert int(s2.P) & FLAG_D != 0


def test_cld_clears_decimal():
    s = _state(PC=0x8000, P=FLAG_U | FLAG_D)
    mem = _make_memory({0x8000: 0xD8})
    s2, _ = step(s, mem)
    assert int(s2.P) & FLAG_D == 0


def test_clv_clears_overflow():
    s = _state(PC=0x8000, P=FLAG_U | FLAG_V)
    mem = _make_memory({0x8000: 0xB8})
    s2, _ = step(s, mem)
    assert int(s2.P) & FLAG_V == 0


# --------------------------------------------------------------------------- #
# NOP
# --------------------------------------------------------------------------- #

def test_nop_advances_pc_and_cycles():
    s = _state(PC=0x8000, P=FLAG_U)
    mem = _make_memory({0x8000: 0xEA})  # NOP
    s2, _ = step(s, mem)
    assert int(s2.PC) == 0x8001
    assert int(s2.cycles) == 2
    assert int(s2.P) == FLAG_U  # untouched
