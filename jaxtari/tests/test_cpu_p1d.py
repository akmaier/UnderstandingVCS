"""P1d tests — conditional branches, JMP, JSR, RTS."""

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
    s = initial_cpu_state()
    return s._replace(**{
        k: jnp.asarray(v).astype(getattr(s, k).dtype) for k, v in fields.items()
    })


# --------------------------------------------------------------------------- #
# Branches — taken vs not taken, page-cross extra cycle
# --------------------------------------------------------------------------- #

def test_beq_not_taken_when_z_clear():
    s = _state(PC=0x8000, P=FLAG_U)  # Z=0
    mem = _make_memory({0x8000: 0xF0, 0x8001: 0x10})  # BEQ +0x10
    s2, _ = step(s, mem)
    assert int(s2.PC) == 0x8002
    assert int(s2.cycles) == 2  # not taken: base only


def test_beq_taken_forward_no_page_cross():
    s = _state(PC=0x8000, P=FLAG_U | FLAG_Z)
    mem = _make_memory({0x8000: 0xF0, 0x8001: 0x10})  # BEQ +0x10
    s2, _ = step(s, mem)
    assert int(s2.PC) == 0x8012  # 0x8002 + 0x10
    assert int(s2.cycles) == 3   # 2 base + 1 taken


def test_beq_taken_forward_with_page_cross():
    # PC=0x80F0, opcode at 0x80F0, operand at 0x80F1, base = PC + 2 = 0x80F2.
    # offset = 0x10, target = 0x8102. Page 0x80 → 0x81, so this DOES cross.
    s = _state(PC=0x80F0, P=FLAG_U | FLAG_Z)
    mem = _make_memory({0x80F0: 0xF0, 0x80F1: 0x10})
    s2, _ = step(s, mem)
    assert int(s2.PC) == 0x8102
    assert int(s2.cycles) == 4  # 2 + 1 taken + 1 page cross


def test_bne_taken_when_z_clear():
    s = _state(PC=0x8000, P=FLAG_U)
    mem = _make_memory({0x8000: 0xD0, 0x8001: 0x05})
    s2, _ = step(s, mem)
    assert int(s2.PC) == 0x8007
    assert int(s2.cycles) == 3


def test_bmi_taken_when_n_set():
    s = _state(PC=0x8000, P=FLAG_U | FLAG_N)
    mem = _make_memory({0x8000: 0x30, 0x8001: 0x04})
    s2, _ = step(s, mem)
    assert int(s2.PC) == 0x8006


def test_bpl_not_taken_when_n_set():
    s = _state(PC=0x8000, P=FLAG_U | FLAG_N)
    mem = _make_memory({0x8000: 0x10, 0x8001: 0x04})
    s2, _ = step(s, mem)
    assert int(s2.PC) == 0x8002
    assert int(s2.cycles) == 2


def test_bcs_taken_when_c_set():
    s = _state(PC=0x8000, P=FLAG_U | FLAG_C)
    mem = _make_memory({0x8000: 0xB0, 0x8001: 0x02})
    s2, _ = step(s, mem)
    assert int(s2.PC) == 0x8004


def test_bvc_taken_when_v_clear():
    s = _state(PC=0x8000, P=FLAG_U)
    mem = _make_memory({0x8000: 0x50, 0x8001: 0x02})
    s2, _ = step(s, mem)
    assert int(s2.PC) == 0x8004


# Fix the bogus backward-cross test by re-asserting correct expectation
def test_beq_taken_backward_within_page():
    s = _state(PC=0x8100, P=FLAG_U | FLAG_Z)
    mem = _make_memory({0x8100: 0xF0, 0x8101: 0xFE})  # offset -2 → target 0x8100
    s2, _ = step(s, mem)
    assert int(s2.PC) == 0x8100
    assert int(s2.cycles) == 3  # taken but no page cross (base 0x8102, target 0x8100 same page)


def test_beq_taken_backward_with_page_cross():
    # PC=0x8002, offset=-0x10 → base=0x8004, target=0x7FF4 → page cross (page 0x80 → 0x7F)
    s = _state(PC=0x8002, P=FLAG_U | FLAG_Z)
    mem = _make_memory({0x8002: 0xF0, 0x8003: 0xF0})  # 0xF0 = -16
    s2, _ = step(s, mem)
    assert int(s2.PC) == 0x7FF4
    assert int(s2.cycles) == 4


# --------------------------------------------------------------------------- #
# JMP
# --------------------------------------------------------------------------- #

def test_jmp_absolute():
    s = _state(PC=0x8000)
    mem = _make_memory({0x8000: 0x4C, 0x8001: 0x34, 0x8002: 0x12})
    s2, _ = step(s, mem)
    assert int(s2.PC) == 0x1234
    assert int(s2.cycles) == 3


def test_jmp_indirect_normal_pointer():
    s = _state(PC=0x8000)
    mem = _make_memory({
        0x8000: 0x6C, 0x8001: 0x00, 0x8002: 0x30,  # JMP ($3000)
        0x3000: 0xCD, 0x3001: 0xAB,                # → $ABCD
    })
    s2, _ = step(s, mem)
    assert int(s2.PC) == 0xABCD
    assert int(s2.cycles) == 5


def test_jmp_indirect_page_wrap_bug():
    """Pointer at $30FF reads low from $30FF and high from $3000 (NOT $3100)."""
    s = _state(PC=0x8000)
    mem = _make_memory({
        0x8000: 0x6C, 0x8001: 0xFF, 0x8002: 0x30,
        0x30FF: 0xCD,   # low byte
        0x3000: 0xAB,   # high byte (page-wrap bug; NOT $3100)
        0x3100: 0x99,   # what a CMOS 65C02 would have read (must NOT be used)
    })
    s2, _ = step(s, mem)
    assert int(s2.PC) == 0xABCD


# --------------------------------------------------------------------------- #
# JSR / RTS
# --------------------------------------------------------------------------- #

def test_jsr_pushes_return_address_and_jumps():
    s = _state(PC=0x8000, SP=0xFD)
    mem = _make_memory({0x8000: 0x20, 0x8001: 0x00, 0x8002: 0x30})  # JSR $3000
    s2, mem2 = step(s, mem)
    assert int(s2.PC) == 0x3000
    assert int(s2.SP) == 0xFB
    # Pushed PC+2 = 0x8002. High byte first → at 0x01FD; low → at 0x01FC.
    assert int(mem2[0x01FD]) == 0x80
    assert int(mem2[0x01FC]) == 0x02
    assert int(s2.cycles) == 6


def test_rts_pops_and_advances():
    s = _state(PC=0x3050, SP=0xFB)
    # Stack has return addr 0x8002 (pushed by a prior JSR)
    mem = _make_memory({
        0x3050: 0x60,
        0x01FC: 0x02,   # low
        0x01FD: 0x80,   # high
    })
    s2, _ = step(s, mem)
    assert int(s2.PC) == 0x8003   # return_addr + 1
    assert int(s2.SP) == 0xFD
    assert int(s2.cycles) == 6


def test_jsr_then_rts_round_trip():
    """JSR at 0x8000 to 0x3000; RTS at 0x3000 lands at 0x8003."""
    s = _state(PC=0x8000, SP=0xFD)
    mem = _make_memory({
        0x8000: 0x20, 0x8001: 0x00, 0x8002: 0x30,  # JSR $3000
        0x3000: 0x60,                              # RTS
    })
    s2, mem2 = step(s, mem)
    assert int(s2.PC) == 0x3000
    s3, _ = step(s2, mem2)
    assert int(s3.PC) == 0x8003
    assert int(s3.SP) == 0xFD  # back to original
