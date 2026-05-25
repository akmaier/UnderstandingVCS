"""P1h SOFT-mode tests — common undocumented NMOS 6502 opcodes in the
differentiable SOFT path.

Mirrors `test_p1h_undocumented.py` (HARD) for the SOFT dispatch
table. Same 37-opcode subset (27 NOPs + 6 LAX + 4 SAX):
  - NOP variants ($1A/$3A/$5A/$7A/$DA/$FA implied; $80/$82/$89/$C2/$E2
    immediate; $04/$44/$64 zp; $14/$34/$54/$74/$D4/$F4 zp,X; $0C abs;
    $1C/$3C/$5C/$7C/$DC/$FC abs,X).
  - LAX (6 modes): load A and X from the same operand; N/Z from value.
  - SAX (4 modes): store (A AND X), no flag side-effects.

The "magic AND" LAX #imm ($AB) + RMW combos (DCP / ISC / RLA / RRA /
SLO / SRE) stay deferred (rare in ROMs, $AB unstable on real
hardware).
"""

import jax.numpy as jnp

from jaxtari.diff import (
    SoftCPUState,
    SOFT_SUPPORTED_OPCODES,
    initial_soft_bus,
    initial_soft_cpu_state,
    soft_step,
)


def _rom_with(opcodes_at_offset_0, size: int = 4096) -> jnp.ndarray:
    rom = jnp.zeros((size,), dtype=jnp.float32)
    for i, b in enumerate(opcodes_at_offset_0):
        rom = rom.at[i].set(jnp.float32(b))
    return rom


def _step(rom_bytes, *, A=0, X=0, Y=0, ram=None):
    """Helper — build a SOFT bus, run one instruction, return (state, bus).
    `ram` is an optional dict of {addr: byte} pre-seeds into bus.ram."""
    bus = initial_soft_bus(_rom_with(rom_bytes))
    if ram:
        ram_arr = bus.ram
        for addr, val in ram.items():
            ram_arr = ram_arr.at[addr & 0x7F].set(jnp.float32(val & 0xFF))
        bus = bus._replace(ram=ram_arr)
    state = initial_soft_cpu_state()
    state = state._replace(
        A=jnp.float32(A),
        X=jnp.float32(X),
        Y=jnp.float32(Y),
    )
    return soft_step(state, bus)


# --------------------------------------------------------------------------- #
# Coverage sanity
# --------------------------------------------------------------------------- #

def test_p1h_soft_dispatch_table_extended():
    """All 37 P1h opcodes are now in the SOFT dispatch table."""
    p1h = {
        # Implied 1-byte NOPs.
        0x1A, 0x3A, 0x5A, 0x7A, 0xDA, 0xFA,
        # Immediate NOPs.
        0x80, 0x82, 0x89, 0xC2, 0xE2,
        # Zero-page NOPs.
        0x04, 0x44, 0x64,
        # Zero-page,X NOPs.
        0x14, 0x34, 0x54, 0x74, 0xD4, 0xF4,
        # Absolute NOP.
        0x0C,
        # Absolute,X NOPs.
        0x1C, 0x3C, 0x5C, 0x7C, 0xDC, 0xFC,
        # LAX.
        0xA7, 0xB7, 0xAF, 0xBF, 0xA3, 0xB3,
        # SAX.
        0x87, 0x97, 0x8F, 0x83,
    }
    assert p1h <= SOFT_SUPPORTED_OPCODES
    assert len(p1h) == 37    # 27 NOPs + 6 LAX + 4 SAX


# --------------------------------------------------------------------------- #
# Undocumented NOPs — bytes consumed, no flag / register changes
# --------------------------------------------------------------------------- #

def test_soft_nop_implied_1byte_all_opcodes():
    """$1A/$3A/$5A/$7A/$DA/$FA — 1 byte, 2 cycles, no state change."""
    initial_pc = float(initial_soft_cpu_state().PC)
    for opcode in (0x1A, 0x3A, 0x5A, 0x7A, 0xDA, 0xFA):
        state, _ = _step([opcode])
        assert float(state.PC) == initial_pc + 1.0, f"opcode ${opcode:02x}"
        assert float(state.cycles) == 2.0, f"opcode ${opcode:02x}"
        assert float(state.A) == 0.0
        assert float(state.X) == 0.0


def test_soft_nop_imm_consumes_operand_byte():
    """$80/$82/$89/$C2/$E2 — 2 bytes, 2 cycles, immediate operand
    fetched but ignored."""
    initial_pc = float(initial_soft_cpu_state().PC)
    for opcode in (0x80, 0x82, 0x89, 0xC2, 0xE2):
        state, _ = _step([opcode, 0x77])
        assert float(state.PC) == initial_pc + 2.0, f"opcode ${opcode:02x}"
        assert float(state.cycles) == 2.0


def test_soft_nop_zp_3_cycles():
    """$04/$44/$64 — 2 bytes, 3 cycles."""
    initial_pc = float(initial_soft_cpu_state().PC)
    for opcode in (0x04, 0x44, 0x64):
        state, _ = _step([opcode, 0x40])
        assert float(state.PC) == initial_pc + 2.0, f"opcode ${opcode:02x}"
        assert float(state.cycles) == 3.0


def test_soft_nop_zp_x_4_cycles():
    """$14/$34/$54/$74/$D4/$F4 — 2 bytes, 4 cycles."""
    initial_pc = float(initial_soft_cpu_state().PC)
    for opcode in (0x14, 0x34, 0x54, 0x74, 0xD4, 0xF4):
        state, _ = _step([opcode, 0x40], X=0x05)
        assert float(state.PC) == initial_pc + 2.0, f"opcode ${opcode:02x}"
        assert float(state.cycles) == 4.0


def test_soft_nop_abs_4_cycles():
    """$0C — 3 bytes, 4 cycles."""
    initial_pc = float(initial_soft_cpu_state().PC)
    state, _ = _step([0x0C, 0x34, 0x12])
    assert float(state.PC) == initial_pc + 3.0
    assert float(state.cycles) == 4.0


def test_soft_nop_abs_x_4_cycles():
    """$1C/$3C/$5C/$7C/$DC/$FC — 3 bytes, 4 cycles."""
    initial_pc = float(initial_soft_cpu_state().PC)
    for opcode in (0x1C, 0x3C, 0x5C, 0x7C, 0xDC, 0xFC):
        state, _ = _step([opcode, 0x00, 0x10], X=0x10)
        assert float(state.PC) == initial_pc + 3.0, f"opcode ${opcode:02x}"
        assert float(state.cycles) == 4.0


# --------------------------------------------------------------------------- #
# LAX — load A and X from the same operand
# --------------------------------------------------------------------------- #

def test_soft_lax_zp():
    """$A7 — load A=X=mem[zp]. N/Z from value. 2 bytes, 3 cycles."""
    state, _ = _step([0xA7, 0x40], ram={0x40: 0x55})
    assert float(state.A) == 0x55
    assert float(state.X) == 0x55
    assert float(state.cycles) == 3.0
    # N=0 (high bit clear), Z=0 (nonzero).
    assert int(state.P) & 0x82 == 0


def test_soft_lax_zp_y():
    """$B7 — zero-page,Y addressing."""
    state, _ = _step([0xB7, 0x40], Y=0x05, ram={0x45: 0xCC})
    assert float(state.A) == 0xCC
    assert float(state.X) == 0xCC
    assert float(state.cycles) == 4.0
    # N=1 (high bit set), Z=0.
    assert int(state.P) & 0x80 != 0
    assert int(state.P) & 0x02 == 0


def test_soft_lax_abs():
    """$AF — absolute. Reading from ROM offset corresponds to bus addr."""
    # Put LAX abs $1003 in ROM at offset 0, with byte at ROM offset 3.
    # Bus addr $1003 → ROM offset 3 (initial_soft_bus maps cart at $1000).
    rom = _rom_with([0xAF, 0x03, 0x10, 0x99])
    bus = initial_soft_bus(rom)
    state = initial_soft_cpu_state()
    state, _ = soft_step(state, bus)
    assert float(state.A) == 0x99
    assert float(state.X) == 0x99
    assert float(state.cycles) == 4.0


def test_soft_lax_zero_sets_z_flag():
    """LAX of 0 → Z=1, N=0."""
    state, _ = _step([0xA7, 0x40], ram={0x40: 0x00})
    assert float(state.A) == 0.0
    assert float(state.X) == 0.0
    assert int(state.P) & 0x02 != 0           # Z set
    assert int(state.P) & 0x80 == 0           # N clear


# --------------------------------------------------------------------------- #
# SAX — store A AND X
# --------------------------------------------------------------------------- #

def test_soft_sax_zp():
    """$87 — store (A AND X) at zp address. No flag change."""
    state, bus = _step([0x87, 0x40], A=0xFC, X=0xAA)
    # 0xFC AND 0xAA = 0xA8.
    assert float(bus.ram[0x40 & 0x7F]) == 0xA8
    assert float(state.cycles) == 3.0
    # Flags unchanged from default.
    assert int(state.P) == int(initial_soft_cpu_state().P)


def test_soft_sax_zp_y():
    """$97 — store at zp,Y."""
    state, bus = _step([0x97, 0x40], A=0xF0, X=0x0F, Y=0x05)
    # 0xF0 AND 0x0F = 0x00 → stored at zp $45.
    assert float(bus.ram[0x45 & 0x7F]) == 0x00
    assert float(state.cycles) == 4.0


def test_soft_sax_abs_into_ram():
    """$8F — absolute store into RAM mirror addr."""
    # $0080 (canonical RAM) is at ram[0].
    state, bus = _step([0x8F, 0x80, 0x00], A=0xFF, X=0x33)
    # 0xFF AND 0x33 = 0x33.
    assert float(bus.ram[0]) == 0x33
    assert float(state.cycles) == 4.0


def test_soft_sax_preserves_flags():
    """A SAX of (A AND X) = 0 doesn't set the Z flag (no flag effects)."""
    state, _ = _step([0x87, 0x40], A=0x00, X=0xFF)
    # Flags must match the default — SAX never touches P.
    assert int(state.P) == int(initial_soft_cpu_state().P)
