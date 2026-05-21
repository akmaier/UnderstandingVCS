"""P3a tests — TIA register file, scanline / frame timing, WSYNC stall.

Playfield rendering, sprites, missiles, ball, collisions, and framebuffer
output land in subsequent P3 sub-phases.
"""

import jax.numpy as jnp

from jaxtari.bus import Bus, initial_bus, peek, poke
from jaxtari.cpu.m6502 import step
from jaxtari.cpu.tables import FLAG_U
from jaxtari.tia.system import (
    NTSC_CPU_CYCLES_PER_SCANLINE,
    NTSC_SCANLINES_PER_FRAME,
    NUM_REGISTERS,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TIAState,
    initial_tia_state,
    tia_advance,
    tia_apply_wsync,
    tia_peek,
    tia_poke,
    W_COLUBK,
    W_WSYNC,
)
from jaxtari.types import CPUState, initial_cpu_state


def _state(**fields) -> CPUState:
    s = initial_cpu_state()
    return s._replace(**{
        k: jnp.asarray(v).astype(getattr(s, k).dtype) for k, v in fields.items()
    })


# --------------------------------------------------------------------------- #
# TIAState alone
# --------------------------------------------------------------------------- #

def test_initial_tia_state_is_zeroed():
    tia = initial_tia_state()
    assert tia.registers.shape == (NUM_REGISTERS,)
    assert int(tia.registers.sum()) == 0
    assert tia.scanline_cycle == 0
    assert tia.scanline == 0
    assert tia.frame == 0
    assert tia.wsync_pending is False
    assert tia.framebuffer.shape == (SCREEN_HEIGHT, SCREEN_WIDTH)
    assert int(tia.framebuffer.sum()) == 0


def test_tia_poke_stores_byte_in_register_file():
    tia = initial_tia_state()
    new_tia = tia_poke(tia, W_COLUBK, 0x1C)
    assert int(new_tia.registers[W_COLUBK]) == 0x1C
    # Original is unchanged (NamedTuple immutability).
    assert int(tia.registers[W_COLUBK]) == 0


def test_tia_poke_wsync_sets_pending_flag():
    tia = initial_tia_state()
    new_tia = tia_poke(tia, W_WSYNC, 0x00)
    assert new_tia.wsync_pending is True
    # Original is unchanged.
    assert tia.wsync_pending is False


def test_tia_poke_register_index_takes_low_6_bits_of_addr():
    """TIA decodes only A0–A5 of the bus, so address $42 (=0x40 | 0x02)
    aliases to register $02 (WSYNC)."""
    tia = initial_tia_state()
    new_tia = tia_poke(tia, 0x42, 0xFF)
    assert new_tia.wsync_pending is True


def test_tia_peek_collisions_default_to_zero():
    """Collision latches (\$30-\$37) start zero. INPT* now have real
    defaults (P6) — covered in test_tia_inputs.py."""
    tia = initial_tia_state()
    for addr in (0x00, 0x07, 0x30, 0x37, 0x3E, 0x3F):
        assert tia_peek(tia, addr) == 0


# --------------------------------------------------------------------------- #
# Timing — tia_advance and tia_apply_wsync
# --------------------------------------------------------------------------- #

def test_tia_advance_within_scanline():
    tia = initial_tia_state()
    new_tia = tia_advance(tia, 30)
    assert new_tia.scanline_cycle == 30
    assert new_tia.scanline == 0
    assert new_tia.frame == 0


def test_tia_advance_crosses_scanline_boundary():
    tia = initial_tia_state()
    new_tia = tia_advance(tia, NTSC_CPU_CYCLES_PER_SCANLINE)
    assert new_tia.scanline_cycle == 0
    assert new_tia.scanline == 1
    assert new_tia.frame == 0


def test_tia_advance_crosses_multiple_scanlines():
    tia = initial_tia_state()
    n = NTSC_CPU_CYCLES_PER_SCANLINE * 5 + 10
    new_tia = tia_advance(tia, n)
    assert new_tia.scanline_cycle == 10
    assert new_tia.scanline == 5
    assert new_tia.frame == 0


def test_tia_advance_crosses_frame_boundary():
    """One full frame is 262 * 76 = 19,912 CPU cycles."""
    tia = initial_tia_state()
    n = NTSC_CPU_CYCLES_PER_SCANLINE * NTSC_SCANLINES_PER_FRAME
    new_tia = tia_advance(tia, n)
    assert new_tia.scanline_cycle == 0
    assert new_tia.scanline == 0
    assert new_tia.frame == 1


def test_tia_apply_wsync_noop_when_no_pending():
    tia = initial_tia_state()
    stall, new_tia = tia_apply_wsync(tia)
    assert stall == 0
    assert new_tia is tia or new_tia == tia


def test_tia_apply_wsync_stalls_to_next_scanline_boundary():
    tia = initial_tia_state()._replace(scanline_cycle=20, wsync_pending=True)
    stall, new_tia = tia_apply_wsync(tia)
    # 76 - 20 = 56 cycles to next scanline boundary
    assert stall == 56
    assert new_tia.scanline_cycle == 0
    assert new_tia.scanline == 1
    assert new_tia.wsync_pending is False


def test_tia_apply_wsync_at_boundary_is_zero_cycle():
    """If we're already at scanline_cycle=0, WSYNC is a no-op stall."""
    tia = initial_tia_state()._replace(scanline_cycle=0, wsync_pending=True)
    stall, new_tia = tia_apply_wsync(tia)
    assert stall == 0
    assert new_tia.scanline == 0           # didn't advance
    assert new_tia.wsync_pending is False


# --------------------------------------------------------------------------- #
# Bus integration — TIA writes via the bus
# --------------------------------------------------------------------------- #

def test_bus_tia_write_records_in_register_file():
    bus = initial_bus()
    bus2 = poke(bus, 0x0009, 0x42)        # COLUBK = 0x42
    assert int(bus2.tia.registers[W_COLUBK]) == 0x42
    # Original bus's TIA is unchanged.
    assert int(bus.tia.registers[W_COLUBK]) == 0


def test_bus_tia_write_to_wsync_sets_pending_flag():
    bus = initial_bus()
    bus2 = poke(bus, 0x0002, 0x00)        # WSYNC
    assert bus2.tia.wsync_pending is True


# --------------------------------------------------------------------------- #
# CPU integration — step() runs TIA post-processing on a Bus
# --------------------------------------------------------------------------- #

def test_step_advances_tia_scanline_cycle_by_instruction_cycles():
    """Each step on a Bus should advance tia.scanline_cycle by the cycles
    consumed by the instruction."""
    rom = jnp.zeros((4096,), dtype=jnp.uint8)
    rom = rom.at[0].set(jnp.uint8(0xEA))   # NOP (2 cycles)
    bus = initial_bus(rom)
    s = _state(PC=0xF000)
    s2, bus2 = step(s, bus)
    assert bus2.tia.scanline_cycle == 2
    assert bus2.tia.scanline == 0


def test_step_crosses_scanline_when_enough_cycles_accumulate():
    """Run enough NOPs (2 cycles each) to cross a scanline boundary."""
    rom = jnp.full((4096,), 0xEA, dtype=jnp.uint8)  # all NOPs
    bus = initial_bus(rom)
    s = _state(PC=0xF000)
    # 76 / 2 = 38 NOPs to exactly hit one scanline boundary.
    for _ in range(38):
        s, bus = step(s, bus)
    assert bus.tia.scanline_cycle == 0
    assert bus.tia.scanline == 1


def test_step_sta_wsync_stalls_to_next_scanline():
    """STA WSYNC at mid-scanline stalls CPU to the next scanline boundary.

    Layout: LDA #$00 (2 cyc), STA WSYNC (3 cyc, lands at scanline_cycle=5),
    expected stall = 76-5 = 71 → state.cycles increases by 5+71=76 total.
    """
    rom = jnp.zeros((4096,), dtype=jnp.uint8)
    rom = rom.at[0].set(jnp.uint8(0xA9))   # LDA #$00
    rom = rom.at[1].set(jnp.uint8(0x00))
    rom = rom.at[2].set(jnp.uint8(0x85))   # STA $02 (WSYNC, zero page)
    rom = rom.at[3].set(jnp.uint8(0x02))
    bus = initial_bus(rom)
    s = _state(PC=0xF000)
    s, bus = step(s, bus)                  # LDA #$00 — 2 cycles
    assert int(s.cycles) == 2
    s, bus = step(s, bus)                  # STA WSYNC — 3 cyc + stall to next scanline
    assert int(s.cycles) == NTSC_CPU_CYCLES_PER_SCANLINE   # 76: one full scanline
    assert bus.tia.scanline == 1
    assert bus.tia.scanline_cycle == 0
    assert bus.tia.wsync_pending is False  # consumed


def test_step_with_flat_memory_does_not_run_tia_postprocess():
    """The TIA timing path runs only when memory is a Bus — flat-memory
    tests (the entire P1 suite) must not pay TIA overhead and must not see
    any TIA-related state changes."""
    mem = jnp.zeros((1 << 16,), dtype=jnp.uint8).at[0].set(jnp.uint8(0xEA))  # NOP at 0
    s = _state(PC=0x0000)
    s2, mem2 = step(s, mem)
    assert int(s2.cycles) == 2
    # mem2 is still a flat ndarray, not a Bus.
    assert not isinstance(mem2, Bus)
