"""P3b tests — TIA playfield rendering (PF0/PF1/PF2 + CTRLPF mirror + COLUPF/COLUBK)."""

import jax.numpy as jnp

from jaxtari.bus import initial_bus, poke
from jaxtari.cpu.m6502 import step
from jaxtari.tia.system import (
    NTSC_CPU_CYCLES_PER_SCANLINE,
    W_COLUBK,
    W_COLUPF,
    W_CTRLPF,
    W_PF0,
    W_PF1,
    W_PF2,
    Y_START,
    _playfield_bits,
    initial_tia_state,
    render_playfield_scanline,
    tia_advance,
    tia_poke,
)
from jaxtari.types import CPUState, initial_cpu_state


def _state(**fields) -> CPUState:
    s = initial_cpu_state()
    return s._replace(**{
        k: jnp.asarray(v).astype(getattr(s, k).dtype) for k, v in fields.items()
    })


# --------------------------------------------------------------------------- #
# Bit layout — _playfield_bits
# --------------------------------------------------------------------------- #

def test_playfield_bits_all_zero():
    assert _playfield_bits(0, 0, 0) == [0] * 20


def test_playfield_bits_pf0_high_nibble_only_used():
    # PF0 = 0x0F → low nibble ignored, all 4 PF0 bits are 0.
    assert _playfield_bits(0x0F, 0, 0) == [0, 0, 0, 0] + [0] * 16
    # PF0 = 0xF0 → high nibble all set, first 4 playfield bits all 1.
    assert _playfield_bits(0xF0, 0, 0) == [1, 1, 1, 1] + [0] * 16


def test_playfield_bits_pf0_bit_order():
    # Only PF0 bit 4 (= 0x10) → playfield pixel 0 only.
    assert _playfield_bits(0x10, 0, 0)[0:4] == [1, 0, 0, 0]
    # Only PF0 bit 7 (= 0x80) → playfield pixel 3 only.
    assert _playfield_bits(0x80, 0, 0)[0:4] == [0, 0, 0, 1]


def test_playfield_bits_pf1_msb_first():
    # PF1 = 0x80 (bit 7) → playfield pixel 4 only.
    bits = _playfield_bits(0, 0x80, 0)
    assert bits[4] == 1
    assert sum(bits) == 1
    # PF1 = 0x01 (bit 0) → playfield pixel 11 only.
    bits = _playfield_bits(0, 0x01, 0)
    assert bits[11] == 1
    assert sum(bits) == 1


def test_playfield_bits_pf2_lsb_first():
    # PF2 = 0x01 (bit 0) → playfield pixel 12 only.
    bits = _playfield_bits(0, 0, 0x01)
    assert bits[12] == 1
    assert sum(bits) == 1
    # PF2 = 0x80 (bit 7) → playfield pixel 19 only (rightmost).
    bits = _playfield_bits(0, 0, 0x80)
    assert bits[19] == 1
    assert sum(bits) == 1


def test_playfield_bits_all_ones():
    bits = _playfield_bits(0xF0, 0xFF, 0xFF)
    assert bits == [1] * 20


# --------------------------------------------------------------------------- #
# Scanline rendering — render_playfield_scanline
# --------------------------------------------------------------------------- #

def _set_regs(tia, **fields):
    for name, value in fields.items():
        tia = tia_poke(tia, globals()[f"W_{name.upper()}"], value)
    return tia


def test_render_all_background():
    tia = initial_tia_state()
    tia = _set_regs(tia, colubk=0x1C, colupf=0x44)
    scanline = render_playfield_scanline(tia)
    assert scanline.shape == (160,)
    assert int(scanline.sum()) == 160 * 0x1C  # all background
    assert int((scanline == 0x1C).sum()) == 160


def test_render_all_playfield():
    tia = initial_tia_state()
    tia = _set_regs(tia, pf0=0xF0, pf1=0xFF, pf2=0xFF, colupf=0x44, colubk=0x1C)
    scanline = render_playfield_scanline(tia)
    assert int((scanline == 0x44).sum()) == 160


def test_render_left_pf0_bit4_lights_first_4_pixels():
    """PF0 bit 4 (=0x10) → playfield pixel 0 → 4 screen pixels at positions 0..3."""
    tia = initial_tia_state()
    tia = _set_regs(tia, pf0=0x10, colupf=0x42, colubk=0x00)
    scanline = render_playfield_scanline(tia)
    assert int(scanline[0]) == 0x42 and int(scanline[3]) == 0x42
    assert int(scanline[4]) == 0x00
    # All other pixels in left half are background
    for i in range(4, 80):
        assert int(scanline[i]) == 0x00


def test_render_right_half_repeated_when_ctrlpf_d0_clear():
    """CTRLPF.D0 = 0 → right half repeats left half (same playfield pattern)."""
    tia = initial_tia_state()
    tia = _set_regs(tia, pf0=0x10, ctrlpf=0x00, colupf=0x42, colubk=0x00)
    scanline = render_playfield_scanline(tia)
    # Left half: playfield pixel 0 at positions 0..3.
    # Right half (positions 80..159): same pattern repeated → 80..83 are colupf.
    assert int(scanline[80]) == 0x42 and int(scanline[83]) == 0x42
    assert int(scanline[84]) == 0x00


def test_render_right_half_mirrored_when_ctrlpf_d0_set():
    """CTRLPF.D0 = 1 → right half is the mirror image of the left half."""
    tia = initial_tia_state()
    tia = _set_regs(tia, pf0=0x10, ctrlpf=0x01, colupf=0x42, colubk=0x00)
    scanline = render_playfield_scanline(tia)
    # Left half: positions 0..3 lit (playfield pixel 0).
    # Mirrored right half: playfield pixel 0 is now the RIGHTMOST → positions 156..159.
    assert int(scanline[156]) == 0x42 and int(scanline[159]) == 0x42
    assert int(scanline[155]) == 0x00
    # Position 80..83 (start of right half = mirror of playfield pixel 19) should be background.
    assert int(scanline[80]) == 0x00


def test_render_full_known_pattern():
    """PF0=0xF0 PF1=0x00 PF2=0x00 + repeat → first 16 pixels are playfield,
    rest of left half is background; right half is same."""
    tia = initial_tia_state()
    tia = _set_regs(tia, pf0=0xF0, ctrlpf=0x00, colupf=0x42, colubk=0x00)
    scanline = render_playfield_scanline(tia)
    for i in range(16):
        assert int(scanline[i]) == 0x42, f"pixel {i}"
    for i in range(16, 80):
        assert int(scanline[i]) == 0x00, f"pixel {i}"
    for i in range(80, 96):
        assert int(scanline[i]) == 0x42, f"pixel {i}"
    for i in range(96, 160):
        assert int(scanline[i]) == 0x00, f"pixel {i}"


# --------------------------------------------------------------------------- #
# Framebuffer integration — tia_advance writes scanlines
# --------------------------------------------------------------------------- #

# Task #83 round 3 (2026-06-11) / test nudge (2026-06-12): `tia_advance`
# gates the framebuffer write on `tia.scanline >= Y_START` (= 34) to
# match xitari's `myClockStartDisplay` offset — pre-Y_START scanlines
# render nowhere. These tests historically started at scanline 0; they
# now start at Y_START so they exercise the framebuffer write at the
# first VISIBLE scanline (same nudge the jutari runtests got in #83).

def test_tia_advance_writes_scanline_on_boundary_cross():
    tia = initial_tia_state()
    tia = _set_regs(tia, pf0=0xF0, colupf=0x42, colubk=0x00)
    tia = tia._replace(scanline=Y_START)
    # Advance exactly one full scanline.
    tia = tia_advance(tia, NTSC_CPU_CYCLES_PER_SCANLINE)
    # Framebuffer row Y_START should now hold the playfield pattern.
    assert int(tia.framebuffer[Y_START, 0]) == 0x42
    assert int(tia.framebuffer[Y_START, 15]) == 0x42
    assert int(tia.framebuffer[Y_START, 16]) == 0x00
    # The next row hasn't been rendered yet.
    assert int(tia.framebuffer[Y_START + 1, 0]) == 0x00


def test_tia_advance_writes_multiple_scanlines():
    tia = initial_tia_state()
    tia = _set_regs(tia, pf0=0xF0, colupf=0x42, colubk=0x00)
    tia = tia._replace(scanline=Y_START)
    # Advance 5 scanlines.
    tia = tia_advance(tia, NTSC_CPU_CYCLES_PER_SCANLINE * 5)
    # All 5 rows should have the same pattern (registers didn't change).
    for line in range(Y_START, Y_START + 5):
        assert int(tia.framebuffer[line, 0]) == 0x42, f"row {line}"
        assert int(tia.framebuffer[line, 15]) == 0x42
        assert int(tia.framebuffer[line, 16]) == 0x00
    # The row after the advance is untouched.
    assert int(tia.framebuffer[Y_START + 5, 0]) == 0x00


def test_tia_advance_top_display_gate():
    """Task #83/#110: scanlines < Y_START (the top display gate)
    do NOT write to the framebuffer.

    Updated for task #110 (PAL): the framebuffer is now 312 rows so no NTSC
    scanline is off-screen via the buffer bound. The active gate is the
    Y_START top-of-display test (xitari `myClockStartDisplay = ... +
    228*myYStart`). Render a scanline at 10 (< Y_START=34); assert nothing
    landed in the buffer."""
    tia = initial_tia_state()
    tia = _set_regs(tia, pf0=0xF0, colupf=0x42, colubk=0x00)
    # Jump to scanline 10 (pre-display-window — < Y_START=34).
    tia = tia._replace(scanline=10)
    tia = tia_advance(tia, NTSC_CPU_CYCLES_PER_SCANLINE)
    # No row in the framebuffer should be touched (top display gate).
    assert int(tia.framebuffer.sum()) == 0


# --------------------------------------------------------------------------- #
# CPU+Bus → end-to-end: a tiny program writes playfield registers
# --------------------------------------------------------------------------- #

def test_program_writes_playfield_then_wsync_renders_scanline():
    """Program at $F000: LDA #$F0 / STA PF0 / LDA #$42 / STA COLUPF / STA WSYNC."""
    rom = jnp.zeros((4096,), dtype=jnp.uint8)
    program = [
        0xA9, 0xF0,        # LDA #$F0
        0x85, 0x0D,        # STA $0D (PF0)
        0xA9, 0x42,        # LDA #$42
        0x85, 0x08,        # STA $08 (COLUPF)
        0x85, 0x02,        # STA $02 (WSYNC)
    ]
    for i, b in enumerate(program):
        rom = rom.at[i].set(jnp.uint8(b))
    bus = initial_bus(rom)
    # Task #83 round 3: nudge tia.scanline to Y_START so the visible-
    # render gate fires (same nudge as the jutari runtests). The
    # program-driven render is otherwise identical.
    bus = bus._replace(tia=bus.tia._replace(scanline=Y_START))
    s = _state(PC=0xF000)
    for _ in range(5):
        s, bus = step(s, bus)
    # WSYNC stalls to next scanline → tia rendered the just-finished line.
    assert int(bus.tia.framebuffer[Y_START, 0]) == 0x42
    assert int(bus.tia.framebuffer[Y_START, 15]) == 0x42
    assert int(bus.tia.framebuffer[Y_START, 16]) == 0x00
    assert bus.tia.scanline == Y_START + 1
