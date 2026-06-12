"""P3f tests — VSYNC-driven frame ending and VBLANK output blanking."""

import jax.numpy as jnp

from jaxtari.tia.system import (
    NTSC_CPU_CYCLES_PER_SCANLINE,
    Y_START,
    initial_tia_state,
    tia_advance,
    tia_poke,
    W_COLUPF,
    W_PF0,
    W_VBLANK,
    W_VSYNC,
)


# --------------------------------------------------------------------------- #
# VSYNC — D1 tracking + frame edge
# --------------------------------------------------------------------------- #

def test_vsync_default_clear():
    tia = initial_tia_state()
    assert tia.vsync_active is False


def test_vsync_d1_sets_flag():
    tia = initial_tia_state()
    tia = tia_poke(tia, W_VSYNC, 0x02)
    assert tia.vsync_active is True
    assert tia.frame == 0           # no frame increment yet (only the 1→0 edge does that)
    assert tia.scanline == 0


def test_vsync_d1_only_bit_1_matters():
    tia = initial_tia_state()
    tia = tia_poke(tia, W_VSYNC, 0x01)
    assert tia.vsync_active is False
    tia = tia_poke(tia, W_VSYNC, 0xFD)
    assert tia.vsync_active is False
    tia = tia_poke(tia, W_VSYNC, 0xFF)
    assert tia.vsync_active is True


def test_vsync_falling_edge_increments_frame_and_resets_scanline():
    """The 1→0 transition is the software-defined frame boundary."""
    tia = initial_tia_state()._replace(scanline=100, scanline_cycle=42)
    tia = tia_poke(tia, W_VSYNC, 0x02)       # set
    assert tia.frame == 0
    assert tia.scanline == 100              # not yet reset
    tia = tia_poke(tia, W_VSYNC, 0x00)       # clear → falling edge
    assert tia.frame == 1
    assert tia.scanline == 0
    assert tia.scanline_cycle == 0
    assert tia.vsync_active is False


def test_clearing_vsync_when_not_active_does_nothing():
    """Two consecutive clears (no rising edge in between) shouldn't trigger
    a frame boundary on the second one."""
    tia = initial_tia_state()._replace(scanline=50)
    tia = tia_poke(tia, W_VSYNC, 0x00)       # already clear
    assert tia.frame == 0
    assert tia.scanline == 50


# --------------------------------------------------------------------------- #
# VBLANK — D1 tracking + framebuffer suppression
# --------------------------------------------------------------------------- #

def test_vblank_default_clear():
    tia = initial_tia_state()
    assert tia.vblank_active is False


def test_vblank_d1_sets_and_clears_flag():
    tia = initial_tia_state()
    tia = tia_poke(tia, W_VBLANK, 0x02)
    assert tia.vblank_active is True
    tia = tia_poke(tia, W_VBLANK, 0x00)
    assert tia.vblank_active is False


def test_vblank_suppresses_framebuffer_writes():
    """With VBLANK active, completed scanlines don't reach the framebuffer."""
    tia = initial_tia_state()
    tia = tia_poke(tia, W_PF0, 0xF0)
    tia = tia_poke(tia, W_COLUPF, 0x42)
    tia = tia_poke(tia, W_VBLANK, 0x02)
    tia = tia_advance(tia, NTSC_CPU_CYCLES_PER_SCANLINE)
    # Row 0 should still be all zeros (blanking suppressed the write).
    assert int(tia.framebuffer[0].sum()) == 0
    # Scanline counter still advanced.
    assert tia.scanline == 1


def test_vblank_clear_resumes_framebuffer_writes():
    # Task #83 round 3 (2026-06-11) / test nudge (2026-06-12): the
    # framebuffer write is gated on `tia.scanline >= Y_START`, so the
    # test starts at Y_START to exercise the resume at the first
    # VISIBLE scanline (same nudge the jutari runtests got in #83).
    tia = initial_tia_state()._replace(scanline=Y_START)
    tia = tia_poke(tia, W_PF0, 0xF0)
    tia = tia_poke(tia, W_COLUPF, 0x42)
    tia = tia_poke(tia, W_VBLANK, 0x02)
    tia = tia_advance(tia, NTSC_CPU_CYCLES_PER_SCANLINE)
    assert int(tia.framebuffer.sum()) == 0
    # Clear VBLANK and run another scanline.
    tia = tia_poke(tia, W_VBLANK, 0x00)
    tia = tia_advance(tia, NTSC_CPU_CYCLES_PER_SCANLINE)
    # Row Y_START+1 should now have the playfield rendered.
    assert int(tia.framebuffer[Y_START + 1, 0]) == 0x42
    assert int(tia.framebuffer[Y_START + 1, 15]) == 0x42


# --------------------------------------------------------------------------- #
# Full frame cycle: VSYNC + VBLANK + visible + repeat
# --------------------------------------------------------------------------- #

def test_full_frame_cycle_via_vsync():
    """Simulate the standard Atari frame structure with VSYNC + VBLANK."""
    tia = initial_tia_state()
    tia = tia_poke(tia, W_PF0, 0xF0)
    tia = tia_poke(tia, W_COLUPF, 0x42)
    # 1. VSYNC: 3 lines (blanked).
    tia = tia_poke(tia, W_VSYNC, 0x02)
    tia = tia_poke(tia, W_VBLANK, 0x02)
    tia = tia_advance(tia, 3 * NTSC_CPU_CYCLES_PER_SCANLINE)
    tia = tia_poke(tia, W_VSYNC, 0x00)              # falling edge → frame=1
    assert tia.frame == 1
    assert tia.scanline == 0
    # 2. VBLANK: 37 lines (blanked). VBLANK still active.
    tia = tia_advance(tia, 37 * NTSC_CPU_CYCLES_PER_SCANLINE)
    assert int(tia.framebuffer.sum()) == 0          # nothing rendered yet
    # 3. Visible: clear VBLANK, render 3 lines.
    tia = tia_poke(tia, W_VBLANK, 0x00)
    tia = tia_advance(tia, 3 * NTSC_CPU_CYCLES_PER_SCANLINE)
    # Rows 37, 38, 39 should have the playfield.
    for row in (37, 38, 39):
        assert int(tia.framebuffer[row, 0]) == 0x42, f"row {row}"
    # Row 40 still untouched.
    assert int(tia.framebuffer[40, 0]) == 0
