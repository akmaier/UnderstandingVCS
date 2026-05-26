"""P3i-a + P3i-b — per-color-clock render kernel scaffolding.

P3i-a deliverable: `render_pixel(tia, color_clock)` returns the same
pixel value as `render_scanline(tia)[color_clock - HBLANK_COLOR_CLOCKS]`
for every visible color clock (68..227). HBLANK positions (0..67) and
positions past the visible region return 0.

P3i-b deliverable: `tia_advance` writes the framebuffer via the new
per-color-clock kernel (instead of the old end-of-scanline single
`render_scanline` write). At this stage both produce bit-exact equal
output — the same register state, the same composition. P3i-c is what
actually starts diverging from the old single-shot write by giving
mid-scanline pokes their own activation color clock.
"""

import jax.numpy as jnp
import pytest

from jaxtari.tia import (
    COLOR_CLOCKS_PER_CPU_CYCLE,
    COLOR_CLOCKS_PER_SCANLINE,
    HBLANK_COLOR_CLOCKS,
    NTSC_CPU_CYCLES_PER_SCANLINE,
    SCREEN_WIDTH,
    initial_tia_state,
    render_pixel,
    render_scanline,
    tia_advance,
    tia_poke,
)
from jaxtari.tia.system import (
    W_COLUBK,
    W_COLUPF,
    W_COLUP0,
    W_CTRLPF,
    W_GRP0,
    W_PF0,
    W_PF1,
    W_PF2,
    W_RESP0,
)


# --------------------------------------------------------------------------- #
# Constants sanity
# --------------------------------------------------------------------------- #

def test_color_clock_constants():
    """228 color clocks per scanline = 76 CPU cycles × 3 color clocks. 68
    of those clocks are HBLANK (chip blanks output during beam retrace);
    the remaining 160 are visible — which matches SCREEN_WIDTH."""
    assert COLOR_CLOCKS_PER_CPU_CYCLE == 3
    assert COLOR_CLOCKS_PER_SCANLINE == NTSC_CPU_CYCLES_PER_SCANLINE * 3
    assert COLOR_CLOCKS_PER_SCANLINE == 228
    assert HBLANK_COLOR_CLOCKS == 68
    assert COLOR_CLOCKS_PER_SCANLINE - HBLANK_COLOR_CLOCKS == SCREEN_WIDTH


def test_initial_color_clock_is_zero():
    tia = initial_tia_state()
    assert tia.color_clock == 0


# --------------------------------------------------------------------------- #
# render_pixel boundary behaviour
# --------------------------------------------------------------------------- #

def test_render_pixel_returns_zero_for_hblank_positions():
    """Color clocks 0..67 are HBLANK — the TIA emits no visible pixel.
    `render_pixel` returns 0 (the framebuffer never receives these)."""
    tia = initial_tia_state()
    tia = tia_poke(tia, W_COLUBK, 0x42)            # non-zero background
    for c in (0, 1, 17, 50, 67):
        assert render_pixel(tia, c) == 0, f"HBLANK position {c} should be 0"


def test_render_pixel_returns_zero_past_visible_region():
    """Positions ≥ 228 are off-the-end of the scanline (shouldn't be
    reached by `tia_advance` but the kernel handles them gracefully)."""
    tia = initial_tia_state()
    tia = tia_poke(tia, W_COLUBK, 0x42)
    assert render_pixel(tia, 228) == 0
    assert render_pixel(tia, 1000) == 0


# --------------------------------------------------------------------------- #
# Equivalence with render_scanline — the P3i-a invariant
# --------------------------------------------------------------------------- #

def _exhaustive_equivalence(tia):
    """`render_pixel(tia, c)` MUST equal `render_scanline(tia)[c-68]`
    for every visible color clock (68..227). This is the foundational
    invariant — P3i-c will break it intentionally by giving pokes
    activation timing, but P3i-a/b must preserve it exactly."""
    scanline = render_scanline(tia)
    for c in range(HBLANK_COLOR_CLOCKS, COLOR_CLOCKS_PER_SCANLINE):
        x = c - HBLANK_COLOR_CLOCKS
        actual = render_pixel(tia, c)
        expected = int(scanline[x])
        assert actual == expected, (
            f"color_clock={c} (x={x}): render_pixel={actual:#04x} but "
            f"render_scanline[{x}]={expected:#04x}"
        )


def test_render_pixel_equals_scanline_all_zero():
    """Default all-zero TIA: every visible pixel = COLUBK = 0."""
    _exhaustive_equivalence(initial_tia_state())


def test_render_pixel_equals_scanline_solid_background():
    """COLUBK = 0x42 → every visible pixel is 0x42."""
    tia = tia_poke(initial_tia_state(), W_COLUBK, 0x42)
    _exhaustive_equivalence(tia)


def test_render_pixel_equals_scanline_with_playfield():
    """PF0/1/2 + COLUPF lit up. Exercises the playfield bit-decode +
    left/right-half mirror logic shared between the two paths."""
    tia = initial_tia_state()
    tia = tia_poke(tia, W_PF0, 0xF0)           # leftmost 4 playfield pixels lit
    tia = tia_poke(tia, W_PF1, 0xAA)           # alternating bits in middle
    tia = tia_poke(tia, W_PF2, 0x55)           # alternating bits at end of left half
    tia = tia_poke(tia, W_COLUPF, 0x42)        # playfield colour
    tia = tia_poke(tia, W_COLUBK, 0x10)        # background distinct
    _exhaustive_equivalence(tia)


def test_render_pixel_equals_scanline_with_player():
    """GRP0 painted at a definite X position — exercises sprite-overlay
    code path that's separate from playfield."""
    tia = initial_tia_state()
    # Position P0 mid-screen via a RESP0 in the visible region.
    tia = tia._replace(scanline_cycle=40)      # → p0_x ≈ 52
    tia = tia_poke(tia, W_RESP0, 0)            # trigger RESP latch
    tia = tia_poke(tia, W_GRP0, 0xAA)          # alternating bits
    tia = tia_poke(tia, W_COLUP0, 0x66)
    tia = tia_poke(tia, W_COLUBK, 0x10)
    _exhaustive_equivalence(tia)


def test_render_pixel_equals_scanline_with_priority_swap():
    """CTRLPF.D2 (PFP) priority mode: PF + BL composite ON TOP of
    sprites. Both render paths must agree."""
    tia = initial_tia_state()
    tia = tia._replace(scanline_cycle=40)
    tia = tia_poke(tia, W_RESP0, 0)
    tia = tia_poke(tia, W_GRP0, 0xFF)          # solid 8-px player
    tia = tia_poke(tia, W_COLUP0, 0x66)
    tia = tia_poke(tia, W_PF1, 0xFF)           # solid PF stripe spans the player
    tia = tia_poke(tia, W_COLUPF, 0x42)
    tia = tia_poke(tia, W_COLUBK, 0x10)
    tia = tia_poke(tia, W_CTRLPF, 0x04)        # PFP priority bit
    _exhaustive_equivalence(tia)


# --------------------------------------------------------------------------- #
# P3i-a: color_clock advances correctly
# --------------------------------------------------------------------------- #

def test_color_clock_advances_with_cpu_cycles():
    """One CPU cycle = 3 color clocks. `tia_advance(n)` advances
    `tia.color_clock` by `3·n` mod 228."""
    tia = initial_tia_state()
    tia = tia_advance(tia, 1)
    assert tia.color_clock == 3
    tia = tia_advance(tia, 5)
    assert tia.color_clock == 18                       # 3 + 5·3 = 18


def test_color_clock_wraps_at_scanline_boundary():
    """At scanline boundary (76 CPU cycles = 228 color clocks),
    `color_clock` wraps to 0 alongside the existing `scanline_cycle`
    wrap."""
    tia = initial_tia_state()
    tia = tia_advance(tia, NTSC_CPU_CYCLES_PER_SCANLINE)
    assert tia.color_clock == 0
    assert tia.scanline_cycle == 0
    assert tia.scanline == 1                           # crossed into next line


# --------------------------------------------------------------------------- #
# P3i-b: tia_advance still produces bit-exact framebuffer output
# --------------------------------------------------------------------------- #

def test_p3i_b_framebuffer_matches_pre_p3i_render():
    """The new per-color-clock framebuffer write must produce the same
    pixels as the pre-P3i `render_scanline`-only write — register state
    is the same at every color clock today (mid-scanline timing lands
    in P3i-c). This pins the no-behavioural-change invariant."""
    tia = initial_tia_state()
    tia = tia_poke(tia, W_PF0, 0xF0)
    tia = tia_poke(tia, W_PF1, 0xAA)
    tia = tia_poke(tia, W_COLUPF, 0x42)
    tia = tia_poke(tia, W_COLUBK, 0x10)
    # Render one full scanline.
    expected = render_scanline(tia)
    tia2 = tia_advance(tia, NTSC_CPU_CYCLES_PER_SCANLINE)
    # The completed scanline's row in the framebuffer should equal
    # the standalone `render_scanline` output.
    assert jnp.array_equal(tia2.framebuffer[0], expected)


def test_p3i_b_vblank_still_suppresses_framebuffer_write():
    """VBLANK output-blanking still works post-P3i-b — collisions
    accumulate but the framebuffer stays untouched."""
    tia = initial_tia_state()
    tia = tia_poke(tia, W_PF0, 0xF0)
    tia = tia_poke(tia, W_COLUPF, 0x42)
    tia = tia._replace(vblank_active=True)
    tia = tia_advance(tia, NTSC_CPU_CYCLES_PER_SCANLINE)
    assert int(tia.framebuffer.sum()) == 0     # nothing written
