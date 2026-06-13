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
    Y_START,
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
    # Task #83 (Y_START gate): the framebuffer is only written for
    # scanlines >= Y_START, so render at a visible line (the completed
    # line lands at framebuffer[Y_START]).
    tia = tia._replace(scanline=Y_START)
    tia = tia_poke(tia, W_PF0, 0xF0)
    tia = tia_poke(tia, W_PF1, 0xAA)
    tia = tia_poke(tia, W_COLUPF, 0x42)
    tia = tia_poke(tia, W_COLUBK, 0x10)
    # Render one full scanline.
    expected = render_scanline(tia)
    tia2 = tia_advance(tia, NTSC_CPU_CYCLES_PER_SCANLINE)
    # The completed scanline's row in the framebuffer should equal
    # the standalone `render_scanline` output.
    assert jnp.array_equal(tia2.framebuffer[Y_START], expected)


def test_p3i_b_vblank_still_suppresses_framebuffer_write():
    """VBLANK output-blanking still works post-P3i-b — collisions
    accumulate but the framebuffer stays untouched."""
    tia = initial_tia_state()
    tia = tia_poke(tia, W_PF0, 0xF0)
    tia = tia_poke(tia, W_COLUPF, 0x42)
    tia = tia._replace(vblank_active=True)
    tia = tia_advance(tia, NTSC_CPU_CYCLES_PER_SCANLINE)
    assert int(tia.framebuffer.sum()) == 0     # nothing written


# --------------------------------------------------------------------------- #
# P3i-c: per-poke write timing (mid-scanline PF0/PF1/PF2 deferred apply)
# --------------------------------------------------------------------------- #

from jaxtari.tia.system import (
    _POKE_DELAY_TABLE,
    _PF_DYNAMIC_DELAY,
    _pf_dynamic_delay,
    _poke_activation_delay,
)


def test_poke_delay_table_matches_xitari():
    """`_POKE_DELAY_TABLE` is the verbatim port of xitari's
    `ourPokeDelayTable[64]`. Spot-check a few well-known entries
    against the reference."""
    # VSYNC, WSYNC, COLU*, COLUBK, CTRLPF, RESP*, RESBL: delay 0
    for reg in (0x00, 0x02, 0x06, 0x07, 0x08, 0x09, 0x0A,
                0x10, 0x11, 0x14):
        assert _POKE_DELAY_TABLE[reg] == 0, f"reg ${reg:02x} should be 0"
    # VBLANK, REFP0, REFP1, GRP0, GRP1: delay 1
    for reg in (0x01, 0x0B, 0x0C, 0x1B, 0x1C):
        assert _POKE_DELAY_TABLE[reg] == 1, f"reg ${reg:02x} should be 1"
    # NUSIZ0/1, RESM0/1: delay 8
    for reg in (0x04, 0x05, 0x12, 0x13):
        assert _POKE_DELAY_TABLE[reg] == 8, f"reg ${reg:02x} should be 8"
    # PF0/PF1/PF2: dynamic (sentinel -1)
    for reg in (0x0D, 0x0E, 0x0F):
        assert _POKE_DELAY_TABLE[reg] == -1, f"reg ${reg:02x} should be -1"


def test_pf_dynamic_delay_cycles_4_5_2_3():
    """The PF dynamic delay cycles through {4, 5, 2, 3} as the
    color clock advances through the 12-clock window. Match xitari's
    `static const uInt32 d[4] = {4, 5, 2, 3}` with index `(x/3) & 3`."""
    assert _PF_DYNAMIC_DELAY == (4, 5, 2, 3)
    # Color clocks 0, 1, 2 → phase 0 → delay 4
    for c in (0, 1, 2):
        assert _pf_dynamic_delay(c) == 4
    # Color clocks 3, 4, 5 → phase 1 → delay 5
    for c in (3, 4, 5):
        assert _pf_dynamic_delay(c) == 5
    # Color clocks 6, 7, 8 → phase 2 → delay 2
    for c in (6, 7, 8):
        assert _pf_dynamic_delay(c) == 2
    # Color clocks 9, 10, 11 → phase 3 → delay 3
    for c in (9, 10, 11):
        assert _pf_dynamic_delay(c) == 3
    # Cycle repeats: 12, 13, 14 → phase 0 → delay 4
    for c in (12, 13, 14):
        assert _pf_dynamic_delay(c) == 4


def test_poke_activation_delay_pf_uses_dynamic():
    """`_poke_activation_delay` returns dynamic for PF0/PF1/PF2,
    table value for everything else."""
    assert _poke_activation_delay(W_PF0, 0) == 4
    assert _poke_activation_delay(W_PF1, 3) == 5
    assert _poke_activation_delay(W_PF2, 6) == 2
    assert _poke_activation_delay(0x04, 100) == 8     # NUSIZ0 fixed 8
    assert _poke_activation_delay(0x00, 100) == 0     # VSYNC fixed 0


def test_pf_write_during_hblank_applies_immediately():
    """Per the P3i-c implementation note: PF writes BEFORE the visible
    region (color_clock < HBLANK_COLOR_CLOCKS = 68) take effect
    immediately for the whole next scanline — same as pre-P3i. This
    keeps the "scanline setup" pattern (write PFs, fall into scanline
    render) producing identical output."""
    tia = initial_tia_state()
    tia = tia_poke(tia, W_COLUBK, 0x10)
    tia = tia_poke(tia, W_COLUPF, 0x42)
    tia = tia_poke(tia, W_PF0, 0xF0)              # color_clock=0 → no defer
    assert tia.pending_writes == ()
    assert int(tia.registers[W_PF0]) == 0xF0


def test_pf_write_mid_scanline_is_deferred():
    """PF write at color_clock >= 68 queues on `pending_writes` and
    does NOT update `tia.registers` immediately."""
    tia = initial_tia_state()
    tia = tia_advance(tia, 38)                    # color_clock = 114
    tia = tia_poke(tia, W_PF0, 0xCC)
    assert tia.pending_writes == ((116, W_PF0, 0xCC),)   # 114 + delay 2
    assert int(tia.registers[W_PF0]) == 0          # unchanged


def test_pf_mid_scanline_change_affects_only_post_activation_pixels():
    """The headline P3i-c property: writing PF0 mid-scanline changes
    only the pixels rendered AFTER the activation color clock. The
    PF0 contribution to pixels 0-15 (left side) is fixed by the
    register value BEFORE the mid-scanline write."""
    tia = initial_tia_state()
    tia = tia._replace(scanline=Y_START)          # render at a visible line (#83 gate)
    tia = tia_poke(tia, W_COLUBK, 0x10)
    tia = tia_poke(tia, W_COLUPF, 0x42)
    tia = tia_poke(tia, W_PF0, 0xF0)              # all 4 PF0 pixels lit
    tia = tia_advance(tia, 38)                    # color_clock = 114
    tia = tia_poke(tia, W_PF0, 0x00)              # mid-scanline: dark PF0
    tia = tia_advance(tia, 38)                    # finish the scanline
    # Pixels 0-15 (PF0 native region, rendered before c=114): still PF colour
    for x in range(16):
        assert int(tia.framebuffer[Y_START, x]) == 0x42, (
            f"left pixel {x} should still be PF colour (0x42), "
            f"got {int(tia.framebuffer[Y_START, x]):#04x}"
        )
    # Pixels 144-159 (PF0 mirror region, rendered after the mid-scanline
    # write): now background (mid-scanline write applied at c=116, well
    # before the mirror region renders).
    for x in range(144, 160):
        assert int(tia.framebuffer[Y_START, x]) == 0x10, (
            f"mirror pixel {x} should be background (0x10), "
            f"got {int(tia.framebuffer[Y_START, x]):#04x}"
        )


def test_pending_writes_cleared_after_advance():
    """`pending_writes` resets to empty after a scanline-crossing
    advance — the writes have been drained into the registers."""
    tia = initial_tia_state()
    tia = tia_advance(tia, 38)
    tia = tia_poke(tia, W_PF0, 0xAA)
    assert len(tia.pending_writes) == 1
    tia = tia_advance(tia, 38)                    # cross scanline boundary
    assert tia.pending_writes == ()
    # And the drained value is in the registers now.
    assert int(tia.registers[W_PF0]) == 0xAA


def test_multiple_mid_scanline_pf_writes_apply_in_beam_order():
    """A canonical 'brick stripe' pattern: write PF0, advance, write
    PF1, advance, write PF2, advance. Each write applies at its own
    activation clock. The resulting framebuffer should show pixels
    drawn between writes using the value current at that beam
    position."""
    tia = initial_tia_state()
    tia = tia_poke(tia, W_COLUBK, 0x10)
    tia = tia_poke(tia, W_COLUPF, 0x42)
    # Drive the scanline through three PF writes spaced apart.
    tia = tia_advance(tia, 30)                    # color_clock = 90
    tia = tia_poke(tia, W_PF0, 0xF0)
    tia = tia_advance(tia, 10)                    # color_clock = 120
    tia = tia_poke(tia, W_PF1, 0xFF)
    tia = tia_advance(tia, 10)                    # color_clock = 150
    tia = tia_poke(tia, W_PF2, 0xFF)
    tia = tia_advance(tia, 26)                    # finish to 76 cycles → wrap
    # Three pending writes queued, all activated mid-scanline.
    assert tia.pending_writes == ()
    # All three values should now be in the register file.
    assert int(tia.registers[W_PF0]) == 0xF0
    assert int(tia.registers[W_PF1]) == 0xFF
    assert int(tia.registers[W_PF2]) == 0xFF


# --------------------------------------------------------------------------- #
# P3i-f: HMOVE blank bug (the "HMOVE comb" — leftmost 8 visible pixels
# of the scanline FOLLOWING a properly-timed HMOVE write get blanked)
# --------------------------------------------------------------------------- #

from jaxtari.tia.system import (
    _hmove_blank_enabled_at,
    _HMOVE_BLANK_ENABLE_CYCLES,
    W_HMOVE,
)


def test_hmove_blank_enable_table_matches_xitari():
    """`_HMOVE_BLANK_ENABLE_CYCLES` should match xitari's
    `ourHMOVEBlankEnableCycles[128]` for indices 0..75. Per xitari:
    True for cycles 0..20 (inclusive — typical HMOVE-after-WSYNC
    placement, with the boundary at cycle 21 flipping to False),
    False for 21..74, True at cycle 75."""
    # 0..20 inclusive should be True
    for c in range(21):
        assert _HMOVE_BLANK_ENABLE_CYCLES[c] is True, f"cycle {c}"
    # 21..74 should be False
    for c in range(21, 75):
        assert _HMOVE_BLANK_ENABLE_CYCLES[c] is False, f"cycle {c}"
    # Cycle 75 should be True
    assert _HMOVE_BLANK_ENABLE_CYCLES[75] is True


def test_hmove_blank_enabled_at_helper():
    """`_hmove_blank_enabled_at(c)` returns the table entry at c modulo
    the scanline (76 CPU cycles)."""
    assert _hmove_blank_enabled_at(0) is True
    assert _hmove_blank_enabled_at(10) is True
    assert _hmove_blank_enabled_at(20) is True
    assert _hmove_blank_enabled_at(21) is False
    assert _hmove_blank_enabled_at(50) is False
    assert _hmove_blank_enabled_at(75) is True


def test_hmove_at_hblank_sets_blank_pending():
    """An HMOVE write at scanline_cycle=0 (typical, right after
    WSYNC) sets `hmove_blank_pending=True`."""
    tia = initial_tia_state()
    tia = tia_poke(tia, W_HMOVE, 0)
    assert tia.hmove_blank_pending is True


def test_hmove_at_mid_scanline_does_not_set_blank():
    """An HMOVE write at scanline_cycle=40 (well into the visible
    region) does NOT set the blank — `_HMOVE_BLANK_ENABLE_CYCLES[40]`
    is False."""
    tia = initial_tia_state()._replace(scanline_cycle=40, color_clock=120)
    tia = tia_poke(tia, W_HMOVE, 0)
    assert tia.hmove_blank_pending is False


def test_hmove_blank_blacks_first_8_visible_pixels():
    """When HMOVE blank fires, the first 8 visible color clocks
    (framebuffer pixels 0..7) render as 0 regardless of background.
    Pixel 8 and beyond render normally."""
    tia = initial_tia_state()
    tia = tia._replace(scanline=Y_START)        # render at a visible line (#83 gate)
    tia = tia_poke(tia, W_COLUBK, 0x42)         # solid background
    tia = tia_poke(tia, W_HMOVE, 0)
    assert tia.hmove_blank_pending is True
    tia = tia_advance(tia, NTSC_CPU_CYCLES_PER_SCANLINE)
    # First 8 pixels blank (0).
    for x in range(8):
        assert int(tia.framebuffer[Y_START, x]) == 0, f"x={x} should be blanked"
    # Pixel 8 and beyond use COLUBK = 0x42.
    for x in (8, 15, 50, 159):
        assert int(tia.framebuffer[Y_START, x]) == 0x42, f"x={x} should be COLUBK"


def test_hmove_blank_clears_after_scanline_renders():
    """After the scanline with the blank renders, the flag clears so
    subsequent scanlines render normally."""
    tia = initial_tia_state()
    tia = tia._replace(scanline=Y_START)        # render at a visible line (#83 gate)
    tia = tia_poke(tia, W_COLUBK, 0x42)
    tia = tia_poke(tia, W_HMOVE, 0)
    tia = tia_advance(tia, NTSC_CPU_CYCLES_PER_SCANLINE)
    assert tia.hmove_blank_pending is False
    # Run another scanline — all pixels should be COLUBK.
    tia = tia_advance(tia, NTSC_CPU_CYCLES_PER_SCANLINE)
    for x in range(160):
        assert int(tia.framebuffer[Y_START + 1, x]) == 0x42


def test_hmove_blank_next_defers_comb_to_following_line():
    """Task #97: an HMOVE write whose *effective* beam already crossed
    into the next scanline (`beam_sc >= 76`, e.g. enduro's free-running
    kernel lines) must NOT blank the CURRENT line's leftmost-8px comb —
    it parks in `hmove_blank_pending_next` and is promoted to
    `hmove_blank_pending` once the beam advances, so the comb lands on
    the FOLLOWING line. This is the bug the wrap fix + `_next` flag
    close (enduro 137 -> 33). The existing hmove-blank tests only
    exercise the beam_sc < 76 path."""
    tia = initial_tia_state()
    tia = tia._replace(scanline=Y_START)
    tia = tia_poke(tia, W_COLUBK, 0x42)
    # HMOVE with beam_sc = 76 → write belongs to line N+1: parked, not live.
    tia = tia_poke(tia, W_HMOVE, 0, beam_sc=76)
    assert tia.hmove_blank_pending is False
    assert tia.hmove_blank_pending_next is True
    # Render line Y_START: NO comb (flag was parked), and the parked flag
    # promotes to hmove_blank_pending for the next line.
    tia = tia_advance(tia, NTSC_CPU_CYCLES_PER_SCANLINE)
    for x in range(8):
        assert int(tia.framebuffer[Y_START, x]) == 0x42, \
            f"line N x={x} must NOT be blanked (comb deferred)"
    assert tia.hmove_blank_pending is True
    assert tia.hmove_blank_pending_next is False
    # Render line Y_START+1: NOW the comb fires (leftmost 8 px blanked).
    tia = tia_advance(tia, NTSC_CPU_CYCLES_PER_SCANLINE)
    for x in range(8):
        assert int(tia.framebuffer[Y_START + 1, x]) == 0, \
            f"line N+1 x={x} should be blanked by the promoted comb"
    for x in (8, 50, 159):
        assert int(tia.framebuffer[Y_START + 1, x]) == 0x42


# --------------------------------------------------------------------------- #
# P3i-g: full xitari ourCompleteMotionTable (mid-scanline-HMOVE quirks)
# --------------------------------------------------------------------------- #

from jaxtari.tia.system import _hmove_motion, _hm_offset, _COMPLETE_MOTION_TABLE


def test_motion_table_dimensions():
    """76 rows × 16 cols (covers scanline cycles 0..75, indexed by
    HM high nibble 0..15)."""
    assert len(_COMPLETE_MOTION_TABLE) == 76
    for row in _COMPLETE_MOTION_TABLE:
        assert len(row) == 16


def test_motion_table_hblank_matches_hm_offset():
    """At scanline_cycle 0..3 (HBLANK), the motion table value should
    equal `_hm_offset(hm)` for every HM nibble — they're the same
    HBLANK pattern."""
    for sc in range(4):
        for nibble in range(16):
            hm = nibble << 4
            assert _hmove_motion(sc, hm) == _hm_offset(hm), \
                f"sc={sc}, nibble={nibble:#x}"


def test_motion_table_mid_scanline_is_zero():
    """Cycles 21..54 — HMOVE in the visible region does NOT move
    sprites (xitari quirk: ourCompleteMotionTable rows 21..54 are
    all zeros)."""
    for sc in range(21, 55):
        for nibble in range(16):
            hm = nibble << 4
            assert _hmove_motion(sc, hm) == 0, \
                f"sc={sc}, nibble={nibble:#x} should be 0"


def test_motion_table_late_scanline_partial():
    """Cycles 55..74 produce cycle-dependent partial-motion deltas
    (xitari rows 55..74). Spot-check a few known table values."""
    # Cycle 64, all nibbles 0..7 → 1..8 (negated in our convention);
    # nibbles 8..15 → 0. From xitari row 64:
    # { 1, 2, 3, 4, 5, 6, 7, 8,  0, 0, 0, 0, 0, 0, 0, 0}
    assert _hmove_motion(64, 0x00) == 1
    assert _hmove_motion(64, 0x70) == 8
    assert _hmove_motion(64, 0x80) == 0   # high half all zero at cycle 64


def test_motion_table_hmove_at_mid_scanline_does_not_move_sprite():
    """End-to-end via tia_poke: write HMP0=0x70 (would normally
    move P0 by +7 in our sign convention), trigger HMOVE at
    scanline_cycle=30 (mid-scanline). P0 X position should NOT
    change."""
    from jaxtari.tia.system import W_HMP0
    tia = initial_tia_state()._replace(p0_x=80)
    tia = tia_poke(tia, W_HMP0, 0x70)
    # Advance to mid-scanline (scanline_cycle = 30).
    tia = tia._replace(scanline_cycle=30, color_clock=30 * 3)
    tia = tia_poke(tia, W_HMOVE, 0)
    assert tia.p0_x == 80, f"P0 should not move mid-scanline, got {tia.p0_x}"


def test_motion_table_hmove_at_hblank_moves_normally():
    """For contrast: HMOVE at scanline_cycle=0 with HMP0=0x70 moves
    P0 by 7 pixels left (our +sign), as before."""
    from jaxtari.tia.system import W_HMP0
    tia = initial_tia_state()._replace(p0_x=80)
    tia = tia_poke(tia, W_HMP0, 0x70)
    # scanline_cycle stays 0.
    tia = tia_poke(tia, W_HMOVE, 0)
    assert tia.p0_x == 73, f"P0 should move from 80 to 73, got {tia.p0_x}"
