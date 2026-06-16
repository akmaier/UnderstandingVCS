"""P3c tests — TIA player sprites (P0/P1) with GRP, REFP, RESP, HMOVE/HMCLR."""

import jax.numpy as jnp

from jaxtari.tia.system import (
    NTSC_CPU_CYCLES_PER_SCANLINE,
    _hm_offset,
    _resp_position,
    initial_tia_state,
    render_scanline,
    tia_advance,
    tia_poke,
    W_COLUBK,
    W_COLUP0,
    W_COLUP1,
    W_GRP0,
    W_GRP1,
    W_HMP0,
    W_HMP1,
    W_HMM0,
    W_HMM1,
    W_HMBL,
    W_HMOVE,
    W_HMCLR,
    W_REFP0,
    W_REFP1,
    W_RESP0,
    W_RESP1,
)


# --------------------------------------------------------------------------- #
# _hm_offset — HM nibble → signed offset
# --------------------------------------------------------------------------- #

def test_hm_offset_positive_nibbles():
    assert _hm_offset(0x00) == 0
    assert _hm_offset(0x10) == 1
    assert _hm_offset(0x70) == 7    # max positive (move left 7)


def test_hm_offset_negative_nibbles():
    assert _hm_offset(0x80) == -8   # max negative (move right 8)
    assert _hm_offset(0xF0) == -1
    assert _hm_offset(0xE0) == -2


def test_hm_offset_low_nibble_ignored():
    assert _hm_offset(0x7F) == 7
    assert _hm_offset(0x8F) == -8


# --------------------------------------------------------------------------- #
# _resp_position — scanline_cycle → sprite x
# --------------------------------------------------------------------------- #

def test_resp_position_hblank_clamps_to_zero():
    """During HBLANK (scanline_cycle ≤ 22 = 68 colour clocks / 3), RESP lands
    at the left edge of the visible area."""
    assert _resp_position(0) == 0
    assert _resp_position(10) == 0
    assert _resp_position(22) == 0


def test_resp_position_just_inside_visible():
    """At scanline_cycle 23, 23*3-68 = 1 pixel into visible area."""
    assert _resp_position(23) == 1


def test_resp_position_clamps_to_159_at_far_right():
    # At scanline_cycle = 76 (just past scanline), 76*3-68 = 160. Clamped to 159.
    assert _resp_position(76) == 159
    # Way past
    assert _resp_position(100) == 159


# --------------------------------------------------------------------------- #
# tia_poke side-effects — RESP, HMOVE, HMCLR
# --------------------------------------------------------------------------- #

def test_resp0_sets_p0_x_from_scanline_cycle():
    # P3i-e: RESP0 uses xitari-exact formula `(color_clock - HBLANK + 5) % 160`
    # at visible color clocks, latched to 3 during HBLANK. With
    # color_clock = 30*3 = 90: (90-68+5) % 160 = 27.
    #
    # Task #115c (faithful object render): a VISIBLE RESP is now DEFERRED
    # to its activation color clock so the player renders at the OLD
    # position left of the strobe (multiplexed-sprite trick). Advance one
    # scanline so the deferred pending_writes entry drains into p0_x.
    tia = initial_tia_state()._replace(scanline_cycle=30, color_clock=90)
    tia = tia_poke(tia, W_RESP0, 0x00)
    tia = tia_advance(tia, NTSC_CPU_CYCLES_PER_SCANLINE)
    assert tia.p0_x == 27


def test_resp1_does_not_touch_p0_x():
    # Task #115c: VISIBLE RESP* is deferred — drain via tia_advance.
    tia = initial_tia_state()._replace(scanline_cycle=30, color_clock=90, p0_x=50)
    tia = tia_poke(tia, W_RESP1, 0x00)
    tia = tia_advance(tia, NTSC_CPU_CYCLES_PER_SCANLINE)
    assert tia.p0_x == 50
    assert tia.p1_x == 27       # same P3i-e formula as RESP0 (deferred)


def test_hmove_applies_hmp_offsets_to_both_players():
    """HMP0=+1 moves p0 left by 1; HMP1=-2 moves p1 right by 2."""
    tia = initial_tia_state()._replace(p0_x=50, p1_x=50)
    tia = tia_poke(tia, W_HMP0, 0x10)   # +1 (left)
    tia = tia_poke(tia, W_HMP1, 0xE0)   # -2 (right)
    tia = tia_poke(tia, W_HMOVE, 0x00)
    assert tia.p0_x == 49
    assert tia.p1_x == 52


def test_hmove_wraps_position():
    tia = initial_tia_state()._replace(p0_x=2)
    tia = tia_poke(tia, W_HMP0, 0x70)    # +7
    tia = tia_poke(tia, W_HMOVE, 0x00)
    # 2 - 7 = -5 → wrap mod 160 → 155
    assert tia.p0_x == 155


def test_hmclr_zeros_all_hm_registers():
    tia = initial_tia_state()
    tia = tia_poke(tia, W_HMP0, 0x70)
    tia = tia_poke(tia, W_HMP1, 0x80)
    tia = tia_poke(tia, W_HMM0, 0x40)
    tia = tia_poke(tia, W_HMM1, 0x20)
    tia = tia_poke(tia, W_HMBL, 0x10)
    tia = tia_poke(tia, W_HMCLR, 0x00)
    for reg in (W_HMP0, W_HMP1, W_HMM0, W_HMM1, W_HMBL):
        assert int(tia.registers[reg]) == 0


# --------------------------------------------------------------------------- #
# render_scanline — player sprite overlay
# --------------------------------------------------------------------------- #

def _setup(p0_x=0, p1_x=0, grp0=0, grp1=0, **regs):
    tia = initial_tia_state()._replace(p0_x=p0_x, p1_x=p1_x)
    if grp0:
        tia = tia_poke(tia, W_GRP0, grp0)
    if grp1:
        tia = tia_poke(tia, W_GRP1, grp1)
    for name, value in regs.items():
        tia = tia_poke(tia, globals()[f"W_{name.upper()}"], value)
    return tia


def test_player0_invisible_when_grp0_zero():
    tia = _setup(p0_x=50, grp0=0, colup0=0x42)
    scanline = render_scanline(tia)
    # All pixels are background (default 0).
    assert int(scanline.sum()) == 0


def test_player0_all_bits_set_paints_8_pixels():
    tia = _setup(p0_x=50, grp0=0xFF, colup0=0x42)
    scanline = render_scanline(tia)
    # Positions 50..57 should be COLUP0.
    for i in range(50, 58):
        assert int(scanline[i]) == 0x42, f"pixel {i}"
    assert int(scanline[49]) == 0
    assert int(scanline[58]) == 0


def test_player0_grp_bit_7_is_leftmost_default():
    """GRP=0x80 → only bit 7 set → leftmost pixel of the 8-pixel span lit."""
    tia = _setup(p0_x=50, grp0=0x80, colup0=0x42)
    scanline = render_scanline(tia)
    assert int(scanline[50]) == 0x42
    assert int(scanline[51]) == 0


def test_player0_grp_bit_0_is_rightmost_default():
    tia = _setup(p0_x=50, grp0=0x01, colup0=0x42)
    scanline = render_scanline(tia)
    assert int(scanline[57]) == 0x42
    assert int(scanline[56]) == 0


def test_player0_refp_reflects_bit_order():
    """REFP0.D3 set → GRP bit 0 becomes LEFTMOST (instead of rightmost)."""
    tia = _setup(p0_x=50, grp0=0x01, colup0=0x42, refp0=0x08)
    scanline = render_scanline(tia)
    assert int(scanline[50]) == 0x42      # bit 0 now leftmost
    assert int(scanline[57]) == 0


def test_player1_independent_of_player0():
    tia = _setup(p0_x=20, p1_x=100, grp0=0xFF, grp1=0xFF,
                 colup0=0x42, colup1=0x66)
    scanline = render_scanline(tia)
    for i in range(20, 28):
        assert int(scanline[i]) == 0x42
    for i in range(100, 108):
        assert int(scanline[i]) == 0x66


def test_player_paints_over_playfield():
    """A player pixel overrides the playfield pixel underneath.

    NB (test nudge 2026-06-12): COLU* writes store `value & 0xFE` —
    xitari masks the LSB (the color-loss bit; see the "COLU& 0xFE
    mask" note in test_screen_conformance.py). COLUPF=$33 therefore
    reads back as $32, which is what the playfield pixels assert.
    """
    tia = _setup(p0_x=4, grp0=0xFF, colup0=0x42)
    # Light up the playfield with PF0 bit 4 → playfield pixels 0..3 → screen 0..15
    tia = tia_poke(tia, 0x0D, 0xF0)         # PF0 = $F0 (all 4 bits)
    tia = tia_poke(tia, W_COLUBK, 0x11)
    tia = tia_poke(tia, 0x08, 0x33)         # COLUPF = $33 → stored $32
    scanline = render_scanline(tia)
    # Pixels 0..3 are still playfield (player not over them).
    assert int(scanline[0]) == 0x32
    # Pixels 4..11 are part playfield (4..15 = pf pixels 1..3), so 4..11 was 0x32
    # but player overrides 4..11 → 0x42.
    for i in range(4, 12):
        assert int(scanline[i]) == 0x42, f"pixel {i}"
    # Pixel 12..15: playfield (pf pixel 3 = bits 12..15)
    assert int(scanline[12]) == 0x32


def test_player_wraps_at_right_edge():
    """A player at x=155 with GRP=0xFF paints pixels 155..159 and 0..2 (mod 160)."""
    tia = _setup(p0_x=155, grp0=0xFF, colup0=0x42)
    scanline = render_scanline(tia)
    for i in (155, 156, 157, 158, 159, 0, 1, 2):
        assert int(scanline[i]) == 0x42, f"pixel {i}"
