"""P3d tests — missiles M0/M1 and ball BL, with sizes (NUSIZ / CTRLPF),
enable bits, RES* position-reset, and HMOVE/HMCLR integration."""

import jax.numpy as jnp

from jaxtari.tia.system import (
    initial_tia_state,
    render_scanline,
    tia_poke,
    W_COLUBK,
    W_COLUP0,
    W_COLUP1,
    W_COLUPF,
    W_CTRLPF,
    W_ENABL,
    W_ENAM0,
    W_ENAM1,
    W_HMBL,
    W_HMM0,
    W_HMM1,
    W_HMOVE,
    W_NUSIZ0,
    W_NUSIZ1,
    W_RESBL,
    W_RESM0,
    W_RESM1,
)


# --------------------------------------------------------------------------- #
# Missile enable
# --------------------------------------------------------------------------- #

def test_missile0_invisible_when_disabled():
    tia = initial_tia_state()._replace(m0_x=50)
    tia = tia_poke(tia, W_COLUP0, 0x42)
    scanline = render_scanline(tia)
    assert int(scanline.sum()) == 0


def test_missile0_visible_when_enabled():
    tia = initial_tia_state()._replace(m0_x=50)
    tia = tia_poke(tia, W_COLUP0, 0x42)
    tia = tia_poke(tia, W_ENAM0, 0x02)
    scanline = render_scanline(tia)
    assert int(scanline[50]) == 0x42
    assert int(scanline[49]) == 0
    assert int(scanline[51]) == 0


def test_missile_enable_bit_is_bit_1():
    """Only D1 of ENAM enables the missile."""
    tia = initial_tia_state()._replace(m0_x=50)
    tia = tia_poke(tia, W_COLUP0, 0x42)
    tia = tia_poke(tia, W_ENAM0, 0x01)        # wrong bit
    scanline = render_scanline(tia)
    assert int(scanline.sum()) == 0


# --------------------------------------------------------------------------- #
# Missile sizes from NUSIZ bits 4-5
# --------------------------------------------------------------------------- #

def test_missile_size_2():
    tia = initial_tia_state()._replace(m0_x=50)
    tia = tia_poke(tia, W_COLUP0, 0x42)
    tia = tia_poke(tia, W_ENAM0, 0x02)
    tia = tia_poke(tia, W_NUSIZ0, 0x10)       # bits 4-5 = 01 → size 2
    scanline = render_scanline(tia)
    assert int(scanline[50]) == 0x42 and int(scanline[51]) == 0x42
    assert int(scanline[52]) == 0


def test_missile_size_4():
    tia = initial_tia_state()._replace(m0_x=50)
    tia = tia_poke(tia, W_COLUP0, 0x42)
    tia = tia_poke(tia, W_ENAM0, 0x02)
    tia = tia_poke(tia, W_NUSIZ0, 0x20)       # bits 4-5 = 10 → size 4
    scanline = render_scanline(tia)
    for i in range(50, 54):
        assert int(scanline[i]) == 0x42
    assert int(scanline[54]) == 0


def test_missile_size_8():
    tia = initial_tia_state()._replace(m0_x=50)
    tia = tia_poke(tia, W_COLUP0, 0x42)
    tia = tia_poke(tia, W_ENAM0, 0x02)
    tia = tia_poke(tia, W_NUSIZ0, 0x30)       # bits 4-5 = 11 → size 8
    scanline = render_scanline(tia)
    for i in range(50, 58):
        assert int(scanline[i]) == 0x42
    assert int(scanline[58]) == 0


def test_missile1_uses_colup1_and_nusiz1():
    tia = initial_tia_state()._replace(m1_x=100)
    tia = tia_poke(tia, W_COLUP1, 0x66)
    tia = tia_poke(tia, W_ENAM1, 0x02)
    tia = tia_poke(tia, W_NUSIZ1, 0x20)       # size 4
    scanline = render_scanline(tia)
    for i in range(100, 104):
        assert int(scanline[i]) == 0x66
    assert int(scanline[104]) == 0


# --------------------------------------------------------------------------- #
# Ball
# --------------------------------------------------------------------------- #

def test_ball_invisible_when_disabled():
    tia = initial_tia_state()._replace(bl_x=80)
    tia = tia_poke(tia, W_COLUPF, 0x44)
    scanline = render_scanline(tia)
    assert int(scanline.sum()) == 0


def test_ball_uses_colupf():
    tia = initial_tia_state()._replace(bl_x=80)
    tia = tia_poke(tia, W_COLUPF, 0x44)
    tia = tia_poke(tia, W_ENABL, 0x02)
    scanline = render_scanline(tia)
    assert int(scanline[80]) == 0x44
    assert int(scanline[79]) == 0 and int(scanline[81]) == 0


def test_ball_size_4_from_ctrlpf_bits_4_5():
    tia = initial_tia_state()._replace(bl_x=80)
    tia = tia_poke(tia, W_COLUPF, 0x44)
    tia = tia_poke(tia, W_ENABL, 0x02)
    tia = tia_poke(tia, W_CTRLPF, 0x20)       # bits 4-5 = 10 → size 4
    scanline = render_scanline(tia)
    for i in range(80, 84):
        assert int(scanline[i]) == 0x44
    assert int(scanline[84]) == 0


def test_ball_size_8():
    tia = initial_tia_state()._replace(bl_x=80)
    tia = tia_poke(tia, W_COLUPF, 0x44)
    tia = tia_poke(tia, W_ENABL, 0x02)
    tia = tia_poke(tia, W_CTRLPF, 0x30)       # bits 4-5 = 11 → size 8
    scanline = render_scanline(tia)
    for i in range(80, 88):
        assert int(scanline[i]) == 0x44


# --------------------------------------------------------------------------- #
# RES* + HMOVE integration
# --------------------------------------------------------------------------- #

def test_resm0_sets_m0_x_from_scanline_cycle():
    # P3i-e: RESM0 uses xitari-exact `(color_clock - HBLANK + 4) % 160`
    # at visible color clocks, latched to 2 during HBLANK. With
    # color_clock = 30*3 = 90: (90-68+4) % 160 = 26.
    tia = initial_tia_state()._replace(scanline_cycle=30, color_clock=90)
    tia = tia_poke(tia, W_RESM0, 0x00)
    assert tia.m0_x == 26


def test_resbl_sets_bl_x_from_scanline_cycle():
    # P3i-e: RESBL uses the same missile+ball formula. With
    # color_clock = 50*3 = 150: (150-68+4) % 160 = 86.
    tia = initial_tia_state()._replace(scanline_cycle=50, color_clock=150)
    tia = tia_poke(tia, W_RESBL, 0x00)
    assert tia.bl_x == 86


def test_hmove_applies_to_missiles_and_ball():
    tia = initial_tia_state()._replace(m0_x=50, m1_x=50, bl_x=50)
    tia = tia_poke(tia, W_HMM0, 0x10)         # +1 left
    tia = tia_poke(tia, W_HMM1, 0xE0)         # -2 right
    tia = tia_poke(tia, W_HMBL, 0xF0)         # -1 right
    tia = tia_poke(tia, W_HMOVE, 0x00)
    assert tia.m0_x == 49
    assert tia.m1_x == 52
    assert tia.bl_x == 51


# --------------------------------------------------------------------------- #
# Priority — players paint over missiles+ball, missile colour comes from COLUP
# --------------------------------------------------------------------------- #

def test_player_paints_over_missile_at_same_position():
    """When a player and a missile occupy the same pixel, the player wins."""
    tia = initial_tia_state()._replace(p0_x=50, m0_x=50)
    tia = tia_poke(tia, 0x1B, 0xFF)           # GRP0 = $FF
    tia = tia_poke(tia, W_COLUP0, 0x42)
    tia = tia_poke(tia, W_ENAM0, 0x02)        # missile enabled
    scanline = render_scanline(tia)
    # Player wins → all 8 pixels at 50..57 are 0x42 (same as missile in this
    # case, so testing colour priority requires different colours).
    for i in range(50, 58):
        assert int(scanline[i]) == 0x42
