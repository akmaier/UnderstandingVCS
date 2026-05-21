"""P3e tests — TIA collision latches (8 read-only registers, CXCLR)."""

import jax.numpy as jnp

from jaxtari.tia.system import (
    NTSC_CPU_CYCLES_PER_SCANLINE,
    initial_tia_state,
    tia_advance,
    tia_peek,
    tia_poke,
    W_CXCLR,
    W_ENABL,
    W_ENAM0,
    W_ENAM1,
    W_GRP0,
    W_GRP1,
    W_NUSIZ0,
    W_PF0,
)


# Address constants for the readable collision registers.
R_CXM0P  = 0x30
R_CXM1P  = 0x31
R_CXP0FB = 0x32
R_CXP1FB = 0x33
R_CXM0FB = 0x34
R_CXM1FB = 0x35
R_CXBLPF = 0x36
R_CXPPMM = 0x37


def _advance_one_scanline(tia):
    return tia_advance(tia, NTSC_CPU_CYCLES_PER_SCANLINE)


def _enable_player(tia, player, x, grp=0xFF, color=0x42):
    """Set GRP, position, and colour for a player so it covers 8 pixels."""
    from jaxtari.tia.system import W_COLUP0, W_COLUP1
    if player == 0:
        tia = tia._replace(p0_x=x)
        tia = tia_poke(tia, W_GRP0, grp)
        tia = tia_poke(tia, W_COLUP0, color)
    else:
        tia = tia._replace(p1_x=x)
        tia = tia_poke(tia, W_GRP1, grp)
        tia = tia_poke(tia, W_COLUP1, color)
    return tia


def _enable_missile(tia, missile, x, size=1):
    """Enable missile at position x with given size (NUSIZ bits 4-5)."""
    if missile == 0:
        tia = tia._replace(m0_x=x)
        tia = tia_poke(tia, W_ENAM0, 0x02)
        if size > 1:
            tia = tia_poke(tia, W_NUSIZ0, {2: 0x10, 4: 0x20, 8: 0x30}[size])
    else:
        tia = tia._replace(m1_x=x)
        tia = tia_poke(tia, W_ENAM1, 0x02)
        from jaxtari.tia.system import W_NUSIZ1
        if size > 1:
            tia = tia_poke(tia, W_NUSIZ1, {2: 0x10, 4: 0x20, 8: 0x30}[size])
    return tia


def _enable_ball(tia, x, size=1):
    from jaxtari.tia.system import W_CTRLPF
    tia = tia._replace(bl_x=x)
    tia = tia_poke(tia, W_ENABL, 0x02)
    if size > 1:
        tia = tia_poke(tia, W_CTRLPF, {2: 0x10, 4: 0x20, 8: 0x30}[size])
    return tia


# --------------------------------------------------------------------------- #
# Default state and CXCLR
# --------------------------------------------------------------------------- #

def test_initial_collisions_are_all_zero():
    tia = initial_tia_state()
    for reg in range(0x30, 0x38):
        assert tia_peek(tia, reg) == 0


def test_inpt_addresses_still_return_zero():
    """\$38-\$3D (INPT*) — still stub for now (input phase, not P3e)."""
    tia = initial_tia_state()
    for reg in range(0x38, 0x3E):
        assert tia_peek(tia, reg) == 0


def test_cxclr_zeros_all_latches():
    tia = initial_tia_state()._replace(
        collisions=jnp.array([0xC0] * 8, dtype=jnp.uint8)
    )
    tia = tia_poke(tia, W_CXCLR, 0x00)
    for reg in range(0x30, 0x38):
        assert tia_peek(tia, reg) == 0


# --------------------------------------------------------------------------- #
# Object pairs that should and should not collide
# --------------------------------------------------------------------------- #

def test_no_collision_when_objects_not_overlapping():
    tia = initial_tia_state()
    tia = _enable_player(tia, 0, x=10)        # P0 at 10..17
    tia = _enable_player(tia, 1, x=100)       # P1 at 100..107
    tia = _advance_one_scanline(tia)
    assert tia_peek(tia, R_CXPPMM) == 0


def test_p0_p1_overlap_sets_cxppmm_d7():
    tia = initial_tia_state()
    tia = _enable_player(tia, 0, x=50)
    tia = _enable_player(tia, 1, x=50)        # exact overlap
    tia = _advance_one_scanline(tia)
    # CXPPMM D7 = P0-P1
    assert (tia_peek(tia, R_CXPPMM) & 0x80) != 0
    # No other collisions should have fired.
    assert (tia_peek(tia, R_CXPPMM) & 0x40) == 0   # M0-M1
    assert tia_peek(tia, R_CXP0FB) == 0            # no PF, no BL


def test_m0_m1_overlap_sets_cxppmm_d6():
    tia = initial_tia_state()
    tia = _enable_missile(tia, 0, x=80)
    tia = _enable_missile(tia, 1, x=80)
    tia = _advance_one_scanline(tia)
    assert (tia_peek(tia, R_CXPPMM) & 0x40) != 0   # M0-M1
    assert (tia_peek(tia, R_CXPPMM) & 0x80) == 0   # no P0-P1


def test_m0_p1_overlap_sets_cxm0p_d7():
    tia = initial_tia_state()
    tia = _enable_missile(tia, 0, x=50)
    tia = _enable_player(tia, 1, x=50)
    tia = _advance_one_scanline(tia)
    assert (tia_peek(tia, R_CXM0P) & 0x80) != 0    # M0-P1
    assert (tia_peek(tia, R_CXM0P) & 0x40) == 0    # M0-P0 (no P0)


def test_m0_p0_overlap_sets_cxm0p_d6():
    tia = initial_tia_state()
    tia = _enable_missile(tia, 0, x=50)
    tia = _enable_player(tia, 0, x=50)
    tia = _advance_one_scanline(tia)
    assert (tia_peek(tia, R_CXM0P) & 0x40) != 0    # M0-P0


def test_m1_p0_overlap_sets_cxm1p_d7():
    tia = initial_tia_state()
    tia = _enable_missile(tia, 1, x=50)
    tia = _enable_player(tia, 0, x=50)
    tia = _advance_one_scanline(tia)
    assert (tia_peek(tia, R_CXM1P) & 0x80) != 0    # M1-P0


def test_p0_pf_overlap_sets_cxp0fb_d7():
    tia = initial_tia_state()
    tia = _enable_player(tia, 0, x=0)             # P0 at 0..7
    tia = tia_poke(tia, W_PF0, 0xF0)              # playfield bits 0..3 → pixels 0..15
    tia = _advance_one_scanline(tia)
    assert (tia_peek(tia, R_CXP0FB) & 0x80) != 0   # P0-PF
    assert (tia_peek(tia, R_CXP0FB) & 0x40) == 0   # no BL


def test_p0_bl_overlap_sets_cxp0fb_d6():
    tia = initial_tia_state()
    tia = _enable_player(tia, 0, x=50)
    tia = _enable_ball(tia, x=50)
    tia = _advance_one_scanline(tia)
    assert (tia_peek(tia, R_CXP0FB) & 0x40) != 0   # P0-BL
    assert (tia_peek(tia, R_CXP0FB) & 0x80) == 0   # no PF


def test_p1_pf_overlap_sets_cxp1fb_d7():
    tia = initial_tia_state()
    tia = _enable_player(tia, 1, x=0)
    tia = tia_poke(tia, W_PF0, 0xF0)
    tia = _advance_one_scanline(tia)
    assert (tia_peek(tia, R_CXP1FB) & 0x80) != 0


def test_m0_pf_overlap_sets_cxm0fb_d7():
    tia = initial_tia_state()
    tia = _enable_missile(tia, 0, x=0)
    tia = tia_poke(tia, W_PF0, 0xF0)
    tia = _advance_one_scanline(tia)
    assert (tia_peek(tia, R_CXM0FB) & 0x80) != 0


def test_bl_pf_overlap_sets_cxblpf_d7():
    tia = initial_tia_state()
    tia = _enable_ball(tia, x=0)
    tia = tia_poke(tia, W_PF0, 0xF0)
    tia = _advance_one_scanline(tia)
    assert (tia_peek(tia, R_CXBLPF) & 0x80) != 0
    # D6 of CXBLPF is unused.
    assert (tia_peek(tia, R_CXBLPF) & 0x40) == 0


# --------------------------------------------------------------------------- #
# Latch semantics — set bits persist; CXCLR resets
# --------------------------------------------------------------------------- #

def test_collision_latch_persists_across_scanlines():
    """Once set, a collision bit stays set until CXCLR."""
    tia = initial_tia_state()
    tia = _enable_player(tia, 0, x=50)
    tia = _enable_player(tia, 1, x=50)
    tia = _advance_one_scanline(tia)            # collision happens here
    assert (tia_peek(tia, R_CXPPMM) & 0x80) != 0
    # Now move P1 so they don't overlap, advance another scanline.
    tia = tia._replace(p1_x=100)
    tia = _advance_one_scanline(tia)
    # Latch should still be set.
    assert (tia_peek(tia, R_CXPPMM) & 0x80) != 0


def test_cxclr_clears_after_collision():
    tia = initial_tia_state()
    tia = _enable_player(tia, 0, x=50)
    tia = _enable_player(tia, 1, x=50)
    tia = _advance_one_scanline(tia)
    assert (tia_peek(tia, R_CXPPMM) & 0x80) != 0
    tia = tia_poke(tia, W_CXCLR, 0x00)
    for reg in range(0x30, 0x38):
        assert tia_peek(tia, reg) == 0
