"""P3h tests — VDELP0 / VDELP1 / VDELBL vertical-delay sprite updates.

When VDELPx bit 0 is set, the player renders from the OLD GRPx — the
value captured the last time the OTHER player's GRP was written.
That's the classic Atari trick for two-frame animations that swap
sets of player graphics atomically.

VDELBL is the ball's analogue, but its shadow updates on GRP1 writes
(not on ENABL writes — that's the TIA hardware convention; Stella
documents it as the ball-delay being driven by a strobe tied into
the GRP1 latch).
"""

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
    W_GRP0,
    W_GRP1,
    W_VDELBL,
    W_VDELP0,
    W_VDELP1,
)


# --------------------------------------------------------------------------- #
# Shadow latch semantics — writes to GRP land in the OTHER's shadow
# --------------------------------------------------------------------------- #

def test_grp1_write_latches_current_grp0_into_shadow():
    tia = initial_tia_state()
    tia = tia_poke(tia, W_GRP0, 0xAA)        # current GRP0 = 0xAA
    tia = tia_poke(tia, W_GRP1, 0x55)        # writing GRP1 captures 0xAA → grp0_old
    assert int(tia.grp0_old) == 0xAA


def test_grp0_write_latches_current_grp1_into_shadow():
    tia = initial_tia_state()
    tia = tia_poke(tia, W_GRP1, 0xBB)
    tia = tia_poke(tia, W_GRP0, 0x44)
    assert int(tia.grp1_old) == 0xBB


def test_grp1_write_also_latches_enabl_old():
    """GRP1 writes also latch the current ENABL for the ball's VDELBL
    delay — xitari/Stella convention."""
    tia = initial_tia_state()
    tia = tia_poke(tia, W_ENABL, 0x02)        # current ENABL = enabled
    tia = tia_poke(tia, W_GRP1, 0x00)
    assert int(tia.enabl_old) == 0x02


def test_grp0_write_does_not_touch_enabl_shadow():
    tia = initial_tia_state()
    tia = tia_poke(tia, W_ENABL, 0x02)
    tia = tia_poke(tia, W_GRP0, 0x00)        # only latches grp1_old
    assert int(tia.enabl_old) == 0


# --------------------------------------------------------------------------- #
# Rendering — VDELP swaps live ↔ shadow GRP
# --------------------------------------------------------------------------- #

def _setup_p0(grp_current: int, grp_old: int | None,
              vdelp0: int) -> "TIAState":
    """Helper: P0 at column 4, colour 0x42, with the given current GRP0
    and (optionally) shadow GRP0_old. `vdelp0` controls VDELP0."""
    tia = initial_tia_state()._replace(p0_x=4)
    tia = tia_poke(tia, W_COLUBK, 0x00)
    tia = tia_poke(tia, W_COLUP0, 0x42)
    tia = tia_poke(tia, W_VDELP0, vdelp0 & 1)
    tia = tia_poke(tia, W_GRP0, grp_current)
    if grp_old is not None:
        tia = tia._replace(grp0_old=grp_old)
    return tia


def test_vdelp0_clear_renders_current_grp0():
    tia = _setup_p0(grp_current=0xFF, grp_old=0x00, vdelp0=0)
    scan = render_scanline(tia)
    # 8-pixel bar at col 4..11 in colour 0x42.
    assert int(scan[4]) == 0x42
    assert int(scan[11]) == 0x42


def test_vdelp0_set_renders_shadow_grp0_not_current():
    """VDELP0=1 + grp0_old=0x00 + current GRP0=0xFF → renders the
    SHADOW (= empty), so the player is invisible."""
    tia = _setup_p0(grp_current=0xFF, grp_old=0x00, vdelp0=1)
    scan = render_scanline(tia)
    assert int(scan[4]) == 0          # invisible — shadow is empty


def test_vdelp0_set_renders_shadow_when_shadow_has_bits():
    """VDELP0=1 + grp0_old=0x80 (bit 7 only) → only the leftmost
    pixel paints."""
    tia = _setup_p0(grp_current=0x00, grp_old=0x80, vdelp0=1)
    scan = render_scanline(tia)
    assert int(scan[4]) == 0x42
    assert int(scan[5]) == 0


def test_vdelp0_doesnt_affect_p1():
    """Sanity: setting VDELP0 only changes P0 rendering, P1 is
    unaffected."""
    tia = initial_tia_state()._replace(p0_x=4, p1_x=20)
    tia = tia_poke(tia, W_COLUBK, 0x00)
    tia = tia_poke(tia, W_COLUP0, 0x42)
    tia = tia_poke(tia, W_COLUP1, 0x84)
    tia = tia_poke(tia, W_VDELP0, 0x01)
    tia = tia_poke(tia, W_GRP1, 0xFF)        # P1 visible
    tia = tia_poke(tia, W_GRP0, 0xFF)        # P0 visible (but VDELP0 → shadow=0 → invisible)
    scan = render_scanline(tia)
    assert int(scan[4]) == 0                  # P0 invisible via VDELP0
    assert int(scan[20]) == 0x84              # P1 unaffected


# --------------------------------------------------------------------------- #
# VDELBL — ball uses shadow ENABL
# --------------------------------------------------------------------------- #

def test_vdelbl_clear_renders_current_enabl():
    # NB (test nudge 2026-06-12): COLU* writes store `value & 0xFE`
    # (xitari color-loss LSB mask) — COLUPF=$33 reads back as $32.
    tia = initial_tia_state()._replace(bl_x=10)
    tia = tia_poke(tia, W_COLUBK, 0x00)
    tia = tia_poke(tia, W_COLUPF, 0x33)
    tia = tia_poke(tia, W_ENABL, 0x02)        # enabled
    scan = render_scanline(tia)
    assert int(scan[10]) == 0x32


def test_vdelbl_set_uses_shadow_enabl():
    """VDELBL=1, shadow ENABL=0, current ENABL=0x02 → ball
    invisible."""
    tia = initial_tia_state()._replace(bl_x=10, enabl_old=0)
    tia = tia_poke(tia, W_COLUBK, 0x00)
    tia = tia_poke(tia, W_COLUPF, 0x33)
    tia = tia_poke(tia, W_VDELBL, 0x01)
    tia = tia_poke(tia, W_ENABL, 0x02)        # current = enabled
    scan = render_scanline(tia)
    assert int(scan[10]) == 0                 # shadow says off


def test_vdelbl_set_with_enabled_shadow_paints_ball():
    """VDELBL=1, shadow ENABL=0x02, current ENABL=0 → ball still
    visible (shadow wins). COLUPF=$33 stores $32 (COLU LSB mask)."""
    tia = initial_tia_state()._replace(bl_x=10, enabl_old=0x02)
    tia = tia_poke(tia, W_COLUBK, 0x00)
    tia = tia_poke(tia, W_COLUPF, 0x33)
    tia = tia_poke(tia, W_VDELBL, 0x01)
    tia = tia_poke(tia, W_ENABL, 0x00)
    scan = render_scanline(tia)
    assert int(scan[10]) == 0x32


# --------------------------------------------------------------------------- #
# End-to-end "swap GRP" pattern — proves the shadow plumbing is
# actually wired the way real Atari ROMs expect.
# --------------------------------------------------------------------------- #

def test_two_frame_swap_idiom():
    """Real-Atari pattern (e.g. Pitfall): load both GRP buffers, set
    VDELP0/1, then alternately update each. Both players' visible
    bitmap should update *together* on every GRP1 write."""
    tia = initial_tia_state()._replace(p0_x=4, p1_x=20)
    tia = tia_poke(tia, W_COLUBK, 0x00)
    tia = tia_poke(tia, W_COLUP0, 0x42)
    tia = tia_poke(tia, W_COLUP1, 0x84)
    tia = tia_poke(tia, W_VDELP0, 0x01)
    tia = tia_poke(tia, W_VDELP1, 0x01)

    # Frame N: prime both buffers. With VDELP0=1, P0 renders the
    # shadow which is still 0 — nothing visible yet for P0.
    tia = tia_poke(tia, W_GRP0, 0xAA)         # current GRP0 = 0xAA, grp1_old ← 0
    tia = tia_poke(tia, W_GRP1, 0x55)         # current GRP1 = 0x55, grp0_old ← 0xAA
    # Now: GRP0 visible (via shadow) = 0xAA. GRP1 visible (via shadow) = 0.
    scan = render_scanline(tia)
    # 0xAA = 1010 1010 — bits 7,5,3,1 set, so cols 4,6,8,10 paint for P0.
    assert int(scan[4]) == 0x42
    assert int(scan[5]) == 0
    assert int(scan[6]) == 0x42
    # P1 not yet visible (its shadow is still 0).
    assert int(scan[20]) == 0
