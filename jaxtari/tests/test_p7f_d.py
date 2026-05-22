"""P7f-d tests — TIA collision detection in the SOFT render.

`soft_collision_registers(bus)` returns the 8 CX latch registers for
the current scanline. Two objects collide when their `(160,)` masks
overlap at any pixel; each register packs its hit flags into D7 / D6.
"""

import jax.numpy as jnp

from jaxtari.diff import SoftBus, soft_collision_registers

# TIA register offsets (= SOFT-bus RAM cells).
_NUSIZ0 = 0x04
_CTRLPF = 0x0A
_PF0    = 0x0D
_RESP0  = 0x10
_RESP1  = 0x11
_RESM0  = 0x12
_RESM1  = 0x13
_RESBL  = 0x14
_GRP0   = 0x1B
_GRP1   = 0x1C
_ENAM0  = 0x1D
_ENAM1  = 0x1E
_ENABL  = 0x1F

# CX register indices in the returned (8,) array.
_CXM0P, _CXM1P, _CXP0FB, _CXP1FB = 0, 1, 2, 3
_CXM0FB, _CXM1FB, _CXBLPF, _CXPPMM = 4, 5, 6, 7


def _bus(regs: dict) -> SoftBus:
    ram = jnp.zeros((128,), dtype=jnp.float32)
    for offset, value in regs.items():
        ram = ram.at[int(offset)].set(jnp.float32(value))
    return SoftBus(ram=ram, rom=jnp.zeros((256,), dtype=jnp.float32))


# --------------------------------------------------------------------------- #
# Shape + empty
# --------------------------------------------------------------------------- #

def test_collision_registers_shape():
    cx = soft_collision_registers(_bus({}))
    assert cx.shape == (8,)


def test_no_objects_no_collisions():
    cx = soft_collision_registers(_bus({}))
    assert jnp.all(cx == 0.0)


def test_objects_present_but_disjoint_do_not_collide():
    """P0 at 0x10, P1 at 0x40 — present but never overlapping."""
    cx = soft_collision_registers(_bus({
        _GRP0: 0xFF, _RESP0: 0x10,
        _GRP1: 0xFF, _RESP1: 0x40,
    }))
    assert float(cx[_CXPPMM]) == 0.0


# --------------------------------------------------------------------------- #
# Player–player
# --------------------------------------------------------------------------- #

def test_p0_p1_overlap_sets_cxppmm_d7():
    """Two solid players at the same X → CXPPMM D7 (P0-P1)."""
    cx = soft_collision_registers(_bus({
        _GRP0: 0xFF, _RESP0: 0x20,
        _GRP1: 0xFF, _RESP1: 0x20,
    }))
    assert int(cx[_CXPPMM]) & 0x80      # P0-P1 hit
    assert not (int(cx[_CXPPMM]) & 0x40)   # no M0-M1


# --------------------------------------------------------------------------- #
# Missile–missile / missile–player
# --------------------------------------------------------------------------- #

def test_m0_m1_overlap_sets_cxppmm_d6():
    cx = soft_collision_registers(_bus({
        _RESM0: 0x30, _ENAM0: 0x02, _NUSIZ0: 0x30,   # M0 8-wide at 0x30
        _RESM1: 0x30, _ENAM1: 0x02,
    }))
    assert int(cx[_CXPPMM]) & 0x40      # M0-M1 hit


def test_m0_over_p0_sets_cxm0p_d6():
    """M0 overlapping P0 → CXM0P D6."""
    cx = soft_collision_registers(_bus({
        _GRP0: 0xFF, _RESP0: 0x20,
        _RESM0: 0x20, _ENAM0: 0x02, _NUSIZ0: 0x30,
    }))
    assert int(cx[_CXM0P]) & 0x40       # M0-P0
    assert not (int(cx[_CXM0P]) & 0x80)    # no M0-P1


def test_m0_over_p1_sets_cxm0p_d7():
    cx = soft_collision_registers(_bus({
        _GRP1: 0xFF, _RESP1: 0x20,
        _RESM0: 0x20, _ENAM0: 0x02, _NUSIZ0: 0x30,
    }))
    assert int(cx[_CXM0P]) & 0x80       # M0-P1


# --------------------------------------------------------------------------- #
# Player–playfield / player–ball
# --------------------------------------------------------------------------- #

def test_p0_over_playfield_sets_cxp0fb_d7():
    """P0 placed over a lit playfield region → CXP0FB D7 (P0-PF)."""
    cx = soft_collision_registers(_bus({
        _PF0: 0xF0,                      # playfield pixels 0..15 lit
        _GRP0: 0xFF, _RESP0: 0x04,
    }))
    assert int(cx[_CXP0FB]) & 0x80      # P0-PF


def test_p0_over_ball_sets_cxp0fb_d6():
    cx = soft_collision_registers(_bus({
        _RESBL: 0x20, _ENABL: 0x02, _CTRLPF: 0x30,   # 8-wide ball at 0x20
        _GRP0: 0xFF, _RESP0: 0x20,
    }))
    assert int(cx[_CXP0FB]) & 0x40      # P0-BL


# --------------------------------------------------------------------------- #
# Ball–playfield / missile–playfield
# --------------------------------------------------------------------------- #

def test_ball_over_playfield_sets_cxblpf_d7():
    cx = soft_collision_registers(_bus({
        _PF0: 0xF0,
        _RESBL: 0x04, _ENABL: 0x02, _CTRLPF: 0x30,
    }))
    assert int(cx[_CXBLPF]) & 0x80      # BL-PF


def test_m0_over_playfield_sets_cxm0fb_d7():
    cx = soft_collision_registers(_bus({
        _PF0: 0xF0,
        _RESM0: 0x04, _ENAM0: 0x02, _NUSIZ0: 0x30,
    }))
    assert int(cx[_CXM0FB]) & 0x80      # M0-PF


def test_disabled_missile_does_not_collide():
    """A missile with ENAM clear has an all-zero mask — no collisions."""
    cx = soft_collision_registers(_bus({
        _PF0: 0xF0,
        _RESM0: 0x04, _ENAM0: 0x00, _NUSIZ0: 0x30,   # disabled
    }))
    assert float(cx[_CXM0FB]) == 0.0
