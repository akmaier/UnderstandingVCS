"""P3g tests — NUSIZ multi-copy + 2×/4×-wide player scaling.

NUSIZ0 / NUSIZ1 low 3 bits select one of 8 sprite layouts (1 / 2 / 3
copies with various spacings, or 1 copy at 2× / 4× width). Pre-P3g
the SOFT and HARD renderers ignored the multi-copy bits entirely —
every sprite was a single 1× copy. This file pins down the new
behaviour for both renderers.
"""

import jax
import jax.numpy as jnp

from jaxtari.diff.soft_state import SoftBus
from jaxtari.diff.soft_tia import soft_render_scanline
from jaxtari.tia.system import (
    initial_tia_state,
    render_scanline,
    tia_poke,
    W_COLUBK,
    W_COLUP0,
    W_COLUP1,
    W_ENAM0,
    W_GRP0,
    W_GRP1,
    W_NUSIZ0,
    W_NUSIZ1,
    W_RESM0,
    W_RESP0,
)


# --------------------------------------------------------------------------- #
# HARD-mode multi-copy
# --------------------------------------------------------------------------- #

def _setup_p0(nusiz: int, x: int = 0):
    """One player with all 8 bits set (so each copy is a solid 8-pixel
    bar at 1× scale; or 16/32 pixels at 2×/4×), at position `x`."""
    tia = initial_tia_state()._replace(p0_x=x)
    tia = tia_poke(tia, W_COLUBK, 0x00)
    tia = tia_poke(tia, W_COLUP0, 0x42)
    tia = tia_poke(tia, W_GRP0,   0xFF)
    tia = tia_poke(tia, W_NUSIZ0, nusiz)
    return tia


def test_hard_nusiz_000_single_copy():
    tia = _setup_p0(0b000, x=0)
    scan = render_scanline(tia)
    # One 8-pixel copy starting at col 0.
    assert all(int(scan[i]) == 0x42 for i in range(8))
    assert int(scan[8]) == 0
    assert int(scan[16]) == 0          # no second copy in mode 0


def test_hard_nusiz_001_two_copies_close():
    """Two copies, 16 pixels apart (center-to-center). First copy at
    col 0..7; second copy at col 16..23."""
    tia = _setup_p0(0b001, x=0)
    scan = render_scanline(tia)
    assert int(scan[0]) == 0x42
    assert int(scan[7]) == 0x42
    assert int(scan[8]) == 0
    assert int(scan[15]) == 0
    assert int(scan[16]) == 0x42      # second copy
    assert int(scan[23]) == 0x42
    assert int(scan[24]) == 0


def test_hard_nusiz_010_two_copies_medium():
    """Two copies, 32 pixels apart. col 0..7 + col 32..39."""
    tia = _setup_p0(0b010, x=0)
    scan = render_scanline(tia)
    assert int(scan[0]) == 0x42
    assert int(scan[7]) == 0x42
    assert int(scan[31]) == 0
    assert int(scan[32]) == 0x42
    assert int(scan[39]) == 0x42
    assert int(scan[40]) == 0


def test_hard_nusiz_011_three_copies_close():
    """Three copies, 16 apart: col 0-7, 16-23, 32-39."""
    tia = _setup_p0(0b011, x=0)
    scan = render_scanline(tia)
    for base in (0, 16, 32):
        for i in range(8):
            assert int(scan[base + i]) == 0x42
    assert int(scan[8])  == 0
    assert int(scan[24]) == 0
    assert int(scan[40]) == 0


def test_hard_nusiz_100_two_copies_wide():
    """Two copies, 64 apart: col 0-7 + col 64-71."""
    tia = _setup_p0(0b100, x=0)
    scan = render_scanline(tia)
    assert int(scan[0]) == 0x42
    assert int(scan[7]) == 0x42
    assert int(scan[63]) == 0
    assert int(scan[64]) == 0x42
    assert int(scan[71]) == 0x42
    assert int(scan[72]) == 0


def test_hard_nusiz_101_double_size():
    """One copy at 2× scale → 16-pixel-wide player. Task #103/NUSIZ: in
    double/quad-size mode the player output is delayed by 1 pixel (xitari
    `computePlayerMaskTable`; mirrored in both ports' `_overlay_player`), so
    at x=0 the 2× player covers cols 1..16, not 0..15."""
    tia = _setup_p0(0b101, x=0)
    scan = render_scanline(tia)
    for i in range(1, 17):
        assert int(scan[i]) == 0x42, f"col {i} expected 0x42, got {int(scan[i]):02x}"
    assert int(scan[0]) == 0          # +1 double-size delay
    assert int(scan[17]) == 0
    # And no second copy.
    assert int(scan[32]) == 0


def test_hard_nusiz_110_three_copies_medium():
    """Three copies, 32 apart: col 0-7, 32-39, 64-71."""
    tia = _setup_p0(0b110, x=0)
    scan = render_scanline(tia)
    for base in (0, 32, 64):
        for i in range(8):
            assert int(scan[base + i]) == 0x42
    assert int(scan[8]) == 0
    assert int(scan[31]) == 0


def test_hard_nusiz_111_quad_size():
    """One copy at 4× scale: 8 bits × 4 = 32-pixel-wide player. Same +1
    double/quad-size delay as NUSIZ=5 → covers cols 1..32 at x=0."""
    tia = _setup_p0(0b111, x=0)
    scan = render_scanline(tia)
    for i in range(1, 33):
        assert int(scan[i]) == 0x42, f"col {i} expected 0x42, got {int(scan[i]):02x}"
    assert int(scan[0]) == 0          # +1 quad-size delay
    assert int(scan[33]) == 0


def test_hard_nusiz_double_size_pattern():
    """Double-size with a non-trivial GRP — verify each bit really does
    paint 2 pixels (bit 7 is leftmost). GRP=0x80 → only bit 7 set →
    2-pixel block at the player's base."""
    tia = initial_tia_state()._replace(p0_x=10)
    tia = tia_poke(tia, W_COLUBK, 0x00)
    tia = tia_poke(tia, W_COLUP0, 0x42)
    tia = tia_poke(tia, W_GRP0,   0x80)       # only bit 7
    tia = tia_poke(tia, W_NUSIZ0, 0b101)      # 2× scale
    scan = render_scanline(tia)
    # +1 double-size delay: base 10 → the 2-pixel block lands at cols 11,12.
    assert int(scan[11]) == 0x42
    assert int(scan[12]) == 0x42
    assert int(scan[10]) == 0
    assert int(scan[13]) == 0


def test_hard_missile_inherits_nusiz_multi_copy():
    """Missile 0 in NUSIZ mode 011 (3 close copies) should paint 3
    missile blocks at the same spacing as the player would."""
    tia = initial_tia_state()._replace(m0_x=0)
    tia = tia_poke(tia, W_COLUBK, 0x00)
    tia = tia_poke(tia, W_COLUP0, 0x55)
    tia = tia_poke(tia, W_ENAM0,  0x02)
    tia = tia_poke(tia, W_NUSIZ0, 0b011)      # 3 copies, 16 apart
    scan = render_scanline(tia)
    # Each missile copy is 1 pixel wide (NUSIZ bits 4-5 = 0). COLUP0=0x55 is
    # stored as `value & 0xFE` → 0x54 (xitari masks the unused LSB of COLU*),
    # so the missile paints 0x54. Missiles get NO double/quad +1 player delay,
    # so the 3 copies land exactly at cols 0, 16, 32.
    assert int(scan[0])  == 0x54
    assert int(scan[16]) == 0x54
    assert int(scan[32]) == 0x54
    # Gaps are background.
    assert int(scan[1])  == 0
    assert int(scan[15]) == 0


# --------------------------------------------------------------------------- #
# SOFT-mode multi-copy + scaling
# --------------------------------------------------------------------------- #

def _soft_bus_p0(nusiz: int, x: int = 0) -> SoftBus:
    ram = jnp.zeros((128,), dtype=jnp.float32)
    ram = ram.at[W_COLUBK].set(0.0)
    ram = ram.at[W_COLUP0].set(0x42)
    ram = ram.at[W_GRP0].set(0xFF)
    ram = ram.at[W_NUSIZ0].set(nusiz)
    ram = ram.at[0x10].set(x)              # W_RESP0 holds X by SOFT convention
    return SoftBus(ram=ram, rom=jnp.zeros((128,), dtype=jnp.float32))


def test_soft_nusiz_011_three_close_copies():
    bus = _soft_bus_p0(0b011, x=0)
    scan = soft_render_scanline(bus)
    for base in (0, 16, 32):
        for i in range(8):
            assert int(scan[base + i]) == 0x42, f"col {base + i}"
    assert int(scan[8])  == 0
    assert int(scan[24]) == 0


def test_soft_nusiz_111_quad_size():
    bus = _soft_bus_p0(0b111, x=0)
    scan = soft_render_scanline(bus)
    for i in range(32):
        assert int(scan[i]) == 0x42, f"col {i}"
    assert int(scan[32]) == 0


def test_soft_colup0_gradient_reaches_second_copy():
    """In NUSIZ mode 001 (2 copies), changing COLUP0 must change BOTH
    copies. The gradient ∂(pixel_at_first_copy)/∂COLUP0 should be 1.0,
    and ditto for pixels inside the second copy."""
    bus0 = _soft_bus_p0(0b001, x=0)

    def pix_at(c):
        return lambda ram: soft_render_scanline(
            SoftBus(ram=ram, rom=bus0.rom))[c]

    g_first  = jax.grad(pix_at(4))(bus0.ram)
    g_second = jax.grad(pix_at(20))(bus0.ram)
    assert float(g_first[W_COLUP0])  == 1.0
    assert float(g_second[W_COLUP0]) == 1.0


def test_soft_nusiz_000_default_single_copy_unchanged():
    """NUSIZ=0 should produce a single 8-pixel copy — same as pre-P3g
    behaviour. Regression guard."""
    bus = _soft_bus_p0(0b000, x=0)
    scan = soft_render_scanline(bus)
    for i in range(8):
        assert int(scan[i]) == 0x42
    assert int(scan[8])  == 0
    assert int(scan[16]) == 0
