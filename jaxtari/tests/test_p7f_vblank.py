"""Tests for VBLANK output-blanking in the SOFT renderer.

When VBLANK ($01) bit 1 is set, real TIA hardware forces the video
output to black. The SOFT renderer used to ignore the VBLANK flag
entirely (it was listed under "still deferred after P7f" in
STATUS.md); this batch wires it in.

The semantics: scanline = scanline * (1 - vblank_bit). The integer
extraction of the VBLANK bit breaks the gradient through that bit
itself, but gradient through colour / sprite registers stays exact
in both the blanked and unblanked branches.
"""

import jax
import jax.numpy as jnp

from jaxtari.diff.soft_state import SoftBus
from jaxtari.diff.soft_tia import soft_render_scanline


W_VBLANK = 0x01
W_COLUBK = 0x09
W_COLUPF = 0x08
W_PF1    = 0x0E


def _bus(*regs) -> SoftBus:
    ram = jnp.zeros((128,), dtype=jnp.float32)
    for addr, val in regs:
        ram = ram.at[addr].set(jnp.float32(val))
    return SoftBus(ram=ram, rom=jnp.zeros((128,), dtype=jnp.float32))


def test_vblank_clear_renders_normally():
    """VBLANK=0 → background colour visible."""
    bus = _bus((W_COLUBK, 0x42), (W_VBLANK, 0x00))
    scan = soft_render_scanline(bus)
    assert int(scan[0]) == 0x42
    assert int(scan[80]) == 0x42


def test_vblank_set_blanks_entire_scanline():
    """VBLANK bit 1 set → scanline is all-zeros."""
    bus = _bus((W_COLUBK, 0x42), (W_VBLANK, 0x02))
    scan = soft_render_scanline(bus)
    assert int(scan[0]) == 0
    assert int(scan[80]) == 0
    assert int(scan[159]) == 0


def test_vblank_blanks_playfield_too():
    """A full playfield + bright colour goes to zero under VBLANK."""
    bus = _bus(
        (W_COLUBK, 0x10), (W_COLUPF, 0x42),
        (W_PF1,    0xFF),                            # large PF block
        (W_VBLANK, 0x02),
    )
    scan = soft_render_scanline(bus)
    # Nothing visible.
    assert jnp.all(scan == 0)


def test_vblank_other_bits_dont_blank():
    """Only bit 1 of VBLANK matters for output-blanking; bit 6 / 7
    (latch / dump) are unrelated."""
    bus = _bus((W_COLUBK, 0x42), (W_VBLANK, 0xFD))   # 1111 1101 — bit 1 clear
    scan = soft_render_scanline(bus)
    assert int(scan[0]) == 0x42


def test_vblank_gradient_through_colubk_under_blank_is_zero():
    """Under VBLANK, ∂pixel/∂COLUBK should be zero — the
    (1 - vblank_bit) factor kills the colour signal."""
    bus0 = _bus((W_COLUBK, 0x42), (W_VBLANK, 0x02))

    def pixel(ram):
        return soft_render_scanline(SoftBus(ram=ram, rom=bus0.rom))[0]

    g = jax.grad(pixel)(bus0.ram)
    assert float(g[W_COLUBK]) == 0.0


def test_vblank_gradient_through_colubk_unblank_is_one():
    """No VBLANK → ∂pixel/∂COLUBK = 1 (the gradient we've always had)."""
    bus0 = _bus((W_COLUBK, 0x42), (W_VBLANK, 0x00))

    def pixel(ram):
        return soft_render_scanline(SoftBus(ram=ram, rom=bus0.rom))[0]

    g = jax.grad(pixel)(bus0.ram)
    assert float(g[W_COLUBK]) == 1.0
