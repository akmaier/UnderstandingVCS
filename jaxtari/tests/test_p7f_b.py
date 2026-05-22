"""P7f-b tests — differentiable player sprites in the SOFT TIA render.

P0 / P1 composite over the playfield with standard TIA priority (P0 on
top). Each player is a single 1x-wide 8-pixel sprite. Per the SOFT-mode
convention, the RESP0 / RESP1 cells hold the player X position.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxtari.diff import (
    SoftBus,
    initial_soft_cpu_state,
    soft_render_scanline,
    soft_run,
)

# TIA register offsets (= SOFT-bus RAM cells).
_COLUP0 = 0x06
_COLUP1 = 0x07
_COLUPF = 0x08
_COLUBK = 0x09
_REFP0  = 0x0B
_REFP1  = 0x0C
_PF0    = 0x0D
_RESP0  = 0x10
_RESP1  = 0x11
_GRP0   = 0x1B
_GRP1   = 0x1C


def _rom_with(bytes_: list[int], size: int = 256) -> jnp.ndarray:
    rom = jnp.zeros((size,), dtype=jnp.float32)
    for i, b in enumerate(bytes_):
        rom = rom.at[i].set(jnp.float32(b))
    return rom


def _bus(regs: dict) -> SoftBus:
    """SoftBus with the given TIA register cells set (offset int → value)."""
    ram = jnp.zeros((128,), dtype=jnp.float32)
    for offset, value in regs.items():
        ram = ram.at[int(offset)].set(jnp.float32(value))
    return SoftBus(ram=ram, rom=jnp.zeros((256,), dtype=jnp.float32))


# --------------------------------------------------------------------------- #
# Forward — player rendering
# --------------------------------------------------------------------------- #

def test_no_player_when_grp_zero():
    """GRP0 = 0 → player 0 paints nothing; the scanline stays background."""
    bus = _bus({_COLUBK: 0x00, _GRP0: 0x00, _RESP0: 0x20})
    scan = soft_render_scanline(bus)
    assert jnp.all(scan == 0x00)


def test_solid_player_paints_eight_pixels():
    """GRP0 = 0xFF → 8 contiguous player pixels at X."""
    bus = _bus({_COLUBK: 0x00, _COLUP0: 0x3A, _GRP0: 0xFF, _RESP0: 0x20})
    scan = soft_render_scanline(bus)
    assert jnp.all(scan[0x20:0x28] == 0x3A)        # the 8 sprite pixels
    assert scan[0x1F] == 0x00                       # just left of the sprite
    assert scan[0x28] == 0x00                       # just right of the sprite


def test_grp_bit7_is_leftmost_pixel():
    """GRP0 = 0x80 (only bit 7) → just the leftmost sprite pixel is drawn."""
    bus = _bus({_COLUBK: 0x00, _COLUP0: 0xFF, _GRP0: 0x80, _RESP0: 0x30})
    scan = soft_render_scanline(bus)
    assert scan[0x30] == 0xFF                        # bit 7 → pixel 0
    assert jnp.all(scan[0x31:0x38] == 0x00)


def test_grp_pattern_0xaa_alternates():
    """GRP0 = 0xAA = 10101010 → sprite pixels 0,2,4,6 on; 1,3,5,7 off."""
    bus = _bus({_COLUBK: 0x00, _COLUP0: 0x0E, _GRP0: 0xAA, _RESP0: 0x10})
    scan = soft_render_scanline(bus)
    for i in range(8):
        expected = 0x0E if (i % 2 == 0) else 0x00
        assert scan[0x10 + i] == expected


def test_refp_reflects_the_sprite():
    """REFP0 bit 3 set → bit order reverses. 0x80 reflected → rightmost pixel."""
    bus = _bus({_COLUBK: 0x00, _COLUP0: 0xFF, _GRP0: 0x80,
                  _RESP0: 0x30, _REFP0: 0x08})
    scan = soft_render_scanline(bus)
    assert scan[0x37] == 0xFF                        # reflected → pixel 7
    assert jnp.all(scan[0x30:0x37] == 0x00)


def test_player_wraps_around_right_edge():
    """A sprite placed near the right edge wraps mod 160."""
    bus = _bus({_COLUBK: 0x00, _COLUP0: 0x22, _GRP0: 0xFF, _RESP0: 158})
    scan = soft_render_scanline(bus)
    assert scan[158] == 0x22
    assert scan[159] == 0x22
    assert scan[0] == 0x22                           # wrapped
    assert scan[1] == 0x22


# --------------------------------------------------------------------------- #
# Priority
# --------------------------------------------------------------------------- #

def test_player_draws_over_playfield():
    """Player pixels overwrite the playfield underneath."""
    bus = _bus({_COLUBK: 0x00, _COLUPF: 0x44, _PF0: 0xF0,   # playfield on
                  _COLUP0: 0x99, _GRP0: 0xFF, _RESP0: 0x04})
    scan = soft_render_scanline(bus)
    # Pixels 4..11 are playfield; the player sits at 4..11 and wins.
    assert jnp.all(scan[0x04:0x0C] == 0x99)


def test_p0_draws_over_p1():
    """Where P0 and P1 overlap, P0 wins (higher priority)."""
    bus = _bus({_COLUBK: 0x00,
                  _COLUP0: 0x11, _GRP0: 0xFF, _RESP0: 0x20,
                  _COLUP1: 0x77, _GRP1: 0xFF, _RESP1: 0x20})
    scan = soft_render_scanline(bus)
    assert jnp.all(scan[0x20:0x28] == 0x11)          # P0 on top


def test_p1_visible_where_p0_absent():
    bus = _bus({_COLUBK: 0x00,
                  _COLUP0: 0x11, _GRP0: 0xFF, _RESP0: 0x20,
                  _COLUP1: 0x77, _GRP1: 0xFF, _RESP1: 0x40})
    scan = soft_render_scanline(bus)
    assert jnp.all(scan[0x20:0x28] == 0x11)          # P0
    assert jnp.all(scan[0x40:0x48] == 0x77)          # P1, no overlap


# --------------------------------------------------------------------------- #
# Gradient — ∂pixel / ∂ROM through a player colour
# --------------------------------------------------------------------------- #

def test_grad_player_pixel_one_hot_at_colour_byte():
    """A four-instruction program positions player 0, gives it a solid
    pattern, and sets its colour from an immediate. `jax.grad` of a
    sprite pixel is one-hot at the COLUP0 immediate byte."""
    # LDA #$20 / STA RESP0   — player 0 X = 0x20
    # LDA #$FF / STA GRP0    — solid sprite
    # LDA #$0C / STA COLUP0  — colour
    rom_init = _rom_with([0xA9, 0x20, 0x85, _RESP0,
                          0xA9, 0xFF, 0x85, _GRP0,
                          0xA9, 0x0C, 0x85, _COLUP0])

    def simulator(rom_arr):
        bus = SoftBus(ram=jnp.zeros((128,), dtype=jnp.float32), rom=rom_arr)
        state = initial_soft_cpu_state()
        state, bus = soft_run(state, bus, n_steps=6)
        return soft_render_scanline(bus)[0x24]       # inside the sprite

    assert float(simulator(rom_init)) == 0x0C
    grad = jax.grad(simulator)(rom_init)
    # rom[9] is the COLUP0 immediate.
    assert float(grad[9]) == pytest.approx(1.0)
    other = float(jnp.sum(jnp.abs(grad))) - float(jnp.abs(grad[9]))
    assert other == pytest.approx(0.0)


def test_grad_background_pixel_not_affected_by_player_colour():
    """A pixel outside the sprite still attributes to COLUBK, not COLUP0."""
    rom_init = _rom_with([0xA9, 0x20, 0x85, _RESP0,
                          0xA9, 0xFF, 0x85, _GRP0,
                          0xA9, 0x0C, 0x85, _COLUP0,
                          0xA9, 0x1E, 0x85, _COLUBK])

    def simulator(rom_arr):
        bus = SoftBus(ram=jnp.zeros((128,), dtype=jnp.float32), rom=rom_arr)
        state = initial_soft_cpu_state()
        state, bus = soft_run(state, bus, n_steps=8)
        return soft_render_scanline(bus)[0x00]       # far from the sprite

    assert float(simulator(rom_init)) == 0x1E
    grad = jax.grad(simulator)(rom_init)
    assert float(grad[13]) == pytest.approx(1.0)     # COLUBK immediate
    assert float(grad[9]) == pytest.approx(0.0)      # COLUP0 immediate — no effect
