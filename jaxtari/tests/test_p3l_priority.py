"""P3l tests — CTRLPF.D2 (PFP) priority swap.

When the PFP bit is clear (default) the player paints on top of the
playfield, the canonical Atari priority. When PFP is set, the
playfield + ball composite on top of players + missiles, which is what
e.g. Pong's net and Combat's tank-track maze rely on so a player can't
visually drive *through* the playfield.

Both the HARD renderer (`jaxtari.tia.system.render_scanline`) and the
SOFT renderer (`jaxtari.diff.soft_tia.soft_render_scanline`) honour
the bit; this file covers both.
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
    W_COLUPF,
    W_CTRLPF,
    W_ENABL,
    W_GRP0,
    W_PF1,
    W_RESBL,
    W_RESP0,
)


# --------------------------------------------------------------------------- #
# HARD-mode priority
# --------------------------------------------------------------------------- #

def _setup_pf_vs_p0(pfp: bool):
    """Place a one-byte playfield bit at column 4..7 (PF0 bit 4) and a
    full-bright P0 at column 4. The two objects overlap. The renderer's
    job: pick the right priority.
    """
    tia = initial_tia_state()._replace(p0_x=4)
    tia = tia_poke(tia, W_COLUBK, 0x00)
    tia = tia_poke(tia, W_COLUPF, 0x42)
    tia = tia_poke(tia, W_COLUP0, 0x84)
    tia = tia_poke(tia, W_GRP0,   0xFF)    # all 8 player bits on
    tia = tia_poke(tia, W_PF1,    0x80)    # PF1 bit 7 → playfield pixels 16..19
    tia = tia_poke(tia, W_CTRLPF, 0x04 if pfp else 0x00)
    return tia


def test_default_priority_player_above_playfield():
    """PFP=0: P0 paints on top of the playfield."""
    tia = _setup_pf_vs_p0(pfp=False)
    scan = render_scanline(tia)
    # PF1 bit 0 = right-most playfield bit on the LEFT half, covering
    # columns 16..19. The player sits at columns 4..11 — no overlap, so
    # the player is exactly its colour and the PF bit is exactly COLUPF.
    assert int(scan[4]) == 0x84
    assert int(scan[16]) == 0x42

    # Now overlap them: drop the player onto the playfield strip.
    tia = tia._replace(p0_x=16)
    scan = render_scanline(tia)
    assert int(scan[16]) == 0x84  # player wins (default priority)


def test_pfp_priority_playfield_above_player():
    """PFP=1: playfield wins over P0 in the overlap."""
    tia = _setup_pf_vs_p0(pfp=True)._replace(p0_x=16)
    scan = render_scanline(tia)
    # In the overlap (col 16..19), PFP=1 says playfield wins.
    assert int(scan[16]) == 0x42
    assert int(scan[19]) == 0x42
    # Outside the playfield (col 20..23), the player still paints.
    assert int(scan[20]) == 0x84
    assert int(scan[23]) == 0x84


def test_pfp_priority_ball_above_player():
    """PFP=1: the ball (which uses COLUPF) also paints on top of P0."""
    tia = initial_tia_state()._replace(p0_x=20, bl_x=20)
    tia = tia_poke(tia, W_COLUBK, 0x00)
    tia = tia_poke(tia, W_COLUPF, 0x42)
    tia = tia_poke(tia, W_COLUP0, 0x84)
    tia = tia_poke(tia, W_GRP0,   0xFF)
    tia = tia_poke(tia, W_ENABL,  0x02)            # ball on
    tia = tia_poke(tia, W_CTRLPF, 0x04)            # PFP set, default-size ball
    scan = render_scanline(tia)
    # Ball is 1 pixel wide at col 20; in PFP mode it wins over the player.
    assert int(scan[20]) == 0x42
    # The rest of the player (col 21..27) is still visible.
    assert int(scan[21]) == 0x84


def test_pfp_background_unchanged_in_clear_region():
    """PFP=1 still leaves the background untouched in cells where
    neither playfield, ball nor sprite paints."""
    tia = _setup_pf_vs_p0(pfp=True)._replace(p0_x=16)
    tia = tia_poke(tia, W_COLUBK, 0x55)
    scan = render_scanline(tia)
    # Column 60 sits well clear of every object — must remain COLUBK.
    assert int(scan[60]) == 0x55


# --------------------------------------------------------------------------- #
# SOFT-mode priority
# --------------------------------------------------------------------------- #

def _soft_bus_p0_over_pf(pfp: bool):
    """SOFT analogue of _setup_pf_vs_p0 with P0 / playfield overlapping
    at column 16. The SOFT bus stores TIA registers in ram[0x00..0x3F],
    and (by SOFT convention) RESP0 carries the player X position."""
    ram = jnp.zeros((128,), dtype=jnp.float32)
    rom = jnp.zeros((128,), dtype=jnp.float32)
    ram = ram.at[W_COLUBK].set(0.0)
    ram = ram.at[W_COLUPF].set(0x42)
    ram = ram.at[W_COLUP0].set(0x84)
    ram = ram.at[W_GRP0].set(0xFF)
    ram = ram.at[W_PF1].set(0x80)           # PF1 bit 7 → playfield columns 16..19
    ram = ram.at[W_RESP0].set(16.0)         # P0 at column 16
    ram = ram.at[W_CTRLPF].set(0x04 if pfp else 0x00)
    return SoftBus(ram=ram, rom=rom)


def test_soft_default_priority_player_above_playfield():
    bus = _soft_bus_p0_over_pf(pfp=False)
    scan = soft_render_scanline(bus)
    # Overlap column: player wins (default).
    assert int(scan[16]) == 0x84


def test_soft_pfp_priority_playfield_above_player():
    bus = _soft_bus_p0_over_pf(pfp=True)
    scan = soft_render_scanline(bus)
    # Overlap column: playfield wins under PFP.
    assert int(scan[16]) == 0x42
    # Just past the playfield strip — player still paints.
    assert int(scan[20]) == 0x84


def test_soft_colupf_gradient_under_pfp():
    """In PFP=1 mode, increasing COLUPF must still increase a pixel that
    sits under the playfield (where the playfield now wins over a
    sprite). The gradient w.r.t. COLUPF at that pixel must be 1.0.
    """
    bus0 = _soft_bus_p0_over_pf(pfp=True)

    def pixel(ram):
        b = SoftBus(ram=ram, rom=bus0.rom)
        return soft_render_scanline(b)[16]                       # overlap col

    g = jax.grad(pixel)(bus0.ram)
    assert float(g[W_COLUPF]) == 1.0


def test_soft_colup0_gradient_outside_overlap():
    """COLUP0 still drives the non-overlap player pixels in either mode."""
    bus0 = _soft_bus_p0_over_pf(pfp=True)

    def pixel(ram):
        b = SoftBus(ram=ram, rom=bus0.rom)
        return soft_render_scanline(b)[20]                       # past PF strip

    g = jax.grad(pixel)(bus0.ram)
    assert float(g[W_COLUP0]) == 1.0
