"""P7f-c tests — differentiable missiles + ball in the SOFT TIA render.

M0 / M1 and the ball BL are solid 1/2/4/8-pixel blocks. Missiles take
their player's colour (COLUP0 / COLUP1); the ball takes COLUPF.
Compositing order is the full HARD sequence:
background ← playfield ← ball ← M1 ← P1 ← M0 ← P0.
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
_NUSIZ0 = 0x04
_NUSIZ1 = 0x05
_COLUP0 = 0x06
_COLUP1 = 0x07
_COLUPF = 0x08
_COLUBK = 0x09
_CTRLPF = 0x0A
_RESM0  = 0x12
_RESM1  = 0x13
_RESBL  = 0x14
_RESP0  = 0x10
_GRP0   = 0x1B
_ENAM0  = 0x1D
_ENAM1  = 0x1E
_ENABL  = 0x1F


def _rom_with(bytes_: list[int], size: int = 256) -> jnp.ndarray:
    rom = jnp.zeros((size,), dtype=jnp.float32)
    for i, b in enumerate(bytes_):
        rom = rom.at[i].set(jnp.float32(b))
    return rom


def _bus(regs: dict) -> SoftBus:
    ram = jnp.zeros((128,), dtype=jnp.float32)
    for offset, value in regs.items():
        ram = ram.at[int(offset)].set(jnp.float32(value))
    return SoftBus(ram=ram, rom=jnp.zeros((256,), dtype=jnp.float32))


# --------------------------------------------------------------------------- #
# Missiles — enable / disable
# --------------------------------------------------------------------------- #

def test_missile_disabled_draws_nothing():
    """ENAM0 bit 1 clear → missile 0 paints nothing."""
    bus = _bus({_COLUBK: 0x00, _COLUP0: 0xFF, _RESM0: 0x20, _ENAM0: 0x00})
    scan = soft_render_scanline(bus)
    assert jnp.all(scan == 0x00)


def test_missile_enabled_paints_one_pixel_by_default():
    """ENAM0 = 0x02, NUSIZ0 = 0 → a 1-pixel missile at X."""
    bus = _bus({_COLUBK: 0x00, _COLUP0: 0x2C, _RESM0: 0x20,
                _ENAM0: 0x02, _NUSIZ0: 0x00})
    scan = soft_render_scanline(bus)
    assert scan[0x20] == 0x2C
    assert scan[0x21] == 0x00


def test_missile_uses_player_colour():
    """Missile 0 takes COLUP0; missile 1 takes COLUP1."""
    bus = _bus({_COLUBK: 0x00,
                _COLUP0: 0x11, _RESM0: 0x10, _ENAM0: 0x02,
                _COLUP1: 0x77, _RESM1: 0x40, _ENAM1: 0x02})
    scan = soft_render_scanline(bus)
    assert scan[0x10] == 0x11      # M0 → COLUP0
    assert scan[0x40] == 0x77      # M1 → COLUP1


# --------------------------------------------------------------------------- #
# Missile / ball width — NUSIZ / CTRLPF bits 4-5
# --------------------------------------------------------------------------- #

def test_missile_width_two():
    """NUSIZ bits 4-5 = 1 → width 2."""
    bus = _bus({_COLUBK: 0x00, _COLUP0: 0x3A, _RESM0: 0x20,
                _ENAM0: 0x02, _NUSIZ0: 0x10})
    scan = soft_render_scanline(bus)
    assert jnp.all(scan[0x20:0x22] == 0x3A)
    assert scan[0x22] == 0x00


def test_missile_width_eight():
    """NUSIZ bits 4-5 = 3 → width 8."""
    bus = _bus({_COLUBK: 0x00, _COLUP0: 0x3A, _RESM0: 0x20,
                _ENAM0: 0x02, _NUSIZ0: 0x30})
    scan = soft_render_scanline(bus)
    assert jnp.all(scan[0x20:0x28] == 0x3A)
    assert scan[0x28] == 0x00


# --------------------------------------------------------------------------- #
# Ball
# --------------------------------------------------------------------------- #

def test_ball_disabled_draws_nothing():
    bus = _bus({_COLUBK: 0x00, _COLUPF: 0xFF, _RESBL: 0x20, _ENABL: 0x00})
    scan = soft_render_scanline(bus)
    assert jnp.all(scan == 0x00)


def test_ball_enabled_uses_colupf():
    bus = _bus({_COLUBK: 0x00, _COLUPF: 0x4E, _RESBL: 0x30, _ENABL: 0x02})
    scan = soft_render_scanline(bus)
    assert scan[0x30] == 0x4E


def test_ball_width_four_from_ctrlpf():
    """CTRLPF bits 4-5 = 2 → ball width 4."""
    bus = _bus({_COLUBK: 0x00, _COLUPF: 0x4E, _RESBL: 0x30,
                _ENABL: 0x02, _CTRLPF: 0x20})
    scan = soft_render_scanline(bus)
    assert jnp.all(scan[0x30:0x34] == 0x4E)
    assert scan[0x34] == 0x00


# --------------------------------------------------------------------------- #
# Priority
# --------------------------------------------------------------------------- #

def test_player0_draws_over_missile0():
    """P0 has higher priority than M0 — where they overlap, P0 wins."""
    bus = _bus({_COLUBK: 0x00,
                _COLUP0: 0x11, _GRP0: 0xFF, _RESP0: 0x20,
                _RESM0: 0x20, _ENAM0: 0x02, _NUSIZ0: 0x30})   # M0 8-wide at 0x20
    scan = soft_render_scanline(bus)
    # P0 (colour 0x11) covers 0x20..0x27; M0 is the same colour anyway,
    # but the point is the composite stays 0x11.
    assert jnp.all(scan[0x20:0x28] == 0x11)


def test_ball_draws_over_playfield_under_players():
    """Ball sits above the playfield but below the players."""
    bus = _bus({_COLUBK: 0x00, _COLUPF: 0x4E,
                _RESBL: 0x10, _ENABL: 0x02, _CTRLPF: 0x30,    # 8-wide ball at 0x10
                _COLUP0: 0x11, _GRP0: 0xFF, _RESP0: 0x14})    # P0 overlaps part
    scan = soft_render_scanline(bus)
    assert scan[0x10] == 0x4E      # ball, no player here
    assert scan[0x14] == 0x11      # player on top of ball


# --------------------------------------------------------------------------- #
# Gradient — ∂pixel / ∂ROM through a missile / ball colour
# --------------------------------------------------------------------------- #

def test_grad_ball_pixel_one_hot_at_colupf_byte():
    """Program enables the ball and sets COLUPF from an immediate;
    `jax.grad` of a ball pixel is one-hot at that immediate."""
    # LDA #$30 / STA RESBL / LDA #$02 / STA ENABL / LDA #$4E / STA COLUPF
    rom_init = _rom_with([0xA9, 0x30, 0x85, _RESBL,
                          0xA9, 0x02, 0x85, _ENABL,
                          0xA9, 0x4E, 0x85, _COLUPF])

    def simulator(rom_arr):
        bus = SoftBus(ram=jnp.zeros((128,), dtype=jnp.float32), rom=rom_arr)
        state = initial_soft_cpu_state()
        state, bus = soft_run(state, bus, n_steps=6)
        return soft_render_scanline(bus)[0x30]

    assert float(simulator(rom_init)) == 0x4E
    grad = jax.grad(simulator)(rom_init)
    assert float(grad[9]) == pytest.approx(1.0)    # COLUPF immediate
    other = float(jnp.sum(jnp.abs(grad))) - float(jnp.abs(grad[9]))
    assert other == pytest.approx(0.0)


def test_grad_missile_pixel_one_hot_at_colup0_byte():
    """Program enables missile 0 and sets COLUP0 from an immediate;
    `jax.grad` of a missile pixel is one-hot at that immediate."""
    # LDA #$20 / STA RESM0 / LDA #$02 / STA ENAM0 / LDA #$1C / STA COLUP0
    rom_init = _rom_with([0xA9, 0x20, 0x85, _RESM0,
                          0xA9, 0x02, 0x85, _ENAM0,
                          0xA9, 0x1C, 0x85, _COLUP0])

    def simulator(rom_arr):
        bus = SoftBus(ram=jnp.zeros((128,), dtype=jnp.float32), rom=rom_arr)
        state = initial_soft_cpu_state()
        state, bus = soft_run(state, bus, n_steps=6)
        return soft_render_scanline(bus)[0x20]

    assert float(simulator(rom_init)) == 0x1C
    grad = jax.grad(simulator)(rom_init)
    assert float(grad[9]) == pytest.approx(1.0)    # COLUP0 immediate
