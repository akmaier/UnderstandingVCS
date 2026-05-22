"""P7f-a tests — differentiable TIA playfield render.

This is the phase that closes the loop the whole project is about:
`jax.grad` of a framebuffer **pixel** w.r.t. the ROM bytes. A SOFT
program writes a colour into a TIA register, `soft_render_scanline`
turns the register file into pixels, and the gradient of a pixel is
one-hot at the ROM byte that supplied the colour.

The renderer reads the TIA register file straight out of
`bus.ram[0x00:0x40]` — the SOFT bus collapses TIA addresses into the
low RAM cells, so `STA $09` (COLUBK) lands in `ram[9]`.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxtari.diff import (
    SoftBus,
    initial_soft_bus,
    initial_soft_cpu_state,
    soft_render_frame,
    soft_render_scanline,
    soft_run,
)

# TIA register offsets (also the SOFT-bus RAM cells they collapse into).
_COLUBK = 0x09
_COLUPF = 0x08
_CTRLPF = 0x0A
_PF0    = 0x0D
_PF1    = 0x0E
_PF2    = 0x0F


def _rom_with(bytes_: list[int], size: int = 256) -> jnp.ndarray:
    rom = jnp.zeros((size,), dtype=jnp.float32)
    for i, b in enumerate(bytes_):
        rom = rom.at[i].set(jnp.float32(b))
    return rom


def _bus_with_regs(**regs) -> SoftBus:
    """A SoftBus whose RAM (= TIA register file) has the given registers
    set. Keyword args are register names: colubk, colupf, ctrlpf, pf0..2."""
    offsets = {"colubk": _COLUBK, "colupf": _COLUPF, "ctrlpf": _CTRLPF,
               "pf0": _PF0, "pf1": _PF1, "pf2": _PF2}
    ram = jnp.zeros((128,), dtype=jnp.float32)
    for name, value in regs.items():
        ram = ram.at[offsets[name]].set(jnp.float32(value))
    return SoftBus(ram=ram, rom=jnp.zeros((256,), dtype=jnp.float32))


# --------------------------------------------------------------------------- #
# Forward render — background
# --------------------------------------------------------------------------- #

def test_empty_playfield_renders_all_background():
    bus = _bus_with_regs(colubk=0x1E)
    scan = soft_render_scanline(bus)
    assert scan.shape == (160,)
    assert jnp.all(scan == 0x1E)


def test_scanline_width_is_160():
    bus = initial_soft_bus(jnp.zeros((256,), dtype=jnp.float32))
    assert soft_render_scanline(bus).shape == (160,)


def test_frame_shape_is_192_by_160():
    bus = initial_soft_bus(jnp.zeros((256,), dtype=jnp.float32))
    frame = soft_render_frame(bus)
    assert frame.shape == (192, 160)


def test_frame_rows_all_equal_the_scanline():
    ram = jnp.zeros((128,), dtype=jnp.float32).at[_COLUBK].set(jnp.float32(0x42))
    bus = SoftBus(ram=ram, rom=jnp.zeros((256,), dtype=jnp.float32))
    frame = soft_render_frame(bus)
    scan = soft_render_scanline(bus)
    assert jnp.all(frame == scan[None, :])


# --------------------------------------------------------------------------- #
# Forward render — playfield pattern
# --------------------------------------------------------------------------- #

def test_pf0_high_nibble_paints_leftmost_pixels():
    """PF0 bit 4 is playfield pixel 0 → screen pixels 0..3. Set PF0 = 0x10
    (only bit 4) with distinct PF/BK colours; pixels 0..3 are playfield."""
    ram = (jnp.zeros((128,), dtype=jnp.float32)
           .at[_COLUBK].set(jnp.float32(0x00))
           .at[_COLUPF].set(jnp.float32(0xFF))
           .at[_PF0].set(jnp.float32(0x10)))      # bit 4 set
    bus = SoftBus(ram=ram, rom=jnp.zeros((256,), dtype=jnp.float32))
    scan = soft_render_scanline(bus)
    assert jnp.all(scan[0:4] == 0xFF)             # playfield pixel 0
    assert jnp.all(scan[4:8] == 0x00)             # background


def test_full_playfield_paints_whole_scanline():
    ram = (jnp.zeros((128,), dtype=jnp.float32)
           .at[_COLUBK].set(jnp.float32(0x00))
           .at[_COLUPF].set(jnp.float32(0x0E))
           .at[_PF0].set(jnp.float32(0xF0))       # all 4 used bits
           .at[_PF1].set(jnp.float32(0xFF))
           .at[_PF2].set(jnp.float32(0xFF)))
    bus = SoftBus(ram=ram, rom=jnp.zeros((256,), dtype=jnp.float32))
    scan = soft_render_scanline(bus)
    assert jnp.all(scan == 0x0E)


def test_reflected_playfield_mirrors_right_half():
    """With CTRLPF reflect set, the right half is the mirror of the left.
    PF0=0x10 lights playfield pixel 0 (screen 0..3). Reflected, that bit
    is the *last* of the 20 → screen pixels 156..159 on the right half."""
    ram = (jnp.zeros((128,), dtype=jnp.float32)
           .at[_COLUBK].set(jnp.float32(0x00))
           .at[_COLUPF].set(jnp.float32(0xFF))
           .at[_PF0].set(jnp.float32(0x10))
           .at[_CTRLPF].set(jnp.float32(0x01)))   # reflect
    bus = SoftBus(ram=ram, rom=jnp.zeros((256,), dtype=jnp.float32))
    scan = soft_render_scanline(bus)
    assert jnp.all(scan[0:4] == 0xFF)             # left: pixel 0
    assert jnp.all(scan[156:160] == 0xFF)         # right: mirrored to the end
    assert jnp.all(scan[80:84] == 0x00)           # right half starts background


# --------------------------------------------------------------------------- #
# The headline — ∂pixel / ∂ROM
# --------------------------------------------------------------------------- #

def test_grad_background_pixel_one_hot_at_colour_rom_byte():
    """**The P7f headline.** Program: LDA #$1E / STA $09 (COLUBK). Render.
    `jax.grad` of any background pixel w.r.t. the ROM is one-hot at the
    immediate-operand byte — "this ROM byte explains this pixel"."""
    rom_init = _rom_with([0xA9, 0x1E, 0x85, _COLUBK])

    def simulator(rom_arr):
        bus = SoftBus(ram=jnp.zeros((128,), dtype=jnp.float32), rom=rom_arr)
        state = initial_soft_cpu_state()
        state, bus = soft_run(state, bus, n_steps=2)     # LDA / STA
        scan = soft_render_scanline(bus)
        return scan[0]                                   # one pixel

    forward = simulator(rom_init)
    assert float(forward) == 0x1E

    grad = jax.grad(simulator)(rom_init)
    assert float(grad[1]) == pytest.approx(1.0)          # the colour immediate
    other = float(jnp.sum(jnp.abs(grad))) - float(jnp.abs(grad[1]))
    assert other == pytest.approx(0.0)


def test_grad_full_frame_sum_back_to_colour_byte():
    """Sum the whole 192x160 frame; with an empty playfield every pixel is
    COLUBK, so ∂(frame sum)/∂rom is 192*160 at the colour byte."""
    rom_init = _rom_with([0xA9, 0x20, 0x85, _COLUBK])

    def simulator(rom_arr):
        bus = SoftBus(ram=jnp.zeros((128,), dtype=jnp.float32), rom=rom_arr)
        state = initial_soft_cpu_state()
        state, bus = soft_run(state, bus, n_steps=2)
        return jnp.sum(soft_render_frame(bus))

    grad = jax.grad(simulator)(rom_init)
    assert float(grad[1]) == pytest.approx(192 * 160)
    other = float(jnp.sum(jnp.abs(grad))) - float(jnp.abs(grad[1]))
    assert other == pytest.approx(0.0)


def test_grad_playfield_pixel_attributes_to_colupf_byte():
    """A playfield pixel depends on COLUPF, not COLUBK. Program sets PF0
    so pixel 0 is playfield, then loads COLUPF from an immediate; the
    gradient of pixel 0 localises to that immediate."""
    # LDA #$FF / STA $0D (PF0)   — light playfield pixel 0
    # LDA #$0C / STA $08 (COLUPF)
    rom_init = _rom_with([0xA9, 0xFF, 0x85, _PF0,
                          0xA9, 0x0C, 0x85, _COLUPF])

    def simulator(rom_arr):
        bus = SoftBus(ram=jnp.zeros((128,), dtype=jnp.float32), rom=rom_arr)
        state = initial_soft_cpu_state()
        state, bus = soft_run(state, bus, n_steps=4)
        return soft_render_scanline(bus)[0]              # a playfield pixel

    assert float(simulator(rom_init)) == 0x0C
    grad = jax.grad(simulator)(rom_init)
    # rom[5] is the COLUPF immediate.
    assert float(grad[5]) == pytest.approx(1.0)
    # rom[1] (the PF0 immediate) feeds the *pattern* — integer-extracted,
    # so its gradient is zero.
    assert float(grad[1]) == pytest.approx(0.0)
