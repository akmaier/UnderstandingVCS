"""P8-c — Attribution on a Pong-like SOFT-mode state.

The original P8 acceptance phrase from PORTING_PLAN.md is *"Integrated
Gradients on ROM bytes recovers a known sprite-defining region in
Pong."* — executing the real `pong.bin` end-to-end in SOFT mode and
attributing a rendered pixel back to its driving ROM bytes.

This file delivers what's reachable today + records the architectural
prerequisite for the literal claim:

  1. **`soft_run_scan` is fast enough for real-ROM traces** — JIT-
     compiled lax.scan body that scales to thousands of instructions
     where the unrolled `soft_run` does not. Smoke test runs the
     actual `pong.bin` for 5000 SOFT instructions and verifies the PC
     advances (so the SOFT step really is executing real Pong code).

  2. **Synthetic Pong-state attribution** — a hand-set `SoftBus`
     representing a Pong-like rendered frame (background, playfield
     walls, two paddles, a ball, all of which a Pong kernel would have
     written by mid-frame). `jax.grad` of a paddle pixel localises to
     the COLUP register; of a background pixel, to COLUBK; of a wall
     pixel, to COLUPF. Different pixels attribute to different bytes —
     the project's headline XAI claim, applied to a realistic frame
     state.

  3. **Real-Pong-execution attribution** — pinned as `xfail`. Pong's
     startup polls the RIOT INTIM timer in a busy loop. The SOFT bus
     currently does not carry RIOT timer state (`_bus_read` returns
     the RAM cell where INTIM would be, which is 0 forever), so Pong
     never escapes init. Closing this needs a small SOFT-mode RIOT
     timer (decrement per soft_step, reset on TIM*T writes, read on
     $0284) — a P7f-e / P8-cx item the harness now makes concrete.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxtari.diff import (
    SoftBus,
    initial_soft_cpu_state,
    soft_render_scanline,
    soft_run_scan,
)
from jaxtari.xai import integrated_gradients

# TIA register offsets (= SOFT-bus RAM cells under the bus-collapse model;
# see P7f-a/b/c).
_NUSIZ0 = 0x04
_COLUP0 = 0x06
_COLUP1 = 0x07
_COLUPF = 0x08
_COLUBK = 0x09
_CTRLPF = 0x0A
_PF0    = 0x0D
_PF1    = 0x0E
_PF2    = 0x0F
_RESP0  = 0x10
_RESP1  = 0x11
_RESBL  = 0x14
_GRP0   = 0x1B
_GRP1   = 0x1C
_ENABL  = 0x1F

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PONG_ROM  = _REPO_ROOT / "xitari" / "roms" / "pong.bin"


# --------------------------------------------------------------------------- #
# 1. soft_run_scan smoke — runs the real pong.bin for thousands of steps
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not _PONG_ROM.exists(),
                    reason="xitari pong.bin not present in this checkout")
def test_soft_run_scan_executes_real_pong_for_5000_steps():
    """The SOFT-mode `soft_run_scan` (jax.lax.scan-backed) runs the
    real `pong.bin` for thousands of CPU instructions without raising.
    Doesn't assert pong reaches the rendering kernel — that needs the
    P8-cx RIOT-timer extension — only that the SOFT execution path
    successfully ticks through real-ROM opcodes."""
    rom_bytes = np.fromfile(_PONG_ROM, dtype=np.uint8)
    rom = jnp.asarray(rom_bytes, dtype=jnp.float32)
    bus = SoftBus(ram=jnp.zeros((128,), dtype=jnp.float32), rom=rom)
    state = initial_soft_cpu_state()
    pc0 = float(state.PC)
    state, bus = soft_run_scan(state, bus, 5000)
    # PC must have advanced from the reset vector — i.e. the cart's code
    # was actually decoded and dispatched.
    assert float(state.PC) != pc0


# --------------------------------------------------------------------------- #
# 2. Synthetic Pong-frame attribution — IG localises per pixel
# --------------------------------------------------------------------------- #

def _pong_like_bus() -> SoftBus:
    """Hand-set TIA registers to look like a Pong frame mid-render:
    background colour, white side walls (via PF0/PF1/PF2 + CTRLPF
    reflect), player 0 paddle (solid 8-px sprite, COLUP0 colour, X near
    the left), player 1 paddle (right), and a ball."""
    ram = jnp.zeros((128,), dtype=jnp.float32)
    # A Pong-like frame: thin top/bottom wall on the far left + (via
    # CTRLPF reflect) far right, two paddles, the ball, and a wide
    # background area in between for the "court".
    settings = {
        _COLUBK: 0x88,    # background (greyish)
        _COLUPF: 0x0F,    # white walls
        _COLUP0: 0x42,    # player 0 paddle colour
        _COLUP1: 0xC6,    # player 1 paddle colour
        _CTRLPF: 0x01,    # playfield reflected (Pong's symmetric wall)
        _PF0:    0x10,    # only PF0 bit 4 lit → screen px 0..3 (and reflected: 156..159)
        _PF1:    0x00,
        _PF2:    0x00,
        _GRP0:   0xFF,    # solid 8-pixel paddle
        _RESP0:  0x18,    # paddle 0 at X = 24
        _GRP1:   0xFF,
        _RESP1:  0x88,    # paddle 1 at X = 136
        _ENABL:  0x02,    # ball enabled
        _RESBL:  0x4E,    # ball at X = 78 (1-px ball — only that one column)
    }
    for off, val in settings.items():
        ram = ram.at[off].set(jnp.float32(val))
    rom = jnp.zeros((4096,), dtype=jnp.float32)
    return SoftBus(ram=ram, rom=rom)


def _attribute_pixel(pixel_x: int):
    """Return ∂(scanline[pixel_x]) / ∂(ram) at the Pong-like state."""
    bus0 = _pong_like_bus()

    def render(ram):
        return soft_render_scanline(SoftBus(ram=ram, rom=bus0.rom))[pixel_x]

    return jax.grad(render)(bus0.ram)


def test_synthetic_pong_paddle_pixel_attributes_to_colup0():
    """A pixel inside the left paddle (X=24..31) attributes only to the
    COLUP0 cell — the colour register the renderer reads to paint it."""
    grad = _attribute_pixel(pixel_x=0x1A)              # inside paddle 0
    assert float(grad[_COLUP0]) == pytest.approx(1.0)
    other = float(jnp.sum(jnp.abs(grad))) - abs(float(grad[_COLUP0]))
    assert other == pytest.approx(0.0, abs=1e-5)


def test_synthetic_pong_other_paddle_pixel_attributes_to_colup1():
    """A pixel inside the right paddle (X=136..143) attributes to COLUP1."""
    grad = _attribute_pixel(pixel_x=0x8A)              # inside paddle 1
    assert float(grad[_COLUP1]) == pytest.approx(1.0)


def test_synthetic_pong_wall_pixel_attributes_to_colupf():
    """A wall pixel (playfield) attributes to COLUPF — at X = 0x00
    (the leftmost playfield pixel from PF0 bit 4)."""
    grad = _attribute_pixel(pixel_x=0x00)
    assert float(grad[_COLUPF]) == pytest.approx(1.0)


def test_synthetic_pong_background_pixel_attributes_to_colubk():
    """A clear-area pixel (no playfield, no sprite) attributes to COLUBK.
    With the Pong-like state, pixel X=0x40 sits between PF and the
    paddles — pure background."""
    grad = _attribute_pixel(pixel_x=0x40)
    assert float(grad[_COLUBK]) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 3. IG completeness on a Pong-like paddle pixel
# --------------------------------------------------------------------------- #

def test_synthetic_pong_paddle_pixel_ig_recovers_pixel_value():
    """IG with a smart baseline (ram with COLUP0 zeroed) on a paddle
    pixel returns the paddle colour exactly — completeness on real
    rendering registers."""
    bus0 = _pong_like_bus()
    target_ram = bus0.ram
    baseline_ram = target_ram.at[_COLUP0].set(0.0)

    def render(ram):
        return soft_render_scanline(SoftBus(ram=ram, rom=bus0.rom))[0x1A]

    ig = integrated_gradients(render, target_ram, baseline=baseline_ram,
                              steps=16)
    # All attribution lands on COLUP0; sum equals the pixel value.
    assert float(ig[_COLUP0]) == pytest.approx(float(target_ram[_COLUP0]),
                                                abs=1e-3)
    other = float(jnp.sum(jnp.abs(ig))) - abs(float(ig[_COLUP0]))
    assert other == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# 4. Real-Pong-execution attribution — xfail-pinned architectural prereq
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not _PONG_ROM.exists(),
                    reason="xitari pong.bin not present in this checkout")
@pytest.mark.xfail(
    strict=True,
    reason=(
        "P8-cx — Pong's startup polls the RIOT INTIM timer in a busy "
        "loop. The SOFT bus does not yet carry RIOT timer state, so "
        "`_bus_read` returns 0 for $0284 forever and Pong never reaches "
        "the rendering kernel — COLUP0 stays 0 even after thousands of "
        "soft_step calls. Closing this needs a small SOFT-mode RIOT "
        "timer (decrement per soft_step, reset on TIM*T writes, read "
        "on $0284). Remove this xfail marker the day soft_run_scan(pong.bin) "
        "actually paints a paddle."
    ),
)
def test_real_pong_execution_attribution_xfails_on_riot_stall():
    rom_bytes = np.fromfile(_PONG_ROM, dtype=np.uint8)
    rom = jnp.asarray(rom_bytes, dtype=jnp.float32)
    bus = SoftBus(ram=jnp.zeros((128,), dtype=jnp.float32), rom=rom)
    state = initial_soft_cpu_state()
    # Run "enough" SOFT steps that Pong should be rendering by now.
    state, bus = soft_run_scan(state, bus, 20_000)
    # After 20K SOFT steps Pong WOULD have set COLUP0 to its paddle
    # colour ($42 or similar non-zero) if execution had progressed.
    # Asserting that pins the architectural gap.
    assert float(bus.ram[_COLUP0]) != 0.0, (
        "Pong stalled in RIOT-INTIM polling loop — needs SOFT-mode "
        "RIOT timer (P8-cx). See test docstring."
    )
