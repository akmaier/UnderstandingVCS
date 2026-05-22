"""SOFT-mode TIA rendering — P7f-a (differentiable background + playfield).

A differentiable parallel to the HARD TIA's playfield render
(`jaxtari.tia.system.render_playfield_scanline`). This is the piece
that finally lets `jax.grad` flow from a framebuffer **pixel** back to
a ROM byte — the headline `∂pixel / ∂ROM` the whole project is built
around.

**Why no SoftBus change.** The SOFT bus collapses the TIA register
file into the low cells of `bus.ram`: a `STA $09` resolves through the
`addr & 0x7F` decode to `ram[9]`, and $09 *is* COLUBK. So the renderer
simply reads its registers out of `bus.ram[0x00:0x40]` — no new field,
no decode change, no churn to the P7b/P7c test suite.

**What flows, what doesn't.** A scanline pixel is

    pixel = pf_mask * COLUPF + (1 - pf_mask) * COLUBK

`COLUBK` / `COLUPF` are read with `soft_ram_peek` — fully
differentiable, so a ROM byte `STA`'d into a colour register reaches
every pixel it paints. The playfield *pattern* (`pf_mask`, from the
PF0/PF1/PF2 bits) is integer bit-extraction — its gradient breaks at
the int cast. That is the right split for XAI: the pattern is
structural, the colours are the differentiable payload.

**Scope.** P7f-a is background + playfield only. Player sprites
(P7f-b), missiles + ball (P7f-c), and collisions + proper TIA/RIOT
register dispatch (P7f-d) follow. VBLANK output-blanking is not
modelled here — the render is the active-display path.
"""

from __future__ import annotations

import jax.numpy as jnp

from jaxtari.diff.soft_step import soft_ram_peek
from jaxtari.tia.system import (
    W_COLUBK,
    W_COLUPF,
    W_CTRLPF,
    W_PF0,
    W_PF1,
    W_PF2,
)

SCREEN_WIDTH = 160
VISIBLE_SCANLINES = 192


def _playfield_mask(pf0, pf1, pf2, ctrlpf) -> jnp.ndarray:
    """The TIA's 20-bit playfield expanded to a `(160,)` float32 mask
    (1.0 = playfield pixel, 0.0 = background pixel).

    Bit order matches `jaxtari.tia.system._playfield_bits`:
      PF0 bits 4..7  → playfield pixels 0..3
      PF1 bits 7..0  → pixels 4..11
      PF2 bits 0..7  → pixels 12..19
    Each of the 20 bits covers 4 screen pixels. The right half repeats
    the pattern, or mirrors it when CTRLPF bit 0 (reflect) is set.
    """
    pf0i = pf0.astype(jnp.int32)
    pf1i = pf1.astype(jnp.int32)
    pf2i = pf2.astype(jnp.int32)

    bits = []
    for b in range(4):
        bits.append((pf0i >> (4 + b)) & 1)        # PF0 4..7
    for b in range(8):
        bits.append((pf1i >> (7 - b)) & 1)        # PF1 7..0
    for b in range(8):
        bits.append((pf2i >> b) & 1)              # PF2 0..7
    bits20 = jnp.stack(bits).astype(jnp.float32)  # (20,)

    left      = jnp.repeat(bits20, 4)             # (80,)
    reflected = (ctrlpf.astype(jnp.int32) & 0x01) != 0
    right     = jnp.where(
        reflected,
        jnp.repeat(jnp.flip(bits20), 4),
        jnp.repeat(bits20, 4),
    )
    return jnp.concatenate([left, right])         # (160,)


def soft_render_scanline(bus) -> jnp.ndarray:
    """Differentiable single-scanline render — background + playfield.

    Reads the TIA registers from `bus.ram[0x00:0x40]` and returns a
    `(160,)` float32 array of colour values. The gradient w.r.t. the
    COLUBK / COLUPF cells (hence back to any ROM byte that wrote them)
    is exact.
    """
    colubk = soft_ram_peek(bus.ram, W_COLUBK)
    colupf = soft_ram_peek(bus.ram, W_COLUPF)
    pf0    = soft_ram_peek(bus.ram, W_PF0)
    pf1    = soft_ram_peek(bus.ram, W_PF1)
    pf2    = soft_ram_peek(bus.ram, W_PF2)
    ctrlpf = soft_ram_peek(bus.ram, W_CTRLPF)

    mask = _playfield_mask(pf0, pf1, pf2, ctrlpf)
    return mask * colupf + (1.0 - mask) * colubk


def soft_render_frame(bus, height: int = VISIBLE_SCANLINES) -> jnp.ndarray:
    """Differentiable full-frame render — `(height, 160)` float32.

    P7f-a's playfield is static for the whole frame (no mid-frame
    register changes are modelled), so every scanline is identical;
    the frame is the scanline broadcast over `height` rows. Beam-
    accurate per-scanline rendering is a later P7f increment.
    """
    scanline = soft_render_scanline(bus)
    return jnp.broadcast_to(scanline, (height, scanline.shape[0]))
