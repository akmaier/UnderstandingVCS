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

**Player sprites (P7f-b).** P0 / P1 composite over the playfield with
standard TIA priority (P0 on top of P1 on top of playfield/background).
Each player is a single 1×-wide 8-pixel sprite — NUSIZ multi-copy /
2×4× scaling is deferred exactly as it is in the HARD port (see P3c).

**SOFT-mode position convention.** On real hardware a sprite's X is
set by *strobing* `RESP0` / `RESP1` at a chosen beam cycle — position
is implicit in timing, which the static-register-file SOFT model does
not track. So SOFT mode adopts a deliberate convention: **the
`RESP0` / `RESP1` cells hold the player X position directly.** A SOFT
program positions a sprite with `LDA #xpos / STA RESP0`, and the X
byte then carries a gradient to every pixel the sprite would move
over. Faithful strobe-timing positioning waits for P7f-d (real TIA
timing state).

**Missiles + ball (P7f-c).** M0 / M1 and the ball BL are solid blocks
of 1/2/4/8 pixels (`_block_mask`) — missile size from NUSIZ bits 4-5,
ball size from CTRLPF bits 4-5, enabled by the ENAM* / ENABL bit 1.
Missiles take their player's colour; the ball takes COLUPF. They use
the same SOFT-mode position convention as players (the RES* cell
holds the X position).

**Collisions (P7f-d).** `soft_collision_registers` detects the 15
pairwise object overlaps and packs them into the 8 CX latch registers,
matching the HARD TIA's P3e layout. Collisions are a structural
(boolean) property — forward feature parity, not a gradient path.

**Scope.** P7f-a…d cover the full visible object set + collisions.
Still deferred: VBLANK output-blanking (the render is the active-
display path), wiring collisions into bus *reads* so a program can
read $00-$07 as collision data ("proper TIA read-register dispatch"),
and cart-hotspot bank-switching from SOFT mode — see STATUS.md.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxtari.diff.soft_step import soft_ram_peek
from jaxtari.tia.system import (
    W_COLUBK,
    W_COLUP0,
    W_COLUP1,
    W_COLUPF,
    W_CTRLPF,
    W_ENABL,
    W_ENAM0,
    W_ENAM1,
    W_GRP0,
    W_GRP1,
    W_NUSIZ0,
    W_NUSIZ1,
    W_PF0,
    W_PF1,
    W_PF2,
    W_REFP0,
    W_REFP1,
    W_RESBL,
    W_RESM0,
    W_RESM1,
    W_RESP0,
    W_RESP1,
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


# P3g — NUSIZ multi-copy + 2×/4×-wide player scaling.
# Mirror of jaxtari.tia.system._NUSIZ_PLAYER_LAYOUT; kept inline here to
# avoid a SOFT-side dependency on HARD-mode internals.
#
#   000  1 copy, 1× wide
#   001  2 copies, 16-pixel spacing, 1× wide
#   010  2 copies, 32-pixel spacing, 1× wide
#   011  3 copies, 16-pixel spacing, 1× wide
#   100  2 copies, 64-pixel spacing, 1× wide
#   101  1 copy, 2× wide
#   110  3 copies, 32-pixel spacing, 1× wide
#   111  1 copy, 4× wide
_NUSIZ_PLAYER_LAYOUT = [
    ((0,),         1),
    ((0, 16),      1),
    ((0, 32),      1),
    ((0, 16, 32),  1),
    ((0, 64),      1),
    ((0,),         2),
    ((0, 32, 64),  1),
    ((0,),         4),
]


def _player_mask_single(grp, refp, xpos, scale) -> jnp.ndarray:
    """One copy of a player sprite at `xpos` with each bit replicated
    `scale` times (1/2/4 — drives the NUSIZ double / quad modes).
    Returns a `(160,)` float32 mask."""
    grp_i     = grp.astype(jnp.int32)
    reflected = (refp.astype(jnp.int32) & 0x08) != 0

    bits = []
    for i in range(8):
        normal = (grp_i >> (7 - i)) & 1       # bit 7 → leftmost
        refl   = (grp_i >> i) & 1
        bits.append(jnp.where(reflected, refl, normal))
    bits8 = jnp.stack(bits).astype(jnp.float32)            # (8,)

    x        = xpos.astype(jnp.int32)
    # Per-pixel weights: bit i contributes `scale` adjacent pixels.
    pix      = jnp.arange(8 * scale)
    bit_idx  = pix // scale                                 # (8*scale,)
    weights  = bits8[bit_idx]                               # (8*scale,)
    cols     = (x + pix) % SCREEN_WIDTH                     # (8*scale,)
    onehots  = jax.nn.one_hot(cols, SCREEN_WIDTH, dtype=jnp.float32)
    return weights @ onehots                                # (160,)


def _player_mask(grp, refp, xpos, nusiz) -> jnp.ndarray:
    """A player sprite expanded to a `(160,)` float32 mask, with
    NUSIZ-driven multi-copy + 2×/4× scaling (P3g).

    All 8 NUSIZ modes are evaluated unconditionally and blended by the
    integer-extracted `nusiz & 7` — same pattern as the CTRLPF.D2
    priority swap. Per-mode masks are summed across copies and clipped
    to [0, 1] so overlapping pixels stay binary.
    """
    nusiz_low = nusiz.astype(jnp.int32) & 0x07
    per_mode = []
    for offsets, scale in _NUSIZ_PLAYER_LAYOUT:
        m = jnp.zeros(SCREEN_WIDTH, dtype=jnp.float32)
        for off in offsets:
            m = m + _player_mask_single(grp, refp, xpos + off, scale)
        per_mode.append(jnp.clip(m, 0.0, 1.0))
    stack = jnp.stack(per_mode)                                  # (8, 160)
    sel   = jax.nn.one_hot(nusiz_low, 8, dtype=jnp.float32)      # (8,)
    return sel @ stack                                           # (160,)


def _block_mask_single(xpos, size_reg, enable_reg) -> jnp.ndarray:
    """One copy of a missile / ball block — `1 << ((size_reg >> 4) & 3)`
    pixels wide at column `xpos`, enabled by `enable_reg` bit 1.
    `(160,)` float32 mask."""
    size_log2 = (size_reg.astype(jnp.int32) >> 4) & 0x03
    width     = 1 << size_log2                                  # 1/2/4/8
    enabled   = ((enable_reg.astype(jnp.int32) >> 1) & 1).astype(jnp.float32)

    x        = xpos.astype(jnp.int32)
    slots    = jnp.arange(8)
    in_block = (slots < width).astype(jnp.float32)              # (8,)
    cols     = (x + slots) % SCREEN_WIDTH
    onehots  = jax.nn.one_hot(cols, SCREEN_WIDTH, dtype=jnp.float32)
    return enabled * (in_block @ onehots)


def _block_mask(xpos, size_reg, enable_reg, nusiz=None) -> jnp.ndarray:
    """A missile / ball block with optional NUSIZ-driven multi-copy.

    The ball ignores `nusiz` (it doesn't share NUSIZ; its sizing comes
    from CTRLPF passed in via `size_reg`). Missiles pass NUSIZ so the
    same multi-copy / spacing the player uses applies. P3g.
    """
    if nusiz is None:
        return _block_mask_single(xpos, size_reg, enable_reg)
    nusiz_low = nusiz.astype(jnp.int32) & 0x07
    per_mode = []
    for offsets, _ in _NUSIZ_PLAYER_LAYOUT:
        m = jnp.zeros(SCREEN_WIDTH, dtype=jnp.float32)
        for off in offsets:
            m = m + _block_mask_single(xpos + off, size_reg, enable_reg)
        per_mode.append(jnp.clip(m, 0.0, 1.0))
    stack = jnp.stack(per_mode)
    sel   = jax.nn.one_hot(nusiz_low, 8, dtype=jnp.float32)
    return sel @ stack


def _object_masks(bus):
    """The six TIA object masks for the current scanline, each a `(160,)`
    float32 array (1.0 where the object is present): playfield, P0, P1,
    M0, M1, ball. Shared by `soft_render_scanline` and
    `soft_collision_registers`.
    """
    ctrlpf = soft_ram_peek(bus.ram, W_CTRLPF)
    pf = _playfield_mask(soft_ram_peek(bus.ram, W_PF0),
                         soft_ram_peek(bus.ram, W_PF1),
                         soft_ram_peek(bus.ram, W_PF2),
                         ctrlpf)
    nusiz0 = soft_ram_peek(bus.ram, W_NUSIZ0)
    nusiz1 = soft_ram_peek(bus.ram, W_NUSIZ1)
    p0 = _player_mask(soft_ram_peek(bus.ram, W_GRP0),
                      soft_ram_peek(bus.ram, W_REFP0),
                      soft_ram_peek(bus.ram, W_RESP0),
                      nusiz0)
    p1 = _player_mask(soft_ram_peek(bus.ram, W_GRP1),
                      soft_ram_peek(bus.ram, W_REFP1),
                      soft_ram_peek(bus.ram, W_RESP1),
                      nusiz1)
    # P3g: missiles inherit NUSIZ low 3 bits for multi-copy; the size
    # bits (4-5) come from the same NUSIZ register passed as size_reg.
    m0 = _block_mask(soft_ram_peek(bus.ram, W_RESM0),
                     nusiz0,
                     soft_ram_peek(bus.ram, W_ENAM0),
                     nusiz0)
    m1 = _block_mask(soft_ram_peek(bus.ram, W_RESM1),
                     nusiz1,
                     soft_ram_peek(bus.ram, W_ENAM1),
                     nusiz1)
    # Ball uses CTRLPF for sizing and does NOT participate in NUSIZ
    # multi-copy (its layout is one solid block).
    bl = _block_mask(soft_ram_peek(bus.ram, W_RESBL),
                     ctrlpf,
                     soft_ram_peek(bus.ram, W_ENABL))
    return pf, p0, p1, m0, m1, bl


def soft_render_scanline(bus) -> jnp.ndarray:
    """Differentiable single-scanline render — background, playfield,
    the two player sprites, the two missiles and the ball.

    Reads the TIA registers from `bus.ram[0x00:0x40]` and returns a
    `(160,)` float32 array of colour values. Compositing order matches
    the HARD TIA and has two modes, selected by CTRLPF bit 2 (PFP):

      PFP=0 (default):    bg ← pf ← bl ← M1 ← P1 ← M0 ← P0
      PFP=1 (priority):   bg ← M1 ← P1 ← M0 ← P0 ← pf ← bl

    Both composites are evaluated unconditionally and then blended by
    the integer-extracted PFP bit. The gradient w.r.t. every colour
    register — COLUBK / COLUPF / COLUP0 / COLUP1, hence back to any
    ROM byte that wrote them — is exact in both modes. The PFP bit
    itself is a structural switch (integer extraction breaks the
    gradient at the cast), matching the existing convention for
    enable / size bits (see `_block_mask`).
    """
    pf, p0, p1, m0, m1, bl = _object_masks(bus)
    colubk = soft_ram_peek(bus.ram, W_COLUBK)
    colupf = soft_ram_peek(bus.ram, W_COLUPF)
    colup0 = soft_ram_peek(bus.ram, W_COLUP0)
    colup1 = soft_ram_peek(bus.ram, W_COLUP1)
    ctrlpf = soft_ram_peek(bus.ram, W_CTRLPF)
    pfp_bit = ((ctrlpf.astype(jnp.int32) >> 2) & 1).astype(jnp.float32)  # 0.0 or 1.0

    # Default-priority composite: background ← playfield ← ball ←
    # missile 1 ← player 1 ← missile 0 ← player 0.
    s_def = pf * colupf + (1.0 - pf) * colubk
    s_def = (1.0 - bl) * s_def + bl * colupf
    s_def = (1.0 - m1) * s_def + m1 * colup1
    s_def = (1.0 - p1) * s_def + p1 * colup1
    s_def = (1.0 - m0) * s_def + m0 * colup0
    s_def = (1.0 - p0) * s_def + p0 * colup0

    # PFP composite: background ← missile 1 ← player 1 ← missile 0 ←
    # player 0 ← playfield ← ball.
    s_pfp = jnp.broadcast_to(colubk, (SCREEN_WIDTH,))
    s_pfp = (1.0 - m1) * s_pfp + m1 * colup1
    s_pfp = (1.0 - p1) * s_pfp + p1 * colup1
    s_pfp = (1.0 - m0) * s_pfp + m0 * colup0
    s_pfp = (1.0 - p0) * s_pfp + p0 * colup0
    s_pfp = (1.0 - pf) * s_pfp + pf * colupf
    s_pfp = (1.0 - bl) * s_pfp + bl * colupf

    return (1.0 - pfp_bit) * s_def + pfp_bit * s_pfp


def soft_render_frame(bus, height: int = VISIBLE_SCANLINES) -> jnp.ndarray:
    """Differentiable full-frame render — `(height, 160)` float32.

    P7f-a's playfield is static for the whole frame (no mid-frame
    register changes are modelled), so every scanline is identical;
    the frame is the scanline broadcast over `height` rows. Beam-
    accurate per-scanline rendering is a later P7f increment.
    """
    scanline = soft_render_scanline(bus)
    return jnp.broadcast_to(scanline, (height, scanline.shape[0]))


# --------------------------------------------------------------------------- #
# P7f-d — collision detection
# --------------------------------------------------------------------------- #

# CX register layout (matches jaxtari.tia.system R_CX*):
#   CXM0P  D7=M0-P1  D6=M0-P0
#   CXM1P  D7=M1-P0  D6=M1-P1
#   CXP0FB D7=P0-PF  D6=P0-BL
#   CXP1FB D7=P1-PF  D6=P1-BL
#   CXM0FB D7=M0-PF  D6=M0-BL
#   CXM1FB D7=M1-PF  D6=M1-BL
#   CXBLPF D7=BL-PF  D6=unused
#   CXPPMM D7=P0-P1  D6=M0-M1

def soft_collision_registers(bus) -> jnp.ndarray:
    """The 8 TIA collision latch registers for the current scanline —
    `(8,)` float32, indexed CXM0P, CXM1P, CXP0FB, CXP1FB, CXM0FB,
    CXM1FB, CXBLPF, CXPPMM.

    Two objects "collide" when their `(160,)` masks are both 1.0 at the
    same pixel; each register packs its hit flags into D7 / D6. The
    result is a structural (boolean) property of the scanline, so the
    gradient through it is ~0 — collisions are forward feature parity
    with the HARD TIA's P3e latches, not a gradient-flow path.

    This function is standalone (like `soft_render_scanline`); wiring it
    into the bus so a program can *read* $00-$07 as collision data is
    the deferred "proper TIA read-register dispatch" part of P7f-d.
    """
    pf, p0, p1, m0, m1, bl = _object_masks(bus)

    def hit(a, b) -> jnp.ndarray:
        return (jnp.sum(a * b) > 0).astype(jnp.float32)

    cxm0p  = hit(m0, p1) * 0x80 + hit(m0, p0) * 0x40
    cxm1p  = hit(m1, p0) * 0x80 + hit(m1, p1) * 0x40
    cxp0fb = hit(p0, pf) * 0x80 + hit(p0, bl) * 0x40
    cxp1fb = hit(p1, pf) * 0x80 + hit(p1, bl) * 0x40
    cxm0fb = hit(m0, pf) * 0x80 + hit(m0, bl) * 0x40
    cxm1fb = hit(m1, pf) * 0x80 + hit(m1, bl) * 0x40
    cxblpf = hit(bl, pf) * 0x80
    cxppmm = hit(p0, p1) * 0x80 + hit(m0, m1) * 0x40
    return jnp.stack([cxm0p, cxm1p, cxp0fb, cxp1fb,
                      cxm0fb, cxm1fb, cxblpf, cxppmm])
