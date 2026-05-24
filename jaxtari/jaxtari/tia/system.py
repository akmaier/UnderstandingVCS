"""TIA — Television Interface Adapter.

The TIA is the Atari 2600's video and audio chip. The CPU talks to it
through 64 memory-mapped registers in the \$0000–\$003F window (with
mirrors), some write-only, a subset readable at the same offsets.

Per-frame timing (NTSC):

    1 TIA color clock    = 1 horizontal pixel (160 visible per scanline)
    228 color clocks     = 1 scanline (40 hblank + 160 visible + 28 hsync)
    3 color clocks       = 1 CPU cycle  →  76 CPU cycles per scanline
    262 scanlines        = 1 frame      →  ~19,912 CPU cycles per frame

WSYNC (write \$02) stalls the CPU until the next horizontal sync, i.e.
the next scanline boundary.

Phase progress:
  - P3a: register file + scanline/frame timing + WSYNC.
  - P3b: playfield rendering (PF0/PF1/PF2 + CTRLPF mirror + COLU*).
  - P3c: player sprites P0/P1 (GRP, REFP, RESP, HMP, HMOVE, HMCLR).
  - P3d: missiles M0/M1 and ball BL (EN*, RES*, HM*, NUSIZ/CTRLPF sizes).
  - P3e: collision latches CXM0P..CXPPMM + CXCLR.
  - P3f (this sub-phase): software-driven frame ending via VSYNC and
    output blanking via VBLANK. A write that clears VSYNC.D1 (1→0
    edge) is treated as the frame boundary — `frame` increments,
    `scanline` resets to 0. When VBLANK.D1 is set, completed scanlines
    are NOT written to the framebuffer (the visible image is blanked).
    The 262-scanline wrap is kept as a safety overflow for ROMs that
    never trigger VSYNC.

P3 is now feature-complete for the documented register set, modulo:
  - NUSIZ multi-copy / 2×/4×-wide player scaling.
  - VDELP* / VDELBL vertical-delay updates.
  - Sub-pixel beam-accurate rendering (mid-scanline register changes).
  - Audio (TIASnd).
  - INPT* input reads.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

# --------------------------------------------------------------------------- #
# Register addresses
# --------------------------------------------------------------------------- #
# Write registers (CPU writes to \$00–\$2C trigger TIA state changes).

W_VSYNC  = 0x00  # vertical sync set/clear
W_VBLANK = 0x01  # vertical blank set/clear
W_WSYNC  = 0x02  # wait for leading edge of horizontal blank
W_RSYNC  = 0x03  # reset horizontal sync counter (debug; rarely used)
W_NUSIZ0 = 0x04  # number/size of player-missile 0
W_NUSIZ1 = 0x05
W_COLUP0 = 0x06  # color/luminance player 0
W_COLUP1 = 0x07
W_COLUPF = 0x08  # color/luminance playfield
W_COLUBK = 0x09  # color/luminance background
W_CTRLPF = 0x0A  # control playfield + ball size + collisions
W_REFP0  = 0x0B  # reflect player 0
W_REFP1  = 0x0C
W_PF0    = 0x0D  # playfield register byte 0 (4 bits in high nibble)
W_PF1    = 0x0E
W_PF2    = 0x0F
W_RESP0  = 0x10  # reset player 0 (latches position to current beam)
W_RESP1  = 0x11
W_RESM0  = 0x12
W_RESM1  = 0x13
W_RESBL  = 0x14
W_AUDC0  = 0x15  # audio control 0
W_AUDC1  = 0x16
W_AUDF0  = 0x17  # audio frequency 0
W_AUDF1  = 0x18
W_AUDV0  = 0x19  # audio volume 0
W_AUDV1  = 0x1A
W_GRP0   = 0x1B  # graphics player 0
W_GRP1   = 0x1C
W_ENAM0  = 0x1D  # enable missile 0
W_ENAM1  = 0x1E
W_ENABL  = 0x1F  # enable ball
W_HMP0   = 0x20  # horizontal motion player 0
W_HMP1   = 0x21
W_HMM0   = 0x22
W_HMM1   = 0x23
W_HMBL   = 0x24
W_VDELP0 = 0x25  # vertical delay player 0
W_VDELP1 = 0x26
W_VDELBL = 0x27
W_RESMP0 = 0x28  # reset missile 0 to player 0 position
W_RESMP1 = 0x29
W_HMOVE  = 0x2A  # apply horizontal motion to all sprites
W_HMCLR  = 0x2B  # clear all horizontal motion registers
W_CXCLR  = 0x2C  # clear all collision latches

# Read registers (decoded with A0-A3 only, so \$30–\$3F maps these; the same
# bits at \$00–\$0F also read them on real hardware).

R_CXM0P  = 0x00  # collision M0 vs P0,P1
R_CXM1P  = 0x01
R_CXP0FB = 0x02  # collision P0 vs PF, BL
R_CXP1FB = 0x03
R_CXM0FB = 0x04  # collision M0 vs PF, BL
R_CXM1FB = 0x05
R_CXBLPF = 0x06  # collision BL vs PF
R_CXPPMM = 0x07  # collision P0,P1 + M0,M1
R_INPT0  = 0x08  # pot port 0 (paddle)
R_INPT1  = 0x09
R_INPT2  = 0x0A
R_INPT3  = 0x0B
R_INPT4  = 0x0C  # trigger 0 (joystick fire)
R_INPT5  = 0x0D


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

NUM_REGISTERS = 64
NTSC_CPU_CYCLES_PER_SCANLINE = 76        # 228 color clocks / 3
NTSC_SCANLINES_PER_FRAME = 262
SCREEN_WIDTH = 160
SCREEN_HEIGHT = 192                       # NTSC visible region (vsync 3 + vblank 37 + visible 192 + overscan 30 = 262)


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

class TIAState(NamedTuple):
    """Snapshot of the TIA's observable state.

    `registers` is a 64-byte record of CPU writes. Later sub-phases
    interpret specific bytes (PF*, GRP*, COLU*, NUSIZ*, HM*, …) to drive
    rendering. Some derived state lives outside the register file
    because writes to RES* / HMOVE / HMCLR have side-effects on positions
    that the register file alone doesn't capture: `p0_x`, `p1_x` are
    those derived player horizontal positions in screen pixels [0..159].

    `framebuffer` is the rendered output, (SCREEN_HEIGHT × SCREEN_WIDTH)
    indexed colour bytes. `m0_x`, `m1_x`, `bl_x` are derived missile /
    ball positions, updated by RESM0 / RESM1 / RESBL / HMOVE writes.
    `collisions` is an 8-byte array holding the latched values of the
    read-only TIA collision registers \$30–\$37 (D6/D7 set on object
    overlap; cleared by CXCLR).

    `vsync_active` and `vblank_active` mirror the D1 bit of the
    respective registers; software toggles these to mark vertical
    retrace and the off-screen vblank/overscan periods.

    `inputs` carries the values games will read from \$30-\$3D:
    indices 0-7 are the collision latches (set by render); indices 8-13
    are INPT0-INPT5. P6 wires INPT4 (P0 trigger) and INPT5 (P1 trigger)
    from front-end joystick state; INPT0-INPT3 (paddle pots) default to
    \$80 ("centred") and proper dump-pot timing is a follow-up.
    """
    registers: jnp.ndarray
    scanline_cycle: int
    scanline: int
    frame: int
    wsync_pending: bool
    framebuffer: jnp.ndarray
    p0_x: int
    p1_x: int
    m0_x: int
    m1_x: int
    bl_x: int
    collisions: jnp.ndarray
    vsync_active: bool
    vblank_active: bool
    inpt: jnp.ndarray   # (6,) uint8 — INPT0..INPT5


def initial_tia_state() -> TIAState:
    # INPT4 / INPT5 idle high (D7=1, no trigger pressed). INPT0-3 (paddle
    # pots) default to $80 — "centred". Proper dump-pot timing is a
    # P6 follow-up.
    inpt_init = jnp.array([0x80, 0x80, 0x80, 0x80, 0x80, 0x80], dtype=jnp.uint8)
    return TIAState(
        registers=jnp.zeros((NUM_REGISTERS,), dtype=jnp.uint8),
        scanline_cycle=0,
        scanline=0,
        frame=0,
        wsync_pending=False,
        framebuffer=jnp.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=jnp.uint8),
        p0_x=0,
        p1_x=0,
        m0_x=0,
        m1_x=0,
        bl_x=0,
        collisions=jnp.zeros((8,), dtype=jnp.uint8),
        vsync_active=False,
        vblank_active=False,
        inpt=inpt_init,
    )


# --------------------------------------------------------------------------- #
# Register access
# --------------------------------------------------------------------------- #

def tia_peek(tia: TIAState, addr: int) -> int:
    """Read a TIA register.

    Reg ranges (the TIA decodes only A0-A3 for reads):
      0..7  → collision latches (\$30-\$37), set by `render_scanline`.
      8..13 → INPT0..INPT5 (\$38-\$3D). P6 wires the trigger inputs
              INPT4 / INPT5 via `set_trigger`; INPT0-INPT3 (paddle pots)
              default to \$80 (centred) — proper dump-pot timing is a
              P6 follow-up.
      14,15 → unused, return 0.
    """
    reg = addr & 0x0F
    if reg < 8:
        return int(tia.collisions[reg])
    if reg < 14:
        return int(tia.inpt[reg - 8])
    return 0


def set_trigger(tia: TIAState, player: int, pressed: bool) -> TIAState:
    """Set the fire-button line for player 0 or 1. Active-low: pressed →
    D7=0, released → D7=1. Touches only the trigger bit; other INPT
    state is preserved."""
    idx = 4 if player == 0 else 5
    new_byte = 0x00 if pressed else 0x80
    return tia._replace(inpt=tia.inpt.at[idx].set(jnp.uint8(new_byte)))


def _resp_position(scanline_cycle: int) -> int:
    """Approximate RESP* timing: maps the current scanline_cycle to a
    sprite horizontal position in [0, 159].

    Real TIA: during HBLANK (~first 22 CPU cycles of the scanline) RESP
    latches to a small constant position; during the visible region the
    position reflects the current beam location. We use a single linear
    formula `scanline_cycle * 3 - 68` (CPU cycles × 3 = colour clocks,
    minus the 68-color-clock HBLANK), clamped into [0, 159]. Off by a
    few pixels vs. a beam-accurate model — good enough for sprite-
    positioning tests; beam-accurate timing lands in P3f.
    """
    pos = scanline_cycle * 3 - 68
    if pos < 0:
        return 0
    if pos > 159:
        return 159
    return pos


def _hm_offset(hm: int) -> int:
    """Convert an HM register byte to a signed motion offset.

    Only the high nibble matters and is interpreted 4-bit two's complement
    (range +7..-8). Positive values move the sprite LEFT, negative RIGHT,
    matching Stella's convention.
    """
    high = (hm >> 4) & 0x0F
    return high - 16 if high >= 8 else high


def tia_poke(tia: TIAState, addr: int, value: int) -> TIAState:
    """Write a TIA register. Stores the byte and applies side-effects:
    WSYNC stall (P3a), player position reset / horizontal motion (P3c).
    Missile + ball, collisions, VSYNC, etc. land in later P3 sub-phases.
    """
    reg = addr & 0x3F                                    # TIA decodes A0–A5
    new_registers = tia.registers.at[reg].set(jnp.uint8(value & 0xFF))
    new_tia = tia._replace(registers=new_registers)

    if reg == W_WSYNC:
        new_tia = new_tia._replace(wsync_pending=True)
    elif reg == W_VSYNC:
        new_vsync = (value & 0x02) != 0
        if tia.vsync_active and not new_vsync:
            # 1→0 edge: frame boundary. Increment frame counter, reset
            # scanline. Beam-accurate emulators reset scanline_cycle too;
            # we follow that convention.
            new_tia = new_tia._replace(
                vsync_active=False,
                frame=tia.frame + 1,
                scanline=0,
                scanline_cycle=0,
            )
        else:
            new_tia = new_tia._replace(vsync_active=new_vsync)
    elif reg == W_VBLANK:
        new_tia = new_tia._replace(vblank_active=(value & 0x02) != 0)
    elif reg == W_RESP0:
        new_tia = new_tia._replace(p0_x=_resp_position(new_tia.scanline_cycle))
    elif reg == W_RESP1:
        new_tia = new_tia._replace(p1_x=_resp_position(new_tia.scanline_cycle))
    elif reg == W_RESM0:
        new_tia = new_tia._replace(m0_x=_resp_position(new_tia.scanline_cycle))
    elif reg == W_RESM1:
        new_tia = new_tia._replace(m1_x=_resp_position(new_tia.scanline_cycle))
    elif reg == W_RESBL:
        new_tia = new_tia._replace(bl_x=_resp_position(new_tia.scanline_cycle))
    elif reg == W_HMOVE:
        hmp0 = int(new_tia.registers[W_HMP0])
        hmp1 = int(new_tia.registers[W_HMP1])
        hmm0 = int(new_tia.registers[W_HMM0])
        hmm1 = int(new_tia.registers[W_HMM1])
        hmbl = int(new_tia.registers[W_HMBL])
        new_tia = new_tia._replace(
            p0_x=(new_tia.p0_x - _hm_offset(hmp0)) % 160,
            p1_x=(new_tia.p1_x - _hm_offset(hmp1)) % 160,
            m0_x=(new_tia.m0_x - _hm_offset(hmm0)) % 160,
            m1_x=(new_tia.m1_x - _hm_offset(hmm1)) % 160,
            bl_x=(new_tia.bl_x - _hm_offset(hmbl)) % 160,
        )
    elif reg == W_HMCLR:
        # HMCLR zeros all five horizontal-motion registers (HMP0/HMP1/HMM0/HMM1/HMBL).
        new_registers = (
            new_tia.registers
                .at[W_HMP0].set(jnp.uint8(0))
                .at[W_HMP1].set(jnp.uint8(0))
                .at[W_HMM0].set(jnp.uint8(0))
                .at[W_HMM1].set(jnp.uint8(0))
                .at[W_HMBL].set(jnp.uint8(0))
        )
        new_tia = new_tia._replace(registers=new_registers)
    elif reg == W_CXCLR:
        new_tia = new_tia._replace(collisions=jnp.zeros((8,), dtype=jnp.uint8))

    return new_tia


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #

def tia_advance(tia: TIAState, cpu_cycles: int) -> TIAState:
    """Advance the TIA by `cpu_cycles` CPU cycles, rendering each
    completed scanline into the framebuffer.

    Rendering uses the register state as of the end-of-scanline moment
    (current register file). Mid-scanline register changes are not yet
    captured — beam-accurate rendering lands in P3f.
    """
    total = tia.scanline_cycle + cpu_cycles
    new_sc = total % NTSC_CPU_CYCLES_PER_SCANLINE
    line_advance = total // NTSC_CPU_CYCLES_PER_SCANLINE

    fb = tia.framebuffer
    new_collisions = tia.collisions
    if line_advance > 0:
        scanline_pixels = render_scanline(tia)
        new_collisions = _detect_collisions(tia._replace(collisions=new_collisions))
        # When VBLANK is active, the TIA blanks its output — completed
        # scanlines are NOT written to the framebuffer. Collision
        # detection still runs (matching real hardware).
        if not tia.vblank_active:
            for i in range(line_advance):
                completed_line = (tia.scanline + i) % NTSC_SCANLINES_PER_FRAME
                if completed_line < SCREEN_HEIGHT:
                    fb = fb.at[completed_line].set(scanline_pixels)

    new_line = tia.scanline + line_advance
    # PXC1-x: don't increment the frame counter on scanline-wrap. The
    # frame counter is driven *only* by the software VSYNC 1→0 edge (see
    # `tia_poke` for `W_VSYNC`). The previous "safety fallback" of also
    # bumping `frame` here caused a double-count for any ROM that drove
    # VSYNC normally: the wrap fired one or two scanlines BEFORE the
    # VSYNC handler did, and both incremented `frame` for the same
    # frame boundary — which is why `run_until_frame` was completing
    # every other "frame" in just ~80 CPU cycles (one scanline) instead
    # of the natural ~19,912.
    new_line = new_line % NTSC_SCANLINES_PER_FRAME
    return tia._replace(
        scanline_cycle=new_sc,
        scanline=new_line,
        framebuffer=fb,
        collisions=new_collisions,
    )


# --------------------------------------------------------------------------- #
# Rendering — P3b: playfield only
# --------------------------------------------------------------------------- #

def _playfield_bits(pf0: int, pf1: int, pf2: int) -> list[int]:
    """The TIA's 20-bit playfield pattern for the LEFT half of the scanline.

    Bit-order quirks of the three registers (matches xitari/Stella):
      PF0 — only the HIGH nibble is used. PF0 bit 4 is the leftmost
            playfield pixel; bits 4, 5, 6, 7 → playfield pixels 0..3.
      PF1 — all 8 bits used, MSB-first. PF1 bit 7 → pixel 4; bit 0 → pixel 11.
      PF2 — all 8 bits used, LSB-first. PF2 bit 0 → pixel 12; bit 7 → pixel 19.
    """
    bits = []
    for b in range(4):
        bits.append((pf0 >> (4 + b)) & 1)        # PF0: bits 4..7
    for b in range(8):
        bits.append((pf1 >> (7 - b)) & 1)        # PF1: bits 7..0
    for b in range(8):
        bits.append((pf2 >> b) & 1)              # PF2: bits 0..7
    return bits


def _playfield_pixels(tia: TIAState) -> list[int]:
    """Internal helper: 160-element Python list of playfield colours."""
    pf0    = int(tia.registers[W_PF0])
    pf1    = int(tia.registers[W_PF1])
    pf2    = int(tia.registers[W_PF2])
    ctrlpf = int(tia.registers[W_CTRLPF])
    colupf = int(tia.registers[W_COLUPF])
    colubk = int(tia.registers[W_COLUBK])
    reflected = (ctrlpf & 0x01) != 0

    left_bits = _playfield_bits(pf0, pf1, pf2)
    right_bits = list(reversed(left_bits)) if reflected else left_bits

    pixels: list[int] = []
    for bit in left_bits:
        pixels.extend([colupf if bit else colubk] * 4)
    for bit in right_bits:
        pixels.extend([colupf if bit else colubk] * 4)
    return pixels


def render_playfield_scanline(tia: TIAState) -> jnp.ndarray:
    """Return a 160-byte uint8 array of palette indices for one scanline,
    rendering only the playfield (no sprites). Kept for unit tests; the
    composite renderer `render_scanline` overlays sprites on top.
    """
    return jnp.array(_playfield_pixels(tia), dtype=jnp.uint8)


# --------------------------------------------------------------------------- #
# P3g — NUSIZ multi-copy + 2×/4×-wide player scaling
# --------------------------------------------------------------------------- #
#
# NUSIZ0 / NUSIZ1 low 3 bits encode the player layout (8 modes — match
# xitari M_NUSIZ table). The high 2 bits (4-5) control the missile
# *width* (1/2/4/8); player multi-copy follows the same low-nibble
# pattern as its missile.
#
#   000  1 copy, 1× wide
#   001  2 copies, 16-pixel spacing, 1× wide   ("close")
#   010  2 copies, 32-pixel spacing, 1× wide   ("medium")
#   011  3 copies, 16-pixel spacing, 1× wide
#   100  2 copies, 64-pixel spacing, 1× wide   ("wide")
#   101  1 copy, 2× wide   (double-sized player)
#   110  3 copies, 32-pixel spacing, 1× wide
#   111  1 copy, 4× wide   (quadruple-sized player)
#
# `_nusiz_player_layout` returns `(copy_offsets, scale)` so the same
# helper drives both the rendering and the collision-set construction.

_NUSIZ_PLAYER_LAYOUT = {
    0b000: ((0,),                 1),
    0b001: ((0, 16),              1),
    0b010: ((0, 32),              1),
    0b011: ((0, 16, 32),          1),
    0b100: ((0, 64),              1),
    0b101: ((0,),                 2),
    0b110: ((0, 32, 64),          1),
    0b111: ((0,),                 4),
}


def _nusiz_player_layout(nusiz: int) -> tuple[tuple[int, ...], int]:
    """Decode NUSIZ low 3 bits into `(copy_offsets, scale)` where
    `copy_offsets` are the per-copy X displacements from the sprite's
    base position and `scale` is the per-bit pixel width (1/2/4)."""
    return _NUSIZ_PLAYER_LAYOUT[nusiz & 0x07]


def _overlay_player(pixels: list[int], tia: TIAState, player: int) -> None:
    """Paint player 0 or 1 onto `pixels` (mutated in place).

    P3g: respects NUSIZ low 3 bits — multi-copy (2 or 3 copies with
    close/medium/wide spacing) and 2×/4× horizontal scaling for the
    single-copy modes. The same NUSIZ value also drives the matching
    missile via `_overlay_missile` and the collision-set computation
    in `_object_pixel_sets`.
    """
    grp_reg = W_GRP0 if player == 0 else W_GRP1
    grp = int(tia.registers[grp_reg])
    if grp == 0:
        return  # nothing to draw
    color_reg = W_COLUP0 if player == 0 else W_COLUP1
    refp_reg = W_REFP0 if player == 0 else W_REFP1
    nusiz_reg = W_NUSIZ0 if player == 0 else W_NUSIZ1
    color = int(tia.registers[color_reg])
    reflected = (int(tia.registers[refp_reg]) & 0x08) != 0
    x = tia.p0_x if player == 0 else tia.p1_x
    copy_offsets, scale = _nusiz_player_layout(int(tia.registers[nusiz_reg]))

    # GRP is rendered with bit 7 as the LEFTMOST pixel by default. When
    # REFP.D3 is set, the bit order is reversed. Each of the 8 bits
    # spans `scale` screen pixels; each of the (1, 2, or 3) copies is
    # painted at `x + copy_offset`.
    for copy_off in copy_offsets:
        base = (x + copy_off) % 160
        for i in range(8):
            bit_idx = i if reflected else (7 - i)
            if (grp >> bit_idx) & 1:
                for k in range(scale):
                    px = (base + i * scale + k) % 160
                    pixels[px] = color


def _overlay_missile(pixels: list[int], tia: TIAState, missile: int) -> None:
    """Paint missile 0 or 1 onto `pixels`. NUSIZ low 3 bits select
    multi-copy + spacing (same layout as the matching player); NUSIZ
    bits 4-5 set each copy's *width* (1/2/4/8 pixels). Missile uses
    the same colour as its associated player.
    """
    enam_reg = W_ENAM0 if missile == 0 else W_ENAM1
    if (int(tia.registers[enam_reg]) & 0x02) == 0:
        return  # disabled
    color = int(tia.registers[W_COLUP0 if missile == 0 else W_COLUP1])
    nusiz = int(tia.registers[W_NUSIZ0 if missile == 0 else W_NUSIZ1])
    width = 1 << ((nusiz >> 4) & 0x03)
    copy_offsets, _ = _nusiz_player_layout(nusiz)
    x = tia.m0_x if missile == 0 else tia.m1_x
    for copy_off in copy_offsets:
        base = (x + copy_off) % 160
        for i in range(width):
            pixels[(base + i) % 160] = color


def _overlay_ball(pixels: list[int], tia: TIAState) -> None:
    """Paint the ball. Uses COLUPF for colour; size 1/2/4/8 from CTRLPF bits 4-5."""
    if (int(tia.registers[W_ENABL]) & 0x02) == 0:
        return
    color = int(tia.registers[W_COLUPF])
    ctrlpf = int(tia.registers[W_CTRLPF])
    size = 1 << ((ctrlpf >> 4) & 0x03)
    x = tia.bl_x
    for i in range(size):
        pixels[(x + i) % 160] = color


def _overlay_playfield(pixels: list[int], tia: TIAState) -> None:
    """Re-paint the playfield (without disturbing the background) over an
    already-rendered scanline. Used by the CTRLPF.D2 (PFP) priority swap:
    when the PF-priority bit is set, the playfield + ball composite *on
    top* of players + missiles, so the renderer paints sprites first and
    then drops the playfield bits on top via this helper.
    """
    pf0    = int(tia.registers[W_PF0])
    pf1    = int(tia.registers[W_PF1])
    pf2    = int(tia.registers[W_PF2])
    ctrlpf = int(tia.registers[W_CTRLPF])
    colupf = int(tia.registers[W_COLUPF])
    reflected = (ctrlpf & 0x01) != 0
    left_bits = _playfield_bits(pf0, pf1, pf2)
    right_bits = list(reversed(left_bits)) if reflected else left_bits
    for i, bit in enumerate(left_bits):
        if bit:
            for k in range(4):
                pixels[i * 4 + k] = colupf
    for i, bit in enumerate(right_bits):
        if bit:
            for k in range(4):
                pixels[80 + i * 4 + k] = colupf


def render_scanline(tia: TIAState) -> jnp.ndarray:
    """Composite renderer (pixel-only — collisions live in `_scanline_with_collisions`).

    Two priority modes, selected by CTRLPF bit 2 (PFP):

      PFP=0 (default):    bg ← pf ← bl ← M1 ← P1 ← M0 ← P0
      PFP=1 (priority):   bg ← M1 ← P1 ← M0 ← P0 ← pf ← bl

    With PFP set, playfield and ball composite *on top* of players and
    missiles — the canonical use case being a paddle game's score
    display or a playfield maze that should never be covered by a
    sprite. P3l adds this; before P3l every game ran in PFP=0 mode.
    """
    ctrlpf = int(tia.registers[W_CTRLPF])
    pfp = (ctrlpf & 0x04) != 0

    if not pfp:
        # default priority
        pixels = _playfield_pixels(tia)
        _overlay_ball(pixels, tia)
        _overlay_missile(pixels, tia, missile=1)
        _overlay_player(pixels, tia, player=1)
        _overlay_missile(pixels, tia, missile=0)
        _overlay_player(pixels, tia, player=0)
    else:
        # PFP set — playfield + ball on top of sprites. Start from the
        # background only (we re-overlay the playfield at the end), draw
        # players + missiles, then drop the playfield + ball on top.
        colubk = int(tia.registers[W_COLUBK])
        pixels = [colubk] * 160
        _overlay_missile(pixels, tia, missile=1)
        _overlay_player(pixels, tia, player=1)
        _overlay_missile(pixels, tia, missile=0)
        _overlay_player(pixels, tia, player=0)
        _overlay_playfield(pixels, tia)
        _overlay_ball(pixels, tia)

    return jnp.array(pixels, dtype=jnp.uint8)


# --------------------------------------------------------------------------- #
# Collision detection (P3e)
# --------------------------------------------------------------------------- #

def _object_pixel_sets(tia: TIAState) -> dict[str, set[int]]:
    """Return per-object sets of screen-pixel indices that the object covers
    on the current scanline. Used by both rendering and collision detection.
    """
    # Playfield
    pf0    = int(tia.registers[W_PF0])
    pf1    = int(tia.registers[W_PF1])
    pf2    = int(tia.registers[W_PF2])
    ctrlpf = int(tia.registers[W_CTRLPF])
    left_bits = _playfield_bits(pf0, pf1, pf2)
    right_bits = list(reversed(left_bits)) if (ctrlpf & 0x01) else left_bits
    pf: set[int] = set()
    for i, bit in enumerate(left_bits):
        if bit:
            pf.update(range(i * 4, i * 4 + 4))
    for i, bit in enumerate(right_bits):
        if bit:
            pf.update(range(80 + i * 4, 80 + i * 4 + 4))

    # Ball
    bl: set[int] = set()
    if int(tia.registers[W_ENABL]) & 0x02:
        size = 1 << ((ctrlpf >> 4) & 0x03)
        for i in range(size):
            bl.add((tia.bl_x + i) % 160)

    # Players — respect NUSIZ multi-copy + scale (P3g) so collisions
    # involving the 2nd/3rd copies and the wide modes are detected.
    def _player_set(grp_reg: int, refp_reg: int, nusiz_reg: int, x: int) -> set[int]:
        grp = int(tia.registers[grp_reg])
        if not grp:
            return set()
        reflected = (int(tia.registers[refp_reg]) & 0x08) != 0
        copy_offsets, scale = _nusiz_player_layout(int(tia.registers[nusiz_reg]))
        out: set[int] = set()
        for copy_off in copy_offsets:
            base = (x + copy_off) % 160
            for i in range(8):
                bit_idx = i if reflected else (7 - i)
                if (grp >> bit_idx) & 1:
                    for k in range(scale):
                        out.add((base + i * scale + k) % 160)
        return out

    p0 = _player_set(W_GRP0, W_REFP0, W_NUSIZ0, tia.p0_x)
    p1 = _player_set(W_GRP1, W_REFP1, W_NUSIZ1, tia.p1_x)

    # Missiles — same multi-copy from NUSIZ low 3 bits, width from
    # NUSIZ bits 4-5.
    def _missile_set(enam_reg: int, nusiz_reg: int, x: int) -> set[int]:
        if (int(tia.registers[enam_reg]) & 0x02) == 0:
            return set()
        nusiz = int(tia.registers[nusiz_reg])
        width = 1 << ((nusiz >> 4) & 0x03)
        copy_offsets, _ = _nusiz_player_layout(nusiz)
        out: set[int] = set()
        for copy_off in copy_offsets:
            base = (x + copy_off) % 160
            for i in range(width):
                out.add((base + i) % 160)
        return out

    m0 = _missile_set(W_ENAM0, W_NUSIZ0, tia.m0_x)
    m1 = _missile_set(W_ENAM1, W_NUSIZ1, tia.m1_x)

    return {"pf": pf, "bl": bl, "p0": p0, "p1": p1, "m0": m0, "m1": m1}


def _detect_collisions(tia: TIAState) -> jnp.ndarray:
    """Inspect this scanline's object sets and OR new collision bits into
    the existing `tia.collisions` latches. Returns the new 8-byte array.

    Bit layout (D6 + D7 per Stella docs):
      CXM0P  ($30): D7 = M0-P1, D6 = M0-P0
      CXM1P  ($31): D7 = M1-P0, D6 = M1-P1
      CXP0FB ($32): D7 = P0-PF, D6 = P0-BL
      CXP1FB ($33): D7 = P1-PF, D6 = P1-BL
      CXM0FB ($34): D7 = M0-PF, D6 = M0-BL
      CXM1FB ($35): D7 = M1-PF, D6 = M1-BL
      CXBLPF ($36): D7 = BL-PF, D6 = unused (0)
      CXPPMM ($37): D7 = P0-P1, D6 = M0-M1
    """
    objs = _object_pixel_sets(tia)
    c = list(int(b) for b in tia.collisions)
    p0, p1, m0, m1, bl, pf = objs["p0"], objs["p1"], objs["m0"], objs["m1"], objs["bl"], objs["pf"]

    def _hit(a: set[int], b: set[int]) -> bool:
        return bool(a) and bool(b) and bool(a & b)

    if _hit(m0, p1): c[0] |= 0x80
    if _hit(m0, p0): c[0] |= 0x40
    if _hit(m1, p0): c[1] |= 0x80
    if _hit(m1, p1): c[1] |= 0x40
    if _hit(p0, pf): c[2] |= 0x80
    if _hit(p0, bl): c[2] |= 0x40
    if _hit(p1, pf): c[3] |= 0x80
    if _hit(p1, bl): c[3] |= 0x40
    if _hit(m0, pf): c[4] |= 0x80
    if _hit(m0, bl): c[4] |= 0x40
    if _hit(m1, pf): c[5] |= 0x80
    if _hit(m1, bl): c[5] |= 0x40
    if _hit(bl, pf): c[6] |= 0x80
    if _hit(p0, p1): c[7] |= 0x80
    if _hit(m0, m1): c[7] |= 0x40
    return jnp.array(c, dtype=jnp.uint8)


def tia_apply_wsync(tia: TIAState) -> tuple[int, TIAState]:
    """Resolve a pending WSYNC stall: stall the CPU to the next scanline
    boundary and clear the flag.

    Returns the stall-cycle count (to be added to `state.cycles`) and the
    updated TIA state with `scanline_cycle == 0` (start of next scanline)
    and `wsync_pending == False`.

    No-op (returns 0 stall) if there is no pending WSYNC or if the TIA
    is already at a scanline boundary.
    """
    if not tia.wsync_pending:
        return 0, tia
    stall = (NTSC_CPU_CYCLES_PER_SCANLINE - tia.scanline_cycle) % NTSC_CPU_CYCLES_PER_SCANLINE
    new_tia = tia_advance(tia, stall)._replace(wsync_pending=False)
    return stall, new_tia
