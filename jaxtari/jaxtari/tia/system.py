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

Phase P3a status (this module): register file + scanline/frame timing +
WSYNC. Reads still return 0 stubs for collisions and INPT*. Playfield,
player sprites, missiles, ball, collisions, HMOVE-driven positioning,
VSYNC-driven frame ending, and framebuffer rendering all land in
subsequent sub-phases (see PORTING_PLAN.md §5 P3).
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

    Attributes
    ----------
    registers : (64,) uint8
        Last byte written to each register address. In P3a this is just a
        record of CPU writes; later sub-phases interpret these fields to
        generate output (e.g. PF0/PF1/PF2 → playfield bits, GRP0 → player 0
        pixels, COLUP0 → player 0 colour).
    scanline_cycle : int
        Current CPU cycle within the current scanline (0..75).
    scanline : int
        Current scanline within the current frame (0..NTSC_SCANLINES_PER_FRAME-1).
    frame : int
        Monotonic frame counter — increments when `scanline` wraps from 261
        back to 0.
    wsync_pending : bool
        Set by a write to WSYNC (\$02); consumed by `tia_apply_wsync` to
        stall the CPU to the next scanline boundary.
    framebuffer : (SCREEN_HEIGHT, SCREEN_WIDTH) uint8
        Indexed pixel colours. Zeros in P3a; populated by the rendering
        sub-phases.
    """
    registers: jnp.ndarray
    scanline_cycle: int
    scanline: int
    frame: int
    wsync_pending: bool
    framebuffer: jnp.ndarray


def initial_tia_state() -> TIAState:
    return TIAState(
        registers=jnp.zeros((NUM_REGISTERS,), dtype=jnp.uint8),
        scanline_cycle=0,
        scanline=0,
        frame=0,
        wsync_pending=False,
        framebuffer=jnp.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=jnp.uint8),
    )


# --------------------------------------------------------------------------- #
# Register access
# --------------------------------------------------------------------------- #

def tia_peek(tia: TIAState, addr: int) -> int:
    """Read a TIA register. P3a stub: all readable registers (collisions,
    INPT*) return 0. Proper values land in P3e (collisions) and a later
    input-handling phase."""
    return 0


def tia_poke(tia: TIAState, addr: int, value: int) -> TIAState:
    """Write a TIA register. Stores the byte and applies P3a side-effects
    (currently just WSYNC). Other side-effecting writes (HMOVE, RES*,
    CXCLR, …) land in later P3 sub-phases.
    """
    reg = addr & 0x3F                                    # TIA decodes A0–A5
    new_registers = tia.registers.at[reg].set(jnp.uint8(value & 0xFF))
    new_tia = tia._replace(registers=new_registers)
    if reg == W_WSYNC:
        new_tia = new_tia._replace(wsync_pending=True)
    return new_tia


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #

def tia_advance(tia: TIAState, cpu_cycles: int) -> TIAState:
    """Advance the TIA by `cpu_cycles` CPU cycles.

    Updates `scanline_cycle`, `scanline`, and `frame`. No rendering or
    side-effects yet — those land in later sub-phases.
    """
    total = tia.scanline_cycle + cpu_cycles
    new_sc, line_advance = total % NTSC_CPU_CYCLES_PER_SCANLINE, total // NTSC_CPU_CYCLES_PER_SCANLINE
    new_line = tia.scanline + line_advance
    frame_advance, new_line = new_line // NTSC_SCANLINES_PER_FRAME, new_line % NTSC_SCANLINES_PER_FRAME
    new_frame = tia.frame + frame_advance
    return tia._replace(scanline_cycle=new_sc, scanline=new_line, frame=new_frame)


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
