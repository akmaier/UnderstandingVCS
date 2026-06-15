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

# P3i-a: color-clock scaffolding. The TIA actually operates at 3× the
# CPU rate — one CPU cycle = 3 color clocks. A scanline is 228 color
# clocks: 68 of HBLANK (the chip blanks output while the CRT beam
# retraces) followed by 160 visible pixels. `color_clock` on the TIA
# state tracks the current beam position 0..227 within the scanline,
# and `render_pixel(tia, c)` is the per-color-clock pixel kernel that
# P3i-b/c/d/e/f progressively replace `render_scanline` with.
COLOR_CLOCKS_PER_CPU_CYCLE = 3
HBLANK_COLOR_CLOCKS = 68
COLOR_CLOCKS_PER_SCANLINE = 228          # = NTSC_CPU_CYCLES_PER_SCANLINE × 3

# P3i-c: per-poke write delays, in color clocks. Ported verbatim from
# xitari `TIA::ourPokeDelayTable[64]` (xitari/emucore/TIA.cxx). The
# index is `addr & 0x3F` (the TIA's 6-bit write-register decode).
# A non-zero entry means a write to that register takes effect that
# many color clocks LATER than the write itself — so a write mid-
# scanline doesn't affect pixels at color clocks <= write_clock+0
# (or < write_clock+delay) but does affect later pixels of the same
# scanline. The -1 sentinel means "compute dynamically based on the
# current scanline phase" (only PF0/PF1/PF2 use this — see
# `_pf_dynamic_delay`).
_POKE_DELAY_TABLE = (
    0,  1,  0,  0,  8,  8,  0,  0,  0,  0,  0,  1,  1, -1, -1, -1,
    0,  0,  8,  8,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
)
# Dynamic-PF delay table — selected by `(color_clock / 3) & 3`.
# Matches xitari's `static const uInt32 d[4] = {4, 5, 2, 3}`.
_PF_DYNAMIC_DELAY = (4, 5, 2, 3)


def _pf_dynamic_delay(color_clock: int) -> int:
    """Compute the per-color-clock activation delay for a PF0/PF1/PF2
    write (which xitari's `ourPokeDelayTable` flags with -1). Real TIA:
    PF takes 2..5 color clocks to apply depending on where in the
    4-color-clock cycle the write lands."""
    return _PF_DYNAMIC_DELAY[(color_clock // 3) & 3]


def _poke_activation_delay(reg: int, color_clock: int) -> int:
    """Return the activation delay in color clocks for a TIA register
    write to `reg` (= addr & 0x3F) at beam position `color_clock`."""
    if reg < 0 or reg >= 64:
        return 0
    delay = _POKE_DELAY_TABLE[reg]
    if delay == -1:                                  # PF0/PF1/PF2 sentinel
        return _pf_dynamic_delay(color_clock)
    return delay


# P3i-f: HMOVE blank enable table, indexed by CPU cycle within the
# scanline (0..75). Ported from xitari's `ourHMOVEBlankEnableCycles[128]`
# (the [76..127] tail is unused — we only index 0..75). A True at cycle
# `x` means "writing HMOVE at this cycle triggers the blank bug, which
# blacks the first 8 visible color clocks of the scanline." Most games
# write HMOVE just after a WSYNC (cycle 0..21) to deliberately trigger
# the blank — visible as the "HMOVE comb" in many games' sprite work.
# The cycle-75 True covers the WSYNC-just-before-HMOVE pattern that
# crosses the scanline wrap.
_HMOVE_BLANK_ENABLE_CYCLES = (
    True,  True,  True,  True,  True,  True,  True,  True,  True,  True,   # 00-09
    True,  True,  True,  True,  True,  True,  True,  True,  True,  True,   # 10-19
    True,  False, False, False, False, False, False, False, False, False,  # 20-29
    False, False, False, False, False, False, False, False, False, False,  # 30-39
    False, False, False, False, False, False, False, False, False, False,  # 40-49
    False, False, False, False, False, False, False, False, False, False,  # 50-59
    False, False, False, False, False, False, False, False, False, False,  # 60-69
    False, False, False, False, False, True,                               # 70-75
)


def _hmove_blank_enabled_at(scanline_cycle: int) -> bool:
    """Return True if an HMOVE write at CPU cycle `scanline_cycle`
    within the scanline triggers the HMOVE-blank bug."""
    sc = int(scanline_cycle) & 0x7F                  # mod 128
    # Task #97 (2026-06-13): beam_sc >= 76 means the write crossed into
    # the NEXT scanline mid-instruction (enduro free-running lines strobe
    # HMOVE at the next line's cyc ~3 → beam_sc 79). Wrap to the next-line
    # cycle so the comb is still recognized — matching `_hmove_motion`
    # which already clamps >=76. Without this the >=76 write returned
    # False and CLOBBERED the current line's real comb. Mirror of jutari.
    if sc >= NTSC_CPU_CYCLES_PER_SCANLINE:
        sc -= NTSC_CPU_CYCLES_PER_SCANLINE
    if sc >= NTSC_CPU_CYCLES_PER_SCANLINE:
        return False                                  # still out of range
    return _HMOVE_BLANK_ENABLE_CYCLES[sc]

# Internal framebuffer height — covers scanlines 0..243 (everything xitari
# would ever show with its default `Display.Height=210` starting at
# `Display.YStart=34`). The unused 0..33 + 244..261 regions stay empty
# (VSYNC + post-overscan) but having them in the buffer lets the renderer
# index the framebuffer directly by absolute scanline number without any
# per-scanline offset arithmetic — which keeps the unit tests' `framebuffer[N]
# == scanline-N render` invariant intact (matches the pre-vertical-alignment
# layout) while still capturing scanlines 192..243 which the old 192-row
# framebuffer was dropping.
SCREEN_HEIGHT = 244

# ALE / xitari visible-region crop. The default `Display.YStart` is 34 and
# the default `Display.Height` is 210 (see xitari/emucore/Props.cxx:300-301).
# `StellaEnvironment.get_screen()` returns `framebuffer[Y_START:Y_START +
# VISIBLE_HEIGHT]` so the user-facing screen matches what xitari / ALE
# present — top 34 lines (VSYNC + VBLANK + any score-header area outside
# xitari's display window) cropped out, height standardised at 210.
Y_START = 34
VISIBLE_HEIGHT = 210


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
    # P3h: VDELP / VDELBL "vertical delay" shadow copies. When VDELPx
    # bit 0 is set, the player renders from the *old* GRPx (the value
    # captured the last time the OTHER player's GRP was written —
    # that's the trick games use to swap two sets of player graphics
    # synchronously). Same logic for the ball with VDELBL + ENABL.
    grp0_old: int   # uint8 shadow of GRP0
    grp1_old: int   # uint8 shadow of GRP1
    enabl_old: int  # uint8 shadow of ENABL
    # P4c (dump-pot): xitari's INPT0_3 implementation reads the paddle
    # pot through a capacitor that charges over `t = 1.6 · r · 0.01µF`
    # seconds (≈ `t · 1.19 MHz` CPU cycles). VBLANK's D7 bit grounds
    # the cap (dump enabled); the 1→0 transition releases it and
    # `dump_disabled_cycle` captures the moment so subsequent INPT
    # reads can compute "have we charged past the threshold yet?".
    # `paddle_use_dump_pot[i]` is the per-pin opt-in: False keeps
    # the simple "INPT returns inpt[i]" semantic used by the existing
    # tests; True activates the cycle-threshold path with
    # `paddle_resistance[i]` as the (xitari-scale) resistance.
    # `total_cycles` is a monotonic CPU cycle counter ticked by
    # `tia_advance` so the threshold comparison has a reference.
    total_cycles: int = 0
    dump_enabled: bool = False
    dump_disabled_cycle: int = 0
    paddle_use_dump_pot: jnp.ndarray = jnp.zeros((4,), dtype=bool)
    paddle_resistance: jnp.ndarray = jnp.zeros((4,), dtype=jnp.uint32)
    # P3i-a: current beam color-clock position within the scanline,
    # 0..227. Always derivable as `scanline_cycle * 3` at CPU-cycle
    # granularity, but stored as a field so it survives across the
    # boundary where scanline_cycle wraps (i.e. mid-CPU-cycle resolution
    # if a future P3i-c lets `tia_poke` apply at sub-cycle color clocks).
    # P3i-a/b only advance it on CPU-cycle boundaries — sub-cycle
    # resolution lands in P3i-c with `ourPokeDelayTable`.
    color_clock: int = 0
    # P3i-c: queue of pending writes (PF0/PF1/PF2 only for now) that
    # were issued mid-scanline by `tia_poke` and will activate during
    # the per-color-clock render loop at their `ourPokeDelayTable`-
    # derived color clock. Each entry is `(activation_color_clock, reg,
    # value)`. The list is drained scanline-by-scanline by `tia_advance`.
    # Other TIA register writes continue to apply immediately — only
    # the rendering-affecting registers whose mid-scanline timing
    # produces a visible artifact (Breakout brick-stripe, score-line
    # rainbows) need the deferred apply.
    pending_writes: tuple = ()
    # P3i-f: HMOVE blank bug — when HMOVE is written at the right
    # cycle within the scanline, the chip black-bars the first 8
    # visible color clocks (the "HMOVE comb"). True from HMOVE write
    # until those 8 pixels finish rendering; auto-cleared by the
    # per-color-clock render loop. xitari tracks this via
    # `myHMOVEBlankEnabled` + `ourHMOVEBlankEnableCycles[x]`.
    hmove_blank_pending: bool = False
    # Task #97 (2026-06-13): HMOVE-blank comb destined for the NEXT
    # scanline (set when an HMOVE write's beam_sc >= 76 crosses the
    # scanline boundary mid-instruction — e.g. enduro free-running lines).
    # Parked here so it doesn't clobber line N's comb; promoted to
    # `hmove_blank_pending` once the beam advances. Mirror of jutari.
    hmove_blank_pending_next: bool = False
    # Task #99: deferred HMOVE object motion for a strobe that crossed into
    # the NEXT scanline (beam_sc >= 76). xitari applies such a late strobe's
    # motion to line N+1, but jaxtari renders whole scanlines in `tia_advance`
    # AFTER the poke, so applying immediately would move M0/BL on the
    # already-completed line N (1 line early — the enduro road-marker offset).
    # Park the per-object motion deltas (p0,p1,m0,m1,bl) here; `tia_advance`
    # applies them after line N commits so N+1 onward use the moved positions.
    # Mirror of `hmove_blank_pending_next` + jutari. (0,0,0,0,0) = none.
    hmove_motion_next: tuple = (0, 0, 0, 0, 0)
    # Task #80: scanlines since the last frame boundary (does NOT wrap at
    # 262, unlike `scanline`). Drives xitari's max-scanlines frame cutoff:
    # xitari force-ends a frame after `myMaximumNumberOfScanlines` (=290
    # NTSC) lines if no software VSYNC came (TIA.cxx:2003). Needed so a
    # VSYNC-less boot-init burst (seaquest: ~455 lines before its first
    # VSYNC) is split into the SAME number of frames as xitari — without
    # it, the 64-frame boot runs one extra cart-loop iteration (one extra
    # RAM[$01] increment → $3f vs xitari's $3e). Reset to 0 on a software
    # VSYNC 1→0 edge (see `tia_poke`) and on the cutoff itself. Dormant
    # for normal frames (VSYNC fires at ~262 < 290). Mirror of jutari.
    lines_since_frame: int = 0
    # Task #103 (air_raid): xitari myVSYNCFinishClock (TIA.cxx:2020). Armed
    # to (frame-relative clock + 228) when VSYNC D1 is SET; a later CLEAR
    # only ends the frame once the beam reaches it — VSYNC must be HELD >= 1
    # scanline. 0x7FFFFFFF = disarmed (xitari's default). Mirror of jutari.
    vsync_finish_clock: int = 0x7FFFFFFF


def initial_tia_state() -> TIAState:
    # INPT4 / INPT5 idle high (D7=1, no trigger pressed). INPT0-3
    # (paddle pots) default to $80 — "centred". xitari's actual
    # behaviour is dynamic: while the pot capacitor is charging
    # through the position-dependent resistor, INPT reads
    # `0x80 | noise`; once charged, INPT reads `noise`. Real
    # paddle-position is encoded in the cycle of that transition.
    # PXC1-x round 4 diagnostic identified the INPT1/INPT3 reads
    # at Pong PCs $F62C/$F633 as the mid-frame divergence source,
    # but a quick static change to $00 *worsens* the gap (15 bytes
    # vs the current 10), so the proper fix is the full dump-pot
    # timing model — a P4c extension that hasn't landed yet.
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
        total_cycles=0,
        dump_enabled=False,
        dump_disabled_cycle=0,
        paddle_use_dump_pot=jnp.zeros((4,), dtype=bool),
        paddle_resistance=jnp.zeros((4,), dtype=jnp.uint32),
        grp0_old=0,
        grp1_old=0,
        enabl_old=0,
        color_clock=0,
        pending_writes=(),
        hmove_blank_pending=False,
        hmove_blank_pending_next=False,
        hmove_motion_next=(0, 0, 0, 0, 0),
        lines_since_frame=0,
        vsync_finish_clock=0x7FFFFFFF,
    )


# --------------------------------------------------------------------------- #
# Register access
# --------------------------------------------------------------------------- #

def tia_peek(tia: TIAState, addr: int) -> int:
    """Read a TIA register.

    Reg ranges (the TIA decodes only A0-A3 for reads):
      0..7  → collision latches (\$30-\$37), set by `render_scanline`.
      8..11 → INPT0..INPT3 (\$38-\$3B). Two modes per pin:
              * `paddle_use_dump_pot[i]` False (default) → return
                `inpt[i]` as-is. Preserves the simple "set_paddle stores
                a byte, INPT returns it" contract the existing P4c
                tests rely on.
              * `paddle_use_dump_pot[i]` True → real xitari-style
                dump-pot timing: D7 = 1 once `total_cycles` exceeds
                `dump_disabled_cycle + (1.6·r·0.01µF·1.19MHz)`,
                else D7 = 0. With `dump_enabled` the cap stays
                grounded → D7 = 0 always. This is what `set_paddle_resistance`
                opts a paddle into for the ALE-style paddle-action
                drive used by paddle games (Breakout, Pong, etc.).
      12,13 → INPT4..INPT5 (\$3C/\$3D), the fire triggers — wired
              by `set_trigger`.
      14,15 → unused, return 0.
    """
    reg = addr & 0x0F
    if reg < 8:
        return int(tia.collisions[reg])
    if reg < 12:
        # INPT0..INPT3 — paddle pots with optional dump-pot timing.
        i = reg - 8
        if bool(tia.paddle_use_dump_pot[i]):
            # Dump enabled: cap grounded → never charges → D7 stays 0.
            if tia.dump_enabled:
                return 0
            # xitari formula:
            #   t       = 1.6 · r · 0.01e-6     seconds
            #   needed  = int(t · 1.19e6)       CPU cycles
            #   D7 = 1 iff cycles > dump_disabled_cycle + needed.
            r       = int(tia.paddle_resistance[i])
            needed  = int(1.6 * r * 0.01e-6 * 1.19e6)
            charged = int(tia.total_cycles) > (int(tia.dump_disabled_cycle) + needed)
            return 0x80 if charged else 0
        return int(tia.inpt[i])
    if reg < 14:
        # INPT4..INPT5 — fire triggers.
        return int(tia.inpt[reg - 8])
    return 0


def set_trigger(tia: TIAState, player: int, pressed: bool) -> TIAState:
    """Set the fire-button line for player 0 or 1. Active-low: pressed →
    D7=0, released → D7=1. Touches only the trigger bit; other INPT
    state is preserved."""
    idx = 4 if player == 0 else 5
    new_byte = 0x00 if pressed else 0x80
    return tia._replace(inpt=tia.inpt.at[idx].set(jnp.uint8(new_byte)))


def set_paddle(tia: TIAState, paddle: int, value: int) -> TIAState:
    """P4c — set the paddle pot reading for one of the four paddles
    (0..3 → INPT0..INPT3).

    `value` is the paddle "wheel position" in the range 0..255. Real
    Atari paddles work via a capacitor that charges through a
    pot-controlled resistor; the INPT register reads D7 = 1 until the
    cap voltage clears a threshold, then D7 = 0. Games measure the
    paddle angle by polling the number of cycles between the start of
    the scanline and the moment INPT D7 flips to 0 — so on real
    hardware the *value* INPT returns at any single instant is just
    a boolean, the analog position is encoded in *timing*.

    The capacitor-charge model is more work than P4c needs; instead
    this helper just stores `value` directly in the INPT cell. Games
    that read INPT once per scanline (the vast majority) get a
    sensible per-paddle value back. Faithful dump-pot timing is a
    separate deferral — see STATUS.md P4c.
    """
    if paddle not in (0, 1, 2, 3):
        raise ValueError(f"paddle must be 0..3, got {paddle}")
    # Reset dump-pot opt-in: callers using the byte-store interface
    # don't want their inpt[i] value overridden by the cycle-threshold
    # formula on the next INPT read.
    new_use = tia.paddle_use_dump_pot.at[paddle].set(False)
    tia = tia._replace(paddle_use_dump_pot=new_use)
    return tia._replace(inpt=tia.inpt.at[paddle].set(jnp.uint8(value & 0xFF)))


def set_paddle_resistance(tia: TIAState, paddle: int, resistance: int) -> TIAState:
    """P4c (dump-pot) — opt one paddle into the cycle-threshold dump-pot
    model with a specific resistance value.

    `resistance` is in xitari's scale: `PADDLE_MIN = 27_450` (fully one
    direction) to `PADDLE_MAX = 790_196` (fully the other direction).
    Once set, INPT0/1/2/3 reads on that paddle no longer return the
    `inpt[i]` byte directly — instead `D7` flips from 0 to 1 after
    enough CPU cycles have elapsed since VBLANK D7's 1→0 transition,
    where `enough = int(1.6 · resistance · 0.01e-6 · 1.19e6)`. Games
    that poll INPT in a tight loop measuring cycle-count to D7 (which
    is essentially every paddle game — Breakout, Pong / Video Olympics,
    Casino, Warlords, …) read the resistance as a "where is the
    paddle?" timing value.

    This is the helper `StellaEnvironment` uses when `use_paddles=True`
    to drive paddle motion from LEFT/RIGHT actions; manual callers
    can also use it directly if they want full xitari-faithful
    paddle-pot semantics.
    """
    if paddle not in (0, 1, 2, 3):
        raise ValueError(f"paddle must be 0..3, got {paddle}")
    new_use  = tia.paddle_use_dump_pot.at[paddle].set(True)
    new_res  = tia.paddle_resistance.at[paddle].set(jnp.uint32(int(resistance) & 0xFFFFFFFF))
    return tia._replace(paddle_use_dump_pot=new_use, paddle_resistance=new_res)


def _resp_position(scanline_cycle: int) -> int:
    """Legacy RESP* timing helper kept for backward compatibility with
    pre-P3i-e tests. Maps `scanline_cycle` to a sprite horizontal
    position via the simple `scanline_cycle * 3 - 68` formula clamped
    to [0, 159].

    New code (post-P3i-e) should use `_resp_player_position(color_clock)`
    or `_resp_missile_ball_position(color_clock)` — they apply xitari's
    exact reset-during-HBLANK constants (3 for players, 2 for
    missiles/ball) and the +5 / +4 visible-region offsets.
    """
    pos = scanline_cycle * 3 - 68
    if pos < 0:
        return 0
    if pos > 159:
        return 159
    return pos


def _resp_player_position(color_clock: int) -> int:
    """P3i-e: xitari-exact RESP0/RESP1 sprite-reset position. During
    HBLANK (color clocks 0..67) the H counter is still in retrace and
    the player latches to constant position 3; in the visible region
    the position is `(color_clock - 68 + 5) % 160` — the 5-color-clock
    offset captures the propagation delay between the RESP* register
    write and the actual sprite-counter latch.

    Mirrors xitari `TIA::poke` case 0x10/0x11:
        `hpos < HBLANK ? 3 : ((hpos - HBLANK + 5) % 160)`
    """
    if color_clock < HBLANK_COLOR_CLOCKS:
        return 3
    return ((color_clock - HBLANK_COLOR_CLOCKS) + 5) % SCREEN_WIDTH


def _resp_missile_ball_position(color_clock: int) -> int:
    """P3i-e: xitari-exact RESM0/RESM1/RESBL reset position. Missile
    and ball share the same formula — HBLANK-constant 2, visible
    offset +4 — differing from the player formula by 1 in both
    constants (a real TIA-circuit asymmetry, not a quirk).

    Mirrors xitari `TIA::poke` case 0x12/0x13/0x14:
        `hpos < HBLANK ? 2 : ((hpos - HBLANK + 4) % 160)`
    """
    if color_clock < HBLANK_COLOR_CLOCKS:
        return 2
    return ((color_clock - HBLANK_COLOR_CLOCKS) + 4) % SCREEN_WIDTH


def _hm_offset(hm: int) -> int:
    """Convert an HM register byte to a signed motion offset.

    Only the high nibble matters and is interpreted 4-bit two's complement
    (range +7..-8). Positive values move the sprite LEFT, negative RIGHT,
    matching Stella's convention. This is the HBLANK-only case — the
    full cycle-dependent table lives at `_COMPLETE_MOTION_TABLE` and is
    exposed via `_hmove_motion(scanline_cycle, hm)`.
    """
    high = (hm >> 4) & 0x0F
    return high - 16 if high >= 8 else high


# P3i-f extension: full HMOVE motion table, indexed by [scanline_cycle][hm_nibble].
# Verbatim port of xitari's `ourCompleteMotionTable[128][16]` (entries 0..76
# used; the [77..127] tail is xitari's defensive padding for the % 228 clock
# modulo computation). Our sign convention is the NEGATION of xitari's:
# xitari stores e.g. `-1` at [0][1] and does `myPOSP0 += table[x][hm]`;
# we keep our existing `p0_x = (p0_x - _hmove_motion(sc, hm)) % 160` style,
# so the table values here are the negation of xitari's.
#
# Three structural regions:
#   * cycles 0..20: "HBLANK" pattern. Motion = `_hm_offset(hm)`-equivalent
#     (mostly +/-N for N in 0..8). HMOVE write right after a WSYNC lands
#     here — the typical case.
#   * cycles 21..54: ALL ZEROS. HMOVE written mid-scanline does NOT move
#     the sprites — this is the "mid-scanline HMOVE doesn't work" quirk
#     real games depend on (and have used both directions of for tricks).
#   * cycles 55..75: late-scanline pattern with extra negative shifts.
#     Used by exotic games that abuse HMOVE timing.
#   * cycle 76: HBLANK wrap — matches cycle 0.
#
# Each row is 16 entries indexed by the high nibble of the HM* register.
_COMPLETE_MOTION_TABLE = (
    # HBLANK (cycles 0..20) — mostly the standard HBLANK pattern.
    # Note xitari's row varies slightly at indices 7+ between rows 0..20.
    ( 0,  1,  2,  3,  4,  5,  6,  7, -8, -7, -6, -5, -4, -3, -2, -1),  # 0
    ( 0,  1,  2,  3,  4,  5,  6,  7, -8, -7, -6, -5, -4, -3, -2, -1),  # 1
    ( 0,  1,  2,  3,  4,  5,  6,  7, -8, -7, -6, -5, -4, -3, -2, -1),  # 2
    ( 0,  1,  2,  3,  4,  5,  6,  7, -8, -7, -6, -5, -4, -3, -2, -1),  # 3
    ( 0,  1,  2,  3,  4,  5,  6,  6, -8, -7, -6, -5, -4, -3, -2, -1),  # 4
    ( 0,  1,  2,  3,  4,  5,  5,  5, -8, -7, -6, -5, -4, -3, -2, -1),  # 5
    ( 0,  1,  2,  3,  4,  5,  5,  5, -8, -7, -6, -5, -4, -3, -2, -1),  # 6
    ( 0,  1,  2,  3,  4,  4,  4,  4, -8, -7, -6, -5, -4, -3, -2, -1),  # 7
    ( 0,  1,  2,  3,  3,  3,  3,  3, -8, -7, -6, -5, -4, -3, -2, -1),  # 8
    ( 0,  1,  2,  2,  2,  2,  2,  2, -8, -7, -6, -5, -4, -3, -2, -1),  # 9
    ( 0,  1,  2,  2,  2,  2,  2,  2, -8, -7, -6, -5, -4, -3, -2, -1),  # 10
    ( 0,  1,  1,  1,  1,  1,  1,  1, -8, -7, -6, -5, -4, -3, -2, -1),  # 11
    ( 0,  0,  0,  0,  0,  0,  0,  0, -8, -7, -6, -5, -4, -3, -2, -1),  # 12
    (-1, -1, -1, -1, -1, -1, -1, -1, -8, -7, -6, -5, -4, -3, -2, -1),  # 13
    (-1, -1, -1, -1, -1, -1, -1, -1, -8, -7, -6, -5, -4, -3, -2, -1),  # 14
    (-2, -2, -2, -2, -2, -2, -2, -2, -8, -7, -6, -5, -4, -3, -2, -2),  # 15
    (-3, -3, -3, -3, -3, -3, -3, -3, -8, -7, -6, -5, -4, -3, -3, -3),  # 16
    (-4, -4, -4, -4, -4, -4, -4, -4, -8, -7, -6, -5, -4, -4, -4, -4),  # 17
    (-4, -4, -4, -4, -4, -4, -4, -4, -8, -7, -6, -5, -4, -4, -4, -4),  # 18
    (-5, -5, -5, -5, -5, -5, -5, -5, -8, -7, -6, -5, -5, -5, -5, -5),  # 19
    (-6, -6, -6, -6, -6, -6, -6, -6, -8, -7, -6, -6, -6, -6, -6, -6),  # 20
    # Mid-scanline (cycles 21..54) — HMOVE does NOTHING here.
    *((0,) * 16 for _ in range(21, 55)),
    # Late-scanline (cycles 55..75) — motion table re-engages with
    # extra negative shifts. xitari rows have specific patterns;
    # we port the literal values (negated for our sign convention).
    ( 0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0),  # 55
    ( 0,  0,  0,  0,  0,  0,  1,  2,  0,  0,  0,  0,  0,  0,  0,  0),  # 56
    ( 0,  0,  0,  0,  0,  1,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0),  # 57
    ( 0,  0,  0,  0,  0,  1,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0),  # 58
    ( 0,  0,  0,  0,  1,  2,  3,  4,  0,  0,  0,  0,  0,  0,  0,  0),  # 59
    ( 0,  0,  0,  1,  2,  3,  4,  5,  0,  0,  0,  0,  0,  0,  0,  0),  # 60
    ( 0,  0,  1,  2,  3,  4,  5,  6,  0,  0,  0,  0,  0,  0,  0,  0),  # 61
    ( 0,  0,  1,  2,  3,  4,  5,  6,  0,  0,  0,  0,  0,  0,  0,  0),  # 62
    ( 0,  1,  2,  3,  4,  5,  6,  7,  0,  0,  0,  0,  0,  0,  0,  0),  # 63
    ( 1,  2,  3,  4,  5,  6,  7,  8,  0,  0,  0,  0,  0,  0,  0,  0),  # 64
    ( 2,  3,  4,  5,  6,  7,  8,  9,  0,  0,  0,  0,  0,  0,  0,  1),  # 65
    ( 2,  3,  4,  5,  6,  7,  8,  9,  0,  0,  0,  0,  0,  0,  0,  1),  # 66
    ( 3,  4,  5,  6,  7,  8,  9, 10,  0,  0,  0,  0,  0,  0,  1,  2),  # 67
    ( 4,  5,  6,  7,  8,  9, 10, 11,  0,  0,  0,  0,  0,  1,  2,  3),  # 68
    ( 5,  6,  7,  8,  9, 10, 11, 12,  0,  0,  0,  0,  1,  2,  3,  4),  # 69
    ( 5,  6,  7,  8,  9, 10, 11, 12,  0,  0,  0,  0,  1,  2,  3,  4),  # 70
    ( 6,  7,  8,  9, 10, 11, 12, 13,  0,  0,  0,  1,  2,  3,  4,  5),  # 71
    ( 7,  8,  9, 10, 11, 12, 13, 14,  0,  0,  1,  2,  3,  4,  5,  6),  # 72
    ( 8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7),  # 73
    ( 8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7),  # 74
    ( 0,  1,  2,  3,  4,  5,  6,  7, -8, -7, -6, -5, -4, -3, -2, -1),  # 75 (HBLANK wrap)
)
# Defensive: confirm we generated exactly 76 rows (0..75).
assert len(_COMPLETE_MOTION_TABLE) == 76


def _hmove_motion(scanline_cycle: int, hm: int) -> int:
    """Compute the HMOVE motion delta for a sprite given the CPU cycle
    within the scanline when HMOVE was written and the HM* register
    byte. Returns our-sign-convention offset (positive = move LEFT,
    negative = move RIGHT — same as `_hm_offset`).

    For `scanline_cycle in [0..20]` (typical HMOVE-after-WSYNC), the
    result matches `_hm_offset(hm)` for the common HM nibble values.
    For `scanline_cycle in [21..54]` (mid-scanline HMOVE), the result
    is 0 — the chip does NOT move the sprite. For [55..75], various
    cycle-dependent partial motions kick in (used by exotic ROMs).
    """
    sc = int(scanline_cycle) & 0x7F
    if sc >= NTSC_CPU_CYCLES_PER_SCANLINE:
        sc = 0                                       # past end → wraps to HBLANK
    return _COMPLETE_MOTION_TABLE[sc][(hm >> 4) & 0x0F]


def tia_poke(tia: TIAState, addr: int, value: int,
             beam_cc: int | None = None, beam_sc: int | None = None) -> TIAState:
    """Write a TIA register. Stores the byte and applies side-effects:
    WSYNC stall (P3a), player position reset / horizontal motion (P3c).
    Missile + ball, collisions, VSYNC, etc. land in later P3 sub-phases.

    **P3i-c**: writes to PF0/PF1/PF2 are *deferred* to their
    `ourPokeDelayTable` activation color clock instead of taking
    effect immediately — this lets games stomp the playfield
    mid-scanline (the Breakout brick stripe, score-line rainbows) and
    have each segment of the scanline render with the value that was
    current at THAT color clock. The deferred write is queued on
    `tia.pending_writes` and drained scanline-by-scanline by
    `tia_advance`'s per-color-clock loop. All other registers continue
    to apply immediately (WSYNC, VSYNC, RESP*, HMOVE, CXCLR, …)
    because their side effects must fire at the write time.

    **P3i-g (timing-only threading)**: `beam_cc`/`beam_sc` are the
    *effective* sub-instruction beam position at the moment this write
    hits the bus (instruction-start beam + cycles consumed so far),
    supplied by `Bus._bus_poke`. The beam-position-sensitive registers
    (PF0/1/2 defer, RES*, HMOVE) consult them so mid-instruction writes
    land at the right color clock — but the TIA is NOT advanced/rendered
    here (that happens once per instruction in `_tia_post_step`, so we
    never render mid-instruction and leak a pre-clear PF pattern into
    later scanlines). They default to the TIA's own counters so direct
    unit-test pokes keep the pre-threading behaviour.
    """
    reg = addr & 0x3F                                    # TIA decodes A0–A5
    value8 = value & 0xFF
    # COLU* (P0/P1/PF/BK): bit 0 of the color-luminance registers is
    # unused on real NMOS hardware — xitari masks `value & 0xFE` on every
    # poke (`case 0x06..0x09` in `TIA::poke`). Otherwise a ROM that writes
    # an odd luminance produces palette indices off by 1 from xitari (the
    # Stella NTSC palette has the same RGB at both N and N+1, so the
    # pixel *looks* identical, but the screen-conformance test counts the
    # byte-level diff — and pong/seaquest/etc. happen to write odd
    # values, accounting for almost all of pong's apparent screen gap).
    if reg in (W_COLUP0, W_COLUP1, W_COLUPF, W_COLUBK):
        value8 &= 0xFE
    if beam_cc is None:
        beam_cc = int(tia.color_clock)
    if beam_sc is None:
        beam_sc = int(tia.scanline_cycle)

    # Task #106 (partial-frame model): xitari's max-scanlines frame cutoff
    # (TIA.cxx:2003-2007) is evaluated at POKE time only — at the top of
    # TIA::poke, before the register switch — NOT every CPU step. If the beam
    # is already past 290 scanlines when a TIA register is written, the frame
    # ends here (immediate increment, like the VSYNC-clear path below). With a
    # poke-LESS stretch > 290 scanlines (qbert's RESET-boot wait loop, task
    # #52) the frame instead runs "grey" until the next poke — producing
    # xitari's boot→step sliver frame. `frame_clock` is the beam's
    # frame-relative color clock = xitari's (clock - myClockWhenFrameStarted);
    # `÷228 > 290` is exactly xitari's
    # `(clock - myClockWhenFrameStarted)/228 > myMaximumNumberOfScanlines`.
    # Beam position (scanline_cycle/color_clock) is preserved, matching the
    # old per-step cutoff that this replaces. Mirror of jutari's poke-time
    # cutoff in tia_poke!.
    _frame_clock_pk = int(tia.lines_since_frame) * COLOR_CLOCKS_PER_SCANLINE + int(beam_cc)
    if _frame_clock_pk // COLOR_CLOCKS_PER_SCANLINE > 290:
        tia = tia._replace(
            frame=tia.frame + 1,
            scanline=0,
            lines_since_frame=0,
            vsync_finish_clock=0x7FFFFFFF,
        )

    # P3i-c: defer PF0/PF1/PF2 to mid-scanline activation. Skip the
    # defer if we're in HBLANK (beam_cc < 68) — the activation
    # would land in the visible region anyway but pre-HBLANK pokes are
    # by convention "scanline setup" and just take effect for the
    # whole next scanline; deferring them generates a no-op gap that
    # produces the same render as the immediate apply.
    # P3i-g pt6: also defer the missile/ball enable bits (ENAM0/ENAM1/
    # ENABL). Their delay in `_POKE_DELAY_TABLE` is 0, so an immediate
    # apply lands at the write cc, but a *whole-scanline* batched apply
    # loses the mid-scanline transition: e.g. breakout disables ENAM1
    # at cc=105 on the last paddle scanline, expecting cols 0..7 (cc
    # 68..75) to still see M1 enabled. With this defer, the per-color-
    # clock render loop applies the ENAM1=0 at cc=105 and M1 paints
    # cols 0..7 as expected. (8 px residual on breakout row 195.)
    # P3i-g pt7: extend the defer set to NUSIZ/REFP/COLU/CTRLPF — all
    # "no-side-effect" render registers (the renderer just reads them).
    # Xitari's `TIA::poke` does `updateFrame(clock + delay)` BEFORE
    # applying any poke, so even a delay=0 write doesn't affect pixels
    # rendered BEFORE the write's CPU cycle on the same scanline. My
    # whole-scanline batched apply was lumping those pre-write pixels
    # into the post-write state (visible in pong's score-digit fragment
    # diffs + cross-scanline COLUBK rows). Deferring them through the
    # same `pending_writes` machinery makes the per-color-clock renderer
    # see the right value at every cc, matching xitari.
    # NOT deferred (have non-trivial side effects beyond a register store):
    #   GRP0/GRP1 (latch VDELP shadows on write — would need to defer the
    #     latch too), WSYNC (stalls CPU), VSYNC (frame-boundary trigger),
    #     RES* (sprite-position update), HMOVE (motion strobe),
    #     CXCLR (collision reset), VBLANK (also has dump-pot side effects).
    if reg in (W_PF0, W_PF1, W_PF2,
               W_ENAM0, W_ENAM1, W_ENABL,
               W_NUSIZ0, W_NUSIZ1,
               W_COLUP0, W_COLUP1, W_COLUPF, W_COLUBK,
               W_CTRLPF, W_REFP0, W_REFP1) \
            and beam_cc >= HBLANK_COLOR_CLOCKS:
        delay = _poke_activation_delay(reg, beam_cc)
        activation_clock = beam_cc + delay
        return tia._replace(
            pending_writes=tia.pending_writes
            + ((activation_clock, reg, value8),),
        )

    # Task #85 (2026-06-11): also defer RESMP0/RESMP1 (render gate
    # value). The 1→0 transition reposition fires IMMEDIATELY (in the
    # elseif chain at the bottom, like xitari does before the store).
    # The deferred *register store* lets the per-color-clock render see
    # the right RESMP value at each cc — pixels rendered BEFORE the
    # write's cc keep the OLD missile visibility, AFTER see the NEW.
    # Without this defer, ROMs that toggle RESMP* mid-scanline (e.g.
    # space_invaders + pitfall) regress in PXC-S. Same gate as
    # jutari's deferred-block addition.
    if reg in (W_RESMP0, W_RESMP1) and beam_cc >= HBLANK_COLOR_CLOCKS:
        delay = _poke_activation_delay(reg, beam_cc)
        activation_clock = beam_cc + delay
        # Task #103 (frostbite): the RESMP* 1→0 (unlock) reposition — snap the
        # missile to its player centre — fires IMMEDIATELY here at poke time
        # (like jutari, and like a RES* position write). This is DISTINCT from
        # putting the reposition in the DEFERRED apply (at the activation clock),
        # which regressed space_invaders (+30 px) / pitfall (+231 px); only the
        # register STORE (the missile-visibility gate) is deferred so the
        # per-color-clock render still sees the right RESMP value at each cc.
        # Before this, a visible-region RESMP write returned here WITHOUT
        # repositioning (the reposition lived only in the immediate elseif
        # chain, reached solely by HBLANK writes), so a missile released
        # mid-scanline kept a stale position and spuriously collided with its
        # player → frostbite CXM1P-D6 (RAM[$34]/[$36] = $47 vs xitari $07),
        # which also broke jaxtari≡jutari (PXC2).
        repl = {}
        old_value = int(tia.registers[reg])
        if (old_value & 0x02) != 0 and (value8 & 0x02) == 0:
            if reg == W_RESMP0:
                nusiz_lo = int(tia.registers[W_NUSIZ0]) & 0x07
                middle = 8 if nusiz_lo == 0x05 else 16 if nusiz_lo == 0x07 else 4
                repl["m0_x"] = (int(tia.p0_x) + middle) % 160
            else:
                nusiz_lo = int(tia.registers[W_NUSIZ1]) & 0x07
                middle = 8 if nusiz_lo == 0x05 else 16 if nusiz_lo == 0x07 else 4
                repl["m1_x"] = (int(tia.p1_x) + middle) % 160
        return tia._replace(
            pending_writes=tia.pending_writes
            + ((activation_clock, reg, value8),),
            **repl,
        )

    # Task #84 (2026-06-10): also defer GRP0/GRP1 (render value).
    # VDELP* / VDELBL latch SIDE EFFECTS must fire IMMEDIATELY at the
    # cart's write moment (matching xitari) — they latch the CURRENT
    # GRP0/GRP1/ENABL into grp0_old/grp1_old/enabl_old. But the
    # RENDERED register value is deferred to its activation_clock so
    # pre-write pixels use the OLD GRP value (matching xitari's
    # incremental render). Without this defer, jaxtari applies GRP1=240
    # to the WHOLE scanline including pre-write pixels — painting
    # pong's right paddle 1 row early (row 95 vs xitari's 96).
    if reg in (W_GRP0, W_GRP1) and beam_cc >= HBLANK_COLOR_CLOCKS:
        delay = _poke_activation_delay(reg, beam_cc)
        activation_clock = beam_cc + delay
        new_pending = tia.pending_writes + ((activation_clock, reg, value8),)
        # Task #91 round 2 (2026-06-12): for deferred GRP0/GRP1, the
        # VDELP / VDELBL shadow latch fires at ACTIVATION time inside
        # `_apply_pending_write` (the render-loop drain) — NOT here at
        # poke time. xitari's case 0x1B/0x1C runs `updateFrame(clock +
        # delay)` first and THEN mutates myGRP* + myDGRP*, so both the
        # register store and the shadow capture are effective at the
        # activation clock, against the register file as of that
        # moment. A poke-time snapshot collapses the shadow to ONE
        # value for the whole scanline — but pitfall's 6-digit kernel
        # rewrites GRP0/GRP1 four times mid-row, and each digit only
        # exists in the shadow for ~2 copy slots.
        return tia._replace(pending_writes=new_pending)

    new_registers = tia.registers.at[reg].set(jnp.uint8(value8))
    new_tia = tia._replace(registers=new_registers)

    if reg == W_WSYNC:
        new_tia = new_tia._replace(wsync_pending=True)
    elif reg == W_VSYNC:
        # Task #103 (air_raid): xitari-faithful VSYNC frame-end with the
        # myVSYNCFinishClock hold-gate (TIA.cxx:2011-2031). A VSYNC CLEAR
        # only ends the frame if VSYNC was HELD >= 1 scanline (228 color
        # clocks). The old logic ended the frame on ANY 1→0 edge, which
        # landed air_raid's 291-line frame one scanline early. frame_clock
        # = beam's frame-relative color clock = xitari's
        # (clock - myClockWhenFrameStarted). Mirror of jutari TIA.jl.
        new_vsync = (value & 0x02) != 0
        bc = int(beam_cc) if beam_cc is not None else int(tia.color_clock)
        frame_clock = int(tia.lines_since_frame) * COLOR_CLOCKS_PER_SCANLINE + bc
        if new_vsync:
            # SET — arm the finish clock 1 scanline out (each set re-arms).
            new_tia = new_tia._replace(
                vsync_active=True,
                vsync_finish_clock=frame_clock + COLOR_CLOCKS_PER_SCANLINE,
            )
        elif frame_clock >= int(tia.vsync_finish_clock):
            # CLEAR and VSYNC held >= 1 scanline → end the frame.
            # Task #80: reset lines_since_frame so the max-scanlines cutoff
            # only fires for genuinely VSYNC-less stretches.
            new_tia = new_tia._replace(
                vsync_active=False,
                frame=tia.frame + 1,
                scanline=0,
                scanline_cycle=0,
                lines_since_frame=0,
                vsync_finish_clock=0x7FFFFFFF,
            )
        else:
            new_tia = new_tia._replace(vsync_active=new_vsync)
    elif reg == W_VBLANK:
        new_tia = new_tia._replace(vblank_active=(value & 0x02) != 0)
        # P4c (dump-pot): VBLANK D7 grounds the paddle-pot cap. The 1→0
        # transition releases it; capture the cycle so subsequent
        # INPT0-3 reads can ask "have we charged past the threshold
        # yet?". Mirrors xitari's myDumpEnabled/myDumpDisabledCycle
        # tracking in TIA::poke for VBLANK.
        new_dump = (value & 0x80) != 0
        if tia.dump_enabled and not new_dump:
            # Falling edge — cap starts charging now.
            new_tia = new_tia._replace(
                dump_enabled=False,
                dump_disabled_cycle=int(new_tia.total_cycles),
            )
        elif new_dump and not tia.dump_enabled:
            # Rising edge — cap grounded.
            new_tia = new_tia._replace(dump_enabled=True)
    elif reg == W_RESP0:
        # P3i-e/g: xitari-exact RESP0 from the *effective* sub-instruction
        # beam color clock (beam_cc), not the stale instruction-start
        # tia.color_clock — HBLANK constant 3, visible +5 offset.
        new_tia = new_tia._replace(p0_x=_resp_player_position(beam_cc))
    elif reg == W_RESP1:
        new_tia = new_tia._replace(p1_x=_resp_player_position(beam_cc))
    elif reg == W_RESM0:
        # P3i-e: missile/ball use the +4 offset / HBLANK constant 2.
        new_tia = new_tia._replace(m0_x=_resp_missile_ball_position(beam_cc))
    elif reg == W_RESM1:
        new_tia = new_tia._replace(m1_x=_resp_missile_ball_position(beam_cc))
    elif reg == W_RESBL:
        new_tia = new_tia._replace(bl_x=_resp_missile_ball_position(beam_cc))
    elif reg == W_HMOVE:
        hmp0 = int(new_tia.registers[W_HMP0])
        hmp1 = int(new_tia.registers[W_HMP1])
        hmm0 = int(new_tia.registers[W_HMM0])
        hmm1 = int(new_tia.registers[W_HMM1])
        hmbl = int(new_tia.registers[W_HMBL])
        # P3i-f + P3i-g: trigger the HMOVE-blank bug if this write
        # lands at a cycle where it would on real hardware, AND use
        # the cycle-dependent motion table (`_hmove_motion` →
        # xitari's `ourCompleteMotionTable[x][hm>>4]`) instead of
        # the HBLANK-only `_hm_offset`. The two combine: most
        # writes hit at cycle 0..20 → blank fires + standard
        # HBLANK motion deltas. Mid-scanline writes (21..54) get
        # zero motion (sprite stays put) AND no blank. Late writes
        # (55..75) get partial-motion deltas. P3i-g: index by the
        # *effective* sub-instruction CPU cycle within the scanline
        # (beam_sc), not the stale instruction-start scanline_cycle.
        sc = int(beam_sc)
        blank = _hmove_blank_enabled_at(sc)
        # Task #97/#99: beam_sc >= 76 means the write crossed into the NEXT
        # scanline — BOTH its blank comb AND its object motion belong to line
        # N+1, not the about-to-be-committed line N. #97 deferred the comb;
        # #99 also defers the motion (`hmove_motion_next`, applied in
        # `tia_advance` after line N commits) — applying it immediately moved
        # M0/BL on line N (1 line early, the enduro road-marker offset).
        # Below 76 the strobe is within the current line: apply immediately.
        if sc >= NTSC_CPU_CYCLES_PER_SCANLINE:
            new_tia = new_tia._replace(
                hmove_blank_pending_next=blank,
                hmove_motion_next=(
                    _hmove_motion(sc, hmp0), _hmove_motion(sc, hmp1),
                    _hmove_motion(sc, hmm0), _hmove_motion(sc, hmm1),
                    _hmove_motion(sc, hmbl),
                ),
            )
        else:
            new_tia = new_tia._replace(
                p0_x=(new_tia.p0_x - _hmove_motion(sc, hmp0)) % 160,
                p1_x=(new_tia.p1_x - _hmove_motion(sc, hmp1)) % 160,
                m0_x=(new_tia.m0_x - _hmove_motion(sc, hmm0)) % 160,
                m1_x=(new_tia.m1_x - _hmove_motion(sc, hmm1)) % 160,
                bl_x=(new_tia.bl_x - _hmove_motion(sc, hmbl)) % 160,
                hmove_blank_pending=blank,
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
    elif reg == W_GRP0:
        # P3h: writing GRP0 latches the CURRENT GRP1 into grp1_old (so
        # a VDELP1-enabled P1 renders the *previous* GRP1 from now on).
        # Mirror for GRP1 below.
        new_tia = new_tia._replace(grp1_old=int(tia.registers[W_GRP1]))
    elif reg == W_GRP1:
        # Writing GRP1 also latches the CURRENT ENABL into enabl_old
        # — xitari/Stella convention: any GRP1 write moves the ball's
        # delayed-enable shadow forward (mirroring the original TIA
        # circuit that ties the ball's delayed-enable to a GRP1 strobe).
        # 2026-06-03 (jutari commit 20b5de0): when a previous ENABL
        # write is still PENDING (queued for deferred activation) and
        # its activation_clock is ≤ the current GRP1 beam, the live
        # tia.registers[W_ENABL] is STALE — we need the new value
        # for the shadow capture. xitari does the ENABL write
        # immediately and then GRP1's shadow captures the new value.
        # Without this lookback, jaxtari's enabl_old retains a
        # bit-1-set value across what should have been a clear,
        # causing spurious BL-PF collisions during VDELBL=1 (the
        # breakout-frame-92 scn 51-52 bug, which jutari hits too).
        effective_enabl = int(tia.registers[W_ENABL])
        for act_cc, w_reg, w_val in tia.pending_writes:
            if w_reg == W_ENABL and act_cc <= beam_cc:
                effective_enabl = int(w_val)
        new_tia = new_tia._replace(
            grp0_old=int(tia.registers[W_GRP0]),
            enabl_old=effective_enabl,
        )
    elif reg == W_RESMP0:
        # Task #85 (2026-06-11): on the RESMP0 1→0 transition, snap M0
        # to (POSP0 + middle) % 160, where middle depends on NUSIZ0's
        # low 3 bits: 0x05 (2× width) → 8; 0x07 (4× width) → 16; else 4.
        # Verbatim port of xitari `case 0x28` (TIA.cxx:2633-2658). Same
        # logic in jutari `tia_poke!`. The hide-while-RESMP=1 gate is in
        # `_missile_set`.
        old_value = int(tia.registers[W_RESMP0])
        if (old_value & 0x02) != 0 and (value & 0x02) == 0:
            nusiz_lo = int(new_tia.registers[W_NUSIZ0]) & 0x07
            middle = 8 if nusiz_lo == 0x05 else 16 if nusiz_lo == 0x07 else 4
            new_tia = new_tia._replace(m0_x=(new_tia.p0_x + middle) % 160)
    elif reg == W_RESMP1:
        old_value = int(tia.registers[W_RESMP1])
        if (old_value & 0x02) != 0 and (value & 0x02) == 0:
            nusiz_lo = int(new_tia.registers[W_NUSIZ1]) & 0x07
            middle = 8 if nusiz_lo == 0x05 else 16 if nusiz_lo == 0x07 else 4
            new_tia = new_tia._replace(m1_x=(new_tia.p1_x + middle) % 160)

    return new_tia


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #

def _apply_pending_write(tia: "TIAState", reg: int, val) -> "TIAState":
    """Apply ONE pending register write at its activation clock,
    INCLUDING the VDELP / VDELBL shadow latch for GRP0 / GRP1.

    Task #91 round 2 (2026-06-12): drain-side counterpart of xitari's
    poke handler — xitari runs `updateFrame(clock + delay)` and THEN
    mutates myGRP* + myDGRP*, i.e. both the store and the shadow
    capture are effective at the activation clock, against the
    register file as of that moment (earlier-activating writes already
    applied). Pitfall's 6-digit kernel depends on this: GRP0/GRP1 are
    rewritten 4× per HUD scanline and each digit value lives in the
    shadow for only ~2 copy slots. A same-scanline ENABL write with an
    earlier activation clock has already been stored by the drain, so
    GRP1's ENABL latch sees the new value — subsuming the 2026-06-03
    ENABL-lookback fix (breakout frame-92) with exact xitari ordering.
    Mirrors jutari `_apply_pending_write!`.
    """
    new_tia = tia._replace(registers=tia.registers.at[reg].set(jnp.uint8(val)))
    if reg == W_GRP0:
        return new_tia._replace(grp1_old=int(new_tia.registers[W_GRP1]))
    if reg == W_GRP1:
        return new_tia._replace(
            grp0_old=int(new_tia.registers[W_GRP0]),
            enabl_old=int(new_tia.registers[W_ENABL]),
        )
    return new_tia


def tia_advance(tia: TIAState, cpu_cycles: int) -> TIAState:
    """Advance the TIA by `cpu_cycles` CPU cycles, rendering each
    completed scanline into the framebuffer.

    **P3i-a / P3i-b**: the framebuffer write now goes through the
    `render_pixel` per-color-clock kernel (160 calls per visible
    scanline, looped over color clocks 68..227). For now the kernel
    sees the same register state at every color clock — so the output
    is bit-exact equivalent to the pre-P3i `render_scanline` write.
    Sub-phases P3i-c onwards add per-poke timing so mid-scanline
    register writes (the actual P3i payoff) start to bite. `tia.color_clock`
    also advances 3× per CPU cycle and is the field P3i-c/d/e/f hang
    write-delay arithmetic off of.
    """
    total = tia.scanline_cycle + cpu_cycles
    new_sc = total % NTSC_CPU_CYCLES_PER_SCANLINE
    line_advance = total // NTSC_CPU_CYCLES_PER_SCANLINE
    # P3i-a: color_clock advances by 3 per CPU cycle, wrapping at 228.
    new_color_clock = (tia.color_clock + cpu_cycles * COLOR_CLOCKS_PER_CPU_CYCLE) \
                      % COLOR_CLOCKS_PER_SCANLINE

    fb = tia.framebuffer
    new_collisions = tia.collisions
    # P3i-c: pending PF0/PF1/PF2 writes that activate during the next
    # scanline render. After we drain them into tia.registers (segment-
    # by-segment), pending_writes resets to empty for the upcoming
    # scanline. Multi-scanline advances apply the LATEST drained state
    # to every completed scanline after the first — equivalent to
    # "pokes affect the scanline they're issued in, plus any wholly-
    # contained subsequent scanlines until the next poke."
    pending = tia.pending_writes
    tia_for_render = tia                                  # local copy we mutate
    # Task #83 round 3: see comment around the assignment inside the
    # visible-render branch. Initialised here so the post-render
    # `new_hmove_blank` decision can read it even when line_advance == 0.
    wrote_framebuffer_this_advance = False
    if line_advance > 0:
        # Sort pending writes by activation_clock (so we apply them in
        # beam-order during the per-color-clock loop).
        pending_sorted = tuple(sorted(pending, key=lambda w: w[0]))

        # First completed scanline: do the per-color-clock render with
        # pending-write activation interleaved. Subsequent scanlines
        # use the post-drain register state (no mid-scanline writes).
        # P3i-d: collision detection now happens per-pixel inside the
        # render loop (via `_apply_pixel_collisions`) instead of via a
        # whole-scanline `_detect_collisions` call up-front — so mid-
        # scanline PF stomping (P3i-c) and future mid-scanline sprite
        # changes correctly affect collision bits at the right beam
        # position.
        running_collisions = list(int(b) for b in new_collisions)
        if not tia.vblank_active:
            row_pixels = []
            write_idx = 0
            cached_sets = _object_pixel_sets(tia_for_render)
            # P3i-f: track the HMOVE-blank window. When
            # `hmove_blank_pending` is true at scanline start, the
            # first 8 visible color clocks (c=68..75 → x=0..7) are
            # blanked to 0; collision detection still runs (sprites
            # are present at those pixels even though the chip
            # blanks the OUTPUT). After c >= 76 the flag clears.
            hmove_blank_active = tia_for_render.hmove_blank_pending
            for c in range(HBLANK_COLOR_CLOCKS, COLOR_CLOCKS_PER_SCANLINE):
                # Apply any pending writes that activate at or before c.
                # `tia.color_clock` was the WRITE position; activation_clock
                # is `>= write_position + delay >= HBLANK_COLOR_CLOCKS + 2`
                # so all pending entries land in the visible region.
                while (write_idx < len(pending_sorted)
                       and pending_sorted[write_idx][0] <= c):
                    _, reg, val = pending_sorted[write_idx]
                    # Task #91 round 2: store + GRP shadow latch at the
                    # activation clock (xitari poke ordering).
                    tia_for_render = _apply_pending_write(tia_for_render, reg, val)
                    cached_sets = _object_pixel_sets(tia_for_render)
                    write_idx += 1
                x = c - HBLANK_COLOR_CLOCKS
                # P3i-f: HMOVE-blank window blanks the leftmost 8
                # visible pixels (c in [68..75]). After that, the
                # flag clears for the rest of the scanline.
                if hmove_blank_active and c >= HBLANK_COLOR_CLOCKS + 8:
                    hmove_blank_active = False
                if hmove_blank_active:
                    row_pixels.append(0)
                else:
                    row_pixels.append(int(render_pixel(tia_for_render, c, cached_sets)))
                # P3i-d: OR in per-pixel collision bits with the
                # cached_sets that reflect post-write state at this c.
                # NOTE: collision still accumulates during the blank
                # window (matches real hardware — blank affects output
                # only, not the collision detection circuitry).
                _apply_pixel_collisions(running_collisions, x, cached_sets)
            # Drain any pending writes that didn't activate in time
            # (shouldn't happen since activation < 228 by construction,
            # but defensive). Apply to the registers.
            while write_idx < len(pending_sorted):
                _, reg, val = pending_sorted[write_idx]
                tia_for_render = _apply_pending_write(tia_for_render, reg, val)
                write_idx += 1

            row = jnp.array(row_pixels, dtype=jnp.uint8)
            # Task #83 round 3 (2026-06-11): only WRITE framebuffer for
            # scanlines at or past Y_START. xitari achieves this
            # implicitly via `myClockStartDisplay = myClockWhenFrameStarted
            # + 228*myYStart`. Pre-Y_START "rendered" pixels go nowhere
            # in xitari AND the HMOVE-blank flag isn't consumed there;
            # this gate matches that behavior so pong's row-0 HMOVE comb
            # lands at display row 0 (= internal scanline 34) instead of
            # the cropped-out internal scanline 27. Per-pixel rendering +
            # collision detection above still runs at every scanline so
            # unit tests that check collisions at scanline 0 keep working.
            if tia.scanline >= Y_START:
                for i in range(line_advance):
                    completed_line = (tia.scanline + i) % NTSC_SCANLINES_PER_FRAME
                    if completed_line < SCREEN_HEIGHT:
                        fb = fb.at[completed_line].set(row)
                wrote_framebuffer_this_advance = True
        else:
            # VBLANK-blanked render — output suppressed. Drain pending
            # writes into the register file. Do NOT run collision
            # detection: xitari's `TIA::updateFrameScanline`
            # (TIA.cxx:1121) memsets the framebuffer and RETURNS when
            # `myVBLANK & 0x02` is set, entirely skipping the per-pixel
            # collision switch. Real hardware DOES keep the collision
            # pipeline live in VBLANK, but xitari does not — and our
            # reference is xitari. (Discovered via per-bus-op trace
            # diff on breakout, commit d66b290 / a8cdcc9, 2026-06-03.)
            for _, reg, val in pending_sorted:
                tia_for_render = _apply_pending_write(tia_for_render, reg, val)
        new_collisions = jnp.array(running_collisions, dtype=jnp.uint8)

    new_line = tia.scanline + line_advance
    # PXC1-x: don't increment the frame counter on scanline-wrap. The
    # frame counter is driven *only* by the software VSYNC 1→0 edge (see
    # `tia_poke` for `W_VSYNC`). The previous "safety fallback" of also
    # bumping `frame` here caused a double-count for any ROM that drove
    # VSYNC normally: the wrap fired one or two scanlines BEFORE the
    # VSYNC handler did, and both incremented `frame` for the same
    # frame boundary — which is why `run_until_frame` was completing
    # every other "frame" in just ~80 CPU cycles (one scanline) instead
    # of the natural ~19,912. The trade-off: a ROM that goes long
    # stretches without toggling VSYNC at all (e.g. Q*Bert's boot
    # sequence) needs `run_until_frame`'s instruction limit to be
    # generous enough — see `console._FRAME_INSTRUCTION_LIMIT`.
    new_line = new_line % NTSC_SCANLINES_PER_FRAME
    # Task #106 (partial-frame model): the frame boundary is now decided
    # ONLY in `tia_poke` (the VSYNC-clear hold-gate OR the max-scanlines
    # cutoff) — exactly like xitari, where both live in TIA::poke. The
    # every-CPU-step `lines_since_frame > 290` cutoff that used to live here
    # (task #80) has been REMOVED: it force-ended qbert's poke-less
    # RESET-boot wait loop at ~291 scanlines, slicing it differently from
    # xitari's 25000-instruction grey-frame budget and producing a
    # persistent +1 frame offset. A truly poke-less runaway frame is now
    # bounded by that budget in `run_until_frame` (a grey frame), as in
    # xitari. Mirror of jutari's `tia_advance!`.
    new_lines_since_frame = tia.lines_since_frame + line_advance
    new_frame = tia.frame
    new_vsync_finish_clock = tia.vsync_finish_clock
    # P3i-c: thread the post-drain register file through to the new
    # state, and clear pending_writes (they've all been applied above
    # OR carried forward in the registers field). `tia_for_render`
    # always equals `tia` if line_advance was 0 (no scanline
    # crossed → no drain), so this is a no-op in that case.
    final_registers = tia_for_render.registers if line_advance > 0 else tia.registers
    # P3i-f: clear hmove_blank_pending after the scanline render — it
    # only blanks the leftmost 8 pixels of the IMMEDIATELY-following
    # scanline. If line_advance was 0 (no scanline crossed), keep the
    # pending flag so the next tia_advance call still applies it.
    # Task #83 round 3 (2026-06-11): also keep the flag if no
    # framebuffer write happened (scanline < Y_START or vblank_active).
    # xitari only clears `myHMOVEBlankEnabled` from inside the
    # visible-region framebuffer-writing branch (TIA.cxx:1776-1786).
    # The `wrote_framebuffer_this_advance` flag is False when:
    #   - line_advance == 0 (no scanline crossed)
    #   - tia.vblank_active (VBLANK-blanked render)
    #   - tia.scanline < Y_START (pre-display-window)
    if line_advance > 0 and wrote_framebuffer_this_advance:
        new_hmove_blank = False
    else:
        new_hmove_blank = tia.hmove_blank_pending
    # Task #97: once the beam crosses a scanline, promote a parked
    # next-line comb (from a beam_sc>=76 HMOVE) into the current flag
    # (OR-semantics, never lose a carried-forward flag), and consume it.
    # If no line crossed, keep it parked for a later advance. Mirror of
    # jutari's promotion in `tia_advance!`.
    if line_advance > 0 and tia.hmove_blank_pending_next:
        new_hmove_blank = True
    new_hmove_blank_next = (
        False if line_advance > 0 else tia.hmove_blank_pending_next
    )
    # Task #99: line N rendered + committed with PRE-motion object positions
    # (the beam_sc>=76 strobe deferred its motion); now apply the deferred
    # motion so line N+1 onward use the moved positions. (0,0,0,0,0) → no-op.
    # If no line crossed, keep it parked for a later advance. Mirror of jutari.
    if line_advance > 0:
        _dp0, _dp1, _dm0, _dm1, _dbl = tia.hmove_motion_next
        new_p0_x = (int(tia.p0_x) - _dp0) % 160
        new_p1_x = (int(tia.p1_x) - _dp1) % 160
        new_m0_x = (int(tia.m0_x) - _dm0) % 160
        new_m1_x = (int(tia.m1_x) - _dm1) % 160
        new_bl_x = (int(tia.bl_x) - _dbl) % 160
        new_hmove_motion_next = (0, 0, 0, 0, 0)
    else:
        new_p0_x, new_p1_x = tia.p0_x, tia.p1_x
        new_m0_x, new_m1_x, new_bl_x = tia.m0_x, tia.m1_x, tia.bl_x
        new_hmove_motion_next = tia.hmove_motion_next
    return tia._replace(
        scanline_cycle=new_sc,
        scanline=new_line,
        p0_x=new_p0_x,
        p1_x=new_p1_x,
        m0_x=new_m0_x,
        m1_x=new_m1_x,
        bl_x=new_bl_x,
        hmove_motion_next=new_hmove_motion_next,
        # Task #80: frame counter is normally bumped only on the VSYNC
        # 1→0 edge (in `tia_poke`); the max-scanlines cutoff above is the
        # second path and writes `new_frame` here. `lines_since_frame`
        # carries the running line count across advances.
        frame=new_frame,
        lines_since_frame=new_lines_since_frame,
        vsync_finish_clock=new_vsync_finish_clock,
        color_clock=new_color_clock,
        framebuffer=fb,
        collisions=new_collisions,
        registers=final_registers,
        pending_writes=() if line_advance > 0 else tia.pending_writes,
        hmove_blank_pending=new_hmove_blank,
        hmove_blank_pending_next=new_hmove_blank_next,
        # Task #91 round 2 (2026-06-12): the drain now latches the
        # VDELP / VDELBL shadows at activation time, so the post-drain
        # shadow values must flow into the new state alongside the
        # register file (they live on `tia_for_render`, which equals
        # `tia` when line_advance == 0).
        grp0_old=tia_for_render.grp0_old if line_advance > 0 else tia.grp0_old,
        grp1_old=tia_for_render.grp1_old if line_advance > 0 else tia.grp1_old,
        enabl_old=tia_for_render.enabl_old if line_advance > 0 else tia.enabl_old,
        # P4c (dump-pot): keep a monotonic CPU-cycle counter so
        # `tia_peek` can decide whether the paddle cap has charged
        # past its xitari-formula threshold yet.
        total_cycles=int(tia.total_cycles) + int(cpu_cycles),
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
    """Internal helper: 160-element Python list of playfield colours.

    Honours CTRLPF.D1 (SCORE mode): when set, the playfield-ON pixels in
    the LEFT half are coloured with COLUP0 and in the RIGHT half with
    COLUP1, instead of COLUPF. Games (Pong, Combat, …) use this to put
    each player's score in their own player colour without burning a
    sprite on it. xitari does the same selection in `TIA::updateFrame`.
    """
    pf0    = int(tia.registers[W_PF0])
    pf1    = int(tia.registers[W_PF1])
    pf2    = int(tia.registers[W_PF2])
    ctrlpf = int(tia.registers[W_CTRLPF])
    colupf = int(tia.registers[W_COLUPF])
    colubk = int(tia.registers[W_COLUBK])
    reflected = (ctrlpf & 0x01) != 0
    score_mode = (ctrlpf & 0x02) != 0
    pf_left  = int(tia.registers[W_COLUP0]) if score_mode else colupf
    pf_right = int(tia.registers[W_COLUP1]) if score_mode else colupf

    left_bits = _playfield_bits(pf0, pf1, pf2)
    right_bits = list(reversed(left_bits)) if reflected else left_bits

    pixels: list[int] = []
    for bit in left_bits:
        pixels.extend([pf_left if bit else colubk] * 4)
    for bit in right_bits:
        pixels.extend([pf_right if bit else colubk] * 4)
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


def _vdel_grp(tia: TIAState, player: int) -> int:
    """P3h: return the player's current GRP byte, honouring VDELP.
    When VDELPx bit 0 is set, render uses the shadow GRPx_old (which
    `tia_poke` latched on the previous GRP-of-the-other-player write)
    instead of the live GRPx.
    """
    if player == 0:
        return int(tia.grp0_old) if (int(tia.registers[W_VDELP0]) & 1) \
            else int(tia.registers[W_GRP0])
    return int(tia.grp1_old) if (int(tia.registers[W_VDELP1]) & 1) \
        else int(tia.registers[W_GRP1])


def _vdel_enabl(tia: TIAState) -> int:
    """P3h: return the ball's current ENABL byte, honouring VDELBL."""
    return int(tia.enabl_old) if (int(tia.registers[W_VDELBL]) & 1) \
        else int(tia.registers[W_ENABL])


def _overlay_player(pixels: list[int], tia: TIAState, player: int) -> None:
    """Paint player 0 or 1 onto `pixels` (mutated in place).

    P3g: respects NUSIZ low 3 bits — multi-copy (2 or 3 copies with
    close/medium/wide spacing) and 2×/4× horizontal scaling for the
    single-copy modes. The same NUSIZ value also drives the matching
    missile via `_overlay_missile` and the collision-set computation
    in `_object_pixel_sets`. **P3h**: GRP comes from `_vdel_grp` which
    routes through the VDELP shadow when the delayed-update bit is set.
    """
    grp = _vdel_grp(tia, player)
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
    # painted at `x + copy_offset`. NUSIZ wide modes (scale > 1, i.e.
    # double-size NUSIZ=5 and quad-size NUSIZ=7) get a +1 pixel offset:
    # xitari (`computePlayerMaskTable`) bakes this into its alignment-0
    # mask with the comment "in double/quad size mode the player's
    # output is delayed by one pixel" — a real NMOS-TIA quirk.
    nusiz_offset = 1 if scale > 1 else 0
    for copy_off in copy_offsets:
        base = (x + copy_off + nusiz_offset) % 160
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
    # Task #85 (2026-06-11): RESMP*=1 → missile invisible. See
    # `_missile_set` for the full rationale; this is the symmetric
    # gate on the (unused) legacy whole-scanline render path.
    resmp_reg = W_RESMP0 if missile == 0 else W_RESMP1
    if (int(tia.registers[resmp_reg]) & 0x02) != 0:
        return
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
    """Paint the ball. Uses COLUPF for colour; size 1/2/4/8 from CTRLPF bits 4-5.
    **P3h**: ENABL routed through `_vdel_enabl` to honour VDELBL.
    """
    if (_vdel_enabl(tia) & 0x02) == 0:
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
    score_mode = (ctrlpf & 0x02) != 0
    pf_left  = int(tia.registers[W_COLUP0]) if score_mode else colupf
    pf_right = int(tia.registers[W_COLUP1]) if score_mode else colupf
    left_bits = _playfield_bits(pf0, pf1, pf2)
    right_bits = list(reversed(left_bits)) if reflected else left_bits
    for i, bit in enumerate(left_bits):
        if bit:
            for k in range(4):
                pixels[i * 4 + k] = pf_left
    for i, bit in enumerate(right_bits):
        if bit:
            for k in range(4):
                pixels[80 + i * 4 + k] = pf_right


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


def render_pixel(tia: TIAState, color_clock: int, cached_sets=None) -> int:
    """P3i-a: per-color-clock pixel kernel.

    Returns the pixel value (palette index, uint8) at beam position
    `color_clock` (0..227) within the current scanline. HBLANK positions
    (0..67) are not visible — returns 0. Visible positions (68..227) map
    to framebuffer X = `color_clock - HBLANK_COLOR_CLOCKS` (0..159).

    Compositing matches `render_scanline` exactly — same two priority
    modes (PFP=0 default, PFP=1 swaps PF/BL above sprites). The function
    is *pure* in `tia` (no mutations) and *bit-exact-equivalent* to
    `render_scanline(tia)[color_clock - HBLANK_COLOR_CLOCKS]` for every
    visible color clock. Sub-phase P3i-c will make the kernel sensitive
    to *mid-scanline* register writes; P3i-a only proves the API is
    equivalent.

    `cached_sets` is an optional precomputed `_object_pixel_sets(tia)`
    return value. Passing it lets `tia_advance` compute the per-object
    coverage once per scanline and reuse it across all 160 visible color
    clocks (P3i-b's per-color-clock loop), keeping per-scanline cost the
    same as before.
    """
    if color_clock < HBLANK_COLOR_CLOCKS:
        return 0                                          # HBLANK — invisible
    x = color_clock - HBLANK_COLOR_CLOCKS                 # 0..159
    if x >= SCREEN_WIDTH:
        return 0                                          # past visible region

    colubk = int(tia.registers[W_COLUBK])
    colupf = int(tia.registers[W_COLUPF])
    colup0 = int(tia.registers[W_COLUP0])
    colup1 = int(tia.registers[W_COLUP1])
    ctrlpf = int(tia.registers[W_CTRLPF])
    pfp    = (ctrlpf & 0x04) != 0                         # P3l priority bit
    # CTRLPF.D1 SCOREMODE: playfield LEFT half uses COLUP0, RIGHT half
    # uses COLUP1 (instead of COLUPF). Ball stays on COLUPF.
    score  = (ctrlpf & 0x02) != 0
    pf_col = (colup0 if x < 80 else colup1) if score else colupf

    sets = cached_sets if cached_sets is not None else _object_pixel_sets(tia)

    pixel = colubk
    if not pfp:
        # Default priority: bg ← pf ← bl ← M1 ← P1 ← M0 ← P0
        if x in sets["pf"]: pixel = pf_col
        if x in sets["bl"]: pixel = colupf
        if x in sets["m1"]: pixel = colup1
        if x in sets["p1"]: pixel = colup1
        if x in sets["m0"]: pixel = colup0
        if x in sets["p0"]: pixel = colup0
    else:
        # PFP priority: bg ← M1 ← P1 ← M0 ← P0 ← pf ← bl
        if x in sets["m1"]: pixel = colup1
        if x in sets["p1"]: pixel = colup1
        if x in sets["m0"]: pixel = colup0
        if x in sets["p0"]: pixel = colup0
        if x in sets["pf"]: pixel = pf_col
        if x in sets["bl"]: pixel = colupf
    return pixel


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
    if _vdel_enabl(tia) & 0x02:                  # P3h: VDELBL-aware
        size = 1 << ((ctrlpf >> 4) & 0x03)
        for i in range(size):
            bl.add((tia.bl_x + i) % 160)

    # Players — respect NUSIZ multi-copy + scale (P3g) so collisions
    # involving the 2nd/3rd copies and the wide modes are detected.
    # P3h: GRP comes via _vdel_grp so VDELP delayed updates also
    # apply to collision detection.
    def _player_set(grp_reg: int, refp_reg: int, nusiz_reg: int, x: int) -> set[int]:
        # `grp_reg` selects the player index (0 or 1) implicitly: W_GRP0 → 0.
        player = 0 if grp_reg == W_GRP0 else 1
        grp = _vdel_grp(tia, player)
        if not grp:
            return set()
        reflected = (int(tia.registers[refp_reg]) & 0x08) != 0
        copy_offsets, scale = _nusiz_player_layout(int(tia.registers[nusiz_reg]))
        # +1 pixel offset for the wide modes (scale > 1) — matches
        # xitari's quad/double-size mask quirk; see `_overlay_player`.
        nusiz_offset = 1 if scale > 1 else 0
        out: set[int] = set()
        for copy_off in copy_offsets:
            base = (x + copy_off + nusiz_offset) % 160
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
    def _missile_set(enam_reg: int, nusiz_reg: int, resmp_reg: int,
                     x: int) -> set[int]:
        if (int(tia.registers[enam_reg]) & 0x02) == 0:
            return set()
        # Task #85 (2026-06-11): RESMP*=1 → missile invisible (locked to
        # player center, no pixels rendered). Without this gate, pong's
        # "color-stripe-below-score" path — which keeps ENAM* enabled
        # but uses RESMP*=$02 to hide the missiles — paints 16 px
        # phantoms (rows 35-37, cols 16-19 + 140-143). Mirrors xitari
        # `case 0x1D/0x1E` which gates `myEnabledObjects` on
        # `ENAM* && !RESMP*` (TIA.cxx:2525, 2536), and the symmetric
        # jutari `_missile_set` fix in `jutari/src/tia/TIA.jl`.
        if (int(tia.registers[resmp_reg]) & 0x02) != 0:
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

    m0 = _missile_set(W_ENAM0, W_NUSIZ0, W_RESMP0, tia.m0_x)
    m1 = _missile_set(W_ENAM1, W_NUSIZ1, W_RESMP1, tia.m1_x)

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

    P3i-d note: this kept-around helper is still used by code paths
    that do per-scanline collision detection in one shot (the
    `_detect_collisions` semantic of "scan whole scanline, OR all
    collisions in"). `tia_advance` now uses the per-pixel cousin
    `_apply_pixel_collisions` so mid-scanline register changes affect
    the collision evaluation correctly.
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


def _apply_pixel_collisions(coll: list[int], x: int, sets) -> None:
    """P3i-d: OR per-pixel collision bits into `coll` (mutated in
    place) for visible pixel `x`. Mirrors `_detect_collisions`'s
    bit-layout exactly; the only difference is the granularity (one
    pixel instead of a whole-scanline OR), which lets mid-scanline
    register changes (PF stomping via P3i-c, future per-pixel
    sprite-position adjustments) affect collision evaluation at the
    exact color clock they apply. Called inside `tia_advance`'s
    per-color-clock loop after pending writes have been drained.
    """
    p0_here = x in sets["p0"]
    p1_here = x in sets["p1"]
    m0_here = x in sets["m0"]
    m1_here = x in sets["m1"]
    bl_here = x in sets["bl"]
    pf_here = x in sets["pf"]

    if m0_here and p1_here: coll[0] |= 0x80          # CXM0P  D7 = M0-P1
    if m0_here and p0_here: coll[0] |= 0x40          # CXM0P  D6 = M0-P0
    if m1_here and p0_here: coll[1] |= 0x80          # CXM1P  D7 = M1-P0
    if m1_here and p1_here: coll[1] |= 0x40          # CXM1P  D6 = M1-P1
    if p0_here and pf_here: coll[2] |= 0x80          # CXP0FB D7 = P0-PF
    if p0_here and bl_here: coll[2] |= 0x40          # CXP0FB D6 = P0-BL
    if p1_here and pf_here: coll[3] |= 0x80          # CXP1FB D7 = P1-PF
    if p1_here and bl_here: coll[3] |= 0x40          # CXP1FB D6 = P1-BL
    if m0_here and pf_here: coll[4] |= 0x80          # CXM0FB D7 = M0-PF
    if m0_here and bl_here: coll[4] |= 0x40          # CXM0FB D6 = M0-BL
    if m1_here and pf_here: coll[5] |= 0x80          # CXM1FB D7 = M1-PF
    if m1_here and bl_here: coll[5] |= 0x40          # CXM1FB D6 = M1-BL
    if bl_here and pf_here: coll[6] |= 0x80          # CXBLPF D7 = BL-PF
    if p0_here and p1_here: coll[7] |= 0x80          # CXPPMM D7 = P0-P1
    if m0_here and m1_here: coll[7] |= 0x40          # CXPPMM D6 = M0-M1


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
