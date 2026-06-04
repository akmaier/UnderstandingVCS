"""
    TIA

Television Interface Adapter — Atari 2600 video/audio chip.

The CPU talks to the TIA through 64 memory-mapped registers in the
\$0000–\$003F window (with mirrors), some write-only, a subset readable.

Per-frame timing (NTSC):

    228 color clocks    = 1 scanline (= 76 CPU cycles)
    262 scanlines       = 1 frame    (≈ 19,912 CPU cycles)

WSYNC (write \$02) stalls the CPU until the next scanline boundary.

Phase progress:
  - P3a: register file + scanline/frame timing + WSYNC.
  - P3b: playfield rendering (PF0/PF1/PF2 + CTRLPF mirror).
  - P3c: player sprites P0/P1 (GRP, REFP, RESP, HMP, HMOVE, HMCLR).
  - P3d: missiles M0/M1 and ball BL (sizes, EN*, RES*, HM*).
  - P3e: collision latches + CXCLR.
  - P3f: software-driven frame ending via VSYNC and output blanking
    via VBLANK. The 1→0 edge on VSYNC.D1 is the frame boundary
    (frame++, scanline=0). When VBLANK.D1 is set, completed scanlines
    are NOT written to the framebuffer (blanked).

P3 is now feature-complete for the documented register set, modulo:
  - NUSIZ multi-copy / 2×/4×-wide player scaling.
  - VDELP* / VDELBL vertical-delay updates.
  - Sub-pixel beam-accurate rendering.
  - Audio (TIASnd).
  - INPT* input reads.
"""
module TIA

export TIAState, initial_tia_state,
       tia_peek, tia_poke!, tia_advance!, tia_apply_wsync!,
       playfield_bits, render_playfield_scanline, render_scanline, render_pixel,
       set_trigger!, set_paddle_resistance!,
       _hm_offset, _resp_position,
       NTSC_CPU_CYCLES_PER_SCANLINE, NTSC_SCANLINES_PER_FRAME,
       NUM_REGISTERS, SCREEN_WIDTH, SCREEN_HEIGHT,
       Y_START, VISIBLE_HEIGHT,
       COLOR_CLOCKS_PER_CPU_CYCLE, COLOR_CLOCKS_PER_SCANLINE, HBLANK_COLOR_CLOCKS,
       W_VSYNC, W_VBLANK, W_WSYNC, W_RSYNC, W_NUSIZ0, W_NUSIZ1,
       W_COLUP0, W_COLUP1, W_COLUPF, W_COLUBK, W_CTRLPF, W_REFP0, W_REFP1,
       W_PF0, W_PF1, W_PF2, W_RESP0, W_RESP1, W_RESM0, W_RESM1, W_RESBL,
       W_AUDC0, W_AUDC1, W_AUDF0, W_AUDF1, W_AUDV0, W_AUDV1,
       W_GRP0, W_GRP1, W_ENAM0, W_ENAM1, W_ENABL,
       W_HMP0, W_HMP1, W_HMM0, W_HMM1, W_HMBL,
       W_VDELP0, W_VDELP1, W_VDELBL, W_RESMP0, W_RESMP1,
       W_HMOVE, W_HMCLR, W_CXCLR

# Write registers ($00..$2C)
const W_VSYNC  = 0x00; const W_VBLANK = 0x01
const W_WSYNC  = 0x02; const W_RSYNC  = 0x03
const W_NUSIZ0 = 0x04; const W_NUSIZ1 = 0x05
const W_COLUP0 = 0x06; const W_COLUP1 = 0x07
const W_COLUPF = 0x08; const W_COLUBK = 0x09
const W_CTRLPF = 0x0A
const W_REFP0  = 0x0B; const W_REFP1  = 0x0C
const W_PF0    = 0x0D; const W_PF1    = 0x0E; const W_PF2    = 0x0F
const W_RESP0  = 0x10; const W_RESP1  = 0x11
const W_RESM0  = 0x12; const W_RESM1  = 0x13; const W_RESBL  = 0x14
const W_AUDC0  = 0x15; const W_AUDC1  = 0x16
const W_AUDF0  = 0x17; const W_AUDF1  = 0x18
const W_AUDV0  = 0x19; const W_AUDV1  = 0x1A
const W_GRP0   = 0x1B; const W_GRP1   = 0x1C
const W_ENAM0  = 0x1D; const W_ENAM1  = 0x1E; const W_ENABL  = 0x1F
const W_HMP0   = 0x20; const W_HMP1   = 0x21
const W_HMM0   = 0x22; const W_HMM1   = 0x23; const W_HMBL   = 0x24
const W_VDELP0 = 0x25; const W_VDELP1 = 0x26; const W_VDELBL = 0x27
const W_RESMP0 = 0x28; const W_RESMP1 = 0x29
const W_HMOVE  = 0x2A; const W_HMCLR  = 0x2B
const W_CXCLR  = 0x2C

# Constants
const NUM_REGISTERS = 64
const NTSC_CPU_CYCLES_PER_SCANLINE = 76
const NTSC_SCANLINES_PER_FRAME = 262
const SCREEN_WIDTH = 160
# Internal framebuffer height — covers scanlines 0..243 (everything
# xitari would ever show with its default `Display.Height=210` starting
# at `Display.YStart=34`). The unused 0..33 + 244..261 regions stay
# empty (VSYNC + post-overscan) but keeping the buffer indexed by
# absolute scanline number preserves the unit-tests' `framebuffer[N+1,:]
# == scanline-N render` invariant.
const SCREEN_HEIGHT = 244
# ALE / xitari visible-region crop. Matches xitari's `Display.YStart=34` +
# `Display.Height=210` (see xitari/emucore/Props.cxx:300-301).
# `StellaEnvironment.get_screen()` returns `framebuffer[Y_START+1 :
# Y_START + VISIBLE_HEIGHT, :]` so the user-facing screen lines up
# vertically with xitari/ALE — top 34 lines (VSYNC + VBLANK + any
# score-header area outside xitari's display window) cropped out.
const Y_START = 34
const VISIBLE_HEIGHT = 210
# P3i-a: color-clock scaffolding. The TIA actually operates at 3× the
# CPU rate — one CPU cycle = 3 color clocks. A scanline is 228 color
# clocks: 68 of HBLANK (chip blanks output while the CRT beam
# retraces) followed by 160 visible pixels. `color_clock` on TIAState
# tracks the beam position 0..227 within the scanline; `render_pixel`
# is the per-color-clock kernel that P3i-c/d/e/f progressively replace
# `render_scanline` with.
const COLOR_CLOCKS_PER_CPU_CYCLE = 3
const HBLANK_COLOR_CLOCKS = 68
const COLOR_CLOCKS_PER_SCANLINE = 228

# P3i-c: per-poke write delays, in color clocks. Verbatim port of
# xitari `TIA::ourPokeDelayTable[64]` (xitari/emucore/TIA.cxx).
# Index is `addr & 0x3F`. A non-zero entry means the write activates
# that many color clocks LATER than the write itself. -1 means
# "compute dynamically based on scanline phase" (only PF0/PF1/PF2
# use this — see `_pf_dynamic_delay`). Matches jaxtari `_POKE_DELAY_TABLE`.
const _POKE_DELAY_TABLE = Int16[
    0,  1,  0,  0,  8,  8,  0,  0,  0,  0,  0,  1,  1, -1, -1, -1,
    0,  0,  8,  8,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
]
const _PF_DYNAMIC_DELAY = (4, 5, 2, 3)

@inline function _pf_dynamic_delay(color_clock::Integer)
    return _PF_DYNAMIC_DELAY[((Int(color_clock) ÷ 3) & 3) + 1]   # 1-indexed
end

@inline function _poke_activation_delay(reg::Integer, color_clock::Integer)
    r = Int(reg)
    (r < 0 || r >= 64) && return 0
    d = _POKE_DELAY_TABLE[r + 1]
    return d == -1 ? _pf_dynamic_delay(color_clock) : Int(d)
end

# P3i-f: HMOVE blank enable table, indexed by CPU cycle within the
# scanline (0..75). Verbatim port of xitari's
# `ourHMOVEBlankEnableCycles[128]` (indices 0..75 used). A True at
# cycle x means "writing HMOVE at this cycle triggers the blank
# bug." Mirrors jaxtari `_HMOVE_BLANK_ENABLE_CYCLES`.
const _HMOVE_BLANK_ENABLE_CYCLES = Bool[
    true,  true,  true,  true,  true,  true,  true,  true,  true,  true,   # 00-09
    true,  true,  true,  true,  true,  true,  true,  true,  true,  true,   # 10-19
    true,  false, false, false, false, false, false, false, false, false,  # 20-29
    false, false, false, false, false, false, false, false, false, false,  # 30-39
    false, false, false, false, false, false, false, false, false, false,  # 40-49
    false, false, false, false, false, false, false, false, false, false,  # 50-59
    false, false, false, false, false, false, false, false, false, false,  # 60-69
    false, false, false, false, false, true,                               # 70-75
]

@inline function _hmove_blank_enabled_at(scanline_cycle::Integer)
    sc = Int(scanline_cycle) & 0x7F                  # mod 128
    sc >= NTSC_CPU_CYCLES_PER_SCANLINE && return false
    return _HMOVE_BLANK_ENABLE_CYCLES[sc + 1]        # Julia 1-indexed
end

# P3i-g: full HMOVE motion table, indexed by [scanline_cycle][hm_nibble].
# Verbatim port of xitari's `ourCompleteMotionTable[128][16]` for cycles
# 0..75. Our sign convention is the NEGATION of xitari's (we subtract,
# xitari adds). Three structural regions: HBLANK (0..20, standard
# offsets), mid-scanline (21..54, all zeros — HMOVE has no effect),
# late-scanline (55..75, partial-motion deltas). Mirrors jaxtari
# `_COMPLETE_MOTION_TABLE`.
const _COMPLETE_MOTION_TABLE = let
    rows = Vector{NTuple{16,Int}}(undef, 76)
    # HBLANK pattern (cycles 0..20). The pattern subtly varies row-by-row
    # (xitari rows show progressive clipping in entries 4..7 and 13..15
    # as the scanline progresses). Spelled out below — see xitari
    # `ourCompleteMotionTable` rows 0..20.
    rows[1]  = ( 0,  1,  2,  3,  4,  5,  6,  7, -8, -7, -6, -5, -4, -3, -2, -1)  # 0
    rows[2]  = ( 0,  1,  2,  3,  4,  5,  6,  7, -8, -7, -6, -5, -4, -3, -2, -1)  # 1
    rows[3]  = ( 0,  1,  2,  3,  4,  5,  6,  7, -8, -7, -6, -5, -4, -3, -2, -1)  # 2
    rows[4]  = ( 0,  1,  2,  3,  4,  5,  6,  7, -8, -7, -6, -5, -4, -3, -2, -1)  # 3
    rows[5]  = ( 0,  1,  2,  3,  4,  5,  6,  6, -8, -7, -6, -5, -4, -3, -2, -1)  # 4
    rows[6]  = ( 0,  1,  2,  3,  4,  5,  5,  5, -8, -7, -6, -5, -4, -3, -2, -1)  # 5
    rows[7]  = ( 0,  1,  2,  3,  4,  5,  5,  5, -8, -7, -6, -5, -4, -3, -2, -1)  # 6
    rows[8]  = ( 0,  1,  2,  3,  4,  4,  4,  4, -8, -7, -6, -5, -4, -3, -2, -1)  # 7
    rows[9]  = ( 0,  1,  2,  3,  3,  3,  3,  3, -8, -7, -6, -5, -4, -3, -2, -1)  # 8
    rows[10] = ( 0,  1,  2,  2,  2,  2,  2,  2, -8, -7, -6, -5, -4, -3, -2, -1)  # 9
    rows[11] = ( 0,  1,  2,  2,  2,  2,  2,  2, -8, -7, -6, -5, -4, -3, -2, -1)  # 10
    rows[12] = ( 0,  1,  1,  1,  1,  1,  1,  1, -8, -7, -6, -5, -4, -3, -2, -1)  # 11
    rows[13] = ( 0,  0,  0,  0,  0,  0,  0,  0, -8, -7, -6, -5, -4, -3, -2, -1)  # 12
    rows[14] = (-1, -1, -1, -1, -1, -1, -1, -1, -8, -7, -6, -5, -4, -3, -2, -1)  # 13
    rows[15] = (-1, -1, -1, -1, -1, -1, -1, -1, -8, -7, -6, -5, -4, -3, -2, -1)  # 14
    rows[16] = (-2, -2, -2, -2, -2, -2, -2, -2, -8, -7, -6, -5, -4, -3, -2, -2)  # 15
    rows[17] = (-3, -3, -3, -3, -3, -3, -3, -3, -8, -7, -6, -5, -4, -3, -3, -3)  # 16
    rows[18] = (-4, -4, -4, -4, -4, -4, -4, -4, -8, -7, -6, -5, -4, -4, -4, -4)  # 17
    rows[19] = (-4, -4, -4, -4, -4, -4, -4, -4, -8, -7, -6, -5, -4, -4, -4, -4)  # 18
    rows[20] = (-5, -5, -5, -5, -5, -5, -5, -5, -8, -7, -6, -5, -5, -5, -5, -5)  # 19
    rows[21] = (-6, -6, -6, -6, -6, -6, -6, -6, -8, -7, -6, -6, -6, -6, -6, -6)  # 20
    # Mid-scanline (cycles 21..54) — HMOVE has no effect.
    for i in 22:55
        rows[i] = ntuple(_ -> 0, 16)
    end
    # Late-scanline (cycles 55..75)
    rows[56] = ( 0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0)  # 55
    rows[57] = ( 0,  0,  0,  0,  0,  0,  1,  2,  0,  0,  0,  0,  0,  0,  0,  0)  # 56
    rows[58] = ( 0,  0,  0,  0,  0,  1,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0)  # 57
    rows[59] = ( 0,  0,  0,  0,  0,  1,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0)  # 58
    rows[60] = ( 0,  0,  0,  0,  1,  2,  3,  4,  0,  0,  0,  0,  0,  0,  0,  0)  # 59
    rows[61] = ( 0,  0,  0,  1,  2,  3,  4,  5,  0,  0,  0,  0,  0,  0,  0,  0)  # 60
    rows[62] = ( 0,  0,  1,  2,  3,  4,  5,  6,  0,  0,  0,  0,  0,  0,  0,  0)  # 61
    rows[63] = ( 0,  0,  1,  2,  3,  4,  5,  6,  0,  0,  0,  0,  0,  0,  0,  0)  # 62
    rows[64] = ( 0,  1,  2,  3,  4,  5,  6,  7,  0,  0,  0,  0,  0,  0,  0,  0)  # 63
    rows[65] = ( 1,  2,  3,  4,  5,  6,  7,  8,  0,  0,  0,  0,  0,  0,  0,  0)  # 64
    rows[66] = ( 2,  3,  4,  5,  6,  7,  8,  9,  0,  0,  0,  0,  0,  0,  0,  1)  # 65
    rows[67] = ( 2,  3,  4,  5,  6,  7,  8,  9,  0,  0,  0,  0,  0,  0,  0,  1)  # 66
    rows[68] = ( 3,  4,  5,  6,  7,  8,  9, 10,  0,  0,  0,  0,  0,  0,  1,  2)  # 67
    rows[69] = ( 4,  5,  6,  7,  8,  9, 10, 11,  0,  0,  0,  0,  0,  1,  2,  3)  # 68
    rows[70] = ( 5,  6,  7,  8,  9, 10, 11, 12,  0,  0,  0,  0,  1,  2,  3,  4)  # 69
    rows[71] = ( 5,  6,  7,  8,  9, 10, 11, 12,  0,  0,  0,  0,  1,  2,  3,  4)  # 70
    rows[72] = ( 6,  7,  8,  9, 10, 11, 12, 13,  0,  0,  0,  1,  2,  3,  4,  5)  # 71
    rows[73] = ( 7,  8,  9, 10, 11, 12, 13, 14,  0,  0,  1,  2,  3,  4,  5,  6)  # 72
    rows[74] = ( 8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7)  # 73
    rows[75] = ( 8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7)  # 74
    rows[76] = ( 0,  1,  2,  3,  4,  5,  6,  7, -8, -7, -6, -5, -4, -3, -2, -1)  # 75 (HBLANK wrap)
    rows
end

@inline function _hmove_motion(scanline_cycle::Integer, hm::Integer)
    sc = Int(scanline_cycle) & 0x7F
    sc >= NTSC_CPU_CYCLES_PER_SCANLINE && (sc = 0)   # past end → HBLANK
    return _COMPLETE_MOTION_TABLE[sc + 1][((Int(hm) >> 4) & 0x0F) + 1]
end

"""
    TIAState

Mutable TIA state. `registers` is the 64-byte register file; `p0_x`,
`p1_x` are the derived player horizontal positions in screen pixels
([0..159]) updated by RESP* / HMOVE writes; `framebuffer` is the
rendered output.
"""
mutable struct TIAState
    registers::Vector{UInt8}
    scanline_cycle::Int
    scanline::Int
    frame::UInt64
    wsync_pending::Bool
    framebuffer::Matrix{UInt8}
    p0_x::Int
    p1_x::Int
    m0_x::Int
    m1_x::Int
    bl_x::Int
    collisions::Vector{UInt8}      # 8 collision-latch bytes ($30..$37)
    vsync_active::Bool
    vblank_active::Bool
    inpt::Vector{UInt8}            # 6 bytes — INPT0..INPT5
    # P3h: VDELP / VDELBL shadow copies. When VDELPx bit 0 is set,
    # the player renders from the *old* GRPx (captured the last time
    # the OTHER player's GRP was written). Same for the ball with
    # VDELBL + ENABL (the ENABL shadow latches on GRP1 writes,
    # matching xitari / Stella convention).
    grp0_old::UInt8
    grp1_old::UInt8
    enabl_old::UInt8
    # P4c (dump-pot) — paddle-pot capacitor charge model. VBLANK D7
    # grounds the cap (`dump_enabled`); on its 1→0 transition the
    # cap starts charging and `dump_disabled_cycle` records when.
    # `paddle_use_dump_pot[i]` is a per-paddle opt-in: False keeps
    # the simple "INPT returns inpt[i]" semantic the existing P4c
    # tests rely on; True activates the cycle-threshold path with
    # `paddle_resistance[i]` as the (xitari-scale) resistance.
    # `total_cycles` is a monotonic CPU cycle counter ticked by
    # `tia_advance!` for the threshold comparison.
    total_cycles::Int
    dump_enabled::Bool
    dump_disabled_cycle::Int
    paddle_use_dump_pot::Vector{Bool}      # length 4
    paddle_resistance::Vector{UInt32}      # length 4
    # P3i-a: current beam color-clock position within the scanline,
    # 0..227. Advances 3× per CPU cycle alongside `scanline_cycle`.
    # P3i-c will hang `ourPokeDelayTable` arithmetic off this field
    # so mid-scanline `tia_poke!` calls land at their real activation
    # color clock instead of taking effect immediately.
    color_clock::Int
    # P3i-c: queue of deferred PF0/PF1/PF2 writes that activate
    # mid-scanline at their `_POKE_DELAY_TABLE` color clock. Each entry
    # is `(activation_color_clock, reg, value)`. Drained scanline-by-
    # scanline by `tia_advance!`. Mirrors jaxtari `pending_writes`.
    pending_writes::Vector{Tuple{Int,Int,UInt8}}
    # P3i-f: HMOVE blank bug — when HMOVE is written at the right
    # cycle within the scanline, the chip black-bars the first 8
    # visible color clocks (the "HMOVE comb"). True from HMOVE write
    # until those 8 pixels finish rendering; auto-cleared by the
    # per-color-clock render loop. Mirrors jaxtari `hmove_blank_pending`.
    hmove_blank_pending::Bool
end

# INPT defaults: paddle pots ($80 = centred), triggers idle high (D7=1).
initial_tia_state() = TIAState(
    zeros(UInt8, NUM_REGISTERS),
    0, 0, UInt64(0), false,
    zeros(UInt8, SCREEN_HEIGHT, SCREEN_WIDTH),
    0, 0, 0, 0, 0,
    zeros(UInt8, 8),
    false, false,
    fill(UInt8(0x80), 6),
    UInt8(0), UInt8(0), UInt8(0),    # grp0_old / grp1_old / enabl_old
    # P4c dump-pot state: cycles=0, not enabled, no opt-ins.
    0, false, 0,
    fill(false, 4),
    zeros(UInt32, 4),
    0,                                # P3i-a: color_clock = 0
    Tuple{Int,Int,UInt8}[],            # P3i-c: empty pending_writes
    false,                             # P3i-f: hmove_blank_pending = false
)

"""
    tia_peek(tia, addr) -> UInt8

Read a TIA register. \$30..\$37 → collision latches; \$38..\$3B → INPT0..3
(paddle pots — default \$80 directly; with `paddle_use_dump_pot[i]` set,
xitari-style cycle-threshold dump-pot model applies); \$3C/\$3D → INPT4/5
(triggers idle high; `set_trigger!` flips them); \$3E/\$3F → unused.
"""
@inline function tia_peek(tia::TIAState, addr::Integer)
    reg = Int(addr) & 0x0F
    if reg < 8
        return tia.collisions[reg + 1]
    elseif reg < 12
        # INPT0..INPT3 — paddle pots with optional dump-pot timing.
        i = reg - 8 + 1                     # Julia 1-based
        if tia.paddle_use_dump_pot[i]
            tia.dump_enabled && return UInt8(0)
            r       = Int(tia.paddle_resistance[i])
            needed  = Int(round(1.6 * r * 0.01e-6 * 1.19e6))
            charged = tia.total_cycles > (tia.dump_disabled_cycle + needed)
            return charged ? UInt8(0x80) : UInt8(0)
        end
        return tia.inpt[i]
    elseif reg < 14
        return tia.inpt[reg - 8 + 1]
    end
    return UInt8(0)
end

"""
    set_trigger!(tia, player, pressed)

Set the fire-button line for player 0 or 1. Active-low: pressed → D7=0.
"""
@inline function set_trigger!(tia::TIAState, player::Integer, pressed::Bool)
    idx = player == 0 ? 5 : 6                   # INPT4 / INPT5 → indices 5 / 6
    tia.inpt[idx] = pressed ? UInt8(0x00) : UInt8(0x80)
    return nothing
end

"""
    set_paddle_resistance!(tia, paddle, resistance)

P4c (dump-pot) — opt the given paddle (0..3) into xitari's
cycle-threshold dump-pot model. `resistance` is in xitari's scale
(PADDLE_MIN = 27_450, PADDLE_MAX = 790_196). After this call INPT
reads on that paddle return D7 = 1 once `total_cycles >
dump_disabled_cycle + int(1.6·r·0.01e-6·1.19e6)`, mirroring the
jaxtari helper of the same name.
"""
@inline function set_paddle_resistance!(tia::TIAState, paddle::Integer, resistance::Integer)
    0 <= paddle <= 3 || error("paddle must be 0..3, got $paddle")
    i = paddle + 1
    tia.paddle_use_dump_pot[i] = true
    tia.paddle_resistance[i]   = UInt32(Int(resistance) & 0xFFFFFFFF)
    return nothing
end

"""
    _resp_position(scanline_cycle) -> Int

Legacy RESP* timing helper retained for backward compatibility with
pre-P3i-e callers. New code should use `_resp_player_position` /
`_resp_missile_ball_position` which apply xitari's exact constants.
"""
@inline function _resp_position(scanline_cycle::Integer)
    pos = Int(scanline_cycle) * 3 - 68
    pos < 0 && return 0
    pos > 159 && return 159
    return pos
end

# P3i-e: xitari-exact RESP0/RESP1 — HBLANK constant 3, visible +5
# offset. Mirrors `jaxtari/jaxtari/tia/system.py::_resp_player_position`.
@inline function _resp_player_position(color_clock::Integer)
    c = Int(color_clock)
    c < HBLANK_COLOR_CLOCKS && return 3
    return ((c - HBLANK_COLOR_CLOCKS) + 5) % SCREEN_WIDTH
end

# P3i-e: xitari-exact RESM0/RESM1/RESBL — HBLANK constant 2, visible
# +4 offset. Mirrors `jaxtari/jaxtari/tia/system.py::_resp_missile_ball_position`.
@inline function _resp_missile_ball_position(color_clock::Integer)
    c = Int(color_clock)
    c < HBLANK_COLOR_CLOCKS && return 2
    return ((c - HBLANK_COLOR_CLOCKS) + 4) % SCREEN_WIDTH
end

"""
    _hm_offset(hm) -> Int

Convert an HM register byte to a signed motion offset. Only the high
nibble matters, interpreted as 4-bit two's complement (+7..-8). Positive
moves the sprite LEFT, negative RIGHT.
"""
@inline function _hm_offset(hm::Integer)
    high = (Int(hm) >> 4) & 0x0F
    return high >= 8 ? high - 16 : high
end

"""
    tia_poke!(tia, addr, value)

Write a TIA register. Stores the byte and applies side-effects: WSYNC
(P3a), player position reset / horizontal motion / HMCLR (P3c).
"""
function tia_poke!(tia::TIAState, addr::Integer, value::Integer,
                   beam_cc::Integer = tia.color_clock,
                   beam_sc::Integer = tia.scanline_cycle)
    reg = Int(addr) & 0x3F                # TIA decodes A0–A5
    value8 = UInt8(Int(value) & 0xFF)
    # COLU* (P0/P1/PF/BK): bit 0 of the color-luminance registers is
    # unused on real NMOS hardware — xitari masks `value & 0xFE` on every
    # poke (`case 0x06..0x09` in `TIA::poke`). Same fix in jaxtari.
    if reg == W_COLUP0 || reg == W_COLUP1 || reg == W_COLUPF || reg == W_COLUBK
        value8 &= 0xFE
    end
    # P3i-g part 2 (timing-only CPU↔TIA threading): `beam_cc` / `beam_sc`
    # are the *effective* sub-instruction beam position at the moment
    # this write hits the bus (instruction-start clock + cycles-so-far*3),
    # supplied by `Bus.poke!`. Verified to match xitari's
    # `mySystem->cycles()*3` write-cycle clock exactly. They default to
    # the TIA's own counters so direct unit-test pokes keep the
    # pre-threading behaviour. Only the beam-position-sensitive
    # registers (PF0/1/2 defer, RES*, HMOVE) consult them.

    # P3i-c: defer PF0/PF1/PF2 mid-scanline writes to their
    # `_POKE_DELAY_TABLE` activation color clock. Skip the defer if
    # we're in HBLANK (beam_cc < 68): the activation would land
    # in the visible region but pre-HBLANK pokes are by convention
    # "scanline setup" and applying them immediately for the whole
    # scanline gives the same render. Same logic as jaxtari.
    # P3i-g pt6: also defer ENAM0/ENAM1/ENABL so a mid-scanline
    # missile/ball disable (e.g. breakout sl 229 ENAM1=0 at cc=105)
    # only blanks the missile from that color clock onwards, not for
    # the whole scanline. Closes the 8 px residual on breakout row 195.
    # P3i-g pt7: extend to NUSIZ/REFP/COLU/CTRLPF — all "no-side-effect"
    # render registers. Xitari's poke does `updateFrame(clock+delay)`
    # before applying any poke, so even delay=0 writes don't affect
    # pixels rendered before the write's CPU cycle. Whole-scanline
    # batched apply was lumping them all into the post-write state.
    # Net: space_invaders 2145→2079, enduro 1972→1954. NOT deferred:
    # GRP*/WSYNC/VSYNC/RES*/HMOVE/CXCLR/VBLANK (they have side effects
    # beyond a register store).
    if (reg == W_PF0 || reg == W_PF1 || reg == W_PF2 ||
        reg == W_ENAM0 || reg == W_ENAM1 || reg == W_ENABL ||
        reg == W_NUSIZ0 || reg == W_NUSIZ1 ||
        reg == W_COLUP0 || reg == W_COLUP1 || reg == W_COLUPF || reg == W_COLUBK ||
        reg == W_CTRLPF || reg == W_REFP0 || reg == W_REFP1) &&
       beam_cc >= HBLANK_COLOR_CLOCKS
        delay = _poke_activation_delay(reg, beam_cc)
        activation_clock = Int(beam_cc) + delay
        # ALWAYS queue — even when activation_clock >= 228 (effect crosses
        # into the next scanline). `tia_advance!`'s drain-remaining step
        # applies those AFTER the per-color-clock loop, in activation
        # order, into the carried-forward registers — so they are NOT
        # clobbered by an earlier same-register pending write. P3i-g
        # bugfix: the old immediate-apply-for->=228 path let Breakout's
        # PF2=$3f@156 clobber a PF2=$00 clear issued at the bottom of the
        # brick band, leaking $3f into the lower screen (the "red columns").
        push!(tia.pending_writes, (activation_clock, reg, value8))
        return nothing                      # do NOT update registers yet
    end

    tia.registers[reg + 1] = value8

    if reg == W_WSYNC
        tia.wsync_pending = true
    elseif reg == W_VSYNC
        new_vsync = (Int(value) & 0x02) != 0
        if tia.vsync_active && !new_vsync
            # 1→0 edge: software frame boundary.
            tia.vsync_active = false
            tia.frame += UInt64(1)
            tia.scanline = 0
            tia.scanline_cycle = 0
        else
            tia.vsync_active = new_vsync
        end
    elseif reg == W_VBLANK
        tia.vblank_active = (Int(value) & 0x02) != 0
        # P4c (dump-pot) — VBLANK D7 grounds the paddle-pot cap. On
        # the 1→0 transition we capture the cycle so subsequent
        # INPT0-3 reads can ask "has the cap charged past the
        # xitari-formula threshold?". Mirrors jaxtari's identical
        # check in tia_poke.
        new_dump = (Int(value) & 0x80) != 0
        if tia.dump_enabled && !new_dump
            tia.dump_enabled = false
            tia.dump_disabled_cycle = tia.total_cycles
        elseif new_dump && !tia.dump_enabled
            tia.dump_enabled = true
        end
    elseif reg == W_RESP0
        # P3i-e/g: xitari-exact player position from the *effective*
        # sub-instruction beam color clock (beam_cc), not the stale
        # instruction-start tia.color_clock — so a RESP0 issued mid-
        # instruction latches the player to the right X (timing-only
        # threading; the TIA itself is NOT advanced here).
        tia.p0_x = _resp_player_position(beam_cc)
    elseif reg == W_RESP1
        tia.p1_x = _resp_player_position(beam_cc)
    elseif reg == W_RESM0
        # P3i-e: missile/ball use the +4 offset / HBLANK constant 2.
        tia.m0_x = _resp_missile_ball_position(beam_cc)
    elseif reg == W_RESM1
        tia.m1_x = _resp_missile_ball_position(beam_cc)
    elseif reg == W_RESBL
        tia.bl_x = _resp_missile_ball_position(beam_cc)
    elseif reg == W_HMOVE
        # P3i-f + P3i-g: use cycle-aware motion table (xitari's
        # ourCompleteMotionTable) indexed by the *effective* CPU cycle
        # within the scanline (beam_sc), not the stale instruction-start
        # scanline_cycle. Mid-scanline HMOVE (cycles 21..54) produces
        # zero motion; late-scanline (55..75) cycle-dependent partials.
        sc = beam_sc
        tia.p0_x = mod(tia.p0_x - _hmove_motion(sc, tia.registers[W_HMP0 + 1]), 160)
        tia.p1_x = mod(tia.p1_x - _hmove_motion(sc, tia.registers[W_HMP1 + 1]), 160)
        tia.m0_x = mod(tia.m0_x - _hmove_motion(sc, tia.registers[W_HMM0 + 1]), 160)
        tia.m1_x = mod(tia.m1_x - _hmove_motion(sc, tia.registers[W_HMM1 + 1]), 160)
        tia.bl_x = mod(tia.bl_x - _hmove_motion(sc, tia.registers[W_HMBL + 1]), 160)
        # HMOVE-blank fires alongside (the cycle ranges overlap with
        # the motion table's HBLANK region).
        tia.hmove_blank_pending = _hmove_blank_enabled_at(sc)
    elseif reg == W_HMCLR
        tia.registers[W_HMP0 + 1] = 0
        tia.registers[W_HMP1 + 1] = 0
        tia.registers[W_HMM0 + 1] = 0
        tia.registers[W_HMM1 + 1] = 0
        tia.registers[W_HMBL + 1] = 0
    elseif reg == W_CXCLR
        fill!(tia.collisions, 0)
    elseif reg == W_GRP0
        # P3h: writing GRP0 latches the CURRENT GRP1 into grp1_old
        # so a VDELP1-enabled P1 renders the previous GRP1 from now
        # on. GRP1 hasn't been touched by this write so reading
        # `registers[W_GRP1+1]` returns the still-current GRP1.
        tia.grp1_old = tia.registers[W_GRP1 + 1]
    elseif reg == W_GRP1
        # Writing GRP1 latches both GRP0 (for VDELP0) and ENABL
        # (for VDELBL — Stella's TIA-circuit convention ties the
        # ball-delay strobe into the GRP1 latch).
        # 2026-06-03: when a previous ENABL write is still PENDING
        # (queued for deferred activation, P3i-g pt6) and its
        # activation_clock is ≤ the current GRP1 beam, the live
        # tia.registers[W_ENABL+1] is STALE — we need the new value
        # for the shadow capture. xitari does the ENABL write
        # immediately and then GRP1's shadow captures the new value.
        # Without this lookback, jutari's `enabl_old` retains a
        # bit-1-set value across what should have been a clear,
        # causing spurious BL-PF collisions during VDELBL=1 (the
        # breakout-frame-92 scn 51-52 bug).
        tia.grp0_old  = tia.registers[W_GRP0 + 1]
        effective_enabl = tia.registers[W_ENABL + 1]
        for (act_cc, w_reg, w_val) in tia.pending_writes
            if w_reg == W_ENABL && act_cc <= Int(beam_cc)
                effective_enabl = w_val   # latest pending overrides
            end
        end
        tia.enabl_old = effective_enabl
    end

    return nothing
end

"""
    tia_advance!(tia, cpu_cycles)

Advance TIA by `cpu_cycles` CPU cycles, rendering each completed
scanline into the framebuffer.

**P3i-a / P3i-b**: the framebuffer write now goes through the
`render_pixel` per-color-clock kernel (160 calls per visible scanline,
looped over color clocks 68..227). For now the kernel sees the same
register state at every color clock — so the output is bit-exact
equivalent to the pre-P3i `render_scanline` write. Sub-phases P3i-c
onwards add per-poke timing so mid-scanline register writes (the
actual P3i payoff) start to bite. `tia.color_clock` also advances 3×
per CPU cycle and is the field P3i-c/d/e/f hang write-delay
arithmetic off of.
"""
function tia_advance!(tia::TIAState, cpu_cycles::Integer)
    # P4c (dump-pot): keep a monotonic CPU-cycle counter so
    # `tia_peek` can decide whether the paddle cap has charged
    # past its xitari-formula threshold yet.
    tia.total_cycles += Int(cpu_cycles)
    total = tia.scanline_cycle + Int(cpu_cycles)
    tia.scanline_cycle = total % NTSC_CPU_CYCLES_PER_SCANLINE
    line_advance = total ÷ NTSC_CPU_CYCLES_PER_SCANLINE
    # P3i-a: color_clock advances 3× per CPU cycle, wrapping at 228.
    tia.color_clock = (tia.color_clock + Int(cpu_cycles) * COLOR_CLOCKS_PER_CPU_CYCLE) %
                      COLOR_CLOCKS_PER_SCANLINE

    if line_advance > 0
        # P3i-c: drain pending PF0/PF1/PF2 writes during the per-
        # color-clock render. Each pending entry (activation_clock,
        # reg, value) applies to the register file at its activation
        # clock during the loop; pixels rendered AT or AFTER the
        # activation clock see the new value, pixels BEFORE see the
        # old. After the first completed scanline, pending_writes
        # has been fully drained; any subsequent scanlines in this
        # advance render with the post-drain state.
        # P3i-d: per-pixel collision detection runs inside the loop
        # (via `_apply_pixel_collisions!`) so the bit OR sees the
        # post-write object sets at each color clock.
        pending_sorted = sort(tia.pending_writes; by = w -> w[1])
        if !tia.vblank_active
            row = Vector{UInt8}(undef, SCREEN_WIDTH)
            write_idx = 1
            cached_sets = _object_pixel_sets(tia)
            # P3i-f: HMOVE-blank window. When `hmove_blank_pending` is
            # true at scanline start, the first 8 visible color clocks
            # (c=68..75 → x=0..7) render as 0 (black bar); collision
            # detection still runs. After c >= 76 the flag clears.
            hmove_blank_active = tia.hmove_blank_pending
            for c in HBLANK_COLOR_CLOCKS:(COLOR_CLOCKS_PER_SCANLINE - 1)
                # Apply any pending writes whose activation_clock <= c
                # before rendering the pixel.
                while write_idx <= length(pending_sorted) &&
                      pending_sorted[write_idx][1] <= c
                    _, reg, val = pending_sorted[write_idx]
                    tia.registers[reg + 1] = val
                    cached_sets = _object_pixel_sets(tia)
                    write_idx += 1
                end
                x = c - HBLANK_COLOR_CLOCKS
                # P3i-f: HMOVE-blank window covers leftmost 8 visible
                # pixels (c in [68..75]). After that, the flag clears.
                if hmove_blank_active && c >= HBLANK_COLOR_CLOCKS + 8
                    hmove_blank_active = false
                end
                if hmove_blank_active
                    row[x + 1] = UInt8(0)
                else
                    row[x + 1] = render_pixel(tia, c, cached_sets)
                end
                # Collision still accumulates during blank (output
                # blanking doesn't disable the collision pipeline).
                _apply_pixel_collisions!(tia, x, cached_sets)
            end
            # Drain any pending writes that didn't activate within the
            # visible region (shouldn't happen — activation < 228 by
            # construction — but defensive).
            while write_idx <= length(pending_sorted)
                _, reg, val = pending_sorted[write_idx]
                tia.registers[reg + 1] = val
                write_idx += 1
            end
            for i in 0:(line_advance - 1)
                completed_line = (tia.scanline + i) % NTSC_SCANLINES_PER_FRAME
                if completed_line < SCREEN_HEIGHT
                    tia.framebuffer[completed_line + 1, :] .= row
                end
            end
        else
            # VBLANK render — output blanked. Drain pending writes but
            # do NOT run collision detection: xitari's
            # `TIA::updateFrameScanline` (TIA.cxx:1121) memsets the
            # framebuffer and RETURNS when `myVBLANK & 0x02` is set,
            # entirely skipping the per-pixel collision switch
            # statement (`myCollision |= ourCollisionTable[...]`).
            # Real hardware DOES keep the collision pipeline live in
            # VBLANK, but xitari does not — and our reference is
            # xitari. Running collisions here was the source of the
            # breakout-frame-92 spurious CXBLPF=$b6 (bit 7 = BL-PF set)
            # — jutari was over-reporting BL-PF collisions during the
            # VBLANK between frames, while xitari did not. (Discovered
            # via per-bus-op trace diff, commit d66b290, 2026-06-03.)
            for (_, reg, val) in pending_sorted
                tia.registers[reg + 1] = val
            end
        end
        # All pending writes have been drained into tia.registers.
        empty!(tia.pending_writes)
        # P3i-f: clear blank after the scanline that consumed it.
        tia.hmove_blank_pending = false
    end

    # PXC1-x: don't increment the frame counter on scanline-wrap. The
    # frame counter is driven *only* by the software VSYNC 1→0 edge
    # (line 189 above). The previous "scanline-wrap as safety fallback"
    # double-counted every frame on ROMs that drove VSYNC normally —
    # the wrap fired one or two scanlines before the VSYNC handler did,
    # and both incremented `frame` for the same frame boundary.
    new_line = tia.scanline + line_advance
    tia.scanline = new_line % NTSC_SCANLINES_PER_FRAME
    return nothing
end

# --------------------------------------------------------------------------- #
# Rendering — P3b: playfield only
# --------------------------------------------------------------------------- #

"""
    playfield_bits(pf0, pf1, pf2) -> Vector{UInt8}

20-bit playfield pattern for the LEFT half of a scanline. Bit-order
quirks (matching xitari/Stella):

  - PF0: only high nibble used; bits 4..7 → playfield pixels 0..3.
  - PF1: all 8 bits, MSB-first; bit 7 → pixel 4, bit 0 → pixel 11.
  - PF2: all 8 bits, LSB-first; bit 0 → pixel 12, bit 7 → pixel 19.
"""
function playfield_bits(pf0::Integer, pf1::Integer, pf2::Integer)
    bits = Vector{UInt8}(undef, 20)
    @inbounds for b in 0:3
        bits[b + 1] = UInt8((Int(pf0) >> (4 + b)) & 1)
    end
    @inbounds for b in 0:7
        bits[4 + b + 1] = UInt8((Int(pf1) >> (7 - b)) & 1)
    end
    @inbounds for b in 0:7
        bits[12 + b + 1] = UInt8((Int(pf2) >> b) & 1)
    end
    return bits
end

"""
    render_playfield_scanline(tia) -> Vector{UInt8}

Return a 160-byte vector of palette indices for one scanline — playfield
only (no sprites). Right half is repeated or mirrored depending on
CTRLPF.D0. Kept for unit tests; `render_scanline` composites sprites on
top.
"""
function render_playfield_scanline(tia::TIAState)
    pf0    = tia.registers[W_PF0 + 1]
    pf1    = tia.registers[W_PF1 + 1]
    pf2    = tia.registers[W_PF2 + 1]
    ctrlpf = tia.registers[W_CTRLPF + 1]
    colupf = tia.registers[W_COLUPF + 1]
    colubk = tia.registers[W_COLUBK + 1]
    reflected = (ctrlpf & 0x01) != 0
    # CTRLPF.D1 SCOREMODE: LEFT half PF pixels coloured COLUP0,
    # RIGHT half COLUP1 (instead of COLUPF). Ball unaffected.
    score = (ctrlpf & 0x02) != 0
    pf_left  = score ? tia.registers[W_COLUP0 + 1] : colupf
    pf_right = score ? tia.registers[W_COLUP1 + 1] : colupf

    left = playfield_bits(pf0, pf1, pf2)
    right = reflected ? reverse(left) : left

    pixels = Vector{UInt8}(undef, 160)
    @inbounds for i in 0:19
        color = left[i + 1] != 0 ? pf_left : colubk
        for k in 0:3
            pixels[i * 4 + k + 1] = color
        end
    end
    @inbounds for i in 0:19
        color = right[i + 1] != 0 ? pf_right : colubk
        for k in 0:3
            pixels[80 + i * 4 + k + 1] = color
        end
    end
    return pixels
end

"""
P3g — NUSIZ low-3-bit decode: `(copy_offsets, scale)`. Matches
jaxtari `_NUSIZ_PLAYER_LAYOUT` byte-for-byte:

  000  1 copy, 1× wide
  001  2 copies, 16-pixel spacing, 1× wide   (close)
  010  2 copies, 32-pixel spacing, 1× wide   (medium)
  011  3 copies, 16-pixel spacing, 1× wide
  100  2 copies, 64-pixel spacing, 1× wide   (wide)
  101  1 copy, 2× wide   (double-sized player)
  110  3 copies, 32-pixel spacing, 1× wide
  111  1 copy, 4× wide   (quadruple-sized player)
"""
const _NUSIZ_PLAYER_LAYOUT = (
    ((0,),         1),
    ((0, 16),      1),
    ((0, 32),      1),
    ((0, 16, 32),  1),
    ((0, 64),      1),
    ((0,),         2),
    ((0, 32, 64),  1),
    ((0,),         4),
)

@inline _nusiz_player_layout(nusiz::Integer) =
    _NUSIZ_PLAYER_LAYOUT[(Int(nusiz) & 0x07) + 1]

"""
    _overlay_player!(pixels, tia, player)

Paint player 0 or 1 onto `pixels` (mutated). **P3g**: NUSIZ low 3
bits decode into per-copy X offsets + per-bit scale (1/2/4), so
multi-copy + 2×/4×-wide layouts paint correctly. The matching
missile inherits the same multi-copy layout via `_overlay_missile!`.
"""
# P3h: `_vdel_grp` / `_vdel_enabl` return the current GRP / ENABL
# byte the renderer should use, honouring VDELP0 / VDELP1 / VDELBL.
# When VDELPx / VDELBL bit 0 is set, the shadow value (latched on
# the previous GRP-of-the-other-player / GRP1 write) is returned;
# otherwise the live register byte. Stella convention.
@inline function _vdel_grp(tia::TIAState, player::Int)
    if player == 0
        return (tia.registers[W_VDELP0 + 1] & UInt8(0x01)) != 0 ?
               tia.grp0_old : tia.registers[W_GRP0 + 1]
    else
        return (tia.registers[W_VDELP1 + 1] & UInt8(0x01)) != 0 ?
               tia.grp1_old : tia.registers[W_GRP1 + 1]
    end
end

@inline _vdel_enabl(tia::TIAState) =
    (tia.registers[W_VDELBL + 1] & UInt8(0x01)) != 0 ?
        tia.enabl_old : tia.registers[W_ENABL + 1]

function _overlay_player!(pixels::Vector{UInt8}, tia::TIAState, player::Int)
    color_reg = player == 0 ? W_COLUP0 : W_COLUP1
    refp_reg  = player == 0 ? W_REFP0  : W_REFP1
    nusiz_reg = player == 0 ? W_NUSIZ0 : W_NUSIZ1
    # P3h: GRP via VDELP-aware lookup.
    grp = _vdel_grp(tia, player)
    grp == 0 && return nothing
    color = tia.registers[color_reg + 1]
    reflected = (tia.registers[refp_reg + 1] & 0x08) != 0
    x = player == 0 ? tia.p0_x : tia.p1_x
    copy_offsets, scale = _nusiz_player_layout(tia.registers[nusiz_reg + 1])

    # GRP bit 7 is leftmost by default; REFP.D3 reverses bit order.
    # Each of the 8 bits paints `scale` adjacent screen pixels, at
    # every requested copy offset. NUSIZ wide modes (scale > 1, ie.
    # double-size NUSIZ=5 and quad-size NUSIZ=7) get a +1 pixel offset
    # (xitari `computePlayerMaskTable` bakes this in: "in double/quad
    # size mode the player's output is delayed by one pixel" — a real
    # NMOS-TIA quirk).
    nusiz_offset = scale > 1 ? 1 : 0
    @inbounds for copy_off in copy_offsets
        base = mod(x + copy_off + nusiz_offset, 160)
        for i in 0:7
            bit_idx = reflected ? i : (7 - i)
            if (Int(grp) >> bit_idx) & 1 != 0
                for k in 0:(scale - 1)
                    pixels[mod(base + i * scale + k, 160) + 1] = color
                end
            end
        end
    end
    return nothing
end

"""
    _overlay_missile!(pixels, tia, missile)

Paint missile 0 or 1. NUSIZ low 3 bits select multi-copy + spacing
(same layout as the matching player); NUSIZ bits 4-5 set each copy's
*width* (1/2/4/8 pixels). Missile inherits the colour of its
associated player (COLUP*).
"""
function _overlay_missile!(pixels::Vector{UInt8}, tia::TIAState, missile::Int)
    enam_reg = missile == 0 ? W_ENAM0 : W_ENAM1
    (tia.registers[enam_reg + 1] & 0x02) == 0 && return nothing
    color_reg = missile == 0 ? W_COLUP0 : W_COLUP1
    nusiz_reg = missile == 0 ? W_NUSIZ0 : W_NUSIZ1
    color = tia.registers[color_reg + 1]
    nusiz = tia.registers[nusiz_reg + 1]
    width = 1 << ((Int(nusiz) >> 4) & 0x03)
    copy_offsets, _ = _nusiz_player_layout(nusiz)
    x = missile == 0 ? tia.m0_x : tia.m1_x
    @inbounds for copy_off in copy_offsets
        base = mod(x + copy_off, 160)
        for i in 0:(width - 1)
            pixels[mod(base + i, 160) + 1] = color
        end
    end
    return nothing
end

"""
    _overlay_ball!(pixels, tia)

Paint the ball. Uses COLUPF; size 1/2/4/8 from CTRLPF bits 4-5.
"""
function _overlay_ball!(pixels::Vector{UInt8}, tia::TIAState)
    # P3h: ENABL via VDELBL-aware lookup.
    (_vdel_enabl(tia) & UInt8(0x02)) == 0 && return nothing
    color = tia.registers[W_COLUPF + 1]
    size  = 1 << ((Int(tia.registers[W_CTRLPF + 1]) >> 4) & 0x03)
    x = tia.bl_x
    @inbounds for i in 0:(size - 1)
        pixels[mod(x + i, 160) + 1] = color
    end
    return nothing
end

"""
    _overlay_playfield!(pixels, tia)

Re-paint the playfield (without disturbing the background) over an
already-rendered scanline. Used by the CTRLPF.D2 (PFP) priority swap:
when the PF-priority bit is set the playfield + ball composite *on top*
of players + missiles, so the renderer paints sprites first and then
drops the playfield bits on top via this helper.
"""
function _overlay_playfield!(pixels::Vector{UInt8}, tia::TIAState)
    pf0    = tia.registers[W_PF0 + 1]
    pf1    = tia.registers[W_PF1 + 1]
    pf2    = tia.registers[W_PF2 + 1]
    ctrlpf = tia.registers[W_CTRLPF + 1]
    colupf = tia.registers[W_COLUPF + 1]
    reflected = (ctrlpf & 0x01) != 0
    score    = (ctrlpf & 0x02) != 0
    pf_left  = score ? tia.registers[W_COLUP0 + 1] : colupf
    pf_right = score ? tia.registers[W_COLUP1 + 1] : colupf
    left  = playfield_bits(pf0, pf1, pf2)
    right = reflected ? reverse(left) : left
    @inbounds for i in 0:19
        if left[i + 1] != 0
            for k in 0:3
                pixels[i * 4 + k + 1] = pf_left
            end
        end
    end
    @inbounds for i in 0:19
        if right[i + 1] != 0
            for k in 0:3
                pixels[80 + i * 4 + k + 1] = pf_right
            end
        end
    end
    return nothing
end

"""
    render_scanline(tia) -> Vector{UInt8}

Composite renderer: playfield + ball + players + missiles. Two priority
modes, selected by CTRLPF bit 2 (PFP):

  PFP=0 (default):  bg ← pf ← bl ← M1 ← P1 ← M0 ← P0
  PFP=1 (priority): bg ← M1 ← P1 ← M0 ← P0 ← pf ← bl

With PFP set the playfield + ball composite on top of sprites — the
canonical use case being a paddle game's score display or a maze that
should never be covered by a sprite. P3l adds the second branch.
"""
function render_scanline(tia::TIAState)
    ctrlpf = tia.registers[W_CTRLPF + 1]
    pfp = (ctrlpf & 0x04) != 0

    if !pfp
        # Default priority.
        pixels = render_playfield_scanline(tia)
        _overlay_ball!(pixels, tia)
        _overlay_missile!(pixels, tia, 1)
        _overlay_player!(pixels, tia, 1)
        _overlay_missile!(pixels, tia, 0)
        _overlay_player!(pixels, tia, 0)
    else
        # PFP set — playfield + ball on top of sprites. Start from the
        # background only (we re-overlay the playfield at the end),
        # draw players + missiles, then drop the playfield + ball on
        # top.
        colubk = tia.registers[W_COLUBK + 1]
        pixels = fill(colubk, 160)
        _overlay_missile!(pixels, tia, 1)
        _overlay_player!(pixels, tia, 1)
        _overlay_missile!(pixels, tia, 0)
        _overlay_player!(pixels, tia, 0)
        _overlay_playfield!(pixels, tia)
        _overlay_ball!(pixels, tia)
    end
    return pixels
end

"""
    render_pixel(tia, color_clock, cached_sets=nothing) -> UInt8

P3i-a: per-color-clock pixel kernel. Returns the pixel value at beam
position `color_clock` (0..227) within the current scanline. HBLANK
positions (0..67) and positions past the visible region return 0.
Visible positions (68..227) map to framebuffer X = `color_clock - 68`.

Bit-exact equivalent to `render_scanline(tia)[color_clock - 67]` for
every visible color clock — P3i-c will start breaking that equivalence
when mid-scanline pokes apply at their `ourPokeDelayTable` activation
color clock instead of immediately.

`cached_sets` is the `_object_pixel_sets(tia)` return value precomputed
once per scanline so `tia_advance!`'s per-color-clock loop doesn't
redo it 160 times. Mirrors `jaxtari/jaxtari/tia/system.py::render_pixel`.
"""
function render_pixel(tia::TIAState, color_clock::Integer, cached_sets=nothing)
    if color_clock < HBLANK_COLOR_CLOCKS
        return UInt8(0)
    end
    x = Int(color_clock) - HBLANK_COLOR_CLOCKS
    if x >= SCREEN_WIDTH
        return UInt8(0)
    end

    colubk = tia.registers[W_COLUBK + 1]
    colupf = tia.registers[W_COLUPF + 1]
    colup0 = tia.registers[W_COLUP0 + 1]
    colup1 = tia.registers[W_COLUP1 + 1]
    ctrlpf = tia.registers[W_CTRLPF + 1]
    pfp   = (ctrlpf & 0x04) != 0
    # CTRLPF.D1 SCOREMODE: PF LEFT half coloured COLUP0, RIGHT COLUP1
    # (ball stays COLUPF).
    score = (ctrlpf & 0x02) != 0
    pf_col = score ? (x < 80 ? colup0 : colup1) : colupf

    sets = cached_sets === nothing ? _object_pixel_sets(tia) : cached_sets

    pixel = colubk
    if !pfp
        # Default: bg ← pf ← bl ← M1 ← P1 ← M0 ← P0
        if x in sets.pf; pixel = pf_col; end
        if x in sets.bl; pixel = colupf; end
        if x in sets.m1; pixel = colup1; end
        if x in sets.p1; pixel = colup1; end
        if x in sets.m0; pixel = colup0; end
        if x in sets.p0; pixel = colup0; end
    else
        # PFP: bg ← M1 ← P1 ← M0 ← P0 ← pf ← bl
        if x in sets.m1; pixel = colup1; end
        if x in sets.p1; pixel = colup1; end
        if x in sets.m0; pixel = colup0; end
        if x in sets.p0; pixel = colup0; end
        if x in sets.pf; pixel = pf_col; end
        if x in sets.bl; pixel = colupf; end
    end
    return pixel
end

# --------------------------------------------------------------------------- #
# Collision detection (P3e)
# --------------------------------------------------------------------------- #

"""
    _object_pixel_sets(tia) -> NamedTuple

Per-object sets of screen-pixel indices for the current scanline. Used
by collision detection.
"""
function _object_pixel_sets(tia::TIAState)
    pf0    = tia.registers[W_PF0 + 1]
    pf1    = tia.registers[W_PF1 + 1]
    pf2    = tia.registers[W_PF2 + 1]
    ctrlpf = tia.registers[W_CTRLPF + 1]
    left_bits  = playfield_bits(pf0, pf1, pf2)
    right_bits = (ctrlpf & 0x01) != 0 ? reverse(left_bits) : left_bits
    pf = Set{Int}()
    @inbounds for (i, b) in enumerate(left_bits)
        if b != 0
            for k in 0:3; push!(pf, (i - 1) * 4 + k); end
        end
    end
    @inbounds for (i, b) in enumerate(right_bits)
        if b != 0
            for k in 0:3; push!(pf, 80 + (i - 1) * 4 + k); end
        end
    end

    bl = Set{Int}()
    # P3h: ENABL via VDELBL-aware lookup.
    if (_vdel_enabl(tia) & UInt8(0x02)) != 0
        size = 1 << ((Int(ctrlpf) >> 4) & 0x03)
        for i in 0:(size - 1)
            push!(bl, mod(tia.bl_x + i, 160))
        end
    end

    # P3g: respect NUSIZ multi-copy + scale so collisions involving
    # the 2nd/3rd copies and the wide modes are detected.
    # P3h: GRP via VDELP-aware lookup.
    function _player_set(grp_reg, refp_reg, nusiz_reg, x)
        player = grp_reg == W_GRP0 ? 0 : 1
        grp = _vdel_grp(tia, player)
        out = Set{Int}()
        grp == 0 && return out
        reflected = (tia.registers[refp_reg + 1] & 0x08) != 0
        copy_offsets, scale = _nusiz_player_layout(tia.registers[nusiz_reg + 1])
        # +1 pixel offset for wide modes (scale > 1) — matches xitari's
        # quad/double-size mask quirk; see `_overlay_player!`.
        nusiz_offset = scale > 1 ? 1 : 0
        for copy_off in copy_offsets
            base = mod(x + copy_off + nusiz_offset, 160)
            for i in 0:7
                bit_idx = reflected ? i : (7 - i)
                if (Int(grp) >> bit_idx) & 1 != 0
                    for k in 0:(scale - 1)
                        push!(out, mod(base + i * scale + k, 160))
                    end
                end
            end
        end
        return out
    end

    function _missile_set(enam_reg, nusiz_reg, x)
        (tia.registers[enam_reg + 1] & 0x02) == 0 && return Set{Int}()
        nusiz = tia.registers[nusiz_reg + 1]
        width = 1 << ((Int(nusiz) >> 4) & 0x03)
        copy_offsets, _ = _nusiz_player_layout(nusiz)
        out = Set{Int}()
        for copy_off in copy_offsets
            base = mod(x + copy_off, 160)
            for i in 0:(width - 1)
                push!(out, mod(base + i, 160))
            end
        end
        return out
    end

    return (
        pf = pf,
        bl = bl,
        p0 = _player_set(W_GRP0, W_REFP0, W_NUSIZ0, tia.p0_x),
        p1 = _player_set(W_GRP1, W_REFP1, W_NUSIZ1, tia.p1_x),
        m0 = _missile_set(W_ENAM0, W_NUSIZ0, tia.m0_x),
        m1 = _missile_set(W_ENAM1, W_NUSIZ1, tia.m1_x),
    )
end

"""
    _detect_collisions!(tia)

OR new collision bits into `tia.collisions` based on the current scanline's
object overlap. See the bit layout in the jaxtari counterpart.
"""
function _detect_collisions!(tia::TIAState)
    objs = _object_pixel_sets(tia)
    p0, p1 = objs.p0, objs.p1
    m0, m1 = objs.m0, objs.m1
    bl, pf = objs.bl, objs.pf
    hit(a, b) = !isempty(a) && !isempty(b) && !isempty(intersect(a, b))

    hit(m0, p1) && (tia.collisions[1] |= 0x80)   # CXM0P D7
    hit(m0, p0) && (tia.collisions[1] |= 0x40)   # CXM0P D6
    hit(m1, p0) && (tia.collisions[2] |= 0x80)   # CXM1P D7
    hit(m1, p1) && (tia.collisions[2] |= 0x40)   # CXM1P D6
    hit(p0, pf) && (tia.collisions[3] |= 0x80)   # CXP0FB D7
    hit(p0, bl) && (tia.collisions[3] |= 0x40)   # CXP0FB D6
    hit(p1, pf) && (tia.collisions[4] |= 0x80)   # CXP1FB D7
    hit(p1, bl) && (tia.collisions[4] |= 0x40)   # CXP1FB D6
    hit(m0, pf) && (tia.collisions[5] |= 0x80)   # CXM0FB D7
    hit(m0, bl) && (tia.collisions[5] |= 0x40)   # CXM0FB D6
    hit(m1, pf) && (tia.collisions[6] |= 0x80)   # CXM1FB D7
    hit(m1, bl) && (tia.collisions[6] |= 0x40)   # CXM1FB D6
    hit(bl, pf) && (tia.collisions[7] |= 0x80)   # CXBLPF D7 (D6 unused)
    hit(p0, p1) && (tia.collisions[8] |= 0x80)   # CXPPMM D7
    hit(m0, m1) && (tia.collisions[8] |= 0x40)   # CXPPMM D6
    return nothing
end

# P3i-d: per-pixel collision evaluation. Mutates `tia.collisions` in
# place, OR'ing the bits for objects that overlap at visible pixel x.
# Mirrors `_detect_collisions!` bit layout exactly; the only difference
# is the granularity (one pixel vs whole-scanline OR), which lets
# mid-scanline register changes (PF stomping via P3i-c, future per-
# pixel sprite-position adjustments) affect collision evaluation at
# the exact color clock they apply.
@inline function _apply_pixel_collisions!(tia::TIAState, x::Integer, sets)
    p0_here = x in sets.p0
    p1_here = x in sets.p1
    m0_here = x in sets.m0
    m1_here = x in sets.m1
    bl_here = x in sets.bl
    pf_here = x in sets.pf

    m0_here && p1_here && (tia.collisions[1] |= 0x80)   # CXM0P D7 = M0-P1
    m0_here && p0_here && (tia.collisions[1] |= 0x40)   # CXM0P D6 = M0-P0
    m1_here && p0_here && (tia.collisions[2] |= 0x80)   # CXM1P D7 = M1-P0
    m1_here && p1_here && (tia.collisions[2] |= 0x40)   # CXM1P D6 = M1-P1
    p0_here && pf_here && (tia.collisions[3] |= 0x80)   # CXP0FB D7 = P0-PF
    p0_here && bl_here && (tia.collisions[3] |= 0x40)   # CXP0FB D6 = P0-BL
    p1_here && pf_here && (tia.collisions[4] |= 0x80)   # CXP1FB D7 = P1-PF
    p1_here && bl_here && (tia.collisions[4] |= 0x40)   # CXP1FB D6 = P1-BL
    m0_here && pf_here && (tia.collisions[5] |= 0x80)   # CXM0FB D7 = M0-PF
    m0_here && bl_here && (tia.collisions[5] |= 0x40)   # CXM0FB D6 = M0-BL
    m1_here && pf_here && (tia.collisions[6] |= 0x80)   # CXM1FB D7 = M1-PF
    m1_here && bl_here && (tia.collisions[6] |= 0x40)   # CXM1FB D6 = M1-BL
    bl_here && pf_here && (tia.collisions[7] |= 0x80)   # CXBLPF D7 = BL-PF
    p0_here && p1_here && (tia.collisions[8] |= 0x80)   # CXPPMM D7 = P0-P1
    m0_here && m1_here && (tia.collisions[8] |= 0x40)   # CXPPMM D6 = M0-M1
    return nothing
end

"""
    tia_apply_wsync!(tia) -> stall_cycles::Int

If `tia.wsync_pending`, advance TIA to the next scanline boundary and
return the cycle count to be added to `state.cycles`. Otherwise return 0.
"""
function tia_apply_wsync!(tia::TIAState)
    tia.wsync_pending || return 0
    stall = mod(NTSC_CPU_CYCLES_PER_SCANLINE - tia.scanline_cycle,
                NTSC_CPU_CYCLES_PER_SCANLINE)
    if stall > 0
        tia_advance!(tia, stall)
    end
    tia.wsync_pending = false
    return stall
end

end # module
