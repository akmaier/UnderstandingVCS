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
       playfield_bits, render_playfield_scanline, render_scanline,
       set_trigger!,
       _hm_offset, _resp_position,
       NTSC_CPU_CYCLES_PER_SCANLINE, NTSC_SCANLINES_PER_FRAME,
       NUM_REGISTERS, SCREEN_WIDTH, SCREEN_HEIGHT,
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
const SCREEN_HEIGHT = 192

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
)

"""
    tia_peek(tia, addr) -> UInt8

Read a TIA register. \$30..\$37 → collision latches; \$38..\$3D → INPT0..5
(paddle pots default \$80, triggers idle high; set_trigger! flips them);
\$3E/\$3F → unused, return 0.
"""
@inline function tia_peek(tia::TIAState, addr::Integer)
    reg = Int(addr) & 0x0F
    if reg < 8
        return tia.collisions[reg + 1]
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
    _resp_position(scanline_cycle) -> Int

Approximate RESP* timing: maps scanline_cycle to sprite position in
[0,159]. Single linear formula `scanline_cycle * 3 - 68`, clamped.
"""
@inline function _resp_position(scanline_cycle::Integer)
    pos = Int(scanline_cycle) * 3 - 68
    pos < 0 && return 0
    pos > 159 && return 159
    return pos
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
function tia_poke!(tia::TIAState, addr::Integer, value::Integer)
    reg = Int(addr) & 0x3F                # TIA decodes A0–A5
    tia.registers[reg + 1] = UInt8(Int(value) & 0xFF)

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
    elseif reg == W_RESP0
        tia.p0_x = _resp_position(tia.scanline_cycle)
    elseif reg == W_RESP1
        tia.p1_x = _resp_position(tia.scanline_cycle)
    elseif reg == W_RESM0
        tia.m0_x = _resp_position(tia.scanline_cycle)
    elseif reg == W_RESM1
        tia.m1_x = _resp_position(tia.scanline_cycle)
    elseif reg == W_RESBL
        tia.bl_x = _resp_position(tia.scanline_cycle)
    elseif reg == W_HMOVE
        tia.p0_x = mod(tia.p0_x - _hm_offset(tia.registers[W_HMP0 + 1]), 160)
        tia.p1_x = mod(tia.p1_x - _hm_offset(tia.registers[W_HMP1 + 1]), 160)
        tia.m0_x = mod(tia.m0_x - _hm_offset(tia.registers[W_HMM0 + 1]), 160)
        tia.m1_x = mod(tia.m1_x - _hm_offset(tia.registers[W_HMM1 + 1]), 160)
        tia.bl_x = mod(tia.bl_x - _hm_offset(tia.registers[W_HMBL + 1]), 160)
    elseif reg == W_HMCLR
        tia.registers[W_HMP0 + 1] = 0
        tia.registers[W_HMP1 + 1] = 0
        tia.registers[W_HMM0 + 1] = 0
        tia.registers[W_HMM1 + 1] = 0
        tia.registers[W_HMBL + 1] = 0
    elseif reg == W_CXCLR
        fill!(tia.collisions, 0)
    end

    return nothing
end

"""
    tia_advance!(tia, cpu_cycles)

Advance TIA by `cpu_cycles` CPU cycles, rendering each completed
scanline into the framebuffer. Rendering uses end-of-scanline register
state; beam-accurate rendering lands in P3f.
"""
function tia_advance!(tia::TIAState, cpu_cycles::Integer)
    total = tia.scanline_cycle + Int(cpu_cycles)
    tia.scanline_cycle = total % NTSC_CPU_CYCLES_PER_SCANLINE
    line_advance = total ÷ NTSC_CPU_CYCLES_PER_SCANLINE

    if line_advance > 0
        scanline_pixels = render_scanline(tia)
        _detect_collisions!(tia)
        # When VBLANK is active, output is blanked — collisions still
        # accumulate but framebuffer writes are suppressed.
        if !tia.vblank_active
            for i in 0:(line_advance - 1)
                completed_line = (tia.scanline + i) % NTSC_SCANLINES_PER_FRAME
                if completed_line < SCREEN_HEIGHT
                    tia.framebuffer[completed_line + 1, :] .= scanline_pixels
                end
            end
        end
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

    left = playfield_bits(pf0, pf1, pf2)
    right = reflected ? reverse(left) : left

    pixels = Vector{UInt8}(undef, 160)
    @inbounds for i in 0:19
        color = left[i + 1] != 0 ? colupf : colubk
        for k in 0:3
            pixels[i * 4 + k + 1] = color
        end
    end
    @inbounds for i in 0:19
        color = right[i + 1] != 0 ? colupf : colubk
        for k in 0:3
            pixels[80 + i * 4 + k + 1] = color
        end
    end
    return pixels
end

"""
    _overlay_player!(pixels, tia, player)

Paint player 0 or 1 onto `pixels` (mutated). Single 1×-wide copy.
"""
function _overlay_player!(pixels::Vector{UInt8}, tia::TIAState, player::Int)
    grp_reg   = player == 0 ? W_GRP0   : W_GRP1
    color_reg = player == 0 ? W_COLUP0 : W_COLUP1
    refp_reg  = player == 0 ? W_REFP0  : W_REFP1
    grp = tia.registers[grp_reg + 1]
    grp == 0 && return nothing
    color = tia.registers[color_reg + 1]
    reflected = (tia.registers[refp_reg + 1] & 0x08) != 0
    x = player == 0 ? tia.p0_x : tia.p1_x

    # GRP bit 7 is leftmost by default; REFP.D3 reverses bit order.
    @inbounds for i in 0:7
        bit_idx = reflected ? i : (7 - i)
        if (Int(grp) >> bit_idx) & 1 != 0
            px = mod(x + i, 160)
            pixels[px + 1] = color
        end
    end
    return nothing
end

"""
    _overlay_missile!(pixels, tia, missile)

Paint missile 0 or 1; width is 1/2/4/8 from NUSIZ bits 4-5. Missile
inherits the colour of its associated player (COLUP*).
"""
function _overlay_missile!(pixels::Vector{UInt8}, tia::TIAState, missile::Int)
    enam_reg = missile == 0 ? W_ENAM0 : W_ENAM1
    (tia.registers[enam_reg + 1] & 0x02) == 0 && return nothing
    color_reg = missile == 0 ? W_COLUP0 : W_COLUP1
    nusiz_reg = missile == 0 ? W_NUSIZ0 : W_NUSIZ1
    color = tia.registers[color_reg + 1]
    size  = 1 << ((Int(tia.registers[nusiz_reg + 1]) >> 4) & 0x03)
    x = missile == 0 ? tia.m0_x : tia.m1_x
    @inbounds for i in 0:(size - 1)
        pixels[mod(x + i, 160) + 1] = color
    end
    return nothing
end

"""
    _overlay_ball!(pixels, tia)

Paint the ball. Uses COLUPF; size 1/2/4/8 from CTRLPF bits 4-5.
"""
function _overlay_ball!(pixels::Vector{UInt8}, tia::TIAState)
    (tia.registers[W_ENABL + 1] & 0x02) == 0 && return nothing
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
    left  = playfield_bits(pf0, pf1, pf2)
    right = reflected ? reverse(left) : left
    @inbounds for i in 0:19
        if left[i + 1] != 0
            for k in 0:3
                pixels[i * 4 + k + 1] = colupf
            end
        end
    end
    @inbounds for i in 0:19
        if right[i + 1] != 0
            for k in 0:3
                pixels[80 + i * 4 + k + 1] = colupf
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
    if (tia.registers[W_ENABL + 1] & 0x02) != 0
        size = 1 << ((Int(ctrlpf) >> 4) & 0x03)
        for i in 0:(size - 1)
            push!(bl, mod(tia.bl_x + i, 160))
        end
    end

    function _player_set(grp_reg, refp_reg, x)
        grp = tia.registers[grp_reg + 1]
        out = Set{Int}()
        grp == 0 && return out
        reflected = (tia.registers[refp_reg + 1] & 0x08) != 0
        for i in 0:7
            bit_idx = reflected ? i : (7 - i)
            if (Int(grp) >> bit_idx) & 1 != 0
                push!(out, mod(x + i, 160))
            end
        end
        return out
    end

    function _missile_set(enam_reg, nusiz_reg, x)
        (tia.registers[enam_reg + 1] & 0x02) == 0 && return Set{Int}()
        size = 1 << ((Int(tia.registers[nusiz_reg + 1]) >> 4) & 0x03)
        return Set(mod(x + i, 160) for i in 0:(size - 1))
    end

    return (
        pf = pf,
        bl = bl,
        p0 = _player_set(W_GRP0, W_REFP0, tia.p0_x),
        p1 = _player_set(W_GRP1, W_REFP1, tia.p1_x),
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
