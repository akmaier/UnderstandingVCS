# SOFT-mode TIA rendering — P7f-a (differentiable background + playfield).
#
# A differentiable parallel to the HARD TIA's playfield render
# (TIA.render_playfield_scanline). This is the piece that lets a
# reverse-mode AD flow from a framebuffer *pixel* back to a ROM byte —
# the headline ∂pixel/∂ROM the project is built around.
#
# The SOFT bus collapses the TIA register file ($00-$3F) into the low
# cells of `bus.ram`: a `STA $09` resolves through the `addr & 0x7F`
# decode to ram[9], and $09 is COLUBK. So the renderer reads its
# registers straight out of `bus.ram[1:64]` (1-based) — no new field,
# no decode change.
#
# A scanline pixel is
#     pixel = pf_mask * COLUPF + (1 - pf_mask) * COLUBK
# COLUBK / COLUPF are read with `soft_ram_peek` (Zygote-differentiable
# — see P7e); the playfield pattern is integer bit-extraction.
#
# P7f-b adds player sprites P0/P1 (single 1x-wide 8-pixel copy each),
# compositing over the playfield with standard TIA priority. SOFT-mode
# convention: the RESP0/RESP1 cells hold the player X position (real
# hardware sets it by strobe timing — faithful positioning is P7f-d).
#
# P7f-c adds missiles M0/M1 and the ball BL — solid 1/2/4/8-pixel
# blocks. Missiles take their player's colour, the ball takes COLUPF.
#
# P7f-d adds collision detection: soft_collision_registers returns the
# 8 CX latch registers (matching the HARD TIA P3e layout).
#
# Scope: P7f-a…d cover the full visible object set + collisions.

const SOFT_SCREEN_WIDTH      = 160
const SOFT_VISIBLE_SCANLINES = 192

# TIA register offsets — also the SOFT-bus RAM cells they collapse into.
const _R_NUSIZ0 = 0x04
const _R_NUSIZ1 = 0x05
const _R_COLUP0 = 0x06
const _R_COLUP1 = 0x07
const _R_COLUPF = 0x08
const _R_COLUBK = 0x09
const _R_REFP0  = 0x0B
const _R_REFP1  = 0x0C
const _R_CTRLPF = 0x0A
const _R_PF0    = 0x0D
const _R_PF1    = 0x0E
const _R_PF2    = 0x0F
const _R_RESP0  = 0x10      # SOFT convention: holds the player 0 X position
const _R_RESP1  = 0x11      # SOFT convention: holds the player 1 X position
const _R_RESM0  = 0x12      # SOFT convention: holds the missile 0 X position
const _R_RESM1  = 0x13      # SOFT convention: holds the missile 1 X position
const _R_RESBL  = 0x14      # SOFT convention: holds the ball X position
const _R_GRP0   = 0x1B
const _R_GRP1   = 0x1C
const _R_ENAM0  = 0x1D
const _R_ENAM1  = 0x1E
const _R_ENABL  = 0x1F

"""
    _playfield_mask(pf0, pf1, pf2, ctrlpf) -> Vector{Float32}

The TIA's 20-bit playfield expanded to a 160-element mask (1.0 =
playfield pixel, 0.0 = background). Bit order matches
`TIA._playfield_bits`: PF0 bits 4..7, PF1 bits 7..0, PF2 bits 0..7.
Each bit covers 4 pixels; the right half repeats, or mirrors when
CTRLPF bit 0 (reflect) is set.

The bit extraction uses **scalar** `>>` inside comprehensions, not a
broadcast `.>>`: Zygote's broadcast adjoint lifts operands to
`ForwardDiff.Dual`, and `>>` has no `Dual` method. `trunc(Int, ·)`
drops the tangent, so `pf0i…pf2i` are plain `Int`s and the whole mask
is a Zygote-constant — `vcat` / `repeat` / `reverse` then keep the
surrounding render traceable (P7e). The colours carry the gradient.
"""
function _playfield_mask(pf0::Real, pf1::Real, pf2::Real, ctrlpf::Real)
    pf0i = trunc(Int, pf0)
    pf1i = trunc(Int, pf1)
    pf2i = trunc(Int, pf2)
    pf0_bits = [(pf0i >> (4 + b)) & 1 for b in 0:3]   # PF0 bits 4..7
    pf1_bits = [(pf1i >> (7 - b)) & 1 for b in 0:7]   # PF1 bits 7..0
    pf2_bits = [(pf2i >> b)       & 1 for b in 0:7]   # PF2 bits 0..7
    bits20   = Float32.(vcat(pf0_bits, pf1_bits, pf2_bits))   # 20

    left      = repeat(bits20, inner = 4)            # 80
    reflected = (trunc(Int, ctrlpf) & 0x01) != 0
    right     = reflected ? repeat(reverse(bits20), inner = 4) :
                            repeat(bits20, inner = 4)
    return vcat(left, right)                         # 160
end

# P3g — NUSIZ multi-copy + 2×/4×-wide player scaling. Mirror of
# jaxtari's _NUSIZ_PLAYER_LAYOUT byte-for-byte:
#
#   000  1 copy, 1× wide
#   001  2 copies, 16-pixel spacing, 1× wide
#   010  2 copies, 32-pixel spacing, 1× wide
#   011  3 copies, 16-pixel spacing, 1× wide
#   100  2 copies, 64-pixel spacing, 1× wide
#   101  1 copy, 2× wide
#   110  3 copies, 32-pixel spacing, 1× wide
#   111  1 copy, 4× wide
const _SOFT_NUSIZ_PLAYER_LAYOUT = (
    ((0,),         1),
    ((0, 16),      1),
    ((0, 32),      1),
    ((0, 16, 32),  1),
    ((0, 64),      1),
    ((0,),         2),
    ((0, 32, 64),  1),
    ((0,),         4),
)

"""
    _player_mask_single(grp, refp, xpos, scale) -> Vector{Float32}

One copy of a player sprite at column `xpos` with each bit replicated
`scale` times (1/2/4 — covers the NUSIZ double / quad modes). Returns
a 160-element Float32 mask. Built from pure broadcasts so the
surrounding render stays Zygote-traceable (P7e).
"""
function _player_mask_single(grp::Real, refp::Real, xpos::Real, scale::Integer)
    grp_i     = trunc(Int, grp)
    reflected = (trunc(Int, refp) & 0x08) != 0
    x         = trunc(Int, xpos)
    d = mod.((0:(SOFT_SCREEN_WIDTH - 1)) .- x, SOFT_SCREEN_WIDTH)   # (160,)
    # Each sprite bit k spans `scale` adjacent screen pixels. The
    # bit-index for a given d is `d ÷ scale`, valid for d in [0, 8*scale).
    spritebit(k) = Float32(reflected ? ((grp_i >> k) & 1) :
                                       ((grp_i >> (7 - k)) & 1))
    width = 8 * scale
    mask  = zeros(Float32, SOFT_SCREEN_WIDTH)
    # Sum-of-per-bit-contributions — broadcast-friendly so Zygote can
    # trace it (a Julia for-loop with `+=` on the local mask also
    # works because Zygote handles vector-valued local accumulation;
    # the dot-product / one-hot form would be cleaner but the
    # accumulator pattern matches the jaxtari `_player_mask_single`
    # logic 1:1 and stays readable).
    for k in 0:7
        bit = spritebit(k)
        bit == 0f0 && continue
        for s in 0:(scale - 1)
            mask = mask .+ bit .* (d .== (k * scale + s))
        end
    end
    return mask
end

"""
    _player_mask(grp, refp, xpos, nusiz) -> Vector{Float32}

P3g: a player sprite with NUSIZ-driven multi-copy + 2×/4× scaling.
All 8 NUSIZ modes are evaluated unconditionally and blended by the
integer-extracted `nusiz & 7` — same pattern as the CTRLPF.D2
priority swap.
"""
function _player_mask(grp::Real, refp::Real, xpos::Real, nusiz::Real)
    nusiz_low = trunc(Int, nusiz) & 0x07
    per_mode = ntuple(8) do mode_idx
        offsets, scale = _SOFT_NUSIZ_PLAYER_LAYOUT[mode_idx]
        m = zeros(Float32, SOFT_SCREEN_WIDTH)
        for off in offsets
            m = m .+ _player_mask_single(grp, refp, xpos + off, scale)
        end
        clamp.(m, 0f0, 1f0)
    end
    # One-hot select by NUSIZ mode (integer extraction breaks the
    # gradient on the bit itself, same convention as `_block_mask`).
    sel = Float32.((0:7) .== nusiz_low)
    return sum(sel[i] .* per_mode[i] for i in 1:8)
end

"""
    _block_mask_single(xpos, size_reg, enable_reg) -> Vector{Float32}

One copy of a missile / ball block at column `xpos`, width
`1 << ((size_reg >> 4) & 3)` (1/2/4/8), enabled by `enable_reg` bit 1.
"""
function _block_mask_single(xpos::Real, size_reg::Real, enable_reg::Real)
    size_log2 = (trunc(Int, size_reg) >> 4) & 0x03
    width     = 1 << size_log2                       # 1/2/4/8
    enabled   = Float32((trunc(Int, enable_reg) >> 1) & 1)
    x = trunc(Int, xpos)
    d = mod.((0:(SOFT_SCREEN_WIDTH - 1)) .- x, SOFT_SCREEN_WIDTH)   # (160,)
    return enabled .* Float32.(d .< width)
end

"""
    _block_mask(xpos, size_reg, enable_reg, nusiz=nothing) -> Vector{Float32}

Missile / ball block with optional NUSIZ-driven multi-copy. The ball
ignores `nusiz` (it doesn't share NUSIZ; its sizing comes from CTRLPF
passed in via `size_reg`). Missiles pass NUSIZ so the same multi-copy
the player uses applies. P3g.
"""
function _block_mask(xpos::Real, size_reg::Real, enable_reg::Real,
                     nusiz::Union{Real,Nothing} = nothing)
    nusiz === nothing && return _block_mask_single(xpos, size_reg, enable_reg)
    nusiz_low = trunc(Int, nusiz) & 0x07
    per_mode = ntuple(8) do mode_idx
        offsets, _ = _SOFT_NUSIZ_PLAYER_LAYOUT[mode_idx]
        m = zeros(Float32, SOFT_SCREEN_WIDTH)
        for off in offsets
            m = m .+ _block_mask_single(xpos + off, size_reg, enable_reg)
        end
        clamp.(m, 0f0, 1f0)
    end
    sel = Float32.((0:7) .== nusiz_low)
    return sum(sel[i] .* per_mode[i] for i in 1:8)
end

"""
    _object_masks(bus::SoftBus)

The six TIA object masks for the current scanline, each a 160-element
Float32 vector (1.0 where the object is present): playfield, P0, P1,
M0, M1, ball. Shared by `soft_render_scanline` and
`soft_collision_registers`.
"""
function _object_masks(bus::SoftBus)
    ctrlpf = soft_ram_peek(bus.ram, _R_CTRLPF)
    pf = _playfield_mask(soft_ram_peek(bus.ram, _R_PF0),
                         soft_ram_peek(bus.ram, _R_PF1),
                         soft_ram_peek(bus.ram, _R_PF2),
                         ctrlpf)
    nusiz0 = soft_ram_peek(bus.ram, _R_NUSIZ0)
    nusiz1 = soft_ram_peek(bus.ram, _R_NUSIZ1)
    p0 = _player_mask(soft_ram_peek(bus.ram, _R_GRP0),
                      soft_ram_peek(bus.ram, _R_REFP0),
                      soft_ram_peek(bus.ram, _R_RESP0),
                      nusiz0)
    p1 = _player_mask(soft_ram_peek(bus.ram, _R_GRP1),
                      soft_ram_peek(bus.ram, _R_REFP1),
                      soft_ram_peek(bus.ram, _R_RESP1),
                      nusiz1)
    # P3g: missiles inherit NUSIZ low 3 bits for multi-copy; the size
    # bits (4-5) come from the same NUSIZ register passed as size_reg.
    m0 = _block_mask(soft_ram_peek(bus.ram, _R_RESM0),
                     nusiz0,
                     soft_ram_peek(bus.ram, _R_ENAM0),
                     nusiz0)
    m1 = _block_mask(soft_ram_peek(bus.ram, _R_RESM1),
                     nusiz1,
                     soft_ram_peek(bus.ram, _R_ENAM1),
                     nusiz1)
    # Ball uses CTRLPF for sizing and does NOT participate in NUSIZ
    # multi-copy (its layout is one solid block).
    bl = _block_mask(soft_ram_peek(bus.ram, _R_RESBL),
                     ctrlpf,
                     soft_ram_peek(bus.ram, _R_ENABL))
    return pf, p0, p1, m0, m1, bl
end

"""
    soft_render_scanline(bus::SoftBus) -> Vector{Float32}

Differentiable single-scanline render — background, playfield, the two
player sprites, the two missiles and the ball. Reads the TIA registers
from `bus.ram[1:64]` and returns a 160-element colour vector.

Two compositing modes, selected by CTRLPF bit 2 (PFP):

  PFP=0 (default):  bg ← pf ← bl ← M1 ← P1 ← M0 ← P0
  PFP=1 (priority): bg ← M1 ← P1 ← M0 ← P0 ← pf ← bl

Both composites are evaluated unconditionally and blended by the
integer-extracted PFP bit. The gradient w.r.t. every colour register
(COLUBK / COLUPF / COLUP0 / COLUP1, hence back to any ROM byte that
wrote them) is exact in both modes. The PFP bit itself is a structural
switch — int extraction breaks the gradient at the cast, matching the
existing convention for enable / size bits in `_block_mask`.
"""
function soft_render_scanline(bus::SoftBus)
    pf, p0, p1, m0, m1, bl = _object_masks(bus)
    colubk = soft_ram_peek(bus.ram, _R_COLUBK)
    colupf = soft_ram_peek(bus.ram, _R_COLUPF)
    colup0 = soft_ram_peek(bus.ram, _R_COLUP0)
    colup1 = soft_ram_peek(bus.ram, _R_COLUP1)
    ctrlpf = soft_ram_peek(bus.ram, _R_CTRLPF)
    pfp_bit = Float32((Int(round(ctrlpf)) >> 2) & 1)

    # Default-priority composite: bg ← pf ← bl ← M1 ← P1 ← M0 ← P0.
    s_def = pf .* colupf .+ (1f0 .- pf) .* colubk
    s_def = (1f0 .- bl) .* s_def .+ bl .* colupf
    s_def = (1f0 .- m1) .* s_def .+ m1 .* colup1
    s_def = (1f0 .- p1) .* s_def .+ p1 .* colup1
    s_def = (1f0 .- m0) .* s_def .+ m0 .* colup0
    s_def = (1f0 .- p0) .* s_def .+ p0 .* colup0

    # PFP composite: bg ← M1 ← P1 ← M0 ← P0 ← pf ← bl.
    s_pfp = fill(colubk, SOFT_SCREEN_WIDTH)
    s_pfp = (1f0 .- m1) .* s_pfp .+ m1 .* colup1
    s_pfp = (1f0 .- p1) .* s_pfp .+ p1 .* colup1
    s_pfp = (1f0 .- m0) .* s_pfp .+ m0 .* colup0
    s_pfp = (1f0 .- p0) .* s_pfp .+ p0 .* colup0
    s_pfp = (1f0 .- pf) .* s_pfp .+ pf .* colupf
    s_pfp = (1f0 .- bl) .* s_pfp .+ bl .* colupf

    return (1f0 .- pfp_bit) .* s_def .+ pfp_bit .* s_pfp
end

"""
    soft_collision_registers(bus::SoftBus) -> Vector{Float32}

The 8 TIA collision latch registers for the current scanline —
indexed CXM0P, CXM1P, CXP0FB, CXP1FB, CXM0FB, CXM1FB, CXBLPF, CXPPMM.
Two objects collide when their masks are both 1.0 at the same pixel;
each register packs its hit flags into D7 / D6. Collisions are a
structural (boolean) property — forward feature parity with the HARD
TIA's P3e latches, not a gradient-flow path.
"""
function soft_collision_registers(bus::SoftBus)
    pf, p0, p1, m0, m1, bl = _object_masks(bus)
    hit(a, b) = sum(a .* b) > 0 ? 1f0 : 0f0
    cxm0p  = hit(m0, p1) * 0x80 + hit(m0, p0) * 0x40
    cxm1p  = hit(m1, p0) * 0x80 + hit(m1, p1) * 0x40
    cxp0fb = hit(p0, pf) * 0x80 + hit(p0, bl) * 0x40
    cxp1fb = hit(p1, pf) * 0x80 + hit(p1, bl) * 0x40
    cxm0fb = hit(m0, pf) * 0x80 + hit(m0, bl) * 0x40
    cxm1fb = hit(m1, pf) * 0x80 + hit(m1, bl) * 0x40
    cxblpf = hit(bl, pf) * 0x80
    cxppmm = hit(p0, p1) * 0x80 + hit(m0, m1) * 0x40
    return [cxm0p, cxm1p, cxp0fb, cxp1fb, cxm0fb, cxm1fb, cxblpf, cxppmm]
end

"""
    soft_render_frame(bus::SoftBus; height=192) -> Matrix{Float32}

Differentiable full-frame render — `height × 160`. P7f-a's playfield
is static for the whole frame, so every row equals the scanline.
"""
function soft_render_frame(bus::SoftBus; height::Integer = SOFT_VISIBLE_SCANLINES)
    scanline = soft_render_scanline(bus)
    return repeat(reshape(scanline, 1, :), height, 1)
end
