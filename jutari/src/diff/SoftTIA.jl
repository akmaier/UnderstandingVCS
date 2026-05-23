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

"""
    _player_mask(grp, refp, xpos) -> Vector{Float32}

A single 8-pixel player sprite expanded to a 160-element mask (1.0
where the sprite paints). GRP bit 7 is the leftmost pixel; REFP bit 3
(0x08) reverses the order. The sprite is placed at column `xpos`,
wrapping mod 160.

Built from pure broadcasts + `.+` only — no array literal, typed
comprehension, or `setindex!` — so the surrounding render stays
Zygote-traceable (P7e). `d` is the per-column offset into the sprite;
each of the 8 sprite bits contributes `bit * (d .== k)`.
"""
function _player_mask(grp::Real, refp::Real, xpos::Real)
    grp_i     = trunc(Int, grp)
    reflected = (trunc(Int, refp) & 0x08) != 0
    x         = trunc(Int, xpos)
    d = mod.((0:(SOFT_SCREEN_WIDTH - 1)) .- x, SOFT_SCREEN_WIDTH)   # (160,)
    # sprite bit k (k = 0 is the leftmost pixel)
    spritebit(k) = Float32(reflected ? ((grp_i >> k) & 1) :
                                       ((grp_i >> (7 - k)) & 1))
    return spritebit(0) .* (d .== 0) .+ spritebit(1) .* (d .== 1) .+
           spritebit(2) .* (d .== 2) .+ spritebit(3) .* (d .== 3) .+
           spritebit(4) .* (d .== 4) .+ spritebit(5) .* (d .== 5) .+
           spritebit(6) .* (d .== 6) .+ spritebit(7) .* (d .== 7)
end

"""
    _block_mask(xpos, size_reg, enable_reg) -> Vector{Float32}

A missile / ball — a solid block of `1 << ((size_reg >> 4) & 3)`
pixels (width 1/2/4/8) at column `xpos`, painted only when
`enable_reg` bit 1 is set. Pure broadcasts (`d .< width`) — Zygote-
traceable. Used for missiles (size from NUSIZ, enable from ENAM) and
the ball (size from CTRLPF, enable from ENABL).
"""
function _block_mask(xpos::Real, size_reg::Real, enable_reg::Real)
    size_log2 = (trunc(Int, size_reg) >> 4) & 0x03
    width     = 1 << size_log2                       # 1/2/4/8
    enabled   = Float32((trunc(Int, enable_reg) >> 1) & 1)
    x = trunc(Int, xpos)
    d = mod.((0:(SOFT_SCREEN_WIDTH - 1)) .- x, SOFT_SCREEN_WIDTH)   # (160,)
    return enabled .* Float32.(d .< width)
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
    p0 = _player_mask(soft_ram_peek(bus.ram, _R_GRP0),
                      soft_ram_peek(bus.ram, _R_REFP0),
                      soft_ram_peek(bus.ram, _R_RESP0))
    p1 = _player_mask(soft_ram_peek(bus.ram, _R_GRP1),
                      soft_ram_peek(bus.ram, _R_REFP1),
                      soft_ram_peek(bus.ram, _R_RESP1))
    m0 = _block_mask(soft_ram_peek(bus.ram, _R_RESM0),
                     soft_ram_peek(bus.ram, _R_NUSIZ0),
                     soft_ram_peek(bus.ram, _R_ENAM0))
    m1 = _block_mask(soft_ram_peek(bus.ram, _R_RESM1),
                     soft_ram_peek(bus.ram, _R_NUSIZ1),
                     soft_ram_peek(bus.ram, _R_ENAM1))
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
