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
# Scope: P7f-a + P7f-b cover background + playfield + players. Missiles
# + ball (P7f-c), collisions + proper register dispatch (P7f-d) follow.

const SOFT_SCREEN_WIDTH      = 160
const SOFT_VISIBLE_SCANLINES = 192

# TIA register offsets — also the SOFT-bus RAM cells they collapse into.
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
const _R_GRP0   = 0x1B
const _R_GRP1   = 0x1C

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
    soft_render_scanline(bus::SoftBus) -> Vector{Float32}

Differentiable single-scanline render — background, playfield, and the
two player sprites. Reads the TIA registers from `bus.ram[1:64]` and
returns a 160-element colour vector. Compositing order matches the
HARD TIA: background ← playfield ← P1 ← P0.
"""
function soft_render_scanline(bus::SoftBus)
    colubk = soft_ram_peek(bus.ram, _R_COLUBK)
    colupf = soft_ram_peek(bus.ram, _R_COLUPF)
    pf0    = soft_ram_peek(bus.ram, _R_PF0)
    pf1    = soft_ram_peek(bus.ram, _R_PF1)
    pf2    = soft_ram_peek(bus.ram, _R_PF2)
    ctrlpf = soft_ram_peek(bus.ram, _R_CTRLPF)
    pf_mask  = _playfield_mask(pf0, pf1, pf2, ctrlpf)
    scanline = pf_mask .* colupf .+ (1f0 .- pf_mask) .* colubk

    # Player sprites — P1 first, then P0 on top (TIA priority).
    colup0  = soft_ram_peek(bus.ram, _R_COLUP0)
    colup1  = soft_ram_peek(bus.ram, _R_COLUP1)
    p1_mask = _player_mask(soft_ram_peek(bus.ram, _R_GRP1),
                           soft_ram_peek(bus.ram, _R_REFP1),
                           soft_ram_peek(bus.ram, _R_RESP1))
    p0_mask = _player_mask(soft_ram_peek(bus.ram, _R_GRP0),
                           soft_ram_peek(bus.ram, _R_REFP0),
                           soft_ram_peek(bus.ram, _R_RESP0))
    scanline = (1f0 .- p1_mask) .* scanline .+ p1_mask .* colup1
    scanline = (1f0 .- p0_mask) .* scanline .+ p0_mask .* colup0
    return scanline
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
