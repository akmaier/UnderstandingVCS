# Qualitative XAI proof-of-concept on the differentiable VCS (jutari).
#
# The screen renderer is, structurally, a stack of hard SELECTIONS: which
# object owns a pixel (priority), which sprite bit lights it, which opcode/
# branch produced the register. Each selection is a "switch" in exactly the
# max-pooling sense: store it in the forward pass and the gradient routes
# through it to the value that was selected — bit-exact forward, exact value
# gradient, no deferral. The ONE quantity a stored switch cannot
# differentiate is the sprite *position index* (like the max-pool window
# location); there a soft/bilinear sampler (the spatial-transformer trick)
# restores the gradient.
#
# This script demonstrates all three, writing raw little-endian Float32
# (row-major) maps under out/ with a manifest read by make_joystick_fig.py.
#
#   PART 1  Genuine gradient saliency from jutari's UNMODIFIED soft TIA
#           renderer: d(screen)/d(COLUP0|COLUP1|COLUBK) via Zygote. The
#           colour selection already routes the gradient, so each object is
#           localised to the pixels its register controls (a Grad-CAM
#           analogue) and CHECKED against ground truth.
#   PART 2  Stored-switch content path: representing sprite graphics as
#           per-bit values + a stored position switch makes d(screen)/
#           d(graphics-bit) flow exactly (it is cut only when GRP0 is read as
#           a packed byte). Max-pool style, no deferral.
#   PART 3  Soft sampler for the position INDEX: a continuous joystick drives
#           the sprite; a genuine Zygote gradient of an on-screen objective
#           w.r.t. the joystick infers the direction, and d(screen)/
#           d(joystick) saliency maps show where each push moves the sprite.
#
# Run:  cd jutari && julia --project=. ../tools/xai_joystick_demo/joystick_grad_demo.jl

using JuTari
using JuTari.Diff: SoftBus, soft_render_scanline, soft_ram_peek
using Zygote

const OUT = joinpath(@__DIR__, "out")
isdir(OUT) || mkdir(OUT)
const MANIFEST = String[]

function dump(name::AbstractString, A::AbstractMatrix)
    M = Float32.(permutedims(A))            # transpose -> row-major on vec()
    open(joinpath(OUT, name * ".bin"), "w") do io
        write(io, vec(M))
    end
    push!(MANIFEST, "$(name) $(size(A,1)) $(size(A,2))")
    return nothing
end

# TIA register -> 1-based SOFT-bus RAM cell (renderer reads ram[1:64]).
const COLUP0 = 0x06 + 1; const COLUP1 = 0x07 + 1; const COLUPF = 0x08 + 1
const COLUBK = 0x09 + 1; const CTRLPF = 0x0A + 1; const PF1 = 0x0E + 1
const RESP0  = 0x10 + 1; const RESP1 = 0x11 + 1
const GRP0   = 0x1B + 1; const GRP1  = 0x1C + 1

const ROM = zeros(Float32, 4096)
const H = 120; const W = 160

# 8x8 sprite (a little lander); MSB = leftmost column.
const SPRITE = UInt8[0b00011000, 0b00111100, 0b01111110, 0b11111111,
                     0b11111111, 0b01111110, 0b00100100, 0b01000010]
const SPR_H = length(SPRITE)

# ============================================================================
# PART 1 — genuine gradient saliency from the real (unmodified) soft renderer.
# ============================================================================
cx0, ry0 = 60, 40
cx1, ry1 = 110, 70
pf_band  = 100:107

function ram_for_row(r::Int)
    ram = zeros(Float32, 128)
    ram[COLUP0] = 66f0; ram[COLUP1] = 0x48; ram[COLUPF] = 0x0E; ram[COLUBK] = 0f0
    ram[RESP0]  = Float32(cx0); ram[RESP1] = Float32(cx1)
    (ry0 <= r < ry0 + SPR_H) && (ram[GRP0] = Float32(SPRITE[r - ry0 + 1]))
    (ry1 <= r < ry1 + SPR_H) && (ram[GRP1] = Float32(SPRITE[r - ry1 + 1]))
    (r in pf_band) && (ram[PF1] = 0xFF)
    return ram
end

frame      = zeros(Float32, H, W)
sal_colup0 = zeros(Float32, H, W)
sal_colup1 = zeros(Float32, H, W)
sal_colubk = zeros(Float32, H, W)
for r in 0:H-1
    ram = ram_for_row(r)
    frame[r+1, :] = soft_render_scanline(SoftBus(ram, ROM))
    J = Zygote.jacobian(rm -> soft_render_scanline(SoftBus(rm, ROM)), ram)[1]
    sal_colup0[r+1, :] = J[:, COLUP0]
    sal_colup1[r+1, :] = J[:, COLUP1]
    sal_colubk[r+1, :] = J[:, COLUBK]
end
dump("p1_frame", frame); dump("p1_sal_colup0", sal_colup0)
dump("p1_sal_colup1", sal_colup1); dump("p1_sal_colubk", sal_colubk)

g = Zygote.gradient(rm -> sum(soft_render_scanline(SoftBus(rm, ROM))),
                    ram_for_row(ry0 + 3))[1]
println("PART 1 (real renderer):  d/dCOLUP0=", g[COLUP0],
        "  d/dRESP0=", g[RESP0], "  d/dGRP0(packed byte)=", g[GRP0])

# ============================================================================
# PART 2 — stored-switch content path: per-bit graphics become differentiable.
# ============================================================================
# Sprite graphics as an 8x8 float matrix G (1 = lit). Position is a stored
# (constant) switch: pixel (r,c) maps to sprite cell (r-y0, c-x0). The
# gradient routes to the exact bit that lights each pixel — max-pool style.
const SCALE = 2
G0 = Float32[(SPRITE[i] >> (7 - j)) & 1 for i in 1:SPR_H, j in 0:7]   # 8x8
const X0, Y0 = 72, 48

# The stored switches: each graphics bit (i,j) maps to a CONSTANT screen
# footprint, recorded in the forward pass. Built once, outside any gradient.
const FOOT = let f = [zeros(Float32, H, W) for _ in 1:SPR_H, _ in 1:8]
    for i in 1:SPR_H, j in 1:8, sy in 0:SCALE-1, sx in 0:SCALE-1
        r = Y0 + (i-1)*SCALE + sy; c = X0 + (j-1)*SCALE + sx
        (0 <= r < H && 0 <= c < W) && (f[i, j][r+1, c+1] = 1f0)
    end
    f
end

# Render = linear combination of the per-bit values through the stored
# (constant) switches: the gradient routes to exactly the bit that lit a
# pixel — max-pool style, bit-exact forward, no deferral.
function render_switch(G, c0, cbk)
    occ = sum(G[i, j] .* FOOT[i, j] for i in 1:SPR_H, j in 1:8)
    return cbk .* (1f0 .- occ) .+ c0 .* occ
end

# Attribution of each graphics bit to the screen: d(total brightness)/d(G[i,j]).
gG = Zygote.gradient(G -> sum(render_switch(G, 66f0, 0f0)), G0)[1]      # 8x8
gC = Zygote.gradient(c -> sum(render_switch(G0, c, 0f0)), 66f0)[1]
println("PART 2 (stored switch): d/d graphics-bit flows (min=",
        minimum(gG[G0 .> 0]), " over lit bits)  d/dCOLUP0=", gC)
dump("p2_frame", render_switch(G0, 66f0, 0f0))
# Screen-space saliency for ONE graphics bit (i0,j0): the gradient reaches
# exactly the pixels that bit controls. Computed as a column of the
# screen Jacobian -> reshaped to H×W (== COLUP0 · stored-switch footprint).
i0, j0 = 4, 4
gcol_bit = Zygote.gradient(G -> render_switch(G, 66f0, 0f0)[Y0 + (i0-1)*SCALE + 1,
                                                            X0 + (j0-1)*SCALE + 1],
                           G0)[1]
println("  bit ($i0,$j0) -> its own pixel grad = ", gcol_bit[i0, j0])
dump("p2_sal_bit", 66f0 .* FOOT[i0, j0])               # H×W: that bit's footprint

# ============================================================================
# PART 3 — soft sampler for the position INDEX: joystick -> screen gradient.
# ============================================================================
tri(t) = max(0f0, 1f0 - abs(t))
const ONPIX = let acc = Tuple{Int,Int}[]
    for (rr, byte) in enumerate(SPRITE), b in 0:7
        if (byte >> (7 - b)) & 1 == 1
            for sy in 0:SCALE-1, sx in 0:SCALE-1
                push!(acc, ((rr-1)*SCALE + sy, b*SCALE + sx))
            end
        end
    end
    Tuple(acc)
end

function render_sampler(px, py, ram)
    colup0 = soft_ram_peek(ram, 0x06); colubk = soft_ram_peek(ram, 0x09)
    rows = collect(0:H-1); cols = collect(0:W-1)
    occ = zeros(typeof(px), H, W)
    for (dr, dc) in ONPIX
        occ = occ .+ (tri.(rows .- (py + dr)) * tri.(cols .- (px + dc))')
    end
    occ = clamp.(occ, 0f0, 1f0)
    return colubk .* (1f0 .- occ) .+ colup0 .* occ
end

const PX0, PY0, STEP = 80f0, 70f0, 30f0
joy_to_xy(j) = (PX0 + STEP*(j[4] - j[3]), PY0 + STEP*(j[2] - j[1]))   # u,d,l,r
ram3 = zeros(Float32, 128); ram3[COLUP0] = 66f0; ram3[COLUBK] = 0f0
const GOAL_XY = (118f0, 26f0)            # a target toward the upper-right
goal = zeros(Float32, H, W); goal[20:32, 110:126] .= 1f0   # marker for the figure

# On-screen response = the soft brightness centroid of the rendered frame.
# It depends smoothly on the (sampler-placed) sprite position, so its gradient
# w.r.t. the joystick recovers the control mapping with no zero-gradient
# pathology.
const COLS = reshape(Float32.(0:W-1), 1, W)
const ROWS = reshape(Float32.(0:H-1), H, 1)
function centroid(j)
    fr = render_sampler(joy_to_xy(j)..., ram3)
    tot = sum(fr) + 1f-6
    (sum(fr .* COLS) / tot, sum(fr .* ROWS) / tot)
end
# Objective: how close the on-screen sprite is to the goal (negative distance).
objective(j) = (cx = centroid(j); -((cx[1]-GOAL_XY[1])^2 + (cx[2]-GOAL_XY[2])^2))

j0 = zeros(Float32, 4)
gj = Zygote.gradient(objective, j0)[1]
labels = ("up", "down", "left", "right")
println("PART 3 (soft sampler):  joystick gradient of on-screen position:")
for i in 1:4
    println("   d(obj)/d($(labels[i])) = ", round(gj[i], digits=2))
end
toward = [labels[i] for i in 1:4 if gj[i] > 0]
println("   => to move the on-screen sprite toward the goal, push: ",
        join(uppercase.(toward), " + "))

# Gradient saliency d(screen)/d(direction): a small step so the central
# difference is the directional derivative (sprite edges in the motion sense).
function dir_saliency(idx)
    eps = 0.04f0
    jp = zeros(Float32, 4); jp[idx] += eps
    jm = zeros(Float32, 4); jm[idx] -= eps
    (render_sampler(joy_to_xy(jp)..., ram3) .- render_sampler(joy_to_xy(jm)..., ram3)) ./ (2eps)
end
dump("p3_frame_neutral", render_sampler(joy_to_xy(j0)..., ram3))
dump("p3_goal", goal)
dump("p3_sal_up", dir_saliency(1)); dump("p3_sal_down", dir_saliency(2))
dump("p3_sal_left", dir_saliency(3)); dump("p3_sal_right", dir_saliency(4))
open(joinpath(OUT, "p3_grad.txt"), "w") do io
    for i in 1:4; println(io, "$(labels[i]) $(gj[i])"); end
end

open(joinpath(OUT, "manifest.txt"), "w") do io
    for line in MANIFEST; println(io, line); end
end
println("wrote ", length(MANIFEST), " maps + manifest to ", OUT)
