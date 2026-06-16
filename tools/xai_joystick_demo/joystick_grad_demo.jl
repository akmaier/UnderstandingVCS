# Qualitative XAI proof-of-concept on the differentiable VCS (jutari).
#
# Scene: a Space-Invaders-style layout rendered by jutari's differentiable
# soft TIA renderer -- a row of invaders (player 1, multi-copied via NUSIZ) and
# a player cannon (player 0) at the bottom that moves left/right.
#
# The renderer is a stack of hard SELECTIONS: which object owns a pixel
# (priority), which sprite bit lights it. Each is a "switch" in the
# max-pooling sense: store it in the forward pass and the gradient routes
# through it to the selected value -- bit-exact forward, exact value gradient,
# no deferral. The one quantity a stored switch cannot differentiate is the
# sprite position INDEX (like the max-pool window location); there a soft
# (bilinear) sampler restores the gradient.
#
#   PART 1  Genuine gradient saliency from the UNMODIFIED soft renderer:
#           d(screen)/d(COLUP0|COLUP1|COLUBK) localises the cannon, the
#           invaders, and the background to exactly the pixels each register
#           controls (a Grad-CAM analogue), checkable against ground truth.
#   PART 2  Stored-switch content: a single cannon graphics bit becomes
#           differentiable once represented as a value behind its switch.
#   PART 3  Soft sampler for the position index: the cannon is moved by a
#           continuous joystick; a Zygote gradient of the on-screen cannon
#           position w.r.t. the joystick recovers the control mapping (the
#           cannon moves left/right) and points toward a target invader.
#
# Run:  cd jutari && julia --project=. ../tools/xai_joystick_demo/joystick_grad_demo.jl

using JuTari
using JuTari.Diff: SoftBus, soft_render_scanline, soft_ram_peek
using Zygote

const OUT = joinpath(@__DIR__, "out")
isdir(OUT) || mkdir(OUT)
const MANIFEST = String[]
function dump(name::AbstractString, A::AbstractMatrix)
    open(joinpath(OUT, name * ".bin"), "w") do io
        write(io, vec(Float32.(permutedims(A))))      # row-major for numpy
    end
    push!(MANIFEST, "$(name) $(size(A,1)) $(size(A,2))")
end

# TIA register -> 1-based SOFT-bus RAM cell (renderer reads ram[1:64]).
const NUSIZ1 = 0x05 + 1
const COLUP0 = 0x06 + 1; const COLUP1 = 0x07 + 1; const COLUBK = 0x09 + 1
const RESP0  = 0x10 + 1; const RESP1 = 0x11 + 1
const GRP0   = 0x1B + 1; const GRP1  = 0x1C + 1

const ROM = zeros(Float32, 4096)
const H = 120; const W = 160

# Recognisable 8x8 sprites (MSB = leftmost column).
const INVADER = UInt8[0b00011000, 0b00111100, 0b01111110, 0b11011011,
                      0b11111111, 0b00100100, 0b01011010, 0b10100101]
const CANNON  = UInt8[0b00010000, 0b00010000, 0b00111000, 0b01111100,
                      0b01111100, 0b11111110, 0b11111110, 0b11111110]
const VS = 2                                # vertical scale (sprites are 16 tall)

# Layout: invader row near the top, cannon near the bottom.
const INV_TOP, CAN_TOP = 16, 92
const INV_X = 28                            # leftmost invader (NUSIZ -> 3 copies)
const NUSIZ_3COPY = 0b110                   # offsets (0,32,64): invaders at 28,60,92
const INV_COLS = (INV_X, INV_X + 32, INV_X + 64)
const CAN_X0 = 50                           # cannon start column

# ============================================================================
# PART 1 — genuine gradient saliency from the unmodified soft renderer.
# ============================================================================
function ram_for_row(r::Int)
    ram = zeros(Float32, 128)
    ram[COLUP0] = 0x3C; ram[COLUP1] = 0x86; ram[COLUBK] = 0f0
    ram[RESP0] = Float32(CAN_X0); ram[RESP1] = Float32(INV_X)
    ram[NUSIZ1] = Float32(NUSIZ_3COPY)
    if INV_TOP <= r < INV_TOP + VS*8
        ram[GRP1] = Float32(INVADER[(r - INV_TOP) ÷ VS + 1])
    end
    if CAN_TOP <= r < CAN_TOP + VS*8
        ram[GRP0] = Float32(CANNON[(r - CAN_TOP) ÷ VS + 1])
    end
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
                    ram_for_row(CAN_TOP + 6))[1]
println("PART 1: d/dCOLUP0=", g[COLUP0], "  d/dRESP0=", g[RESP0],
        "  d/dGRP0(packed)=", g[GRP0])

# ============================================================================
# PART 2 — stored-switch content: per-bit cannon graphics become differentiable.
# ============================================================================
const SCALE = 2                # vertical only; horizontal stays 1x like Part 1
const CX0, CY0 = CAN_X0, CAN_TOP
const FOOT = let f = [zeros(Float32, H, W) for _ in 1:8, _ in 1:8]
    for i in 1:8, j in 1:8, sy in 0:SCALE-1
        r = CY0 + (i-1)*SCALE + sy; c = CX0 + (j-1)
        (0 <= r < H && 0 <= c < W) && (f[i, j][r+1, c+1] = 1f0)
    end
    f
end
G0 = Float32[(CANNON[i] >> (7 - j)) & 1 for i in 1:8, j in 0:7]
render_switch(G, c0, cbk) =
    (occ = sum(G[i, j] .* FOOT[i, j] for i in 1:8, j in 1:8);
     cbk .* (1f0 .- occ) .+ c0 .* occ)
gG = Zygote.gradient(G -> sum(render_switch(G, 60f0, 0f0)), G0)[1]
println("PART 2: d/d graphics-bit flows (min over lit bits = ",
        minimum(gG[G0 .> 0]), ")")
i0, j0 = 6, 4
dump("p2_sal_bit", 60f0 .* FOOT[i0, j0])
# cannon SHAPE (lit bits only, not the full 8x8 grid) for the figure backdrop
dump("p2_cannon",  60f0 .* sum(G0[i, j] .* FOOT[i, j] for i in 1:8, j in 1:8))

# ============================================================================
# PART 3 — soft sampler: a continuous joystick moves the cannon left/right.
# ============================================================================
tri(t) = max(0f0, 1f0 - abs(t))
const CAN_ONPIX = let acc = Tuple{Int,Int}[]
    for (rr, byte) in enumerate(CANNON), b in 0:7
        if (byte >> (7 - b)) & 1 == 1
            for sy in 0:VS-1               # vertical 2x; horizontal 1x (8 wide)
                push!(acc, ((rr-1)*VS + sy, b))
            end
        end
    end
    Tuple(acc)
end
# Static invaders (context, drawn through constant switches; do not move).
const INV_FOOT = let m = zeros(Float32, H, W)
    for cx in INV_COLS, (rr, byte) in enumerate(INVADER), b in 0:7
        if (byte >> (7 - b)) & 1 == 1
            for sy in 0:VS-1
                r = INV_TOP + (rr-1)*VS + sy; c = cx + b
                (0 <= r < H && 0 <= c < W) && (m[r+1, c+1] = 1f0)
            end
        end
    end
    m
end

ram3 = zeros(Float32, 128); ram3[COLUP0] = 0x3C; ram3[COLUP1] = 0x86; ram3[COLUBK] = 0f0
const ROWS = reshape(Float32.(0:H-1), H, 1); const COLS = reshape(Float32.(0:W-1), 1, W)

cannon_occ(px, py) = clamp.(sum((tri.(ROWS .- (py + dr))) .* (tri.(COLS .- (px + dc)))
                               for (dr, dc) in CAN_ONPIX), 0f0, 1f0)
function render_sampler(px, py)
    c0 = soft_ram_peek(ram3, 0x06); c1 = soft_ram_peek(ram3, 0x07)
    cbk = soft_ram_peek(ram3, 0x09)
    occ = cannon_occ(px, py)
    base = cbk .* (1f0 .- INV_FOOT) .+ c1 .* INV_FOOT       # static invaders
    return (1f0 .- occ) .* base .+ occ .* c0                # cannon on top
end

const CAN_PX0, CAN_PY = Float32(CX0), Float32(CAN_TOP)
const STEP = 26f0
const TARGET_X = Float32(INV_COLS[3])      # align the cannon under the right invader
joy_x(j) = CAN_PX0 + STEP*(j[4] - j[3])    # only left/right move the cannon
function cannon_cx(j)
    occ = cannon_occ(joy_x(j), CAN_PY)
    sum(occ .* COLS) / (sum(occ) + 1f-6)
end
objective(j) = -(cannon_cx(j) - TARGET_X)^2

j0 = zeros(Float32, 4)
gj = Zygote.gradient(objective, j0)[1]
labels = ("up", "down", "left", "right")
println("PART 3: joystick gradient of on-screen cannon position:")
for i in 1:4; println("   d(obj)/d($(labels[i])) = ", round(gj[i], digits=2)); end
println("   => to move the cannon under the target invader, push: ",
        join(uppercase.([labels[i] for i in 1:4 if gj[i] > 1f0]), " + "))

function dir_saliency(idx)
    eps = 0.04f0
    jp = zeros(Float32, 4); jp[idx] += eps; jm = zeros(Float32, 4); jm[idx] -= eps
    (render_sampler(joy_x(jp), CAN_PY) .- render_sampler(joy_x(jm), CAN_PY)) ./ (2eps)
end
# goal marker: the target invader's footprint (for the figure)
goal = zeros(Float32, H, W)
for (rr, byte) in enumerate(INVADER), b in 0:7
    if (byte >> (7 - b)) & 1 == 1
        for sy in 0:VS-1
            r = INV_TOP + (rr-1)*VS + sy; c = INV_COLS[3] + b
            (0 <= r < H && 0 <= c < W) && (goal[r+1, c+1] = 1f0)
        end
    end
end
dump("p3_frame_neutral", render_sampler(joy_x(j0), CAN_PY))
dump("p3_cannon", cannon_occ(joy_x(j0), CAN_PY))    # cannon mask for the figure
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
