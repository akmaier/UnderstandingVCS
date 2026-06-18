# Real-ROM Space Invaders joystick gradient via the paper's differentiable
# sampler (main paper, "Proof of Concept: Ground-Truth Gradients" + the
# sub-pixel bilinear sampler that "restores the position gradient").
#
# We take the REAL 35 s scene (frame 1050 @ 30 fps — classic colours), extract
# the real player-cannon footprint, and apply the bilinear sampler over its
# horizontal position. A continuous joystick drives the position (joy_x), NOT the
# controller-to-CPU path (which is the discrete index the paper says kills the
# naive gradient). We compute, for the three SOFT variants (SOFT-STE, relaxed
# alpha=6/T=0.14, relaxed alpha=5.5/T=0.145):
#   FORWARD  d screen / d RIGHT     -> the on-screen saliency (cannon edges)
#   NAIVE    d screen / d RIGHT     through the integer index -> 0 (vanishes)
#   INVERSE  d (move-right) / d joy -> recovers "push RIGHT" (up/down vanish)
#
# Reads colours through soft_ram_peek so the relaxation genuinely enters; by
# Theorem 1 the forward is bit-exact, so the three variants coincide. NO changes
# to jutari. Run:
#   cd jutari && julia --project=. ../tools/xai_si_gradient/si_joystick_gradient.jl

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen, get_ram
using JuTari.Diff: soft_ram_peek, set_relax!, using_relax
import Zygote

const NOOP = 0
const CANNON_COLOR = 196          # classic-state player cannon colour (frame 1050)
const SCENE_FRAME = 1050          # 35 s at the video's 30 fps
outdir = joinpath(@__DIR__, "out"); isdir(outdir) || mkpath(outdir)
_cflat(s) = vec(permutedims(s))

# ---- 1. real 35 s scene (frame 1050 @ 30 fps, classic colours) -------------
rom_bytes = read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin"))
e = StellaEnvironment(rom_bytes); env_reset!(e; boot_noop_steps = 60, boot_reset_steps = 4)
for _ in 1:SCENE_FRAME; env_step!(e, NOOP); end
scene = copy(get_screen(e)); H, W = size(scene)
write(joinpath(outdir, "ji_scene.raw"), UInt8.(_cflat(scene)))
write(joinpath(outdir, "ji_scene.shape"), "$H $W")
println("scene frame $SCENE_FRAME (35 s @30fps); size $H x $W; bg=", Int(scene[60, 3]),
        " player-X=", Int(get_ram(e)[29]))

# ---- 2. extract the real cannon footprint (colour 114) ---------------------
canpx = [(r, c) for r in 178:198 for c in 24:62 if Int(scene[r, c]) == CANNON_COLOR]
py0 = minimum(first.(canpx)); px0 = minimum(last.(canpx))
offs = Tuple((r - py0, c - px0) for (r, c) in canpx)
bg_color = Float32(scene[py0 - 2, px0])              # background just above cannon
println("cannon: ", length(canpx), " px  origin (row=$py0,col=$px0)  bg=", Int(bg_color))

# ---- 3. bilinear sampler over the cannon's horizontal position -------------
const R0, R1 = py0 - 3, py0 + 13
const C0, C1 = px0 - 10, px0 + 28          # room to slide right
const ROWS = reshape(Float32.(R0:R1), :, 1)
const COLS = reshape(Float32.(C0:C1), 1, :)
tri(t) = max(0f0, 1f0 - abs(t))
occ(px) = clamp.(sum(tri.(ROWS .- (py0 + dr)) .* tri.(COLS .- (px + dc))
                     for (dr, dc) in offs), 0f0, 1f0)

# colours via soft_ram_peek (relaxation-aware); $06 = cannon, $09 = background
cram = zeros(Float32, 128); cram[0x06 + 1] = Float32(CANNON_COLOR); cram[0x09 + 1] = bg_color
render(px) = (o = occ(px);
              (1f0 .- o) .* soft_ram_peek(cram, 0x09) .+ o .* soft_ram_peek(cram, 0x06))

const STEP = 1f0                            # 1 unit of joystick = 1 px
joy_x(j) = Float32(px0) + STEP * (j[4] - j[3])      # (up,down,left,right): right - left

# FORWARD directional saliency  d render / d RIGHT   (finite diff on the joystick)
function fwd_saliency()
    ε = 0.5f0
    jp = Float32[0, 0, 0, ε]; jm = Float32[0, 0, 0, -ε]
    (render(joy_x(jp)) .- render(joy_x(jm))) ./ (2ε)
end

# NAIVE saliency: the real renderer places the sprite at an INTEGER column;
# d screen / d joystick through round(.) is identically zero (the vanishing).
mask_int(px) = begin
    pc = round(Int, px)
    m = zeros(Float32, length(R0:R1), length(C0:C1))
    for (dr, dc) in offs
        r = (py0 + dr) - R0 + 1; c = (pc + dc) - C0 + 1
        (1 <= r <= size(m, 1) && 1 <= c <= size(m, 2)) && (m[r, c] = 1f0)
    end
    m
end
function naive_saliency()
    ε = 0.5f0
    (Float32(CANNON_COLOR) - bg_color) .*
        (mask_int(joy_x(Float32[0, 0, 0, ε])) .- mask_int(joy_x(Float32[0, 0, 0, -ε]))) ./ (2ε)
end

# INVERSE: push the cannon toward a target to its right; recover the control map
cannon_cx(j) = (o = occ(joy_x(j)); sum(o .* COLS) / (sum(o) + 1f-6))
const TARGET = Float32(px0) + 18f0
objective(j) = -(cannon_cx(j) - TARGET)^2

# ---- 4. three SOFT variants ------------------------------------------------
variants = [("STE", false, 10.0, 0.10),
            ("relax_a6_T0.14", true, 6.0, 0.14),
            ("relax_a5.5_T0.145", true, 5.5, 0.145)]
labels = ("up", "down", "left", "right")
sal_maps = Dict{String,Matrix{Float32}}()
open(joinpath(outdir, "ji_grad.txt"), "w") do io
    for (nm, on, al, T) in variants
        sal, gj = using_relax(on = on, alpha = al, temperature = T) do
            (fwd_saliency(), Zygote.gradient(objective, zeros(Float32, 4))[1])
        end
        sal_maps[nm] = sal
        write(joinpath(outdir, "ji_sal_$(nm).raw"), _cflat(sal))
        println(io, "VARIANT $nm  on=$on alpha=$al T=$T")
        for i in 1:4; println(io, "  d(move_right)/d($(labels[i])) = ", round(gj[i], digits = 4)); end
        println("variant $nm: inverse grad (u,d,l,r) = ", round.(gj, digits = 3))
    end
end
# naive (vanishing) map + crop geometry
write(joinpath(outdir, "ji_naive.raw"), _cflat(naive_saliency()))
write(joinpath(outdir, "ji_crop.txt"), "$R0 $R1 $C0 $C1")

# cross-variant agreement (Theorem 1: forward-exact -> saliency identical)
ks = collect(keys(sal_maps))
maxdiff = maximum(maximum(abs.(sal_maps[ks[1]] .- sal_maps[k])) for k in ks)
println("\nmax |saliency difference| across the 3 variants = ", maxdiff,
        "   (≈0 ⇒ sampler gradient is identical for all three)")
println("forward saliency max |val| = ", round(maximum(abs.(sal_maps[ks[1]])), digits = 3))
