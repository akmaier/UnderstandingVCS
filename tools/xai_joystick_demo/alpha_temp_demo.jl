# Supplementary exploration (jutari only): how the relaxation parameters
# shape the gradient maps. We render a single player cannon whose horizontal
# position is produced by jutari's actual soft relaxations, and look at the
# screen-space directional-derivative saliency d(screen)/d(control):
#
#   * soft branch sharpness alpha  (jutari `soft_branch`, gate g = sigma(alpha z))
#     The cannon position is a gated blend of a LEFT and a RIGHT placement.
#     Small alpha: a soft 50/50 blend (two faint cannons), broad gradient.
#     Large alpha: the gate snaps to one side, gradient concentrates and then
#     vanishes away from the decision (the hard / straight-through limit).
#
#   * soft select temperature T  (jutari `soft_select`, weights softmax(l/T))
#     The cannon position is a softmax mixture over K candidate columns.
#     High T: the cannon is smeared over many columns, gradient spread wide.
#     Low T: the mixture collapses to one column, gradient localises.
#
# This is qualitative — it shows the EFFECT OF THE VALUES on the gradient, not
# a full game trace. Run:
#   cd jutari && julia --project=. ../tools/xai_joystick_demo/alpha_temp_demo.jl

using JuTari
using JuTari.Diff: soft_select, soft_branch

const OUT = joinpath(@__DIR__, "out")
isdir(OUT) || mkdir(OUT)
const MANIFEST = String[]
function dump(name::AbstractString, A::AbstractMatrix)
    open(joinpath(OUT, name * ".bin"), "w") do io
        write(io, vec(Float32.(permutedims(A))))      # row-major for numpy
    end
    push!(MANIFEST, "$(name) $(size(A,1)) $(size(A,2))")
end

const H, W, VS = 80, 140, 2
const CANNON = UInt8[0b00010000, 0b00010000, 0b00111000, 0b01111100,
                     0b01111100, 0b11111110, 0b11111110, 0b11111110]

# Hard cannon occupancy (8 wide x 16 tall) with its left edge at column `x`.
function cannon_occ(x::Int; yc::Int = 30)
    occ = zeros(Float32, H, W)
    for (rr, byte) in enumerate(CANNON), b in 0:7
        if (byte >> (7 - b)) & 1 == 1
            for sy in 0:VS-1
                r = yc + (rr-1)*VS + sy; c = x + b
                (1 <= r <= H && 1 <= c <= W) && (occ[r, c] = 1f0)
            end
        end
    end
    occ
end

# ----------------------------------------------------------------------------
# soft branch (sharpness alpha): gated blend of a left/right cannon placement.
# ----------------------------------------------------------------------------
const OCC_L = cannon_occ(45)
const OCC_R = cannon_occ(95)
render_branch(z::Real, alpha::Real) =
    (g = soft_branch(z, 0f0, 1f0; alpha = alpha); (1f0 - g) .* OCC_L .+ g .* OCC_R)

function grad_branch(z::Real, alpha::Real; eps = 0.02f0)
    (render_branch(z + eps, alpha) .- render_branch(z - eps, alpha)) ./ (2eps)
end

const Z0 = 0.25f0      # slightly toward "branch taken" so alpha shifts the blend
for (i, a) in enumerate((2f0, 6f0, 20f0))
    dump("at_branch_scene$i", render_branch(Z0, a))
    dump("at_branch_grad$i",  grad_branch(Z0, a))
    g = 1.0f0 / (1.0f0 + exp(-a * Z0))
    println("soft_branch alpha=$a: gate g(z=$Z0)=", round(g, digits=3))
end

# ----------------------------------------------------------------------------
# soft select (temperature T): softmax mixture over K candidate columns.
# ----------------------------------------------------------------------------
const CAND = collect(30:8:110)               # K candidate cannon columns
const K = length(CAND)
const V = let m = zeros(Float32, K, H*W)
    for (k, xc) in enumerate(CAND); m[k, :] = vec(cannon_occ(xc)); end
    m
end
function render_select(target::Real, T::Real)
    logits = Float32[-((xc - target) / 6f0)^2 for xc in CAND]
    reshape(soft_select(logits, V; temperature = T), H, W)
end
function grad_select(target::Real, T::Real; eps = 0.5f0)
    (render_select(target + eps, T) .- render_select(target - eps, T)) ./ (2eps)
end

const TARGET0 = 70f0
for (i, T) in enumerate((2f0, 0.5f0, 0.1f0))
    dump("at_select_scene$i", render_select(TARGET0, T))
    dump("at_select_grad$i",  grad_select(TARGET0, T))
    w = let l = Float32[-((xc - TARGET0)/6f0)^2 for xc in CAND]
        e = exp.((l .- maximum(l)) ./ T); e ./ sum(e)
    end
    println("soft_select T=$T: max weight=", round(maximum(w), digits=3),
            "  effective #cols=", round(1 / sum(w.^2), digits=1))
end

# Per-candidate HARD cannon occupancies (one image per candidate column), so the
# pixel-space sampled heatmap (make_temp_heatmap_fig.py) can draw sprite positions
# ~ softmax(logits/T), render each, and average them in the screen domain.
for (k, xc) in enumerate(CAND)
    dump("at_cand_occ$k", cannon_occ(xc))
end

open(joinpath(OUT, "alpha_temp_manifest.txt"), "w") do io
    for line in MANIFEST; println(io, line); end
end
println("wrote ", length(MANIFEST), " maps to ", OUT)
