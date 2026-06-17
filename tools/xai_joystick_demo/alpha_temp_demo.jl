# Supplementary exploration (jutari only): how the soft-branch sharpness alpha
# shapes the gradient map. We render a single target sprite (an invader) whose
# horizontal placement is produced by jutari's actual `soft_branch` relaxation,
# a gate g = sigma(alpha z) blending a LEFT and a RIGHT placement, and look at
# the screen-space directional-derivative saliency d(screen)/d(control):
#
#   small alpha: a soft 50/50 blend (two faint sprites), broad weak gradient;
#   moderate alpha: the gradient is a sharp dipole between the two placements;
#   large alpha: the gate snaps to one side -> a single sprite and the gradient
#                vanishes (the hard / straight-through limit).
#
# The soft-SELECT (temperature T) effect is shown separately, in pixel space, by
# make_temp_heatmap_fig.py (a sampled sprite-occupancy heatmap), so it is not
# duplicated here. Run:
#   cd jutari && julia --project=. ../tools/xai_joystick_demo/alpha_temp_demo.jl

using JuTari
using JuTari.Diff: soft_branch

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
# 8-wide "invader" target sprite (the object being positioned is a target).
const INVADER = UInt8[0b00011000, 0b00111100, 0b01111110, 0b11011011,
                      0b11111111, 0b10100101, 0b00100100, 0b01000010]

# Hard sprite occupancy (8 wide x 16 tall) with its left edge at column `x`.
function sprite_occ(x::Int; yc::Int = 30)
    occ = zeros(Float32, H, W)
    for (rr, byte) in enumerate(INVADER), b in 0:7
        if (byte >> (7 - b)) & 1 == 1
            for sy in 0:VS-1
                r = yc + (rr-1)*VS + sy; c = x + b
                (1 <= r <= H && 1 <= c <= W) && (occ[r, c] = 1f0)
            end
        end
    end
    occ
end

# soft branch (sharpness alpha): gated blend of a LEFT/RIGHT placement, kept
# close together so the dipole reads as one object shifting, not two far apart.
const OCC_L = sprite_occ(56)
const OCC_R = sprite_occ(84)
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

open(joinpath(OUT, "alpha_temp_manifest.txt"), "w") do io
    for line in MANIFEST; println(io, line); end
end
println("wrote ", length(MANIFEST), " maps to ", OUT)
