# Compute the screen FOOTPRINT of each candidate RAM cell for Pong at the
# analysis state (boot → frame 120, then a 30-frame window). The footprint of a
# cell is the set of screen pixels that change when the cell is perturbed and the
# emulator is re-run bit-exact — i.e. the region of the picture that cell controls.
# This is what lets the website paint a RAM-cell attribution onto the game frame
# (image domain, like Paper 1's joystick-gradient figure).
#
# Output (raw, little-endian, row-major) into docs/assets/methods/:
#   fp_pong_base.raw     210*160 uint8   baseline palette-index screen
#   fp_pong.raw          C*210*160 float32  per-cell footprint intensity (0..1)
#   fp_pong_cells.txt    the C candidate RAM indices, one per line
#
# Run:  julia --project=jutari docs/cell_footprints.jl

include(joinpath(@__DIR__, "..", "tools", "xai_study", "common", "jutari_oracle.jl"))
using .JutariOracle: boot_replay, intervene_ram!, snapshot
using JuTari.Env: env_step!, get_screen, get_ram

const GAME  = "pong"
const CELLS = [13, 14, 45, 46, 49, 50, 51, 54]   # pong candidate RAM cells
const TARGET = 120
const WINDOW = 30
const DELTAS = [-64, -16, 16, 64]
outdir = joinpath(@__DIR__, "assets", "methods")

actions = zeros(Int, TARGET + WINDOW)            # NOOP throughout
ckpt = boot_replay(actions, TARGET; game = GAME)

# baseline per-frame screens over the window
base = deepcopy(ckpt)
base_frames = Vector{Matrix{UInt8}}()
for _ in 1:WINDOW
    env_step!(base, 0); push!(base_frames, Matrix{UInt8}(get_screen(base)))
end
H, W = size(base_frames[end])

# per-cell footprint = how often a pixel changes across the window, unioned over
# a few symmetric perturbations of the cell value.
foot = zeros(Float32, length(CELLS), H, W)
for (ci, cell) in enumerate(CELLS)
    cur = Int(get_ram(ckpt)[cell + 1])
    for d in DELTAS
        env = deepcopy(ckpt)
        intervene_ram!(env, cell, (cur + d) & 0xFF)
        for t in 1:WINDOW
            env_step!(env, 0)
            s = get_screen(env)
            @inbounds for r in 1:H, c in 1:W
                if s[r, c] != base_frames[t][r, c]
                    foot[ci, r, c] += 1f0
                end
            end
        end
    end
end
# normalise each cell's footprint to its own max
for ci in 1:length(CELLS)
    m = maximum(@view foot[ci, :, :])
    m > 0 && (foot[ci, :, :] ./= m)
end

mkpath(outdir)
# baseline screen (row-major)
open(joinpath(outdir, "fp_pong_base.raw"), "w") do io
    write(io, UInt8.(vec(permutedims(base_frames[end]))))
end
# footprints (row-major per cell: C,H,W)
open(joinpath(outdir, "fp_pong.raw"), "w") do io
    for ci in 1:length(CELLS)
        write(io, Float32.(vec(permutedims(foot[ci, :, :]))))
    end
end
open(joinpath(outdir, "fp_pong_cells.txt"), "w") do io
    for c in CELLS; println(io, c); end
end
println("wrote footprints for $(length(CELLS)) cells, $H x $W, to $outdir")
for (ci, cell) in enumerate(CELLS)
    println("  RAM[$cell]: footprint pixels = ", count(>(0f0), @view foot[ci, :, :]))
end
