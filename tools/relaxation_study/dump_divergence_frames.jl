# Dump beam-timed Space Invaders frames for the hard/soft-exact/relaxed
# divergence video (4-B). Runs the cycle-accurate HARD Console twice:
#   * relax OFF  -> EXACT stream  (= HARD = SOFT-exact, Theorem 1)
#   * relax ON   -> RELAXED stream (fully-relaxed at alpha, T)
# and writes each as a flat uint8 (n, h, 160) C-order .raw + a .shape sidecar,
# matching tools/breakout_video so the compositor can ingest them.
#
# Run: cd jutari && julia --project=. ../tools/relaxation_study/dump_divergence_frames.jl [N alpha T]

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen
import JuTari.CPU: set_cpu_relax!

N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 180
A = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 5.5
T = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.15

rom = read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin"))
outdir = joinpath(@__DIR__, "video_out"); isdir(outdir) || mkpath(outdir)

# C-order (h, w) flatten so numpy fromfile().reshape(n, h, 160) is correct.
_cflat(s) = vec(permutedims(s))

function dump(label::String; relax::Bool)
    set_cpu_relax!(on = false)
    env = StellaEnvironment(rom)
    relax && set_cpu_relax!(on = true, alpha = A, temperature = T)
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    buf = UInt8[]; h = 0
    for _ in 1:N
        env_step!(env, 0)
        s = get_screen(env); h = size(s, 1)
        append!(buf, _cflat(s))
    end
    set_cpu_relax!(on = false)
    write(joinpath(outdir, "$(label)_frames.raw"), buf)
    write(joinpath(outdir, "$(label)_frames.shape"), "$N $h 160")
    println("wrote $(label)_frames.raw  ($N x $h x 160)  nz@last=", count(!=(0), get_screen(env)))
    return h
end

println("Space Invaders divergence dump: N=$N  relaxed alpha=$A T=$T")
dump("exact"; relax = false)
dump("relaxed"; relax = true)
