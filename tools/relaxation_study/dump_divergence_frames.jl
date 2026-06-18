# Dump beam-timed Space Invaders frames for the hard/soft-STE/soft divergence
# video (4-B). Runs the cycle-accurate HARD Console once relax-OFF (EXACT = HARD
# = SOFT-STE, Theorem 1) and once per relaxed (alpha,T) setting, writing each as
# a flat uint8 (n,h,160) C-order .raw + .shape, plus a manifest the compositor
# reads. Relax is on throughout (boot included) for the soft streams.
#
# Run: cd jutari && julia --project=. ../tools/relaxation_study/dump_divergence_frames.jl [N a1 T1 a2 T2 ...]
#   default: N=1800, settings (6.0,0.14) and (5.5,0.145)

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen
import JuTari.CPU: set_cpu_relax!

args = ARGS
N = length(args) >= 1 ? parse(Int, args[1]) : 1800
function parse_settings(a)
    s = Tuple{Float64,Float64}[]
    i = 2
    while i + 1 <= length(a)
        push!(s, (parse(Float64, a[i]), parse(Float64, a[i+1]))); i += 2
    end
    return s
end
settings = parse_settings(args)
isempty(settings) && (settings = [(6.0, 0.14), (5.5, 0.145)])

rom = read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin"))
outdir = joinpath(@__DIR__, "video_out"); isdir(outdir) || mkpath(outdir)
_cflat(s) = vec(permutedims(s))               # C-order (h,w) flatten for numpy

function dump(label; relax::Bool, A = 0.0, T = 0.0)
    set_cpu_relax!(on = false)
    env = StellaEnvironment(rom)
    relax && set_cpu_relax!(on = true, alpha = A, temperature = T)
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    buf = UInt8[]; h = 0
    for _ in 1:N
        env_step!(env, 0); s = get_screen(env); h = size(s, 1); append!(buf, _cflat(s))
    end
    set_cpu_relax!(on = false)
    write(joinpath(outdir, "$(label)_frames.raw"), buf)
    write(joinpath(outdir, "$(label)_frames.shape"), "$N $h 160")
    println("wrote $(label)_frames.raw  ($N x $h x 160)  nz@last=", count(!=(0), get_screen(env)))
end

println("Divergence dump: N=$N  settings=", settings)
dump("exact"; relax = false)
open(joinpath(outdir, "manifest.txt"), "w") do io
    for (i, (A, T)) in enumerate(settings)
        dump("soft$(i-1)"; relax = true, A = A, T = T)
        println(io, "soft$(i-1) $A $T")
    end
end
