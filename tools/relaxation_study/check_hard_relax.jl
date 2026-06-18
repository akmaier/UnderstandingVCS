# Functional check of the default-off hard-CPU relaxation hook: relax-off is
# deterministic; relax-on (alpha=5, T=0.15) produces a DIVERGED beam-timed frame
# on Space Invaders (the basis for the divergence video). Not a conformance gate
# (that's the 64/64 sweeps) -- just confirms the ON path does something real.
#
# Run: cd jutari && julia --project=. ../tools/relaxation_study/check_hard_relax.jl

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen
import JuTari.CPU: set_cpu_relax!

rom = read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin"))

function run_frames(n; relax::Bool, a = 5.0, T = 0.15)
    set_cpu_relax!(on = false)
    env = StellaEnvironment(rom)
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    relax && set_cpu_relax!(on = true, alpha = a, temperature = T)
    for _ in 1:n
        env_step!(env, 0)                       # action 0 = NOOP
    end
    s = copy(get_screen(env))
    set_cpu_relax!(on = false)
    return s
end

N = 30
off1 = run_frames(N; relax = false)
off2 = run_frames(N; relax = false)
on   = run_frames(N; relax = true)

println("relax-off deterministic           : ", off1 == off2)
println("relax-on differs from relax-off   : ", off1 != on,
        "   (", count(off1 .!= on), " / ", length(on), " px differ)")
println("relax-on frame is a valid screen  : ", size(on), "  nonzero px=", count(!=(0), on))
