# "Boots then diverges later": boot Space Invaders with relax OFF (clean, exact
# boot to the running game), THEN enable the relaxation for the rollout frames.
# This is the realistic usage (boot to a state exactly, then run the relaxed /
# differentiable rollout) and -- unlike relax-during-boot -- it can diverge K>1
# frames into the rollout. Report the first frame the relaxed rollout deviates
# from the exact one, and whether the relaxed screen is still recognizable.
#
# Run: cd jutari && julia --project=. ../tools/relaxation_study/scan_margin2.jl [N]

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen
import JuTari.CPU: set_cpu_relax!

rom = read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin"))
N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 600

set_cpu_relax!(on = false)
refenv = StellaEnvironment(rom); env_reset!(refenv; boot_noop_steps = 60, boot_reset_steps = 4)
ref = Vector{Matrix{UInt8}}(undef, N)
for f in 1:N; env_step!(refenv, 0); ref[f] = copy(get_screen(refenv)); end

function first_div_postboot(A, T)
    set_cpu_relax!(on = false)
    env = StellaEnvironment(rom); env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    set_cpu_relax!(on = true, alpha = A, temperature = T)     # relax ON only for the rollout
    fd = -1; nz = -1; nza = -1
    for f in 1:N
        env_step!(env, 0); s = get_screen(env)
        if s != ref[f] && fd < 0
            fd = f; nz = count(!=(0), s)
        end
        fd > 0 && f == min(fd + 20, N) && (nza = count(!=(0), s))
    end
    set_cpu_relax!(on = false)
    return fd, nz, nza
end

println("clean boot + relaxed rollout. exact nz(30)≈", count(!=(0), ref[30]), "  N=$N")
println("alpha   T    | first_div_frame | relaxed_nz @div / @div+20")
for (A, T) in [(20.0,0.145),(20.0,0.146),(20.0,0.147),(20.0,0.148),(20.0,0.15),
               (5.45,0.10),(5.4,0.10),(5.3,0.10),(5.0,0.13),(5.0,0.15),(5.5,0.146)]
    fd, nz, nza = first_div_postboot(A, T)
    println("  $A\t$T\t|  ", fd == -1 ? "no div in $N" : string(fd),
            "\t|  ", fd == -1 ? "—" : "$nz / $nza")
end
