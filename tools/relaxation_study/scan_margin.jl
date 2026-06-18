# Fine margin scan: find an (alpha, T) for the relaxed hard CPU that BOOTS Space
# Invaders cleanly (survives the ~64-frame ALE boot relaxed) but DIVERGES LATER
# (first-divergence frame > 1). Relax is ON throughout (boot included), so
# first_div_frame == 1 means it broke during boot; a larger value means it booted
# and then drifted. N must be large enough to catch late divergence.
#
# Run: cd jutari && julia --project=. ../tools/relaxation_study/scan_margin.jl [N]

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen
import JuTari.CPU: set_cpu_relax!

rom = read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin"))
N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 600

# exact reference frames (relax off) once
set_cpu_relax!(on = false)
refenv = StellaEnvironment(rom); env_reset!(refenv; boot_noop_steps = 60, boot_reset_steps = 4)
ref = Vector{Matrix{UInt8}}(undef, N)
for f in 1:N; env_step!(refenv, 0); ref[f] = copy(get_screen(refenv)); end

function first_div(A, T)
    set_cpu_relax!(on = true, alpha = A, temperature = T)
    env = StellaEnvironment(rom); env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    fd = -1; nz_at = -1; nz_after = -1
    for f in 1:N
        env_step!(env, 0); s = get_screen(env)
        if s != ref[f] && fd < 0
            fd = f; nz_at = count(!=(0), s)
        end
        fd > 0 && f == min(fd + 10, N) && (nz_after = count(!=(0), s))
    end
    set_cpu_relax!(on = false)
    return fd, nz_at, nz_after
end

println("exact nz (frame 30) ≈ ", count(!=(0), ref[30]), "   N=$N")
println("alpha    T    | first_div_frame | relaxed_nz @div / @div+10")
settings = [(20.0,0.135),(20.0,0.138),(20.0,0.140),(20.0,0.141),(20.0,0.142),
            (20.0,0.143),(20.0,0.144),(20.0,0.145),(20.0,0.146),(20.0,0.148),
            (5.5,0.140),(5.5,0.142),(5.5,0.144),
            (5.45,0.10),(5.4,0.10),(5.35,0.10),(5.3,0.10)]
for (A, T) in settings
    fd, nz, nza = first_div(A, T)
    println("  $A\t$T\t|  ", fd == -1 ? "BOOTS (no div in $N)" : string(fd),
            "\t|  ", fd == -1 ? "—" : "$nz / $nza")
end
