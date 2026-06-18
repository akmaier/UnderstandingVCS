# Scan (alpha,T) for the relaxed hard CPU on Space Invaders: for each setting,
# the first frame the relaxed screen differs from the exact hard run, and whether
# the relaxed screen is still DISPLAYING (nonzero) at a few checkpoints vs going
# black. Goal: find a setting that boots into a recognizable screen and THEN
# glitches (good for the divergence video), between "exact" and "black-at-boot".
#
# Run: cd jutari && julia --project=. ../tools/relaxation_study/scan_relax_boot.jl

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen
import JuTari.CPU: set_cpu_relax!

rom = read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin"))
const N = 90

# exact reference frames (relax off), captured once
set_cpu_relax!(on = false)
refenv = StellaEnvironment(rom); env_reset!(refenv; boot_noop_steps = 60, boot_reset_steps = 4)
ref = Vector{Matrix{UInt8}}(undef, N)
for f in 1:N; env_step!(refenv, 0); ref[f] = copy(get_screen(refenv)); end

function probe(A, T)
    set_cpu_relax!(on = true, alpha = A, temperature = T)
    env = StellaEnvironment(rom); env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    firstdiff = -1; nz = Dict{Int,Int}()
    for f in 1:N
        env_step!(env, 0); s = get_screen(env)
        (s != ref[f] && firstdiff < 0) && (firstdiff = f)
        f in (10, 30, 60, 90) && (nz[f] = count(!=(0), s))
    end
    set_cpu_relax!(on = false)
    return firstdiff, nz
end

println("alpha   T   | first_diff | on_nz @10/30/60/90  (off_nz ~", count(!=(0), ref[30]), ")")
for (A, T) in [(5.0,0.13),(5.25,0.13),(5.25,0.14),(5.5,0.14),(5.5,0.15),
               (6.0,0.14),(6.0,0.15),(5.0,0.14)]
    fd, nz = probe(A, T)
    println("  $A  $T  |  ", fd == -1 ? "exact" : string(fd),
            "  |  ", get(nz,10,-1), "/", get(nz,30,-1), "/", get(nz,60,-1), "/", get(nz,90,-1))
end
