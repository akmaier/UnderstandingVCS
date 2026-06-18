# Boot Space Invaders with the relaxed hard CPU at (alpha, T) and compare,
# frame by frame, to the exact (relax-off) hard run: when does the relaxed
# screen first differ, and does it stay DISPLAYING (nonzero) or collapse to
# black? Used to pick a setting that boots to a recognizable screen before it
# glitches (richer for the divergence video than a boot-time black-out).
#
# Run: cd jutari && julia --project=. ../tools/relaxation_study/check_hard_relax_boot.jl [alpha T N]

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen
import JuTari.CPU: set_cpu_relax!

rom = read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin"))
A = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 5.5
T = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.13
N = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 150

set_cpu_relax!(on = false)
env_off = StellaEnvironment(rom); env_reset!(env_off; boot_noop_steps = 60, boot_reset_steps = 4)
set_cpu_relax!(on = true, alpha = A, temperature = T)
env_on = StellaEnvironment(rom); env_reset!(env_on; boot_noop_steps = 60, boot_reset_steps = 4)
set_cpu_relax!(on = false)

println("Space Invaders relaxed boot: alpha=$A T=$T  (frame | diff_px | on_nz | off_nz)")
firstdiff = -1
for f in 1:N
    set_cpu_relax!(on = false); env_step!(env_off, 0); s_off = get_screen(env_off)
    set_cpu_relax!(on = true, alpha = A, temperature = T); env_step!(env_on, 0); s_on = get_screen(env_on)
    set_cpu_relax!(on = false)
    d = count(s_off .!= s_on)
    (d > 0 && firstdiff < 0) && (global firstdiff = f)
    if f in (1, 5, 10, 20, 30, 45, 60, 90, 120, 150)
        println("  $f\t$d\t", count(!=(0), s_on), "\t", count(!=(0), s_off))
    end
end
println("first differing frame: ", firstdiff == -1 ? "none (identical for $N frames)" : firstdiff)
