# Verify the analytic "boots then diverges later" window: clean boot (relax off),
# then relax ON for the rollout. analyze_divergence.jl predicts (alpha>=6):
#   T < 0.1465        -> never diverges (boots forever)
#   0.1465 <= T < 0.166 -> exact for ~64 rollout frames, then diverges (~step 64,
#                          when byte $15AC is first fetched)
#   T >= 0.166        -> diverges almost immediately (kernel byte fails)
# Reports the first STEPPED (post-boot) frame the relaxed rollout deviates.
#
# Run: cd jutari && julia --project=. ../tools/relaxation_study/verify_window.jl [N]

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen
import JuTari.CPU: set_cpu_relax!

rom = read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin"))
N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 200

set_cpu_relax!(on = false)
refenv = StellaEnvironment(rom); env_reset!(refenv; boot_noop_steps = 60, boot_reset_steps = 4)
ref = [(env_step!(refenv, 0); copy(get_screen(refenv))) for _ in 1:N]

function first_div_postboot(A, T)
    set_cpu_relax!(on = false)
    env = StellaEnvironment(rom); env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    set_cpu_relax!(on = true, alpha = A, temperature = T)
    fd = -1; nz = -1
    for f in 1:N
        env_step!(env, 0); s = get_screen(env)
        if s != ref[f] && fd < 0; fd = f; nz = count(!=(0), s); end
    end
    set_cpu_relax!(on = false)
    return fd, nz
end

println("clean boot + relaxed rollout (alpha=20 unless noted). exact nz≈", count(!=(0), ref[100]))
println("  T      | first_div_step | relaxed_nz @div")
for (A, T) in [(20.0,0.1445),(20.0,0.145),(20.0,0.1455),(20.0,0.146),(20.0,0.147),
               (20.0,0.148),(20.0,0.149),(20.0,0.150),(20.0,0.152),(20.0,0.155)]
    fd, nz = first_div_postboot(A, T)
    lbl = A == 20.0 ? "T=$T" : "a=$A,T=$T"
    println("  $lbl\t| ", fd == -1 ? "no div in $N (boots)" : string(fd), "\t| ", fd == -1 ? "—" : nz)
end
