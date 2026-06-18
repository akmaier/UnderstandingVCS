# Long, fine, low-T search for a DELAYED divergence (clean boot, then relax on
# for the rollout). The empirical run includes ALL effects -- crucially the RAM
# reads, whose neighbour-pull evolves with game state, so a read can be exact for
# many frames and only cross 1/2 once the RAM reaches a high-contrast layout.
# If first_div_step lands at a moderate/large frame for some T, "boots then
# diverges later" exists.
#
# Run: cd jutari && julia --project=. ../tools/relaxation_study/long_scan.jl [N]

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen
import JuTari.CPU: set_cpu_relax!

rom = read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin"))
N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3000

set_cpu_relax!(on = false)
refenv = StellaEnvironment(rom); env_reset!(refenv; boot_noop_steps = 60, boot_reset_steps = 4)
ref = [(env_step!(refenv, 0); copy(get_screen(refenv))) for _ in 1:N]

function first_div(A, T)
    set_cpu_relax!(on = false)
    env = StellaEnvironment(rom); env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    set_cpu_relax!(on = true, alpha = A, temperature = T)
    fd = -1
    for f in 1:N
        env_step!(env, 0)
        if get_screen(env) != ref[f] && fd < 0; fd = f; break; end
    end
    set_cpu_relax!(on = false)
    return fd
end

println("clean boot + relaxed rollout, alpha=20, N=$N  (looking for moderate first_div_step)")
for T in (0.138, 0.140, 0.142, 0.143, 0.1435, 0.144, 0.1442, 0.1444, 0.1446, 0.1448)
    fd = first_div(20.0, T)
    println("  T=$T\t-> first_div_step = ", fd == -1 ? "none ($N frames)" : string(fd))
end
