# Phase 0a (HARD grounding) for the real-ROM Space Invaders joystick-gradient
# study. Establishes, with NO changes to jutari:
#   (1) when the SI game field is live (screen evolution vs wall-clock seconds),
#   (2) the player-ship X position RAM byte, found empirically by diffing a
#       RIGHT-held vs NOOP-held trajectory from the ~35 s scene,
#   (3) that holding RIGHT actually moves that byte (game is in play, not attract).
#
# Run: cd jutari && julia --project=. ../tools/xai_si_gradient/probe_hard.jl

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen, get_ram,
                  frame_number

const NOOP = 0
const RIGHT = 3
const LEFT = 4

rom = read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin"))
fresh() = (e = StellaEnvironment(rom); env_reset!(e; boot_noop_steps = 60,
                                                  boot_reset_steps = 4); e)

# (1) screen evolution under NOOP --------------------------------------------
println("=== screen evolution (NOOP), 60 fps ===")
let env = fresh()
    for f in 1:2200
        env_step!(env, NOOP)
        if f in (35, 60, 120, 300) || f % 300 == 0
            s = get_screen(env)
            println("  frame $f  t=$(round(f / 60, digits = 2))s  nz_px=",
                    count(!=(0), s), "  game_frame=", frame_number(env))
        end
    end
end

# (2)+(3) player-X discovery at the 35 s scene -------------------------------
# Roll NOOP to `nstart`, then hold `action` for `nhold` frames, recording RAM.
function ram_traj(nstart, nhold, action)
    env = fresh()
    for _ in 1:nstart; env_step!(env, NOOP); end
    rams = [copy(get_ram(env))]
    for _ in 1:nhold; env_step!(env, action); push!(rams, copy(get_ram(env))); end
    return rams
end

NSTART = 2100          # 35 s
NHOLD = 24
tR = ram_traj(NSTART, NHOLD, RIGHT)
tN = ram_traj(NSTART, NHOLD, NOOP)
tL = ram_traj(NSTART, NHOLD, LEFT)

dR = Int.(tR[end]) .- Int.(tR[1])
dN = Int.(tN[end]) .- Int.(tN[1])
dL = Int.(tL[end]) .- Int.(tL[1])

println("\n=== RAM bytes that move under RIGHT but differ from NOOP (start=35 s) ===")
println("  addr   start  ΔRIGHT  ΔNOOP  ΔLEFT   (monotone-R trace)")
for a in 1:128
    (dR[a] == dN[a]) && continue
    dR[a] == 0 && continue
    trace = [Int(tR[k][a]) for k in 1:length(tR)]
    println("  \$", lpad(string(a - 1, base = 16), 2, '0'),
            "   ", lpad(Int(tR[1][a]), 4), "  ", lpad(dR[a], 5), "  ",
            lpad(dN[a], 5), "  ", lpad(dL[a], 5), "   ", trace)
end
println("\n(Player-X = the byte with a clean monotone ΔRIGHT>0, ΔLEFT<0, ΔNOOP≈0.)")
