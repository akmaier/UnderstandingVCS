# Verify the parameter-setting recipe is reliable ACROSS ROMs (ROM-independent
# safe bounds): the fully-relaxed forward must stay bit-exact at the recommended
# operating point (alpha=6, T=0.1) and at a safer point (alpha=8, T=0.05), and
# must diverge just outside the bounds (alpha=5 or T=0.15) -- confirming the
# bounds are tight, not loose.
#
# Run: cd jutari && julia --project=. ../tools/relaxation_study/recipe_check.jl

using JuTari
using JuTari.Diff: SoftBus, soft_step, soft_rom_peek, initial_soft_cpu_state, set_relax!

const N = 6000
const ROMS = ("pong.bin", "breakout.bin", "space_invaders.bin")

function first_div(rom, pc0, alpha, T)
    fresh() = (s = initial_soft_cpu_state(); s.PC = Float32(pc0); (s, SoftBus(zeros(Float32, 128), rom)))
    set_relax!(on = false); sR, bR = fresh()
    set_relax!(on = true, alpha = alpha, temperature = T); sX, bX = fresh()
    for i in 1:N
        set_relax!(on = false); sR, bR = soft_step(sR, bR)
        set_relax!(on = true, alpha = alpha, temperature = T); sX, bX = soft_step(sX, bX)
        if Int(round(sR.PC)) != Int(round(sX.PC)) || round.(Int, bR.ram) != round.(Int, bX.ram)
            set_relax!(on = false); return i
        end
    end
    set_relax!(on = false); return -1
end

POINTS = [("recommended a=6,T=0.14", 6.0, 0.14),
          ("safer a=8,T=0.05",       8.0, 0.05),
          ("below a-bound a=5,T=0.14", 5.0, 0.14),
          ("above T-bound a=6,T=0.15", 6.0, 0.15)]

println("first-divergence step over N=$N  (-1 / exact = bit-exact whole run)\n")
print(rpad("ROM", 26))
for (lbl, _, _) in POINTS; print(rpad(lbl, 26)); end; println()
for name in ROMS
    path = joinpath(@__DIR__, "..", "..", "xitari", "roms", name)
    rom = Float32.(read(path))
    pc0 = Int(soft_rom_peek(rom, 0xFFD)) * 256 + Int(soft_rom_peek(rom, 0xFFC))
    print(rpad(name, 26))
    for (_, a, T) in POINTS
        d = first_div(rom, pc0, a, T)
        print(rpad(d == -1 ? "exact" : string(d), 26))
    end
    println()
end
