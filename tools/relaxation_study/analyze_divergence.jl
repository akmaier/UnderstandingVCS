# Analytic divergence-after-boot for the temperature-T relaxed read.
#
# A read at ROM byte a fails (rounds wrong) once the temperature blend pulls it
# past 1/2:  pull(a,T) = sum_{k!=0} w_k(T) (rom[a+k]-rom[a]),  w_k = e^{-|k|/T}/Z.
# pull grows with T, so each byte has a threshold T_fail(a). As T rises from 0
# the FIRST byte to cross is the most-borderline one; the run diverges when that
# byte is first FETCHED. So "boots cleanly then diverges later" exists iff some
# byte first fetched AFTER boot has a lower T_fail than every byte fetched during
# boot -- then a T between the two is exact through boot and fails on that byte.
#
# This script computes T_fail for all 4096 bytes and the first-fetch frame of
# each (opcode + operand bytes), splits boot (frame<64) vs rollout (>=64), and
# reports the running-min T_fail over rollout frames = the T -> divergence-frame map.
#
# Run: cd jutari && julia --project=. ../tools/relaxation_study/analyze_divergence.jl

using JuTari
using JuTari.ConsoleModule: initial_console, console_reset!, console_step!

const ROM = Int.(read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin")))
const M = length(ROM)
b(a) = ROM[mod(a, M) + 1]

# Smallest T at which |pull| >= 0.5 (pull is 0 at T->0, grows with T). Inf = never.
function tfail(a)
    prev = 0.0
    T = 0.02
    while T <= 3.0
        Z = sum(exp(-abs(k) / T) for k in -4:4)
        pull = sum((exp(-abs(k) / T) / Z) * (b(a + k) - b(a)) for k in -4:4 if k != 0)
        abs(pull) >= 0.5 && return T
        T += 0.0005
    end
    return Inf
end
Tf = [tfail(a) for a in 0:M-1]

# Per-instruction fetch trace: first frame each ROM byte is read (opcode @PC and
# operand bytes between consecutive PCs).
const NF = 200
function build_trace()
    con = initial_console(read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin")))
    console_reset!(con)
    firstframe = fill(typemax(Int), M)
    fr = Int(con.bus.tia.frame); guard = 0
    while fr < NF && guard < 3_000_000
        pc = Int(con.cpu.PC)
        for off in 0:2                       # opcode + up to 2 operand bytes
            a = (pc + off) & 0x0FFF
            firstframe[a + 1] > fr && (firstframe[a + 1] = fr)
        end
        console_step!(con)
        fr = Int(con.bus.tia.frame); guard += 1
    end
    return firstframe, guard, fr
end
firstframe, guard, fr = build_trace()

bootmin = minimum(Tf[a + 1] for a in 0:M-1 if firstframe[a + 1] < 64; init = Inf)
println("instrs stepped=$guard  reached frame=$fr")
println("min T_fail over BOOT-fetched bytes (frame<64): ", round(bootmin, digits=4))

# 5 most-borderline FETCHED bytes overall + their first-fetch frame
fetched = [a for a in 0:M-1 if firstframe[a + 1] != typemax(Int) && isfinite(Tf[a + 1])]
order = sort(fetched, by = a -> Tf[a + 1])
println("\nmost-borderline fetched bytes (addr, T_fail, first_frame):")
for a in order[1:min(8, length(order))]
    println("  \$", uppercase(string(0x1000 + a, base = 16)), "  T_fail=", round(Tf[a + 1], digits=4),
            "  first_frame=", firstframe[a + 1], firstframe[a + 1] < 64 ? " (boot)" : " (ROLLOUT)")
end

# Running-min T_fail over rollout frames = T -> first-divergence-frame map
println("\nrollout running-min T_fail (clean boot, relax on from frame 64):")
println("  a byte first-fetched at frame f with this T_fail diverges there iff T>=it")
function rollout_runmin()
    rmin = Inf
    for f in 64:NF-1
        fmin = minimum((Tf[a + 1] for a in 0:M-1 if firstframe[a + 1] == f && isfinite(Tf[a + 1])); init = Inf)
        rmin = min(rmin, fmin)
        f in (64, 65, 70, 80, 100, 130, 170, 199) &&
            println("  through rollout-frame $f: running-min T_fail = ", round(rmin, digits=4))
    end
end
rollout_runmin()
println("\nboots-then-diverges-later possible iff rollout running-min DROPS below boot-min (",
        round(bootmin, digits=4), ") at a frame > 64.")
