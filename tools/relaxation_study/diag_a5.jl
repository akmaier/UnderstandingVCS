using JuTari
using JuTari.Diff: SoftBus, soft_step, soft_rom_peek, initial_soft_cpu_state, set_relax!
romf = Float32.(read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin")))
romb = round.(Int, romf); M = length(romb)
pc0 = Int(soft_rom_peek(romf, 0xFFD)) * 256 + Int(soft_rom_peek(romf, 0xFFC)); STEPS = 3000
BRANCH = Set(Int[0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0])
fresh() = (s = initial_soft_cpu_state(); s.PC = Float32(pc0); (s, SoftBus(zeros(Float32, 128), romf)))

# executed-trace branches with |offset| >= 75 (would fail at alpha=5)
set_relax!(on = false); s, b = fresh(); big = Tuple{Int,Int}[]
for i in 1:STEPS
    a = mod(Int(round(s.PC)), M); op = romb[a+1]
    if op in BRANCH
        raw = romb[mod(a+1, M)+1]; off = raw < 128 ? raw : raw - 256
        abs(off) >= 75 && push!(big, (i, off))
    end
    global s, b = soft_step(s, b)
end
println("executed branches with |offset|>=75: ", length(big),
        "   first few (step,offset): ", first(big, min(6, length(big))))

function firstdiv(al, T)
    set_relax!(on = false); sR, bR = fresh()
    set_relax!(on = true, alpha = al, temperature = T); sX, bX = fresh()
    for i in 1:STEPS
        set_relax!(on = false); sR, bR = soft_step(sR, bR)
        set_relax!(on = true, alpha = al, temperature = T); sX, bX = soft_step(sX, bX)
        if Int(round(sR.PC)) != Int(round(sX.PC)) || round.(Int, bR.ram) != round.(Int, bX.ram)
            set_relax!(on = false); return i
        end
    end
    set_relax!(on = false); return -1
end
for al in (4.0, 5.0, 6.0)
    println("first_divergence(alpha=$al, T=0.1) = ", firstdiv(al, 0.1))
end
