# Empirical alpha x T grid: forward bit-exactness of the FULLY-RELAXED soft
# pass (soft branch without STE + temperature-T reads) on the full jutari
# simulator running Space Invaders, vs the executed (STE) bit-exact reference.
# Cells: % frame pixels exact and % RAM bytes exact after N instructions.
#
# Run: cd jutari && julia --project=. ../tools/relaxation_study/grid_relax.jl

using JuTari
using JuTari.Diff: SoftBus, soft_step, soft_rom_peek, initial_soft_cpu_state,
                   set_relax!, soft_render_frame
using Printf

rom = Float32.(read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin")))
pc0 = Int(soft_rom_peek(rom, 0xFFD)) * 256 + Int(soft_rom_peek(rom, 0xFFC))
N = 3000

fresh() = (s = initial_soft_cpu_state(); s.PC = Float32(pc0);
           (s, SoftBus(zeros(Float32, 128), rom)))
function runN(s, b, n)
    for _ in 1:n
        s, b = soft_step(s, b)
    end
    return s, b
end

set_relax!(on = false)
_, bref = runN(fresh()..., N)
ram_ref = round.(Int, bref.ram)
F_ref = round.(Int, soft_render_frame(bref)); px_tot = length(F_ref)

ALPHAS = [1.0, 2.0, 4.0, 8.0, 20.0]
TS     = [0.05, 0.1, 0.3, 1.0, 3.0]

pix = Dict{Tuple{Float64,Float64},Float64}()
ram = Dict{Tuple{Float64,Float64},Float64}()
for a in ALPHAS, T in TS
    set_relax!(on = true, alpha = a, temperature = T)
    _, b = runN(fresh()..., N)
    set_relax!(on = false)
    pix[(a, T)] = 100 * count(round.(Int, soft_render_frame(b)) .== F_ref) / px_tot
    ram[(a, T)] = 100 * count(round.(Int, b.ram) .== ram_ref) / 128
end

function show_table(title, d)
    println("\n== $title  (N=$N instrs, Space Invaders, jutari) ==")
    print(rpad("alpha\\T", 8)); for T in TS; print(lpad("T=$T", 9)); end; println()
    for a in ALPHAS
        print(rpad("a=$a", 8))
        for T in TS; @printf("%9.1f", d[(a, T)]); end
        println()
    end
end
show_table("% frame pixels exact", pix)
show_table("% RAM bytes exact", ram)
