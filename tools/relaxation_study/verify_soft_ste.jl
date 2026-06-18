# SOFT-STE bit-exactness regression check for the relaxation toggle.
#
# Proves that with the relaxation OFF (the default / executed STE path):
#   (1) soft_rom_peek / soft_ram_peek are EXACTLY the original one-hot dot
#       product (the relaxation `if _relax_on[]` guard is inert);
#   (2) the executed soft forward on real ROMs is deterministic and UNAFFECTED
#       by having toggled the relaxation on and back off (no state leak), i.e.
#       RAM and the rendered frame are byte-identical before/after a perturbed
#       relaxed run.
#
# Run: cd jutari && julia --project=. ../tools/relaxation_study/verify_soft_ste.jl

using JuTari
using JuTari.Diff: SoftBus, soft_step, soft_rom_peek, soft_ram_peek,
                   initial_soft_cpu_state, set_relax!, relax_config,
                   soft_render_frame
using Random

@assert relax_config().on == false "relaxation must default OFF"
@assert relax_config().alpha == 10.0 "relax alpha default must be 10.0 (matches executed branch)"

fail = false

# (1) relax-off peeks == original one-hot formula, bit-for-bit -----------------
Random.seed!(1)
ok_peek = true
for _ in 1:5000
    rom = Float32.(rand(0:255, 4096)); ram = Float32.(rand(0:255, 128))
    a_rom = rand(0:8191); a_ram = rand(0:127)
    ref_rom = let n = length(rom); idx = mod(Int(a_rom), n)
        sum(Float32.((0:n-1) .== idx) .* rom) end
    ref_ram = let n = length(ram)
        sum(Float32.((0:n-1) .== Int(a_ram)) .* ram) end
    set_relax!(on = false)
    global ok_peek &= (soft_rom_peek(rom, a_rom) === ref_rom) &
                      (soft_ram_peek(ram, a_ram) === ref_ram)
end
println("(1) relax-off peeks identical to one-hot original : ", ok_peek)
global fail |= !ok_peek

# (2) executed-soft on real ROMs is byte-stable across a relax toggle ----------
function run_rom(rom, pc, n)
    bus = SoftBus(zeros(Float32, 128), rom)
    st = initial_soft_cpu_state(); st.PC = Float32(pc)
    for _ in 1:n
        st, bus = soft_step(st, bus)
    end
    return Int.(round.(bus.ram)), Int.(round.(soft_render_frame(bus)))
end

for name in ("pong.bin", "breakout.bin", "space_invaders.bin")
    path = joinpath(@__DIR__, "..", "..", "xitari", "roms", name)
    isfile(path) || (println("  skip $name (missing)"); continue)
    rom = Float32.(read(path))
    pc = Int(soft_rom_peek(rom, 0xFFD)) * 256 + Int(soft_rom_peek(rom, 0xFFC))

    set_relax!(on = false); ram1, fr1 = run_rom(rom, pc, 3000)
    set_relax!(on = true, alpha = 1.0, temperature = 5.0); run_rom(rom, pc, 3000)  # perturb
    set_relax!(on = false); ram2, fr2 = run_rom(rom, pc, 3000)

    same = (ram1 == ram2) && (fr1 == fr2)
    println("(2) $name executed-soft byte-stable after relax toggle : ", same,
            "  (ram=", ram1 == ram2, " frame=", fr1 == fr2, ")")
    global fail |= !same
end

set_relax!(on = false)
println(fail ? "RESULT: FAIL — soft-STE bit-exactness broken" :
               "RESULT: PASS — soft-STE bit-exactness preserved")
exit(fail ? 1 : 0)
