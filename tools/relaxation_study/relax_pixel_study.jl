# Effect of the relaxation parameters (alpha, T) on pixel exactness, run on the
# FULL jutari simulator (Space Invaders). Companion to Theorem 2 / the
# "exact for small T and large alpha" claim, on a real ROM rather than a toy.
#
# Method (self-contained, per Theorem 1 "the soft and hard one-step maps are
# equal by design"): the EXECUTED soft path (straight-through branch + one-hot
# reads) is the bit-exact reference. We run the FULLY-RELAXED forward at (alpha,
# T) for the same trace and measure how far it drifts:
#   * RAM bytes  — fraction of the 128-byte RAM that still matches the reference
#   * pixels     — fraction of the rendered frame that still matches (the frame
#                  is rendered with relaxation OFF on both states, so this
#                  isolates the CPU-forward drift, not a render relaxation)
#
# Usage (one conservative run):
#   cd jutari && julia --project=. ../tools/relaxation_study/relax_pixel_study.jl
#   # optional: --steps N --alpha A --temp T

using JuTari
using JuTari.Diff: SoftBus, soft_step, soft_rom_peek, initial_soft_cpu_state,
                   set_relax!, soft_render_frame

# -- args --------------------------------------------------------------------
steps = 3000; alpha = 20.0; temp = 0.05
let i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        a == "--steps" && (global steps = parse(Int, ARGS[i+1]); i += 2; continue)
        a == "--alpha" && (global alpha = parse(Float64, ARGS[i+1]); i += 2; continue)
        a == "--temp"  && (global temp  = parse(Float64, ARGS[i+1]); i += 2; continue)
        error("unknown arg $a")
    end
end

rom_path = joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin")
rom = Float32.(read(rom_path))
lo = soft_rom_peek(rom, 0xFFC); hi = soft_rom_peek(rom, 0xFFD)
reset_pc = Int(hi) * 256 + Int(lo)

function fresh()
    bus = SoftBus(zeros(Float32, 128), rom)
    st = initial_soft_cpu_state(); st.PC = Float32(reset_pc)
    return st, bus
end
function run_n(st, bus, n)
    for _ in 1:n
        st, bus = soft_step(st, bus)
    end
    return st, bus
end

println("ROM space_invaders.bin  reset \$", uppercase(string(reset_pc, base = 16)),
        "  steps=$steps  (functional soft_step path)")

# Reference: executed (bit-exact) forward.
set_relax!(on = false)
t0 = time(); st_ref, bus_ref = run_n(fresh()..., steps); t_ref = time() - t0

# Test: fully-relaxed forward at the (alpha, T) corner.
set_relax!(on = true, alpha = alpha, temperature = temp)
t0 = time(); st_rel, bus_rel = run_n(fresh()..., steps); t_rel = time() - t0
set_relax!(on = false)            # render exactly on both states

# -- compare -----------------------------------------------------------------
ram_ref = round.(Int, bus_ref.ram); ram_rel = round.(Int, bus_rel.ram)
ram_match = count(ram_ref .== ram_rel)

F_ref = soft_render_frame(bus_ref)
F_rel = soft_render_frame(bus_rel)
px_match = count(F_ref .== F_rel); px_tot = length(F_ref)

regs(s) = (A = Int(round(s.A)), X = Int(round(s.X)), Y = Int(round(s.Y)),
           SP = Int(round(s.SP)), PC = Int(round(s.PC)), P = Int(round(s.P)))
r1 = regs(st_ref); r2 = regs(st_rel)

println("\n--- conservative corner: alpha=$alpha  T=$temp ---")
println("exec  $(round(t_ref, digits = 2))s   relaxed $(round(t_rel, digits = 2))s ",
        "($(round(Int, steps / t_rel)) steps/s)")
println("RAM bytes match : $ram_match / 128   ", ram_match == 128 ? "(bit-exact)" : "")
println("frame pixels    : $px_match / $px_tot  ",
        "($(round(100 * px_match / px_tot, digits = 3))%)  ",
        px_match == px_tot ? "(pixel-exact)" : "")
println("regs exec   : ", r1)
println("regs relaxed: ", r2, r1 == r2 ? "   (identical)" : "   (DIVERGED)")
