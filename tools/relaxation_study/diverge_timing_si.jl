# Divergence-timing study for the FULLY-RELAXED forward on Space Invaders.
#
# Reuses the patterns in recipe_check.jl / diag_a5.jl (set_relax!, soft_step,
# lockstep executed-vs-relaxed first-divergence) and adds:
#   (a) first-divergence INSTRUCTION for relaxed(alpha=5, T=0.15) vs executed-soft
#   (b) how the divergence evolves -- mismatching RAM-byte count + PC-match at
#       500/1000/2000/5000/10000 instructions, and whether it reconverges
#   (c) CPU instructions per Atari FRAME for Space Invaders (HARD Console),
#       plus mean cycles/instruction and the cycles-based cross-check.
#
# Run: cd jutari && julia --project=. ../tools/relaxation_study/diverge_timing_si.jl

using JuTari
using JuTari.Diff: SoftBus, soft_step, soft_rom_peek, initial_soft_cpu_state, set_relax!
using JuTari.ConsoleModule: Console, initial_console, console_reset!
using JuTari.CPU: step
import JuTari.Bus: peek

const ALPHA = 5.0
const T     = 0.15
const N     = 12000           # lockstep horizon (>= 6000 as requested, headroom for 10k checkpoint)

const ROMPATH = joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin")
romf = Float32.(read(ROMPATH))
pc0  = Int(soft_rom_peek(romf, 0xFFD)) * 256 + Int(soft_rom_peek(romf, 0xFFC))
println("Space Invaders reset vector PC0 = 0x", string(pc0, base = 16, pad = 4))

fresh() = (s = initial_soft_cpu_state(); s.PC = Float32(pc0); (s, SoftBus(zeros(Float32, 128), romf)))

# ---------------------------------------------------------------------------
# (a) + (b): lockstep executed-soft vs relaxed(alpha=5, T=0.15)
# ---------------------------------------------------------------------------
ram_mismatch(bR, bX) = count(round.(Int, bR.ram) .!= round.(Int, bX.ram))
pc_match(sR, sX)     = Int(round(sR.PC)) == Int(round(sX.PC))

set_relax!(on = false);                                sR, bR = fresh()
set_relax!(on = true, alpha = ALPHA, temperature = T); sX, bX = fresh()

const CHECKPOINTS = (500, 1000, 2000, 5000, 10000)
first_div      = -1
evol           = Dict{Int,Tuple{Int,Bool}}()   # instr => (ram_mismatch_count, pc_match)
diverged_once  = false
reconverged    = false

for i in 1:N
    global sR, bR, sX, bX, first_div, diverged_once, reconverged
    set_relax!(on = false);                                global sR, bR = soft_step(sR, bR)
    set_relax!(on = true, alpha = ALPHA, temperature = T); global sX, bX = soft_step(sX, bX)
    mm  = ram_mismatch(bR, bX)
    pcm = pc_match(sR, sX)
    differs = (!pcm) || (mm != 0)
    if differs && first_div == -1
        first_div = i
    end
    if differs
        diverged_once = true
    elseif diverged_once && !reconverged
        # state had diverged earlier and is now byte-identical again (PC + all RAM)
        reconverged = true
    end
    if i in CHECKPOINTS
        evol[i] = (mm, pcm)
    end
end
set_relax!(on = false)

# Whether they are still diverged at the END of the run (PC or any RAM byte).
still_diverged_at_end = (!pc_match(sR, sX)) || (ram_mismatch(bR, bX) != 0)

println("\n=== (a) first-divergence (relaxed alpha=$ALPHA, T=$T vs executed-soft) ===")
println("first_div_instruction = ", first_div)

println("\n=== (b) divergence evolution ===")
for c in CHECKPOINTS
    mm, pcm = evol[c]
    println(rpad("@ $c instr:", 16),
            "RAM-byte mismatches = ", rpad(mm, 4),
            "  PC matches = ", pcm)
end
println("diverged_once = ", diverged_once,
        "   reconverged (became byte-identical again) = ", reconverged)
println("still_diverged_at_end (after $N instr) = ", still_diverged_at_end,
        "   final RAM mismatches = ", ram_mismatch(bR, bX),
        "   final PC match = ", pc_match(sR, sX))

# ---------------------------------------------------------------------------
# (c) CPU instructions per Atari FRAME (HARD Console), Space Invaders
# ---------------------------------------------------------------------------
println("\n=== (c) CPU instructions per Atari frame (HARD Console) ===")
romb = read(ROMPATH)            # UInt8 bytes for the HARD console
console = initial_console(romb)
console_reset!(console)         # cold power-on, PC <- reset vector

# Warm up a few frames (boot/title), then measure steady-state frames.
const WARM_FRAMES    = 3
const MEASURE_FRAMES = 12
const INSTR_BUDGET   = 25000    # mirror xitari execute(25000) guard

function run_one_frame!(console)
    tia = console.bus.tia
    if tia.buffer_swap_pending
        tia.framebuffer, tia.framebuffer_prev = tia.framebuffer_prev, tia.framebuffer
        tia.buffer_swap_pending = false
    end
    start_frame  = tia.frame
    instr        = 0
    cyc_before   = console.cpu.cycles
    for _ in 1:INSTR_BUDGET
        step(console.cpu, console.bus)
        instr += 1
        if console.bus.tia.frame != start_frame
            break
        end
    end
    cyc = Int(console.cpu.cycles - cyc_before)
    return instr, cyc
end

for _ in 1:WARM_FRAMES
    run_one_frame!(console)
end

instr_counts = Int[]
cyc_counts   = Int[]
for f in 1:MEASURE_FRAMES
    ins, cyc = run_one_frame!(console)
    push!(instr_counts, ins)
    push!(cyc_counts, cyc)
end

mean(v) = sum(v) / length(v)
instr_per_frame_measured = mean(instr_counts)
cyc_per_frame_measured   = mean(cyc_counts)
mean_cyc_per_instr       = sum(cyc_counts) / sum(instr_counts)

println("per-frame instruction counts : ", instr_counts)
println("per-frame CPU cycle counts   : ", cyc_counts)
println("mean instructions / frame    = ", round(instr_per_frame_measured, digits = 2))
println("mean CPU cycles / frame      = ", round(cyc_per_frame_measured,   digits = 2))
println("mean cycles / instruction    = ", round(mean_cyc_per_instr,        digits = 4))

# Cross-check via NTSC frame timing: 262 scanlines x 76 colour clocks
# = 19912 colour clocks; 3 colour clocks per CPU cycle => ~6637.3 CPU cycles.
const NTSC_CYC_PER_FRAME = 262 * 76 / 3
instr_per_frame_from_cycles = NTSC_CYC_PER_FRAME / mean_cyc_per_instr
println("NTSC cycles/frame (262*76/3) = ", round(NTSC_CYC_PER_FRAME, digits = 2))
println("instr/frame from cycles      = ", round(instr_per_frame_from_cycles, digits = 2),
        "  (NTSC_cyc_per_frame / mean_cyc_per_instr)")

ipf = round(Int, instr_per_frame_measured)
println("\n=== SUMMARY ===")
println("first_div_instruction   = ", first_div)
println("instructions_per_frame  = ", ipf, "  (measured mean, HARD Console)")
println("first_div_frame         = ", round(first_div / ipf, digits = 4))
println("reconverges             = ", reconverged)
