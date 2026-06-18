# Conservative probe: does jutari's executed SOFT simulator make forward
# progress through a real ROM (Space Invaders)? This is the prerequisite for
# the relaxation study (effect of alpha/T on pixel exactness on the full
# simulator) — before sweeping alpha/T we confirm the soft CPU runs the ROM
# end-to-end instead of stalling in a polling loop or decaying into garbage.
#
# Run: cd jutari && julia --project=. ../tools/relaxation_study/probe_soft_run.jl

using JuTari
using JuTari.Diff: SoftBus, soft_step!, soft_rom_peek, initial_soft_cpu_state

rom_path = joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin")
rom = Float32.(read(rom_path))
println("ROM: ", basename(rom_path), "  (", length(rom), " bytes)")

# Reset vector lives at $FFFC/$FFFD; cart range maps via addr & 0x0FFF.
lo = soft_rom_peek(rom, 0xFFC); hi = soft_rom_peek(rom, 0xFFD)
reset_pc = Int(hi) * 256 + Int(lo)
println("reset vector = \$", uppercase(string(reset_pc, base = 16)))

bus = SoftBus(zeros(Float32, 128), rom)
state = initial_soft_cpu_state()
state.PC = Float32(reset_pc)

N = 5000
pcs = Vector{Int}(undef, N)
t0 = time()
for i in 1:N
    soft_step!(state, bus)
    pcs[i] = Int(round(state.PC))
end
wall = time() - t0

println("ran $N soft steps in $(round(wall, digits = 2)) s ",
        "($(round(Int, N / wall)) steps/s)")
println("PC final  = \$", uppercase(string(Int(round(state.PC)), base = 16)))
println("PC range  = \$", uppercase(string(minimum(pcs), base = 16)), " .. \$",
        uppercase(string(maximum(pcs), base = 16)))
println("unique PCs over run = ", length(unique(pcs)), " / $N")
nz = count(!=(0f0), bus.ram)
println("RAM nonzero bytes = $nz / 128")
# last-100-step PC spread tells us whether we ended stuck in a tight wait loop
println("unique PCs in last 100 steps = ", length(unique(pcs[end-99:end])))
