# Dump the two 1D relaxation profiles p_branch(alpha) and p_read(T) (plus rho,
# f_b) on fine meshes, for the overview heatmap. P_step is separable:
#   P_step(alpha,T) = p_read(T)^rho * p_branch(alpha)^f_b
# so the 2D landscape is reconstructed by the Python plotter as an outer product.
#
# Run: cd jutari && julia --project=. ../tools/relaxation_study/dump_profiles.jl

using JuTari
using JuTari.Diff: SoftBus, soft_step, soft_rom_peek, initial_soft_cpu_state, set_relax!

const ROMB = round.(Int, Float32.(read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin"))))
const M = length(ROMB)
const PC0 = ROMB[0xFFD+1] * 256 + ROMB[0xFFC+1]
const STEPS = 3000
const BRANCH = Set(Int[0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0])

set_relax!(on = false)
s = initial_soft_cpu_state(); s.PC = Float32(PC0); b = SoftBus(zeros(Float32, 128), Float32.(ROMB))
pcs = Vector{Int}(undef, STEPS)
for i in 1:STEPS; pcs[i] = Int(round(s.PC)); global s, b = soft_step(s, b); end
fetch = [mod(p, M) for p in pcs]; isbr = [ROMB[a+1] in BRANCH for a in fetch]
OFFS = Int[]; for i in 1:STEPS
    if isbr[i]; raw = ROMB[mod(fetch[i]+1, M)+1]; push!(OFFS, raw<128 ? raw : raw-256); end
end
lens = Int[]; for i in 1:STEPS-1
    if !isbr[i]; d = pcs[i+1]-pcs[i]; (1<=d<=3) && push!(lens, d); end
end
RHO = sum(lens)/length(lens); FB = count(isbr)/STEPS
HIST = Dict{Int,Int}(); for a in fetch; HIST[a] = get(HIST, a, 0)+1; end

p_branch(a) = count(d->abs(d)<0.5*(1+exp(a)), OFFS)/length(OFFS)
function p_read(T)
    Z = sum(exp(-abs(k)/T) for k in -4:4); num=0; den=0
    for (a,c) in HIST
        pull = sum((exp(-abs(k)/T)/Z)*(ROMB[mod(a+k,M)+1]-ROMB[a+1]) for k in -4:4 if k!=0)
        den += c; abs(pull)<0.5 && (num += c)
    end
    num/den
end

# log-spaced meshes spanning the interesting range
amesh = exp.(range(log(1.0), log(24.0), length = 180))
tmesh = exp.(range(log(0.04), log(1.2), length = 180))
pb = p_branch.(amesh)
pr = p_read.(tmesh)

open(joinpath(@__DIR__, "relax_profiles.txt"), "w") do io
    println(io, "RHO ", RHO)
    println(io, "FB ", FB)
    println(io, "ALPHA ", join(amesh, " "))
    println(io, "PBRANCH ", join(pb, " "))
    println(io, "TEMP ", join(tmesh, " "))
    println(io, "PREAD ", join(pr, " "))
end
println("wrote relax_profiles.txt  (rho=", round(RHO, digits=3), " f_b=", round(FB, digits=3),
        ")  alpha-exact from a=", round(amesh[findfirst(>=(1.0), pb)], digits=2),
        "  read-exact up to T=", round(tmesh[findlast(>=(0.999), pr)], digits=3))
