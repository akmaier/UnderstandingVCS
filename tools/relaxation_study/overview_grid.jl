# Full 2D (alpha x T) overview grid of the per-step bit-exactness likelihood
# P_step, plus the two 1D marginals p_branch(alpha) and p_read(T). Used to design
# the supplement presentation (overview + detail). P_step is separable:
#   P_step(alpha,T) = p_read(T)^rho * p_branch(alpha)^f_b.
#
# Run: cd jutari && julia --project=. ../tools/relaxation_study/overview_grid.jl

using JuTari
using JuTari.Diff: SoftBus, soft_step, soft_rom_peek, initial_soft_cpu_state, set_relax!
using Printf

const ROMB = round.(Int, Float32.(read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin"))))
const M = length(ROMB)
const PC0 = (ROMB[0xFFD+1]) * 256 + (ROMB[0xFFC+1])
const STEPS = 3000
const BRANCH = Set(Int[0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0])

set_relax!(on = false)
s = initial_soft_cpu_state(); s.PC = Float32(PC0); b = SoftBus(zeros(Float32, 128), ROMB .|> Float32)
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
Pstep(a,T) = p_read(T)^RHO * p_branch(a)^FB

A = [1.0,2.0,3.0,4.0,5.0,6.0,8.0,20.0]
T = [0.05,0.10,0.12,0.15,0.20,0.30,0.50,1.0]

@printf("rho=%.2f f_b=%.3f\n\nP_step grid (rows alpha, cols T):\n", RHO, FB)
print(rpad("a\\T",6)); for t in T; @printf("%7.2f", t); end; println("   | p_branch")
for a in A
    print(rpad(@sprintf("%.0f",a),6)); for t in T; @printf("%7.2f", Pstep(a,t)); end
    @printf("   | %.3f\n", p_branch(a))
end
print(rpad("p_read",6)); for t in T; @printf("%7.3f", p_read(t)); end; println()
