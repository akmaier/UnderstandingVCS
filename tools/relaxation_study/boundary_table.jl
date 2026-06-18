# Combined boundary table: measured (empirical) vs predicted (cast-margin model)
# forward bit-exactness of the fully-relaxed soft pass, sampled densely around
# the two critical boundaries (the branch sharpness alpha and the read
# temperature T) on the full jutari simulator running Space Invaders.
#
#   Measured : run the relaxed forward N steps, % RAM bytes / % frame pixels
#              still matching the executed (STE) reference.
#   Predicted: p_branch(alpha)=frac offsets with |d|<0.5(1+e^alpha);
#              p_read(T)=frac fetches with neighbour pull <0.5;
#              P_step = p_read^rho * p_branch^f_b  (rho, f_b measured).
#
# Run: cd jutari && julia --project=. ../tools/relaxation_study/boundary_table.jl

using JuTari
using JuTari.Diff: SoftBus, soft_step, soft_rom_peek, initial_soft_cpu_state,
                   set_relax!, soft_render_frame
using Printf

const ROMF = Float32.(read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin")))
const ROMB = round.(Int, ROMF)
const M = length(ROMB)
const PC0 = Int(soft_rom_peek(ROMF, 0xFFD)) * 256 + Int(soft_rom_peek(ROMF, 0xFFC))
const STEPS = 3000
const BRANCH = Set(Int[0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0])

fresh() = (s = initial_soft_cpu_state(); s.PC = Float32(PC0);
           (s, SoftBus(zeros(Float32, 128), ROMF)))
runN(s, b, n) = (for _ in 1:n; s, b = soft_step(s, b); end; (s, b))

# ---- trace stats + executed reference ----
set_relax!(on = false)
let
    global OFFS, RHO, FB, HIST, RAM_REF, F_REF, PXTOT
    s, b = fresh(); pcs = Vector{Int}(undef, STEPS)
    for i in 1:STEPS; pcs[i] = Int(round(s.PC)); s, b = soft_step(s, b); end
    RAM_REF = round.(Int, b.ram); F_REF = round.(Int, soft_render_frame(b)); PXTOT = length(F_REF)
    fetch = [mod(p, M) for p in pcs]; isbr = [ROMB[a+1] in BRANCH for a in fetch]
    OFFS = Int[]; for i in 1:STEPS
        if isbr[i]; raw = ROMB[mod(fetch[i]+1, M)+1]; push!(OFFS, raw<128 ? raw : raw-256); end
    end
    lens = Int[]; for i in 1:STEPS-1
        if !isbr[i]; d = pcs[i+1]-pcs[i]; (1<=d<=3) && push!(lens, d); end
    end
    RHO = sum(lens)/length(lens); FB = count(isbr)/STEPS
    HIST = Dict{Int,Int}(); for a in fetch; HIST[a] = get(HIST, a, 0)+1; end
end

taub(a) = 0.5*(1+exp(a))
p_branch(a) = count(d->abs(d)<taub(a), OFFS)/length(OFFS)
function p_read(T)
    Z = sum(exp(-abs(k)/T) for k in -4:4); num=0; den=0
    for (a,c) in HIST
        pull = sum((exp(-abs(k)/T)/Z)*(ROMB[mod(a+k,M)+1]-ROMB[a+1]) for k in -4:4 if k!=0)
        den += c; abs(pull)<0.5 && (num += c)
    end
    num/den
end
Pstep(a,T) = p_read(T)^RHO * p_branch(a)^FB

# Faithful bit-exactness measure: first step at which the relaxed trajectory
# deviates from the executed one (-1 = exact for all STEPS). The endpoint
# RAM/frame match is NOT used: the game re-initialises state each frame, so a
# diverged run can partially re-synchronise and falsely look exact at the end.
function first_div(a, T)
    set_relax!(on = false); sR, bR = fresh()
    set_relax!(on = true, alpha = a, temperature = T); sX, bX = fresh()
    for i in 1:STEPS
        set_relax!(on = false); sR, bR = soft_step(sR, bR)
        set_relax!(on = true, alpha = a, temperature = T); sX, bX = soft_step(sX, bX)
        if Int(round(sR.PC)) != Int(round(sX.PC)) || round.(Int, bR.ram) != round.(Int, bX.ram)
            set_relax!(on = false); return i
        end
    end
    set_relax!(on = false); return -1
end

@printf("rho=%.2f  f_b=%.3f  #branches=%d  max|offset|=%d\n\n", RHO, FB, length(OFFS), maximum(abs.(OFFS)))
println("  alpha     T | first_div | p_branch  p_read   P_step  pred_steps")
println("  -----------------------------------------------------------------")
function row(a, T)
    d = first_div(a, T); P = Pstep(a, T)
    pred = P >= 1 ? "inf" : @sprintf("%.0f", 1 / (1 - P))
    @printf("  %5.1f  %5.3f | %9s | %7.3f %7.3f %8.4f  %9s\n",
            a, T, d == -1 ? "exact" : string(d), p_branch(a), p_read(T), P, pred)
end
println("  [branch boundary: T = 0.10 fixed]")
for a in (2.0, 3.0, 4.0, 5.0, 6.0); row(a, 0.10); end
println("  [temperature boundary: alpha = 20 fixed]")
for T in (0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25); row(20.0, T); end
