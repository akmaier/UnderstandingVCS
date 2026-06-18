# Bit-exactness LIKELIHOOD model for the fully-relaxed SOFT pass (soft branch
# without STE + temperature-T reads), and one explicit relaxed attempt.
#
# Mechanism: each instruction casts back to Int — the sigmoid-blended program
# counter and the temperature-blended memory reads. The forward stays bit-exact
# only while every cast rounds to the executed value. Two per-cast probabilities,
# computed from the cast margins on the real Space Invaders trace:
#
#   branch (alpha): a branch with signed offset d is exact iff the rounded
#       soft-PC equals the hard target, i.e. |d| < tau_b(alpha) = 0.5*(1+e^alpha).
#       p_branch(alpha) = fraction of executed branch offsets below tau_b.
#
#   read (T): a read at address a returns round( sum_k w_k * mem[a+k] ),
#       w_k ∝ exp(-|k|/T). It is exact iff the neighbour "pull"
#       |sum_{k!=0} w_k (mem[a+k]-mem[a])| < 0.5.
#       p_read(T) = fraction of executed opcode fetches whose pull < 0.5
#       (frequency-weighted over the ROM).
#
# Per-step likelihood  P_step = p_read(T)^rho * p_branch(alpha)^f_b
#   (rho = mean reads/instruction, f_b = branch fraction, both measured).
# Per-run likelihood   P_run(N) = P_step^N ;  expected exact steps = 1/(1-P_step).
#
# Run: cd jutari && julia --project=. ../tools/relaxation_study/likelihood_model.jl

using JuTari
using JuTari.Diff: SoftBus, soft_step, soft_rom_peek, initial_soft_cpu_state, set_relax!
using Printf

const ROMF = Float32.(read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin")))
const ROMB = round.(Int, ROMF)
const M = length(ROMB)                       # 4096
const PC0 = Int(soft_rom_peek(ROMF, 0xFFD)) * 256 + Int(soft_rom_peek(ROMF, 0xFFC))
const STEPS = 3000
const BRANCH = Set(Int[0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0])

fresh() = (s = initial_soft_cpu_state(); s.PC = Float32(PC0);
           (s, SoftBus(zeros(Float32, 128), ROMF)))

# ---- executed trace: PCs, branch offsets, reads/step, fetch histogram --------
function trace_stats()
    set_relax!(on = false)
    s, b = fresh()
    pcs = Vector{Int}(undef, STEPS)
    for i in 1:STEPS
        pcs[i] = Int(round(s.PC)); s, b = soft_step(s, b)
    end
    fetch = [mod(p, M) for p in pcs]
    ops = [ROMB[a + 1] for a in fetch]
    isbr = [o in BRANCH for o in ops]
    offs = Int[]
    for i in 1:STEPS
        if isbr[i]
            raw = ROMB[mod(fetch[i] + 1, M) + 1]
            push!(offs, raw < 128 ? raw : raw - 256)
        end
    end
    lens = Int[]
    for i in 1:STEPS-1
        if !isbr[i]
            d = pcs[i+1] - pcs[i]
            (1 <= d <= 3) && push!(lens, d)
        end
    end
    hist = Dict{Int,Int}()
    for a in fetch; hist[a] = get(hist, a, 0) + 1; end
    return (; offs, rho = sum(lens) / length(lens), f_b = count(isbr) / STEPS, hist)
end

# ---- per-cast probabilities ---------------------------------------------------
taub(alpha) = 0.5 * (1 + exp(alpha))
p_branch(offs, alpha) = isempty(offs) ? 1.0 :
    count(d -> abs(d) < taub(alpha), offs) / length(offs)

function pull_exact(a, T)
    Z = sum(exp(-abs(k) / T) for k in -4:4)
    pull = sum((exp(-abs(k) / T) / Z) * (ROMB[mod(a + k, M) + 1] - ROMB[a + 1])
               for k in -4:4 if k != 0)
    return abs(pull) < 0.5
end
function p_read(hist, T)
    num = 0; den = 0
    for (a, c) in hist
        den += c; pull_exact(a, T) && (num += c)
    end
    return num / den
end

# ---- one explicit relaxed attempt: steps to first divergence ------------------
function first_divergence(alpha, T)
    set_relax!(on = false); sR, bR = fresh()
    set_relax!(on = true, alpha = alpha, temperature = T); sX, bX = fresh()
    div_at = -1
    for i in 1:STEPS
        set_relax!(on = false); sR, bR = soft_step(sR, bR)
        set_relax!(on = true, alpha = alpha, temperature = T); sX, bX = soft_step(sX, bX)
        if Int(round(sR.PC)) != Int(round(sX.PC)) ||
           round.(Int, bR.ram) != round.(Int, bX.ram)
            div_at = i; break
        end
    end
    set_relax!(on = false)
    return div_at
end

function main()
    st = trace_stats()
    @printf("trace: rho(reads/step)=%.2f  f_b(branch frac)=%.3f  #branches=%d  |offset|max=%d\n",
            st.rho, st.f_b, length(st.offs), maximum(abs.(st.offs)))

    ALPHAS = [1.0, 2.0, 4.0, 6.0, 8.0, 20.0]
    TS     = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

    println("\n-- per-cast: p_branch(alpha) --")
    for a in ALPHAS; @printf("  alpha=%4.1f  tau_b=%9.1f  p_branch=%.4f\n", a, taub(a), p_branch(st.offs, a)); end
    println("-- per-cast: p_read(T) --")
    for T in TS; @printf("  T=%4.2f  p_read=%.4f\n", T, p_read(st.hist, T)); end

    Pstep(a, T) = p_read(st.hist, T)^st.rho * p_branch(st.offs, a)^st.f_b

    println("\n== OVERVIEW: per-step bit-exactness likelihood  P_step(alpha,T) ==")
    print(rpad("a\\T", 7)); for T in TS; @printf("%9.2f", T); end; println()
    for a in ALPHAS
        print(rpad(@sprintf("%.0f", a), 7))
        for T in TS; @printf("%9.4f", Pstep(a, T)); end; println()
    end

    println("\n== expected # bit-exact steps  1/(1-P_step)  (>=N means exact for the run) ==")
    print(rpad("a\\T", 7)); for T in TS; @printf("%9.2f", T); end; println()
    for a in ALPHAS
        print(rpad(@sprintf("%.0f", a), 7))
        for T in TS
            p = Pstep(a, T)
            e = p >= 1 ? Inf : 1 / (1 - p)
            @printf("%9s", e == Inf ? "inf" : (e > 1e5 ? @sprintf("%.0e", e) : @sprintf("%.0f", e)))
        end
        println()
    end

    # one explicit reasonable attempt + a few validation points
    println("\n== one attempt + validation: first-divergence step (-1 = exact for all $STEPS) ==")
    for (a, T) in [(6.0, 0.1), (6.0, 0.2), (8.0, 0.1), (4.0, 0.05), (20.0, 0.3), (1.0, 0.05)]
        d = first_divergence(a, T)
        @printf("  alpha=%4.1f T=%4.2f : first_div=%s  P_step=%.4f  pred_exact_steps=%s\n",
                a, T, d == -1 ? "exact" : string(d), Pstep(a, T),
                Pstep(a, T) >= 1 ? "inf" : @sprintf("%.0f", 1 / (1 - Pstep(a, T))))
    end
end

main()
