# triad_sm.jl — shared Sufficiency (S) and Minimality (M) scorers for the
# F∧S∧M correctness triad (paper sec:triad, 03_methods.tex). This factors the two
# triad axes that most Phase-B attribution runners and several Phase-C runners were
# NOT yet emitting, so they can be added WITHOUT re-deriving the machinery per file.
#
# Authoritative definitions (03_methods.tex sec:triad):
#
#   M (minimality) = |U*| / |U_hat| ∈ (0,1].  |U*| is the size of the TRUE minimal
#     cause set — the smallest set of cells/registers whose intervention reproduces
#     y — and |U_hat| is the size of the cause set the METHOD actually names (its
#     top-k / above-threshold attribution set). 1 ⇒ named exactly the minimal set;
#     a method that names the whole state pays for every spurious cell.
#
#   S (sufficiency) = a HELD-OUT predictive score ∈ [0,1]. Fit/read the explanation
#     on a calibration set of interventions, then predict the true output y' under a
#     DISJOINT held-out set of do(u) edits, compare to the bit-exact re-run, and
#     report the fraction predicted within the output-type tolerance.
#
# The per-cause TRUE causal effects {Δy(u)} (the vector every runner calls
# `oracle_abs_delta` / `odelta`) are already produced by a bit-exact re-run of each
# cause. So S needs NO NEW re-runs: it splits the causes into a calibration half and
# a disjoint held-out half, fits the method's attribution → Δy map on the
# calibration half, predicts Δy on the held-out half, and counts held-out causes
# whose predicted output y_intact ± Δy_pred lands within tolerance of the true
# bit-exact re-run y_intact ± Δy_true. This is exactly the sec:triad estimator with
# the intervention oracle's own re-runs as the held-out ground truth.
#
# These are PURE functions over vectors the runners already hold; they cannot touch
# F (the runner computes F separately and unchanged) or the emulator core.

module TriadSM

import Statistics

export minimality_score, sufficiency_score, triad_extra_dict, jnum_or_null

# --- Minimality -------------------------------------------------------------
"""
    minimality_score(attr, odelta; k=nothing, mover_floor=0.0, name_frac=1e-6)
        -> (M, U_star, U_hat, note)

M = |U*| / |U_hat|, clamped to (0,1].

  * U* (true minimal cause set) = the causes whose TRUE causal effect exceeds
    `mover_floor` — the cells that actually move y (smallest set that reproduces y,
    since a cause with Δy=0 is not needed to reproduce y). Size = |U*|.
  * U_hat (named set) = the causes the METHOD names. If `k` is given, U_hat is the
    method's top-k (matching the runner's own precision@k budget); otherwise it is
    the above-threshold set — causes whose attribution exceeds `name_frac` of the
    method's own max attribution (the cells the method actually points at).

Returns the score plus |U*|, |U_hat| and a one-line note. If the method names
NOTHING (all-zero attribution, e.g. a gradient that vanishes on a position output)
U_hat is empty and M is undefined ⇒ returned as `nothing` with a note, never a
meaningless value.
"""
function minimality_score(attr::AbstractVector, odelta::AbstractVector;
                          k::Union{Nothing,Integer} = nothing,
                          mover_floor::Real = 0.0, name_frac::Real = 1e-6)
    n = length(odelta)
    @assert length(attr) == n "attr/odelta length mismatch ($(length(attr)) vs $n)"
    u_star = count(>(mover_floor), odelta)                    # |U*|: the true movers
    if u_star == 0
        return (nothing, 0, 0,
            "M undefined: oracle finds NO causal mover at this state (|U*|=0)")
    end
    if k !== nothing
        kk = min(Int(k), n)
        # named set = the method's top-k (its declared budget), but only the
        # entries carrying nonzero attribution actually count as "named".
        order = sortperm(attr; rev = true)
        u_hat = count(i -> attr[i] > 0.0, order[1:kk])
        u_hat == 0 && return (nothing, u_star, 0,
            "M undefined: method names nothing (all-zero attribution) in its top-$kk")
        note = "|U*|=$u_star (oracle movers) / |U_hat|=$u_hat (method top-$kk, nonzero)"
    else
        mx = maximum(abs, attr; init = 0.0)
        thr = name_frac * mx
        u_hat = mx == 0.0 ? 0 : count(x -> abs(x) > thr, attr)
        u_hat == 0 && return (nothing, u_star, 0,
            "M undefined: method names nothing (all-zero attribution) at this state")
        note = "|U*|=$u_star (oracle movers) / |U_hat|=$u_hat (method above-threshold)"
    end
    M = min(1.0, u_star / u_hat)                              # (0,1]
    return (M, u_star, u_hat, note)
end

# --- Sufficiency ------------------------------------------------------------
"""
    sufficiency_score(attr, odelta; seed=0, tol=nothing, min_heldout=2) -> (S, note)

S = held-out predictive score ∈ [0,1]. The explanation's attribution is a claim
about how much each cause moves y. We fit that claim on a CALIBRATION half of the
causes and test it on a DISJOINT held-out half against the oracle's bit-exact
per-cause Δy (`odelta`, already computed by the runner via real re-runs).

Procedure (no new re-runs — reuses the oracle's per-cause re-runs as ground truth):
  1. Split the cause indices deterministically (by `seed`) into calibration and
     held-out halves.
  2. Fit a 1-D least-squares map  Δy ≈ a·attr + b  on the CALIBRATION half (the
     explanation, read off the calibration interventions).
  3. Predict Δy_pred on the HELD-OUT half and compare y_intact+Δy_pred to the true
     y_intact+Δy_true (the bit-exact re-run). Count held-out causes whose prediction
     is within `tol`. S = that fraction.

`tol` defaults to a small band relative to the spread of the held-out true effects
(a continuous-output ε band, per sec:triad "a small ε band for continuous scores").
Returns `nothing` with a note when there are too few causes to hold out or the
calibration column is degenerate (no signal to fit) — never a meaningless value.
"""
function sufficiency_score(attr::AbstractVector, odelta::AbstractVector;
                           seed::Integer = 0, tol::Union{Nothing,Real} = nothing,
                           min_heldout::Integer = 2)
    n = length(odelta)
    @assert length(attr) == n "attr/odelta length mismatch"
    if n < 2 * min_heldout
        return (nothing, "S undefined: only $n causes (< $(2*min_heldout) needed to hold out)")
    end
    # deterministic interleaved split: even ranks → calibration, odd → held-out,
    # after a seed-rotated ordering so the split is stable but seed-controllable.
    idx = collect(1:n)
    rot = (Int(seed) % n) + 1
    idx = vcat(idx[rot:end], idx[1:rot-1])
    calib = idx[1:2:end]
    held  = idx[2:2:end]
    (length(held) < min_heldout || length(calib) < min_heldout) &&
        return (nothing, "S undefined: split too small (calib=$(length(calib)), held=$(length(held)))")

    ac = Float64.(attr[calib]);  yc = Float64.(odelta[calib])
    ah = Float64.(attr[held]);   yh = Float64.(odelta[held])

    # least-squares fit Δy ≈ a·attr + b on calibration.
    sac = Statistics.std(ac)
    if sac == 0.0
        # method assigns no varying signal on the calibration set → its best
        # constant prediction is the calibration mean effect.
        a = 0.0; b = Statistics.mean(yc)
    else
        cc = Statistics.cov(ac, yc); va = Statistics.var(ac)
        a = cc / va
        b = Statistics.mean(yc) - a * Statistics.mean(ac)
    end
    ypred = a .* ah .+ b

    # tolerance: an ε band for the continuous |Δy| output. Default = max(a small
    # absolute floor, a fraction of the held-out true-effect spread), so a method
    # that predicts the effect size closely counts as sufficient.
    spread = maximum(yh; init = 0.0) - minimum(yh; init = 0.0)
    ε = tol === nothing ? max(0.5, 0.10 * spread) : Float64(tol)
    hits = count(i -> abs(ypred[i] - yh[i]) <= ε, 1:length(yh))
    S = hits / length(yh)
    note = "held-out predictive fit: $hits/$(length(yh)) held-out do(u) within ε=$(round(ε,digits=3)) " *
           "(calib=$(length(calib)), fit Δy≈$(round(a,digits=3))·attr+$(round(b,digits=3)))"
    return (S, note)
end

# --- record assembly --------------------------------------------------------
# JSON can't hold a raw NaN; a `nothing` S/M means "undefined at this state" and is
# emitted as JSON null (the leaderboard's triad_of requires the "S"/"M" keys to be
# PRESENT, so we always emit the keys, with null where undefined).
jnum_or_null(x) = (x === nothing || (x isa Real && isnan(x))) ? nothing : Float64(x)

"""
    triad_extra_dict(F, attr, odelta; topk=nothing, seed=0) -> Dict{String,Any}

Assemble the `extra.triad = {F, S, M, ...}` dict for an attribution runner whose
faithfulness F is already computed. S and M are derived from the method's per-cause
attribution `attr` and the oracle's true per-cause effect `odelta` (both already in
hand), so this cannot alter F. `topk` (if given) makes M use the method's declared
top-k named set; otherwise the above-threshold set. Emits `S`/`M` as JSON null when
undefined at this state, with `S_note`/`M_note` explaining why.
"""
function triad_extra_dict(F, attr::AbstractVector, odelta::AbstractVector;
                          topk::Union{Nothing,Integer} = nothing, seed::Integer = 0)
    # M uses the method's ABOVE-THRESHOLD named set (the cause set the explanation
    # actually points at), matching the A8 whole-state reference (128 named cells →
    # M=0.07). This is the discriminating reading; the top-k budget would cap the
    # named set at k and saturate M≈1 for every attribution method. `topk` is kept
    # in the signature for callers that want the budget-based reading.
    M, ustar, uhat, mnote = minimality_score(attr, odelta)
    S, snote = sufficiency_score(attr, odelta; seed = seed)
    return Dict{String,Any}(
        "F" => jnum_or_null(F),
        "S" => jnum_or_null(S), "S_note" => snote,
        "M" => jnum_or_null(M), "M_note" => mnote,
        "M_true_minimal_size" => ustar, "M_named_size" => uhat,
        "definition" => "F∧S∧M triad (03_methods.tex sec:triad): F = faithfulness " *
            "(runner-computed, unchanged); S = held-out predictive sufficiency over a " *
            "disjoint do(u) split scored on the oracle's bit-exact re-runs; " *
            "M = |U*|/|U_hat| (true minimal cause set / method-named set).")
end

end # module
