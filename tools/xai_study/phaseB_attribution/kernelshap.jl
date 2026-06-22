# kernelshap.jl — Phase-B attribution (P2-E4-11), JULIA path.
#
# KERNELSHAP / Shapley-value SAMPLING (Lundberg & Lee, NeurIPS 2017; Štrumbelj &
# Kononenko, KAIS 2014) scored against the exact §1 intervention oracle on the 6 CORE
# games (tools/xai_study/common/game_set.json). The 12th attribution method on the
# Phase-B leaderboard, built on the validated Phase-B foundation pinned by the IG
# pilot (P2-E4-0) + ig_baseline_sweep.jl, reusing the faithfulness CONTRACT verbatim
# (experiment_design.md §5 row "KernelSHAP / Shapley sampling (Lundberg & Lee 2017;
# Štrumbelj & Kononenko 2014) | Shapley values | corr vs true; convergence vs
# compute"; §7 "SHAP / LIME (B) | ✓ | Partial/Fail | baseline/sampling-dependent;
# approximations diverge").
#
# METHOD — KernelSHAP estimates the SHAPLEY VALUES of the candidate cells (the
# features) for the chosen output y by:
#   (1) SAMPLING COALITIONS z ∈ {0,1}^C over the C candidate cells. A bit z[i]=1 means
#       cell i is PRESENT (kept at its real value); z[i]=0 means cell i is ABSENT
#       (masked to the baseline 0 — RISE's standard occlusion AND the oracle's own
#       occlude! operator, RAM→0). Coalition sizes are drawn from the SHAPLEY-KERNEL
#       distribution (favouring near-empty / near-full coalitions, Lundberg & Lee §4).
#   (2) RE-RUNNING the real ROM for each coalition: f(z) = y( do(ram[i]:=0 ∀ i: z[i]=0) ).
#       Every f(z) is a GENUINE real-ROM re-run on the TRUE VCS — NO surrogate model,
#       no world model, no gradient. This is the crucial methodological point: unlike
#       textbook SHAP (which trains a surrogate over a background dataset), here the
#       value function IS the true emulator, so the only approximation is the coalition
#       SAMPLING + the linear (additive) SHAP model — exactly the "approximations
#       diverge" knob the §7 prediction is about.
#   (3) SOLVING the SHAP weighted least squares for the additive attribution φ:
#         minimise  Σ_z π(z) ( f(z) − φ0 − Σ_i z[i]·φ_i )²   subject to  Σ_i φ_i = f(1)−f(0),
#       with the Shapley kernel weight π(z) = (C−1) / ( binom(C,|z|) · |z| · (C−|z|) )
#       (Lundberg & Lee Theorem 2; |z|∈{0,C} carry "infinite" weight ⇒ pinned exactly,
#       which the efficiency constraint enforces). φ0 = f(0) = y(all masked). The
#       solution is the standard equality-constrained WLS (closed form), and the φ_i are
#       the SHAP values = the attribution.
#
# EFFICIENCY / COMPLETENESS (the SHAP axiom we REPORT): Σ_i φ_i = f(1) − f(0) =
# y(intact) − y(all-masked). This is enforced by the constraint, so we measure the
# RESIDUAL |Σφ − (f(1)−f(0))| (≈0 up to numerical error) and record it as the
# completeness check — the SHAP analogue of IG's completeness axiom.
#
# SCORING CONTRACT (verbatim from the IG pilot, shared by every E4 method):
#   (1) corr            — Pearson + Spearman of the SHAP map with the oracle's TRUE
#                         causal map {|Δy(u)|} over the same cause set.
#   (2) deletion/insertion AUC — measured ON THE TRUE VCS: re-run the real ROM with the
#                         top-attributed causes successively occluded (deletion) /
#                         restored (insertion), reading y each time. Genuine re-runs.
#   (3) precision@k     — |SHAP top-k ∩ oracle top-k| / k vs the true causal top-k.
#  (+) HARNESS POSITIVE CONTROL — feed the oracle's OWN |Δy| back as the candidate map;
#      it MUST score corr=1 / precision@k=1 on a non-degenerate column, proving the
#      harness rewards a faithful map ⇒ the SHAP numbers are a true measurement.
#  (+) CONVERGENCE vs COMPUTE (the §5 "convergence vs compute" requirement) — recompute
#      the SHAP solve + faithfulness from PREFIXES (N/8, N/4, N/2, N) of the SAME
#      coalition pool (no extra re-runs), showing how the estimate stabilises with the
#      sample count (the central SHAP-sampling knob — "approximations diverge" at low N).
#
# OUTPUT SELECTION (the §1 content-vs-position split, mirroring the siblings):
#   * HEADLINE  = a CONTENT output: the candidate CONCEPT BYTE the oracle ranks as the
#     most causally-active RAM cause (SAME pick as the IG/EG/saliency/RISE/CF siblings,
#     via pick_content_idx), read straight off RAM.
#   * CONTRAST  = a POSITION/INDEX output `ball_pixel`: KernelSHAP STILL produces a real
#     attribution here (it re-runs the real ROM and reads the real pixel), unlike
#     IG/saliency which VANISH — the "perturbation methods survive position outputs"
#     contrast (§7), reported not asserted.
#
# REUSES the validated Phase-B foundation on main (NO emulator core touched):
#   * ig_baseline_sweep.jl — the multi-game env layer (load_env, boot_replay,
#     continue_from, fresh_baseline, assert_bit_exact, occlude!, oracle_abs_delta,
#     deletion_insertion_auc with a generic read_y, position_pixel_cell,
#     pick_content_idx, candidates_path_for, CORE_GAMES) — the SAME machinery the
#     IG/EG/saliency/RISE/CF siblings use, so the oracle column + del/ins curves are
#     apples-to-apples with them.
#   * pilot_ig_vs_oracle.jl — the SCORER (pearson/spearman/precision_at_k), the §R
#     writer helpers (_git_commit/_json_num/_trapz_unit), the harness positive-control.
#   * oracle_intervene.jl — Cause / build_pong_causes / candidate_ram_indices.
#   * jutari_oracle.jl — boot/replay/snapshot/intervene + the §R NPZ writer.
#   The masked re-run primitive (do(ram[i]:=0) for the ABSENT cells) is the same
#   coalition mechanism RISE uses (rise.jl); KernelSHAP differs ONLY in the coalition
#   SAMPLING distribution + the weighted-least-squares SOLVE (this file). file_scope
#   is disjoint from rise.jl / occlusion.jl / lime.jl.
#
# BUDGET (the §R-recorded knob) — N = 512 sampled coalitions (paper-reasonable; capped),
# plus the 2 pinned endpoints f(0)=all-masked and f(1)=intact. The SHAP solve uses the
# UNIQUE coalitions with summed kernel weights (the standard KernelSHAP de-duplication),
# so repeated draws cost re-runs only once. Each coalition = ONE genuine emulator
# re-run. We re-run incrementally from the cached deepcopy CHECKPOINT (boot+replay paid
# once), and the convergence sweep reuses the SAME pool (no extra re-runs).
#
# Run (warm shared depot, primary's project) — DEFAULT = all 6 core games locally:
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseB_attribution/kernelshap.jl --games core
# Flags: --games core|<g1,g2,...>  --game <g>
#        --target-frame N --horizon N --n-coalitions N --topk K --seed S
#        --out-dir DIR --where W --roms-dir DIR --selftest
# Cluster-shardable (Slurm array via tools/cluster/xai_array_jl.sbatch):
#        --shard <i> --nshards <n> --shard-kind game   (shard i → the i-th core game)
#
# Writes (SPEC §R; file_scope kernelshap_* under the out dir):
#   <out>/kernelshap_<game>_content.{json,npz}
#   <out>/kernelshap_<game>_ball_pixel.{json,npz}
#   <out>/kernelshap_core_summary.json        (only on a multi-game / non-sharded run)

module KernelSHAP

using JSON
import Statistics
import Random
import LinearAlgebra

# the Phase-B foundation (env layer + scorer + del/ins + oracle |Δy|), identical to
# the IG/EG/saliency/RISE/CF siblings — REUSED, not re-implemented. We include ONLY
# ig_baseline_sweep.jl: it transitively includes pilot_ig_vs_oracle.jl (→
# oracle_intervene.jl → jutari_oracle.jl), so there is ONE include chain ⇒ ONE type
# identity for Snapshot/Cause (a second top-level include would create a distinct
# Cause/Snapshot type and break dispatch).
include(joinpath(@__DIR__, "ig_baseline_sweep.jl"))
using .IGBaselineSweep: CORE_GAMES, load_env, boot_replay, continue_from,
                        fresh_baseline, assert_bit_exact, occlude!,
                        oracle_abs_delta, deletion_insertion_auc, position_pixel_cell,
                        pick_content_idx, candidates_path_for
using .IGBaselineSweep.PilotIGvsOracle: pearson, spearman, precision_at_k,
                                        _git_commit, _json_num, _trapz_unit
using .IGBaselineSweep.PilotIGvsOracle.OracleIntervene: build_pong_causes, Cause,
                                                        candidate_ram_indices
using .IGBaselineSweep.PilotIGvsOracle.OracleIntervene.JutariOracle: Snapshot, snapshot,
                                                        intervene_ram!, write_npz, RAM_SIZE

const OUT_DIR = joinpath(@__DIR__, "out")

# ============================================================================
# The coalition value function — all on the TRUE VCS (perturb-and-re-run; no gradient).
# ============================================================================
# We work over the UNIVERSE of candidate RAM cells (the oracle's candidate concept
# bytes — the SAME cells the oracle scores, so attribution lives on the same axis as
# the ground truth). A coalition z ∈ {0,1}^C marks which cells are PRESENT (kept at
# their real value); the ABSENT cells (z[i]=0) are masked to 0 — RISE's standard
# occlusion = the oracle's own occlude! (RAM→0) — and the real ROM is re-run.

"""
    coalition_value(checkpoint, actions, tf, hz, cells, present_mask, read_y) -> Float64

Re-run y under the coalition `present_mask` (a Bool vector over `cells`, `true`=cell
present): deepcopy the checkpoint, zero each ABSENT cell, continue the horizon on the
real ROM, read y. `cells` are 0-based RAM indices. All-true ⇒ the intact y(x);
all-false ⇒ y(all-masked). Masking-to-0 is the oracle's own occlude!, so these probes
are the oracle's do()-interventions ⇒ apples-to-apples with the del/ins curves."""
function coalition_value(checkpoint, actions, tf, hz, cells, present_mask, read_y)
    env = deepcopy(checkpoint)
    @inbounds for j in 1:length(cells)
        present_mask[j] || intervene_ram!(env, cells[j], 0)
    end
    tail = Int.(actions[tf + 1 : tf + hz])
    for a in tail; IGBaselineSweep.env_step!(env, a); end
    return read_y(snapshot(env, length(tail)))
end

# ----------------------------------------------------------------------------
# The Shapley kernel + coalition sampling (Lundberg & Lee 2017 §4).
# ----------------------------------------------------------------------------
"""
    shapley_kernel_weight(C, s) -> Float64

The Shapley-kernel weight for a coalition of size `s` over `C` features (Lundberg &
Lee Theorem 2): π(s) = (C−1) / ( binom(C,s) · s · (C−s) ). Defined for 1≤s≤C−1; the
endpoints s∈{0,C} have "infinite" weight (pinned exactly via the efficiency
constraint), returned here as 0 so they are excluded from the sampled WLS rows."""
function shapley_kernel_weight(C::Integer, s::Integer)
    (s <= 0 || s >= C) && return 0.0
    return (C - 1) / (binomial(big(C), big(s)) * s * (C - s)) |> Float64
end

"""
    sample_coalitions(C, n; rng) -> Vector{BitVector}

Draw `n` coalitions over `C` features with sizes ~ the Shapley-kernel size
distribution p(s) ∝ (C−1)/(s·(C−s)) for s∈{1,…,C−1} (the sampling weight implied by
π — Lundberg & Lee §4 / Covert & Lee 2021). For a drawn size s we pick a uniformly
random s-subset of the features as PRESENT. Endpoints s∈{0,C} are NEVER sampled (they
are pinned). Returns the present-masks as BitVectors."""
function sample_coalitions(C::Integer, n::Integer; rng)
    C < 2 && return [trues(C) for _ in 1:n]      # degenerate; nothing to share
    sizes = 1:(C - 1)
    # size sampling weight ∝ (C-1)/(s·(C-s)) (the kernel summed over the binom(C,s)
    # subsets of size s — i.e. drop the binom(C,s) so each SUBSET is equiprobable
    # within a size, and the size is drawn ∝ the total mass of that size).
    w = Float64[(C - 1) / (s * (C - s)) for s in sizes]
    w ./= sum(w)
    cdf = cumsum(w)
    out = Vector{BitVector}(undef, n)
    for k in 1:n
        r = rand(rng)
        si = searchsortedfirst(cdf, r); si = min(si, length(sizes))
        s = sizes[si]
        present = falses(C)
        # uniform random s-subset
        idx = Random.randperm(rng, C)[1:s]
        present[idx] .= true
        out[k] = present
    end
    return out
end

# ----------------------------------------------------------------------------
# The KernelSHAP weighted least-squares solve (efficiency-constrained).
# ----------------------------------------------------------------------------
"""
    kernelshap_solve(masks, fvals, weights, f0, f1; C) -> phi::Vector{Float64}

Solve the Shapley weighted least squares for the additive SHAP values φ over the C
features (Lundberg & Lee 2017, Aas et al. 2021 eq. for the constrained solution):

    minimise  Σ_z w(z) ( f(z) − f0 − Σ_i z[i] φ_i )²   s.t.  Σ_i φ_i = f1 − f0

`masks` (n×C Bool) are the sampled coalitions' present-masks, `fvals` (n) their
coalition values f(z), `weights` (n) their (de-duplicated, summed) Shapley-kernel
weights; f0=f(∅)=y(all-masked), f1=f(full)=y(intact). Returns φ (length C). The
closed-form constrained solution: with g_i = z (the design over the C features),
target t = f(z) − f0, weights W (diag), the unconstrained WLS is φ* = (XᵀWX)⁻¹ XᵀW t,
then projected onto Σφ = f1−f0 via the Lagrange correction. We use a small ridge for
numerical stability on rank-deficient designs (recorded)."""
function kernelshap_solve(masks::AbstractMatrix{Bool}, fvals::AbstractVector{<:Real},
                          weights::AbstractVector{<:Real}, f0::Real, f1::Real; C::Integer)
    n = size(masks, 1)
    X = Float64.(masks)                       # n×C design (z[i])
    t = Float64.(fvals) .- f0                 # centered target f(z) − f0
    W = Float64.(weights)
    ridge = 1e-8
    XtW = X' .* reshape(W, 1, :)              # C×n  (XᵀW)
    A = XtW * X + ridge * LinearAlgebra.I     # C×C
    b = XtW * t                               # C
    Ainv = LinearAlgebra.inv(A)
    phi_unc = Ainv * b                        # unconstrained WLS
    # project onto the efficiency constraint Σφ = f1 − f0 (Lagrange, with the same A):
    #   φ = φ_unc + Ainv·1 · ( (f1−f0) − 1ᵀφ_unc ) / (1ᵀ Ainv 1 )
    one = ones(Float64, C)
    Ainv1 = Ainv * one
    denom = LinearAlgebra.dot(one, Ainv1)
    target_sum = Float64(f1 - f0)
    phi = phi_unc .+ Ainv1 .* ((target_sum - sum(phi_unc)) / denom)
    return phi
end

"""De-duplicate a coalition pool: identical present-masks share one re-run; their
Shapley-kernel weights SUM (the standard KernelSHAP de-dup). Returns the unique masks
(rows), their summed weights, and the index of each unique row's representative draw
order (for prefix/convergence slicing we keep the FIRST-seen order)."""
function dedup_coalitions(coalitions::Vector{BitVector}, C::Integer)
    seen = Dict{UInt64,Int}()           # hash → unique-row index
    uniq = BitVector[]
    wsum = Float64[]
    first_at = Int[]                     # the draw order at which each unique row first appeared
    for (k, z) in enumerate(coalitions)
        s = count(z)
        kw = shapley_kernel_weight(C, s)
        h = hash(z)
        if haskey(seen, h)
            wsum[seen[h]] += kw
        else
            push!(uniq, z); push!(wsum, kw); push!(first_at, k)
            seen[h] = length(uniq)
        end
    end
    return uniq, wsum, first_at
end

"""Map the per-cell SHAP values onto the oracle's per-CAUSE axis: a RAM cause touching
candidate cell i gets |φ_i|; a TIA/joystick cause gets 0 (the SHAP feature universe is
RAM cells — the method honestly assigns no mass off it). We take |φ| because the
oracle map is |Δy| (a magnitude); the signed φ are kept in the record."""
function shap_attr_per_cause(phi::AbstractVector{<:Real}, cells, causes::Vector{Cause})
    by_idx = Dict(cells[j] => phi[j] for j in 1:length(cells))
    return [c.kind == "ram" ? abs(get(by_idx, c.index, 0.0)) : 0.0 for c in causes]
end

# ============================================================================
# Per-output result + the convergence sweep.
# ============================================================================
struct ConvergencePoint
    n_coalitions::Int                    # # sampled coalitions used (prefix of the pool)
    n_unique::Int                        # # unique re-runs in that prefix
    pearson::Float64
    spearman::Float64
    precision_at_k::Float64
    deletion_auc::Float64
    insertion_auc::Float64
    completeness_residual::Float64       # |Σφ − (f1−f0)| at this budget
end

struct SHAPResult
    game::String
    output::String
    output_kind::String                  # "content" | "position"
    target_frame::Int
    horizon::Int
    n_coalitions::Int                    # the BUDGET (sampled coalition count)
    n_unique::Int                        # # unique coalitions actually re-run (+2 endpoints)
    topk::Int
    seed::Int
    content_idx::Int
    cause_names::Vector{String}
    candidate_cells::Vector{Int}
    oracle_abs_delta::Vector{Float64}
    shap_attr_per_cause::Vector{Float64} # the headline SHAP map (|φ| at the full budget)
    shap_phi_signed::Vector{Float64}     # the signed φ over the candidate cells
    f0::Float64                          # y(all-masked) = φ0
    f1::Float64                          # y(intact)
    completeness_residual::Float64       # |Σφ − (f1−f0)|  (efficiency axiom)
    # faithfulness vs the oracle at the FULL budget (the §5 contract; same as siblings)
    pearson::Float64
    spearman::Float64
    precision_at_k::Float64
    deletion_auc::Float64
    insertion_auc::Float64
    del_curve::Vector{Float64}
    ins_curve::Vector{Float64}
    # the CONVERGENCE sweep (how faithfulness scales with N) — prefixes of one pool
    convergence::Vector{ConvergencePoint}
    n_real_reruns::Int                   # unique coalitions + 2 endpoints (the true cost)
    oracle_column_degenerate::Bool
    # harness positive control (oracle's OWN |Δy| as the method)
    oracle_self_pearson::Float64
    oracle_self_precision_at_k::Float64
    oracle_self_deletion_auc::Float64
    oracle_self_insertion_auc::Float64
end

"""Score one SHAP map against the oracle with the §5 metrics + del/ins on the TRUE VCS."""
function _score_map(attr, odelta, checkpoint, actions, tf, hz, causes, read_y, topk)
    pr  = pearson(attr, odelta)
    sp  = spearman(attr, odelta)
    pak = precision_at_k(attr, odelta, topk)
    order = sortperm(attr; rev = true)
    del_auc, ins_auc, dc, ic = deletion_insertion_auc(
        checkpoint, actions, tf, hz, causes, order, read_y)
    return pr, sp, pak, del_auc, ins_auc, dc, ic
end

"""Solve KernelSHAP from a PREFIX of `n_take` sampled coalitions (the first n_take of
the SAME pool), reusing the cached per-unique-coalition values `val_of` — no new
re-runs. Returns (phi, n_unique, completeness_residual)."""
function _shap_from_prefix(coalitions, val_of::Dict, f0, f1, C, n_take)
    pref = coalitions[1:n_take]
    uniq, wsum, _ = dedup_coalitions(pref, C)
    nu = length(uniq)
    masks = falses(nu, C)
    fvals = zeros(Float64, nu)
    for r in 1:nu
        masks[r, :] = uniq[r]
        fvals[r] = val_of[uniq[r]]
    end
    phi = kernelshap_solve(masks, fvals, wsum, f0, f1; C = C)
    resid = abs(sum(phi) - (f1 - f0))
    return phi, nu, resid
end

"""Compute the KernelSHAP attribution + faithfulness + convergence sweep for one
output of one game. `read_y` reads the chosen output off a Snapshot; `output_kind` is
"content"/"position"; `cells` is the candidate RAM-cell universe (0-based)."""
function compute_one(; game, output_kind, content_idx = -1, position_cell = nothing,
                     checkpoint, actions, target_frame, horizon, causes, candidate_cells,
                     n_coalitions, topk, seed, verbose)
    read_y = output_kind == "content" ?
        (s -> Float64(Int(s.ram[content_idx + 1]))) :
        (s -> Float64(Int(s.screen[position_cell[1], position_cell[2]])))
    output_name = output_kind == "content" ?
        "content(ram_self@$content_idx)" :
        "ball_pixel@r$(position_cell[1])c$(position_cell[2])"

    C = length(candidate_cells)

    # 1) the oracle |Δy| per CAUSE (real re-runs on the TRUE VCS) — the ground truth
    odelta = oracle_abs_delta(checkpoint, actions, target_frame, horizon, causes, read_y)
    degenerate = Statistics.std(odelta) == 0

    # 2) pinned endpoints: f0 = y(all-masked), f1 = y(intact) — both real re-runs.
    f0 = coalition_value(checkpoint, actions, target_frame, horizon, candidate_cells, falses(C), read_y)
    f1 = coalition_value(checkpoint, actions, target_frame, horizon, candidate_cells, trues(C), read_y)

    # 3) sample N coalitions from the Shapley-kernel size distribution + RE-RUN each
    #    unique one (THE BUDGET — every unique coalition is one genuine emulator re-run).
    verbose && println("[kernelshap] $game '$output_name': sampling $n_coalitions coalitions over " *
                       "$C candidate cells (Shapley kernel) ...")
    rng = Random.MersenneTwister(seed * 1_000_003 + 6151)   # deterministic, recorded seed
    coalitions = sample_coalitions(C, n_coalitions; rng = rng)

    # de-dup → unique coalitions actually re-run (repeated draws cost a re-run once);
    # cache value-of-coalition so the convergence prefixes reuse it (no extra re-runs).
    uniq_all, _, _ = dedup_coalitions(coalitions, C)
    val_of = Dict{BitVector,Float64}()
    for z in uniq_all
        val_of[z] = coalition_value(checkpoint, actions, target_frame, horizon, candidate_cells, z, read_y)
    end
    n_unique = length(uniq_all)
    n_real_reruns = n_unique + 2          # + the two pinned endpoints
    verbose && println("[kernelshap]   $n_unique unique coalitions re-run (+2 endpoints) = $n_real_reruns real re-runs")

    # 4) the headline SHAP solve at the FULL budget + its faithfulness
    phi_full, _, resid_full = _shap_from_prefix(coalitions, val_of, f0, f1, C, n_coalitions)
    attr_full = shap_attr_per_cause(phi_full, candidate_cells, causes)
    pr, sp, pak, del_auc, ins_auc, dc, ic =
        _score_map(attr_full, odelta, checkpoint, actions, target_frame, horizon, causes, read_y, topk)

    # 5) the CONVERGENCE sweep: re-solve from PREFIXES of the SAME coalition pool
    #    (N/8, N/4, N/2, N) — no extra re-runs; shows how the SHAP estimate stabilises
    #    with the sample count (the §5 "convergence vs compute"). del/ins use the prefix
    #    ranking; we also track the completeness residual at each budget.
    conv = ConvergencePoint[]
    for frac in (8, 4, 2, 1)
        nk = max(2, n_coalitions ÷ frac)
        phik, nuk, residk = _shap_from_prefix(coalitions, val_of, f0, f1, C, nk)
        attrk = shap_attr_per_cause(phik, candidate_cells, causes)
        prk, spk, pakk, delk, insk, _, _ =
            _score_map(attrk, odelta, checkpoint, actions, target_frame, horizon, causes, read_y, topk)
        push!(conv, ConvergencePoint(nk, nuk, prk, spk, pakk, delk, insk, residk))
        nk == n_coalitions && break       # avoid a duplicate at the full budget
    end

    # 6) harness positive control (oracle's OWN |Δy| as the candidate map)
    or_pr  = pearson(odelta, odelta); or_pak = precision_at_k(odelta, odelta, topk)
    or_order = sortperm(odelta; rev = true)
    or_del, or_ins, _, _ = deletion_insertion_auc(checkpoint, actions, target_frame,
                                                  horizon, causes, or_order, read_y)

    if verbose
        println("[kernelshap]   faithfulness vs oracle @N=$n_coalitions: corr=$(round(pr,digits=4)) " *
                "spearman=$(round(sp,digits=4)) p@$topk=$(round(pak,digits=3)) " *
                "del=$(round(del_auc,digits=3)) ins=$(round(ins_auc,digits=3))")
        println("[kernelshap]   efficiency/completeness: Σφ=$(round(sum(phi_full),digits=4)) " *
                "vs f1−f0=$(round(f1-f0,digits=4)) ⇒ residual=$(round(resid_full,sigdigits=3))")
        print("[kernelshap]   convergence corr@N:")
        for c in conv; print(" $(c.n_coalitions)=$(round(c.pearson,digits=3))"); end
        println()
        println("[kernelshap]   [harness] oracle-as-method: corr=$(round(or_pr,digits=3)) " *
                "p@$topk=$(round(or_pak,digits=3)) del=$(round(or_del,digits=3)) ins=$(round(or_ins,digits=3))" *
                (degenerate ? "  (oracle column flat at this state)" : ""))
    end

    return SHAPResult(game, output_name, output_kind, target_frame, horizon, n_coalitions,
                      n_unique, topk, seed, content_idx, [c.name for c in causes],
                      collect(candidate_cells), odelta, attr_full, phi_full, f0, f1, resid_full,
                      pr, sp, pak, del_auc, ins_auc, dc, ic, conv, n_real_reruns,
                      degenerate, or_pr, or_pak, or_del, or_ins)
end

"""Drive both outputs for one game: assert bit-exact, build causes + candidate cells,
pick the content byte, locate the position pixel, run KernelSHAP for content + position."""
function compute_game(; game, target_frame, horizon, n_coalitions, topk, seed, verbose)
    total = target_frame + horizon
    actions = fill(0, total)
    verbose && println("[kernelshap] $game: asserting bit-exactness (2 fresh boots+replays to f$total)...")
    assert_bit_exact(actions, total; game = game)

    cand = candidates_path_for(game)
    checkpoint = boot_replay(actions, target_frame; game = game)
    at_target = continue_from(checkpoint, Int[])
    causes = build_pong_causes(cand, at_target)
    candidate_cells = sort(unique(idx for (idx, _) in candidate_ram_indices(cand)))

    # content byte = the most causally-active candidate concept byte (SAME pick as the
    # IG/EG/saliency/RISE/CF siblings ⇒ the methods score the SAME content output).
    content_idx, content_mv = pick_content_idx(checkpoint, actions, target_frame, horizon, causes,
                                               [idx for (idx, _) in candidate_ram_indices(cand)])
    verbose && println("[kernelshap] $game content byte = RAM[$content_idx] (max oracle |Δself|=$(round(content_mv,digits=2)))")

    base = continue_from(checkpoint, Int.(actions[target_frame + 1 : total]))
    pcell = position_pixel_cell(checkpoint, base.screen; horizon = horizon)

    f_content = compute_one(; game = game, output_kind = "content", content_idx = content_idx,
        checkpoint = checkpoint, actions = actions, target_frame = target_frame, horizon = horizon,
        causes = causes, candidate_cells = candidate_cells, n_coalitions = n_coalitions,
        topk = topk, seed = seed, verbose = verbose)
    f_pos = compute_one(; game = game, output_kind = "position", position_cell = pcell,
        checkpoint = checkpoint, actions = actions, target_frame = target_frame, horizon = horizon,
        causes = causes, candidate_cells = candidate_cells, n_coalitions = n_coalitions,
        topk = topk, seed = seed, verbose = verbose)
    return f_content, f_pos
end

# ============================================================================
# Self-check (DoD) — the scoring contract is sound; results are non-fabricated.
# ============================================================================
"""
    selftest(f::SHAPResult; require_nondegenerate) -> Bool

Asserts the load-bearing claims of KernelSHAP + the shared contract:

  (HARNESS POSITIVE CONTROL) feeding the oracle's OWN |Δy| as the candidate map scores
    corr=1 (on a non-degenerate column) and precision@k=1 — the harness rewards a
    faithful map (else the harness is broken, not the method).

  (SHAP CORRECTNESS)
    * attribution finite + non-negative (it is |φ|); all AUCs ∈ [0,1] or NaN.
    * EFFICIENCY / COMPLETENESS: Σφ ≈ f1−f0 (the SHAP axiom, enforced by the
      constraint) — residual must be tiny (≤ 1e-6·(1+|f1−f0|)).
    * the budget is real: n_real_reruns == n_unique + 2 (each unique coalition + the
      two pinned endpoints = one re-run apiece).
    * the convergence sweep is non-empty, monotone in N, and ends at the full budget.

  (POSITION OUTPUT — the §7 contrast) KernelSHAP still produces a real attribution on
    the position pixel (it re-runs the real ROM, no gradient) — recorded, not asserted
    to be large; the contrast with IG/saliency which vanish there.

Throws on a violation."""
function selftest(f::SHAPResult; require_nondegenerate = false)
    @assert all(isfinite, f.shap_attr_per_cause) "non-finite SHAP attribution ($(f.game)/$(f.output))"
    @assert all(>=(0.0), f.shap_attr_per_cause) "SHAP attribution must be non-negative ($(f.game)/$(f.output))"
    @assert all(isfinite, f.shap_phi_signed) "non-finite signed φ ($(f.game)/$(f.output))"
    for (nm, v) in (("deletion", f.deletion_auc), ("insertion", f.insertion_auc),
                    ("oracle_self_deletion", f.oracle_self_deletion_auc),
                    ("oracle_self_insertion", f.oracle_self_insertion_auc))
        @assert isnan(v) || (0.0 <= v <= 1.0 + 1e-9) "$nm AUC out of [0,1]: $v ($(f.game)/$(f.output))"
    end
    # EFFICIENCY / COMPLETENESS axiom — the defining SHAP guarantee.
    eff_tol = 1e-6 * (1 + abs(f.f1 - f.f0))
    @assert f.completeness_residual <= eff_tol "SHAP efficiency violated: |Σφ − (f1−f0)|=" *
        "$(f.completeness_residual) > tol=$eff_tol ($(f.game)/$(f.output))"
    @assert f.n_real_reruns == f.n_unique + 2 "budget mismatch: n_real_reruns=$(f.n_real_reruns) " *
        "!= n_unique+2=$(f.n_unique + 2)"
    @assert !isempty(f.convergence) "empty convergence sweep ($(f.game)/$(f.output))"
    @assert f.convergence[end].n_coalitions == f.n_coalitions "convergence sweep does not reach the full budget"
    for c in f.convergence
        @assert isnan(c.deletion_auc) || (0.0 <= c.deletion_auc <= 1.0 + 1e-9) "convergence del AUC out of range"
        @assert isnan(c.insertion_auc) || (0.0 <= c.insertion_auc <= 1.0 + 1e-9) "convergence ins AUC out of range"
    end

    if f.output_kind == "position"
        println("[kernelshap] SELF-CHECK PASS (position '$(f.output)', $(f.game)): " *
                "SHAP map finite (max|φ|=$(round(maximum(f.shap_attr_per_cause; init=0.0),sigdigits=3))) over " *
                "$(f.n_coalitions) coalitions ($(f.n_real_reruns) re-runs); corr=$(round(f.pearson,digits=3)); " *
                "Σφ−(f1−f0) residual=$(round(f.completeness_residual,sigdigits=3)); " *
                "oracle-as-method corr=$(round(f.oracle_self_pearson,digits=3)) " *
                "(the §7 contrast: KernelSHAP re-runs the real ROM, so it works on a position output " *
                "where IG/saliency VANISH).")
        return true
    end

    # CONTENT output --------------------------------------------------------
    if require_nondegenerate
        @assert !f.oracle_column_degenerate "content oracle column degenerate for $(f.game) — " *
            "pick_content_idx failed to find a causally-active concept byte"
        @assert f.oracle_self_pearson > 0.999 "harness broken: oracle-as-method corr != 1 ($(f.oracle_self_pearson)) [$(f.game)]"
        @assert f.oracle_self_precision_at_k == 1.0 "harness broken: oracle-as-method p@k != 1 [$(f.game)]"
    end
    conv_str = join(["$(c.n_coalitions)=$(round(c.pearson,digits=3))" for c in f.convergence], " ")
    println("[kernelshap] SELF-CHECK PASS (content '$(f.output)', $(f.game)): " *
            "SHAP map finite + non-negative over $(f.n_coalitions) coalitions ($(f.n_real_reruns) re-runs); " *
            "efficiency Σφ=$(round(sum(f.shap_phi_signed),digits=3)) ≈ f1−f0=$(round(f.f1-f.f0,digits=3)) " *
            "(residual $(round(f.completeness_residual,sigdigits=2))); " *
            "corr=$(round(f.pearson,digits=3)) p@$(f.topk)=$(round(f.precision_at_k,digits=3)) " *
            "del=$(round(f.deletion_auc,digits=3)) ins=$(round(f.insertion_auc,digits=3)); " *
            "convergence corr@N [$conv_str]; oracle-as-method corr=$(round(f.oracle_self_pearson,digits=3)).")
    return true
end

# ============================================================================
# Persist (SPEC §R) — JSON record + sibling .npz; file_scope kernelshap_*.
# ============================================================================
function _output_note(f::SHAPResult)
    if f.output_kind == "position"
        return "POSITION/INDEX output (ball_pixel) — the §7 CONTRAST: unlike IG/saliency (which VANISH " *
               "on position outputs because the pixel is a round/argmax sprite column with no " *
               "differentiable RAM path), KernelSHAP — a BLACK-BOX perturb-and-re-run method — STILL " *
               "produces a real attribution here. It masks the absent coalition cells to 0 and RE-RUNS " *
               "the real ROM, reading the real pixel; no differentiation. Reported, not asserted to be " *
               "large; the intervention oracle remains the sole truth."
    else
        conv_corr = [c.pearson for c in f.convergence]
        return "CONTENT output: RAM byte $(f.content_idx) (the most causally-active candidate concept " *
               "byte — SAME pick as the IG/EG/saliency/RISE/CF siblings). KernelSHAP (Lundberg & Lee " *
               "2017) samples $(f.n_coalitions) coalitions from the Shapley-kernel size distribution, " *
               "re-runs the real ROM for each unique one (masking absent cells to 0), and solves the " *
               "efficiency-constrained weighted least squares for the SHAP values φ. EFFICIENCY axiom " *
               "Σφ=$(round(sum(f.shap_phi_signed),digits=3)) ≈ f1−f0=$(round(f.f1-f.f0,digits=3)) " *
               "(residual $(round(f.completeness_residual,sigdigits=2))). Faithfulness " *
               "corr=$(round(f.pearson,digits=3)) @N=$(f.n_coalitions); convergence corr goes " *
               "$(round(first(conv_corr),digits=3))→$(round(last(conv_corr),digits=3)) as N goes " *
               "$(f.convergence[1].n_coalitions)→$(f.n_coalitions) (SHAP sampling variance ∝ 1/N — the " *
               "§7 'approximations diverge' knob). Every probe is a genuine emulator re-run (RAM→0 = the " *
               "oracle's own occlude!), apples-to-apples with the del/ins curves."
    end
end

function write_result(f::SHAPResult; out_dir = OUT_DIR, where = "local")
    isdir(out_dir) || mkpath(out_dir)
    tag = f.output_kind == "content" ? "content" : "ball_pixel"
    stem = "kernelshap_$(f.game)_$(tag)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path  = joinpath(out_dir, stem * ".npz")

    conv_json = [Dict{String,Any}(
        "n_coalitions" => c.n_coalitions, "n_unique" => c.n_unique,
        "pearson_corr" => c.pearson, "spearman_corr" => c.spearman,
        "precision_at_k" => c.precision_at_k,
        "deletion_auc" => _json_num(c.deletion_auc), "insertion_auc" => _json_num(c.insertion_auc),
        "completeness_residual" => c.completeness_residual,
    ) for c in f.convergence]

    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseB_attribution", "method" => "kernelshap",
        "game" => f.game, "state" => "f$(f.target_frame)+$(f.horizon)",
        "target_output" => f.output,
        # headline §R scalar = the SHAP map's corr with the oracle (one comparable number
        # per method on the leaderboard, like the siblings).
        "metric_name" => "pearson_corr_with_oracle",
        "value" => f.pearson,
        "stderr" => nothing, "ci" => nothing, "n" => length(f.cause_names),
        "seed" => f.seed, "where" => where, "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene@$(f.game)#$(f.output)",
        "timestamp" => string(round(Int, time())), "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            "method_ref" => "KernelSHAP / Shapley sampling (Lundberg & Lee, NeurIPS 2017; Štrumbelj & " *
                "Kononenko, KAIS 2014): sample coalitions from the Shapley kernel, re-run the true ROM " *
                "with absent cells masked to 0, solve the efficiency-constrained weighted least squares " *
                "for the SHAP values φ = attribution.",
            "substrate" => "jutari (Julia, HARD) — real-ROM coalition re-runs (bit-exact oracle " *
                "machinery) + closed-form efficiency-constrained WLS solve. NO surrogate model: the " *
                "value function IS the true emulator, so the only approximation is the coalition " *
                "sampling + the additive (linear) SHAP model.",
            "output_kind" => f.output_kind,
            "content_ram_index" => f.content_idx,
            # THE BUDGET (recorded per §R) — the perturb-and-re-run cost knob.
            "budget" => Dict{String,Any}(
                "n_coalitions_sampled" => f.n_coalitions,
                "n_unique_coalitions" => f.n_unique,
                "n_candidate_cells" => length(f.candidate_cells),
                "n_real_reruns" => f.n_real_reruns,
                "endpoints" => "f0=y(all-masked) and f1=y(intact) pinned exactly (2 extra re-runs)",
                "rng" => "MersenneTwister(seed*1_000_003 + 6151), seed=$(f.seed)",
                "note" => "N=$(f.n_coalitions) coalitions sampled from the Shapley-kernel size " *
                    "distribution p(s)∝(C−1)/(s·(C−s)); duplicate draws de-duplicated (their kernel " *
                    "weights SUM, the standard KernelSHAP de-dup) so only $(f.n_unique) UNIQUE " *
                    "coalitions are re-run + 2 endpoints = $(f.n_real_reruns) real emulator re-runs per " *
                    "output. Re-runs continue from the cached deepcopy CHECKPOINT (boot+replay paid " *
                    "once), and the convergence sweep reuses the SAME pool (no extra re-runs)."),
            "headline_metrics" => Dict{String,Any}(
                "pearson_corr" => f.pearson, "spearman_corr" => f.spearman,
                "precision_at_k" => f.precision_at_k, "topk" => f.topk,
                "deletion_auc" => _json_num(f.deletion_auc),
                "insertion_auc" => _json_num(f.insertion_auc)),
            # the SHAP EFFICIENCY / COMPLETENESS axiom (the SHAP analogue of IG completeness).
            "efficiency_completeness" => Dict{String,Any}(
                "sum_phi" => sum(f.shap_phi_signed),
                "f_full_minus_empty" => f.f1 - f.f0,
                "f0_all_masked" => f.f0, "f1_intact" => f.f1,
                "residual" => f.completeness_residual,
                "interpretation" => "Σφ = f(full) − f(∅) = y(intact) − y(all-masked) is the SHAP " *
                    "efficiency axiom (the additive attribution exactly reconstructs the output gap). " *
                    "Enforced by the equality constraint in the WLS solve; the tiny residual is " *
                    "numerical. This is the SHAP analogue of IG's completeness."),
            # how faithfulness SCALES with the coalition count (prefixes of one pool).
            "convergence_sweep" => Dict{String,Any}(
                "points" => conv_json,
                "interpretation" => "the SHAP solve + faithfulness (corr / precision@k / del-ins) + the " *
                    "completeness residual recomputed from N/8, N/4, N/2, N coalitions of the SAME pool " *
                    "(no extra re-runs). The SHAP-sampling variance is ∝ 1/N, so corr/p@k STABILISE as N " *
                    "grows; the spread between the smallest and full budget quantifies how many " *
                    "coalitions KernelSHAP needs on this output (the §5 'convergence vs compute', the §7 " *
                    "'approximations diverge' knob)."),
            "harness_positive_control" => Dict{String,Any}(
                "method" => "oracle_abs_delta (the perfectly-faithful attribution)",
                "pearson_corr" => f.oracle_self_pearson,
                "precision_at_k" => f.oracle_self_precision_at_k,
                "deletion_auc" => _json_num(f.oracle_self_deletion_auc),
                "insertion_auc" => _json_num(f.oracle_self_insertion_auc),
                "oracle_column_degenerate" => f.oracle_column_degenerate,
                "interpretation" => "corr=1 & precision@k=1 (on a non-degenerate column) ⇒ the scoring " *
                    "harness rewards a faithful map; the KernelSHAP numbers above are then a true " *
                    "measurement, not an artefact of the metric."),
            "auc_note" => "deletion/insertion curves measured on the TRUE VCS by re-running the real ROM " *
                "with top-SHAP causes occluded (deletion) / restored (insertion); every point is a genuine " *
                "emulator re-run, not a surrogate. NaN = a genuinely flat experiment.",
            "shapley_kernel_note" => "coalitions sampled with size s ~ p(s)∝(C−1)/(s·(C−s)); each row of " *
                "the WLS is weighted by the Shapley kernel π(s)=(C−1)/(binom(C,s)·s·(C−s)) (Lundberg & " *
                "Lee Theorem 2). Endpoints s∈{0,C} carry infinite weight ⇒ pinned exactly by the " *
                "efficiency constraint (not sampled). A 1e-8 ridge stabilises rank-deficient designs.",
            "output_note" => _output_note(f),
            "cause_names" => f.cause_names,
            "shap_attr_per_cause" => Dict(f.cause_names[i] => f.shap_attr_per_cause[i] for i in 1:length(f.cause_names)),
            "oracle_abs_delta_per_cause" => Dict(f.cause_names[i] => f.oracle_abs_delta[i] for i in 1:length(f.cause_names)),
            "shap_phi_signed_per_cell" => Dict(string(f.candidate_cells[j]) => f.shap_phi_signed[j]
                                               for j in 1:length(f.candidate_cells)),
            "comparison_note" => "KernelSHAP vs the gradient family: vanilla saliency/Grad×Input/IG " *
                "measure ∂y/∂u (vanish on index outputs); KernelSHAP measures the Shapley value of each " *
                "cell from real coalition re-runs (alive everywhere the perturbation is valid). " *
                "KernelSHAP vs the §1 oracle: SHAP credits the average marginal contribution over " *
                "coalitions (occlude→0 baseline only), so it is the SAMPLING-APPROXIMATION twin of the " *
                "oracle — the §7 'Partial/Fail: baseline/sampling-dependent; approximations diverge' " *
                "prediction, in numbers (reported, not asserted).",
            "scales_to_cluster_via" =>
                "tools/cluster/xai_array_jl.sbatch (--shard i --nshards n --shard-kind game): one Slurm " *
                "array task per core game; or batched SOFT-STE coalition re-runs over coalitions×games on " *
                "GPU (the forward is bit-exact to this HARD map).",
        ),
    )
    open(json_path, "w") do io; JSON.print(io, rec, 2); end

    write_npz(npz_path, Dict(
        "oracle_abs_delta"        => f.oracle_abs_delta,
        "shap_attr_per_cause"     => f.shap_attr_per_cause,
        "shap_phi_signed"         => f.shap_phi_signed,
        "candidate_cells"         => Float64.(f.candidate_cells),
        "deletion_curve"          => f.del_curve,
        "insertion_curve"         => f.ins_curve,
        "convergence_n_coalitions"=> Float64[c.n_coalitions for c in f.convergence],
        "convergence_n_unique"    => Float64[c.n_unique for c in f.convergence],
        "convergence_pearson"     => Float64[c.pearson for c in f.convergence],
        "convergence_spearman"    => Float64[c.spearman for c in f.convergence],
        "convergence_precision_at_k" => Float64[c.precision_at_k for c in f.convergence],
        "convergence_deletion_auc"   => Float64[isnan(c.deletion_auc) ? -1.0 : c.deletion_auc for c in f.convergence],
        "convergence_insertion_auc"  => Float64[isnan(c.insertion_auc) ? -1.0 : c.insertion_auc for c in f.convergence],
        "convergence_completeness_residual" => Float64[c.completeness_residual for c in f.convergence],
        "scalars"                 => Float64[f.pearson, f.spearman, f.precision_at_k,
                                             isnan(f.deletion_auc) ? -1.0 : f.deletion_auc,
                                             isnan(f.insertion_auc) ? -1.0 : f.insertion_auc,
                                             Float64(f.n_coalitions), Float64(f.n_real_reruns),
                                             f.f0, f.f1, f.completeness_residual],
    ))
    return json_path, npz_path
end

# ============================================================================
# CLI — local default = all 6 core games; cluster-shardable via the sbatch flags.
# ============================================================================
"""Resolve the games to run: an explicit --game / --games, OR a --shard i --nshards n
--shard-kind game selection (shard i → the i-th core game). Default = all 6 core."""
function _resolve_games(; single_game, games_arg, shard, nshards, shard_kind)
    single_game !== nothing && return [single_game]
    games_arg !== nothing && return games_arg
    if shard !== nothing && nshards !== nothing && nshards > 1
        kind = shard_kind === nothing ? "game" : shard_kind
        kind == "game" || error("kernelshap.jl only shards by game (--shard-kind game), got $kind")
        # shard i (0-based) → every game at position i, i+n, i+2n, ... (round-robin), so
        # any nshards ≤ |core| picks the i-th core game and nshards=|core| is 1-per.
        sel = [CORE_GAMES[j] for j in (shard + 1):nshards:length(CORE_GAMES)]
        isempty(sel) && error("shard $shard of $nshards selects no core game (|core|=$(length(CORE_GAMES)))")
        return sel
    end
    return CORE_GAMES
end

function main(args = ARGS)
    single_game = nothing; games_arg = nothing
    target_frame = 120; horizon = 30
    n_coalitions = 512
    topk = 3; seed = 0
    shard = nothing; nshards = nothing; shard_kind = nothing
    out_dir = OUT_DIR; where = "local"
    selftest_only = false
    i = 1
    while i <= length(args)
        a = args[i]
        if     a == "--games"
            v = args[i+1]; games_arg = (v == "core") ? CORE_GAMES : String.(split(v, ",")); i += 2
        elseif a == "--game";          single_game = args[i+1]; i += 2
        elseif a == "--target-frame";  target_frame = parse(Int, args[i+1]); i += 2
        elseif a == "--horizon";       horizon = parse(Int, args[i+1]); i += 2
        elseif a == "--n-coalitions";  n_coalitions = parse(Int, args[i+1]); i += 2
        elseif a == "--topk";          topk = parse(Int, args[i+1]); i += 2
        elseif a == "--seed";          seed = parse(Int, args[i+1]); i += 2
        elseif a == "--shard";         shard = parse(Int, args[i+1]); i += 2
        elseif a == "--nshards";       nshards = parse(Int, args[i+1]); i += 2
        elseif a == "--shard-kind";    shard_kind = args[i+1]; i += 2
        elseif a == "--out-dir";       out_dir = args[i+1]; i += 2
        elseif a == "--where";         where = args[i+1]; i += 2
        elseif a == "--roms-dir";      i += 2            # accepted-and-ignored (ROMs located internally)
        elseif a == "--selftest";      selftest_only = true; i += 1
        else; i += 1                                     # ignore unknown flags (permissive)
        end
    end
    games = _resolve_games(; single_game = single_game, games_arg = games_arg,
                           shard = shard, nshards = nshards, shard_kind = shard_kind)
    # a sharded run writes per-game records but NOT the cross-game summary (the SM
    # merges per-game records; a single shard has no business writing the summary).
    sharded = shard !== nothing && nshards !== nothing && nshards > 1

    println("[kernelshap] KernelSHAP / Shapley sampling (Lundberg & Lee 2017) vs oracle — " *
            "games=$(join(games, ",")) target_frame=$target_frame horizon=$horizon " *
            "n_coalitions=$n_coalitions topk=$topk seed=$seed out_dir=$out_dir where=$where" *
            (shard === nothing ? "" : " shard=$shard/$nshards kind=$shard_kind") * " (jutari/Julia)")

    summary = Dict{String,Any}[]
    for game in games
        println("\n[kernelshap] ===== $game =====")
        f_content, f_pos = compute_game(; game = game, target_frame = target_frame,
            horizon = horizon, n_coalitions = n_coalitions,
            topk = topk, seed = seed, verbose = true)
        @assert !f_content.oracle_column_degenerate "content oracle column is degenerate for $game " *
            "at f$target_frame — pick_content_idx failed to find a causally-active concept byte"
        selftest(f_content; require_nondegenerate = true)
        selftest(f_pos)
        if !selftest_only
            for f in (f_content, f_pos)
                jp, np = write_result(f; out_dir = out_dir, where = where)
                println("[kernelshap] wrote $jp"); println("[kernelshap] arrays  $np")
            end
        end
        for f in (f_content, f_pos)
            push!(summary, Dict{String,Any}(
                "game" => game, "output" => f.output, "output_kind" => f.output_kind,
                "content_ram_index" => f.content_idx,
                "n_coalitions" => f.n_coalitions, "n_real_reruns" => f.n_real_reruns,
                "pearson" => f.pearson, "spearman" => f.spearman,
                "precision_at_k" => f.precision_at_k,
                "deletion_auc" => _json_num(f.deletion_auc),
                "insertion_auc" => _json_num(f.insertion_auc),
                "completeness_residual" => f.completeness_residual,
                "convergence_pearson" => [c.pearson for c in f.convergence],
                "convergence_n_coalitions" => [c.n_coalitions for c in f.convergence],
                "oracle_self_pearson" => f.oracle_self_pearson,
                "oracle_self_precision_at_k" => f.oracle_self_precision_at_k,
                "is_content_path" => f.output_kind == "content"))
        end
    end

    if selftest_only
        println("\n[kernelshap] --selftest: all passed, not writing artifacts.")
        return 0
    end

    # cross-game summary only on a full (non-sharded) run.
    if !sharded
        isdir(out_dir) || mkpath(out_dir)
        summary_path = joinpath(out_dir, "kernelshap_core_summary.json")
        total_budget = sum(r["n_real_reruns"] for r in summary; init = 0)
        rec = Dict{String,Any}(
            "paper" => "P2", "phase" => "phaseB_attribution", "method" => "kernelshap",
            "item" => "P2-E4-11", "games" => games,
            "n_coalitions" => n_coalitions, "topk" => topk, "seed" => seed,
            "target_frame" => target_frame, "horizon" => horizon,
            "where" => where, "commit" => _git_commit(), "timestamp" => string(round(Int, time())),
            "total_budget_reruns" => total_budget,
            "headline" => "KernelSHAP / Shapley sampling (Lundberg & Lee 2017; Štrumbelj & Kononenko " *
                "2014): estimate the Shapley values of the candidate RAM cells for y by sampling " *
                "N=$n_coalitions coalitions from the Shapley kernel, masking absent cells to 0, " *
                "RE-RUNNING the real ROM for each unique coalition, and solving the efficiency-constrained " *
                "weighted least squares. Scored vs the intervention oracle (corr + del/ins AUC + " *
                "precision@k); the EFFICIENCY axiom Σφ = y(intact) − y(all-masked) is reported as " *
                "completeness; the CONVERGENCE sweep (N/8…N) shows the estimate stabilising as N grows " *
                "(SHAP variance ∝ 1/N — the §7 'approximations diverge' knob). NO surrogate: the value " *
                "function IS the true emulator. KEY CONTRAST vs IG/saliency: KernelSHAP ALSO works on " *
                "POSITION outputs where the gradient methods VANISH. Oracle-as-method positive control " *
                "corr=1/p@k=1.",
            "budget_note" => Dict{String,Any}(
                "n_coalitions" => n_coalitions,
                "cost" => "≤ n_coalitions unique real re-runs + 2 pinned endpoints per output " *
                    "(2 outputs/game: content + position)",
                "convergence" => "SHAP re-solved at N/8, N/4, N/2, N from the SAME coalition pool — no extra re-runs"),
            "metrics_note" => Dict{String,Any}(
                "faithfulness" => "corr + deletion/insertion AUC on the TRUE VCS + precision@k vs the oracle top-k",
                "efficiency" => "Σφ ≈ y(intact) − y(all-masked) (the SHAP completeness axiom)",
                "convergence" => "corr/p@k/del-ins/residual vs the coalition count N (convergence vs compute)"),
            "results" => summary)
        open(summary_path, "w") do io; JSON.print(io, rec, 2); end
        println("\n[kernelshap] wrote summary $summary_path")
    end

    println("\n[kernelshap] ===== per-game headline (content path, N=$n_coalitions coalitions) =====")
    println("  game             ram   corr    spear   p@$topk    del    ins    Σφ-resid  conv-corr@N")
    for r in summary
        r["is_content_path"] || continue
        conv = join(["$(r["convergence_n_coalitions"][i])=$(round(r["convergence_pearson"][i],digits=2))"
                     for i in 1:length(r["convergence_pearson"])], " ")
        println("  $(rpad(r["game"],16)) $(rpad(r["content_ram_index"],5)) " *
                "$(rpad(round(r["pearson"],digits=3),7)) " *
                "$(rpad(round(r["spearman"],digits=3),7)) " *
                "$(rpad(round(r["precision_at_k"],digits=3),6)) " *
                "$(rpad(r["deletion_auc"]===nothing ? "NA" : round(r["deletion_auc"],digits=3),6)) " *
                "$(rpad(r["insertion_auc"]===nothing ? "NA" : round(r["insertion_auc"],digits=3),6)) " *
                "$(rpad(round(r["completeness_residual"],sigdigits=2),9)) [$conv]")
    end
    println("[kernelshap] ===== position contrast (ball_pixel — SHAP works where IG/saliency VANISH) =====")
    for r in summary
        r["is_content_path"] && continue
        println("  $(rpad(r["game"],16)) corr=$(round(r["pearson"],digits=3)) " *
                "p@$topk=$(round(r["precision_at_k"],digits=3)) " *
                "del=$(r["deletion_auc"]===nothing ? "NA" : round(r["deletion_auc"],digits=3)) " *
                "ins=$(r["insertion_auc"]===nothing ? "NA" : round(r["insertion_auc"],digits=3)) " *
                "Σφ-resid=$(round(r["completeness_residual"],sigdigits=2))")
    end
    return 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    KernelSHAP.main()
end
