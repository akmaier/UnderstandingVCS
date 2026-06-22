# lime.jl — Phase-B attribution (P2-E4-10), JULIA path.
#
# LIME (Ribeiro, Singh & Guestrin, KDD 2016) — Local Interpretable Model-agnostic
# Explanations — scored against the exact intervention oracle on the 6 CORE games
# (tools/xai_study/common/game_set.json). The 10th attribution method on the
# Phase-B leaderboard, built on the validated Phase-B foundation pinned by the IG
# pilot (P2-E4-0) + ig_baseline_sweep.jl, reusing the faithfulness CONTRACT
# verbatim (experiment_design.md §5 row "LIME (Ribeiro et al. 2016): local linear
# weights → corr vs true map; stability").
#
# METHOD — a BLACK-BOX, perturb-and-re-run LOCAL SURROGATE. LIME explains one
# instance x by (1) drawing N random perturbations in an INTERPRETABLE binary space
# (here: each candidate RAM cell PRESENT(1) or ABSENT(0)), (2) mapping each
# perturbation back to a real input — present cells keep their value, ABSENT cells
# are masked to the baseline (0) — and re-running the real model to read y, then (3)
# fitting a LOCALITY-WEIGHTED LINEAR surrogate g(z)=w·z+b by weighted least squares,
# the perturbations weighted by their proximity π(z) to the instance. The surrogate
# COEFFICIENTS w are the attribution (Ribeiro 2016 Eq. 1–3, the standard
# LinearLIME / "LIME with a sparse-linear g and the exponential cosine kernel"):
#
#   for n = 1..N:  draw z_n ∈ {0,1}^C   (cell j PRESENT w.p. keep_prob)
#                  ŷ_n = y( do(ram[j] := 0  for every j with z_n[j]=0) )   ← REAL re-run
#                  π_n = exp(-D(z_n, 1)^2 / σ^2)     (cosine-distance kernel to x=all-present)
#   w, b = argmin  Σ_n π_n · (ŷ_n − (w·z_n + b))^2          (WEIGHTED least squares)
#   importance(j) = w_j                                       (surrogate coefficient)
#
# i.e. cell j's attribution is its locally-weighted linear coefficient: how much y
# moves, near x, per unit of cell-j PRESENCE. Each ŷ_n is a GENUINE re-run of the
# real ROM through the bit-exact oracle machinery (no surrogate-of-a-surrogate, no
# world model, no gradient) — so LIME applies UNCHANGED to the §1 POSITION/INDEX
# output (ball_pixel) where the gradient methods VANISH (the §7 contrast it shares
# with Occlusion / RISE / the counterfactual). "Masking absent cells to 0" is the
# oracle's own occlude! operator (RAM→0), so the re-runs are EXACTLY the oracle's
# do()-interventions ⇒ apples-to-apples with del/ins.
#
# WHAT LIME ADDS over RISE/Occlusion (the §5 "stability" + the surrogate's own
# faithfulness): LIME does not read importance straight off the perturbations — it
# FITS a linear model, so two extra numbers fall out:
#   * SURROGATE R^2 — the weighted coefficient of determination of g on the sampled
#     neighbourhood. This is the LOCAL FAITHFULNESS OF THE SURROGATE ITSELF (how
#     well a linear g approximates the real ROM near x) — distinct from, and
#     reported ALONGSIDE, the attribution faithfulness vs the oracle. A low R^2 is
#     LIME's own confession that a linear surrogate is a poor local model of the VCS.
#   * STABILITY across SEEDS — LIME is sampling-dependent (the §7 prediction:
#     Partial/Fail, "baseline/sampling-dependent; approximations diverge"). We refit
#     the surrogate from K INDEPENDENT perturbation pools (different seeds) and
#     report the mean ± std of the attribution↔oracle corr AND the pairwise
#     coefficient-vector cosine similarity across fits — the SPEC's "stability"
#     metric, measured not asserted.
#
# BUDGET (the §R-recorded knob) — N = 300 perturbations per fit (the paper-reasonable
# default for a ~12–14-dim interpretable space), keep_prob = 0.5, K = 5 stability
# seeds. Cost = N real re-runs per fit; the K stability fits reuse the SAME budget
# knob with fresh seeds (so 5×N re-runs/output if stability is on). The kernel width
# σ defaults to the LIME library's 0.75·sqrt(#features) (recorded).
#
# OUTPUT SELECTION (the §1 content-vs-position split, mirroring the siblings):
#   * HEADLINE  = a CONTENT output: the candidate CONCEPT BYTE the oracle ranks as
#     the most causally-active RAM cause (SAME pick as the IG/EG/RISE/CF siblings,
#     via pick_content_idx), read straight off RAM.
#   * CONTRAST  = a POSITION/INDEX output `ball_pixel`: LIME STILL produces a real
#     attribution here (it re-runs the real ROM and reads the real pixel), unlike
#     IG/saliency which vanish — the "perturbation methods survive position outputs"
#     contrast (§7), reported not asserted.
#
# REUSES the validated Phase-B foundation on main (NO emulator core touched):
#   * ig_baseline_sweep.jl — the multi-game env layer (load_env, boot_replay,
#     continue_from, assert_bit_exact, occlude!, oracle_abs_delta,
#     deletion_insertion_auc with a generic read_y, position_pixel_cell,
#     pick_content_idx, candidates_path_for, env_step!, CORE_GAMES) — the SAME
#     machinery the IG/EG/RISE/CF siblings use, so the oracle column + del/ins curves
#     are apples-to-apples with them.
#   * pilot_ig_vs_oracle.jl — the SCORER (pearson/spearman/precision_at_k), the §R
#     writer helpers (_git_commit/_json_num/_trapz_unit), the harness
#     positive-control idea (oracle-as-method ⇒ corr=1/p@k=1).
#   * oracle_intervene.jl — Cause / build_pong_causes / candidate_ram_indices.
#   * jutari_oracle.jl — boot/replay/snapshot/intervene + the §R NPZ writer.
#
# Run (warm shared depot, primary's project) — DEFAULT = all 6 core games locally:
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseB_attribution/lime.jl --games core
# Flags: --games core|<g1,g2,...>  --game <g>
#        --target-frame N --horizon N --n-samples N --keep-prob P --topk K --seed S
#        --stability-seeds K --kernel-width SIGMA --selftest
# Cluster-shardable (Slurm array via tools/cluster/xai_array_jl.sbatch):
#        --shard <i> --nshards <n> --shard-kind game   (shard i → the i-th core game)
#        --out-dir <dir>   (where the §R records land; default = ./out)
#        (--roms-dir / --where accepted-and-ignored: ROMs located internally.)
#
# Writes (SPEC §R; file_scope lime_* under the out dir):
#   <out>/lime_<game>_content.{json,npz}
#   <out>/lime_<game>_ball_pixel.{json,npz}
#   <out>/lime_core_summary.json

module LIME

using JSON
import Statistics
import Random
import LinearAlgebra

# the Phase-B foundation (env layer + scorer + del/ins + oracle |Δy|), identical to
# the IG/EG/RISE/CF siblings — REUSED, not re-implemented.
include(joinpath(@__DIR__, "ig_baseline_sweep.jl"))
using .IGBaselineSweep: CORE_GAMES, load_env, boot_replay, continue_from,
                        assert_bit_exact, occlude!,
                        oracle_abs_delta, deletion_insertion_auc, position_pixel_cell,
                        pick_content_idx, candidates_path_for, env_step!
using .IGBaselineSweep.PilotIGvsOracle: pearson, spearman, precision_at_k,
                                        _git_commit, _json_num, _trapz_unit
using .IGBaselineSweep.PilotIGvsOracle.OracleIntervene: build_pong_causes, Cause,
                                                        candidate_ram_indices
using .IGBaselineSweep.PilotIGvsOracle.OracleIntervene.JutariOracle: Snapshot, snapshot,
                                                        intervene_ram!, write_npz, RAM_SIZE

const OUT_DIR = joinpath(@__DIR__, "out")

# ============================================================================
# The LIME machinery — all on the TRUE VCS (perturb-and-re-run; no gradient).
# ============================================================================
# We work over the UNIVERSE of candidate RAM cells (the oracle's candidate concept
# bytes — the SAME cells the oracle scores, so attribution lives on the same axis
# as the ground truth). The INTERPRETABLE space is binary present/absent over those
# cells. ABSENT ⇒ masked to 0 (the oracle's own occlude! operator) and re-run.

"""
    masked_read_y(checkpoint, actions, tf, hz, drop_idx, read_y) -> Float64

Re-run y under the LIME perturbation `do(ram[j] := 0 for j in drop_idx)`: deepcopy
the checkpoint, zero each ABSENT cell, continue the horizon on the real ROM, read y.
`drop_idx` is a collection of 0-based RAM indices. Empty ⇒ the intact y(x). Masking
absent cells to 0 is the oracle's own occlude! (RAM→0), so each probe is the
oracle's do()-intervention."""
function masked_read_y(checkpoint, actions, tf, hz, drop_idx, read_y)
    env = deepcopy(checkpoint)
    for i in drop_idx
        intervene_ram!(env, i, 0)
    end
    tail = Int.(actions[tf + 1 : tf + hz])
    for a in tail; env_step!(env, a); end
    return read_y(snapshot(env, length(tail)))
end

"""
    lime_sample(checkpoint, actions, tf, hz, cells, read_y; n_samples, keep_prob, rng)
        -> (Z::Matrix{Float64}, ys::Vector{Float64})

Draw `n_samples` random binary perturbations in the interpretable space over the
candidate `cells` (each cell PRESENT with probability `keep_prob`), and, for each,
RE-RUN the real ROM with the absent cells masked to 0, reading y. Returns the N×C
perturbation matrix Z (rows = perturbations, columns = cells; 1.0 = cell present)
and the N outputs. The FIRST row is forced to the all-present instance z=1 (the
anchor x), so the surrogate sees the unperturbed point. Every output is a genuine
emulator re-run — this is the entire compute budget."""
function lime_sample(checkpoint, actions, tf, hz, cells, read_y;
                     n_samples::Integer, keep_prob::Real, rng)
    C = length(cells)
    Z = zeros(Float64, n_samples, C)
    ys = zeros(Float64, n_samples)
    for n in 1:n_samples
        if n == 1
            present = trues(C)                 # anchor: the all-present instance x
        else
            present = rand(rng, C) .< keep_prob
        end
        Z[n, :] = Float64.(present)
        drop = [cells[j] for j in 1:C if !present[j]]
        ys[n] = masked_read_y(checkpoint, actions, tf, hz, drop, read_y)
    end
    return Z, ys
end

"""
    lime_weights(Z; kernel_width) -> Vector{Float64}

The LIME locality kernel π(z) = exp(-D(z, 1)^2 / σ^2), D = COSINE distance between
the perturbation z and the all-present instance 1 (Ribeiro 2016's default kernel for
LinearLIME). σ = `kernel_width`. The anchor row z=1 has D=0 ⇒ weight 1 (the highest
weight, as it should be)."""
function lime_weights(Z::AbstractMatrix{<:Real}; kernel_width::Real)
    N, C = size(Z)
    ref = ones(Float64, C)
    nref = LinearAlgebra.norm(ref)
    w = zeros(Float64, N)
    @inbounds for n in 1:N
        z = view(Z, n, :)
        nz = LinearAlgebra.norm(z)
        cos_sim = (nz == 0 || nref == 0) ? 0.0 : LinearAlgebra.dot(z, ref) / (nz * nref)
        d = 1.0 - cos_sim                       # cosine distance ∈ [0, 1]
        w[n] = exp(-(d^2) / (kernel_width^2))
    end
    return w
end

"""
    fit_weighted_linear(Z, ys, weights) -> (coef::Vector{Float64}, intercept, r2)

Weighted least squares fit of g(z) = coef·z + intercept to (Z, ys) with the LIME
locality weights, via LinearAlgebra (the normal equations with a Tikhonov ridge for
numerical stability — the LIME reference uses ridge regression too). Returns the
per-cell coefficients (the attribution), the intercept, and the WEIGHTED R^2 of g
on the sampled neighbourhood (the surrogate's own local faithfulness)."""
function fit_weighted_linear(Z::AbstractMatrix{<:Real}, ys::AbstractVector{<:Real},
                             weights::AbstractVector{<:Real}; ridge::Real = 1e-6)
    N, C = size(Z)
    X = hcat(Z, ones(Float64, N))               # design matrix: [cells | intercept]
    sw = sqrt.(weights)
    Xw = X .* sw                                  # √π-scaled rows
    yw = ys .* sw
    # ridge-regularised normal equations (ridge on the cell coefficients, not the
    # intercept) so the fit is well-posed even when a cell never varies in the pool.
    P = size(X, 2)
    R = Matrix(ridge * LinearAlgebra.I, P, P); R[P, P] = 0.0
    β = (Xw' * Xw .+ R) \ (Xw' * yw)
    coef = β[1:C]; intercept = β[C + 1]
    # weighted R^2 of g over the neighbourhood (the surrogate's local faithfulness)
    ŷ = X * β
    wsum = sum(weights)
    ybar = wsum == 0 ? Statistics.mean(ys) : sum(weights .* ys) / wsum
    ss_res = sum(weights .* (ys .- ŷ).^2)
    ss_tot = sum(weights .* (ys .- ybar).^2)
    r2 = ss_tot == 0 ? NaN : 1.0 - ss_res / ss_tot
    return coef, intercept, r2
end

"""Map the per-cell LIME coefficients onto the oracle's per-CAUSE axis: a RAM cause
touching candidate cell j gets |coef_j| (magnitude = importance); a TIA/joystick
cause gets 0 (the LIME interpretable space is RAM cells — the method honestly
assigns no mass off it). Magnitude because the oracle column is |Δy| (a non-negative
importance), so we score importance-vs-importance like the IG/RISE siblings."""
function lime_attr_per_cause(coef::AbstractVector{<:Real}, cells, causes::Vector{Cause})
    by_idx = Dict(cells[j] => coef[j] for j in 1:length(cells))
    return [c.kind == "ram" ? abs(get(by_idx, c.index, 0.0)) : 0.0 for c in causes]
end

# ============================================================================
# Per-output result + the stability sweep.
# ============================================================================
struct StabilityFit
    seed::Int
    surrogate_r2::Float64
    pearson::Float64
    spearman::Float64
    precision_at_k::Float64
end

struct LIMEResult
    game::String
    output::String
    output_kind::String                  # "content" | "position"
    target_frame::Int
    horizon::Int
    n_samples::Int                       # the BUDGET (perturbations per fit)
    keep_prob::Float64
    kernel_width::Float64
    stability_seeds::Int
    topk::Int
    seed::Int
    content_idx::Int
    cause_names::Vector{String}
    candidate_cells::Vector{Int}
    oracle_abs_delta::Vector{Float64}
    lime_attr_per_cause::Vector{Float64} # the headline LIME map (primary-seed fit)
    lime_coef_cells::Vector{Float64}
    surrogate_r2::Float64                # LOCAL faithfulness of the surrogate itself
    surrogate_intercept::Float64
    # faithfulness vs the oracle (the §5 contract; same as siblings) — primary fit
    pearson::Float64
    spearman::Float64
    precision_at_k::Float64
    deletion_auc::Float64
    insertion_auc::Float64
    del_curve::Vector{Float64}
    ins_curve::Vector{Float64}
    # the STABILITY sweep across K seeds (§5 "stability"): mean±std of corr/R^2 +
    # the pairwise coefficient-cosine across fits.
    stability::Vector{StabilityFit}
    stability_corr_mean::Float64
    stability_corr_std::Float64
    stability_r2_mean::Float64
    stability_r2_std::Float64
    stability_coef_cosine_mean::Float64  # pairwise coef-vector cosine across fits
    n_real_reruns::Int                   # total emulator re-runs (incl. stability)
    y_intact::Float64
    oracle_column_degenerate::Bool
    # harness positive control (oracle's OWN |Δy| as the method)
    oracle_self_pearson::Float64
    oracle_self_precision_at_k::Float64
    oracle_self_deletion_auc::Float64
    oracle_self_insertion_auc::Float64
end

"""Pairwise mean cosine similarity across a set of coefficient vectors (the
stability of the explanation direction across seeds; 1 = perfectly stable)."""
function _mean_pairwise_cosine(coefs::Vector{Vector{Float64}})
    K = length(coefs)
    K < 2 && return NaN
    acc = 0.0; cnt = 0
    for a in 1:K, b in (a + 1):K
        na = LinearAlgebra.norm(coefs[a]); nb = LinearAlgebra.norm(coefs[b])
        c = (na == 0 || nb == 0) ? 0.0 : LinearAlgebra.dot(coefs[a], coefs[b]) / (na * nb)
        acc += c; cnt += 1
    end
    return cnt == 0 ? NaN : acc / cnt
end

"""One LIME fit (sample N perturbations, fit the weighted linear surrogate) for a
given RNG seed. Returns the per-cause attribution, the per-cell coefficients, and
the surrogate R^2. `n_real_reruns` is N (every perturbation is one re-run)."""
function _one_fit(; checkpoint, actions, tf, hz, cells, read_y, n_samples, keep_prob,
                  kernel_width, causes, seed)
    rng = Random.MersenneTwister(seed * 1_000_003 + 6271)   # deterministic, recorded seed
    Z, ys = lime_sample(checkpoint, actions, tf, hz, cells, read_y;
                        n_samples = n_samples, keep_prob = keep_prob, rng = rng)
    weights = lime_weights(Z; kernel_width = kernel_width)
    coef, intercept, r2 = fit_weighted_linear(Z, ys, weights)
    attr = lime_attr_per_cause(coef, cells, causes)
    return attr, coef, intercept, r2
end

"""Score one LIME map (attribution per cause) against the oracle with the §5 metrics."""
function _score_map(attr, odelta, checkpoint, actions, tf, hz, causes, read_y, topk)
    pr  = pearson(attr, odelta)
    sp  = spearman(attr, odelta)
    pak = precision_at_k(attr, odelta, topk)
    order = sortperm(attr; rev = true)
    del_auc, ins_auc, dc, ic = deletion_insertion_auc(
        checkpoint, actions, tf, hz, causes, order, read_y)
    return pr, sp, pak, del_auc, ins_auc, dc, ic
end

"""Compute the LIME attribution + faithfulness + stability sweep for one output of
one game. `read_y` reads the chosen output off a Snapshot; `output_kind` is
"content"/"position"; `cells` is the candidate RAM-cell universe (0-based)."""
function compute_one(; game, output_kind, content_idx = -1, position_cell = nothing,
                     checkpoint, actions, target_frame, horizon, causes, candidate_cells,
                     n_samples, keep_prob, kernel_width, stability_seeds, topk, seed, verbose)
    read_y = output_kind == "content" ?
        (s -> Float64(Int(s.ram[content_idx + 1]))) :
        (s -> Float64(Int(s.screen[position_cell[1], position_cell[2]])))
    output_name = output_kind == "content" ?
        "content(ram_self@$content_idx)" :
        "ball_pixel@r$(position_cell[1])c$(position_cell[2])"

    # 1) the oracle |Δy| per CAUSE (real re-runs on the TRUE VCS) — the ground truth
    odelta = oracle_abs_delta(checkpoint, actions, target_frame, horizon, causes, read_y)
    degenerate = Statistics.std(odelta) == 0

    # 2) intact y(x)
    base_snap = continue_from(checkpoint, Int.(actions[target_frame + 1 : target_frame + horizon]))
    y_intact = read_y(base_snap)

    # 3) the PRIMARY LIME fit (seed = the run seed): sample N perturbations, re-run
    #    the real ROM for each (THE BUDGET), fit the locality-weighted linear
    #    surrogate, read off the coefficients = the headline attribution.
    verbose && println("[lime] $game '$output_name': sampling $n_samples perturbations " *
                       "(keep_prob=$keep_prob, kernel_width=$(round(kernel_width,digits=3))) over " *
                       "$(length(candidate_cells)) candidate cells [$n_samples real re-runs]; " *
                       "fitting weighted-linear surrogate...")
    attr, coef, intercept, r2 = _one_fit(; checkpoint = checkpoint, actions = actions,
        tf = target_frame, hz = horizon, cells = candidate_cells, read_y = read_y,
        n_samples = n_samples, keep_prob = keep_prob, kernel_width = kernel_width,
        causes = causes, seed = seed)
    pr, sp, pak, del_auc, ins_auc, dc, ic =
        _score_map(attr, odelta, checkpoint, actions, target_frame, horizon, causes, read_y, topk)

    # 4) the STABILITY sweep across K seeds (§5 "stability"): refit from K
    #    independent perturbation pools and report the spread of corr / R^2 / the
    #    pairwise coefficient cosine. The primary fit is fit #1; the extra K−1 fits
    #    use fresh seeds (so K×N re-runs total when stability is on).
    stab = StabilityFit[]
    coefs = Vector{Float64}[coef]
    push!(stab, StabilityFit(seed, r2, pr, sp, pak))
    n_real = n_samples
    for s in 1:(stability_seeds - 1)
        sd = seed + 101 * s
        a2, c2, _, r22 = _one_fit(; checkpoint = checkpoint, actions = actions,
            tf = target_frame, hz = horizon, cells = candidate_cells, read_y = read_y,
            n_samples = n_samples, keep_prob = keep_prob, kernel_width = kernel_width,
            causes = causes, seed = sd)
        pr2 = pearson(a2, odelta); sp2 = spearman(a2, odelta)
        pak2 = precision_at_k(a2, odelta, topk)
        push!(stab, StabilityFit(sd, r22, pr2, sp2, pak2))
        push!(coefs, c2)
        n_real += n_samples
    end
    corrs = [f.pearson for f in stab]; r2s = [f.surrogate_r2 for f in stab]
    valid_r2 = filter(isfinite, r2s)
    corr_mean = Statistics.mean(corrs)
    corr_std  = length(corrs) > 1 ? Statistics.std(corrs) : 0.0
    r2_mean   = isempty(valid_r2) ? NaN : Statistics.mean(valid_r2)
    r2_std    = length(valid_r2) > 1 ? Statistics.std(valid_r2) : 0.0
    coef_cos  = _mean_pairwise_cosine(coefs)

    # 5) harness positive control (oracle's OWN |Δy| as the candidate map)
    or_pr  = pearson(odelta, odelta); or_pak = precision_at_k(odelta, odelta, topk)
    or_order = sortperm(odelta; rev = true)
    or_del, or_ins, _, _ = deletion_insertion_auc(checkpoint, actions, target_frame,
                                                  horizon, causes, or_order, read_y)

    if verbose
        println("[lime]   surrogate R^2=$(round(r2,digits=4)) (local faithfulness of g itself)")
        println("[lime]   faithfulness vs oracle: corr=$(round(pr,digits=4)) " *
                "spearman=$(round(sp,digits=4)) p@$topk=$(round(pak,digits=3)) " *
                "del=$(round(del_auc,digits=3)) ins=$(round(ins_auc,digits=3))")
        println("[lime]   stability over $(stability_seeds) seeds: corr=$(round(corr_mean,digits=3))±" *
                "$(round(corr_std,digits=3)) R^2=$(round(r2_mean,digits=3))±$(round(r2_std,digits=3)) " *
                "coef-cosine=$(round(coef_cos,digits=3))")
        println("[lime]   [harness] oracle-as-method: corr=$(round(or_pr,digits=3)) " *
                "p@$topk=$(round(or_pak,digits=3)) del=$(round(or_del,digits=3)) ins=$(round(or_ins,digits=3))" *
                (degenerate ? "  (oracle column flat at this state)" : ""))
    end

    return LIMEResult(game, output_name, output_kind, target_frame, horizon, n_samples,
                      Float64(keep_prob), Float64(kernel_width), stability_seeds, topk, seed,
                      content_idx, [c.name for c in causes], collect(candidate_cells), odelta,
                      attr, coef, r2, intercept,
                      pr, sp, pak, del_auc, ins_auc, dc, ic,
                      stab, corr_mean, corr_std, r2_mean, r2_std, coef_cos,
                      n_real, y_intact, degenerate, or_pr, or_pak, or_del, or_ins)
end

"""Drive both outputs for one game: assert bit-exact, build causes + candidate
cells, pick the content byte, locate the position pixel, run LIME for content +
position."""
function compute_game(; game, target_frame, horizon, n_samples, keep_prob, kernel_width,
                      stability_seeds, topk, seed, verbose)
    total = target_frame + horizon
    actions = fill(0, total)
    verbose && println("[lime] $game: asserting bit-exactness (2 fresh boots+replays to f$total)...")
    assert_bit_exact(actions, total; game = game)

    cand = candidates_path_for(game)
    checkpoint = boot_replay(actions, target_frame; game = game)
    at_target = continue_from(checkpoint, Int[])
    causes = build_pong_causes(cand, at_target)
    candidate_cells = sort(unique(idx for (idx, _) in candidate_ram_indices(cand)))

    # content byte = the most causally-active candidate concept byte (SAME pick as
    # the IG/EG/RISE/CF siblings ⇒ the methods score the SAME content output).
    content_idx, content_mv = pick_content_idx(checkpoint, actions, target_frame, horizon, causes,
                                               [idx for (idx, _) in candidate_ram_indices(cand)])
    verbose && println("[lime] $game content byte = RAM[$content_idx] (max oracle |Δself|=$(round(content_mv,digits=2)))")

    base = continue_from(checkpoint, Int.(actions[target_frame + 1 : total]))
    pcell = position_pixel_cell(checkpoint, base.screen; horizon = horizon)

    f_content = compute_one(; game = game, output_kind = "content", content_idx = content_idx,
        checkpoint = checkpoint, actions = actions, target_frame = target_frame, horizon = horizon,
        causes = causes, candidate_cells = candidate_cells, n_samples = n_samples, keep_prob = keep_prob,
        kernel_width = kernel_width, stability_seeds = stability_seeds, topk = topk, seed = seed,
        verbose = verbose)
    f_pos = compute_one(; game = game, output_kind = "position", position_cell = pcell,
        checkpoint = checkpoint, actions = actions, target_frame = target_frame, horizon = horizon,
        causes = causes, candidate_cells = candidate_cells, n_samples = n_samples, keep_prob = keep_prob,
        kernel_width = kernel_width, stability_seeds = stability_seeds, topk = topk, seed = seed,
        verbose = verbose)
    return f_content, f_pos
end

# ============================================================================
# Self-check (DoD) — the scoring contract is sound; results are non-fabricated.
# ============================================================================
"""
    selftest(f::LIMEResult; require_nondegenerate) -> Bool

Asserts the load-bearing claims of LIME + the shared contract:

  (HARNESS POSITIVE CONTROL) feeding the oracle's OWN |Δy| as the candidate map
    scores corr=1 (on a non-degenerate column) and precision@k=1 — the harness
    rewards a faithful map (else the harness is broken, not the method).

  (LIME CORRECTNESS)
    * attribution finite + non-negative (it is |coef|); all AUCs ∈ [0,1] or NaN.
    * the surrogate R^2 is finite or NaN, and ≤ 1 (a coefficient of determination).
    * the budget is real: n_real_reruns == n_samples × stability_seeds.
    * the stability sweep has `stability_seeds` fits; corr-std / R^2-std / coef-cosine
      recorded (the §7 "sampling-dependent" prediction — measured, not asserted).

  (POSITION OUTPUT — the §7 contrast) LIME still produces a real attribution on the
    position pixel (it re-runs the real ROM, no gradient) — recorded, not asserted
    to be large; the contrast with IG/saliency which vanish there.

Throws on a violation."""
function selftest(f::LIMEResult; require_nondegenerate = false)
    @assert all(isfinite, f.lime_attr_per_cause) "non-finite LIME attribution ($(f.game)/$(f.output))"
    @assert all(>=(0.0), f.lime_attr_per_cause) "LIME attribution must be non-negative ($(f.game)/$(f.output))"
    for (nm, v) in (("deletion", f.deletion_auc), ("insertion", f.insertion_auc),
                    ("oracle_self_deletion", f.oracle_self_deletion_auc),
                    ("oracle_self_insertion", f.oracle_self_insertion_auc))
        @assert isnan(v) || (0.0 <= v <= 1.0 + 1e-9) "$nm AUC out of [0,1]: $v ($(f.game)/$(f.output))"
    end
    @assert isnan(f.surrogate_r2) || f.surrogate_r2 <= 1.0 + 1e-9 "surrogate R^2 > 1: $(f.surrogate_r2) ($(f.game)/$(f.output))"
    @assert f.n_real_reruns == f.n_samples * f.stability_seeds "budget mismatch: " *
        "n_real_reruns=$(f.n_real_reruns) != n_samples($(f.n_samples))×stability_seeds($(f.stability_seeds))"
    @assert length(f.stability) == f.stability_seeds "stability sweep has $(length(f.stability)) fits, expected $(f.stability_seeds)"

    if f.output_kind == "position"
        println("[lime] SELF-CHECK PASS (position '$(f.output)', $(f.game)): " *
                "LIME map finite (max|attr|=$(round(maximum(f.lime_attr_per_cause),sigdigits=3))) over " *
                "$(f.n_samples) perturbations; surrogate R^2=$(round(f.surrogate_r2,digits=3)); " *
                "corr=$(round(f.pearson,digits=3)); oracle-as-method corr=$(round(f.oracle_self_pearson,digits=3)) " *
                "(the §7 contrast: LIME re-runs the real ROM, so it works on a position output where " *
                "IG/saliency VANISH).")
        return true
    end

    # CONTENT output --------------------------------------------------------
    if require_nondegenerate
        @assert !f.oracle_column_degenerate "content oracle column degenerate for $(f.game) — " *
            "pick_content_idx failed to find a causally-active concept byte"
        @assert f.oracle_self_pearson > 0.999 "harness broken: oracle-as-method corr != 1 ($(f.oracle_self_pearson)) [$(f.game)]"
        @assert f.oracle_self_precision_at_k == 1.0 "harness broken: oracle-as-method p@k != 1 [$(f.game)]"
    end
    println("[lime] SELF-CHECK PASS (content '$(f.output)', $(f.game)): " *
            "LIME map finite + non-negative over $(f.n_samples) perturbations (keep_prob=$(f.keep_prob)); " *
            "surrogate R^2=$(round(f.surrogate_r2,digits=3)) (local faithfulness of g); " *
            "corr=$(round(f.pearson,digits=3)) p@$(f.topk)=$(round(f.precision_at_k,digits=3)) " *
            "del=$(round(f.deletion_auc,digits=3)) ins=$(round(f.insertion_auc,digits=3)); " *
            "stability corr=$(round(f.stability_corr_mean,digits=3))±$(round(f.stability_corr_std,digits=3)) " *
            "coef-cosine=$(round(f.stability_coef_cosine_mean,digits=3)); " *
            "oracle-as-method corr=$(round(f.oracle_self_pearson,digits=3)).")
    return true
end

# ============================================================================
# Persist (SPEC §R) — JSON record + sibling .npz; file_scope lime_*.
# ============================================================================
function _output_note(f::LIMEResult)
    if f.output_kind == "position"
        return "POSITION/INDEX output (ball_pixel) — the §7 CONTRAST: unlike IG/saliency (which " *
               "VANISH on position outputs because the pixel is a round/argmax sprite column with no " *
               "differentiable RAM path), LIME — a BLACK-BOX perturb-and-re-run surrogate — STILL " *
               "produces a real attribution here. It masks candidate RAM cells to 0, RE-RUNS the real " *
               "ROM, reads the real pixel, and fits a locality-weighted linear surrogate; no " *
               "differentiation needed. Reported, not asserted to be large; the intervention oracle " *
               "remains the sole truth."
    else
        return "CONTENT output: RAM byte $(f.content_idx) (the most causally-active candidate concept " *
               "byte — SAME pick as the IG/EG/RISE/CF siblings). LIME (Ribeiro 2016) draws " *
               "$(f.n_samples) RANDOM BINARY PERTURBATIONS (keep_prob=$(f.keep_prob)) in the interpretable " *
               "present/absent space over the candidate cells, masks absent cells to 0, RE-RUNS the real " *
               "ROM, weights each perturbation by the cosine-distance kernel π(z)=exp(-D(z,1)^2/σ^2) " *
               "(σ=$(round(f.kernel_width,digits=3))), and fits a weighted linear surrogate g(z)=w·z+b; the " *
               "coefficients w are the attribution. SURROGATE R^2=$(round(f.surrogate_r2,digits=3)) (how well " *
               "a linear g models the real ROM near x — LIME's own local-faithfulness confession). " *
               "Faithfulness vs the oracle corr=$(round(f.pearson,digits=3)). STABILITY over " *
               "$(f.stability_seeds) seeds: corr=$(round(f.stability_corr_mean,digits=3))±" *
               "$(round(f.stability_corr_std,digits=3)), coef-cosine=$(round(f.stability_coef_cosine_mean,digits=3)) " *
               "(the §7 'sampling-dependent' prediction — measured). Every probe is a genuine emulator " *
               "re-run (RAM→0 = the oracle's own occlude!), so LIME is apples-to-apples with the del/ins curves."
    end
end

function write_result(f::LIMEResult; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    tag = f.output_kind == "content" ? "content" : "ball_pixel"
    stem = "lime_$(f.game)_$(tag)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path  = joinpath(out_dir, stem * ".npz")

    stab_json = [Dict{String,Any}(
        "seed" => s.seed, "surrogate_r2" => _json_num(s.surrogate_r2),
        "pearson_corr" => s.pearson, "spearman_corr" => s.spearman,
        "precision_at_k" => s.precision_at_k,
    ) for s in f.stability]

    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseB_attribution", "method" => "lime",
        "game" => f.game, "state" => "f$(f.target_frame)+$(f.horizon)",
        "target_output" => f.output,
        # headline §R scalar = the LIME map's corr with the oracle (one comparable
        # number per method on the leaderboard, like the siblings).
        "metric_name" => "pearson_corr_with_oracle",
        "value" => f.pearson,
        "stderr" => _json_num(f.stability_corr_std), "ci" => nothing, "n" => length(f.cause_names),
        "seed" => f.seed, "where" => "local", "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene@$(f.game)#$(f.output)",
        "timestamp" => string(round(Int, time())), "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            "substrate" => "jutari (Julia, HARD) — LIME (Ribeiro et al. 2016) local linear surrogate: draw " *
                "N random binary perturbations in the interpretable present/absent space over the candidate " *
                "RAM cells, mask absent cells to 0, RE-RUN the real ROM (bit-exact oracle machinery), weight " *
                "each perturbation by the cosine-distance kernel, fit a weighted linear surrogate by least " *
                "squares (LinearAlgebra); coefficients = attribution. Every probe is a genuine emulator " *
                "re-run; no gradient, no surrogate-of-a-surrogate.",
            "output_kind" => f.output_kind,
            "content_ram_index" => f.content_idx,
            # THE BUDGET (recorded per §R) — the perturb-and-re-run cost knob.
            "budget" => Dict{String,Any}(
                "n_samples" => f.n_samples,
                "keep_prob" => f.keep_prob,
                "kernel_width" => f.kernel_width,
                "stability_seeds" => f.stability_seeds,
                "n_candidate_cells" => length(f.candidate_cells),
                "n_real_reruns" => f.n_real_reruns,
                "rng" => "MersenneTwister(seed*1_000_003 + 6271), per-fit seed offset 101·k",
                "note" => "N=$(f.n_samples) perturbations per fit at keep_prob=$(f.keep_prob) " *
                    "(paper-reasonable for a ~$(length(f.candidate_cells))-dim interpretable space). Each " *
                    "perturbation = ONE real emulator re-run; the STABILITY sweep refits over " *
                    "$(f.stability_seeds) seeds ⇒ $(f.n_real_reruns) real re-runs total. Kernel = cosine-" *
                    "distance exp(-D^2/σ^2), σ=$(round(f.kernel_width,digits=3)) (LIME default)."),
            "surrogate" => Dict{String,Any}(
                "model" => "weighted linear g(z)=w·z+b, ridge-regularised normal equations (LinearAlgebra)",
                "kernel" => "cosine-distance exponential π(z)=exp(-D(z,1)^2/σ^2)",
                "r2" => _json_num(f.surrogate_r2),
                "intercept" => f.surrogate_intercept,
                "interpretation" => "the WEIGHTED R^2 of the linear surrogate on the sampled neighbourhood " *
                    "= the LOCAL FAITHFULNESS OF THE SURROGATE ITSELF (how well a linear g approximates the " *
                    "real ROM near x), distinct from the attribution↔oracle corr below. A low R^2 is LIME's " *
                    "own confession that a linear model is a poor local fit to the VCS computation."),
            "headline_metrics" => Dict{String,Any}(
                "pearson_corr" => f.pearson, "spearman_corr" => f.spearman,
                "precision_at_k" => f.precision_at_k, "topk" => f.topk,
                "deletion_auc" => _json_num(f.deletion_auc),
                "insertion_auc" => _json_num(f.insertion_auc),
                "surrogate_r2" => _json_num(f.surrogate_r2)),
            # the §5 "stability" metric — how the explanation varies across seeds.
            "stability" => Dict{String,Any}(
                "seeds" => f.stability_seeds,
                "fits" => stab_json,
                "corr_mean" => f.stability_corr_mean, "corr_std" => f.stability_corr_std,
                "surrogate_r2_mean" => _json_num(f.stability_r2_mean),
                "surrogate_r2_std" => f.stability_r2_std,
                "coef_cosine_mean" => _json_num(f.stability_coef_cosine_mean),
                "interpretation" => "LIME refit from $(f.stability_seeds) INDEPENDENT perturbation pools " *
                    "(different seeds). corr_std + (1−coef_cosine_mean) quantify the SAMPLING DEPENDENCE — " *
                    "the §7 prediction (Partial/Fail: 'baseline/sampling-dependent; approximations " *
                    "diverge'). High coef_cosine (→1) ⇒ a stable explanation direction; large corr_std ⇒ " *
                    "the faithfulness verdict itself wobbles with the random seed."),
            "harness_positive_control" => Dict{String,Any}(
                "method" => "oracle_abs_delta (the perfectly-faithful attribution)",
                "pearson_corr" => f.oracle_self_pearson,
                "precision_at_k" => f.oracle_self_precision_at_k,
                "deletion_auc" => _json_num(f.oracle_self_deletion_auc),
                "insertion_auc" => _json_num(f.oracle_self_insertion_auc),
                "oracle_column_degenerate" => f.oracle_column_degenerate,
                "interpretation" => "corr=1 & precision@k=1 (on a non-degenerate column) ⇒ the scoring " *
                    "harness rewards a faithful map; the LIME numbers above are then a true measurement, " *
                    "not an artefact of the metric."),
            "auc_note" => "deletion/insertion curves measured on the TRUE VCS by re-running the real ROM " *
                "with top-LIME causes occluded (deletion) / restored (insertion); every point is a genuine " *
                "emulator re-run, not a surrogate. NaN = a genuinely flat experiment.",
            "output_note" => _output_note(f),
            "cause_names" => f.cause_names,
            "lime_attr_per_cause" => Dict(f.cause_names[i] => f.lime_attr_per_cause[i] for i in 1:length(f.cause_names)),
            "oracle_abs_delta_per_cause" => Dict(f.cause_names[i] => f.oracle_abs_delta[i] for i in 1:length(f.cause_names)),
            "y_intact" => f.y_intact,
            "scales_to_cluster_via" =>
                "tools/cluster/xai_array_jl.sbatch (--shard i --nshards n --shard-kind game): one Slurm " *
                "array task per core game; or batched SOFT-STE masked re-runs over perturbations×cells×games " *
                "on GPU (the forward is bit-exact to this HARD map), the linear fit done on the host.",
        ),
    )
    open(json_path, "w") do io; JSON.print(io, rec, 2); end

    write_npz(npz_path, Dict(
        "oracle_abs_delta"       => f.oracle_abs_delta,
        "lime_attr_per_cause"    => f.lime_attr_per_cause,
        "lime_coef_cells"        => f.lime_coef_cells,
        "candidate_cells"        => Float64.(f.candidate_cells),
        "deletion_curve"         => f.del_curve,
        "insertion_curve"        => f.ins_curve,
        "stability_seeds"        => Float64[s.seed for s in f.stability],
        "stability_pearson"      => Float64[s.pearson for s in f.stability],
        "stability_spearman"     => Float64[s.spearman for s in f.stability],
        "stability_precision_at_k" => Float64[s.precision_at_k for s in f.stability],
        "stability_surrogate_r2" => Float64[isnan(s.surrogate_r2) ? -2.0 : s.surrogate_r2 for s in f.stability],
        "scalars"                => Float64[f.pearson, f.spearman, f.precision_at_k,
                                            isnan(f.deletion_auc) ? -1.0 : f.deletion_auc,
                                            isnan(f.insertion_auc) ? -1.0 : f.insertion_auc,
                                            isnan(f.surrogate_r2) ? -2.0 : f.surrogate_r2,
                                            Float64(f.n_samples), f.keep_prob, f.kernel_width,
                                            f.stability_corr_mean, f.stability_corr_std,
                                            isnan(f.stability_coef_cosine_mean) ? -2.0 : f.stability_coef_cosine_mean],
    ))
    return json_path, npz_path
end

# ============================================================================
# CLI — local default = all 6 core games; cluster-shardable via the sbatch flags.
# ============================================================================
"""Resolve the games to run from the parsed CLI: an explicit --game / --games, OR
a --shard i --nshards n --shard-kind game selection (shard i → the i-th core
game). Default (no selection) = all 6 core games."""
function _resolve_games(; single_game, games_arg, shard, nshards, shard_kind)
    single_game !== nothing && return [single_game]
    games_arg !== nothing && return games_arg
    if shard !== nothing
        kind = shard_kind === nothing ? "game" : shard_kind
        kind == "game" || error("lime.jl only shards by game (--shard-kind game), got $kind")
        n = nshards === nothing ? length(CORE_GAMES) : nshards
        # shard i (0-based) → every game at position i, i+n, i+2n, ... (round-robin),
        # so any nshards ≤ |core| picks the i-th core game and nshards=|core| is 1-per.
        sel = [CORE_GAMES[j] for j in (shard + 1):n:length(CORE_GAMES)]
        isempty(sel) && error("shard $shard of $n selects no core game (|core|=$(length(CORE_GAMES)))")
        return sel
    end
    return CORE_GAMES
end

# default kernel width = LIME library's 0.75·sqrt(#features); resolved per-game once
# the candidate-cell count is known. A negative sentinel ⇒ "auto".
_resolve_kernel_width(kw, n_cells) = kw < 0 ? 0.75 * sqrt(n_cells) : kw

function main(args = ARGS)
    single_game = nothing; games_arg = nothing
    target_frame = 120; horizon = 30
    n_samples = 300; keep_prob = 0.5
    kernel_width = -1.0          # sentinel: auto = 0.75·sqrt(#features)
    stability_seeds = 5
    topk = 3; seed = 0
    shard = nothing; nshards = nothing; shard_kind = nothing
    out_dir = OUT_DIR
    selftest_only = false
    i = 1
    while i <= length(args)
        a = args[i]
        if     a == "--games"
            v = args[i+1]; games_arg = (v == "core") ? CORE_GAMES : String.(split(v, ",")); i += 2
        elseif a == "--game";            single_game = args[i+1]; i += 2
        elseif a == "--target-frame";    target_frame = parse(Int, args[i+1]); i += 2
        elseif a == "--horizon";         horizon = parse(Int, args[i+1]); i += 2
        elseif a == "--n-samples";       n_samples = parse(Int, args[i+1]); i += 2
        elseif a == "--keep-prob";       keep_prob = parse(Float64, args[i+1]); i += 2
        elseif a == "--kernel-width";    kernel_width = parse(Float64, args[i+1]); i += 2
        elseif a == "--stability-seeds"; stability_seeds = parse(Int, args[i+1]); i += 2
        elseif a == "--topk";            topk = parse(Int, args[i+1]); i += 2
        elseif a == "--seed";            seed = parse(Int, args[i+1]); i += 2
        elseif a == "--shard";           shard = parse(Int, args[i+1]); i += 2
        elseif a == "--nshards";         nshards = parse(Int, args[i+1]); i += 2
        elseif a == "--shard-kind";      shard_kind = args[i+1]; i += 2
        elseif a == "--out-dir";         out_dir = args[i+1]; i += 2
        elseif a == "--roms-dir";        i += 2            # accepted-and-ignored (ROMs located internally)
        elseif a == "--where";           i += 2            # accepted-and-ignored (record always says where=local/cluster via env)
        elseif a == "--selftest";        selftest_only = true; i += 1
        else; i += 1
        end
    end
    games = _resolve_games(; single_game = single_game, games_arg = games_arg,
                           shard = shard, nshards = nshards, shard_kind = shard_kind)

    println("[lime] LIME (Ribeiro 2016) local-linear-surrogate attribution vs oracle — " *
            "games=$(join(games, ",")) target_frame=$target_frame horizon=$horizon " *
            "n_samples=$n_samples keep_prob=$keep_prob kernel_width=$(kernel_width < 0 ? "auto" : kernel_width) " *
            "stability_seeds=$stability_seeds topk=$topk seed=$seed out_dir=$out_dir (jutari/Julia)")

    summary = Dict{String,Any}[]
    for game in games
        println("\n[lime] ===== $game =====")
        # resolve the (auto) kernel width from this game's candidate-cell count.
        cand = candidates_path_for(game)
        n_cells = length(unique(idx for (idx, _) in candidate_ram_indices(cand)))
        kw = _resolve_kernel_width(kernel_width, n_cells)
        f_content, f_pos = compute_game(; game = game, target_frame = target_frame,
            horizon = horizon, n_samples = n_samples, keep_prob = keep_prob, kernel_width = kw,
            stability_seeds = stability_seeds, topk = topk, seed = seed, verbose = true)
        @assert !f_content.oracle_column_degenerate "content oracle column is degenerate for $game " *
            "at f$target_frame — pick_content_idx failed to find a causally-active concept byte"
        selftest(f_content; require_nondegenerate = true)
        selftest(f_pos)
        if !selftest_only
            for f in (f_content, f_pos)
                jp, np = write_result(f; out_dir = out_dir)
                println("[lime] wrote $jp"); println("[lime] arrays  $np")
            end
        end
        for f in (f_content, f_pos)
            push!(summary, Dict{String,Any}(
                "game" => game, "output" => f.output, "output_kind" => f.output_kind,
                "content_ram_index" => f.content_idx,
                "n_samples" => f.n_samples, "keep_prob" => f.keep_prob,
                "kernel_width" => f.kernel_width, "stability_seeds" => f.stability_seeds,
                "pearson" => f.pearson, "spearman" => f.spearman,
                "precision_at_k" => f.precision_at_k,
                "surrogate_r2" => _json_num(f.surrogate_r2),
                "deletion_auc" => _json_num(f.deletion_auc),
                "insertion_auc" => _json_num(f.insertion_auc),
                "stability_corr_mean" => f.stability_corr_mean,
                "stability_corr_std" => f.stability_corr_std,
                "stability_coef_cosine_mean" => _json_num(f.stability_coef_cosine_mean),
                "oracle_self_pearson" => f.oracle_self_pearson,
                "oracle_self_precision_at_k" => f.oracle_self_precision_at_k,
                "is_content_path" => f.output_kind == "content"))
        end
    end

    if selftest_only
        println("\n[lime] --selftest: all passed, not writing artifacts.")
        return 0
    end

    isdir(out_dir) || mkpath(out_dir)
    summary_path = joinpath(out_dir, "lime_core_summary.json")
    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseB_attribution", "method" => "lime",
        "item" => "P2-E4-10", "games" => games,
        "n_samples" => n_samples, "keep_prob" => keep_prob,
        "kernel_width" => (kernel_width < 0 ? "auto(0.75*sqrt(#features))" : kernel_width),
        "stability_seeds" => stability_seeds, "topk" => topk, "seed" => seed,
        "target_frame" => target_frame, "horizon" => horizon,
        "where" => "local", "commit" => _git_commit(), "timestamp" => string(round(Int, time())),
        "headline" => "LIME (Ribeiro, Singh & Guestrin 2016): local interpretable model-agnostic " *
            "explanations — draw N=$n_samples random binary perturbations in the interpretable " *
            "present/absent space over the candidate RAM cells (keep_prob=$keep_prob), mask absent cells " *
            "to 0, RE-RUN the real ROM, weight each perturbation by the cosine-distance kernel, and fit a " *
            "WEIGHTED LINEAR SURROGATE g(z)=w·z+b by least squares; coefficients w = attribution. A " *
            "black-box perturb-and-re-run method (no gradient), scored vs the intervention oracle (corr + " *
            "del/ins AUC + precision@k). Reports BOTH the surrogate R^2 (local faithfulness of g itself) " *
            "AND the attribution faithfulness vs the true causal map, PLUS the §5 STABILITY across " *
            "$stability_seeds seeds (corr-std + coef-cosine — the §7 'sampling-dependent' prediction). " *
            "BUDGET = N perturbations × $stability_seeds seeds real re-runs per output. KEY CONTRAST vs " *
            "IG/saliency: LIME ALSO works on POSITION outputs (it re-runs the real ROM) where the " *
            "gradient methods VANISH. Oracle-as-method positive control corr=1/p@k=1.",
        "budget_note" => Dict{String,Any}(
            "n_samples" => n_samples, "keep_prob" => keep_prob, "stability_seeds" => stability_seeds,
            "cost" => "n_samples × stability_seeds real emulator re-runs per output (2 outputs/game: " *
                "content + position)",
            "kernel" => "cosine-distance exponential, σ=auto(0.75·sqrt(#features)) unless --kernel-width"),
        "metrics_note" => Dict{String,Any}(
            "surrogate_r2" => "weighted R^2 of the linear surrogate g on the local neighbourhood (g's own " *
                "local faithfulness — distinct from the oracle corr)",
            "faithfulness" => "corr + deletion/insertion AUC on the TRUE VCS + precision@k vs the oracle top-k",
            "stability" => "corr-std + surrogate-R^2-std + pairwise coef-cosine across $stability_seeds seeds"),
        "results" => summary)
    open(summary_path, "w") do io; JSON.print(io, rec, 2); end
    println("\n[lime] wrote summary $summary_path")

    println("\n[lime] ===== per-game headline (content path, N=$n_samples perturbations) =====")
    println("  game             ram   corr    spear   p@$topk    R^2     del    ins    stab(corr±sd / coef-cos)")
    for r in summary
        r["is_content_path"] || continue
        cc = r["stability_coef_cosine_mean"]
        println("  $(rpad(r["game"],16)) $(rpad(r["content_ram_index"],5)) " *
                "$(rpad(round(r["pearson"],digits=3),7)) " *
                "$(rpad(round(r["spearman"],digits=3),7)) " *
                "$(rpad(round(r["precision_at_k"],digits=3),6)) " *
                "$(rpad(r["surrogate_r2"]===nothing ? "NA" : round(r["surrogate_r2"],digits=3),7)) " *
                "$(rpad(r["deletion_auc"]===nothing ? "NA" : round(r["deletion_auc"],digits=3),6)) " *
                "$(rpad(r["insertion_auc"]===nothing ? "NA" : round(r["insertion_auc"],digits=3),6)) " *
                "$(round(r["stability_corr_mean"],digits=2))±$(round(r["stability_corr_std"],digits=2)) / " *
                "$(cc===nothing ? "NA" : round(cc,digits=2))")
    end
    println("[lime] ===== position contrast (ball_pixel — LIME works where IG/saliency VANISH) =====")
    for r in summary
        r["is_content_path"] && continue
        println("  $(rpad(r["game"],16)) corr=$(round(r["pearson"],digits=3)) " *
                "p@$topk=$(round(r["precision_at_k"],digits=3)) " *
                "R^2=$(r["surrogate_r2"]===nothing ? "NA" : round(r["surrogate_r2"],digits=3)) " *
                "del=$(r["deletion_auc"]===nothing ? "NA" : round(r["deletion_auc"],digits=3)) " *
                "ins=$(r["insertion_auc"]===nothing ? "NA" : round(r["insertion_auc"],digits=3))")
    end
    return 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    LIME.main()
end
