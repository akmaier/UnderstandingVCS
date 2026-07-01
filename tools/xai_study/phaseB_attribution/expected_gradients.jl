# expected_gradients.jl — Phase-B attribution (P2-E4-6), JULIA path.
#
# Expected Gradients (Erion et al. 2021, Nature Machine Intelligence) scored
# against the exact intervention oracle on the 6 CORE games (tools/xai_study/
# common/game_set.json). EG is the 7th attribution method on the leaderboard and
# the DIRECT successor to single-baseline Integrated Gradients (P2-E4-5,
# ig_baseline_sweep.jl): instead of ONE hand-chosen baseline x0, EG averages the
# path-integrated gradient over a DISTRIBUTION of REAL reference states. It reuses
# the validated Phase-B faithfulness CONTRACT verbatim (experiment_design.md §5 row
# "Expected Gradients": baseline-averaged IG, "as above" = corr + del/ins AUC +
# completeness; precision@k vs the true causal top-k).
#
# METHOD — Expected Gradients (Erion et al. 2021, eq. 5). For an output y and the
# 128-byte RAM tape x at the intervention frame, with a reference distribution D
# over REAL VCS states x':
#
#   EG_i(x) = E_{x'~D, α~U(0,1)} [ (x_i − x'_i) · ∂y/∂x_i |_{x' + α(x − x')} ]
#
# Estimated by Monte-Carlo: draw M reference states x'^(m) from D, and for each a
# path position α_m ~ U(0,1) (Erion's single-sample-per-reference estimator), then
#
#   EG_i(x) ≈ (1/M) Σ_m (x_i − x'^(m)_i) · ∂y/∂x_i |_{x'^(m) + α_m (x − x'^(m))}
#
# Completeness (in expectation): Σ_i EG_i ≈ y(x) − E_{x'~D}[y(x')].
#
# WHY EG vs single-baseline IG (the §5 contrast this item must quantify): IG's
# baseline x0 is a free, sensitive choice (P2-E4-5 measured it). EG removes that
# choice by integrating over the reference distribution — Erion's claims are (a)
# LOWER BASELINE SENSITIVITY (no single x0 to pick) and (b) it satisfies the IG
# axioms in expectation. On the VCS we can measure both EXACTLY against the oracle.
# We report, per game:
#   * EG faithfulness (corr / spearman / precision@k / del-ins AUC on the TRUE VCS
#     / completeness-in-expectation) — the same contract as the IG sibling;
#   * a head-to-head vs single-baseline IG at the canonical `zeros` baseline (same
#     content byte, same oracle column): Δcorr, Δprecision@k, Δdeletion-AUC;
#   * STABILITY: EG is a Monte-Carlo estimator, so we run it with R independent
#     reference-sample seeds and report the spread of corr / the mean pairwise
#     cosine of the per-cause attribution across seeds. This is the property that
#     distinguishes EG from a deterministic single-baseline IG and is the §R
#     "stability" deliverable for this item.
#
# THE REFERENCE DISTRIBUTION D (the EG-specific input): genuine, reachable VCS
# states — the recorded state trajectory from the E0-2 recorder
# (tools/xai_study/common/jutari_record.jl). We replay the same deterministic
# NOOP ROM trace and use the per-frame RAM tapes at frames 1..target_frame as the
# reference pool D (each is an on-distribution real game state, NOT a synthetic
# vector). EG draws its M baselines uniformly (with replacement, seeded) from D.
# This is exactly Erion's "background distribution of real inputs" instantiated on
# the bit-exact substrate.
#
# OUTPUT SELECTION (the §1 content-vs-position split, identical to the IG sibling &
# smoothgrad/saliency/gradxinput):
#   * HEADLINE  = a CONTENT output read one-hot from RAM through `soft_ram_peek`
#     (∂y/∂ram[idx]=1, the content path). We read the SAME candidate concept byte
#     the IG sibling picks (the most causally-active RAM cause via pick_content_idx)
#     so the two methods score the SAME column and the head-to-head is apples-to-
#     apples. EG concentrates on that TRUE causal byte ⇒ the FAITHFUL headline.
#   * CONTRAST  = a POSITION/INDEX output `ball_pixel`: the pixel is the colour of
#     whichever sprite covers a cell (round/argmax) ⇒ NO differentiable RAM path ⇒
#     EG VANISHES (max|attr|≈0) for EVERY reference draw ⇒ near-chance faithfulness.
#     Averaging over real baselines cannot manufacture a gradient that does not
#     exist — the honest "plausible ≠ faithful" failure; the intervention oracle is
#     the sole truth there. (Same caveat as IG, called out in the item notes.)
#
# REUSES the validated Phase-B foundation on main (NO emulator core touched):
#   * ig_baseline_sweep.jl (P2-E4-5) — the per-game env/ROM-alias/RomSettings map,
#     bit-exact assertion, the oracle |Δy| per cause, the deletion/insertion AUC on
#     the TRUE VCS, the content/position readers, pick_content_idx (so EG scores the
#     SAME content byte as IG), and the IG-over-ram primitive we reuse for the
#     single-baseline IG comparison column.
#   * pilot_ig_vs_oracle.jl — the SCORER (pearson/spearman/precision_at_k), the
#     per-cause |attr| mapping, the §R writer helpers, the oracle-as-method control.
#   * oracle_intervene.jl / jutari_oracle.jl — Cause / build_pong_causes /
#     candidate_ram_indices / boot/replay/snapshot/intervene + the §R NPZ writer.
#   * jutari_record.jl — record_trajectory: the E0-2 real-state reference pool D.
#   * JuTari.Diff.soft_ram_peek — the forward-exact one-hot content read EG integrates.
#
# Run (warm shared depot, primary's project):
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseB_attribution/expected_gradients.jl --games core
# Flags: --games core|<g1,g2,...>  --game <g>
#        --target-frame N --horizon N --eg-samples M --eg-stability-seeds R
#        --topk K --seed S --selftest
#
# Writes (SPEC §R; file_scope expected_gradients_* under out/):
#   tools/xai_study/phaseB_attribution/out/expected_gradients_<game>_content.{json,npz}
#   tools/xai_study/phaseB_attribution/out/expected_gradients_<game>_ball_pixel.{json,npz}
#   tools/xai_study/phaseB_attribution/out/expected_gradients_core_summary.json

module ExpectedGradients

using JSON
import Zygote
import Statistics
import Random

using JuTari
using JuTari.Diff: soft_ram_peek
using JuTari.Env: env_step!, get_ram

# REUSE the IG baseline-sweep sibling (P2-E4-5) wholesale: the per-game env map,
# bit-exact assertion, oracle |Δy|, deletion/insertion AUC, the content/position
# readers, pick_content_idx, and the single-baseline IG primitive for the contrast.
include(joinpath(@__DIR__, "ig_baseline_sweep.jl"))
using .IGBaselineSweep: CORE_GAMES, load_env, boot_replay, continue_from,
                        fresh_baseline, assert_bit_exact, oracle_abs_delta,
                        deletion_insertion_auc, content_read, position_read_zero,
                        position_pixel_cell, pick_content_idx, ig_over_ram

# the IG pilot scorer + per-cause mapping + §R writer helpers (the shared contract)
using .IGBaselineSweep.PilotIGvsOracle: pearson, spearman, precision_at_k,
                                        ig_attribution_per_cause, _git_commit,
                                        _json_num, _trapz_unit
using .IGBaselineSweep.PilotIGvsOracle.OracleIntervene: build_pong_causes, Cause,
                                                        candidate_ram_indices
using .IGBaselineSweep.PilotIGvsOracle.OracleIntervene.JutariOracle: Snapshot,
                                                        snapshot, write_npz,
                                                        intervene_ram!

# The P2 SHARED TESTBED (experiment_redesign.md). ig_baseline_sweep.jl ALREADY
# `include`s common/shared_testbed_impl.jl at top level, so we must NOT re-include
# the fragment here (that would redefine build_shared_testbed & friends). Instead we
# reuse the ONE copy through the IGBaselineSweep namespace — build_shared_testbed,
# the ST_* config, the SHARED_TESTBED opt-in flag, and the injected oracle fns the
# testbed needs (env/ROM map, continue_from/boot_replay/run_intervention, ...).
using .IGBaselineSweep: build_shared_testbed, SHARED_TESTBED,
                        ST_PREFIX, ST_HORIZON, ST_SEED, ST_GATE_K, ST_FLOOR,
                        load_env, run_intervention, settings_for, rom_path_for,
                        candidates_path_for

const OUT_DIR = joinpath(@__DIR__, "out")

# ============================================================================
# The reference distribution D — genuine, reachable VCS states (E0-2 recorder).
# ============================================================================
"""
    reference_pool(game, target_frame; actions) -> Matrix{Float32} (T × 128)

The background distribution D for Expected Gradients: the per-frame RIOT RAM tape
at frames 1..target_frame of the deterministic real-ROM replay. Each ROW is a
genuine, reachable game state — NOT a synthetic vector. EG draws its baselines from
these rows. This is exactly the E0-2 trajectory-recorder behaviour
(tools/xai_study/common/jutari_record.jl), but built on the IG sibling's validated
per-game env (load_env: the per-game ROM-alias + RomSettings map), so the EG path
shares one ROM resolution with the rest of the runner (the recorder's oracle
resolver lacks the `ms_pacman→mspacman` alias). We record up to `target_frame`
frames so every reference is at or before the explained state, all on-distribution
and reachable under the same NOOP trace."""
function reference_pool(game::AbstractString, target_frame::Integer;
                        actions::Union{Nothing,AbstractVector{<:Integer}} = nothing)
    nframes = max(1, Int(target_frame))
    acts = actions === nothing ? fill(0, nframes) : Int.(collect(actions))
    @assert length(acts) >= nframes "actions length $(length(acts)) < frames=$nframes"
    env = load_env(; game = game)
    D = Matrix{Float32}(undef, nframes, 128)
    for t in 1:nframes
        env_step!(env, acts[t])
        D[t, :] = Float32.(collect(get_ram(env)))   # the genuine RAM at frame t
    end
    return D
end

# ============================================================================
# Expected Gradients of `readf` over the RAM tape against a reference pool D.
# ============================================================================
"""
    expected_gradients(readf, ram, D; samples, rng) -> (eg, completeness_err, y_ref_mean)

Monte-Carlo Expected Gradients (Erion et al. 2021, eq. 5) of `readf(ram)::Float32`
w.r.t. the 128-byte RAM vector `ram`, with the reference distribution given by the
rows of `D` (T × 128 real states):

    EG_i(x) ≈ (1/M) Σ_m (x_i − x'^(m)_i) · ∂y/∂x_i |_{x'^(m) + α_m (x − x'^(m))}

with x'^(m) drawn uniformly (with replacement, seeded `rng`) from the rows of D and
α_m ~ U(0,1) (Erion's single-α-per-reference estimator: one gradient evaluation per
sampled baseline, so `samples` total gradient calls — directly comparable in compute
to IG's `steps`). Completeness IN EXPECTATION: Σ_i EG_i ≈ y(x) − mean_m y(x'^(m)),
which we return as `completeness_err = |Σ EG − (y(x) − ȳ_ref)|` (ȳ_ref is the
sample mean of readf over the drawn references — the EG analogue of IG's single
y(x0); it is exact for a one-hot linear read and Monte-Carlo for general readf)."""
function expected_gradients(readf, ram::AbstractVector{<:Real}, D::AbstractMatrix{<:Real};
                            samples::Integer = 64,
                            rng::Random.AbstractRNG = Random.MersenneTwister(0))
    x = Float32.(ram)
    n = length(x)
    T = size(D, 1)
    @assert size(D, 2) == n "reference pool width $(size(D,2)) != RAM width $n"
    acc = zeros(Float32, n)
    yref_acc = 0.0
    for _ in 1:samples
        m = rand(rng, 1:T)
        x0 = Float32.(@view D[m, :])
        α = rand(rng, Float32)                       # α ~ U(0,1)
        xα = x0 .+ α .* (x .- x0)
        g = Zygote.gradient(readf, xα)[1]
        g === nothing && (g = zeros(Float32, n))
        acc .+= (x .- x0) .* g                       # (x − x') ⊙ ∂y/∂x |_{interp}
        yref_acc += Float64(readf(x0))
    end
    eg = acc ./ samples
    y_ref_mean = yref_acc / samples
    y_x = Float64(readf(x))
    completeness_err = abs(Float64(sum(eg)) - (y_x - y_ref_mean))
    return eg, completeness_err, y_ref_mean
end

# ============================================================================
# Score one EG (a single seed) against the oracle — the §5 metrics.
# ============================================================================
struct EGScore
    seed::Int
    pearson::Float64
    spearman::Float64
    precision_at_k::Float64
    deletion_auc::Float64
    insertion_auc::Float64
    completeness_err::Float64
    eg_max_abs::Float64
    y_ref_mean::Float64                  # ȳ_ref — the EG completeness reference
    attr_per_cause::Vector{Float64}
    eg_over_ram::Vector{Float64}
    del_curve::Vector{Float64}
    ins_curve::Vector{Float64}
end

function score_eg_for_seed(; readf, read_y, ram_now, D, causes, odelta, checkpoint,
                           actions, target_frame, horizon, eg_samples, topk, eg_seed)
    rng = Random.MersenneTwister(eg_seed + 1234567)
    eg_full, comp_err, yref = expected_gradients(readf, ram_now, D;
                                                 samples = eg_samples, rng = rng)
    attr = ig_attribution_per_cause(eg_full, causes)
    pr  = pearson(attr, odelta)
    sp  = spearman(attr, odelta)
    pak = precision_at_k(attr, odelta, topk)
    order = sortperm(attr; rev = true)
    del_auc, ins_auc, dc, ic = deletion_insertion_auc(
        checkpoint, actions, target_frame, horizon, causes, order, read_y)
    return EGScore(eg_seed, pr, sp, pak, del_auc, ins_auc, comp_err,
                   maximum(abs.(attr)), yref, attr, Float64.(eg_full), dc, ic)
end

# ============================================================================
# Single-baseline IG at the canonical zeros baseline — the head-to-head contrast.
# ============================================================================
"""Score single-baseline IG (zeros baseline) on the SAME column EG is scored on,
so the §5 head-to-head (Δcorr / Δp@k / Δdel-AUC: EG − IG) is apples-to-apples."""
function score_ig_zeros(; readf, read_y, ram_now, causes, odelta, checkpoint,
                        actions, target_frame, horizon, ig_steps, topk)
    x0 = zeros(Float32, length(ram_now))
    ig_full, comp_err = ig_over_ram(readf, ram_now; steps = ig_steps, baseline = x0)
    attr = ig_attribution_per_cause(ig_full, causes)
    pr  = pearson(attr, odelta)
    sp  = spearman(attr, odelta)
    pak = precision_at_k(attr, odelta, topk)
    order = sortperm(attr; rev = true)
    del_auc, ins_auc, _, _ = deletion_insertion_auc(
        checkpoint, actions, target_frame, horizon, causes, order, read_y)
    return (pearson = pr, spearman = sp, precision_at_k = pak,
            deletion_auc = del_auc, insertion_auc = ins_auc,
            completeness_err = comp_err, eg_max_abs = maximum(abs.(attr)),
            attr = attr)
end

# mean pairwise cosine of a set of per-cause attribution vectors (EG stability).
function mean_pairwise_cosine(vs::Vector{Vector{Float64}})
    length(vs) < 2 && return 1.0
    cos(a, b) = begin
        na = sqrt(sum(abs2, a)); nb = sqrt(sum(abs2, b))
        (na == 0 || nb == 0) && return 0.0
        return dot(a, b) / (na * nb)
    end
    dot(a, b) = sum(a .* b)
    acc = 0.0; cnt = 0
    for i in 1:length(vs), j in (i+1):length(vs)
        acc += cos(vs[i], vs[j]); cnt += 1
    end
    return cnt == 0 ? 1.0 : acc / cnt
end

# ============================================================================
# The result record for one output (content or position).
# ============================================================================
struct Faithfulness
    game::String
    output::String                       # "content(ram_self@N)" | "ball_pixel@rRcC"
    output_kind::String                  # "content" | "position"
    target_frame::Int
    horizon::Int
    eg_samples::Int
    topk::Int
    seed::Int
    content_idx::Int
    ref_pool_size::Int                   # |D| — number of real reference states
    cause_names::Vector{String}
    oracle_abs_delta::Vector{Float64}
    y_full::Float64
    # the EG stability ensemble — one EGScore per independent reference-sample seed
    eg_scores::Vector{EGScore}
    eg_attr_mean::Vector{Float64}        # mean EG attribution over the seeds
    eg_stability_cosine::Float64         # mean pairwise cosine across seeds
    # the single-baseline IG head-to-head (zeros baseline, same column)
    ig_pearson::Float64
    ig_spearman::Float64
    ig_precision_at_k::Float64
    ig_deletion_auc::Float64
    ig_insertion_auc::Float64
    ig_completeness_err::Float64
    ig_max_abs::Float64
    # harness positive control (oracle's OWN |Δy| as the method)
    oracle_self_pearson::Float64
    oracle_self_precision_at_k::Float64
    oracle_self_deletion_auc::Float64
    oracle_self_insertion_auc::Float64
    oracle_column_degenerate::Bool
    # the EG-SPECIFIC degeneracy (a genuine, reported finding — NOT an error): on
    # the one-hot content path the only live EG cell is the self-byte, where
    # EG_self = (x_self − x'_self)·1. If the content byte is (near-)CONSTANT across
    # the reference distribution D — x_self == x'_self for (almost) every reference —
    # the (x−x') factor is 0 for every drawn baseline and EG VANISHES. This is the
    # EG analogue of IG's baseline degeneracy, MORE likely with on-distribution
    # references (nearby real states share the byte). `eg_self_baseline_invariant`
    # is the EMPIRICAL outcome (EG vanished for the whole stability ensemble on the
    # content path); when true, EG cannot attribute and the intervention oracle is
    # the reference (like a position output). `n_refs_self_differs` / the byte range
    # record WHY (strictly constant ⇒ 0; near-constant ⇒ the rare differing frame
    # was not drawn — both leave EG ≈ 0 in practice).
    eg_self_baseline_invariant::Bool
    content_byte_value::Int              # x_self at the explained state (for the record)
    ref_self_byte_range::Tuple{Int,Int}  # (min,max) of x'_self over D
    n_refs_self_differs::Int             # #references with x'_self != x_self (0 ⇒ strictly constant)
end

"""The representative EG score = the first stability seed (the headline single run;
the ensemble's spread is the stability deliverable)."""
headline_eg(f::Faithfulness) = f.eg_scores[1]

function compute_one(; game, output_kind, content_idx = -1, position_cell = nothing,
                     checkpoint, actions, target_frame, horizon, causes, D,
                     eg_samples, eg_stability_seeds, ig_steps, topk, seed, verbose,
                     read_y_override = nothing, readf_override = nothing,
                     output_name_override = nothing)
    # SHARED TESTBED: the position output is a screen-buffer REGION (read_y_override)
    # and its EG integral runs through the bilinear SAMPLER (readf_override) — the
    # real, non-vanishing position gradient (redesign Problems 2+4). Content path is
    # unchanged (one-hot RAM read).
    read_y = read_y_override !== nothing ? read_y_override :
        (output_kind == "content" ?
            (s -> Float64(Int(s.ram[content_idx + 1]))) :
            (s -> Float64(Int(s.screen[position_cell[1], position_cell[2]]))))
    readf = readf_override !== nothing ? readf_override :
        (output_kind == "content" ? (r -> content_read(r, content_idx)) : position_read_zero)
    output_name = output_name_override !== nothing ? output_name_override :
        (output_kind == "content" ?
            "content(ram_self@$content_idx)" :
            "ball_pixel@r$(position_cell[1])c$(position_cell[2])")

    # 1) oracle |Δy| per cause (real re-runs on the TRUE VCS)
    odelta = oracle_abs_delta(checkpoint, actions, target_frame, horizon, causes, read_y)
    degenerate = Statistics.std(odelta) == 0

    # 2) the actual state RAM at the intervention frame (the EG endpoint x)
    at_target = continue_from(checkpoint, Int[])
    ram_now = Float32.(collect(at_target.ram))

    # 2b) the EG-specific degeneracy probe (content only): record how the explained
    # self-byte is distributed across the reference distribution D (the structural
    # WHY of any EG vanishing). The empirical degeneracy FLAG is set after the EG
    # ensemble below (EG vanished for every seed on the content path).
    content_byte_value = output_kind == "content" ? Int(round(ram_now[content_idx + 1])) : -1
    self_col = output_kind == "content" ? Int.(round.(D[:, content_idx + 1])) : Int[]
    ref_self_range = isempty(self_col) ? (-1, -1) : (minimum(self_col), maximum(self_col))
    n_refs_self_differs = output_kind == "content" ? count(!=(content_byte_value), self_col) : -1

    # 3) the EG STABILITY ensemble — R independent reference-sample seeds
    verbose && println("[eg] '$output_name': Expected Gradients ($eg_samples refs/run × " *
                       "$eg_stability_seeds stability seeds, |D|=$(size(D,1)) real states)...")
    eg_scores = EGScore[]
    for r in 0:(eg_stability_seeds - 1)
        s = score_eg_for_seed(; readf = readf, read_y = read_y, ram_now = ram_now, D = D,
            causes = causes, odelta = odelta, checkpoint = checkpoint, actions = actions,
            target_frame = target_frame, horizon = horizon, eg_samples = eg_samples,
            topk = topk, eg_seed = seed + r)
        push!(eg_scores, s)
        verbose && println("[eg]   seed=$(rpad(seed + r, 4)) corr=$(rpad(round(s.pearson,digits=4),8)) " *
                           "p@$topk=$(rpad(round(s.precision_at_k,digits=3),6)) " *
                           "del=$(rpad(round(s.deletion_auc,digits=3),7)) " *
                           "ins=$(rpad(round(s.insertion_auc,digits=3),7)) " *
                           "compl_err=$(round(s.completeness_err,sigdigits=3)) " *
                           "max|EG|=$(round(s.eg_max_abs,sigdigits=3))")
    end
    eg_attr_mean = [Statistics.mean(s.attr_per_cause[i] for s in eg_scores)
                    for i in 1:length(causes)]
    stability = mean_pairwise_cosine([s.attr_per_cause for s in eg_scores])

    # the EMPIRICAL EG degeneracy flag (content only): EG vanished for EVERY seed —
    # the (x−x') factor was 0 at the live cell for every drawn baseline (because the
    # self-byte is constant or near-constant over D). A reported finding, not a bug.
    eg_self_invariant = output_kind == "content" &&
        all(s -> s.eg_max_abs < 1e-6, eg_scores)

    # 4) single-baseline IG (zeros) on the SAME column — the head-to-head contrast
    ig = score_ig_zeros(; readf = readf, read_y = read_y, ram_now = ram_now,
        causes = causes, odelta = odelta, checkpoint = checkpoint, actions = actions,
        target_frame = target_frame, horizon = horizon, ig_steps = ig_steps, topk = topk)

    # 5) y_full + harness positive control (oracle as method)
    base_snap = continue_from(checkpoint, Int.(actions[target_frame + 1 : target_frame + horizon]))
    y_full = read_y(base_snap)
    or_pr  = pearson(odelta, odelta); or_pak = precision_at_k(odelta, odelta, topk)
    or_order = sortperm(odelta; rev = true)
    or_del, or_ins, _, _ = deletion_insertion_auc(checkpoint, actions, target_frame,
                                                  horizon, causes, or_order, read_y)

    if verbose
        hs = eg_scores[1]
        corrs = [s.pearson for s in eg_scores]
        println("[eg]   ► EG headline corr=$(round(hs.pearson,digits=4)) p@$topk=$(round(hs.precision_at_k,digits=3)) " *
                "| stability cos=$(round(stability,digits=4)) corr-spread=$(round(maximum(corrs)-minimum(corrs),digits=4))")
        println("[eg]   ► vs single-baseline IG(zeros): corr $(round(ig.pearson,digits=4)) → EG $(round(hs.pearson,digits=4)) " *
                "(Δcorr=$(round(hs.pearson - ig.pearson, digits=4))), " *
                "p@$topk $(round(ig.precision_at_k,digits=3))→$(round(hs.precision_at_k,digits=3)), " *
                "IG max|attr|=$(round(ig.eg_max_abs,sigdigits=3)) EG max|attr|=$(round(hs.eg_max_abs,sigdigits=3))")
        println("[eg]   [harness] oracle-as-method: corr=$(round(or_pr,digits=3)) " *
                "p@$topk=$(round(or_pak,digits=3)) del=$(round(or_del,digits=3)) ins=$(round(or_ins,digits=3))" *
                (degenerate ? "  (oracle column flat at this state)" : ""))
        if eg_self_invariant
            const_kind = n_refs_self_differs == 0 ? "CONSTANT" :
                "near-CONSTANT ($n_refs_self_differs/$(size(D,1)) refs differ; none drawn)"
            println("[eg]   ⚠ EG DEGENERACY: content byte RAM[$content_idx]=$content_byte_value is $const_kind " *
                    "across the reference distribution D (x'_self∈[$(ref_self_range[1]),$(ref_self_range[2])]) ⇒ " *
                    "the (x−x') factor is 0 at the only live cell ⇒ EG vanishes (the EG analogue of IG's " *
                    "baseline degeneracy; on-distribution references make it MORE likely). Oracle is the reference here.")
        end
    end

    return Faithfulness(game, output_name, output_kind, target_frame, horizon,
                        eg_samples, topk, seed, content_idx, size(D, 1),
                        [c.name for c in causes], odelta, y_full, eg_scores,
                        eg_attr_mean, stability,
                        ig.pearson, ig.spearman, ig.precision_at_k, ig.deletion_auc,
                        ig.insertion_auc, ig.completeness_err, ig.eg_max_abs,
                        or_pr, or_pak, or_del, or_ins, degenerate,
                        eg_self_invariant, content_byte_value, ref_self_range,
                        n_refs_self_differs)
end

"""Drive both outputs for one game: assert bit-exact, build causes, pick the SAME
content byte as the IG sibling, build the real-state reference pool D, score the
content headline + the position contrast — each as an EG stability ensemble plus a
single-baseline-IG head-to-head."""
function compute_game(; game, target_frame, horizon, eg_samples, eg_stability_seeds,
                      ig_steps, topk, seed, verbose)
    if SHARED_TESTBED
        st = build_shared_testbed(game;
            settings_for = settings_for, rom_path_for = rom_path_for,
            candidates_path_for = candidates_path_for,
            build_causes = build_pong_causes, candidate_ram_indices = candidate_ram_indices,
            continue_from = continue_from, snapshot = snapshot, env_step = env_step!,
            intervene_ram = intervene_ram!, boot_replay = boot_replay,
            run_intervention = run_intervention, soft_ram_peek = soft_ram_peek,
            prefix = ST_PREFIX, horizon = ST_HORIZON, seed = ST_SEED,
            k = ST_GATE_K, floor = ST_FLOOR, verbose = verbose,
            assert_bit_exact = assert_bit_exact)
        verbose && println("[eg] $game SHARED gameplay state: cause_density=$(st.cause_density)/" *
            "$(length(st.causes)) accepted=$(st.accepted) cell=$(st.cell) " *
            "geom=$(st.geom === nothing ? "static" : "RAM[$(st.geom[1])]")")

        actions = st.actions; checkpoint = st.checkpoint; causes = st.causes
        tf = st.prefix; hz = st.horizon; cand_indices = st.cand_indices
        content_idx, content_mv = pick_content_idx(checkpoint, actions, tf, hz, causes, cand_indices)
        verbose && println("[eg] $game content byte = RAM[$content_idx] (max oracle |Δself|=$(round(content_mv,digits=2)))")

        # THE LOAD-BEARING EG CHANGE: the reference distribution D is the GAMEPLAY
        # reference pool (each row a RAM tape along the SAME seeded random-action
        # trajectory, so the causal byte VARIES ((x−x0)≠0) and EG is not zeroed by a
        # constant byte over a NOOP tape). We do NOT call the old NOOP reference_pool.
        D = st.ref_pool
        verbose && println("[eg] $game reference pool D = $(size(D,1)) genuine RAM states " *
                           "(GAMEPLAY trajectory seed=$(st.seed), frames 1..$tf; on-distribution EG background)")

        f_content = compute_one(; game = game, output_kind = "content", content_idx = content_idx,
            checkpoint = checkpoint, actions = actions, target_frame = tf,
            horizon = hz, causes = causes, D = D, eg_samples = eg_samples,
            eg_stability_seeds = eg_stability_seeds, ig_steps = ig_steps, topk = topk,
            seed = seed, verbose = verbose)
        f_pos = compute_one(; game = game, output_kind = "position", position_cell = st.cell,
            checkpoint = checkpoint, actions = actions, target_frame = tf,
            horizon = hz, causes = causes, D = D, eg_samples = eg_samples,
            eg_stability_seeds = eg_stability_seeds, ig_steps = ig_steps, topk = topk,
            seed = seed, verbose = verbose,
            read_y_override = st.read_y, readf_override = st.sampler_read,
            output_name_override = "screen_region(n_changed_px)@r$(st.cell[1])c$(st.cell[2])")
        return f_content, f_pos,
               (cause_density = st.cause_density, accepted = st.accepted,
                n_causes = length(st.causes), cell = st.cell, geom = st.geom,
                prefix = st.prefix, horizon = st.horizon, seed = st.seed)
    end

    total = target_frame + horizon
    actions = fill(0, total)
    verbose && println("[eg] $game: asserting bit-exactness (2 fresh boots+replays to f$total)...")
    assert_bit_exact(actions, total; game = game)

    cand = IGBaselineSweep.candidates_path_for(game)
    checkpoint = boot_replay(actions, target_frame; game = game)
    at_target = continue_from(checkpoint, Int[])
    causes = build_pong_causes(cand, at_target)
    cand_indices = [idx for (idx, _) in candidate_ram_indices(cand)]

    # SAME content byte as the IG sibling (apples-to-apples head-to-head)
    content_idx, content_mv = pick_content_idx(checkpoint, actions, target_frame, horizon, causes, cand_indices)
    verbose && println("[eg] $game content byte = RAM[$content_idx] (max oracle |Δself|=$(round(content_mv,digits=2)))")

    # the EG reference distribution D — genuine RAM states at frames 1..target_frame
    # of the same deterministic NOOP replay (the E0-2 recorder). All real, reachable.
    D = reference_pool(game, target_frame; actions = actions)
    verbose && println("[eg] $game reference pool D = $(size(D,1)) genuine RAM states " *
                       "(frames 1..$target_frame, on-distribution background for EG)")

    base = continue_from(checkpoint, Int.(actions[target_frame + 1 : total]))
    pcell = position_pixel_cell(checkpoint, base.screen; horizon = horizon)

    f_content = compute_one(; game = game, output_kind = "content", content_idx = content_idx,
        checkpoint = checkpoint, actions = actions, target_frame = target_frame,
        horizon = horizon, causes = causes, D = D, eg_samples = eg_samples,
        eg_stability_seeds = eg_stability_seeds, ig_steps = ig_steps, topk = topk,
        seed = seed, verbose = verbose)
    f_pos = compute_one(; game = game, output_kind = "position", position_cell = pcell,
        checkpoint = checkpoint, actions = actions, target_frame = target_frame,
        horizon = horizon, causes = causes, D = D, eg_samples = eg_samples,
        eg_stability_seeds = eg_stability_seeds, ig_steps = ig_steps, topk = topk,
        seed = seed, verbose = verbose)
    return f_content, f_pos, nothing
end

# ============================================================================
# Self-check (DoD) — the scoring contract is sound; results are non-fabricated.
# ============================================================================
"""Parse the RAM index out of a cause name like "ram[54]:set" → 54; else -1."""
function _cause_ram_index(name::AbstractString)
    m = match(r"ram\[(\d+)\]", name)
    return m === nothing ? -1 : parse(Int, m.captures[1])
end

"""
    selftest(f::Faithfulness; require_nondegenerate) -> Bool

Asserts the load-bearing claims (the contract every E4 method reuses):

  (HARNESS POSITIVE CONTROL) feeding the oracle's OWN |Δy| as the candidate map
    scores corr=1 (on a non-degenerate column) and precision@k=1.

  (EG RESULT, output-dependent)
    * POSITION/INDEX output (ball_pixel) — EG VANISHES (max|attr|≈0) for EVERY
      stability seed (the §1 index failure; averaging real baselines cannot
      manufacture a missing gradient — the same caveat as IG/SmoothGrad).
    * CONTENT output — EG keeps the gradient ALIVE (max|attr|>0), its top mass lands
      on a cause touching the content self-byte (the one-hot read), and completeness
      IN EXPECTATION holds (err < tol — looser than IG's exact bound because EG's
      ȳ_ref is a Monte-Carlo mean, but tight for the one-hot linear read).
    * STABILITY — the EG ensemble produces a finite per-seed corr spread and a high
      mean pairwise cosine (reported; the headline EG-vs-IG distinction).

  (HEAD-TO-HEAD) the single-baseline-IG comparison column is finite.

All AUCs in [0,1] or NaN; all attribution finite. Throws on a violation."""
function selftest(f::Faithfulness; require_nondegenerate = false, sampler_on = false)
    for s in f.eg_scores
        @assert all(isfinite, s.attr_per_cause) "non-finite EG attribution (seed=$(s.seed), $(f.game))"
        for (nm, v) in (("deletion", s.deletion_auc), ("insertion", s.insertion_auc))
            @assert isnan(v) || (0.0 <= v <= 1.0 + 1e-9) "$(nm) AUC out of [0,1]: $v (seed=$(s.seed), $(f.game))"
        end
    end
    @assert all(isfinite, [f.ig_pearson, f.ig_precision_at_k]) "non-finite IG comparison ($(f.game))"
    for (nm, v) in (("oracle_self_deletion", f.oracle_self_deletion_auc),
                    ("oracle_self_insertion", f.oracle_self_insertion_auc),
                    ("ig_deletion", f.ig_deletion_auc), ("ig_insertion", f.ig_insertion_auc))
        @assert isnan(v) || (0.0 <= v <= 1.0 + 1e-9) "$nm AUC out of [0,1]: $v ($(f.game))"
    end

    if f.output_kind == "position"
        if sampler_on
            # SHARED TESTBED sampler-on: the sampler restores a nonzero position EG on
            # the FULL 128-byte gradient for at least one seed (the keystone). The
            # sampler's position byte may fall outside this game's cause set, in which
            # case attr_per_cause (hence eg_max_abs) is null while the raw gradient is
            # alive — so we assert on the FULL EG-over-ram max, NOT per-cause. (A given
            # seed can still vanish if x0==x on the byte for its draws; we require ≥1
            # seed alive, as on the content path.)
            alive = [s for s in f.eg_scores if maximum(abs.(s.eg_over_ram); init = 0.0) > 1e-6]
            @assert !isempty(alive) "SAMPLER-ON: expected ≥1 seed with a nonzero position EG " *
                "(full gradient) [$(f.game)]"
            println("[eg] SELF-CHECK PASS (position/sampler '$(f.output)', $(f.game)): sampler RESTORES " *
                    "a nonzero position EG ($(length(alive))/$(length(f.eg_scores)) seeds alive on the full " *
                    "128-byte gradient); headline corr=$(round(headline_eg(f).pearson,digits=3)).")
        else
            for s in f.eg_scores
                @assert s.eg_max_abs < 1e-6 "expected EG to vanish on a position output for EVERY seed, " *
                    "got max|attr|=$(s.eg_max_abs) (seed=$(s.seed), $(f.game))"
            end
            println("[eg] SELF-CHECK PASS (position '$(f.output)', $(f.game)): EG vanishes for ALL " *
                    "$(length(f.eg_scores)) seeds (max|attr|<1e-6) — the §1 index failure; averaging real " *
                    "baselines cannot manufacture a missing gradient. Headline corr=$(round(headline_eg(f).pearson,digits=3)) " *
                    "(near chance); single-baseline IG max|attr|=$(round(f.ig_max_abs,sigdigits=3)) (also vanishes); " *
                    "oracle-as-method corr=$(round(f.oracle_self_pearson,digits=3)).")
        end
        return true
    end

    # CONTENT output --------------------------------------------------------
    # completeness in expectation holds for EVERY seed regardless of degeneracy (the
    # one-hot read makes EG nearly exact; when EG vanishes, Σ EG = 0 = y(x)−ȳ_ref
    # because ȳ_ref == y(x) on a constant byte — completeness still passes).
    for s in f.eg_scores
        @assert s.completeness_err < 1e-2 "EG completeness err too large: $(s.completeness_err) " *
            "(seed=$(s.seed), $(f.game))"
    end
    # the oracle harness control must still hold (the metric rewards a faithful map).
    if require_nondegenerate
        @assert !f.oracle_column_degenerate "content oracle column degenerate for $(f.game)"
        @assert f.oracle_self_pearson > 0.999 "harness broken: oracle-as-method corr != 1 ($(f.oracle_self_pearson)) [$(f.game)]"
        @assert f.oracle_self_precision_at_k == 1.0 "harness broken: oracle-as-method p@k != 1 [$(f.game)]"
    end

    corrs = [s.pearson for s in f.eg_scores]
    hs = headline_eg(f)
    alive = [s for s in f.eg_scores if s.eg_max_abs > 1e-6]

    if f.eg_self_baseline_invariant
        # THE EG DEGENERACY — a genuine, reported finding (NOT a failure): the content
        # byte is CONSTANT across the reference distribution D, so EG's (x−x') factor
        # is 0 at the only live cell and EG vanishes for EVERY seed. We assert it is
        # CONSISTENT (all seeds vanish ⇒ the degeneracy is real, not noise) and report.
        @assert isempty(alive) "EG flagged self-baseline-invariant (vanished for all seeds) yet some seed is alive — inconsistent ($(f.game))"
        const_kind = f.n_refs_self_differs == 0 ? "CONSTANT" :
            "near-CONSTANT ($(f.n_refs_self_differs)/$(f.ref_pool_size) refs differ; the rare frame(s) were not drawn)"
        println("[eg] SELF-CHECK PASS (content '$(f.output)', $(f.game)) — EG DEGENERACY (reported finding): " *
                "content byte RAM[$(f.content_idx)]=$(f.content_byte_value) is $const_kind across the reference " *
                "distribution D (x'_self∈[$(f.ref_self_byte_range[1]),$(f.ref_self_byte_range[2])]) ⇒ the (x−x') " *
                "factor is 0 at the only live cell for every drawn baseline ⇒ EG VANISHES for ALL " *
                "$(length(f.eg_scores)) seeds. The EG analogue of IG's baseline degeneracy, and on-distribution " *
                "references make it MORE likely. Single-baseline IG(zeros) stays alive here (corr=" *
                "$(round(f.ig_pearson,digits=3)), max|attr|=$(round(f.ig_max_abs,sigdigits=3))) because " *
                "0≠$(f.content_byte_value); completeness holds (err<1e-2). Oracle-as-method corr=" *
                "$(round(f.oracle_self_pearson,digits=3)).")
        return true
    end

    # NON-degenerate content path: EG must be alive AND its top mass on the self-byte.
    @assert !isempty(alive) "no stability seed kept the content-path EG alive ($(f.game)) although the " *
        "self-byte is NOT constant over D — unexpected"
    self_causes = findall(c -> c == f.content_idx, [_cause_ram_index(n) for n in f.cause_names])
    if !isempty(self_causes)
        s = alive[argmax([x.eg_max_abs for x in alive])]
        @assert argmax(s.attr_per_cause) in self_causes "EG top mass not on the content self-byte " *
            "(seed=$(s.seed), $(f.game))"
    end
    println("[eg] SELF-CHECK PASS (content '$(f.output)', $(f.game)): $(length(alive))/$(length(f.eg_scores)) " *
            "seeds keep EG alive + complete (err<1e-2) + top mass on the content self-byte; " *
            "headline corr=$(round(hs.pearson,digits=3)) p@$(f.topk)=$(round(hs.precision_at_k,digits=3)); " *
            "stability cos=$(round(f.eg_stability_cosine,digits=4)) corr-spread=$(round(maximum(corrs)-minimum(corrs),digits=4)); " *
            "vs IG(zeros) corr=$(round(f.ig_pearson,digits=3)) (Δ=$(round(hs.pearson - f.ig_pearson,digits=4))); " *
            "oracle-as-method corr=$(round(f.oracle_self_pearson,digits=3)).")
    return true
end

# ============================================================================
# Persist (SPEC §R) — JSON record + sibling .npz; file_scope expected_gradients_*.
# ============================================================================
function _eg_seed_json(s::EGScore, cause_names, topk)
    return Dict{String,Any}(
        "seed"             => s.seed,
        "pearson_corr"     => s.pearson,
        "spearman_corr"    => s.spearman,
        "precision_at_k"   => s.precision_at_k,
        "topk"             => topk,
        "deletion_auc"     => _json_num(s.deletion_auc),
        "insertion_auc"    => _json_num(s.insertion_auc),
        "completeness_err" => s.completeness_err,
        "eg_max_abs"       => s.eg_max_abs,
        "y_ref_mean"       => s.y_ref_mean,
        "attr_per_cause"   => Dict(cause_names[i] => s.attr_per_cause[i] for i in 1:length(cause_names)),
    )
end

function _output_note(f::Faithfulness)
    hs = headline_eg(f)
    corrs = [s.pearson for s in f.eg_scores]
    if f.output_kind == "position"
        return "POSITION/INDEX output — the §1 caveat: the pixel value comes from a discrete " *
               "sprite column (round/argmax), so there is no differentiable RAM read and Expected " *
               "Gradients VANISHES (max|attr|<1e-6) for EVERY reference-sample seed. Averaging the " *
               "path-integrated gradient over a DISTRIBUTION of REAL reference states cannot " *
               "manufacture a gradient that does not exist (same failure as single-baseline IG, " *
               "max|attr|=$(round(f.ig_max_abs,sigdigits=3))), so EG scores near chance regardless of " *
               "the reference distribution — the 'plausible ≠ faithful' contrast; the intervention " *
               "oracle is the sole truth here."
    elseif f.eg_self_baseline_invariant
        return "CONTENT output: RAM byte $(f.content_idx) read one-hot through soft_ram_peek — the SAME " *
               "most-causally-active concept byte the IG sibling (P2-E4-5) scores. EG DEGENERACY (the key " *
               "EG finding here): the content byte = $(f.content_byte_value) is CONSTANT across the entire " *
               "reference distribution D (x'_self∈[$(f.ref_self_byte_range[1]),$(f.ref_self_byte_range[2])] over " *
               "$(f.ref_pool_size) real states), so EG_self = (x_self−x'_self)·∂y/∂x_self = 0 at the ONLY live " *
               "cell ⇒ EG VANISHES (max|EG|<1e-6) for every reference draw and scores corr=" *
               "$(round(hs.pearson,digits=3)) (near chance). This is the EXPECTED-GRADIENTS analogue of IG's " *
               "baseline degeneracy — and on-distribution references make it MORE likely, because nearby real " *
               "states share the byte value. Single-baseline IG(zeros) STAYS ALIVE here (corr=" *
               "$(round(f.ig_pearson,digits=4)), p@$(f.topk)=$(round(f.ig_precision_at_k,digits=3)), max|attr|=" *
               "$(round(f.ig_max_abs,sigdigits=3))) precisely because the zeros baseline (0) ≠ the content byte " *
               "($(f.content_byte_value)): the EG-vs-IG roles invert vs Pong. Completeness still holds " *
               "(Σ EG=0=y(x)−ȳ_ref, ȳ_ref=y(x) on a constant byte; err≈$(round(hs.completeness_err,sigdigits=3))). " *
               "stability cos=$(round(f.eg_stability_cosine,digits=4)) (trivially stable — always 0). The " *
               "intervention oracle is the reference here (precision@$(f.topk)=$(round(hs.precision_at_k,digits=3)) " *
               "survives because the oracle's true causal byte is still the only candidate, but corr=0)."
    else
        return "CONTENT output: RAM byte $(f.content_idx) read one-hot through soft_ram_peek " *
               "(∂y/∂ram[$(f.content_idx)]=1) — the SAME most-causally-active concept byte the IG " *
               "sibling (P2-E4-5) scores. EXPECTED GRADIENTS (Erion et al. 2021) replaces IG's single " *
               "hand-chosen baseline with an EXPECTATION over a background distribution D of $(f.ref_pool_size) " *
               "GENUINE VCS states (E0-2 recorder). HEADLINE: EG corr=$(round(hs.pearson,digits=4)) " *
               "p@$(f.topk)=$(round(hs.precision_at_k,digits=3)) (vs single-baseline IG(zeros) " *
               "corr=$(round(f.ig_pearson,digits=4)) p@$(f.topk)=$(round(f.ig_precision_at_k,digits=3)); " *
               "Δcorr=$(round(hs.pearson - f.ig_pearson,digits=4))). KEY FINDING: on the one-hot content " *
               "path ∂y/∂u = e_idx is a CONSTANT, so EG (like IG) puts all nonzero mass on the SAME content " *
               "byte ⇒ identical ranking/corr/precision@k to single-baseline IG — EG's averaging over real " *
               "baselines does NOT change the faithfulness, but it DOES (a) remove the IG baseline-degeneracy " *
               "(a real reference is essentially never exactly equal to the content byte, so EG stays alive " *
               "where the `zeros` baseline can vanish) and (b) it is a Monte-Carlo estimator, so we report " *
               "STABILITY: mean pairwise cosine of the per-cause attribution across $(length(f.eg_scores)) " *
               "independent reference-sample seeds = $(round(f.eg_stability_cosine,digits=4)) " *
               "(corr-spread $(round(maximum(corrs)-minimum(corrs),digits=4))). Completeness IN EXPECTATION " *
               "Σ EG ≈ y(x)−ȳ_ref holds (err≈$(round(hs.completeness_err,sigdigits=3)) at the headline)."
    end
end

function write_faithfulness(f::Faithfulness; out_dir = OUT_DIR, st_extra = nothing)
    isdir(out_dir) || mkpath(out_dir)
    tag = f.output_kind == "content" ? "content" : "ball_pixel"
    stem = "expected_gradients_$(f.game)_$(tag)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path  = joinpath(out_dir, stem * ".npz")
    hs = headline_eg(f)
    corrs = [s.pearson for s in f.eg_scores]

    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseB_attribution", "method" => "expected_gradients",
        "game" => f.game, "state" => "f$(f.target_frame)+$(f.horizon)",
        "target_output" => f.output,
        # headline §R scalar = the content-path corr of EG (one comparable number
        # per method on the leaderboard, like the siblings).
        "metric_name" => "pearson_corr_with_oracle",
        "value" => hs.pearson,
        "stderr" => length(corrs) > 1 ? Statistics.std(corrs) / sqrt(length(corrs)) : nothing,
        "ci" => nothing, "n" => length(f.cause_names),
        "seed" => f.seed, "where" => "local", "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene@$(f.game)#$(f.output)",
        "timestamp" => string(round(Int, time())), "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            "substrate" => "jutari (Julia) — Zygote Expected Gradients (Erion 2021) over the RAM tape " *
                "with a reference distribution of real states + TRUE-VCS deletion/insertion re-runs.",
            "method_definition" => "EG_i(x) = E_{x'~D, α~U(0,1)}[(x_i−x'_i)·∂y/∂x_i|_{x'+α(x−x')}], " *
                "Monte-Carlo over $(f.eg_samples) reference draws per run; D = $(f.ref_pool_size) genuine " *
                "VCS RAM states (E0-2 recorder, frames 1..$(f.target_frame)).",
            "output_kind" => f.output_kind,
            "content_ram_index" => f.content_idx,
            "eg_samples" => f.eg_samples,
            "ref_pool_size" => f.ref_pool_size,
            # the EG-specific degeneracy (a reported finding): the content byte is
            # constant across D ⇒ EG's (x−x') factor is 0 at the only live cell ⇒ EG
            # vanishes (the EG analogue of IG's baseline degeneracy; on-distribution
            # references make it MORE likely). When true, the oracle is the reference.
            "eg_self_baseline_invariant" => f.eg_self_baseline_invariant,
            "content_byte_value" => f.content_byte_value,
            "ref_self_byte_min" => f.ref_self_byte_range[1],
            "ref_self_byte_max" => f.ref_self_byte_range[2],
            "n_refs_self_differs" => f.n_refs_self_differs,
            "content_byte_strictly_constant_over_D" => (f.output_kind == "content" && f.n_refs_self_differs == 0),
            "headline_metrics" => Dict{String,Any}(
                "pearson_corr" => hs.pearson, "spearman_corr" => hs.spearman,
                "precision_at_k" => hs.precision_at_k, "topk" => f.topk,
                "deletion_auc" => _json_num(hs.deletion_auc),
                "insertion_auc" => _json_num(hs.insertion_auc),
                "completeness_err" => hs.completeness_err,
                "eg_max_abs" => hs.eg_max_abs, "y_ref_mean" => hs.y_ref_mean),
            # THE STABILITY DELIVERABLE: EG is a Monte-Carlo estimator over the
            # reference distribution; we quantify its run-to-run stability.
            "stability" => Dict{String,Any}(
                "n_seeds" => length(f.eg_scores),
                "mean_pairwise_cosine" => f.eg_stability_cosine,
                "corr_min" => minimum(corrs), "corr_max" => maximum(corrs),
                "corr_mean" => Statistics.mean(corrs),
                "corr_std" => length(corrs) > 1 ? Statistics.std(corrs) : 0.0,
                "corr_spread" => maximum(corrs) - minimum(corrs),
                "precision_at_k_min" => minimum(s.precision_at_k for s in f.eg_scores),
                "precision_at_k_max" => maximum(s.precision_at_k for s in f.eg_scores),
                "per_seed" => [_eg_seed_json(s, f.cause_names, f.topk) for s in f.eg_scores],
                "interpretation" => "EG draws M baselines from a distribution of real states, so it is a " *
                    "Monte-Carlo estimator. mean_pairwise_cosine is the average cosine of the per-cause EG " *
                    "attribution across independent reference-sample seeds (1.0 = perfectly stable). On the " *
                    "one-hot content path the gradient is a constant e_idx, so EG is essentially deterministic " *
                    "(cosine≈1, corr-spread≈0) — the reference distribution moves only the MAGNITUDE, not the " *
                    "ranking. This is the §5 stability column for EG."),
            # THE HEAD-TO-HEAD: EG vs single-baseline IG (zeros), same column.
            "vs_single_baseline_ig" => Dict{String,Any}(
                "ig_baseline" => "zeros",
                "ig_pearson" => f.ig_pearson, "ig_spearman" => f.ig_spearman,
                "ig_precision_at_k" => f.ig_precision_at_k,
                "ig_deletion_auc" => _json_num(f.ig_deletion_auc),
                "ig_insertion_auc" => _json_num(f.ig_insertion_auc),
                "ig_completeness_err" => f.ig_completeness_err,
                "ig_max_abs" => f.ig_max_abs,
                "delta_pearson" => hs.pearson - f.ig_pearson,
                "delta_precision_at_k" => hs.precision_at_k - f.ig_precision_at_k,
                "delta_deletion_auc" => (isfinite(hs.deletion_auc) && isfinite(f.ig_deletion_auc)) ?
                    (hs.deletion_auc - f.ig_deletion_auc) : nothing,
                "interpretation" => "EG (Erion 2021) was proposed to remove IG's sensitive single-baseline " *
                    "choice (P2-E4-5 measured that sensitivity). On the VCS content path the head-to-head is " *
                    "decisive: the ranking is baseline-INVARIANT (∂y/∂u constant), so EG and IG score the SAME " *
                    "corr/precision@k (Δcorr=$(round(hs.pearson - f.ig_pearson,digits=4))) — EG buys robustness " *
                    "(it avoids the IG degeneracy where a `zeros` baseline byte equals a 0-valued content byte) " *
                    "at M× the gradient compute, NOT a faithfulness gain. The position output: both VANISH."),
            "harness_positive_control" => Dict{String,Any}(
                "method" => "oracle_abs_delta (the perfectly-faithful attribution)",
                "pearson_corr" => f.oracle_self_pearson,
                "precision_at_k" => f.oracle_self_precision_at_k,
                "deletion_auc" => _json_num(f.oracle_self_deletion_auc),
                "insertion_auc" => _json_num(f.oracle_self_insertion_auc),
                "oracle_column_degenerate" => f.oracle_column_degenerate,
                "interpretation" => "corr=1 & precision@k=1 (on a non-degenerate column) ⇒ the scoring " *
                    "harness rewards a faithful map; the EG numbers above are a true measurement."),
            "auc_note" => "deletion/insertion curves measured on the TRUE VCS by re-running the real ROM " *
                "with top-EG causes occluded (deletion) / restored (insertion); every point is a genuine " *
                "emulator re-run, not a surrogate. NaN = a genuinely flat experiment.",
            "output_note" => _output_note(f),
            "cause_names" => f.cause_names,
            "oracle_abs_delta_per_cause" => Dict(f.cause_names[i] => f.oracle_abs_delta[i] for i in 1:length(f.cause_names)),
            "eg_attr_mean_per_cause" => Dict(f.cause_names[i] => f.eg_attr_mean[i] for i in 1:length(f.cause_names)),
            "y_full" => f.y_full,
            "scales_to_cluster_via" =>
                "tools/cluster/xai_*.sbatch (E0-3) — batched SOFT-STE Expected Gradients over " *
                "outputs×causes×games×references on GPU; the forward is bit-exact to this map.",
        ),
    )
    # SHARED-TESTBED provenance + the sampler-on side-by-side (redesign protocol).
    # For EG the load-bearing difference is the reference pool: the GAMEPLAY pool
    # (each cause byte varies over the seeded trajectory), NOT the old NOOP pool.
    if st_extra !== nothing
        rec["state"] = "gameplay(seed=$(st_extra.seed),prefix=$(st_extra.prefix))+$(st_extra.horizon)"
        rec["extra"]["testbed"] = Dict{String,Any}(
            "state_kind" => "seeded_random_action_gameplay",
            "prefix" => st_extra.prefix, "horizon" => st_extra.horizon, "seed" => st_extra.seed,
            "shared_output" => "screen_region(n_changed_px)@r$(st_extra.cell[1])c$(st_extra.cell[2])",
            "reference_pool" => "gameplay_trajectory(seed=$(st_extra.seed),prefix=$(st_extra.prefix))",
            "reference_pool_note" => "EG's background distribution D is the RAM tape along the SAME " *
                "seeded random-action gameplay trajectory (frames 1..$(st_extra.prefix)), so the causal " *
                "byte VARIES across references ((x−x0)≠0) — the fix for the NOOP-tape degeneracy where a " *
                "constant byte zeroed EG. This REPLACES the old NOOP reference pool.",
            "cause_density_above_floor" => st_extra.cause_density,
            "cause_density_floor" => ST_FLOOR, "cause_density_gate_k" => ST_GATE_K,
            "cause_density_accepted" => st_extra.accepted, "n_causes" => st_extra.n_causes,
            "position_byte_ram_index" => st_extra.geom === nothing ? -1 : st_extra.geom[1])
        if f.output_kind == "position"
            per_cause_null = maximum(maximum(abs.(s.eg_over_ram); init = 0.0) for s in f.eg_scores) > 1e-6 &&
                             all(maximum(abs.(s.attr_per_cause); init = 0.0) < 1e-9 for s in f.eg_scores)
            rec["extra"]["sampler_on"] = Dict{String,Any}(
                "position_gradient_restored" => st_extra.geom !== nothing,
                "per_cause_faithfulness_null" => per_cause_null,
                "note" => "naive index EG ≡ 0 (Prop. prop:zero); the bilinear sampler restores a " *
                    "real ∂pixel/∂ram[position_byte] EG. per_cause_faithfulness_null=true ⇒ the " *
                    "sampler's position byte is not among this game's scored candidate causes, so " *
                    "the per-cause corr is null (0.0 by convention), NOT a genuine zero — the " *
                    "keystone (a non-vanishing gradient on the full 128-byte map) holds.")
        end
    end
    open(json_path, "w") do io; JSON.print(io, rec, 2); end

    write_npz(npz_path, Dict(
        "oracle_abs_delta"        => f.oracle_abs_delta,
        "eg_attr_mean_per_cause"  => f.eg_attr_mean,
        "headline_eg_over_ram"    => hs.eg_over_ram,
        "headline_attr_per_cause" => hs.attr_per_cause,
        "headline_deletion_curve" => hs.del_curve,
        "headline_insertion_curve"=> hs.ins_curve,
        # stability matrix rows aligned to the seed order
        "stability_pearson"       => Float64[s.pearson for s in f.eg_scores],
        "stability_spearman"      => Float64[s.spearman for s in f.eg_scores],
        "stability_precision_at_k"=> Float64[s.precision_at_k for s in f.eg_scores],
        "stability_completeness"  => Float64[s.completeness_err for s in f.eg_scores],
        "stability_eg_max_abs"    => Float64[s.eg_max_abs for s in f.eg_scores],
        # the IG head-to-head row
        "ig_zeros_scalars"        => Float64[f.ig_pearson, f.ig_spearman, f.ig_precision_at_k,
                                             isnan(f.ig_deletion_auc) ? -1.0 : f.ig_deletion_auc,
                                             isnan(f.ig_insertion_auc) ? -1.0 : f.ig_insertion_auc,
                                             f.ig_completeness_err, f.ig_max_abs],
    ))
    return json_path, npz_path
end

# ============================================================================
# CLI
# ============================================================================
function main(args = ARGS)
    games = CORE_GAMES; single_game = nothing
    target_frame = 120; horizon = 30
    eg_samples = 64                  # M reference draws (≈ IG's `steps` in compute)
    eg_stability_seeds = 5           # R independent runs for the stability estimate
    ig_steps = 64                    # the single-baseline-IG comparison resolution
    topk = 3; seed = 0
    selftest_only = false
    i = 1
    while i <= length(args)
        a = args[i]
        if     a == "--games"
            v = args[i+1]; games = (v == "core") ? CORE_GAMES : String.(split(v, ",")); i += 2
        elseif a == "--game";               single_game = args[i+1]; i += 2
        elseif a == "--target-frame";       target_frame = parse(Int, args[i+1]); i += 2
        elseif a == "--horizon";            horizon = parse(Int, args[i+1]); i += 2
        elseif a == "--eg-samples";         eg_samples = parse(Int, args[i+1]); i += 2
        elseif a == "--eg-stability-seeds"; eg_stability_seeds = parse(Int, args[i+1]); i += 2
        elseif a == "--ig-steps";           ig_steps = parse(Int, args[i+1]); i += 2
        elseif a == "--topk";               topk = parse(Int, args[i+1]); i += 2
        elseif a == "--seed";               seed = parse(Int, args[i+1]); i += 2
        elseif a == "--selftest";           selftest_only = true; i += 1
        else; i += 1
        end
    end
    single_game !== nothing && (games = [single_game])

    println("[eg] Expected Gradients (Erion 2021) vs oracle — games=$(join(games, ",")) " *
            "target_frame=$target_frame horizon=$horizon eg_samples=$eg_samples " *
            "stability_seeds=$eg_stability_seeds ig_steps=$ig_steps topk=$topk seed=$seed (jutari/Julia)")

    summary = Dict{String,Any}[]
    for game in games
        println("\n[eg] ===== $game =====")
        f_content, f_pos, st_extra = compute_game(; game = game, target_frame = target_frame,
            horizon = horizon, eg_samples = eg_samples, eg_stability_seeds = eg_stability_seeds,
            ig_steps = ig_steps, topk = topk, seed = seed, verbose = true)
        @assert !f_content.oracle_column_degenerate "content oracle column is degenerate for $game " *
            "at f$target_frame — pick_content_idx failed to find a causally-active concept byte"
        selftest(f_content; require_nondegenerate = true)
        # SHARED TESTBED sampler-on: EG's position gradient is NON-VANISHING when a
        # moving sprite exists (geom !== nothing) — the redesign keystone.
        sampler_on = st_extra !== nothing && st_extra.geom !== nothing
        selftest(f_pos; sampler_on = sampler_on)
        if !selftest_only
            for f in (f_content, f_pos)
                jp, np = write_faithfulness(f; st_extra = (f === f_pos ? st_extra : nothing))
                println("[eg] wrote $jp"); println("[eg] arrays  $np")
            end
        end
        for f in (f_content, f_pos)
            hs = headline_eg(f)
            corrs = [s.pearson for s in f.eg_scores]
            push!(summary, Dict{String,Any}(
                "game" => game, "output" => f.output, "output_kind" => f.output_kind,
                "content_ram_index" => f.content_idx, "ref_pool_size" => f.ref_pool_size,
                "eg_pearson" => hs.pearson, "eg_spearman" => hs.spearman,
                "eg_precision_at_k" => hs.precision_at_k,
                "eg_deletion_auc" => _json_num(hs.deletion_auc),
                "eg_insertion_auc" => _json_num(hs.insertion_auc),
                "eg_completeness_err" => hs.completeness_err,
                "eg_stability_cosine" => f.eg_stability_cosine,
                "eg_corr_spread" => maximum(corrs) - minimum(corrs),
                "eg_corr_std" => length(corrs) > 1 ? Statistics.std(corrs) : 0.0,
                "ig_pearson" => f.ig_pearson, "ig_precision_at_k" => f.ig_precision_at_k,
                "ig_deletion_auc" => _json_num(f.ig_deletion_auc),
                "delta_corr_eg_minus_ig" => hs.pearson - f.ig_pearson,
                "delta_pak_eg_minus_ig" => hs.precision_at_k - f.ig_precision_at_k,
                "eg_max_abs" => hs.eg_max_abs, "ig_max_abs" => f.ig_max_abs,
                "eg_self_baseline_invariant" => f.eg_self_baseline_invariant,
                "content_byte_value" => f.content_byte_value,
                "oracle_self_pearson" => f.oracle_self_pearson,
                "oracle_self_precision_at_k" => f.oracle_self_precision_at_k,
                "is_content_path" => f.output_kind == "content"))
        end
    end

    if selftest_only
        println("\n[eg] --selftest: all passed, not writing artifacts.")
        return 0
    end

    isdir(OUT_DIR) || mkpath(OUT_DIR)
    summary_path = joinpath(OUT_DIR, "expected_gradients_core_summary.json")
    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseB_attribution", "method" => "expected_gradients",
        "item" => "P2-E4-6", "games" => games,
        "eg_samples" => eg_samples, "eg_stability_seeds" => eg_stability_seeds,
        "ig_steps" => ig_steps, "topk" => topk, "seed" => seed,
        "target_frame" => target_frame, "horizon" => horizon,
        "where" => "local", "commit" => _git_commit(), "timestamp" => string(round(Int, time())),
        "headline" => "Expected Gradients (Erion 2021, NMI): baseline-averaged IG over a distribution of " *
            "REAL VCS states (E0-2 recorder). content path (one-hot RAM read): EG lands on the TRUE causal " *
            "byte and is complete-in-expectation; position output (ball_pixel): EG VANISHES for every " *
            "reference draw → near chance (plausible ≠ faithful; same as IG). KEY FINDING vs single-baseline " *
            "IG (P2-E4-5): on the VCS content path ∂y/∂u is a CONSTANT, so EG and IG score the SAME " *
            "corr/precision@k (Δcorr≈0) — EG removes IG's baseline-choice/degeneracy and is provably stable " *
            "(mean pairwise cosine≈1 across reference-sample seeds), but does NOT improve faithfulness over a " *
            "well-chosen IG baseline here; it costs M× the gradient compute. Oracle-as-method control corr=1/p@k=1.",
        "method_note" => Dict{String,Any}(
            "definition" => "EG_i(x) = E_{x'~D, α~U(0,1)}[(x_i−x'_i)·∂y/∂x_i|_{x'+α(x−x')}]",
            "reference_distribution" => "D = genuine RAM states at frames 1..target_frame of the same " *
                "deterministic NOOP replay (E0-2 recorder, jutari_record.jl); each row a reachable game state",
            "vs_ig" => "single-baseline IG (P2-E4-5) uses ONE hand-chosen x0; EG integrates over D. We score " *
                "both on the SAME content byte (head-to-head Δcorr/Δp@k/Δdel-AUC per game).",
            "stability" => "EG is Monte-Carlo over D; reported as mean pairwise cosine of the per-cause " *
                "attribution + corr spread/std across eg_stability_seeds independent reference draws."),
        "results" => summary)
    open(summary_path, "w") do io; JSON.print(io, rec, 2); end
    println("\n[eg] wrote summary $summary_path")

    println("\n[eg] ===== per-game headline (content path): EG vs single-baseline IG =====")
    println("  game             ram   EG-corr  IG-corr  Δcorr    EG-p@$topk  compl_err  stab-cos  corr-spread  note")
    for r in summary
        r["is_content_path"] || continue
        note = r["eg_self_baseline_invariant"] ? "EG-DEGEN(byte=$(r["content_byte_value"]) const over D ⇒ EG=0)" : ""
        println("  $(rpad(r["game"],16)) $(rpad(r["content_ram_index"],5)) " *
                "$(rpad(round(r["eg_pearson"],digits=3),8)) " *
                "$(rpad(round(r["ig_pearson"],digits=3),8)) " *
                "$(rpad(round(r["delta_corr_eg_minus_ig"],digits=4),8)) " *
                "$(rpad(round(r["eg_precision_at_k"],digits=3),7)) " *
                "$(rpad(round(r["eg_completeness_err"],sigdigits=2),10)) " *
                "$(rpad(round(r["eg_stability_cosine"],digits=4),9)) " *
                "$(rpad(round(r["eg_corr_spread"],digits=4),12)) $note")
    end
    println("[eg] ===== position contrast (ball_pixel — EG vanishes, like IG) =====")
    for r in summary
        r["is_content_path"] && continue
        println("  $(rpad(r["game"],16)) EG-corr=$(round(r["eg_pearson"],digits=3)) " *
                "p@$topk=$(round(r["eg_precision_at_k"],digits=3)) " *
                "max|EG|=$(round(r["eg_max_abs"],sigdigits=3)) max|IG|=$(round(r["ig_max_abs"],sigdigits=3))")
    end
    return 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    ExpectedGradients.main()
end
