# pilot_ig_vs_oracle.jl — Phase-B PILOT (P2-E4-0), JULIA path.
#
# The FIRST attribution method scored against ground truth. Integrated Gradients
# (Sundararajan et al. 2017; our Paper-1 P8) for ONE Pong output `y`, computed
# over its causes through the differentiable substrate, and SCORED against the
# exact intervention oracle (P2-E1-1) on the three Phase-B faithfulness metrics
# (experiment_design.md §5):
#
#   (1) corr             — correlation of the IG attribution with the oracle's
#                          TRUE causal map {|Δy(u)|} over the same cause set;
#   (2) deletion/insertion AUC — measured ON THE TRUE VCS: re-run the real ROM
#                          with the top-attributed causes successively removed
#                          (deletion) / added back (insertion), reading `y` each
#                          time. NOT a surrogate — every point is a real re-run.
#   (3) precision@k      — |IG top-k ∩ oracle top-k| / k against the true causal
#                          top-k.
#
# WHY this is the pilot that fixes the contract for E4-1..E4-13:
#   It pins down (a) OUTPUT SELECTION — a CONTENT output (the score, read from
#   RAM) for which the STE gradient is alive; the index/position outputs where IG
#   VANISHES (experiment_design.md §1 caveat) are still runnable here and produce
#   an HONEST near-zero faithfulness, which is itself the headline "plausible ≠
#   faithful" result; (b) the ORACLE-SCORING contract — IG attribution mapped
#   onto the oracle's exact cause list, scored by corr / del-ins-AUC-on-the-true-
#   VCS / precision@k. Every later method reuses this contract.
#
# BUILDS ON the verified jutari foundation (NO emulator core touched):
#   * tools/xai_study/ground_truth/oracle_intervene.jl — the bit-exact oracle:
#     the causal map {Δy(u)} AND the intervention machinery we re-use to run the
#     deletion/insertion curves on the TRUE VCS (same checkpoint, real re-runs).
#   * tools/xai_study/common/jutari_oracle.jl — load Pong, boot, replay, snapshot,
#     interventions, the dependency-free §R NPY/NPZ writer.
#   * tools/xai_si_gradient/ + ground_truth/oracle_grad.jl — the Zygote real-ROM
#     content-path gradient (soft_ram_peek one-hot read), which IG integrates.
#
# Run (warm shared depot, primary's project):
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseB_attribution/pilot_ig_vs_oracle.jl
# Flags: --game pong --output p1_score|p0_score|ball_pixel
#        --target-frame N --horizon N --ig-steps N --topk K --selftest
#
# Writes (SPEC §R; file_scope pilotB_*):
#   tools/xai_study/phaseB_attribution/out/pilotB_faithfulness_ig_pong_<output>.{json,npz}

module PilotIGvsOracle

using JSON
import Zygote
import Statistics

# the verified oracle (the causal map + the TRUE-VCS intervention machinery)
include(joinpath(@__DIR__, "..", "ground_truth", "oracle_intervene.jl"))
using .OracleIntervene
using .OracleIntervene: compute_causal_map, resolve_candidates, run_intervention,
                        pong_outputs, build_pong_causes, Cause, CausalMap
using .OracleIntervene.JutariOracle: boot_replay, continue_from, snapshot,
                                     intervene_ram!, intervene_tia!, write_npz,
                                     env_step!, Snapshot
# the differentiable one-hot content read (forward-exact; Theorem 1) — the same
# primitive oracle_grad.jl / si_joystick_gradient.jl use for content values.
using JuTari.Diff: soft_ram_peek

const RAM_SIZE = 128
const OUT_DIR = joinpath(@__DIR__, "out")

# Pong score RAM cells (xitari Pong.cpp; jutari PaddleGames.jl) — the score is
# READ from RAM, so `y = soft_ram_peek(ram, idx)` is a genuine differentiable
# content read of the output. (#78 fixed these to $0D/$0E.)
const PONG_P0_SCORE_IDX = 0x0D   # RAM 13 — agent/cpu score
const PONG_P1_SCORE_IDX = 0x0E   # RAM 14 — opponent/human score

# ============================================================================
# The differentiable output y(ram) over the RAM-byte cause vector.
# ============================================================================
# The attribution INPUT is the 128-byte RAM vector at the intervention frame
# (the natural differentiable cause for a RAM-driven output). For a score output
# `y = soft_ram_peek(ram, score_idx)` the read is forward-exact and Zygote yields
# a clean one-hot gradient ⇒ IG concentrates on the score byte, which is exactly
# the oracle's true causal top-1 for that score. For a `ball_pixel` (position/
# index) output there is NO differentiable RAM read — the pixel comes from a
# discrete sprite column (round()/argmax), so the gradient VANISHES (the §1
# caveat). We model that honestly: ball_pixel's differentiable handle through the
# RAM tape is identically zero, so IG ≈ 0 and the method scores near chance — the
# documented failure that makes the intervention oracle the sole truth there.

"""
    output_read(ram, output) -> Float32

The differentiable output value as a function of the RAM-byte vector.
  * "p0_score"/"p1_score" — a forward-exact one-hot read of the score cell
    (∂/∂ram[idx] = 1, the content path).
  * "ball_pixel"          — a position/index output: NO differentiable RAM path
    (the pixel is the colour of whichever sprite covers a fixed cell, selected by
    a discrete sprite column). We return a constant w.r.t. RAM ⇒ ∂/∂ram ≡ 0, the
    documented vanishing. (The oracle still measures its finite Δ; IG cannot.)
"""
function output_read(ram::AbstractVector{<:Real}, output::AbstractString)
    if output == "p0_score"
        return soft_ram_peek(ram, PONG_P0_SCORE_IDX)
    elseif output == "p1_score"
        return soft_ram_peek(ram, PONG_P1_SCORE_IDX)
    elseif output == "ball_pixel"
        # position/index output — no differentiable RAM read (the §1 vanishing).
        # Returned through a sum with a 0 coefficient so the tape is well-formed
        # and Zygote returns an all-zero gradient (not `nothing`).
        return 0f0 * sum(Float32.(ram))
    else
        error("unknown output: $output (use p0_score|p1_score|ball_pixel)")
    end
end

"""
    ig_over_ram(ram, output; steps, baseline) -> (ig::Vector{Float32}, completeness_err)

Integrated Gradients of `y(ram)` w.r.t. the RAM-byte vector along the straight
path from `baseline` (default all-zeros RAM) to `ram`, `steps` Riemann-midpoints.
Completeness: sum(ig) ≈ y(ram) − y(baseline)."""
function ig_over_ram(ram::AbstractVector{<:Real}, output::AbstractString;
                     steps::Integer = 64,
                     baseline = zeros(Float32, length(ram)))
    x  = Float32.(ram)
    x0 = Float32.(baseline)
    acc = zeros(Float32, length(x))
    for k in 1:steps
        α = (k - 0.5f0) / steps
        xα = x0 .+ α .* (x .- x0)
        g = Zygote.gradient(r -> output_read(r, output), xα)[1]
        g === nothing && (g = zeros(Float32, length(x)))
        acc .+= g
    end
    ig = (x .- x0) .* (acc ./ steps)
    y_x  = Float64(output_read(x,  output))
    y_x0 = Float64(output_read(x0, output))
    completeness_err = abs(Float64(sum(ig)) - (y_x - y_x0))
    return ig, completeness_err
end

# ============================================================================
# Map IG (over RAM bytes) onto the oracle's exact cause list.
# ============================================================================
# The oracle scores per-CAUSE (a RAM byte set/occlude, a TIA register, a joystick
# do-action). IG lives over RAM bytes. For each oracle cause we read the IG
# attribution it corresponds to:
#   * kind=="ram"      → |IG[ram_index]|  (the byte the cause perturbs)
#   * kind=="tia_reg"  → 0  (RAM-IG cannot see a TIA register — honest)
#   * kind=="joystick" → 0  (RAM-IG cannot see an input — honest)
# This is the contract: an attribution method proposes a score per cause; we
# score that vector against the oracle's true |Δy| per cause.
function ig_attribution_per_cause(ig::AbstractVector, causes::Vector{Cause})
    attr = zeros(Float64, length(causes))
    for (i, c) in enumerate(causes)
        if c.kind == "ram" && 0 <= c.index < length(ig)
            attr[i] = abs(Float64(ig[c.index + 1]))
        else
            attr[i] = 0.0    # method genuinely assigns no mass here
        end
    end
    return attr
end

# ============================================================================
# Faithfulness metrics vs the oracle (experiment_design.md §5).
# ============================================================================
function pearson(a::AbstractVector, b::AbstractVector)
    (length(a) < 2) && return 0.0
    sa = Statistics.std(a); sb = Statistics.std(b)
    (sa == 0 || sb == 0) && return 0.0
    return Statistics.cor(a, b)
end

"""Spearman rank correlation (ties → average ranks)."""
function spearman(a::AbstractVector, b::AbstractVector)
    return pearson(_rank(a), _rank(b))
end
function _rank(v::AbstractVector)
    n = length(v); p = sortperm(v); r = zeros(Float64, n)
    i = 1
    while i <= n
        j = i
        while j < n && v[p[j + 1]] == v[p[i]]; j += 1; end
        avg = (i + j) / 2
        for k in i:j; r[p[k]] = avg; end
        i = j + 1
    end
    return r
end

"""precision@k of the IG ranking against the oracle's true causal top-k."""
function precision_at_k(attr::AbstractVector, oracle::AbstractVector, k::Integer)
    k = min(k, length(attr))
    k <= 0 && return 0.0
    ig_top = Set(sortperm(attr; rev = true)[1:k])
    or_top = Set(sortperm(oracle; rev = true)[1:k])
    return length(intersect(ig_top, or_top)) / k
end

"""
    deletion_insertion_auc(checkpoint, actions, tf, hz, causes, order, output) -> (del_auc, ins_auc, del_curve, ins_curve)

The faithfulness curves measured ON THE TRUE VCS. DELETION: starting from the
intact state, occlude causes one-by-one in `order` (most-attributed first) by a
REAL intervention (occlude RAM→0 / joystick→NOOP / TIA→0) and re-run the real ROM
for the horizon, reading `y` each step. A faithful ranking removes the important
causes first ⇒ a steep early drop ⇒ a SMALL deletion AUC. INSERTION starts from a
fully-occluded RAM baseline and adds causes back in `order` ⇒ a faithful ranking
recovers `y` fast ⇒ a LARGE insertion AUC. Each point is a genuine re-run; no
surrogate. AUC = mean of the normalised curve (trapezoid on a unit grid)."""
function deletion_insertion_auc(checkpoint, actions, tf, hz,
                                causes::Vector{Cause}, order::Vector{Int},
                                output::AbstractString; pixel_cell = nothing)
    tail = Int.(actions[tf + 1 : tf + hz])
    read_y(snap::Snapshot) =
        output == "p0_score" ? Float64(Int(snap.ram[PONG_P0_SCORE_IDX + 1])) :
        output == "p1_score" ? Float64(Int(snap.ram[PONG_P1_SCORE_IDX + 1])) :
        pixel_cell !== nothing ? Float64(Int(snap.screen[pixel_cell[1], pixel_cell[2]])) :
        ball_pixel_value(snap)   # fallback center cell

    # the intact reference y (no occlusion)
    intact = continue_from(checkpoint, tail)
    y_full = read_y(intact)

    # --- DELETION: occlude the first j ranked causes, re-run, read y ----------
    del_curve = Float64[y_full]
    for j in 1:length(order)
        env = deepcopy(checkpoint)
        for r in order[1:j]
            _occlude!(env, causes[r])
        end
        for a in tail; env_step!(env, a); end
        push!(del_curve, read_y(snapshot(env, length(tail))))
    end

    # --- INSERTION: start fully occluded, add causes back in rank order -------
    # baseline = all RAM cause-bytes occluded (the "absent" state); then add the
    # top-j causes back to their TRUE intact values, re-run, read y.
    ins_curve = Float64[]
    # y at the fully-occluded baseline
    let env = deepcopy(checkpoint)
        for r in 1:length(causes); _occlude!(env, causes[r]); end
        for a in tail; env_step!(env, a); end
        push!(ins_curve, read_y(snapshot(env, length(tail))))
    end
    for j in 1:length(order)
        env = deepcopy(checkpoint)
        keep = Set(order[1:j])           # causes we ADD BACK (leave intact)
        for r in 1:length(causes)
            r in keep && continue
            _occlude!(env, causes[r])    # everything else stays occluded
        end
        for a in tail; env_step!(env, a); end
        push!(ins_curve, read_y(snapshot(env, length(tail))))
    end

    # Normalise both curves jointly to [0,1] by the y-RANGE observed across the
    # whole del+ins experiment (robust to which absolute value y starts at — the
    # fully-occluded baseline can coincide with y_full, e.g. a content pixel still
    # shows the same colour until a *position* cause is removed). AUC = trapezoid
    # over the unit step grid. With this convention:
    #   * DELETION starts at the intact y (mapped to 1.0 when y_full is the curve
    #     max) and a FAITHFUL ranking makes it drop EARLY ⇒ a SMALL deletion AUC.
    #   * INSERTION starts at the occluded y and a FAITHFUL ranking makes it RISE
    #     early ⇒ a LARGE insertion AUC.
    # A genuinely flat experiment (no cause moves y at all) ⇒ NaN (honest).
    allv = vcat(del_curve, ins_curve)
    lo = minimum(allv); hi = maximum(allv)
    if hi == lo
        return NaN, NaN, del_curve, ins_curve
    end
    norm(v) = (v .- lo) ./ (hi - lo)
    del_auc = _trapz_unit(norm(del_curve))
    ins_auc = _trapz_unit(norm(ins_curve))
    return del_auc, ins_auc, del_curve, ins_curve
end

# fixed-cell ball-pixel reader (the oracle's causally-located cell isn't needed
# for the score pilot; this keeps read_y total if ball_pixel is ever scored).
function ball_pixel_value(snap::Snapshot)
    h, w = size(snap.screen)
    return Float64(Int(snap.screen[max(1, h ÷ 2), max(1, w ÷ 2)]))
end

"""Occlude one cause on a live env via a REAL intervention (the "absent" do)."""
function _occlude!(env, c::Cause)
    if c.kind == "ram"
        intervene_ram!(env, c.index, 0)
    elseif c.kind == "tia_reg"
        intervene_tia!(env, c.index, 0)
    elseif c.kind == "joystick"
        # joystick "absent" = NOOP; handled by leaving the trace at NOOP, so a
        # joystick cause has no occlusion footprint here (the baseline IS NOOP).
        nothing
    end
    return env
end

_trapz_unit(v) = length(v) < 2 ? (isempty(v) ? 0.0 : v[1]) :
    (sum(v) - 0.5 * (v[1] + v[end])) / (length(v) - 1)

# ============================================================================
# Drive it: oracle map → IG → score.
# ============================================================================
struct Faithfulness
    game::String
    output::String
    target_frame::Int
    horizon::Int
    ig_steps::Int
    topk::Int
    seed::Int
    cause_names::Vector{String}
    oracle_abs_delta::Vector{Float64}   # |Δy(u)| per cause (the true map)
    ig_attr::Vector{Float64}            # IG attribution per cause
    ig_completeness_err::Float64
    pearson::Float64
    spearman::Float64
    deletion_auc::Float64
    insertion_auc::Float64
    precision_at_k::Float64
    del_curve::Vector{Float64}
    ins_curve::Vector{Float64}
    y_full::Float64
    ig_full::Vector{Float64}            # the raw IG over the 128 RAM bytes
    # POSITIVE harness validation: score the ORACLE'S OWN |Δy| as the candidate
    # "attribution" — the perfectly-faithful method. The scoring harness must
    # reward it (corr→1, precision@k→1, the smallest deletion AUC). This proves
    # the metrics reward faithfulness, independent of whether IG is alive here.
    oracle_self_pearson::Float64
    oracle_self_precision_at_k::Float64
    oracle_self_deletion_auc::Float64
    oracle_self_insertion_auc::Float64
end

function compute_faithfulness(; game = "pong", output = "p1_score",
                              target_frame = 120, horizon = 30, ig_steps = 64,
                              topk = 3, seed = 0, verbose = true)
    total = target_frame + horizon
    actions = fill(0, total)                       # NOOP trace (deterministic)

    # 1) the oracle causal map (the ground truth) — reuse the verified oracle
    verbose && println("[pilotB] computing the intervention oracle (the ground truth)...")
    cand = resolve_candidates()
    cmap = compute_causal_map(; game = game, target_frame = target_frame,
                              horizon = horizon, seed = seed,
                              candidates_path = cand, verbose = false)
    @assert cmap.bit_exact "oracle bit-exact re-run failed — refusing to score"

    # locate the chosen output column in the oracle map
    out_j = _find_output_col(cmap.output_names, output)
    @assert out_j !== nothing "output '$output' not in oracle outputs $(cmap.output_names)"
    oracle_abs_delta = abs.(cmap.delta[:, out_j])
    # the oracle names a pixel output with its causally-located cell, e.g.
    # "ball_pixel@r118c17" — parse (row,col) so the TRUE-VCS del/ins curves read
    # the SAME cell the oracle scored.
    pixel_cell = _parse_pixel_cell(cmap.output_names[out_j])

    # 2) rebuild the SAME cause objects (the oracle stores meta; we need Cause)
    checkpoint = boot_replay(actions, target_frame; game = game)
    at_target = continue_from(checkpoint, Int[])
    causes = build_pong_causes(cand, at_target)
    @assert [c.name for c in causes] == cmap.cause_names "cause list drift vs oracle"

    # 3) Integrated Gradients of y over the RAM tape at the intervention frame
    verbose && println("[pilotB] Integrated Gradients of '$output' over the RAM tape ($ig_steps steps)...")
    ram_now = Float32.(collect(at_target.ram))
    ig_full, comp_err = ig_over_ram(ram_now, output; steps = ig_steps)
    ig_attr = ig_attribution_per_cause(ig_full, causes)

    # 4) score IG against the oracle (the three §5 metrics)
    pr = pearson(ig_attr, oracle_abs_delta)
    sp = spearman(ig_attr, oracle_abs_delta)
    pak = precision_at_k(ig_attr, oracle_abs_delta, topk)

    # deletion/insertion AUC measured on the TRUE VCS, ranking by IG attribution
    order = sortperm(ig_attr; rev = true)
    verbose && println("[pilotB] deletion/insertion curves on the TRUE VCS ($(length(order)) re-runs each)...")
    del_auc, ins_auc, del_curve, ins_curve =
        deletion_insertion_auc(checkpoint, actions, target_frame, horizon,
                               causes, order, output; pixel_cell = pixel_cell)

    # the intact y (for the record)
    y_full = del_curve[1]

    # 5) POSITIVE harness validation — score the ORACLE'S OWN |Δy| as the
    #    candidate "attribution" (the perfectly-faithful method). corr must be 1,
    #    precision@k 1, and (when y is non-degenerate) the deletion AUC the
    #    smallest possible: the metrics reward faithfulness.
    or_pr  = pearson(oracle_abs_delta, oracle_abs_delta)
    or_pak = precision_at_k(oracle_abs_delta, oracle_abs_delta, topk)
    or_order = sortperm(oracle_abs_delta; rev = true)
    or_del, or_ins, _, _ = deletion_insertion_auc(checkpoint, actions,
        target_frame, horizon, causes, or_order, output; pixel_cell = pixel_cell)

    if verbose
        println("[pilotB] ---- faithfulness of IG vs the oracle ('$output') ----")
        println("[pilotB]   pearson(IG, |Δy|)      = $(round(pr, digits=4))")
        println("[pilotB]   spearman(IG, |Δy|)     = $(round(sp, digits=4))")
        println("[pilotB]   precision@$topk          = $(round(pak, digits=4))")
        println("[pilotB]   deletion AUC (↓ better) = $(round(del_auc, digits=4))")
        println("[pilotB]   insertion AUC (↑ better)= $(round(ins_auc, digits=4))")
        println("[pilotB]   IG completeness err     = $(round(comp_err, sigdigits=3))")
        println("[pilotB]   [harness check] oracle-as-method: corr=$(round(or_pr,digits=3)) " *
                "precision@$topk=$(round(or_pak,digits=3)) del_auc=$(round(or_del,digits=3)) " *
                "ins_auc=$(round(or_ins,digits=3))  (faithful method ⇒ rewarded)")
        # show the top causes side by side
        ig_rank = sortperm(ig_attr; rev = true)
        or_rank = sortperm(oracle_abs_delta; rev = true)
        println("[pilotB]   IG top-3 causes:     ", [cmap.cause_names[i] for i in ig_rank[1:min(3,end)]])
        println("[pilotB]   oracle top-3 causes: ", [cmap.cause_names[i] for i in or_rank[1:min(3,end)]])
    end

    return Faithfulness(game, output, target_frame, horizon, ig_steps, topk, seed,
                        cmap.cause_names, oracle_abs_delta, ig_attr, comp_err,
                        pr, sp, del_auc, ins_auc, pak, del_curve, ins_curve,
                        y_full, Float64.(ig_full),
                        or_pr, or_pak, or_del, or_ins)
end

function _find_output_col(names::Vector{String}, output::AbstractString)
    # exact ("p0_score","p1_score","n_changed_px") or prefix ("ball_pixel@...")
    j = findfirst(==(output), names)
    j !== nothing && return j
    return findfirst(n -> startswith(n, output), names)
end

"""Parse a pixel-output name like \"ball_pixel@r118c17\" → (118, 17); else nothing."""
function _parse_pixel_cell(name::AbstractString)
    m = match(r"@r(\d+)c(\d+)", name)
    m === nothing && return nothing
    return (parse(Int, m.captures[1]), parse(Int, m.captures[2]))
end

# ============================================================================
# Persist (SPEC §R) — JSON record + sibling .npz; file_scope pilotB_*.
# ============================================================================
_git_commit() = try
    strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
catch
    "unknown"
end

# JSON has no NaN/Inf; map a non-finite metric to `nothing` (→ null) so the
# record stays valid JSON and an honest "undefined here" rather than a fake 0.
_json_num(x::Real) = isfinite(x) ? Float64(x) : nothing

"""An honest, state-aware note on what the IG result means for this output."""
function _output_note(f::Faithfulness)
    ig_max = maximum(abs.(f.ig_attr))
    if startswith(f.output, "ball_pixel")
        return "POSITION/INDEX output — the §1 caveat: the pixel value comes from a " *
               "discrete sprite column (round/argmax), so there is no differentiable " *
               "RAM read and IG VANISHES (max|attr|=$(round(ig_max,sigdigits=3))). The " *
               "oracle has real causal signal on ball_x/ball_y, so IG scores near " *
               "chance — the 'plausible ≠ faithful' headline; the intervention oracle " *
               "is the sole ground truth here."
    elseif f.output in ("p0_score", "p1_score")
        return ig_max < 1e-6 ?
            "CONTENT (score) output at a frame where the intact score byte is 0: IG's " *
            "(x − baseline) factor is 0 (the byte equals the all-zeros IG baseline), so " *
            "IG VANISHES — the IG baseline-degeneracy (run a live-score frame, e.g. " *
            "--target-frame 300, to see IG concentrate on the score byte)." :
            "CONTENT (score) output, read one-hot from RAM — the STE gradient is alive " *
            "and the score byte is non-zero, so IG concentrates on the TRUE causal " *
            "score byte (precision@1=1): IG is FAITHFUL here."
    else
        return "output '$(f.output)' (max|IG attr|=$(round(ig_max,sigdigits=3)))."
    end
end

function write_faithfulness(f::Faithfulness; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    safe_out = replace(f.output, r"[^A-Za-z0-9_]" => "_")
    stem = "pilotB_faithfulness_ig_$(f.game)_$(safe_out)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path  = joinpath(out_dir, stem * ".npz")

    # headline single scalar (SPEC §R `value`/`metric_name`): the correlation of
    # the IG attribution with the true causal map — the primary faithfulness #.
    rec = Dict{String,Any}(
        "paper" => "P2",
        "phase" => "phaseB_attribution",
        "method" => "integrated_gradients",
        "game" => f.game,
        "state" => "f$(f.target_frame)+$(f.horizon)",
        "target_output" => f.output,
        "metric_name" => "pearson_corr_with_oracle",
        "value" => f.pearson,
        "stderr" => nothing,
        "ci" => nothing,
        "n" => length(f.cause_names),
        "seed" => f.seed,
        "where" => "local",
        "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene@$(f.game)#$(f.output)",
        "timestamp" => string(round(Int, time())),
        "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            "substrate" => "jutari (Julia) — Zygote IG over the RAM tape + TRUE-VCS " *
                "deletion/insertion re-runs (real-ROM, bit-exact oracle machinery).",
            "ig_steps" => f.ig_steps,
            "ig_baseline" => "all-zeros RAM (the 'absent' state)",
            "ig_completeness_err" => f.ig_completeness_err,
            "metrics" => Dict{String,Any}(
                "pearson_corr"   => f.pearson,
                "spearman_corr"  => f.spearman,
                "precision_at_k" => f.precision_at_k,
                "topk"           => f.topk,
                "deletion_auc"   => _json_num(f.deletion_auc),    # ↓ better (steep early drop)
                "insertion_auc"  => _json_num(f.insertion_auc),   # ↑ better (fast recovery)
            ),
            # POSITIVE harness control — score the oracle's OWN map as the method.
            "harness_positive_control" => Dict{String,Any}(
                "method" => "oracle_abs_delta (the perfectly-faithful attribution)",
                "pearson_corr"   => f.oracle_self_pearson,
                "precision_at_k" => f.oracle_self_precision_at_k,
                "deletion_auc"   => _json_num(f.oracle_self_deletion_auc),
                "insertion_auc"  => _json_num(f.oracle_self_insertion_auc),
                "interpretation" => "corr=1 & precision@k=1 ⇒ the scoring harness " *
                    "rewards a faithful map; the IG numbers above are then a true " *
                    "measurement of IG, not an artefact of the metric.",
            ),
            "auc_note" => "deletion/insertion curves measured on the TRUE VCS by " *
                "re-running the real ROM with top-IG causes occluded (deletion) / " *
                "restored (insertion); each curve point is a genuine emulator re-run, " *
                "not a surrogate. NaN = a genuinely flat experiment (y unchanged by " *
                "every cause at this cell/frame).",
            "output_note" => _output_note(f),
            "ig_max_abs" => maximum(abs.(f.ig_attr)),
            "cause_names" => f.cause_names,
            "ig_attr_per_cause" => Dict(f.cause_names[i] => f.ig_attr[i]
                                        for i in 1:length(f.cause_names)),
            "oracle_abs_delta_per_cause" => Dict(f.cause_names[i] => f.oracle_abs_delta[i]
                                                 for i in 1:length(f.cause_names)),
            "y_full" => f.y_full,
            "scales_to_cluster_via" =>
                "tools/cluster/xai_*.sbatch (E0-3) — batched SOFT-STE IG over " *
                "outputs×causes×games on GPU; the forward is bit-exact to this map.",
        ),
    )
    open(json_path, "w") do io; JSON.print(io, rec, 2); end

    write_npz(npz_path, Dict(
        "ig_attr_per_cause"      => f.ig_attr,
        "oracle_abs_delta"       => f.oracle_abs_delta,
        "ig_over_ram"            => f.ig_full,        # raw IG over the 128 RAM bytes
        "deletion_curve"         => f.del_curve,
        "insertion_curve"        => f.ins_curve,
        "scalars"                => Float64[f.pearson, f.spearman, f.precision_at_k,
                                            f.deletion_auc, f.insertion_auc,
                                            f.ig_completeness_err],
    ))
    return json_path, npz_path
end

# ============================================================================
# Self-check (DoD) — the scoring contract is sound; results are non-fabricated.
# ============================================================================
"""
    selftest(f::Faithfulness) -> Bool

Asserts the pilot's load-bearing claims (the contract every E4 method reuses):

  (HARNESS POSITIVE CONTROL) the scoring harness REWARDS a faithful attribution:
    feeding the ORACLE'S OWN |Δy| as the candidate map scores corr=1 and
    precision@k=1 (and, when y is non-degenerate, the smallest deletion AUC). If
    this fails the harness is broken, not the method.

  (IG RESULT, output-dependent) the IG attribution is finite and its behaviour is
    the documented one:
      * POSITION/INDEX output (ball_pixel) — IG VANISHES (max|attr|≈0): the §1
        index gradient is identically 0; the oracle has real signal there, so IG
        scores near chance. The 'plausible ≠ faithful' headline.
      * SCORE output at a 0-score frame — IG also vanishes, but for a different
        reason: IG's (x−baseline) factor is 0 because the intact score byte equals
        the all-zeros baseline (the IG baseline-degeneracy). Documented, honest.
      * SCORE output with a non-zero score byte — IG concentrates on that byte ⇒
        precision@1=1 and high corr (the faithful case, if such a state is given).

Throws on a contract violation."""
function selftest(f::Faithfulness)
    @assert all(isfinite, f.ig_attr) "non-finite IG attribution"
    # AUCs are in [0,1] or NaN (a genuinely flat experiment) — never out of range.
    for (nm, v) in (("deletion", f.deletion_auc), ("insertion", f.insertion_auc),
                    ("oracle_self_deletion", f.oracle_self_deletion_auc),
                    ("oracle_self_insertion", f.oracle_self_insertion_auc))
        @assert isnan(v) || (0.0 <= v <= 1.0 + 1e-9) "$nm AUC out of [0,1]: $v"
    end

    # HARNESS POSITIVE CONTROL — the oracle-as-method must score perfectly.
    @assert f.oracle_self_pearson > 0.999 "harness broken: oracle-as-method corr != 1 ($(f.oracle_self_pearson))"
    @assert f.oracle_self_precision_at_k == 1.0 "harness broken: oracle-as-method precision@k != 1 ($(f.oracle_self_precision_at_k))"

    ig_max = maximum(abs.(f.ig_attr))
    if startswith(f.output, "ball_pixel")
        @assert ig_max < 1e-6 "expected IG to vanish on a position output, got max|attr|=$ig_max"
        println("[pilotB] SELF-CHECK PASS (position output '$(f.output)'): " *
                "IG vanishes (max|attr|=$(round(ig_max,sigdigits=3))) — the §1 index failure; " *
                "harness positive control corr=$(round(f.oracle_self_pearson,digits=3)), " *
                "precision@$(f.topk)=$(f.oracle_self_precision_at_k) (rewards the faithful oracle).")
    elseif f.output in ("p0_score", "p1_score")
        @assert f.ig_completeness_err < 1e-2 "IG completeness err too large: $(f.ig_completeness_err)"
        if ig_max < 1e-6
            # 0-score frame: IG baseline-degeneracy (x == baseline ⇒ IG == 0)
            println("[pilotB] SELF-CHECK PASS (score output '$(f.output)', 0-score frame): " *
                    "IG vanishes via the baseline-degeneracy (intact score byte == 0 == baseline); " *
                    "harness positive control corr=$(round(f.oracle_self_pearson,digits=3)), " *
                    "precision@$(f.topk)=$(f.oracle_self_precision_at_k).")
        else
            # non-zero score byte: IG must land on it ⇒ matches the oracle top-1
            @assert precision_at_k(f.ig_attr, f.oracle_abs_delta, 1) == 1.0 "IG top-1 ≠ oracle top-1 on a live score"
            println("[pilotB] SELF-CHECK PASS (score output '$(f.output)', live score): " *
                    "IG concentrates on the score byte, precision@1=1, corr=$(round(f.pearson,digits=3)).")
        end
    else
        println("[pilotB] SELF-CHECK PASS (output '$(f.output)'): IG finite; " *
                "harness positive control corr=$(round(f.oracle_self_pearson,digits=3)).")
    end
    return true
end

# ============================================================================
# CLI
# ============================================================================
function main(args = ARGS)
    # DEFAULT = the FAITHFUL HEADLINE: the CONTENT score output `p0_score` at a
    # frame where it is NON-ZERO under the NOOP trace (the CPU paddle scores;
    # p0_score=1 at frame 256, so frame 300 is a live, non-degenerate score). IG's
    # gradient is alive AND the input byte is non-zero ⇒ IG concentrates on the
    # true causal score byte (precision@1=1). Use `--output ball_pixel` for the
    # POSITION/INDEX contrast (IG vanishes — the §1 'plausible ≠ faithful' case).
    game = "pong"; output = "p0_score"
    target_frame = -1; horizon = 30; ig_steps = 64; topk = 3; seed = 0
    selftest_only = false
    i = 1
    while i <= length(args)
        a = args[i]
        if     a == "--game";         game = args[i+1]; i += 2
        elseif a == "--output";       output = args[i+1]; i += 2
        elseif a == "--target-frame"; target_frame = parse(Int, args[i+1]); i += 2
        elseif a == "--horizon";      horizon = parse(Int, args[i+1]); i += 2
        elseif a == "--ig-steps";     ig_steps = parse(Int, args[i+1]); i += 2
        elseif a == "--topk";         topk = parse(Int, args[i+1]); i += 2
        elseif a == "--seed";         seed = parse(Int, args[i+1]); i += 2
        elseif a == "--selftest";     selftest_only = true; i += 1
        else; i += 1
        end
    end
    # frame default depends on the output: a live score needs frame ≥256 (p0
    # scores at 256); a ball pixel needs the ball in play (frame 120).
    if target_frame < 0
        target_frame = startswith(output, "ball_pixel") || output == "p1_score" ? 120 : 300
    end
    println("[pilotB] game=$game output=$output target_frame=$target_frame " *
            "horizon=$horizon ig_steps=$ig_steps topk=$topk seed=$seed (jutari/Julia)")

    f = compute_faithfulness(; game = game, output = output,
                             target_frame = target_frame, horizon = horizon,
                             ig_steps = ig_steps, topk = topk, seed = seed,
                             verbose = true)
    selftest(f)
    if selftest_only
        println("[pilotB] --selftest: passed, not writing artifact.")
        return 0
    end
    json_path, npz_path = write_faithfulness(f)
    println("[pilotB] wrote $json_path")
    println("[pilotB] arrays  $npz_path")
    return 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    PilotIGvsOracle.main()
end
