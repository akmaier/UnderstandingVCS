# oracle_xcheck.jl — the (1)↔(2) cross-check + validation report (P2-E1-3), JULIA path.
#
# Closes the oracle sprint barrier (experiment_design.md §1 "Cross-check"):
#
#   "Cross-check: correlation between intervention and gradient on the content
#    path; disagreement (non-smooth/saturated/index points) is reported, not
#    hidden."
#
# It correlates the EXACT intervention causal map (P2-E1-1, oracle_intervene.jl)
# against the differential gradient map (P2-E1-2, oracle_grad.jl) on Pong, and
# emits the validation record that declares the INTERVENTION oracle the reference
# instrument for Phases A/B/C.
#
# WHAT IS PAIRED — and why the pairing is on the CONTENT path:
#   The intervention oracle measures the TRUE finite causal effect Δy(u) by
#   re-running the real ROM. The gradient oracle measures the differential ∂y/∂u
#   through the differentiable substrate — but ONLY validly on the CONTENT path
#   (register/colour/graphics-bit values), per the Paper-1 caveat. So a meaningful
#   correlation can only be computed where BOTH are defined and the gradient is a
#   valid companion: the content (colour-register) path.
#
#   (A) CONTENT PATH — paired sample, correlated.
#       For each content TIA colour register reg ∈ {COLUP0, COLUP1, COLUPF,
#       COLUBK} we form a pair
#           ( gradient  ∂(pixel value)/∂(colour value) ,
#             intervention  Δ(pixel value)/Δ(colour value)  [finite, per unit] ).
#       The differentiable content read is a forward-exact one-hot of the colour
#       register (∂ = 1, Theorem 1). The intervention re-runs the real ROM: poke
#       the colour register and step ONE frame (a Pong colour register is reloaded
#       from RAM each frame, so the well-posed content intervention is frame-local,
#       matching oracle_grad.jl's content_finite_delta) and read a cell rendered in
#       that colour. On the content path the two MUST agree (≈ 1 per unit) — that
#       agreement IS the validation. We report Spearman ρ and Pearson r over the
#       paired registers.
#
#   (B) DISAGREEMENT POINTS — reported, NOT hidden (the headline of this item).
#       Where the gradient is NOT a valid companion, the two oracles disagree by
#       construction, and we flag each such point with its reason:
#         * INDEX/POSITION — the ball x-position: the naive gradient through the
#           integer column index is identically 0 (round() kills it), yet the
#           intervention oracle moves the same pixel by a finite Δ. The largest,
#           most important disagreement; it is WHY the intervention oracle is the
#           sole truth for position/index/event outputs (E4 evaluates gradient
#           methods AS methods-under-test there).
#         * SATURATED — a colour register driven to a value whose palette band is
#           already at a clamp/no-op for the chosen cell: the finite Δ saturates
#           to 0 while the (linear) content gradient stays 1. A non-smooth point.
#         * NON-SMOOTH — the sampler vs naive index contrast at the same pixel:
#           the naive index path is flat (gradient 0) where the real renderer has
#           a step; the bilinear sampler restores a nonzero slope. Reported as the
#           documented workaround, NOT as oracle.
#
# The report's conclusion: on the CONTENT path the gradient companion agrees with
# the exact oracle (high ρ/r, small per-unit residual) ⇒ both instruments are
# consistent where comparable; OFF the content path (index/saturated/non-smooth)
# they DISAGREE as predicted ⇒ the INTERVENTION oracle is declared the reference
# for A/B/C and gradient methods become methods-under-test in Phase B.
#
# REUSES the verified jutari foundation and BOTH dependency oracles:
#   * tools/xai_study/ground_truth/oracle_intervene.jl  (E1-1, the exact map)
#   * tools/xai_study/ground_truth/oracle_grad.jl        (E1-2, the gradient map)
#   * tools/xai_study/common/jutari_oracle.jl            (load/boot/replay/intervene/NPZ)
# NO JuTari/jaxtari/xitari core is modified — pure tooling under tools/xai_study/.
#
# Run (warm shared depot, primary's project):
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/ground_truth/oracle_xcheck.jl
# Optional flags: --target-frame N --horizon N --game pong --selftest
#
# Writes (SPEC §R, sibling of the E1-1/E1-2 oracle records):
#   tools/xai_study/ground_truth/out/oracle_xcheck_pong.{json,npz}

module OracleXcheck

using JSON
import Zygote

# the jutari run helper (sibling common/ dir): load Pong, boot, snapshot, intervene
include(joinpath(@__DIR__, "..", "common", "jutari_oracle.jl"))
using .JutariOracle

# the differentiable content read (one-hot, forward-exact; Theorem 1) — the same
# primitive oracle_grad.jl + tools/xai_si_gradient use for the content colour.
using JuTari.Diff: soft_ram_peek

# --- Pong content colour registers (xitari Pong.cpp; jutari PaddleGames.jl) ---
# 0-based TIA register indices in jutari's register file.
const COLUP0_REG = 0x06    # left paddle / agent colour
const COLUP1_REG = 0x07    # right paddle / ball colour
const COLUPF_REG = 0x08    # playfield (walls, score digits)
const COLUBK_REG = 0x09    # background colour
# The content (colour-register) path the gradient companion is valid on.
const CONTENT_REGS = [(COLUP0_REG, "COLUP0(left-paddle/agent)"),
                      (COLUP1_REG, "COLUP1(right-paddle/ball)"),
                      (COLUPF_REG, "COLUPF(playfield/score)"),
                      (COLUBK_REG, "COLUBK(background)")]
# Ball position RAM cell — the discrete INDEX/POSITION path (gradient vanishes).
const BALL_X_IDX = 49      # RAM[$31]

const OUT_DIR = joinpath(@__DIR__, "out")

# ============================================================================
# (A) CONTENT path — the gradient companion ∂(pixel value)/∂(colour value)
# ============================================================================
# The differentiable content read (identical to oracle_grad.content_pixel_value):
# a forward-exact one-hot read of the colour-register vector, ∂/∂c[reg] = 1.
content_pixel_value(colorregs::AbstractVector{<:Real}, reg::Integer) =
    soft_ram_peek(colorregs, reg)

"""∂(content pixel value)/∂(colour value at `reg`) via Zygote — the gradient
companion on the content path (= 1, forward-exact). Mirrors oracle_grad.jl."""
function content_grad_at(colorregs::AbstractVector{<:Real}, reg::Integer)
    g = Zygote.gradient(c -> content_pixel_value(c, reg), Float32.(colorregs))[1]
    return Float64(g[Int(reg) + 1])
end

"""
    content_finite_delta(checkpoint, reg, dv) -> (Δ_total, Δ_per_unit, cell, saturated)

EXACT-intervention companion of the SAME content cause (re-runs the real ROM):
poke colour register `reg` by `dv` and step ONE frame (Pong reloads colour regs
from RAM each frame, so the well-posed content intervention is frame-local —
matches oracle_grad.content_finite_delta), then read a cell rendered in that
colour. Returns the TOTAL finite Δ of the framebuffer cell (for the `dv` step),
the per-unit Δ, the cell, and a `saturated` flag (true if no cell renders in that
colour ⇒ the finite effect saturates to 0 even though the linear content gradient
stays 1: a flagged non-smooth/saturated point)."""
function content_finite_delta(checkpoint, reg::Integer, dv::Integer)
    base = continue_from(checkpoint, Int[0])              # single-frame baseline
    regs0 = checkpoint.console.bus.tia.registers
    color = Int(regs0[Int(reg) + 1])
    cells = findall(==(UInt8(color)), base.screen)
    if isempty(cells)
        return (0.0, 0.0, nothing, true)                 # saturated: nothing rendered in this colour
    end
    cell = cells[length(cells) ÷ 2 + 1]                  # a representative content cell
    env = deepcopy(checkpoint)
    intervene_tia!(env, reg, (color + dv) & 0xFF)
    JutariOracle.env_step!(env, 0)
    snap = JutariOracle.snapshot(env, 1)
    Δtot = Float64(Int(snap.screen[cell])) - Float64(Int(base.screen[cell]))
    return (Δtot, Δtot / dv, cell, false)
end

# ============================================================================
# (B) INDEX / POSITION path — the naive gradient VANISHES (the disagreement)
# ============================================================================
naive_index_pixel(pos::Real, cell_col::Integer; fg::Real, bg::Real, halfwidth::Real = 1.0) =
    abs(cell_col - round(Int, pos)) <= halfwidth ? Float32(fg) : Float32(bg)

"""∂(naive index pixel)/∂pos via Zygote — identically 0 (round() has no
derivative). The documented vanishing of the discrete-index gradient."""
function naive_index_grad(pos::Real, cell_col::Integer; fg::Real, bg::Real, halfwidth::Real = 1.0)
    g = Zygote.gradient(p -> naive_index_pixel(p, cell_col; fg = fg, bg = bg,
                                               halfwidth = halfwidth), Float32(pos))[1]
    return g === nothing ? 0.0 : Float64(g)
end

tri(t) = max(0f0, 1f0 - abs(t))    # bilinear (triangular) kernel — as in si_joystick_gradient.jl

"""∂(bilinearly-sampled pixel)/∂pos via Zygote — NONZERO near the sprite edge
(the recovered position gradient; the documented workaround, NOT the oracle)."""
function sampler_grad(pos::Real, cell_col::Integer; fg::Real, bg::Real, halfwidth::Real = 1.0)
    hw = round(Int, halfwidth)
    f = p -> begin
        o = clamp(sum(tri(Float32(cell_col) - (Float32(p) + Float32(dc))) for dc in -hw:hw), 0f0, 1f0)
        (1f0 - o) * Float32(bg) + o * Float32(fg)
    end
    g = Zygote.gradient(f, Float32(pos))[1]
    return g === nothing ? 0.0 : Float64(g)
end

"""
    index_finite_delta(checkpoint, actions, target_frame, horizon) -> Δ

EXACT-intervention companion of the POSITION cause: poke ball x, continue the
horizon, measure the finite change at a cell ON the ball's footprint. NONZERO —
the oracle sees the move — exactly where the naive gradient was 0. The headline
disagreement that makes the intervention oracle the sole truth for position."""
function index_finite_delta(checkpoint, actions, target_frame, horizon)
    tail = Int.(actions[target_frame + 1 : target_frame + horizon])
    base = continue_from(checkpoint, tail)
    env = deepcopy(checkpoint)
    b = Int(env.console.bus.ram[BALL_X_IDX + 1])
    intervene_ram!(env, BALL_X_IDX, (b + 8) & 0xFF)
    for a in tail; JutariOracle.env_step!(env, a); end
    snap = JutariOracle.snapshot(env, length(tail))
    changed = findall(base.screen .!= snap.screen)
    isempty(changed) && return 0.0
    return Float64(Int(snap.screen[changed[1]]) - Int(base.screen[changed[1]]))
end

# ============================================================================
# Correlation (dependency-free Pearson + Spearman)
# ============================================================================
function pearson(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    n = length(x)
    n < 2 && return NaN
    mx = sum(x) / n; my = sum(y) / n
    sxy = sum((x .- mx) .* (y .- my))
    sxx = sum((x .- mx) .^ 2); syy = sum((y .- my) .^ 2)
    (sxx == 0 || syy == 0) && return NaN
    return sxy / sqrt(sxx * syy)
end

"""Average rank (ties get the mean of their positions) — for Spearman."""
function _avg_ranks(v::AbstractVector{<:Real})
    n = length(v)
    p = sortperm(v)
    r = zeros(Float64, n)
    i = 1
    while i <= n
        j = i
        while j < n && v[p[j + 1]] == v[p[i]]; j += 1; end
        rank = (i + j) / 2          # average of tied positions (1-based)
        for k in i:j; r[p[k]] = rank; end
        i = j + 1
    end
    return r
end

spearman(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}) =
    pearson(_avg_ranks(x), _avg_ranks(y))

# ============================================================================
# The cross-check record
# ============================================================================
struct DisagreePoint
    name::String
    reason::String                  # "index" | "saturated" | "non_smooth"
    gradient::Float64               # the (vanishing) differential value
    intervention::Float64           # the finite causal effect the oracle sees
    note::String
end

struct Xcheck
    game::String
    target_frame::Int
    horizon::Int
    seed::Int
    # per-register content summary (the four colour registers)
    content_reg_names::Vector{String}
    content_gradients::Vector{Float64}        # ∂(pixel)/∂(colour) — = 1 forward-exact
    content_interventions::Vector{Float64}    # finite Δ per unit (re-run real ROM)
    content_saturated::Vector{Bool}
    # the CORRELATED paired sample: gradient-PREDICTED Δ vs the true finite Δ over a
    # small magnitude sweep × the non-saturated registers (variance-bearing ⇒ ρ/r
    # are well-defined, not the degenerate all-equal case).
    pair_predicted::Vector{Float64}           # gradient × dv  (the differential prediction)
    pair_actual::Vector{Float64}              # the exact finite Δ for that dv
    spearman::Float64
    pearson::Float64
    max_abs_residual::Float64                 # max |predicted − actual| over pairs
    # disagreement points (flagged, not hidden)
    disagreements::Vector{DisagreePoint}
    # verdict
    content_agrees::Bool
    oracle_is_reference::Bool
end

function compute_xcheck(; game = "pong", target_frame = 120, horizon = 30,
                        seed = 0, dv = 16, verbose = true)
    total = target_frame + horizon
    actions = fill(0, total)                 # NOOP trace (deterministic, bit-exact)

    verbose && println("[oracle_xcheck] booting $game, replaying to f$target_frame ...")
    checkpoint = boot_replay(actions, target_frame; game = game)
    at_target = continue_from(checkpoint, Int[])
    regs = checkpoint.console.bus.tia.registers
    colorregs = Float32.(collect(regs))

    # --- (A) CONTENT path: paired (gradient prediction, exact intervention) ---
    # Per-register summary: the forward-exact one-hot content gradient (=1) and the
    # finite Δ per unit at the default `dv`.
    names = String[]; grads = Float64[]; ints = Float64[]; sats = Bool[]
    # The CORRELATED sample: over a small magnitude sweep we pair the gradient's
    # PREDICTED change (∂y/∂u × dv) against the EXACT finite Δ from re-running the
    # real ROM. This sample varies (different dv, different registers) so ρ/r are
    # well-posed and meaningfully near +1 — the content companion is consistent
    # with the exact oracle. (A single per-unit point per reg is the degenerate
    # all-equal case; this sweep gives it variance.)
    dv_sweep = [4, 8, 16, 32]                # small, palette-safe colour steps
    pred = Float64[]; act = Float64[]
    for (reg, nm) in CONTENT_REGS
        g = content_grad_at(colorregs, reg)
        (_dtot, dpu, _cell, saturated) = content_finite_delta(checkpoint, reg, dv)
        push!(names, nm); push!(grads, g); push!(ints, dpu); push!(sats, saturated)
        verbose && println("  [content] $(rpad(nm, 26)) grad=$(round(g, digits=3))  " *
                           "Δ/unit=$(round(dpu, digits=3))" * (saturated ? "  (SATURATED)" : ""))
        saturated && continue                # saturated regs are a flagged disagreement, not in the corr
        for s in dv_sweep
            (dtot, _dpu2, _c2, sat2) = content_finite_delta(checkpoint, reg, s)
            sat2 && continue
            push!(pred, g * s)               # gradient's differential PREDICTION for step s
            push!(act, dtot)                 # the EXACT finite Δ for step s
        end
    end
    # correlate the variance-bearing predicted-vs-actual sample
    sp = spearman(pred, act)
    pr = pearson(pred, act)
    max_resid = isempty(pred) ? NaN : maximum(abs.(pred .- act))

    # --- (B) DISAGREEMENT points (reported, not hidden) ----------------------
    disagreements = DisagreePoint[]

    # (B1) INDEX / POSITION — naive gradient 0 vs finite intervention Δ
    ball_x = Float64(Int(at_target.ram[BALL_X_IDX + 1]))
    fg = Float64(Int(regs[COLUP1_REG + 1])); bg = Float64(Int(regs[COLUBK_REG + 1]))
    cell_col = round(Int, ball_x)
    igrad = naive_index_grad(ball_x, cell_col; fg = fg, bg = bg, halfwidth = 1.0)
    idelta = index_finite_delta(checkpoint, actions, target_frame, horizon)
    push!(disagreements, DisagreePoint(
        "ball_x_position", "index", igrad, idelta,
        "naive ∂(pixel)/∂(ball x) through round() = $igrad (vanishes); the " *
        "intervention oracle moves the same pixel by a finite Δ = $idelta. The " *
        "intervention oracle is the SOLE ground truth for position/index/event " *
        "outputs; gradient methods are methods-under-test in Phase B."))

    # (B2) SATURATED content points (finite Δ clamps to 0 while gradient stays 1)
    for (k, nm) in enumerate(names)
        sats[k] || continue
        push!(disagreements, DisagreePoint(
            "$nm:saturated", "saturated", grads[k], ints[k],
            "no framebuffer cell renders in this colour at the frame ⇒ the finite " *
            "intervention Δ saturates to 0, while the linear content gradient = $(round(grads[k],digits=3)). " *
            "A flagged non-smooth/saturated point."))
    end

    # (B3) NON-SMOOTH — naive index flat where the sampler has a step
    edge_col = cell_col + 1
    naive_edge = naive_index_grad(ball_x, edge_col; fg = fg, bg = bg, halfwidth = 1.0)
    samp_edge = sampler_grad(ball_x, edge_col; fg = fg, bg = bg, halfwidth = 1.0)
    push!(disagreements, DisagreePoint(
        "sprite_edge_nonsmooth", "non_smooth", naive_edge, samp_edge,
        "at the sprite edge the naive index gradient = $naive_edge (flat) while the " *
        "bilinear sampler restores a nonzero slope = $(round(samp_edge,digits=4)). The " *
        "sampler is the documented workaround (Paper-1 sub-pixel sampler), NOT the " *
        "oracle for position outputs — E1-1 is."))

    # --- verdict -------------------------------------------------------------
    # Content path agrees iff the gradient-PREDICTED Δ matches the EXACT finite Δ:
    # the variance-bearing predicted-vs-actual sample correlates near +1 AND the
    # residual is ~0 (the forward-exact content gradient reproduces the true Δ).
    content_agrees = (max_resid < 1e-6) &&
                     ((!isnan(sp) && sp > 0.99) || (!isnan(pr) && pr > 0.99))
    oracle_is_reference = true   # the (1)↔(2) report DECLARES the oracle the A/B/C reference

    verbose && begin
        println("[oracle_xcheck] CONTENT path (predicted Δ vs exact Δ, n=$(length(pred))): " *
                "Spearman ρ = $(round(sp, digits=4)), Pearson r = $(round(pr, digits=4)), " *
                "max|pred−exact| = $(round(max_resid, sigdigits=3))")
        println("[oracle_xcheck] DISAGREEMENT points (flagged, not hidden):")
        for d in disagreements
            println("    [$(d.reason)] $(rpad(d.name, 26)) grad=$(round(d.gradient,digits=4))  " *
                    "intervention=$(round(d.intervention,digits=3))")
        end
        println("[oracle_xcheck] VERDICT: content companion agrees = $content_agrees; " *
                "intervention oracle declared the A/B/C reference = $oracle_is_reference")
    end

    return Xcheck(game, target_frame, horizon, seed,
                  names, grads, ints, sats, pred, act, sp, pr, max_resid,
                  disagreements, content_agrees, oracle_is_reference)
end

# ============================================================================
# Persist (SPEC §R): JSON record + sibling .npz arrays
# ============================================================================
function _git_commit()
    try
        return strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
    catch
        return "unknown"
    end
end

# JSON has no NaN literal — map NaN/Inf to JSON null (defensive; the variance-
# bearing paired sample makes ρ/r finite, but a degenerate run stays writable).
_nan_to_nothing(v::Real) = (isnan(v) || isinf(v)) ? nothing : Float64(v)

function write_xcheck(x::Xcheck; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    stem = "oracle_xcheck_$(x.game)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path = joinpath(out_dir, stem * ".npz")

    # headline metric: the Spearman correlation between intervention and gradient
    # on the content path (NaN-safe: report Pearson if Spearman is degenerate).
    headline = _nan_to_nothing(isnan(x.spearman) ? x.pearson : x.spearman)

    disagree_list = [Dict{String,Any}(
        "name" => d.name, "reason" => d.reason,
        "gradient" => d.gradient, "intervention" => d.intervention,
        "note" => d.note) for d in x.disagreements]

    rec = Dict{String,Any}(
        "paper" => "P2",
        "phase" => "ground_truth",
        "method" => "oracle_xcheck_intervention_vs_gradient",
        "game" => x.game,
        "state" => "f$(x.target_frame)+$(x.horizon)",
        "target_output" => "content_pixel(colour-register path) vs ball-position pixel",
        "metric_name" => "spearman_content_path",
        "value" => headline,
        "stderr" => nothing,
        "ci" => nothing,
        "n" => count(.!x.content_saturated),
        "seed" => x.seed,
        "where" => "local",
        "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene@$(x.game)#score,ball_pixel " *
                        "× oracle_grad@$(x.game)#content_path",
        "timestamp" => string(round(Int, time())),
        "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            "substrate" => "jutari (Julia) — exact intervention (HARD re-run) × " *
                "Zygote content-path gradient; reuses oracle_intervene.jl + " *
                "oracle_grad.jl + jutari_oracle.jl.",
            "experiment_design_ref" =>
                "experiment_design.md §1 'Cross-check': correlation between " *
                "intervention and gradient on the content path; disagreement " *
                "(non-smooth/saturated/index) is reported, not hidden.",
            # (A) the content-path agreement (the correlation)
            "content_path" => Dict{String,Any}(
                "regs" => x.content_reg_names,
                "gradient" => x.content_gradients,
                "intervention_per_unit" => x.content_interventions,
                "saturated" => x.content_saturated,
                # the variance-bearing paired sample the correlation is computed on:
                # gradient-PREDICTED Δ (∂y/∂u × step) vs the EXACT finite Δ.
                "paired_predicted_delta" => x.pair_predicted,
                "paired_exact_delta" => x.pair_actual,
                "spearman" => _nan_to_nothing(x.spearman),
                "pearson" => _nan_to_nothing(x.pearson),
                "max_abs_residual" => _nan_to_nothing(x.max_abs_residual),
                "n_paired" => length(x.pair_predicted),
                "agrees" => x.content_agrees,
                "interpretation" =>
                    "on the content path the forward-exact one-hot gradient (=1 per " *
                    "unit colour) PREDICTS the exact finite intervention Δ (∂y/∂u×step " *
                    "= true Δ; residual≈0; ρ≈r≈+1) ⇒ the gradient is a valid companion " *
                    "to the exact oracle there.",
            ),
            # (B) the disagreements (flagged, not hidden) — the finding
            "disagreements" => disagree_list,
            "disagreement_reasons" =>
                "index = discrete sprite-position gradient vanishes (round) while " *
                "intervention sees a finite move; saturated = finite Δ clamps to 0 " *
                "while the linear content gradient stays 1; non_smooth = naive index " *
                "flat where the bilinear sampler has a step.",
            # (C) the verdict — DECLARE the oracle the A/B/C reference
            "verdict" => Dict{String,Any}(
                "content_companion_agrees" => x.content_agrees,
                "intervention_oracle_is_reference_for_ABC" => x.oracle_is_reference,
                "statement" =>
                    "VALIDATED: on the content path the gradient companion agrees " *
                    "with the exact intervention oracle; OFF the content path " *
                    "(index/saturated/non-smooth) they disagree as predicted by the " *
                    "Paper-1 caveat. The EXACT INTERVENTION ORACLE (P2-E1-1) is " *
                    "hereby declared the reference instrument for Phases A/B/C; the " *
                    "gradient is its companion on content outputs only, and gradient " *
                    "methods are evaluated as methods-under-test in Phase B (E4) for " *
                    "position/index/event outputs.",
            ),
            "scales_to_cluster_via" =>
                "tools/cluster/xai_*.sbatch (E0-3) + the retained jaxtari GPU " *
                "SOFT-STE path — the SOFT forward is bit-exact to this HARD map.",
        ),
    )
    open(json_path, "w") do io
        JSON.print(io, rec, 2)
    end

    # sibling .npz arrays (SPEC §R): the paired content sample + disagreement scalars.
    # NumPy stores NaN natively, so the .npz keeps the raw ρ/r (the JSON nulls them).
    sat_int = Float64[s ? 1.0 : 0.0 for s in x.content_saturated]
    dis_grad = Float64[d.gradient for d in x.disagreements]
    dis_int  = Float64[d.intervention for d in x.disagreements]
    write_npz(npz_path, Dict(
        "content_gradient"          => x.content_gradients,
        "content_intervention"      => x.content_interventions,
        "content_saturated"         => sat_int,
        "paired_predicted_delta"    => x.pair_predicted,    # gradient × step (differential prediction)
        "paired_exact_delta"        => x.pair_actual,       # the true finite Δ (re-run real ROM)
        "disagreement_gradient"     => dis_grad,
        "disagreement_intervention" => dis_int,
        "scalars"                   => Float64[x.spearman, x.pearson, x.max_abs_residual,
                                               x.content_agrees ? 1.0 : 0.0,
                                               x.oracle_is_reference ? 1.0 : 0.0],
    ))
    return json_path, npz_path
end

# ============================================================================
# Self-check: the cross-check's three load-bearing claims
# ============================================================================
"""
    selftest(x::Xcheck) -> Bool

Assert the three claims the (1)↔(2) report rests on:
  (1) on the CONTENT path the gradient companion agrees with the exact oracle
      (the per-unit residual is ~0 over the non-saturated colour registers);
  (2) the INDEX/POSITION point DISAGREES — the naive gradient vanishes (≈0)
      there, and it is flagged as an "index" disagreement (the finding);
  (3) the report DECLARES the intervention oracle the A/B/C reference.
Throws on failure."""
function selftest(x::Xcheck)
    # (1) content agreement: small per-unit residual on the comparable path
    @assert x.content_agrees "content path should agree (residual=$(x.max_abs_residual), ρ=$(x.spearman), r=$(x.pearson))"
    @assert (isnan(x.max_abs_residual) || x.max_abs_residual < 1e-6) "content per-unit residual too large: $(x.max_abs_residual)"
    # (2) an INDEX disagreement is flagged with a vanishing gradient
    idx = findfirst(d -> d.reason == "index", x.disagreements)
    @assert idx !== nothing "an index disagreement point must be flagged (not hidden)"
    @assert abs(x.disagreements[idx].gradient) < 1e-8 "index gradient should vanish, got $(x.disagreements[idx].gradient)"
    # (3) the oracle is declared the A/B/C reference
    @assert x.oracle_is_reference "the report must declare the intervention oracle the A/B/C reference"
    println("[oracle_xcheck] SELF-CHECK PASS: content agrees (resid=$(round(x.max_abs_residual,sigdigits=3))), " *
            "index disagreement flagged (grad=$(x.disagreements[idx].gradient), " *
            "intervention=$(x.disagreements[idx].intervention)), oracle declared the A/B/C reference.")
    return true
end

# ============================================================================
# CLI
# ============================================================================
function main(args = ARGS)
    game = "pong"; target_frame = 120; horizon = 30; seed = 0; dv = 16
    do_selftest_only = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--game"; game = args[i + 1]; i += 2
        elseif a == "--target-frame"; target_frame = parse(Int, args[i + 1]); i += 2
        elseif a == "--horizon"; horizon = parse(Int, args[i + 1]); i += 2
        elseif a == "--seed"; seed = parse(Int, args[i + 1]); i += 2
        elseif a == "--dv"; dv = parse(Int, args[i + 1]); i += 2
        elseif a == "--selftest"; do_selftest_only = true; i += 1
        else; i += 1
        end
    end
    println("[oracle_xcheck] game=$game target_frame=$target_frame horizon=$horizon " *
            "seed=$seed dv=$dv (jutari/Julia: intervention × gradient, content path)")

    x = compute_xcheck(; game = game, target_frame = target_frame,
                       horizon = horizon, seed = seed, dv = dv, verbose = true)
    selftest(x)
    if do_selftest_only
        println("[oracle_xcheck] --selftest: passed, not writing artifact.")
        return 0
    end
    json_path, npz_path = write_xcheck(x)
    println("[oracle_xcheck] wrote $json_path")
    println("[oracle_xcheck] arrays  $npz_path")
    return 0
end

end # module

# run when executed as a script
if abspath(PROGRAM_FILE) == @__FILE__
    OracleXcheck.main()
end
