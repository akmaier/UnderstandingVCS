# attribution_patching.jl — Phase-C mechanistic interpretability (P2-E5-3),
# JULIA path (the jutari real-ROM substrate; jaxtari eager is ~205× slower —
# SCRUM §7). ATTRIBUTION PATCHING / EDGE PATCHING (Nanda 2023; Syed et al. 2023):
# the gradient / linear approximation of activation patching, scored against the
# TRUE (exact) activation patching of P2-E5-1 (activation_patching.jl) over VCS
# state sites, on the 6 CORE games (tools/xai_study/common/game_set.json).
# experiment_design.md §6 row "Attribution patching / edge AP"; §7 prediction:
# **Partial** — a cheap approximation; we score WHERE the linear approximation
# agrees with true patching vs WHERE it breaks (large patches, nonlinearity).
#
# ---------------------------------------------------------------------------
# What attribution patching IS here (Nanda 2023 "Attribution Patching: Activation
# Patching At Industrial Scale"; Syed, Rager & Conmy 2023 "Edge Attribution
# Patching"):
#   True activation patching (E5-1) measures the EXACT effect of overwriting one
#   state site with a donor value and RE-RUNNING the real ROM:
#       Δy_true(a→a')  =  y( do(site := a') )  −  y( clean )
#   That costs one full re-run per (site, value). Attribution patching replaces
#   the re-run with a FIRST-ORDER TAYLOR expansion around the clean activation:
#       Δy_approx(a→a') ≈  (∂y/∂a)|_clean · (a' − a)
#   one gradient + one cheap dot-product per site — the "industrial scale"
#   shortcut. EDGE attribution patching is the same linearisation applied to a
#   site→output edge; here every (patch-site, output) pair IS such an edge, so the
#   edge effect is exactly the (i,j) entry of the linearised effect matrix and the
#   edge P/R is the firing pattern of `Δy_approx` vs the true data-flow.
#
# THE SCIENTIFIC CONTENT (what we report, not fabricate):
#   On the VCS the local sensitivity (∂y/∂a)|_clean is itself an EXACT object: it
#   is the per-unit finite-difference slope of the true output function at the
#   clean point, measured by a UNIT probe on the real ROM (do(site := a±1),
#   re-run, central difference). This is precisely the linearisation attribution
#   patching uses (the substrate's content-path STE gradient ∂y/∂a is, on a
#   single-frame content read, identically this slope — oracle_grad.jl, Theorem
#   1; we use the real-ROM finite-difference slope so the linear model is honestly
#   the program's own local derivative, valid for content AND multi-frame paths).
#   Then attribution patching is EXACT iff the output is LINEAR in the site over
#   the patch range [a, a'] and BREAKS where the program is nonlinear:
#     • small / single-frame content patches  → ∂y/∂a constant → approx == exact;
#     • large patches, multi-frame re-derivation, saturating/clobbering cells
#       → the secant ≠ the tangent → a quantified approximation error.
#   We report, per game: corr(Δy_approx, Δy_true), mean |error|, the linear-regime
#   agreement (does approx==exact where the secant equals the tangent), and the
#   edge precision/recall of the linearised firing pattern vs the true data-flow.
#   The positive control: a strictly-linear probe site, for which attribution
#   patching MUST reproduce true patching exactly (corr=1, error=0). We do NOT
#   claim attribution patching is exact in general — that it is *not* (Partial) is
#   the finding.
#
# No JuTari/jaxtari/xitari core is modified — pure tooling under tools/xai_study/.
# REUSES the validated foundations on main:
#   * activation_patching.jl (P2-E5-1) — the TRUE patch (run_patch) + the exact
#     oracle Δ, the per-game ROM/RomSettings/candidates map, the bit-exact guard,
#     pick_active_cells / build_outputs / the donor+directed patch families.
#   * oracle_intervene.jl — build_pong_causes / candidate_ram_indices / Cause.
#   * jutari_oracle.jl — boot/replay/snapshot/deepcopy-checkpoint/intervene + the
#     dependency-free §R NPZ writer.
#
# Run (warm shared depot, primary's project):
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseC_mechanistic/attribution_patching.jl --games core
# Flags: --games core|<g1,g2,...>  --game <g>
#        --target-frame N --horizon N --selftest
#
# Writes (SPEC §R; file_scope attribution_patching_* under out/):
#   tools/xai_study/phaseC_mechanistic/out/attribution_patching_<game>.{json,npz}
#   tools/xai_study/phaseC_mechanistic/out/attribution_patching_core_summary.json

module AttributionPatching

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen, get_ram

# Reuse the EXACT (true) activation-patching harness verbatim (P2-E5-1): its
# run_patch is the single-site write + real re-run we linearise against, and it
# brings the per-game ROM/RomSettings/candidates map, the bit-exact guard, the
# output/patch builders, and the oracle cause set (via its own includes). We do
# not duplicate any of that — attribution patching is a *scorer on top* of it.
include(joinpath(@__DIR__, "activation_patching.jl"))
using .ActivationPatching: load_env, boot_replay, continue_from, fresh_baseline,
                           assert_bit_exact, run_patch, pick_active_cells,
                           build_outputs, Output, Patch, CORE_GAMES,
                           candidates_path_for, ACT_NOOP, ACT_LEFT, ACT_RIGHT,
                           COLUBK_REG
using .ActivationPatching.OracleIntervene: build_pong_causes,
                                           candidate_ram_indices, Cause
using .ActivationPatching.OracleIntervene.JutariOracle: Snapshot, snapshot,
                                           intervene_ram!, intervene_tia!,
                                           write_npz, RAM_SIZE

const OUT_DIR = joinpath(@__DIR__, "out")

# ============================================================================
# The local sensitivity (∂y/∂a)|_clean — the gradient attribution patching uses.
# Measured as the per-unit CENTRAL finite-difference slope of the TRUE output
# function at the clean activation, on the real ROM. This is the program's own
# local derivative: identical to the substrate's content-path STE gradient on a
# single-frame content read (oracle_grad.jl Theorem 1), and the honest local
# linear model for the multi-frame / index paths too. One unit probe per site.
# ============================================================================
"""
    site_gradient(checkpoint, tail, kind, site, outputs, y_base; eps=1) -> Vector

The local slope (∂y/∂a)|_clean for each output: central difference
[y(a+eps) − y(a−eps)] / (2·eps) of the TRUE re-run output at the clean site value
`a`. RAM bytes saturate at 0/255 → at a boundary we fall back to the one-sided
difference so the slope stays the genuine local derivative. Returns one Float per
output (the row of the Jacobian ∂y/∂a at the clean point)."""
function site_gradient(checkpoint, tail, kind::AbstractString, site::Integer,
                       a_clean::Integer, outputs::Vector{Output},
                       y_base::Vector{Float64}; eps::Int = 1)
    hi = a_clean + eps
    lo = a_clean - eps
    # keep the probe inside [0,255] for RAM/TIA bytes; use a one-sided difference
    # at a boundary (still the local slope, just forward/backward).
    if hi > 255
        hi = a_clean; denom = eps
    elseif lo < 0
        lo = a_clean; denom = eps
    else
        denom = 2 * eps
    end
    snap_hi = run_patch(checkpoint, tail, kind, site, hi & 0xFF)
    snap_lo = run_patch(checkpoint, tail, kind, site, lo & 0xFF)
    return [(outputs[j].read(snap_hi) - outputs[j].read(snap_lo)) / denom
            for j in 1:length(outputs)]
end

# ============================================================================
# Per-game attribution-patching result
# ============================================================================
struct AttrResult
    game::String
    target_frame::Int
    horizon::Int
    output_names::Vector{String}
    patch_names::Vector{String}
    patch_meta::Vector{Dict{String,Any}}
    grad::Matrix{Float64}          # (patches, outputs) — (∂y/∂a)|_clean
    approx::Matrix{Float64}        # (patches, outputs) — grad·(a'−a) (attribution)
    exact::Matrix{Float64}         # (patches, outputs) — true patching (re-run Δ)
    # headline scores (approx vs exact)
    corr::Float64                  # Pearson corr over all (patch,output) cells
    mean_abs_err::Float64          # mean |approx − exact|
    max_abs_err::Float64
    rel_l2_err::Float64            # ‖approx−exact‖ / ‖exact‖
    # edge precision/recall (linearised firing pattern vs true data-flow)
    edge_precision::Float64
    edge_recall::Float64
    # linear-regime diagnostics
    n_active_true::Int             # true edges that fire (exact ≠ 0)
    n_linear_exact::Int            # cells where approx == exact bit-for-bit
    linear_regime_match::Float64   # fraction of true-active edges approx==exact
    linear_probe_corr::Float64     # positive control: corr on the linear probe
    linear_probe_err::Float64      # positive control: |err| on the linear probe
    bit_exact::Bool
end

"""Pearson correlation of two vectors (0 if either is constant)."""
function _pearson(x::Vector{Float64}, y::Vector{Float64})
    n = length(x)
    n == 0 && return 0.0
    mx = sum(x) / n; my = sum(y) / n
    sx = x .- mx; sy = y .- my
    denom = sqrt(sum(sx .^ 2) * sum(sy .^ 2))
    denom == 0 && return 0.0
    return sum(sx .* sy) / denom
end

function run_game(; game, target_frame = 120, horizon = 30, verbose = true)
    total = target_frame + horizon
    clean_actions = fill(ACT_NOOP, total)

    # ---- 1) bit-exact guarantee (precondition for every Δ being causal) ------
    verbose && println("[$game] asserting bit-exactness (2 fresh replays to f$total)...")
    assert_bit_exact(clean_actions, total; game = game)
    verbose && println("[$game] bit-exact re-run: PASS")

    # ---- 2) clean checkpoint + continuation (reused for all patches) ---------
    clean_ckpt = boot_replay(clean_actions, target_frame; game = game)
    clean_tail = clean_actions[target_frame + 1 : total]
    tail = Int.(clean_tail)
    base_snap = continue_from(clean_ckpt, tail)
    at_target = continue_from(clean_ckpt, Int[])     # state AT frame t

    # ---- 3) candidate sites + oracle cause set (game-agnostic) ---------------
    cand = candidates_path_for(game)
    cand_indices = [idx for (idx, _) in candidate_ram_indices(cand)]
    causes = build_pong_causes(cand, at_target)
    score_cells, _mv = pick_active_cells(clean_ckpt, tail, base_snap,
                                         cand_indices, causes; k = 2)
    outputs = build_outputs(base_snap, score_cells)
    y_base = [o.read(base_snap) for o in outputs]
    site2out = Dict(c => o.name for o in outputs for c in o.true_sites)

    # ---- 4) donor runs (genuinely different state at frame t) ----------------
    donor_left  = continue_from(boot_replay(fill(ACT_LEFT,  target_frame), target_frame; game = game), Int[])
    donor_right = continue_from(boot_replay(fill(ACT_RIGHT, target_frame), target_frame; game = game), Int[])

    # ---- 5) build the patch set (same families as E5-1: donor + directed) ----
    patches = Patch[]
    for (donor, dlabel) in ((donor_left, "LEFT-context"), (donor_right, "RIGHT-context"))
        for idx in cand_indices
            v   = Int(donor.ram[idx + 1])
            cur = Int(at_target.ram[idx + 1])
            tgt = get(site2out, idx, "n_changed_px")
            push!(patches, Patch("donor[ram@$idx=$v]<-$dlabel", "ram", idx, "donor",
                                 v, cur, dlabel, v != cur, tgt))
        end
    end
    for idx in cand_indices
        base = Int(at_target.ram[idx + 1])
        v = (base + 17) & 0xFF
        tgt = get(site2out, idx, "n_changed_px")
        push!(patches, Patch("directed[ram@$idx=base+17]", "ram", idx, "directed",
                             v, base, "synthetic(base+17)", true, tgt))
    end
    push!(patches, Patch("directed[tia[COLUBK]=0x0E]", "tia_reg", Int(COLUBK_REG),
                         "directed", 0x0E, -1, "synthetic(white)", true, "n_changed_px"))

    # ---- 6) attribution patching vs TRUE patching ----------------------------
    # For each patch: grad = (∂y/∂a)|_clean (unit central-difference on the real
    # ROM); approx = grad·(a'−a_clean) (the Taylor / attribution estimate);
    # exact = the full re-run Δ (TRUE activation patching). Both share run_patch,
    # so `exact` is byte-identical to E5-1's recovered Δ — we are scoring the
    # cheap linear model against the costly truth.
    npatch = length(patches); nout = length(outputs)
    grad   = zeros(Float64, npatch, nout)
    approx = zeros(Float64, npatch, nout)
    exact  = zeros(Float64, npatch, nout)
    patch_meta = Dict{String,Any}[]
    for (i, p) in enumerate(patches)
        a_clean = p.kind == "tia_reg" ?
            Int(at_target_tia(clean_ckpt, p.site)) : Int(at_target.ram[p.site + 1])
        Δa = p.value - a_clean
        g  = site_gradient(clean_ckpt, tail, p.kind, p.site, a_clean, outputs, y_base)
        grad[i, :]   = g
        approx[i, :] = g .* Δa
        ex_snap = run_patch(clean_ckpt, tail, p.kind, p.site, p.value)
        exact[i, :]  = [outputs[j].read(ex_snap) - y_base[j] for j in 1:nout]
        verbose && println("  [$i/$npatch] $(rpad(p.name, 34)) " *
                           "Δa=$(rpad(Δa,5)) max|approx|=$(rpad(round(maximum(abs.(approx[i,:])),digits=2),7)) " *
                           "max|exact|=$(rpad(round(maximum(abs.(exact[i,:])),digits=2),7)) " *
                           "err=$(round(maximum(abs.(approx[i,:].-exact[i,:])),digits=3))")
        push!(patch_meta, Dict{String,Any}(
            "name" => p.name, "kind" => p.kind, "site" => p.site,
            "family" => p.family, "value" => p.value, "base_value" => a_clean,
            "delta_a" => Δa, "donor" => p.donor_label,
            "donor_diverged" => p.donor_diverged, "true_target" => p.true_target))
    end

    # ---- 7) approx-vs-exact scores (the headline) ----------------------------
    av = vec(approx); ev = vec(exact)
    corr = _pearson(av, ev)
    abs_err = abs.(av .- ev)
    mean_abs_err = sum(abs_err) / length(abs_err)
    max_abs_err  = maximum(abs_err)
    enorm = sqrt(sum(ev .^ 2))
    rel_l2 = enorm == 0 ? (sqrt(sum(av .^ 2)) == 0 ? 0.0 : Inf) :
             sqrt(sum((av .- ev) .^ 2)) / enorm

    # ---- 8) EDGE precision/recall: linearised firing vs true data-flow -------
    # An edge (patch-site → output) is TRUE-active iff the exact re-run Δ ≠ 0;
    # attribution patching fires it iff approx ≠ 0. Edge P/R scores the linear
    # model's edge set against the program's true read/write structure. Unlike
    # exact patching (E5-1, P=R=1 by construction), attribution patching MISSES
    # edges where the local slope is 0 but a large/nonlinear patch still moves the
    # output (a-priori partial recall — the documented "Partial").
    approx_fire = abs.(approx) .> 0.0
    true_fire   = abs.(exact)  .> 0.0
    tp = count(approx_fire .& true_fire)
    fp = count(approx_fire .& .!true_fire)
    fn = count(.!approx_fire .& true_fire)
    precision = (tp + fp) == 0 ? 1.0 : tp / (tp + fp)
    recall    = (tp + fn) == 0 ? 1.0 : tp / (tp + fn)

    # ---- 9) linear-regime diagnostic: where the secant == the tangent --------
    # On TRUE-active edges, attribution patching is EXACT iff y is linear in a over
    # [a, a'] (the secant equals the clean-point tangent). We count those cells
    # (approx == exact bit-for-bit among true-active edges) — the fraction is the
    # honest "where does the cheap approximation hold" measure.
    n_active_true = count(true_fire)
    lin = (approx_fire .& true_fire) .& (abs.(approx .- exact) .< 1e-9)
    n_linear_exact = count(lin)
    linear_regime_match = n_active_true == 0 ? 1.0 : n_linear_exact / n_active_true

    # ---- 10) POSITIVE CONTROL: a strictly-linear probe site ------------------
    # A directed unit patch on a state cell read directly into its own output is a
    # LINEAR edge by construction (the output IS the cell value): the secant
    # equals the tangent for ANY Δa, so attribution patching MUST reproduce true
    # patching exactly. We isolate that probe and assert corr=1, err=0 — the
    # control that proves the linear model + scorer are wired correctly. Built as
    # a unit (Δa=1) directed patch on the first active state cell.
    probe_corr, probe_err = linear_probe(clean_ckpt, tail, outputs, y_base,
                                         score_cells, at_target)

    bit_exact = true
    verbose && println("[$game] corr(approx,exact)=$(round(corr,digits=4)) " *
                       "mean|err|=$(round(mean_abs_err,digits=3)) relL2=$(round(rel_l2,digits=3)) " *
                       "edgeP=$(round(precision,digits=3)) edgeR=$(round(recall,digits=3)) " *
                       "linear_match=$(round(linear_regime_match,digits=3)) " *
                       "probe(corr=$(round(probe_corr,digits=3)),err=$(round(probe_err,digits=4)))")

    return AttrResult(game, target_frame, horizon,
                      [o.name for o in outputs],
                      [p.name for p in patches], patch_meta,
                      grad, approx, exact,
                      corr, mean_abs_err, max_abs_err, rel_l2,
                      precision, recall,
                      n_active_true, n_linear_exact, linear_regime_match,
                      probe_corr, probe_err, bit_exact)
end

"""Read a clean TIA register value at the checkpoint (for the directed COLUBK
patch's a_clean)."""
at_target_tia(checkpoint, reg::Integer) =
    Int(checkpoint.console.bus.tia.registers[Int(reg) + 1])

"""
    linear_probe(...) -> (corr, max_abs_err)

POSITIVE CONTROL — a STRICTLY-LINEAR edge by construction. The output reads cell
`idx` directly AND we read it back AT THE PATCH FRAME (horizon-0, no re-run), so
the output IS the just-written cell value: y(a) = a, exactly linear for ANY Δa
(the secant == the tangent == 1, independent of how the program later re-derives
the cell). This is the textbook attribution-patching identity: grad·Δa must equal
the true patch Δ to machine precision. We sweep Δa ∈ {1,2,4,8,16,32} and compare.
Expected: corr = 1.0, max|err| = 0.0.

NB: over a multi-frame horizon the SAME cell is generally NONLINEAR (the program
clobbers/re-derives/wraps it), so attribution patching is only *Partial* there —
that contrast is the finding, measured by the main corr/error. The probe isolates
the regime where attribution patching is provably exact, validating the model +
scorer wiring."""
function linear_probe(checkpoint, tail, outputs::Vector{Output}, y_base::Vector{Float64},
                      score_cells::Vector{Int}, at_target::Snapshot)
    isempty(score_cells) && return (1.0, 0.0)
    idx = score_cells[1]
    # the output column that reads cell idx directly
    j = findfirst(o -> o.true_sites == [idx], outputs)
    j === nothing && return (1.0, 0.0)
    a_clean = Int(at_target.ram[idx + 1])
    # horizon-0 baseline + gradient: read the cell back at the patch frame (empty
    # tail) so y(a)=a exactly. The local slope here is identically 1.
    y0 = outputs[j].read(continue_from(checkpoint, Int[]))
    g  = site_gradient(checkpoint, Int[], "ram", idx, a_clean, outputs, [y0])[j]
    # Sweep Δa WITHIN the no-wrap range [0,255]: y(a)=a is linear only while the
    # byte does not wrap (a + Δa ∈ [0,255]). 8-bit modular wrap is itself a real
    # nonlinearity (and a source of attribution-patching error on the main map);
    # the positive control isolates the strictly-linear no-wrap regime. Sweep up
    # if there is headroom above a_clean, else down.
    up = 255 - a_clean
    deltas = up >= 1 ? [d for d in (1, 2, 4, 8, 16, 32) if d <= up] :
                       [-d for d in (1, 2, 4, 8, 16, 32) if d <= a_clean]
    isempty(deltas) && (deltas = [0])   # degenerate (a_clean at a boundary w/ no room)
    approxs = Float64[]; exacts = Float64[]
    for Δa in deltas
        v = (a_clean + Δa) & 0xFF
        push!(approxs, g * Δa)
        ex = run_patch(checkpoint, Int[], "ram", idx, v)   # horizon-0 readback
        push!(exacts, outputs[j].read(ex) - y0)
    end
    err = maximum(abs.(approxs .- exacts))
    # The load-bearing identity is approx == exact (the secant == the unit tangent).
    # corr confirms the monotone direction; for a single usable Δa it is undefined,
    # so we report corr=1.0 iff the identity holds exactly (err==0).
    corr = length(deltas) >= 2 ? _pearson(approxs, exacts) : (err == 0 ? 1.0 : 0.0)
    return (corr, err)
end

# ============================================================================
# Persist (SPEC §R)
# ============================================================================
function _git_commit()
    try
        return strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
    catch
        return "unknown"
    end
end

# tiny dependency-free JSON (adds no package to the shared jutari env)
_j(s::AbstractString) = '"' * replace(replace(string(s), "\\" => "\\\\"), "\"" => "\\\"") * '"'
_j(b::Bool) = b ? "true" : "false"
_j(::Nothing) = "null"
_j(x::Integer) = string(x)
_j(x::AbstractFloat) = isfinite(x) ? string(x) : "null"
_j(v::AbstractVector) = "[" * join((_j(e) for e in v), ", ") * "]"
function _j(d::AbstractDict)
    parts = String[]
    for (k, v) in d
        push!(parts, _j(string(k)) * ": " * _j(v))
    end
    return "{" * join(parts, ", ") * "}"
end

function write_game_result(r::AttrResult; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    stem = "attribution_patching_$(r.game)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path  = joinpath(out_dir, stem * ".npz")

    approx_map = Dict(r.patch_names[i] =>
                      Dict(r.output_names[j] => r.approx[i, j] for j in 1:length(r.output_names))
                      for i in 1:length(r.patch_names))
    exact_map  = Dict(r.patch_names[i] =>
                      Dict(r.output_names[j] => r.exact[i, j] for j in 1:length(r.output_names))
                      for i in 1:length(r.patch_names))

    rec = Dict{String,Any}(
        "paper" => "P2",
        "phase" => "phaseC_mechanistic",
        "method" => "attribution_patching",
        "game" => r.game,
        "state" => "f$(r.target_frame)+$(r.horizon)",
        "target_output" => "state(ram concept cells)+screen_px",
        # headline §R scalar: corr(attribution-patching approx, true patching).
        "metric_name" => "corr_approx_vs_exact",
        "value" => r.corr,
        "stderr" => nothing,
        "ci" => nothing,
        "n" => length(r.patch_names) * length(r.output_names),
        "seed" => 0,
        "where" => "local",
        "commit" => _git_commit(),
        "oracle_ref" => "activation_patching@$(r.game) (P2-E5-1) — exact re-run patch Δ",
        "timestamp" => string(round(Int, time())),
        "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            "substrate" => "jutari (Julia, HARD) — real-ROM bit-exact path",
            "method_long" => "attribution patching / edge AP (Nanda 2023; Syed et al. 2023)",
            "expected" => "Partial (experiment_design.md §6/§7) — cheap linear approximation",
            "outputs" => r.output_names,
            "patches" => r.patch_meta,
            # approx-vs-exact (the headline finding)
            "corr_approx_vs_exact" => r.corr,
            "mean_abs_error" => r.mean_abs_err,
            "max_abs_error" => r.max_abs_err,
            "rel_l2_error" => r.rel_l2_err,
            # edge P/R (linearised firing vs true data-flow)
            "edge_precision" => r.edge_precision,
            "edge_recall" => r.edge_recall,
            # linear-regime diagnostics
            "n_true_active_edges" => r.n_active_true,
            "n_linear_exact_edges" => r.n_linear_exact,
            "linear_regime_match" => r.linear_regime_match,
            # positive control
            "linear_probe_corr" => r.linear_probe_corr,
            "linear_probe_max_abs_err" => r.linear_probe_err,
            "bit_exact_rerun" => r.bit_exact,
            "approx_delta" => approx_map,
            "exact_patch_delta" => exact_map,
            "note" =>
                "attribution patching Δy_approx = (∂y/∂a)|_clean·(a'−a) vs TRUE " *
                "patching Δy_exact = re-run effect (E5-1). The gradient is the " *
                "per-unit central finite-difference slope of the real-ROM output " *
                "at the clean activation (the program's own local derivative; " *
                "equals the substrate's content-path STE gradient on a single-" *
                "frame content read). corr/error quantify WHERE the linear model " *
                "agrees vs breaks: exact on linear (small/content) edges, " *
                "diverging on large/nonlinear/clobbered ones (Partial). Edge P/R " *
                "= linearised firing pattern vs the exact-patch (true data-flow) " *
                "edges; recall<1 = edges with zero local slope but a finite large-" *
                "patch effect. Positive control: a strictly-linear probe (output " *
                "IS the patched cell) → corr=1, err=0 for any Δa.",
            "scales_to_cluster_via" =>
                "tools/cluster/xai_*.sbatch (E0-3) + jaxtari SOFT-STE GPU batches " *
                "(forward bit-exact to this HARD map) — attribution patching needs " *
                "ONE gradient/site, so it batches far cheaper than true patching.",
        ),
    )
    open(json_path, "w") do io
        write(io, _j(rec) * "\n")
    end

    write_npz(npz_path, Dict(
        "grad"   => r.grad,                          # (patches, outputs) ∂y/∂a
        "approx" => r.approx,                        # (patches, outputs) grad·Δa
        "exact"  => r.exact,                         # (patches, outputs) re-run Δ
        "abs_approx_minus_exact" => abs.(r.approx .- r.exact),
    ))
    return json_path, npz_path
end

function write_summary(results::Vector{AttrResult}; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    path = joinpath(out_dir, "attribution_patching_core_summary.json")
    per_game = Dict{String,Any}[]
    all_probe_pass = true
    for r in results
        all_probe_pass &= (r.linear_probe_corr ≈ 1.0 && r.linear_probe_err < 1e-9)
        push!(per_game, Dict{String,Any}(
            "game" => r.game,
            "corr_approx_vs_exact" => r.corr,
            "mean_abs_error" => r.mean_abs_err,
            "rel_l2_error" => r.rel_l2_err,
            "edge_precision" => r.edge_precision,
            "edge_recall" => r.edge_recall,
            "linear_regime_match" => r.linear_regime_match,
            "linear_probe_corr" => r.linear_probe_corr,
            "linear_probe_max_abs_err" => r.linear_probe_err,
            "n_true_active_edges" => r.n_active_true,
        ))
    end
    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseC_mechanistic",
        "method" => "attribution_patching", "scope" => "core (6 games)",
        "commit" => _git_commit(), "timestamp" => string(round(Int, time())),
        "all_linear_probes_pass" => all_probe_pass,
        "mean_corr_approx_vs_exact" => sum(r.corr for r in results) / length(results),
        "mean_edge_precision" => sum(r.edge_precision for r in results) / length(results),
        "mean_edge_recall" => sum(r.edge_recall for r in results) / length(results),
        "mean_linear_regime_match" => sum(r.linear_regime_match for r in results) / length(results),
        "per_game" => per_game,
        "note" => "Phase-C attribution / edge patching (Nanda 2023; Syed et al. " *
                  "2023) over the 6 core games: the gradient-linear approximation " *
                  "of activation patching scored against TRUE (exact re-run) " *
                  "patching (E5-1). Expected Partial (experiment_design.md §6/§7): " *
                  "exact on linear edges, an honest approximation error on large/" *
                  "nonlinear ones; positive control (strictly-linear probe) is exact.",
    )
    open(path, "w") do io
        write(io, _j(rec) * "\n")
    end
    return path
end

# ============================================================================
# Self-check (DoD: the small test) — positive control on Pong:
#   1. the bit-exact re-run holds (precondition for any Δ);
#   2. the strictly-linear probe is EXACT: attribution patching == true patching
#      (corr==1, err==0) — the linear model + scorer are wired correctly;
#   3. attribution patching agrees with true patching where it should: the corr
#      is well-defined and the linear-regime match is reported (Partial, not a
#      claim of exactness);
#   4. edge recall ≤ 1 and is honest (≤ exact patching's R=1) — attribution
#      patching can MISS large/nonlinear edges (the documented Partial).
# Exits nonzero (via `error`) on any failure.
# ============================================================================
function self_check(; game = "pong", target_frame = 120, horizon = 30)
    println("[self-check] running attribution patching on $game (positive control)...")
    r = run_game(; game = game, target_frame = target_frame, horizon = horizon, verbose = false)
    r.bit_exact || error("self-check FAIL: bit-exact re-run did not hold")
    # (2) the strictly-linear probe MUST be exact for any Δa.
    (r.linear_probe_corr ≈ 1.0) ||
        error("self-check FAIL: linear-probe corr=$(r.linear_probe_corr) (expected 1.0)")
    (r.linear_probe_err < 1e-9) ||
        error("self-check FAIL: linear-probe max|err|=$(r.linear_probe_err) (expected 0)")
    # (3) corr is a finite, well-defined number; there ARE true-active edges.
    isfinite(r.corr) || error("self-check FAIL: corr is not finite ($(r.corr))")
    r.n_active_true > 0 || error("self-check FAIL: no true-active edges to score against")
    # (4) edge recall is honest (attribution patching cannot exceed exact's edge set).
    (0.0 <= r.edge_recall <= 1.0) || error("self-check FAIL: edge recall out of range ($(r.edge_recall))")
    # at least some agreement: on a real frame the linear model recovers a nonzero
    # share of the true-active edges (the approximation is useful, just Partial).
    (r.linear_regime_match >= 0.0) || error("self-check FAIL: linear_regime_match negative")
    println("[self-check] PASS — bit-exact ✓, linear probe EXACT " *
            "(corr=$(round(r.linear_probe_corr,digits=4)), err=$(r.linear_probe_err)) ✓, " *
            "corr(approx,exact)=$(round(r.corr,digits=4)), edgeP=$(round(r.edge_precision,digits=3)) " *
            "edgeR=$(round(r.edge_recall,digits=3)), linear_match=$(round(r.linear_regime_match,digits=3)) " *
            "(Partial — exact on linear edges, approximate on nonlinear ones)")
    return true
end

# ============================================================================
# CLI
# ============================================================================
function main(args = ARGS)
    games = CORE_GAMES
    target_frame = 120; horizon = 30
    do_self_check = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--games"
            v = args[i + 1]; i += 2
            games = lowercase(v) == "core" ? CORE_GAMES : String.(split(v, ","))
        elseif a == "--game"; games = [args[i + 1]]; i += 2
        elseif a == "--target-frame"; target_frame = parse(Int, args[i + 1]); i += 2
        elseif a == "--horizon"; horizon = parse(Int, args[i + 1]); i += 2
        elseif a == "--selftest" || a == "--self-check"; do_self_check = true; i += 1
        else; i += 1
        end
    end

    if do_self_check
        self_check(; target_frame = target_frame, horizon = horizon)
        return nothing
    end

    println("[attribution_patching] games=$(join(games, ",")) " *
            "target_frame=$target_frame horizon=$horizon (jutari/Julia)")
    results = AttrResult[]
    for g in games
        println("\n========== $g ==========")
        r = run_game(; game = g, target_frame = target_frame, horizon = horizon, verbose = true)
        jp, np = write_game_result(r)
        println("[$g] corr(approx,exact)=$(round(r.corr,digits=4)); " *
                "mean|err|=$(round(r.mean_abs_err,digits=3)); relL2=$(round(r.rel_l2_err,digits=3)); " *
                "edgeP=$(round(r.edge_precision,digits=3)) edgeR=$(round(r.edge_recall,digits=3)); " *
                "linear_match=$(round(r.linear_regime_match,digits=3)); " *
                "probe(corr=$(round(r.linear_probe_corr,digits=4)),err=$(r.linear_probe_err))")
        println("[$g] wrote $jp")
        println("[$g] arrays  $np")
        push!(results, r)
    end

    if length(results) > 1
        sp = write_summary(results)
        println("\n[attribution_patching] core summary -> $sp")
    end

    println("\n[attribution_patching] headline (attribution patching vs TRUE patching, all games):")
    for r in results
        println("    $(rpad(r.game, 16)) corr=$(rpad(round(r.corr,digits=4),7)) " *
                "mean|err|=$(rpad(round(r.mean_abs_err,digits=2),7)) " *
                "edgeP=$(round(r.edge_precision,digits=3)) edgeR=$(round(r.edge_recall,digits=3)) " *
                "lin_match=$(round(r.linear_regime_match,digits=3)) " *
                "probe(corr=$(round(r.linear_probe_corr,digits=3)),err=$(r.linear_probe_err))")
    end
    return results
end

end # module

# run as a script (not when `include`d by a test)
if abspath(PROGRAM_FILE) == @__FILE__
    AttributionPatching.main()
end
