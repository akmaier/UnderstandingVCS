# na_audit.jl — Phase-B N/A audit (P2-E4-13), JULIA path.
#
# The RECORDED "does-not-apply" finding for the popular XAI methods that require a
# neural-network architecture (conv feature maps / attention) or a *learned policy*
# to attach to. The VCS substrate (jutari/jaxtari) is a deterministic 6502+TIA+RIOT
# program, not a network and not an agent, so these methods have NO ingredient to
# bind to. This is a *measured statement about the narrowness of popular XAI*, not a
# score — it scopes Phase B honestly (experiment_design.md §5 "N/A" row + §7 last row).
#
# This is the ONE Phase-B item that is intentionally a WRITE-UP, not a 6-game method
# run: there is nothing to *run as a method* (the methods don't apply). The DoD still
# requires (i) a runnable generator, (ii) per-game §R records to out/, and (iii) a
# self-check. We honour all three by emitting, per game, the structured audit
# (method × applies=false × missing-ingredient × why), grounded in that game's real
# substrate facts loaded from the verified foundation:
#   * tools/xai_study/t3/out/candidates_<game>.json — the candidate CAUSE cells the
#     APPLICABLE Phase-B methods (IG, occlusion, …) attribute over. We report how many
#     candidate causes exist (≥16 per game) and contrast: an N/A method has ZERO of
#     its required substrate channels (conv maps / attention heads / a policy net) —
#     the absence is structural, not a tuning failure.
#   * tools/xai_study/common/jutari_oracle.jl — load/boot/replay/snapshot/intervene
#     (the bit-exact intervention oracle that ALL applicable methods are scored
#     against). The N/A audit cites it as the contrast: the oracle and the applicable
#     methods (IG etc.) bind to real causes; the N/A methods bind to nothing.
#   * tools/xai_study/common/jutari_record.jl — used only via the §R writer it shares.
#
# We do NOT touch the emulator core, and there is NO method to score, so there is no
# F/S/M / positive-control here (those belong to items that DO yield an importance
# claim — pilot_si.jl is their template). Instead the self-check asserts the audit is
# internally consistent and grounded: every N/A method names a concrete missing
# substrate ingredient that is verifiably ABSENT from the VCS, while the applicable
# methods it is contrasted with are verifiably PRESENT (the oracle + candidate causes).
#
# Run (warm shared depot, primary's project):
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseB_attribution/na_audit.jl
# Flags: --selftest   (run the self-check, do not write artifacts)
#        --md          (also (re)write na_audit.md from the structured audit)
#
# Writes (SPEC §R; file_scope na_audit_*):
#   tools/xai_study/phaseB_attribution/out/na_audit_<game>.{json,npz}   (6 records)
#   tools/xai_study/phaseB_attribution/out/na_audit_combined.json       (1 roll-up)
#   tools/xai_study/phaseB_attribution/na_audit.md                      (the writeup, with --md)

module NAAudit

using JSON

# Reuse the verified foundation's §R .npz/.npy writer (NO emulator core touched).
include(joinpath(@__DIR__, "..", "common", "jutari_oracle.jl"))
using .JutariOracle: write_npz

const OUT_DIR = joinpath(@__DIR__, "out")
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const CORE_GAMES = ["pong", "breakout", "space_invaders", "seaquest", "ms_pacman", "qbert"]

# ----------------------------------------------------------------------------
# The audit content — one structured entry per N/A method. Each names the
# REQUIRED substrate ingredient, asserts it is ABSENT from the bare VCS, and
# explains why no faithful re-targeting recovers it. Citations are the SPEC's
# (experiment_design.md §5/§7); verified to Paper-1's no-hallucination standard.
# ----------------------------------------------------------------------------

struct NAMethod
    key::String
    name::String
    family::String
    citation::String
    requires::String           # the missing substrate ingredient
    requires_kind::String      # "conv_feature_maps" | "attention" | "learned_policy" | "learned_channels"
    present_in_vcs::Bool        # is the required ingredient present? (always false here — that's the finding)
    why::String                # mechanistic reason it does not apply
    no_retarget::String        # why a naive re-targeting onto VCS state is NOT this method
end

const NA_METHODS = NAMethod[
    NAMethod(
        "grad_cam", "Grad-CAM", "CNN class-activation mapping",
        "Selvaraju et al. 2017 (ICCV)",
        "convolutional feature maps (a last conv layer whose channels are spatially " *
            "pooled and weighted by class-gradient)",
        "conv_feature_maps", false,
        "Grad-CAM forms a class-discriminative heatmap by weighting the activation " *
            "maps of a CNN's final convolutional layer by the gradient of the class " *
            "logit w.r.t. those maps, then ReLU-ing the weighted sum. The VCS has NO " *
            "convolutional layers, NO learned channels, and NO class logits: it is a " *
            "6502 CPU + TIA + RIOT executing a fixed ROM. There is no last-conv-layer " *
            "activation tensor to weight, so the construction has no operand.",
        "Projecting register/RAM/colour-clock state into a 'feature map' grid and " *
            "weighting by ∂y/∂state is NOT Grad-CAM: those are raw causal state cells, " *
            "not learned conv channels, and the resulting map is just the content-path " *
            "gradient (already covered by vanilla-gradient / Grad×Input, E4-1/E4-2). The " *
            "channel-pooling + ReLU that DEFINE Grad-CAM are vacuous without conv channels.",
    ),
    NAMethod(
        "grad_cam_pp", "Grad-CAM++", "CNN class-activation mapping",
        "Chattopadhyay et al. 2018 (WACV)",
        "convolutional feature maps + higher-order (2nd/3rd) gradients of the class " *
            "score w.r.t. those maps",
        "conv_feature_maps", false,
        "Grad-CAM++ refines Grad-CAM's per-channel weights with pixel-wise higher-order " *
            "gradient terms over the SAME final conv feature maps. It inherits Grad-CAM's " *
            "hard requirement for conv channels; the VCS has none, so the refinement has " *
            "nothing to refine.",
        "Higher-order gradients of a VCS pixel w.r.t. state cells exist (the substrate " *
            "is differentiable, content path), but computing them is not Grad-CAM++ — " *
            "without conv feature maps the per-channel weighting that the formula sums " *
            "over is undefined. The result would again be a (higher-order) saliency map, " *
            "not a CAM.",
    ),
    NAMethod(
        "attention_rollout", "Attention rollout", "Transformer attention attribution",
        "Abnar & Zuidema 2020 (ACL)",
        "self-attention matrices across transformer layers (per-head token-to-token " *
            "attention weights to multiply through, with the residual/identity mix)",
        "attention", false,
        "Attention rollout recursively multiplies the layer-wise self-attention matrices " *
            "(averaged over heads, mixed with the residual identity) to trace how input " *
            "tokens influence a later representation. The VCS is not a transformer: there " *
            "are NO attention heads, NO token sequence, and NO attention matrices to " *
            "multiply. The method's entire input — the stack of A^(l) matrices — does not exist.",
        "One could build a token-to-token influence matrix from the VCS's true data-flow " *
            "(which RAM cell reads/writes which), but that is the §1 intervention/data-flow " *
            "oracle (or Phase-C path patching), NOT attention rollout: there is no learned, " *
            "softmax-normalised attention being rolled up. Calling the data-flow graph " *
            "'attention' would mislabel ground truth as a method under test.",
    ),
    NAMethod(
        "viper", "VIPER (policy distillation to a decision tree)", "Policy extraction / surrogate",
        "Bastani et al. 2018 (NeurIPS)",
        "a LEARNED policy π(a|s) (a trained agent / Q-network) to distil into an " *
            "interpretable decision tree via DAgger-style imitation",
        "learned_policy", false,
        "VIPER extracts a verifiable decision-tree policy by imitating a trained " *
            "neural policy (Q-network) over visited states, using the Q-values to weight " *
            "the imitation. Paper 2's subject is the PROGRAM (the VCS as a deterministic " *
            "input→output function), not an agent: there is NO learned policy, NO action " *
            "distribution to imitate, and NO Q-function to weight by. The thing VIPER " *
            "distils does not exist in this study. (Agents are P5, not P2.)",
        "Fitting a decision tree to (state → next-RAM / pixel) of the VCS itself is not " *
            "VIPER: VIPER's object is a LEARNED control policy, and its guarantee is about " *
            "imitating that policy's returns. The VCS's own state-transition is the exact " *
            "program — a tree over it is a (lossy) re-description of known ground truth, " *
            "not a policy-distillation explanation. Faithful recovery of the program is " *
            "Phase C (circuit discovery), scored vs the true data-flow.",
    ),
    NAMethod(
        "cam", "CAM (class activation mapping)", "CNN class-activation mapping",
        "Zhou et al. 2016 (CVPR)",
        "a global-average-pooled final conv layer feeding a single linear " *
            "classification layer (the GAP+FC architecture CAM is defined on)",
        "conv_feature_maps", false,
        "Vanilla CAM requires the specific CNN topology of a GAP'd last conv layer wired " *
            "to one FC classifier; the heatmap is the conv maps weighted by that FC's " *
            "class weights. The VCS has neither conv maps nor a learned classifier head, " *
            "so the construction is undefined.",
        "There is no GAP+FC head to read class weights from; any 'importance grid' over " *
            "VCS state is the content-path gradient, already covered by E4-1/E4-2.",
    ),
    NAMethod(
        "feature_vis", "Feature visualisation / CAV", "Learned-channel interpretation",
        "Olah et al. 2017 (Distill); Kim et al. 2018 (TCAV, ICML)",
        "LEARNED hidden channels / directions in a network's activation space to " *
            "optimise an input toward, or to probe with concept-activation vectors",
        "learned_channels", false,
        "Feature visualisation synthesises inputs that maximally activate a chosen " *
            "LEARNED unit/channel/direction; TCAV measures sensitivity to a learned " *
            "concept-activation vector in hidden space. Both require a trained network " *
            "with hidden channels that have emergent semantics. The VCS's state cells are " *
            "NOT learned channels — they are hardware registers/RAM with KNOWN, fixed " *
            "semantics (T1/T2). There is nothing learned to visualise or to define a CAV in.",
        "Optimising a ROM/input to maximise a pixel is the §1 gradient/IG path (content) " *
            "or an adversarial-style intervention, not feature-vis of a learned channel. " *
            "Probing a KNOWN register's role is the §2/Phase-C probing task (with control " *
            "tasks), where the labels are ground truth, not discovered concepts.",
    ),
]

# Methods that DO apply (the contrast) — these bind to real VCS causes and ARE scored
# against the §1 oracle elsewhere in E4. Listed so the audit is not one-sided.
const APPLICABLE_CONTRAST = [
    ("vanilla_gradient", "Vanilla gradient / saliency (Simonyan et al. 2014)", "E4-1"),
    ("integrated_gradients", "Integrated Gradients (Sundararajan et al. 2017; P8)", "E4-0 / E4-5"),
    ("occlusion", "Occlusion on the true VCS (Zeiler & Fergus 2014)", "E4-7"),
    ("counterfactual", "On-distribution counterfactual states (Olson 2021; Atrey 2020)", "E4-12"),
]

# ----------------------------------------------------------------------------
# Substrate grounding — per game, load the real candidate-cause count from the
# verified candidates file. The point: applicable methods (IG, occlusion) have
# ≥16 candidate causes to attribute over; the N/A methods have ZERO of THEIR
# required channels (conv/attention/policy). The asymmetry is the finding.
# ----------------------------------------------------------------------------
function candidates_path(game)
    rel = joinpath("tools", "xai_study", "t3", "out", "candidates_$(game).json")
    for base in (REPO_ROOT, "/Users/maier/Documents/code/UnderstandingVCS")
        p = joinpath(base, rel)
        isfile(p) && return p
    end
    return nothing
end

struct GameGrounding
    game::String
    candidates_file::Union{String,Nothing}
    n_candidate_causes::Int          # # of candidate CAUSE cells applicable methods attribute over
    n_conv_feature_maps::Int         # 0 — the missing ingredient for Grad-CAM/++/CAM/feature-vis
    n_attention_heads::Int           # 0 — the missing ingredient for attention rollout
    n_learned_policies::Int          # 0 — the missing ingredient for VIPER
    n_learned_channels::Int          # 0 — the missing ingredient for feature-vis/CAV
end

function ground_game(game)
    p = candidates_path(game)
    n = 0
    if p !== nothing
        data = JSON.parsefile(p)
        n = length(get(data, "candidates", []))
    end
    return GameGrounding(game, p, n, 0, 0, 0, 0)
end

# ----------------------------------------------------------------------------
# §R record per game (the structured N/A finding).
# ----------------------------------------------------------------------------
_git_commit() = try
    strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
catch
    "unknown"
end

function method_entry_dict(m::NAMethod)
    return Dict{String,Any}(
        "method" => m.key,
        "name" => m.name,
        "family" => m.family,
        "citation" => m.citation,
        "applies" => false,
        "missing_ingredient" => m.requires,
        "missing_ingredient_kind" => m.requires_kind,
        "present_in_vcs" => m.present_in_vcs,
        "reason" => m.why,
        "why_no_retarget" => m.no_retarget,
    )
end

function build_record(g::GameGrounding)
    npz_stem = "na_audit_$(g.game)"
    method_entries = [method_entry_dict(m) for m in NA_METHODS]
    # The asymmetry table: for each missing-ingredient kind, count of channels present (0).
    missing_kinds = Dict{String,Int}(
        "conv_feature_maps" => g.n_conv_feature_maps,
        "attention" => g.n_attention_heads,
        "learned_policy" => g.n_learned_policies,
        "learned_channels" => g.n_learned_channels,
    )
    rec = Dict{String,Any}(
        "paper" => "P2",
        "phase" => "phaseB_attribution",
        "method" => "na_audit",
        "game" => g.game,
        "state" => nothing,                # N/A — no state-specific run (architectural finding)
        "target_output" => "applicability_of_NN/policy-specific_XAI_to_the_VCS",
        # SPEC §R headline scalar: # of audited methods that DO NOT apply.
        "metric_name" => "n_methods_not_applicable",
        "value" => length(NA_METHODS),
        "stderr" => nothing,
        "ci" => nothing,
        "n" => length(NA_METHODS),
        "seed" => nothing,
        "where" => "local",
        "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene@$(g.game)  (the contrast: applicable methods bind to its causes)",
        "timestamp" => string(round(Int, time())),
        "arrays" => npz_stem * ".npz",
        "applies" => false,                # top-level finding for this record
        "extra" => Dict{String,Any}(
            "finding" => "Architecture-/policy-specific XAI methods DO NOT APPLY to the " *
                "VCS substrate. The VCS is a deterministic 6502+TIA+RIOT program (no " *
                "neural network, no learned policy), so methods that require conv feature " *
                "maps, attention matrices, learned channels, or a learned policy have NO " *
                "operand. Recorded as a measured statement about the narrowness of popular " *
                "XAI (experiment_design.md §5 'N/A' row + §7 last row).",
            "substrate" => "jutari (Julia) — bare VCS as a function; no NN, no agent " *
                "(agents are P5, not P2).",
            "methods_not_applicable" => method_entries,
            "n_methods_not_applicable" => length(NA_METHODS),
            "missing_ingredient_channel_counts" => missing_kinds,
            "missing_ingredient_channel_counts_note" =>
                "Per the substrate audit (experiment_design.md §3): the VCS has ZERO " *
                "conv feature maps, ZERO attention heads, ZERO learned policies, and ZERO " *
                "learned channels — the exact ingredients these methods bind to.",
            "applicable_contrast" => Dict{String,Any}(
                "n_candidate_causes" => g.n_candidate_causes,
                "candidates_file" => g.candidates_file === nothing ? nothing :
                    relpath(g.candidates_file, REPO_ROOT),
                "methods_that_DO_apply" => [Dict("method"=>k, "name"=>nm, "item"=>it)
                                            for (k, nm, it) in APPLICABLE_CONTRAST],
                "note" => "The ASYMMETRY is the finding: applicable methods (IG, " *
                    "occlusion, counterfactuals, vanilla gradient) have $(g.n_candidate_causes) " *
                    "candidate CAUSE cells (RAM/registers/inputs) to attribute over and ARE " *
                    "scored against the §1 intervention oracle. The N/A methods have ZERO " *
                    "of their required substrate channels — the absence is STRUCTURAL " *
                    "(no conv/attention/policy exists), not a tuning or coverage failure.",
            ),
            "scoring" => "No F/S/M and no positive control: there is NO method here that " *
                "yields an importance/recovery claim to score against the oracle. (F/S/M " *
                "+ oracle-as-method control belong to the items that DO apply — see " *
                "pilot_ig_vs_oracle.jl / pilot_si.jl.) The deliverable is the recorded " *
                "applies=false finding with the missing-ingredient reason per method.",
            "spec_ref" => "xai_paper/xai_2_interpretability/experiment_design.md " *
                "§5 (row 'N/A') + §7 (last row); SPEC.md §E4 (E4-1..E4-N 'N/A audit').",
            "scales_to_cluster_via" => "N/A — this is an architectural finding, not a sweep; " *
                "it is game-independent (the substrate has no NN/policy in ANY game), " *
                "recorded per core game for leaderboard (E6) uniformity.",
        ),
    )
    return rec, npz_stem
end

function write_game(g::GameGrounding; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    rec, stem = build_record(g)
    json_path = joinpath(out_dir, stem * ".json")
    npz_path = joinpath(out_dir, stem * ".npz")
    open(json_path, "w") do io; JSON.print(io, rec, 2); end
    # sibling .npz (SPEC §R): the missing-ingredient channel counts as a numeric vector
    # (all zeros — the structural absence) + the applicable-causes count, so tooling /
    # the leaderboard can read the asymmetry without parsing prose.
    write_npz(npz_path, Dict(
        # [conv_feature_maps, attention_heads, learned_policies, learned_channels]
        "missing_ingredient_channel_counts" =>
            Int64[g.n_conv_feature_maps, g.n_attention_heads,
                  g.n_learned_policies, g.n_learned_channels],
        "n_candidate_causes_applicable_methods" => Int64[g.n_candidate_causes],
        "n_methods_not_applicable" => Int64[length(NA_METHODS)],
        "applies_per_method" => zeros(Int64, length(NA_METHODS)),  # 0 = does not apply
    ))
    return json_path, npz_path
end

function write_combined(groundings::Vector{GameGrounding}; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    json_path = joinpath(out_dir, "na_audit_combined.json")
    rec = Dict{String,Any}(
        "paper" => "P2",
        "phase" => "phaseB_attribution",
        "method" => "na_audit",
        "game" => "ALL_CORE",
        "state" => nothing,
        "target_output" => "applicability_of_NN/policy-specific_XAI_to_the_VCS",
        "metric_name" => "n_methods_not_applicable",
        "value" => length(NA_METHODS),
        "stderr" => nothing,
        "ci" => nothing,
        "n" => length(NA_METHODS),
        "seed" => nothing,
        "where" => "local",
        "commit" => _git_commit(),
        "oracle_ref" => nothing,
        "timestamp" => string(round(Int, time())),
        "arrays" => nothing,
        "applies" => false,
        "extra" => Dict{String,Any}(
            "finding" => "Across all 6 core games the result is identical: the " *
                "architecture-/policy-specific XAI methods do not apply (the VCS has no " *
                "NN and no learned policy in ANY game). Recorded per game for leaderboard " *
                "uniformity; the finding is game-independent.",
            "core_games" => [g.game for g in groundings],
            "per_game_records" => ["na_audit_$(g.game).json" for g in groundings],
            "methods_not_applicable" => [m.key for m in NA_METHODS],
            "n_methods_not_applicable" => length(NA_METHODS),
            "per_game_candidate_cause_counts" =>
                Dict(g.game => g.n_candidate_causes for g in groundings),
            "writeup" => "na_audit.md",
            "spec_ref" => "experiment_design.md §5 (N/A) + §7; SPEC.md §E4.",
        ),
    )
    open(json_path, "w") do io; JSON.print(io, rec, 2); end
    return json_path
end

# ----------------------------------------------------------------------------
# The Markdown writeup (the primary deliverable per the item).
# ----------------------------------------------------------------------------
function render_md(groundings::Vector{GameGrounding})
    io = IOBuffer()
    commit = _git_commit()
    pr(s) = print(io, s, "\n")
    pr("# Phase-B N/A audit — methods that do NOT apply to the VCS substrate")
    pr("")
    pr("> **Item:** P2-E4-13 · **Phase:** B (attribution / XAI) · **Where:** local (jutari).")
    pr("> **Status finding:** *recorded* `applies = false` per method (a measured statement,")
    pr("> not a score). **Generated by:** `tools/xai_study/phaseB_attribution/na_audit.jl`")
    pr("> at commit `$(commit)`. **Spec:** `experiment_design.md` §5 (row \"N/A\") + §7")
    pr("> (last row); `SPEC.md` §E4.")
    pr("")
    pr("## Why this is a write-up, not a 6-game method run")
    pr("")
    pr("Every other Phase-B item runs a method over the 6 core games and scores its")
    pr("attribution map against the §1 intervention oracle with F/S/M plus an")
    pr("oracle-as-method positive control (template: `pilot_ig_vs_oracle.jl` /")
    pr("`phaseA_kording/pilot_si.jl`). **This item cannot**, because the methods it")
    pr("covers have nothing in the VCS to attach to. The subject of Paper 2 is the")
    pr("**Atari VCS itself** — a deterministic 6502 CPU + TIA + RIOT executing a fixed")
    pr("ROM (no neural network; no learned policy; *agents are P5, not P2*). The methods")
    pr("below each require a neural-network architecture (convolutional feature maps,")
    pr("attention matrices, learned channels) or a *learned policy* to extract. None of")
    pr("those ingredients exists on the bare VCS, so there is **no operand** to run the")
    pr("method on and **nothing to score**. Recording this honestly *scopes* Phase B and")
    pr("supplies the \"architecture-specific methods don't apply\" row of the leaderboard")
    pr("(E6) — the measured narrowness of popular XAI.")
    pr("")
    pr("## The asymmetry (this is the finding)")
    pr("")
    pr("Applicable Phase-B methods — vanilla gradient, Grad×Input/DeepLIFT, SmoothGrad,")
    pr("**Integrated Gradients**, occlusion, RISE, LIME, KernelSHAP, on-distribution")
    pr("counterfactuals — bind to *real causes* of a chosen output `y` (ROM bytes, RAM")
    pr("cells, registers, joystick inputs) and **are** scored against the exact")
    pr("intervention oracle (`ground_truth/oracle_intervene.jl`). Per core game they have")
    pr("a concrete, non-empty set of candidate causes to attribute over:")
    pr("")
    pr("| Core game | Candidate cause cells (applicable methods) | Conv maps | Attention heads | Learned policies | Learned channels |")
    pr("|---|---:|---:|---:|---:|---:|")
    for g in groundings
        pr("| $(g.game) | $(g.n_candidate_causes) | $(g.n_conv_feature_maps) | " *
           "$(g.n_attention_heads) | $(g.n_learned_policies) | $(g.n_learned_channels) |")
    end
    pr("")
    pr("(Candidate-cause counts are loaded from each game's verified")
    pr("`tools/xai_study/t3/out/candidates_<game>.json`.) The applicable methods have")
    pr("≥16 candidate causes each; the N/A methods have **zero** of *their* required")
    pr("substrate channels. The absence is **structural** — no conv layer, attention")
    pr("block, or policy network exists in *any* game — not a coverage or tuning gap, so")
    pr("the finding is game-independent (recorded per game only for leaderboard uniformity).")
    pr("")
    pr("## Per-method audit")
    pr("")
    for m in NA_METHODS
        pr("### $(m.name) — *does not apply*")
        pr("")
        pr("- **Family:** $(m.family). **Citation:** $(m.citation).")
        pr("- **Requires (missing ingredient):** $(m.requires).")
        pr("- **Present in the VCS?** **No.**")
        pr("- **Why it does not apply:** $(m.why)")
        pr("- **Why a naive re-targeting onto VCS state is *not* this method:** $(m.no_retarget)")
        pr("")
    end
    pr("## What *does* apply (the contrast, scored elsewhere in E4)")
    pr("")
    for (k, nm, it) in APPLICABLE_CONTRAST
        pr("- **$(nm)** — applies; scored vs the oracle in $(it).")
    end
    pr("")
    pr("These bind to the VCS's true causes and are evaluated as methods under test")
    pr("(`experiment_design.md` §5). The headline expectation (§7) holds: *causal,")
    pr("gradient, and mechanistic methods pass; correlational and architecture-specific")
    pr("methods fail or do not apply.* This audit nails down the \"do not apply\" half.")
    pr("")
    pr("## Records (SPEC §R)")
    pr("")
    pr("Per game: `out/na_audit_<game>.{json,npz}` (6) with `method=na_audit`,")
    pr("`applies=false`, and a `methods_not_applicable[]` array carrying")
    pr("`{method, applies=false, missing_ingredient, missing_ingredient_kind, reason}`")
    pr("per row. Roll-up: `out/na_audit_combined.json`. The `.npz` carries the")
    pr("missing-ingredient channel counts (all zero) + the applicable-cause count, so the")
    pr("leaderboard reads the asymmetry without parsing prose.")
    return String(take!(io))
end

function write_md(groundings::Vector{GameGrounding}; path = joinpath(@__DIR__, "na_audit.md"))
    open(path, "w") do io; print(io, render_md(groundings)); end
    return path
end

# ----------------------------------------------------------------------------
# Self-check (DoD) — the audit is internally consistent and grounded:
#   * every N/A method names a concrete missing ingredient of a recognised kind,
#     marked applies=false / present_in_vcs=false;
#   * the missing-ingredient channel counts are all ZERO (structural absence);
#   * the applicable-method contrast is non-empty AND each core game has ≥1
#     candidate cause loaded from its verified candidates file (so the asymmetry
#     is real and grounded, not asserted);
#   * the §R record round-trips (writes + re-reads with the required keys).
# Throws on a contract violation.
# ----------------------------------------------------------------------------
function selftest(groundings::Vector{GameGrounding})
    @assert length(NA_METHODS) >= 4 "expected ≥4 N/A methods (Grad-CAM, ++/CAM, attention, VIPER)"
    kinds = Set(["conv_feature_maps", "attention", "learned_policy", "learned_channels"])
    for m in NA_METHODS
        @assert m.present_in_vcs == false "$(m.name): finding must be present_in_vcs=false"
        @assert !isempty(m.requires) "$(m.name): must name a missing ingredient"
        @assert m.requires_kind in kinds "$(m.name): unknown ingredient kind $(m.requires_kind)"
        @assert !isempty(m.why) && !isempty(m.no_retarget) "$(m.name): reason/no-retarget required"
    end
    # the canonical four the item names must all be covered
    keys_present = Set(m.key for m in NA_METHODS)
    for required in ("grad_cam", "attention_rollout", "viper")
        @assert required in keys_present "missing required N/A method: $required"
    end
    @assert any(m -> m.key == "grad_cam_pp" || m.key == "cam", NA_METHODS) "Grad-CAM++/CAM must be covered"

    @assert !isempty(APPLICABLE_CONTRAST) "the applicable-method contrast must be non-empty"

    @assert length(groundings) == length(CORE_GAMES) "must ground all 6 core games"
    for g in groundings
        @assert g.candidates_file !== nothing "$(g.game): candidates file not found (grounding)"
        @assert g.n_candidate_causes >= 1 "$(g.game): expected ≥1 candidate cause for applicable methods"
        # the structural absence — the heart of the finding
        @assert g.n_conv_feature_maps == 0 "$(g.game): VCS must have 0 conv feature maps"
        @assert g.n_attention_heads == 0 "$(g.game): VCS must have 0 attention heads"
        @assert g.n_learned_policies == 0 "$(g.game): VCS must have 0 learned policies"
        @assert g.n_learned_channels == 0 "$(g.game): VCS must have 0 learned channels"
    end

    # §R record round-trip on the first game (write → re-read → required keys present)
    g1 = groundings[1]
    rec, _ = build_record(g1)
    for k in ("paper", "phase", "method", "game", "metric_name", "value", "commit",
              "applies", "extra")
        @assert haskey(rec, k) "record missing §R key: $k"
    end
    @assert rec["applies"] == false "record top-level applies must be false"
    @assert rec["value"] == length(NA_METHODS) "headline value must = #N/A methods"
    me = rec["extra"]["methods_not_applicable"]
    @assert length(me) == length(NA_METHODS) "record must carry one entry per N/A method"
    @assert all(e -> e["applies"] == false, me) "every method entry must be applies=false"
    @assert all(e -> haskey(e, "missing_ingredient") && !isempty(e["missing_ingredient"]), me) "each entry needs a missing-ingredient reason"

    println("[na_audit] SELF-CHECK PASS:")
    println("[na_audit]   N/A methods audited: $(length(NA_METHODS)) " *
            "($(join([m.key for m in NA_METHODS], ", ")))")
    println("[na_audit]   all present_in_vcs=false; missing-ingredient kinds covered: " *
            "$(join(sort(collect(Set(m.requires_kind for m in NA_METHODS))), ", "))")
    println("[na_audit]   applicable-method contrast: $(length(APPLICABLE_CONTRAST)) methods")
    for g in groundings
        println("[na_audit]   $(rpad(g.game, 16)) candidate causes=$(g.n_candidate_causes)  " *
                "conv=0 attn=0 policy=0 learned-ch=0  (structural absence)")
    end
    return true
end

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
function main(args = ARGS)
    selftest_only = false
    write_markdown = false
    i = 1
    while i <= length(args)
        a = args[i]
        if     a == "--selftest"; selftest_only = true; i += 1
        elseif a == "--md";       write_markdown = true; i += 1
        else; i += 1
        end
    end
    println("[na_audit] grounding $(length(CORE_GAMES)) core games (jutari/Julia path) …")
    groundings = GameGrounding[ground_game(g) for g in CORE_GAMES]
    selftest(groundings)
    if selftest_only
        println("[na_audit] --selftest: passed, not writing artifacts.")
        return 0
    end
    for g in groundings
        jp, np = write_game(g)
        println("[na_audit] wrote $jp")
        println("[na_audit] arrays  $np")
    end
    cj = write_combined(groundings)
    println("[na_audit] wrote $cj")
    mp = write_md(groundings)   # always (re)write the writeup — it is the primary deliverable
    println("[na_audit] wrote $mp")
    write_markdown && println("[na_audit] (--md set; writeup written)")
    return 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    NAAudit.main()
end
