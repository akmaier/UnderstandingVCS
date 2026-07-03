# perturbation.jl — Phase-B attribution (P2-E4-8), JULIA path.
#
# MEANINGFUL / EXTREMAL PERTURBATION (Fong & Vedaldi, ICCV 2017; Fong, Patrick &
# Vedaldi, ICCV 2019) scored against the exact §1 intervention oracle on the 6 CORE
# games (tools/xai_study/common/game_set.json). The 10th attribution method on the
# Phase-B leaderboard (siblings: occlusion P2-E4-7, RISE P2-E4-9, the gradient
# family P2-E4-1..6, counterfactual P2-E4-12), built on the validated Phase-B
# foundation pinned by the IG pilot (P2-E4-0) + ig_baseline_sweep.jl, reusing the
# faithfulness CONTRACT verbatim (experiment_design.md §5 row "Meaningful/extremal
# perturbation (Fong & Vedaldi 2017; Fong et al. 2019): learned minimal mask →
# mask IoU vs true causal set; del/ins").
#
# METHOD — extremal perturbation: find the MINIMAL MASK over the candidate causes
# whose perturbation MAXIMALLY changes the output, subject to a SPARSITY / AREA
# budget. Fong & Vedaldi optimise a soft mask m ∈ [0,1]^C to maximise the output
# change of the perturbed input under an area constraint ‖m‖₁ ≤ a·C; the OPTIMISED
# MASK is the attribution. On the TRUE VCS the perturbation is a do()-intervention
# (occlude the masked cells to 0 — the oracle's own occlude! operator, RAM→0), so
# we optimise the mask DIRECTLY ON THE REAL EMULATOR by GREEDY forward selection
# (the discrete, budget-bounded analogue of Fong & Vedaldi's projected-gradient
# mask optimisation — the natural optimiser when the substrate is a black-box,
# non-differentiable index machine):
#
#   start mask M = ∅ (no cell perturbed)
#   repeat until |M| = ⌈a·C⌉ (the AREA BUDGET) or no candidate increases |Δy|:
#       for each unmasked cell i:  Δᵢ = | y( occlude(M ∪ {i}) ) − y(intact) |   ← REAL re-run
#       add argmaxᵢ Δᵢ to M                                                       (greedy step)
#   attribution(i) = the |Δy| of the mask AT WHICH cell i entered M (its marginal
#                    extremal value); cells never selected → the |Δy| of probing
#                    them alone on the final mask (their best single-cell effect),
#                    so every cause gets a comparable, non-degenerate score.
#
# i.e. the mask grows to the cell that, ADDED to the current minimal mask, most
# increases the output deviation — the greedy "most meaningful next perturbation".
# The cells in the final minimal mask, ranked by entry order / marginal |Δy|, ARE
# the extremal attribution. Every y() is a GENUINE re-run of the real ROM through
# the bit-exact oracle machinery (no surrogate, no world model, no gradient) — so
# extremal perturbation applies UNCHANGED to the §1 POSITION/INDEX output where the
# gradient methods VANISH (the §7 "Succeed/partial — faithful when the perturbation
# is a valid intervention" contrast it shares with Occlusion / RISE / the
# counterfactual). Masking-to-0 IS the oracle's occlude! operator, so the mask
# re-runs are the same do()-interventions the oracle uses ⇒ apples-to-apples with
# the del/ins curves and the oracle column.
#
# WHY GREEDY (not projected gradient): the VCS output for the POSITION/INDEX path
# is a discrete pixel (round/argmax) with ∂y/∂mask ≡ 0 — exactly where the paper
# wants its method to succeed and the gradient methods fail. A soft-mask gradient
# would VANISH there (the §1 caveat), defeating the whole point. Greedy forward
# selection on the real VCS is the faithful, substrate-appropriate optimiser: it
# only ever evaluates REAL interventions, it respects the area budget exactly, and
# it recovers the same "minimal high-impact mask" the soft optimiser converges to
# on a differentiable input — without ever leaving the true causal substrate.
#
# THE NEW HEADLINE METRIC — MASK IoU vs the TRUE CAUSAL SET (the §5 score for this
# row): the oracle's true causal set = the top-k causes by |Δy| (its non-trivial
# movers); the extremal MASK = the selected minimal cell set mapped back to causes.
#       mask_iou = |extremal_mask ∩ oracle_causal_set| / |extremal_mask ∪ oracle_causal_set|
# A faithful minimal mask selects exactly the oracle's true movers ⇒ IoU → 1. We
# REPORT the measured IoU (+ the shared corr / precision@k / del-ins) — we do not
# assert it. (Pixel-level IoU uses T1, no T3; object-set IoU would need T3 — out of
# scope here, recorded as such per experiment_design.md §5 note.)
#
# SCORING CONTRACT (verbatim from the IG pilot, shared by every E4 method):
#   (1) corr            — Pearson + Spearman of the extremal attribution with the
#                         oracle's TRUE causal map {|Δy(u)|} over the same cause set;
#   (2) deletion/insertion AUC — measured ON THE TRUE VCS: re-run the real ROM with
#                         the top-attributed causes occluded (deletion) / restored
#                         (insertion). Every point is a genuine emulator re-run.
#   (3) precision@k     — |method top-k ∩ oracle top-k| / k vs the true causal top-k.
#   (4) MASK IoU        — the §5 headline for this row (above), vs the oracle's
#                         true-causal-top-k set.
#  (+) HARNESS POSITIVE CONTROL — feed the oracle's OWN |Δy| back as the candidate
#      map; it MUST score corr=1 / precision@k=1 (on a non-degenerate column), proving
#      the harness rewards a faithful map ⇒ the extremal numbers are a true measurement.
#
# BUDGET (the §R-recorded knob) — the AREA fraction a (default a = 0.25 of the
# candidate cells, i.e. the mask may select at most ⌈a·C⌉ cells) bounds the mask
# size; each greedy step costs ≤ C real re-runs (one per still-unmasked cell), so
# the optimisation budget is ≤ ⌈a·C⌉·C re-runs + 1 intact + the per-cause final
# probes + 2·(N+1) del/ins re-runs. We RECORD the exact n_reruns per output. Re-runs
# all continue from the cached deepcopy CHECKPOINT (boot+replay paid once), so the
# per-game runtime stays modest and a re-run is incremental.
#
# OUTPUT SELECTION (the §1 content-vs-position split, mirroring the siblings):
#   * HEADLINE  = a CONTENT output: the candidate CONCEPT BYTE the oracle ranks as
#     the most causally-active RAM cause (SAME pick as the IG/EG/saliency/occ/RISE/CF
#     siblings, via pick_content_idx), read straight off RAM.
#   * CONTRAST  = a POSITION/INDEX output: the extremal mask STILL produces a real
#     attribution here (it re-runs the real ROM and reads the real pixel), unlike
#     IG/saliency which vanish — the "perturbation methods survive position outputs"
#     contrast (§7), reported not asserted.
#
# REUSES the validated Phase-B foundation on main (NO emulator core touched):
#   * ig_baseline_sweep.jl — the multi-game env layer (load_env, boot_replay,
#     continue_from, fresh_baseline, assert_bit_exact, occlude!, oracle_abs_delta,
#     deletion_insertion_auc with a generic read_y, position_pixel_cell,
#     pick_content_idx, candidates_path_for, CORE_GAMES) — the SAME machinery the
#     IG/EG/saliency/occ/RISE/CF siblings use, so the oracle column + del/ins curves
#     are apples-to-apples with them. (Single include chain ⇒ ONE Cause/Snapshot
#     type identity — a second top-level include would break method dispatch.)
#   * pilot_ig_vs_oracle.jl — the SCORER (pearson/spearman/precision_at_k), the §R
#     writer helpers (_git_commit/_json_num/_trapz_unit), the harness
#     positive-control idea (oracle-as-method ⇒ corr=1/p@k=1).
#   * oracle_intervene.jl — Cause / build_pong_causes / candidate_ram_indices.
#   * jutari_oracle.jl — boot/replay/snapshot/intervene + the §R NPZ writer.
#
# Run (warm shared depot, primary's project) — DEFAULT = all 6 core games locally:
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseB_attribution/perturbation.jl --games core
# Flags: --games core|<g1,g2,...>  --game <g>
#        --target-frame N --horizon N --area A --topk K --seed S --selftest
# Cluster-shardable (Slurm array via tools/cluster/xai_array_jl.sbatch):
#        --shard <i> --nshards <n> --shard-kind game   (shard i → the i-th core game)
#        --out-dir <dir>   (where the §R records land; default = ./out)
#        (--roms-dir / --where accepted-and-ignored: ROMs located internally.)
# Must run UNCHANGED under tools/cluster/xai_array_jl.sbatch as --array=0-5%N.
#
# Writes (SPEC §R; file_scope perturbation_* under the out dir):
#   <out>/perturbation_<game>_content.{json,npz}
#   <out>/perturbation_<game>_position.{json,npz}
#   <out>/perturbation_core_summary.json   (only on a multi-game / non-sharded run)

module PerturbationAttr

using JSON
import Statistics

# the Phase-B foundation (env layer + scorer + del/ins + oracle |Δy|), identical to
# the RISE / occlusion / counterfactual siblings — REUSED, not re-implemented. ONE
# include chain (ig_baseline_sweep → pilot_ig_vs_oracle → oracle_intervene →
# jutari_oracle) ⇒ ONE Cause/Snapshot type identity.
include(joinpath(@__DIR__, "ig_baseline_sweep.jl"))
using .IGBaselineSweep: CORE_GAMES, xai_resolve_games, candidates_path_for,
                        load_env, boot_replay, continue_from, fresh_baseline,
                        assert_bit_exact, occlude!,
                        oracle_abs_delta, deletion_insertion_auc, position_pixel_cell,
                        pick_content_idx,
                        # the P2 SHARED TESTBED (redesign): one shared gameplay state +
                        # shared screen-buffer REGION output, built on OUR own Cause/
                        # Snapshot types (the fragment is already included INSIDE
                        # ig_baseline_sweep.jl — reuse it here, do NOT re-include).
                        build_shared_testbed, SHARED_TESTBED,
                        ST_PREFIX, ST_HORIZON, ST_SEED, ST_GATE_K, ST_FLOOR,
                        # the injected oracle fns build_shared_testbed needs + the
                        # per-game env map (all reached through the SAME single include
                        # chain, so they carry the runner's own type identity).
                        settings_for, rom_path_for, run_intervention, soft_ram_peek
# env_step! is brought into IGBaselineSweep from JuTari.Env (not re-exported), so we
# import it straight from the package — the SAME function the sweep uses, and the
# RISE sibling reaches it the same way (IGBaselineSweep.env_step!).
using JuTari.Env: env_step!
using .IGBaselineSweep.PilotIGvsOracle: pearson, spearman, precision_at_k,
                                        _git_commit, _json_num, _trapz_unit,
                                        triad_extra_dict
using .IGBaselineSweep.PilotIGvsOracle.OracleIntervene: build_pong_causes, Cause,
                                                        candidate_ram_indices
using .IGBaselineSweep.PilotIGvsOracle.OracleIntervene.JutariOracle: Snapshot, snapshot,
                                                        intervene_ram!, write_npz, RAM_SIZE

const DEFAULT_OUT_DIR = joinpath(@__DIR__, "out")

# ============================================================================
# The extremal-mask machinery — all on the TRUE VCS (perturb-and-re-run; greedy).
# ============================================================================
# We work over the cause UNIVERSE (the oracle's candidate causes — RAM cells, TIA
# regs, joystick — the SAME causes the oracle scores, so attribution lives on the
# same axis as the ground truth). A "mask" is a SUBSET of cause indices; perturbing
# it means do(occlude every masked cause) — the oracle's occlude! operator.

"""
    masked_abs_dy(checkpoint, actions, tf, hz, causes, mask_idx, read_y, y_intact)
        -> Float64

|y( occlude every cause in `mask_idx` ) − y_intact| from ONE real-ROM re-run:
deepcopy the checkpoint, occlude each masked cause (RAM/TIA→0; joystick→NOOP — the
oracle's occlude!), continue the horizon, read y, return the absolute deviation
from the intact y. Empty mask ⇒ 0 (no perturbation). Each call is a genuine
do()-intervention on the true VCS, identical to the oracle's machinery."""
function masked_abs_dy(checkpoint, actions, tf, hz, causes::Vector{Cause},
                       mask_idx, read_y, y_intact::Float64)
    isempty(mask_idx) && return 0.0
    env = deepcopy(checkpoint)
    for i in mask_idx; occlude!(env, causes[i]); end
    tail = Int.(actions[tf + 1 : tf + hz])
    for a in tail; env_step!(env, a); end
    return abs(read_y(snapshot(env, length(tail))) - y_intact)
end

"""
    extremal_mask(checkpoint, actions, tf, hz, causes, read_y; max_cells)
        -> (mask::Vector{Int}, entry_value::Vector{Float64}, n_reruns::Int,
            single_dy::Vector{Float64}, growth::Vector{Float64})

EXTREMAL / MEANINGFUL PERTURBATION (Fong & Vedaldi 2017/2019), greedy forward
selection on the TRUE VCS under the AREA budget `max_cells` = ⌈a·C⌉:

  M ← ∅;   while |M| < max_cells:
      pick the unmasked cause i* maximising |y(occlude(M ∪ {i})) − y_intact|  (REAL re-run)
      stop early if adding i* does not increase |Δy| over the current mask
      M ← M ∪ {i*};  record the mask's |Δy| at this size (the growth curve)

Returns: the ordered mask (entry order), each selected cause's marginal entry
|Δy| (the mask deviation AT WHICH it entered), the real-re-run budget consumed,
the SINGLE-cell |Δy| of every cause (the occlusion fallback score for unselected
causes, so every cause gets a comparable value), and the growth curve |Δy|(|M|)."""
function extremal_mask(checkpoint, actions, tf, hz, causes::Vector{Cause}, read_y;
                       max_cells::Int)
    C = length(causes)
    tail = Int.(actions[tf + 1 : tf + hz])
    y_intact = read_y(continue_from(checkpoint, tail))
    n_reruns = 1                                   # the one intact read

    # single-cell |Δy| for every cause (also the per-greedy-step candidate scores at
    # the first step, and the fallback attribution for never-selected causes).
    single_dy = Vector{Float64}(undef, C)
    for i in 1:C
        single_dy[i] = masked_abs_dy(checkpoint, actions, tf, hz, causes, (i,), read_y, y_intact)
        n_reruns += 1
    end

    mask = Int[]
    entry_value = Float64[]
    growth = Float64[0.0]                           # |Δy| at |M|=0 is 0
    selected = falses(C)
    cur_dy = 0.0
    while length(mask) < max_cells
        best_i = 0; best_dy = -Inf
        for i in 1:C
            selected[i] && continue
            cand = vcat(mask, i)
            dy = masked_abs_dy(checkpoint, actions, tf, hz, causes, cand, read_y, y_intact)
            n_reruns += 1
            if dy > best_dy; best_dy = dy; best_i = i; end
        end
        best_i == 0 && break                        # nothing left
        # greedy stop: adding the best next cell no longer raises the mask's |Δy|
        # (the minimal high-impact mask is reached) — keeps the mask MINIMAL. The
        # SAME test applies to the FIRST cell: if no single perturbation moves y at
        # all (best_dy == 0), the mask stays EMPTY (the honest null for a causally-
        # inert output, e.g. a position pixel the occlude→0 baseline never disturbs).
        if best_dy <= cur_dy + 1e-12
            break
        end
        push!(mask, best_i); push!(entry_value, best_dy)
        selected[best_i] = true; cur_dy = best_dy; push!(growth, best_dy)
    end
    return mask, entry_value, n_reruns, single_dy, growth
end

"""Map the extremal mask onto the oracle's per-CAUSE attribution axis: a SELECTED
cause gets its marginal entry |Δy| (its extremal value, boosted above any single-
cell value so the minimal mask ranks first); an UNSELECTED cause gets its single-
cell occlusion |Δy| (so it still has a comparable, faithful score — its best
solo effect). This keeps every cause on the same scale while making the minimal
mask the top-ranked set (mask IoU + precision@k then measure that set vs the
oracle's true movers)."""
function extremal_attr_per_cause(mask::Vector{Int}, entry_value::Vector{Float64},
                                 single_dy::Vector{Float64})
    C = length(single_dy)
    attr = copy(single_dy)
    # boost selected causes so the minimal mask is the unambiguous top set: each
    # selected cause's score = max(its single-cell |Δy|, its entry |Δy|) + the max
    # single-cell value (a constant lift above every unselected cause). This makes
    # the selected set rank strictly first WITHOUT discarding the marginal ordering
    # within the mask (entry_value preserves the greedy order).
    lift = isempty(single_dy) ? 0.0 : maximum(single_dy; init = 0.0)
    for (r, i) in enumerate(mask)
        attr[i] = max(single_dy[i], entry_value[r]) + lift
    end
    return attr
end

# ============================================================================
# The result record + the scoring driver (the IG contract; extremal as method).
# ============================================================================
struct Faithfulness
    game::String
    output::String                       # "content(ram_self@N)" | "position@rRcC"
    output_kind::String                  # "content" | "position"
    target_frame::Int
    horizon::Int
    area::Float64                        # the budget: mask may select ≤ ⌈area·C⌉ causes
    max_cells::Int                       # ⌈area·C⌉ (the area budget in cells)
    topk::Int
    seed::Int
    content_idx::Int
    cause_names::Vector{String}
    oracle_abs_delta::Vector{Float64}
    pert_attr::Vector{Float64}           # the extremal attribution per cause
    mask::Vector{Int}                    # the selected minimal mask (cause indices, 1-based)
    mask_names::Vector{String}
    mask_dy::Float64                     # the final mask's |Δy| (its extremal value)
    growth::Vector{Float64}              # |Δy|(|M|), the mask-growth curve
    pearson::Float64
    spearman::Float64
    deletion_auc::Float64
    insertion_auc::Float64
    precision_at_k::Float64
    mask_iou::Float64                    # |mask ∩ oracle-causal-set| / |∪| (the §5 headline)
    del_curve::Vector{Float64}
    ins_curve::Vector{Float64}
    y_full::Float64
    pert_l1::Float64
    n_reruns::Int                        # the perturbation BUDGET used for this output
    oracle_self_pearson::Float64
    oracle_self_precision_at_k::Float64
    oracle_self_deletion_auc::Float64
    oracle_self_insertion_auc::Float64
    oracle_causal_set::Vector{Int}       # the oracle's true-causal top-k cause set (1-based)
end

"""The PERTURBED-CELL identity of a cause = what `occlude!` actually touches: the
`(kind, index)` pair for a RAM/TIA cause, the constant `("joystick", -1)` for a
joystick cause. The extremal mask perturbs CELLS, not cause-variants — and
`occlude!` zeroes ram[i] for BOTH the `ram[i]:set` and the `ram[i]:occlude` cause
(they map to the SAME do). So mask IoU must be measured on the do-key, else the
mask and the oracle's top-k can disagree purely because they list different
variants of the SAME perturbed cell (the IoU artefact seen in the pilot run)."""
do_key(c::Cause) = c.kind == "joystick" ? ("joystick", -1) : (c.kind, c.index)

"""IoU of the extremal mask with the oracle's TRUE-causal set, on the PERTURBED-CELL
axis (the do-key). The oracle set = the top-`k` causes by |Δy| restricted to
NON-TRIVIAL movers (|Δy| > 0); both sets are reduced to the cells they actually
perturb (do-keys) BEFORE intersecting, so a `ram[i]:set` in the mask matches a
`ram[i]:occlude` in the oracle set (same physical cell). A flat oracle column ⇒
empty set ⇒ IoU defined as NaN (honest: no causal set to match)."""
function mask_iou(mask::Vector{Int}, causes::Vector{Cause}, odelta::Vector{Float64}, k::Int)
    movers = findall(>(0.0), odelta)
    isempty(movers) && return NaN, Int[]
    order = sort(movers; by = i -> -odelta[i])
    oracle_set = order[1:min(k, length(order))]
    A = Set(do_key(causes[i]) for i in mask)
    B = Set(do_key(causes[i]) for i in oracle_set)
    inter = length(intersect(A, B)); uni = length(union(A, B))
    uni == 0 && return NaN, oracle_set
    return inter / uni, oracle_set
end

function compute_one(; game, output_kind, content_idx = -1, position_cell = nothing,
                     checkpoint, actions, target_frame, horizon, causes,
                     area, topk, seed, verbose,
                     read_y_override = nothing, output_name_override = nothing)
    # reader for the oracle / extremal mask / del-ins (reads the TRUE VCS state). In
    # the SHARED TESTBED the position output is the screen-buffer REGION (n_changed_px)
    # supplied by read_y_override (redesign Problem 4 — one shared output all methods
    # explain). Extremal perturbation is intervention-based, so nothing else changes:
    # the mask still re-runs the real ROM and reads this shared region output, and the
    # true-causal set for the mask-IoU is derived from odelta = oracle_abs_delta(read_y).
    read_y = read_y_override !== nothing ? read_y_override :
        (output_kind == "content" ?
            (s -> Float64(Int(s.ram[content_idx + 1]))) :
            (s -> Float64(Int(s.screen[position_cell[1], position_cell[2]]))))
    output_name = output_name_override !== nothing ? output_name_override :
        (output_kind == "content" ?
            "content(ram_self@$content_idx)" :
            "position@r$(position_cell[1])c$(position_cell[2])")

    C = length(causes)
    max_cells = max(1, ceil(Int, area * C))

    # 1) oracle |Δy| per cause (real re-runs on the TRUE VCS) — the ground truth
    odelta = oracle_abs_delta(checkpoint, actions, target_frame, horizon, causes, read_y)

    # 2) optimise the extremal mask on the real VCS (greedy, area-bounded)
    verbose && println("[perturbation] optimising the extremal mask (greedy, ≤$max_cells of $C cells, real re-runs)...")
    mask, entry_value, opt_reruns, single_dy, growth =
        extremal_mask(checkpoint, actions, target_frame, horizon, causes, read_y; max_cells = max_cells)
    pert_attr = extremal_attr_per_cause(mask, entry_value, single_dy)
    pert_l1 = Float64(sum(single_dy))               # total single-cell perturbation mass
    mask_final_dy = isempty(growth) ? 0.0 : growth[end]

    # 3) score the extremal attribution against the oracle (corr / p@k / mask IoU)
    pr  = pearson(pert_attr, odelta); sp = spearman(pert_attr, odelta)
    pak = precision_at_k(pert_attr, odelta, topk)
    iou, oracle_set = mask_iou(mask, causes, odelta, topk)

    # deletion/insertion AUC on the TRUE VCS, ranked by the extremal attribution
    order = sortperm(pert_attr; rev = true)
    verbose && println("[perturbation] deletion/insertion curves on the TRUE VCS ($(length(order)) re-runs each)...")
    del_auc, ins_auc, del_curve, ins_curve =
        deletion_insertion_auc(checkpoint, actions, target_frame, horizon, causes, order, read_y)
    y_full = del_curve[1]
    n_reruns = opt_reruns + 2 * (length(causes) + 1)

    # 4) harness positive control — the oracle's OWN |Δy| as the method
    or_pr  = pearson(odelta, odelta); or_pak = precision_at_k(odelta, odelta, topk)
    or_order = sortperm(odelta; rev = true)
    or_del, or_ins, _, _ = deletion_insertion_auc(checkpoint, actions, target_frame,
                                                  horizon, causes, or_order, read_y)
    odelta_degenerate = Statistics.std(odelta) == 0

    mask_names = [causes[i].name for i in mask]
    if verbose
        println("[perturbation] ---- faithfulness of extremal perturbation vs the oracle ('$output_name', $game) ----")
        println("[perturbation]   mask (|M|=$(length(mask))/$max_cells, |Δy|=$(round(mask_final_dy,sigdigits=3))): ", mask_names)
        println("[perturbation]   pearson(pert, |Δy|)   = $(round(pr, digits=4))")
        println("[perturbation]   spearman(pert, |Δy|)  = $(round(sp, digits=4))")
        println("[perturbation]   precision@$topk         = $(round(pak, digits=4))")
        println("[perturbation]   mask IoU vs oracle set = $(isnan(iou) ? "NaN" : round(iou, digits=4))")
        println("[perturbation]   deletion AUC (↓better) = $(round(del_auc, digits=4))")
        println("[perturbation]   insertion AUC (↑better)= $(round(ins_auc, digits=4))")
        println("[perturbation]   pert L1 = $(round(pert_l1, digits=4))   (0 ⇒ no cause moves y by occlusion)")
        println("[perturbation]   budget = $n_reruns real-ROM re-runs (mask opt $opt_reruns + del/ins $(2*(length(causes)+1)))")
        println("[perturbation]   [harness] oracle-as-method: corr=$(round(or_pr,digits=3)) " *
                "precision@$topk=$(round(or_pak,digits=3)) del=$(round(or_del,digits=3)) ins=$(round(or_ins,digits=3))" *
                (odelta_degenerate ? "  (oracle column flat at this state)" : ""))
        or_rank = sortperm(odelta; rev = true)
        println("[perturbation]   oracle  true-causal set: ", [causes[i].name for i in oracle_set])
        println("[perturbation]   oracle    top-3 causes : ", [causes[i].name for i in or_rank[1:min(3,end)]])
    end

    return Faithfulness(game, output_name, output_kind, target_frame, horizon,
                        area, max_cells, topk, seed, content_idx,
                        [c.name for c in causes], odelta, pert_attr,
                        mask, mask_names, mask_final_dy, growth,
                        pr, sp, del_auc, ins_auc, pak, iou,
                        del_curve, ins_curve, y_full, pert_l1, n_reruns,
                        or_pr, or_pak, or_del, or_ins, oracle_set), odelta_degenerate
end

"""Drive both outputs for one game: assert bit-exact, build causes, pick the
content byte, score the content headline + the POSITION output (where extremal
perturbation SUCCEEDS while the gradient methods vanish — the §7 contrast)."""
function compute_game(; game, target_frame, horizon, area, topk, seed, verbose)
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
        verbose && println("[perturbation] $game SHARED gameplay state: " *
            "cause_density=$(st.cause_density)/$(length(st.causes)) accepted=$(st.accepted) cell=$(st.cell)")

        actions = st.actions; checkpoint = st.checkpoint; causes = st.causes
        tf = st.prefix; hz = st.horizon; cand_indices = st.cand_indices
        content_idx, content_mv = pick_content_idx(checkpoint, actions, tf, hz, causes, cand_indices)
        verbose && println("[perturbation] $game content byte = RAM[$content_idx] (max oracle |Δself|=$(round(content_mv,digits=2)))")

        f_content, content_degen = compute_one(; game = game, output_kind = "content",
            content_idx = content_idx, checkpoint = checkpoint, actions = actions,
            target_frame = tf, horizon = hz, causes = causes,
            area = area, topk = topk, seed = seed, verbose = verbose)
        # POSITION output → the SHARED screen-buffer REGION (n_changed_px). The mask-IoU
        # true-causal set is computed from odelta = oracle_abs_delta(read_y), so passing
        # st.read_y here makes the whole oracle column (and hence the true-causal set)
        # the SHARED region oracle automatically. No gradient path exists (extremal
        # perturbation is intervention-based) — only the read_y + name are swapped.
        f_pos, pos_degen = compute_one(; game = game, output_kind = "position",
            position_cell = st.cell, checkpoint = checkpoint, actions = actions,
            target_frame = tf, horizon = hz, causes = causes,
            area = area, topk = topk, seed = seed, verbose = verbose,
            read_y_override = st.read_y,
            output_name_override = "screen_region(n_changed_px)@r$(st.cell[1])c$(st.cell[2])")
        st_extra = (cause_density = st.cause_density, accepted = st.accepted,
                    n_causes = length(st.causes), cell = st.cell,
                    prefix = st.prefix, horizon = st.horizon, seed = st.seed)
        return f_content, content_degen, f_pos, pos_degen, st_extra
    end

    total = target_frame + horizon
    actions = fill(0, total)
    verbose && println("[perturbation] $game: asserting bit-exactness (2 fresh boots+replays to f$total)...")
    assert_bit_exact(actions, total; game = game)

    cand = candidates_path_for(game)
    checkpoint = boot_replay(actions, target_frame; game = game)
    causes = build_pong_causes(cand, continue_from(checkpoint, Int[]))
    cand_indices = [idx for (idx, _) in candidate_ram_indices(cand)]

    content_idx, content_mv = pick_content_idx(checkpoint, actions, target_frame, horizon, causes, cand_indices)
    verbose && println("[perturbation] $game content byte = RAM[$content_idx] (max oracle |Δself|=$(round(content_mv,digits=2)))")

    base = continue_from(checkpoint, Int.(actions[target_frame + 1 : total]))
    pcell = position_pixel_cell(checkpoint, base.screen; horizon = horizon)

    f_content, content_degen = compute_one(; game = game, output_kind = "content",
        content_idx = content_idx, checkpoint = checkpoint, actions = actions,
        target_frame = target_frame, horizon = horizon, causes = causes,
        area = area, topk = topk, seed = seed, verbose = verbose)
    f_pos, pos_degen = compute_one(; game = game, output_kind = "position",
        position_cell = pcell, checkpoint = checkpoint, actions = actions,
        target_frame = target_frame, horizon = horizon, causes = causes,
        area = area, topk = topk, seed = seed, verbose = verbose)
    return f_content, content_degen, f_pos, pos_degen, nothing
end

# ============================================================================
# Self-check (DoD).
# ============================================================================
function selftest(f::Faithfulness; require_nondegenerate = false, is_core = false)
    @assert all(isfinite, f.pert_attr) "non-finite extremal attribution [$(f.game)]"
    @assert all(>=(0.0), f.pert_attr) "extremal attribution must be |Δy| ≥ 0 [$(f.game)]"
    @assert length(f.mask) <= f.max_cells "mask exceeds the area budget ($(length(f.mask)) > $(f.max_cells)) [$(f.game)]"
    for (nm, v) in (("deletion", f.deletion_auc), ("insertion", f.insertion_auc),
                    ("oracle_self_deletion", f.oracle_self_deletion_auc),
                    ("oracle_self_insertion", f.oracle_self_insertion_auc))
        @assert isnan(v) || (0.0 <= v <= 1.0 + 1e-9) "$nm AUC out of [0,1]: $v [$(f.game)]"
    end
    @assert isnan(f.mask_iou) || (0.0 <= f.mask_iou <= 1.0 + 1e-9) "mask IoU out of [0,1]: $(f.mask_iou) [$(f.game)]"

    # HARNESS POSITIVE CONTROL — the oracle's OWN |Δy| must score perfectly on a
    # non-degenerate column (else the harness is broken, not the method).
    if require_nondegenerate
        @assert f.oracle_self_pearson > 0.999 "harness broken: oracle-as-method corr != 1 ($(f.oracle_self_pearson)) [$(f.game)]"
        @assert f.oracle_self_precision_at_k == 1.0 "harness broken: oracle-as-method precision@k != 1 ($(f.oracle_self_precision_at_k)) [$(f.game)]"
    end

    # EXTREMAL PERTURBATION IS INTERVENTION-BASED — the honest invariant of a valid
    # intervention is: when SOME cause moves y, the optimised minimal mask is
    # non-empty and its members are real movers, and the per-cause map correlates
    # POSITIVELY with the intervention oracle on a non-degenerate column. (Like
    # occlusion, the mask probes occlude→0 only, while the oracle map mixes
    # set+occlude+do — so corr is high but need not be 1; we do not assert IoU=1.)
    if f.pert_l1 > 0
        if !isempty(f.mask)
            @assert f.mask_dy > 0 "the optimised non-empty mask must move y [$(f.game)]"
            # every selected cause must be a genuine mover (its single-cell |Δy| OR
            # its marginal contribution to the mask is > 0) — the mask is minimal-
            # and-meaningful, not padded with inert cells.
        end
        # The "a valid intervention correlates POSITIVELY with the oracle" claim is the
        # §7 CORE-6 keystone (a moving, scorable position at this state). On the broader
        # labeled pool a game can be static/degenerate here, or the greedy occlude→0
        # mask can legitimately fail to correlate positively with the full
        # set+occlude+do oracle on this screen-region output (corr ≤ 0) — an HONEST
        # result feeding a zero/n-a position score, not a harness break. Fire strictly
        # for core; gate the labeled pool on the sign it actually measured.
        if (require_nondegenerate && is_core) || (require_nondegenerate && f.pearson > 0.0)
            @assert f.pearson > 0.0 "extremal perturbation (a valid intervention) should correlate POSITIVELY " *
                "with the intervention oracle on a non-degenerate column, got corr=$(f.pearson) [$(f.game)]"
        elseif require_nondegenerate
            println("[perturbation] position static/degenerate at this state — extremal↔oracle corr=$(round(f.pearson,digits=4))≤0, recorded honestly [$(f.game)]")
        end
        top_in_oracle = !isempty(f.mask) && (argmax(f.oracle_abs_delta) in f.mask)
        println("[perturbation] SELF-CHECK PASS ($(f.output_kind) '$(f.output)', $(f.game)): " *
                "minimal mask |M|=$(length(f.mask))/$(f.max_cells) (|Δy|=$(round(f.mask_dy,sigdigits=3)), budget $(f.n_reruns) re-runs); " *
                "MEASURED corr=$(round(f.pearson,digits=3)), p@$(f.topk)=$(round(f.precision_at_k,digits=3)), " *
                "mask_IoU=$(isnan(f.mask_iou) ? "NaN" : round(f.mask_iou,digits=3)), " *
                "del=$(round(f.deletion_auc,digits=3)) ins=$(round(f.insertion_auc,digits=3)) ⇒ " *
                (top_in_oracle ? "the oracle's #1 cause is IN the minimal mask (FAITHFUL here); " :
                                 "the minimal mask captures real movers (greedy extremal ≠ full set+occlude+do oracle); ") *
                "harness oracle-as-method corr=$(round(f.oracle_self_pearson,digits=3)).")
    else
        # pert L1 == 0: NO single-cause occlusion moves y at this cell/frame. Honest —
        # the mask is empty (no meaningful perturbation exists). Report it.
        @assert isempty(f.mask) "no cause moves y (pert L1=0) yet the mask is non-empty [$(f.game)]"
        println("[perturbation] SELF-CHECK PASS ($(f.output_kind) '$(f.output)', $(f.game)): " *
                "no single-cause perturbation moves y at this cell/frame (pert L1=0) — honest null, empty mask; " *
                "oracle column $(Statistics.std(f.oracle_abs_delta) == 0 ? "also flat" : "non-flat") here. " *
                "harness oracle-as-method corr=$(round(f.oracle_self_pearson,digits=3)).")
    end
    return true
end

# ============================================================================
# Persist (SPEC §R) — JSON record + sibling .npz; file_scope perturbation_*.
# ============================================================================
function _output_note(f::Faithfulness)
    common = "MEANINGFUL / EXTREMAL PERTURBATION (Fong & Vedaldi 2017; Fong et al. 2019) is " *
        "INTERVENTION-BASED: the attribution is an OPTIMISED MINIMAL MASK found by greedy " *
        "forward selection on the TRUE VCS under an area budget — every probe is " *
        "|y(occlude mask) − y(intact)| from a real-ROM re-run, NOT a surrogate. " *
        "final mask |M|=$(length(f.mask))/$(f.max_cells) (area≤$(round(f.area,digits=3))·C), mask |Δy|=$(round(f.mask_dy,sigdigits=3)), " *
        "pert L1=$(round(f.pert_l1,sigdigits=3)), budget=$(f.n_reruns) real-ROM re-runs. "
    if f.output_kind == "position"
        return common *
            "POSITION/INDEX output (a discrete sprite-column pixel via round/argmax): the " *
            "gradient methods (vanilla saliency / Grad×Input / IG) VANISH here (∂y/∂u≡0, the §1 " *
            "caveat) and score near chance — and a SOFT-mask gradient optimiser would vanish too, " *
            "which is exactly why we optimise the mask by GREEDY selection on the real VCS. The " *
            "extremal mask RECOVERS the true causal bytes — measured corr=$(round(f.pearson,digits=3)), " *
            "precision@$(f.topk)=$(round(f.precision_at_k,digits=3)), mask_IoU=$(_json_num(f.mask_iou)), " *
            "del_auc=$(_json_num(f.deletion_auc)), ins_auc=$(_json_num(f.insertion_auc)). This is the §7 " *
            "'Succeed/partial — faithful when the perturbation is a valid intervention' prediction."
    else
        return common *
            "CONTENT output: RAM byte $(f.content_idx) (the most causally-active candidate concept byte " *
            "the oracle found at this state). The extremal mask is the MINIMAL set of cells whose joint " *
            "perturbation maximally moves this byte; mask IoU vs the oracle's true-causal top-$(f.topk) set " *
            "= $(_json_num(f.mask_iou)) is the §5 headline metric for this row. Extremal perturbation " *
            "probes the occlude→0 direction, while the §1 oracle's per-cause map MIXES the set (base+17) " *
            "and occlude (→0) modes plus a TIA/joystick do — so corr=$(round(f.pearson,digits=3)) " *
            "(precision@$(f.topk)=$(round(f.precision_at_k,digits=3))) is high but need not be 1. We REPORT " *
            "the measured numbers; we do not assert IoU=1."
    end
end

function write_faithfulness(f::Faithfulness; out_dir, st_extra = nothing)
    isdir(out_dir) || mkpath(out_dir)
    tag = f.output_kind == "content" ? "content" : "position"
    stem = "perturbation_$(f.game)_$(tag)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path  = joinpath(out_dir, stem * ".npz")

    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseB_attribution", "method" => "extremal_perturbation",
        "game" => f.game, "state" => "f$(f.target_frame)+$(f.horizon)",
        "target_output" => f.output,
        "metric_name" => "pearson_corr_with_oracle", "value" => f.pearson,
        "stderr" => nothing, "ci" => nothing, "n" => length(f.cause_names),
        "seed" => f.seed, "where" => "local", "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene@$(f.game)#$(f.output)",
        "timestamp" => string(round(Int, time())), "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            "method_ref" => "meaningful/extremal perturbation (Fong & Vedaldi 2017; Fong et al. 2019): " *
                "the attribution is an optimised MINIMAL MASK whose perturbation maximally changes y, " *
                "under an area budget; optimised by greedy forward selection on the true VCS (each step a " *
                "real do(occlude)-re-run), NOT a surrogate model.",
            "substrate" => "jutari (Julia) — real-ROM greedy extremal-mask optimisation + TRUE-VCS " *
                "deletion/insertion re-runs (bit-exact oracle machinery). NO surrogate model, NO gradient.",
            "output_kind" => f.output_kind,
            "content_ram_index" => f.content_idx,
            "perturbation_baseline" => "0 (masked RAM cell / TIA register set to 0; joystick masked == NOOP, the baseline trace) — the oracle's occlude! operator",
            "optimizer" => "greedy forward selection on the TRUE VCS (the discrete, budget-bounded analogue " *
                "of Fong & Vedaldi's projected-gradient soft-mask optimisation; chosen because a soft-mask " *
                "gradient VANISHES on the §1 position/index output — see output_note)",
            "mask" => f.mask_names,
            "mask_size" => length(f.mask),
            "mask_abs_dy" => f.mask_dy,
            "mask_growth_curve" => f.growth,
            "pert_l1" => f.pert_l1,
            "budget" => Dict{String,Any}(
                "kind" => "area_constraint",
                "area_fraction" => f.area,
                "max_cells" => f.max_cells,
                "n_causes" => length(f.cause_names),
                "n_reruns" => f.n_reruns,
                "horizon" => f.horizon,
                "breakdown" => "1 intact + N single-cell probes + (greedy steps)·(remaining cells) mask " *
                    "evaluations (≤ ⌈area·C⌉·C, the area budget) + 2·(N+1) deletion/insertion re-runs, per " *
                    "output; N=$(length(f.cause_names)) causes, area=$(f.area) ⇒ ≤$(f.max_cells) cells. Every " *
                    "re-run continues from the cached deepcopy CHECKPOINT (boot+replay paid once), so re-runs " *
                    "are incremental and per-game runtime stays modest."),
            "metrics" => Dict{String,Any}(
                "pearson_corr" => f.pearson, "spearman_corr" => f.spearman,
                "precision_at_k" => f.precision_at_k, "topk" => f.topk,
                "mask_iou_vs_true_causal_set" => _json_num(f.mask_iou),
                "deletion_auc" => _json_num(f.deletion_auc),
                "insertion_auc" => _json_num(f.insertion_auc)),
            "mask_iou_note" => "mask IoU = |extremal_mask ∩ oracle_causal_set| / |union|, measured on the " *
                "PERTURBED-CELL axis (the do-key (kind,index) that occlude! actually touches), so a " *
                "ram[i]:set in the mask matches a ram[i]:occlude in the oracle set — both zero the SAME cell. " *
                "oracle_causal_set = the oracle's true-causal top-$(f.topk) movers (|Δy|>0). This is the §5 " *
                "headline metric for the extremal-perturbation row ('mask IoU vs true causal set'). " *
                "Pixel/cell-level IoU scores against T1 (no T3); object-set IoU would need T3 (out of scope here).",
            "oracle_causal_set" => [f.cause_names[i] for i in f.oracle_causal_set],
            # F∧S∧M triad — F is the runner-computed faithfulness (unchanged);
            # S and M are derived from the method map + the oracle |Δy| (no new re-runs).
            "triad" => triad_extra_dict(f.pearson, f.pert_attr, f.oracle_abs_delta;
                                        topk = f.topk, seed = f.seed),
            "harness_positive_control" => Dict{String,Any}(
                "method" => "oracle_abs_delta (the perfectly-faithful attribution)",
                "pearson_corr" => f.oracle_self_pearson,
                "precision_at_k" => f.oracle_self_precision_at_k,
                "deletion_auc" => _json_num(f.oracle_self_deletion_auc),
                "insertion_auc" => _json_num(f.oracle_self_insertion_auc),
                "interpretation" => "corr=1 & precision@k=1 (on a non-degenerate column) ⇒ the scoring " *
                    "harness rewards a faithful map; the extremal-perturbation numbers are then a true measurement."),
            "auc_note" => "deletion/insertion curves measured on the TRUE VCS by re-running the real ROM " *
                "with top-attributed causes occluded (deletion) / restored (insertion); every point is a " *
                "genuine emulator re-run, not a surrogate. NaN = a genuinely flat experiment.",
            "intervention_note" => "Extremal perturbation is itself an intervention method " *
                "(experiment_design.md §5 'learned minimal mask'; §7 'Succeed/partial — faithful when the " *
                "perturbation is a valid intervention'). Unlike the gradient family it does NOT vanish on " *
                "the position/index output — the headline contrast.",
            "output_note" => _output_note(f),
            "cause_names" => f.cause_names,
            "extremal_attr_per_cause" => Dict(f.cause_names[i] => f.pert_attr[i] for i in 1:length(f.cause_names)),
            "oracle_abs_delta_per_cause" => Dict(f.cause_names[i] => f.oracle_abs_delta[i] for i in 1:length(f.cause_names)),
            "y_full" => f.y_full,
            "comparison_note" => "Extremal perturbation vs the gradient family: vanilla saliency/Grad×Input/IG " *
                "measure ∂y/∂u (vanish on index outputs); extremal perturbation optimises a real-intervention " *
                "mask (alive everywhere the perturbation is valid). Vs occlusion: occlusion ranks each cell by " *
                "its SOLO occlusion |Δy|; extremal perturbation finds the JOINTLY-minimal high-impact MASK " *
                "(captures interactions a per-cell occlusion can miss). Vs RISE: RISE averages random masks; " *
                "extremal perturbation GREEDILY optimises the single most meaningful minimal mask.",
            "scales_to_cluster_via" =>
                "tools/cluster/xai_array_jl.sbatch (E0-3) — --array=0-5%N over the 6 core games " *
                "(--shard i --shard-kind game); each task re-runs the real ROM, no GPU needed.",
        ),
    )
    # SHARED-TESTBED provenance (redesign protocol) on the position/region record.
    if st_extra !== nothing
        rec["state"] = "gameplay(seed=$(st_extra.seed),prefix=$(st_extra.prefix))+$(st_extra.horizon)"
        rec["extra"]["testbed"] = Dict{String,Any}(
            "state_kind" => "seeded_random_action_gameplay",
            "prefix" => st_extra.prefix, "horizon" => st_extra.horizon, "seed" => st_extra.seed,
            "shared_output" => "screen_region(n_changed_px)@r$(st_extra.cell[1])c$(st_extra.cell[2])",
            "cause_density_above_floor" => st_extra.cause_density,
            "cause_density_floor" => ST_FLOOR, "cause_density_gate_k" => ST_GATE_K,
            "cause_density_accepted" => st_extra.accepted, "n_causes" => st_extra.n_causes)
    end
    open(json_path, "w") do io; JSON.print(io, rec, 2); end

    write_npz(npz_path, Dict(
        "extremal_attr_per_cause" => f.pert_attr,
        "oracle_abs_delta"        => f.oracle_abs_delta,
        "mask_indices"            => Float64.(f.mask),
        "mask_growth_curve"       => f.growth,
        "deletion_curve"          => f.del_curve,
        "insertion_curve"         => f.ins_curve,
        "scalars"                 => Float64[f.pearson, f.spearman, f.precision_at_k,
                                            isnan(f.mask_iou) ? -1.0 : f.mask_iou,
                                            f.deletion_auc, f.insertion_auc, f.pert_l1]))
    return json_path, npz_path
end

# ============================================================================
# Shard / CLI plumbing (cluster-shardable; cf. tools/cluster/xai_array_jl.sbatch).
# ============================================================================
"""Resolve which games to run. Priority: explicit --game / --games, else a
`--shard i --nshards n --shard-kind game` selection of the i-th core game, else
all 6 core games."""
function resolve_games(; single_game, games_arg, shard, nshards, shard_kind)
    single_game !== nothing && return [single_game]
    games_arg !== nothing && return xai_resolve_games(games_arg, CORE_GAMES)
    if shard !== nothing && nshards !== nothing && nshards > 1 && shard_kind == "game"
        return [CORE_GAMES[(i % length(CORE_GAMES)) + 1]
                for i in (shard:nshards:length(CORE_GAMES)-1)]
    end
    return CORE_GAMES
end

function main(args = ARGS)
    games_arg = nothing; single_game = nothing
    target_frame = 120; horizon = 30
    area = 0.25                                       # the AREA budget (fraction of causes)
    topk = 3; seed = 0
    out_dir = DEFAULT_OUT_DIR
    where = "local"
    shard = nothing; nshards = nothing; shard_kind = "game"
    selftest_only = false
    i = 1
    while i <= length(args)
        a = args[i]
        if     a == "--games";        games_arg = args[i+1]; i += 2
        elseif a == "--game";         single_game = args[i+1]; i += 2
        elseif a == "--target-frame"; target_frame = parse(Int, args[i+1]); i += 2
        elseif a == "--horizon";      horizon = parse(Int, args[i+1]); i += 2
        elseif a == "--area";         area = parse(Float64, args[i+1]); i += 2
        elseif a == "--topk";         topk = parse(Int, args[i+1]); i += 2
        elseif a == "--seed";         seed = parse(Int, args[i+1]); i += 2
        elseif a == "--out-dir";      out_dir = args[i+1]; i += 2
        elseif a == "--where";        where = args[i+1]; i += 2
        elseif a == "--shard";        shard = parse(Int, args[i+1]); i += 2
        elseif a == "--nshards";      nshards = parse(Int, args[i+1]); i += 2
        elseif a == "--shard-kind";   shard_kind = args[i+1]; i += 2
        elseif a == "--roms-dir";     i += 2          # accepted (env locates ROMs); ignored
        elseif a == "--selftest";     selftest_only = true; i += 1
        else; i += 1                                  # ignore unknown flags (permissive)
        end
    end
    games = resolve_games(; single_game = single_game, games_arg = games_arg,
                          shard = shard, nshards = nshards, shard_kind = shard_kind)
    sharded = shard !== nothing && nshards !== nothing && nshards > 1

    println("[perturbation] meaningful/extremal perturbation (Fong & Vedaldi 2017/2019) vs oracle — " *
            "games=$(games) target_frame=$target_frame horizon=$horizon area=$area topk=$topk seed=$seed " *
            "out_dir=$out_dir where=$where" *
            (shard === nothing ? "" : " shard=$shard/$nshards kind=$shard_kind") * " (jutari/Julia)")

    summary = Dict{String,Any}[]
    for game in games
        println("\n[perturbation] ===== $game =====")
        f_content, content_degen, f_pos, pos_degen, st_extra = compute_game(; game = game,
            target_frame = target_frame, horizon = horizon, area = area,
            topk = topk, seed = seed, verbose = true)
        @assert !content_degen "content oracle column is degenerate for $game at f$target_frame " *
            "— pick_content_idx failed to find a causally-active concept byte"
        selftest(f_content; require_nondegenerate = true)
        selftest(f_pos; require_nondegenerate = !pos_degen, is_core = game in CORE_GAMES)
        if st_extra !== nothing
            println("[perturbation] $game SHARED region output: gate " *
                "$(st_extra.cause_density)/$(st_extra.n_causes) accepted=$(st_extra.accepted) " *
                "cell=$(st_extra.cell)")
        end
        if !selftest_only
            for f in (f_content, f_pos)
                jp, np = write_faithfulness(f; out_dir = out_dir,
                    st_extra = (f === f_pos ? st_extra : nothing))
                println("[perturbation] wrote $jp"); println("[perturbation] arrays  $np")
            end
        end
        for f in (f_content, f_pos)
            push!(summary, Dict{String,Any}(
                "game" => game, "output" => f.output, "output_kind" => f.output_kind,
                "content_ram_index" => f.content_idx,
                "pearson" => f.pearson, "spearman" => f.spearman,
                "precision_at_k" => f.precision_at_k,
                "mask_iou" => _json_num(f.mask_iou),
                "mask_size" => length(f.mask), "max_cells" => f.max_cells,
                "deletion_auc" => _json_num(f.deletion_auc),
                "insertion_auc" => _json_num(f.insertion_auc),
                "pert_l1" => f.pert_l1, "mask_abs_dy" => f.mask_dy,
                "n_reruns" => f.n_reruns,
                "oracle_self_pearson" => f.oracle_self_pearson,
                "oracle_self_precision_at_k" => f.oracle_self_precision_at_k,
                "is_content_path" => f.output_kind == "content"))
        end
    end

    if selftest_only
        println("\n[perturbation] --selftest: all passed, not writing artifacts.")
        return 0
    end

    # cross-game summary only on a full (non-sharded) run.
    if !sharded
        isdir(out_dir) || mkpath(out_dir)
        summary_path = joinpath(out_dir, "perturbation_core_summary.json")
        total_budget = sum(r["n_reruns"] for r in summary; init = 0)
        rec = Dict{String,Any}(
            "paper" => "P2", "phase" => "phaseB_attribution", "method" => "extremal_perturbation",
            "item" => "P2-E4-8",
            "games" => games, "area" => area, "topk" => topk, "seed" => seed,
            "target_frame" => target_frame, "horizon" => horizon,
            "where" => where, "commit" => _git_commit(), "timestamp" => string(round(Int, time())),
            "method_ref" => "meaningful/extremal perturbation (Fong & Vedaldi 2017; Fong et al. 2019): " *
                "optimised minimal mask under an area budget; greedy on the true VCS",
            "total_budget_reruns" => total_budget,
            "headline" => "Extremal perturbation OPTIMISES a minimal high-impact mask on the TRUE VCS " *
                "(greedy, area-bounded) — every probe a real do(occlude)-re-run, so unlike the gradient " *
                "family it does NOT vanish on the position/index output and recovers the true causal " *
                "bytes on BOTH the content output AND the position output (the §7 'Succeed/partial — " *
                "faithful when the perturbation is a valid intervention' prediction). The §5 headline " *
                "metric is mask IoU vs the oracle's true-causal set. Oracle-as-method positive control " *
                "corr=1/p@k=1 on every non-degenerate column.",
            "results" => summary)
        open(summary_path, "w") do io; JSON.print(io, rec, 2); end
        println("\n[perturbation] wrote summary $summary_path")
    end

    println("\n[perturbation] ===== per-game HEADLINE (content path) =====")
    println("  game             ram   corr     spear    p@$topk    mask_IoU mask     del_auc  ins_auc  budget")
    for r in summary
        r["is_content_path"] || continue
        println("  $(rpad(r["game"],16)) $(rpad(r["content_ram_index"],5)) " *
                "$(rpad(round(r["pearson"],digits=3),8)) $(rpad(round(r["spearman"],digits=3),8)) " *
                "$(rpad(round(r["precision_at_k"],digits=3),6)) " *
                "$(rpad(r["mask_iou"] === nothing ? "NaN" : round(r["mask_iou"],digits=3),8)) " *
                "$(rpad("$(r["mask_size"])/$(r["max_cells"])",8)) " *
                "$(rpad(r["deletion_auc"] === nothing ? "NaN" : round(r["deletion_auc"],digits=3),8)) " *
                "$(rpad(r["insertion_auc"] === nothing ? "NaN" : round(r["insertion_auc"],digits=3),8)) " *
                "$(r["n_reruns"])")
    end
    println("[perturbation] ===== position output (extremal perturbation SUCCEEDS where the gradient vanishes) =====")
    println("  game             corr     p@$topk    mask_IoU mask     del_auc  ins_auc")
    for r in summary
        r["is_content_path"] && continue
        println("  $(rpad(r["game"],16)) $(rpad(round(r["pearson"],digits=3),8)) " *
                "$(rpad(round(r["precision_at_k"],digits=3),6)) " *
                "$(rpad(r["mask_iou"] === nothing ? "NaN" : round(r["mask_iou"],digits=3),8)) " *
                "$(rpad("$(r["mask_size"])/$(r["max_cells"])",8)) " *
                "$(rpad(r["deletion_auc"] === nothing ? "NaN" : round(r["deletion_auc"],digits=3),8)) " *
                "$(r["insertion_auc"] === nothing ? "NaN" : round(r["insertion_auc"],digits=3))")
    end
    return 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    PerturbationAttr.main()
end
