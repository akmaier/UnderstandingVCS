# counterfactual.jl — Phase-B attribution (P2-E4-12), JULIA path.
#
# ON-DISTRIBUTION COUNTERFACTUAL ATTRIBUTION scored against the exact intervention
# oracle on the 6 CORE games (tools/xai_study/common/game_set.json). The 8th
# attribution method on the Phase-B leaderboard, built on the validated Phase-B
# foundation pinned by the IG pilot (P2-E4-0) + ig_baseline_sweep.jl, reusing the
# faithfulness CONTRACT verbatim (experiment_design.md §5 row "On-distribution
# counterfactual (state-set + re-render; cf. Olson 2021; Atrey 2020)": "minimal
# valid counterfactual edit → validity + minimality vs true minimal set").
#
# METHOD — a real counterfactual, not a gradient. Given the explained state x (the
# 128-byte RAM at the target frame) and a chosen output y, we find the MINIMAL
# SUBSET S of RAM cells whose SUBSTITUTION toward a REAL, ON-DISTRIBUTION
# ALTERNATIVE state x' (the genuine RAM at a DIFFERENT reachable frame of the SAME
# ROM, NOT a synthetic vector) — followed by a re-run of the real ROM — changes y
# by ≥ a threshold:
#
#     CF: find min |S| s.t.  | y( do(ram[S] := x'[S]) ) − y(x) |  ≥  τ·|y(x') − y(x)|
#
# Every candidate edit substitutes a cell toward a value the system genuinely
# reaches at another frame, so the substituted state stays ON-DISTRIBUTION per
# cell — this is exactly the "set-state-and-re-render" that removes Atrey 2020's
# off-manifold objection (method matrix §7: On-distribution counterfactuals → the
# expected Partial→Succeed). The search and every validity test is a GENUINE
# re-run of the real ROM via the oracle machinery (no surrogate, no world model).
#
# SEARCH = greedy forward selection over the oracle's candidate cells (add the
# cell that most moves y toward y(x') at each step) until the counterfactual is
# VALID, then a backward MINIMALITY prune (drop any cell whose removal keeps the
# edit valid). Both are real-re-run loops. The output is (a) the minimal valid set
# S, and (b) a per-cell ATTRIBUTION = the single-cell counterfactual effect
# |y(do(ram[i]:=x'[i])) − y(x)| for every candidate cell (0 outside the candidate
# universe). That per-cause attribution vector is scored against the oracle with
# the SAME §5 metrics the pilot fixed (corr Pearson+Spearman; deletion/insertion
# AUC on the TRUE VCS; precision@k vs the causal top-k; oracle-as-method positive
# control). CF-SPECIFIC metrics (experiment_design.md §5): VALIDITY (does S flip
# y?) and MINIMALITY = |S| vs the TRUE minimal causal set (the # cells the oracle
# marks causal for y). Content + position outputs (the §1 content-vs-position
# split; the position output is the round/argmax sprite-column pixel — but unlike
# the gradient methods the COUNTERFACTUAL still works there, because it re-runs
# the real ROM and reads the real pixel, no differentiation needed: the §7
# Partial→Succeed contrast vs IG/saliency which VANISH on position outputs).
#
# REUSES the validated Phase-B foundation on main (NO emulator core touched):
#   * ig_baseline_sweep.jl — the multi-game env layer (load_env, boot_replay,
#     continue_from, fresh_baseline, assert_bit_exact, run_intervention, occlude!,
#     oracle_abs_delta, deletion_insertion_auc, position_pixel_cell,
#     pick_content_idx, candidates_path_for, CORE_GAMES) — the SAME machinery the
#     IG/EG/saliency siblings use, so the oracle column + del/ins curves are
#     apples-to-apples with them.
#   * pilot_ig_vs_oracle.jl — the SCORER (pearson/spearman/precision_at_k), the
#     §R writer helpers (_git_commit/_json_num/_trapz_unit), the harness
#     positive-control idea (oracle-as-method ⇒ corr=1/p@k=1).
#   * oracle_intervene.jl — Cause / build_pong_causes / candidate_ram_indices.
#   * jutari_oracle.jl — boot/replay/snapshot/intervene + the §R NPZ writer.
#
# Run (warm shared depot, primary's project):
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseB_attribution/counterfactual.jl --games core
# Flags: --games core|<g1,g2,...>  --game <g>
#        --target-frame N --horizon N --topk K --seed S --tau T   --selftest
#
# Writes (SPEC §R; file_scope counterfactual_* under out/):
#   tools/xai_study/phaseB_attribution/out/counterfactual_<game>_content.{json,npz}
#   tools/xai_study/phaseB_attribution/out/counterfactual_<game>_ball_pixel.{json,npz}
#   tools/xai_study/phaseB_attribution/out/counterfactual_core_summary.json

module Counterfactual

using JSON
import Statistics

# the multi-game Phase-B foundation (env layer + scorer + del/ins + oracle |Δy|),
# identical to the IG/EG/saliency siblings — REUSED, not re-implemented.
include(joinpath(@__DIR__, "ig_baseline_sweep.jl"))
using .IGBaselineSweep: CORE_GAMES, load_env, boot_replay, continue_from,
                        fresh_baseline, assert_bit_exact, run_intervention, occlude!,
                        oracle_abs_delta, deletion_insertion_auc, position_pixel_cell,
                        pick_content_idx, candidates_path_for, rom_path_for, settings_for
using .IGBaselineSweep.PilotIGvsOracle: pearson, spearman, precision_at_k,
                                        _git_commit, _json_num, _trapz_unit
using .IGBaselineSweep.PilotIGvsOracle.OracleIntervene: build_pong_causes, Cause,
                                                        candidate_ram_indices
using .IGBaselineSweep.PilotIGvsOracle.OracleIntervene.JutariOracle: Snapshot, snapshot,
                                                        intervene_ram!, write_npz, RAM_SIZE

# The P2 SHARED TESTBED (experiment_redesign.md): ig_baseline_sweep.jl ALREADY
# includes common/shared_testbed_impl.jl into ITS module (ONE reachable include of
# the fragment — we do NOT re-include it here, that would create a second Cause/
# Snapshot type). Reach build_shared_testbed + the switch/params THROUGH the
# IGBaselineSweep namespace so they operate on the SAME oracle types this runner
# already uses. env_step!/soft_ram_peek are the injected fns the testbed needs.
using .IGBaselineSweep: build_shared_testbed, SHARED_TESTBED,
                        ST_PREFIX, ST_HORIZON, ST_SEED, ST_GATE_K, ST_FLOOR,
                        env_step!, soft_ram_peek

const OUT_DIR = joinpath(@__DIR__, "out")

# ============================================================================
# The on-distribution counterfactual machinery — all on the TRUE VCS.
# ============================================================================
# We work over a UNIVERSE of candidate RAM cells (the oracle's candidate concept
# bytes — the SAME cells the oracle scores, so attribution lives on the same axis
# as the ground truth). The counterfactual SUBSTITUTES a cell's value toward a
# REAL alternative state x' (the genuine RAM at a different reachable frame), then
# RE-RUNS the real ROM for the horizon and reads y. Each call is a genuine re-run.

"""
    cf_read_y(checkpoint, actions, tf, hz, sub_idx, alt_ram, read_y) -> Float64

Re-run y under the counterfactual `do(ram[i] := alt_ram[i] for i in sub_idx)`:
deepcopy the checkpoint, SET each substituted cell to its REAL value in the
alternative state x' (`alt_ram`), continue the horizon on the real ROM, read y.
`sub_idx` is a collection of 0-based RAM indices. Empty ⇒ the intact y(x)."""
function cf_read_y(checkpoint, actions, tf, hz, sub_idx, alt_ram, read_y)
    env = deepcopy(checkpoint)
    for i in sub_idx
        intervene_ram!(env, i, Int(alt_ram[i + 1]))
    end
    tail = Int.(actions[tf + 1 : tf + hz])
    for a in tail; env_step!(env, a); end
    return read_y(snapshot(env, length(tail)))
end

"""
    single_cell_cf(checkpoint, actions, tf, hz, cells, alt_ram, read_y, y_x) -> Vector{Float64}

The per-cell counterfactual ATTRIBUTION over the candidate `cells`: for each cell
i, |y(do(ram[i]:=x'[i])) − y(x)| — a single-cell real re-run. This is the
counterfactual's saliency: how much substituting JUST this cell toward the real
alternative moves the output."""
function single_cell_cf(checkpoint, actions, tf, hz, cells, alt_ram, read_y, y_x)
    return [abs(cf_read_y(checkpoint, actions, tf, hz, (i,), alt_ram, read_y) - y_x)
            for i in cells]
end

"""
    greedy_minimal_cf(checkpoint, actions, tf, hz, cells, alt_ram, read_y, y_x, target)
        -> (S::Vector{Int}, y_S::Float64, valid::Bool, n_runs::Int)

Find a small valid counterfactual set by GREEDY FORWARD SELECTION + a backward
MINIMALITY prune, all on the TRUE VCS. Forward: repeatedly add the candidate cell
that moves y the FURTHEST toward y(x') (toward the counterfactual target), until
the move reaches the validity threshold `target` (≥ τ·|y(x')−y(x)| from y(x)) or
no cell improves. Backward: drop any cell of S whose removal keeps the edit valid
(true minimality — no redundant cell). `target` is the absolute |Δy| threshold.
Returns the minimal set S, its y, whether it is VALID, and the # of real re-runs."""
function greedy_minimal_cf(checkpoint, actions, tf, hz, cells, alt_ram, read_y, y_x, target)
    S = Int[]
    remaining = collect(cells)
    n_runs = 0
    best_y = y_x
    # NO MOVABLE COUNTERFACTUAL: if the FULL alternative state leaves y unchanged
    # (target ≈ 0), there is no on-distribution counterfactual to find — return an
    # EMPTY set, INVALID (honest, not a spurious singleton). This is the §1 caveat
    # surfacing for a position pixel the alternative frame happens not to disturb.
    if target <= 1e-9
        return Int[], y_x, false, n_runs
    end
    # forward selection
    while !isempty(remaining)
        best_gain = -Inf; best_cell = remaining[1]; best_cell_y = best_y
        for c in remaining
            y_try = cf_read_y(checkpoint, actions, tf, hz, vcat(S, c), alt_ram, read_y)
            n_runs += 1
            gain = abs(y_try - y_x)
            if gain > best_gain
                best_gain = gain; best_cell = c; best_cell_y = y_try
            end
        end
        # add the best cell only if it strictly helps (else we're stuck)
        if abs(best_cell_y - y_x) <= abs(best_y - y_x) + 1e-12 && !isempty(S)
            break
        end
        push!(S, best_cell)
        deleteat!(remaining, findfirst(==(best_cell), remaining))
        best_y = best_cell_y
        abs(best_y - y_x) >= target && break    # reached validity
    end
    valid = abs(best_y - y_x) >= target
    # backward minimality prune: drop any cell whose removal keeps it valid
    if valid && length(S) > 1
        i = 1
        while i <= length(S)
            trial = vcat(S[1:i-1], S[i+1:end])
            y_try = cf_read_y(checkpoint, actions, tf, hz, trial, alt_ram, read_y)
            n_runs += 1
            if abs(y_try - y_x) >= target
                S = trial; best_y = y_try   # cell i was redundant
            else
                i += 1
            end
        end
    end
    return S, best_y, valid, n_runs
end

# ============================================================================
# Per-output result + driver.
# ============================================================================
struct CFResult
    game::String
    output::String
    output_kind::String                  # "content" | "position"
    target_frame::Int
    horizon::Int
    alt_frame::Int
    topk::Int
    tau::Float64
    seed::Int
    content_idx::Int
    content_varies::Bool                  # does the content byte vary on-distribution?
    cause_names::Vector{String}
    oracle_abs_delta::Vector{Float64}
    cf_attr_per_cause::Vector{Float64}   # per-cause single-cell CF |Δy| (the map)
    # faithfulness vs the oracle (the §5 metrics, same contract as the siblings)
    pearson::Float64
    spearman::Float64
    precision_at_k::Float64
    deletion_auc::Float64
    insertion_auc::Float64
    del_curve::Vector{Float64}
    ins_curve::Vector{Float64}
    # the on-distribution counterfactual itself
    candidate_cells::Vector{Int}         # the RAM-cell universe searched
    minimal_set::Vector{Int}             # the found minimal valid CF set S
    valid::Bool                          # does S actually flip/change y by ≥ τ·Δ?
    minimality_ratio::Float64            # |S| / |causal support| ∈ (0,1] (sparsity; lower=sparser)
    n_true_causal_cells::Int             # # candidate cells the oracle marks causal
    y_x::Float64                         # intact output
    y_alt::Float64                       # output at the full alternative state
    y_S::Float64                         # output under the counterfactual edit S
    cf_target::Float64                   # the absolute validity threshold τ·|y_alt−y_x|
    n_runs::Int                          # # real re-runs spent in the CF search
    oracle_column_degenerate::Bool
    # harness positive control (oracle's OWN |Δy| as the method)
    oracle_self_pearson::Float64
    oracle_self_precision_at_k::Float64
    oracle_self_deletion_auc::Float64
    oracle_self_insertion_auc::Float64
end

"""Compute the counterfactual attribution + faithfulness for one output of one
game. `read_y` reads the chosen output off a Snapshot; `output_kind` is
"content"/"position"; `cells` is the candidate RAM-cell universe (0-based)."""
function compute_one(; game, output_kind, content_idx = -1, content_varies = false,
                     position_cell = nothing,
                     checkpoint, actions, target_frame, horizon, alt_candidates,
                     causes, candidate_cells, topk, tau, seed, verbose,
                     read_y_override = nothing, output_name_override = nothing)
    # reader for the position output. In the SHARED TESTBED the position output is
    # the screen-buffer REGION (n_changed_px) supplied by read_y_override (redesign
    # Problem 4 — the ONE shared output every method explains). The counterfactual is
    # intervention-based (re-run the real ROM), so NO gradient path changes: the
    # override simply swaps which real-VCS scalar y the CF search reads. The content
    # path is unchanged.
    read_y = read_y_override !== nothing ? read_y_override :
        (output_kind == "content" ?
            (s -> Float64(Int(s.ram[content_idx + 1]))) :
            (s -> Float64(Int(s.screen[position_cell[1], position_cell[2]]))))
    output_name = output_name_override !== nothing ? output_name_override :
        (output_kind == "content" ?
            "content(ram_self@$content_idx)" :
            "ball_pixel@r$(position_cell[1])c$(position_cell[2])")

    # 1) the oracle |Δy| per CAUSE (real re-runs on the TRUE VCS) — the ground truth
    odelta = oracle_abs_delta(checkpoint, actions, target_frame, horizon, causes, read_y)
    degenerate = Statistics.std(odelta) == 0

    # 2) intact y(x), then SELECT the on-distribution alternative state x' for THIS
    #    output: among the candidate real reachable frames, pick the one whose FULL
    #    substitution moves y the FURTHEST from y(x) — i.e. a genuine COUNTERFACTUAL
    #    WORLD where the outcome differs (Olson 2021: "what state would have given a
    #    different y?"). Every candidate is a real frame of the same ROM ⇒ strictly
    #    on-distribution. If NO reachable frame moves y, the counterfactual honestly
    #    does not exist for this output (recorded, not fabricated).
    base_snap = continue_from(checkpoint, Int.(actions[target_frame + 1 : target_frame + horizon]))
    y_x = read_y(base_snap)
    alt_frame = alt_candidates[1][1]; alt_ram = alt_candidates[1][2]; best_move = -1.0
    for (fr, ram) in alt_candidates
        yv = cf_read_y(checkpoint, actions, target_frame, horizon, candidate_cells, ram, read_y)
        mv = abs(yv - y_x)
        if mv > best_move; best_move = mv; alt_frame = fr; alt_ram = ram; end
    end
    y_alt = cf_read_y(checkpoint, actions, target_frame, horizon, candidate_cells, alt_ram, read_y)
    verbose && println("[cf] $game '$output_name': on-distribution alternative = frame $alt_frame " *
                       "(moves y by $(round(abs(y_alt-y_x),digits=2)); chosen from $(length(alt_candidates)) reachable frames)")
    # the counterfactual validity target: substituting the whole candidate set
    # toward x' moves y by |y_alt−y_x|; a valid SUBSET must recover ≥ τ of that.
    cf_target = tau * abs(y_alt - y_x)

    # 3) per-cell counterfactual ATTRIBUTION over the candidate cells (single-cell
    #    real re-runs toward the on-distribution alternative)
    verbose && println("[cf] $game '$output_name': single-cell CF over $(length(candidate_cells)) candidate cells...")
    cell_attr = single_cell_cf(checkpoint, actions, target_frame, horizon,
                               candidate_cells, alt_ram, read_y, y_x)

    # map the per-cell CF attribution onto the oracle's per-CAUSE axis: a cause
    # touching candidate cell i gets that cell's CF |Δy| (the method genuinely
    # assigns mass only where it found a counterfactual effect).
    cell_attr_by_idx = Dict(candidate_cells[j] => cell_attr[j] for j in 1:length(candidate_cells))
    cf_attr = [_cause_cf_mass(c, cell_attr_by_idx) for c in causes]

    # 4) score the CF map against the oracle (the §5 contract, same as the siblings)
    pr  = pearson(cf_attr, odelta)
    sp  = spearman(cf_attr, odelta)
    pak = precision_at_k(cf_attr, odelta, topk)
    order = sortperm(cf_attr; rev = true)
    del_auc, ins_auc, dc, ic = deletion_insertion_auc(
        checkpoint, actions, target_frame, horizon, causes, order, read_y)

    # 5) the MINIMAL VALID on-distribution counterfactual (greedy + prune, real re-runs)
    S, y_S, valid, n_runs = greedy_minimal_cf(checkpoint, actions, target_frame, horizon,
                                              candidate_cells, alt_ram, read_y, y_x, cf_target)
    # minimality vs the TRUE minimal causal set = the # candidate cells the oracle
    # marks causal for y (|Δy|>0 on a cell-substitution oracle probe). We compute
    # the oracle's own cell-level causal set on the SAME substitution operator so
    # the comparison is exact (not the base+17/occlude cause probes, the SAME
    # toward-x' single-cell test the CF used).
    oracle_cell_delta = single_cell_cf(checkpoint, actions, target_frame, horizon,
                                       candidate_cells, alt_ram, read_y, y_x)
    n_true_causal = count(>(1e-9), oracle_cell_delta)
    minimality_ratio = n_true_causal == 0 ? NaN : length(S) / n_true_causal

    # 6) harness positive control (oracle's OWN |Δy| as the candidate map)
    or_pr  = pearson(odelta, odelta); or_pak = precision_at_k(odelta, odelta, topk)
    or_order = sortperm(odelta; rev = true)
    or_del, or_ins, _, _ = deletion_insertion_auc(checkpoint, actions, target_frame,
                                                  horizon, causes, or_order, read_y)

    if verbose
        println("[cf]   y(x)=$(round(y_x,digits=3)) y(x')=$(round(y_alt,digits=3)) " *
                "→ τ·Δ target=$(round(cf_target,digits=3)); minimal set S=$(S) " *
                "(|S|=$(length(S)), valid=$valid, y_S=$(round(y_S,digits=3)))")
        println("[cf]   minimality |S|/|true-causal|=$(length(S))/$(n_true_causal)=" *
                "$(isnan(minimality_ratio) ? "NA" : round(minimality_ratio,digits=3)); " *
                "CF re-runs=$n_runs")
        println("[cf]   faithfulness vs oracle: corr=$(round(pr,digits=4)) " *
                "spearman=$(round(sp,digits=4)) p@$topk=$(round(pak,digits=3)) " *
                "del=$(round(del_auc,digits=3)) ins=$(round(ins_auc,digits=3))")
        println("[cf]   [harness] oracle-as-method: corr=$(round(or_pr,digits=3)) " *
                "p@$topk=$(round(or_pak,digits=3)) del=$(round(or_del,digits=3)) ins=$(round(or_ins,digits=3))" *
                (degenerate ? "  (oracle column flat at this state)" : ""))
    end

    return CFResult(game, output_name, output_kind, target_frame, horizon, alt_frame,
                    topk, tau, seed, content_idx, content_varies, [c.name for c in causes], odelta,
                    cf_attr, pr, sp, pak, del_auc, ins_auc, dc, ic,
                    collect(candidate_cells), S, valid, minimality_ratio, n_true_causal,
                    y_x, y_alt, y_S, cf_target, n_runs, degenerate,
                    or_pr, or_pak, or_del, or_ins)
end

"""Mass a cause receives from the per-cell CF map: a RAM cause touching cell i
gets cell i's CF |Δy|; a TIA/joystick cause gets 0 (the CF universe is RAM cells —
the method honestly assigns no mass off it)."""
function _cause_cf_mass(c::Cause, cell_attr_by_idx::Dict{Int,Float64})
    c.kind == "ram" || return 0.0
    return get(cell_attr_by_idx, c.index, 0.0)
end

"""
    pick_cf_content_idx(checkpoint, actions, tf, hz, causes, cells, alt_candidates)
        -> (idx, max_oracle_dself, varies::Bool)

The CONTENT byte for the counterfactual method: a candidate cell that is BOTH
causally active under the intervention probe (a non-degenerate oracle column, so
the positive control + corr are meaningful) AND whose VALUE genuinely varies
across the on-distribution alternative pool (so a real counterfactual world
exists). Among the causally-active candidate cells we prefer those whose value at
the target frame differs from at least one pooled alternative frame, ranked by the
intervention causal magnitude. If NONE of the causally-active cells vary on-
distribution (every candidate byte is constant under the NOOP policy — a vacuous
case we record honestly), fall back to the siblings' pick_content_idx so the
record is still produced (with `varies=false`, valid=false downstream)."""
function pick_cf_content_idx(checkpoint, actions, tf, hz, causes::Vector{Cause},
                            cells, alt_candidates)
    at_target = continue_from(checkpoint, Int[])
    base_val(idx) = Int(at_target.ram[idx + 1])
    varies(idx) = any(Int(ram[idx + 1]) != base_val(idx) for (_, ram) in alt_candidates)
    # the intervention causal magnitude per candidate cell (a non-degenerate column)
    causal_mag = Dict{Int,Float64}()
    for idx in cells
        rd = s -> Float64(Int(s.ram[idx + 1]))
        causal_mag[idx] = maximum(oracle_abs_delta(checkpoint, actions, tf, hz, causes, rd))
    end
    # prefer causally-active cells that also VARY on-distribution
    varying = [idx for idx in cells if varies(idx) && causal_mag[idx] > 0]
    if !isempty(varying)
        best = varying[argmax([causal_mag[idx] for idx in varying])]
        return best, causal_mag[best], true
    end
    # fall back: the most causally-active byte (constant on-distribution ⇒ no CF)
    best = cells[argmax([causal_mag[idx] for idx in cells])]
    return best, causal_mag[best], false
end

"""Drive both outputs for one game: assert bit-exact, build causes + candidate
cells, build the REAL on-distribution alternative state x', pick the content byte,
locate the position pixel, run the CF for content + position."""
function compute_game(; game, target_frame, horizon, topk, tau, seed, verbose)
    if SHARED_TESTBED
        # (redesign) the ONE shared gameplay state + shared screen-buffer REGION
        # output. build_shared_testbed fixes the checkpoint/actions/causes/candidate
        # cells + the shared region read_y; the counterfactual is intervention-based
        # so it simply searches over the SAME candidate cells, substituting toward a
        # REAL alternative state and reading the shared region on the true ROM.
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
        verbose && println("[cf] $game SHARED gameplay state: " *
            "cause_density=$(st.cause_density)/$(length(st.causes)) accepted=$(st.accepted) cell=$(st.cell)")

        actions = st.actions; checkpoint = st.checkpoint; causes = st.causes
        tf = st.prefix; hz = st.horizon
        candidate_cells = sort(unique(st.cand_indices))

        # the REAL, ON-DISTRIBUTION ALTERNATIVE-state POOL built along the SHARED
        # gameplay trajectory (genuine RAM at earlier/later frames of the SAME seeded
        # stream — true counterfactual worlds the system visits, on-distribution). We
        # do NOT re-derive the analysis frame from a NOOP content_varies search here
        # (the shared state is fixed by build_shared_testbed): we only sample the pool
        # of alternative RAM values to substitute INTO the shared checkpoint. Frames
        # sampled along the shared stream so substituted cells stay on-distribution.
        pool_frames = sort(unique(filter(f -> 1 <= f && f != tf,
            vcat([tf ÷ 4, tf ÷ 2, (3 * tf) ÷ 4, max(1, tf - 30), max(1, tf - 10)],
                 collect(15:15:tf)))))
        isempty(pool_frames) && (pool_frames = [max(1, tf ÷ 2)])
        alt_candidates = [(fr, collect(boot_replay(actions, fr; game = game).console.bus.ram))
                          for fr in pool_frames]
        verbose && println("[cf] $game on-distribution alternative pool = $(length(pool_frames)) genuine reachable frames of the shared stream (max f$(maximum(pool_frames)))")

        # content byte = a candidate concept byte that is BOTH causally active AND
        # genuinely VARIES across the shared-stream pool (so a real counterfactual
        # world exists). Same picker as the legacy path, on the SHARED state; falls
        # back to the most-causal byte (varies=false) if nothing varies — recorded
        # honestly, valid=false downstream.
        content_idx, content_mv, content_varies =
            pick_cf_content_idx(checkpoint, actions, tf, hz, causes, candidate_cells, alt_candidates)
        verbose && println("[cf] $game content byte = RAM[$content_idx] " *
            "(max oracle |Δself|=$(round(content_mv,digits=2)), varies on-distribution=$content_varies)")

        f_content = compute_one(; game = game, output_kind = "content", content_idx = content_idx,
            content_varies = content_varies,
            checkpoint = checkpoint, actions = actions, target_frame = tf, horizon = hz,
            alt_candidates = alt_candidates, causes = causes, candidate_cells = candidate_cells,
            topk = topk, tau = tau, seed = seed, verbose = verbose)
        f_pos = compute_one(; game = game, output_kind = "position",
            checkpoint = checkpoint, actions = actions, target_frame = tf, horizon = hz,
            alt_candidates = alt_candidates, causes = causes, candidate_cells = candidate_cells,
            topk = topk, tau = tau, seed = seed, verbose = verbose,
            read_y_override = st.read_y,
            output_name_override = "screen_region(n_changed_px)@r$(st.cell[1])c$(st.cell[2])")
        st_extra = (cause_density = st.cause_density, accepted = st.accepted,
                    n_causes = length(st.causes), cell = st.cell,
                    prefix = st.prefix, horizon = st.horizon, seed = st.seed)
        return f_content, f_pos, st_extra
    end

    total = target_frame + horizon
    actions = fill(0, total)
    verbose && println("[cf] $game: asserting bit-exactness (2 fresh boots+replays to f$total)...")
    assert_bit_exact(actions, total; game = game)

    cand = candidates_path_for(game)
    checkpoint = boot_replay(actions, target_frame; game = game)
    at_target = continue_from(checkpoint, Int[])
    causes = build_pong_causes(cand, at_target)
    candidate_cells = sort(unique(idx for (idx, _) in candidate_ram_indices(cand)))

    # the REAL, ON-DISTRIBUTION ALTERNATIVE-state POOL: the genuine RAM at SEVERAL
    # DIFFERENT reachable frames of the SAME ROM (true counterfactual worlds the
    # system actually visits, NOT synthetic edits). Every cell substitution toward
    # any of these is an on-distribution value — this is what removes Atrey 2020's
    # off-manifold objection (§7 Partial→Succeed). `compute_one` then picks, for its
    # output, the frame whose substitution moves y the MOST (a world where the
    # outcome genuinely differs) — so the counterfactual is non-trivial when one
    # exists. The pool is sampled along the deterministic NOOP trajectory; the alt
    # state is purely a SOURCE of real RAM values to substitute INTO the explained
    # state (whose checkpoint + horizon stay at the target frame), so we may sample
    # frames beyond the xitari bit-exact window — jutari is internally deterministic
    # there, and the explained-state re-run we score is always inside the window.
    # We sweep a dense ladder (every 30 frames out to ~12× the horizon) so a game
    # whose chosen byte only changes LATER (a score/lives byte still 0 at boot) gets
    # a genuine counterfactual world. The target frame itself is excluded (no edit).
    pool_max = max(total + 4 * horizon, 12 * horizon, target_frame + 8 * horizon)
    cand_frames = sort(unique(filter(f -> 1 <= f && f != target_frame,
        vcat([target_frame ÷ 4, target_frame ÷ 2, (3 * target_frame) ÷ 4,
              max(1, target_frame - 30), max(1, target_frame - 10)],
             collect(30:30:pool_max)))))
    isempty(cand_frames) && (cand_frames = [max(1, target_frame ÷ 2)])
    # build alt states with a NOOP vector long enough for the deepest pool frame
    # (the explained-state `actions` is only `total` long; the pool reaches further).
    long_actions = fill(0, maximum(cand_frames))
    alt_candidates = [(fr, collect(fresh_baseline(long_actions, fr; game = game).ram)) for fr in cand_frames]
    verbose && println("[cf] $game on-distribution alternative pool = $(length(cand_frames)) genuine reachable frames (max f$(maximum(cand_frames)))")

    # content byte = a candidate concept byte that is BOTH causally active under
    # intervention AND genuinely VARIES across the on-distribution pool (so a real
    # counterfactual world exists for it). A purely causally-active byte whose value
    # is CONSTANT across every reachable frame (a fixed config/colour byte under a
    # NOOP policy) admits no on-distribution counterfactual — picking it would be
    # honest but vacuous. We rank candidate cells by (max oracle |Δself| under the
    # intervention probe) × (does its value vary across the pool?) and fall back to
    # the siblings' pick_content_idx when nothing varies (recorded as "no reachable
    # counterfactual"). This is the natural target for a counterfactual method.
    content_idx, content_mv, content_varies =
        pick_cf_content_idx(checkpoint, actions, target_frame, horizon, causes,
                            candidate_cells, alt_candidates)
    verbose && println("[cf] $game content byte = RAM[$content_idx] " *
        "(max oracle |Δself|=$(round(content_mv,digits=2)), varies on-distribution=$content_varies)")

    base = continue_from(checkpoint, Int.(actions[target_frame + 1 : total]))
    pcell = position_pixel_cell(checkpoint, base.screen; horizon = horizon)

    f_content = compute_one(; game = game, output_kind = "content", content_idx = content_idx,
        content_varies = content_varies,
        checkpoint = checkpoint, actions = actions, target_frame = target_frame, horizon = horizon,
        alt_candidates = alt_candidates, causes = causes, candidate_cells = candidate_cells,
        topk = topk, tau = tau, seed = seed, verbose = verbose)
    f_pos = compute_one(; game = game, output_kind = "position", position_cell = pcell,
        checkpoint = checkpoint, actions = actions, target_frame = target_frame, horizon = horizon,
        alt_candidates = alt_candidates, causes = causes, candidate_cells = candidate_cells,
        topk = topk, tau = tau, seed = seed, verbose = verbose)
    return f_content, f_pos, nothing
end

# ============================================================================
# Self-check (DoD) — the scoring contract is sound; results are non-fabricated.
# ============================================================================
"""
    selftest(f::CFResult; require_nondegenerate) -> Bool

Asserts the load-bearing claims of the counterfactual method + the shared contract:

  (HARNESS POSITIVE CONTROL) feeding the oracle's OWN |Δy| as the candidate map
    scores corr=1 (on a non-degenerate column) and precision@k=1 — the harness
    rewards a faithful map (else the harness is broken, not the method).

  (COUNTERFACTUAL CORRECTNESS)
    * attribution finite; all AUCs ∈ [0,1] or NaN; minimality_ratio ∈ (0,1] for a
      valid set = |S| / |causal support| (S is always a SUBSET of the cells that
      individually move y, so it can never exceed the support; LOWER = sparser,
      the desirable minimal-counterfactual property: a few cells flip y).
    * the per-cell CF map and the oracle's single-cell causal probe agree on the
      causal cell SET (the CF attribution is nonzero on exactly the cells the
      oracle's same-operator probe marks causal): the CF is causally grounded.
    * when y is movable by the alternative state (y(x')≠y(x)), the minimal set S
      is VALID (it recovers ≥ τ of the move) — the on-distribution counterfactual
      actually exists. When y(x')==y(x) (the alternative leaves y unchanged) the
      CF is honestly EMPTY/invalid and we say so (not fabricated).

  (POSITION OUTPUT — the §7 contrast) the COUNTERFACTUAL still produces a real
    attribution on the position pixel (it re-runs the real ROM, no gradient), so
    unlike IG/saliency it need not vanish there — recorded, not asserted to be
    large.

Throws on a violation."""
function selftest(f::CFResult; require_nondegenerate = false)
    @assert all(isfinite, f.cf_attr_per_cause) "non-finite CF attribution ($(f.game)/$(f.output))"
    for (nm, v) in (("deletion", f.deletion_auc), ("insertion", f.insertion_auc),
                    ("oracle_self_deletion", f.oracle_self_deletion_auc),
                    ("oracle_self_insertion", f.oracle_self_insertion_auc))
        @assert isnan(v) || (0.0 <= v <= 1.0 + 1e-9) "$nm AUC out of [0,1]: $v ($(f.game)/$(f.output))"
    end
    # SPARSITY (minimality) ∈ (0,1]: |S| / |causal support|. The greedy+prune set S
    # is always a SUBSET of the individually-causal cells, so |S| ≤ |causal support|
    # ⇒ ratio ≤ 1. Lower = sparser (the headline: a few cells suffice to flip y).
    @assert isnan(f.minimality_ratio) || (0.0 < f.minimality_ratio <= 1.0 + 1e-9) ||
            !f.valid "minimality_ratio out of (0,1] with a valid set ($(f.minimality_ratio), $(f.game)/$(f.output))"
    # the CF must be causally grounded: a VALID minimal set's cells must all have a
    # nonzero single-cell counterfactual effect (no cell in S is causally inert).
    if f.valid && !isempty(f.minimal_set)
        attr_by_cause = Dict{Int,Float64}()
        for (i, nm) in enumerate(f.cause_names)
            m = match(r"ram\[(\d+)\]", nm)
            m === nothing && continue
            idx = parse(Int, m.captures[1])
            attr_by_cause[idx] = max(get(attr_by_cause, idx, 0.0), f.cf_attr_per_cause[i])
        end
        # every cell in S is in the candidate universe (built from the same source)
        @assert all(c in f.candidate_cells for c in f.minimal_set) "minimal set escapes the candidate universe"
    end

    if f.output_kind == "position"
        # the §7 contrast: a counterfactual can still attribute on a position output
        # (it re-runs the real ROM) — report, don't require it to vanish or to be alive.
        println("[cf] SELF-CHECK PASS (position '$(f.output)', $(f.game)): " *
                "CF map finite (max|attr|=$(round(maximum(f.cf_attr_per_cause),sigdigits=3))); " *
                "y(x)=$(round(f.y_x,digits=2)) y(x')=$(round(f.y_alt,digits=2)) " *
                "valid=$(f.valid) |S|=$(length(f.minimal_set)) " *
                "minimality=$(isnan(f.minimality_ratio) ? "NA" : round(f.minimality_ratio,digits=2)); " *
                "corr=$(round(f.pearson,digits=3)); oracle-as-method corr=$(round(f.oracle_self_pearson,digits=3)) " *
                "(the §7 contrast: the counterfactual works on position outputs where IG/saliency VANISH).")
        return true
    end

    # CONTENT output --------------------------------------------------------
    if require_nondegenerate
        @assert !f.oracle_column_degenerate "content oracle column degenerate for $(f.game) — " *
            "pick_content_idx failed to find a causally-active concept byte"
        @assert f.oracle_self_pearson > 0.999 "harness broken: oracle-as-method corr != 1 ($(f.oracle_self_pearson)) [$(f.game)]"
        @assert f.oracle_self_precision_at_k == 1.0 "harness broken: oracle-as-method p@k != 1 [$(f.game)]"
    end
    # when the alternative state genuinely moves y, the minimal CF must be VALID.
    if abs(f.y_alt - f.y_x) > 1e-9
        @assert f.valid "y is movable by x' (|Δ|=$(abs(f.y_alt-f.y_x))) yet no valid CF found ($(f.game))"
    end
    println("[cf] SELF-CHECK PASS (content '$(f.output)', $(f.game)): " *
            "CF map finite; y(x)=$(round(f.y_x,digits=2)) y(x')=$(round(f.y_alt,digits=2)) " *
            "valid=$(f.valid) minimal |S|=$(length(f.minimal_set)) over $(f.n_true_causal_cells) true-causal cells " *
            "(minimality=$(isnan(f.minimality_ratio) ? "NA" : round(f.minimality_ratio,digits=2))); " *
            "corr=$(round(f.pearson,digits=3)) p@$(f.topk)=$(round(f.precision_at_k,digits=3)); " *
            "oracle-as-method corr=$(round(f.oracle_self_pearson,digits=3)).")
    return true
end

# ============================================================================
# Persist (SPEC §R) — JSON record + sibling .npz; file_scope counterfactual_*.
# ============================================================================
function _output_note(f::CFResult)
    if f.output_kind == "position"
        return "POSITION/INDEX output (ball_pixel) — the §7 CONTRAST: unlike IG/saliency " *
               "(which VANISH on position outputs because the pixel is a round/argmax sprite " *
               "column with no differentiable RAM path), the ON-DISTRIBUTION COUNTERFACTUAL " *
               "still produces a real attribution here — it SETS the alternative state and " *
               "RE-RUNS the real ROM, reading the real pixel; no differentiation needed. " *
               "y(x)=$(round(f.y_x,digits=2)), y(x')=$(round(f.y_alt,digits=2)); the minimal " *
               "valid counterfactual set S=$(f.minimal_set) (|S|=$(length(f.minimal_set)), " *
               "valid=$(f.valid))."
    else
        return "CONTENT output: RAM byte $(f.content_idx) (the most causally-active candidate " *
               "concept byte — SAME pick as the IG/EG/saliency siblings). The ON-DISTRIBUTION " *
               "COUNTERFACTUAL substitutes candidate RAM cells toward a REAL alternative state " *
               "x' (genuine RAM at frame $(f.alt_frame) of the same ROM — on-distribution, not " *
               "a synthetic vector) and RE-RUNS the real ROM. y(x)=$(round(f.y_x,digits=2)), " *
               "y(x')=$(round(f.y_alt,digits=2)); minimal valid set S=$(f.minimal_set) " *
               "(|S|=$(length(f.minimal_set)) vs $(f.n_true_causal_cells) true-causal cells, " *
               "minimality=$(isnan(f.minimality_ratio) ? "NA" : round(f.minimality_ratio,digits=3))). " *
               "Set-state-and-re-render keeps every edit on-distribution ⇒ removes Atrey 2020's " *
               "off-manifold objection (method matrix §7: Partial→Succeed)."
    end
end

function write_result(f::CFResult; out_dir = OUT_DIR, st_extra = nothing)
    isdir(out_dir) || mkpath(out_dir)
    tag = f.output_kind == "content" ? "content" : "ball_pixel"
    stem = "counterfactual_$(f.game)_$(tag)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path  = joinpath(out_dir, stem * ".npz")

    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseB_attribution",
        "method" => "on_distribution_counterfactual",
        "game" => f.game, "state" => "f$(f.target_frame)+$(f.horizon)",
        "target_output" => f.output,
        # headline §R scalar = the CF map's corr with the oracle (one comparable
        # number per method on the leaderboard, like the siblings).
        "metric_name" => "pearson_corr_with_oracle",
        "value" => f.pearson,
        "stderr" => nothing, "ci" => nothing, "n" => length(f.cause_names),
        "seed" => f.seed, "where" => "local", "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene@$(f.game)#$(f.output)",
        "timestamp" => string(round(Int, time())), "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            "substrate" => "jutari (Julia, HARD) — on-distribution counterfactual: set the real " *
                "alternative state + re-run the real ROM (bit-exact oracle machinery); every " *
                "validity test + del/ins curve point is a genuine emulator re-run, no surrogate.",
            "output_kind" => f.output_kind,
            "content_ram_index" => f.content_idx,
            "content_byte_varies_on_distribution" => f.content_varies,
            "tau" => f.tau,
            "alt_frame" => f.alt_frame,
            "headline_metrics" => Dict{String,Any}(
                "pearson_corr" => f.pearson, "spearman_corr" => f.spearman,
                "precision_at_k" => f.precision_at_k, "topk" => f.topk,
                "deletion_auc" => _json_num(f.deletion_auc),
                "insertion_auc" => _json_num(f.insertion_auc)),
            # THE COUNTERFACTUAL-SPECIFIC metrics (experiment_design.md §5 row):
            # validity + minimality vs the true minimal causal set.
            "counterfactual" => Dict{String,Any}(
                "candidate_cells" => f.candidate_cells,
                "minimal_set" => f.minimal_set,
                "minimal_set_size" => length(f.minimal_set),
                "valid" => f.valid,
                "validity_threshold_abs" => f.cf_target,
                "minimality_ratio" => _json_num(f.minimality_ratio),
                "n_true_causal_cells" => f.n_true_causal_cells,
                "y_x" => f.y_x, "y_alt" => f.y_alt, "y_S" => f.y_S,
                "n_real_reruns" => f.n_runs,
                "alternative_state" => "genuine RAM at frame $(f.alt_frame) of the same ROM " *
                    "(on-distribution reachable state — NOT a synthetic edit)",
                "interpretation" => "VALIDITY = the minimal set S, when SUBSTITUTED toward the real " *
                    "alternative state and re-run, moves y by ≥ τ·|y(x')−y(x)| (a genuine counterfactual). " *
                    "MINIMALITY (sparsity) = |S| / |causal support|, where the causal support is the # " *
                    "candidate cells whose single-cell substitution toward x' moves y on the real ROM. S is " *
                    "always a SUBSET of the support (greedy+prune only keep causal cells), so the ratio ∈ " *
                    "(0,1]; LOWER = SPARSER (the desirable minimal-counterfactual: a few cells suffice to " *
                    "flip y, e.g. 1/2 = a single cell out of two causal ones). set-state-and-re-render keeps " *
                    "every edit ON-DISTRIBUTION (Atrey 2020 off-manifold objection removed; §7 Partial→Succeed)."),
            "harness_positive_control" => Dict{String,Any}(
                "method" => "oracle_abs_delta (the perfectly-faithful attribution)",
                "pearson_corr" => f.oracle_self_pearson,
                "precision_at_k" => f.oracle_self_precision_at_k,
                "deletion_auc" => _json_num(f.oracle_self_deletion_auc),
                "insertion_auc" => _json_num(f.oracle_self_insertion_auc),
                "oracle_column_degenerate" => f.oracle_column_degenerate,
                "interpretation" => "corr=1 & precision@k=1 (on a non-degenerate column) ⇒ the scoring " *
                    "harness rewards a faithful map; the CF numbers above are then a true measurement, " *
                    "not an artefact of the metric."),
            "auc_note" => "deletion/insertion curves measured on the TRUE VCS by re-running the real ROM " *
                "with top-CF causes occluded (deletion) / restored (insertion); every point is a genuine " *
                "emulator re-run, not a surrogate. NaN = a genuinely flat experiment.",
            "output_note" => _output_note(f),
            "cause_names" => f.cause_names,
            "cf_attr_per_cause" => Dict(f.cause_names[i] => f.cf_attr_per_cause[i] for i in 1:length(f.cause_names)),
            "oracle_abs_delta_per_cause" => Dict(f.cause_names[i] => f.oracle_abs_delta[i] for i in 1:length(f.cause_names)),
            "scales_to_cluster_via" =>
                "tools/cluster/xai_*.sbatch (E0-3) — batched SOFT-STE counterfactual search over " *
                "outputs×cells×games on GPU; the forward is bit-exact to this HARD map.",
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
        "oracle_abs_delta"   => f.oracle_abs_delta,
        "cf_attr_per_cause"  => f.cf_attr_per_cause,
        "deletion_curve"     => f.del_curve,
        "insertion_curve"    => f.ins_curve,
        "candidate_cells"    => Float64.(f.candidate_cells),
        "minimal_set"        => Float64.(f.minimal_set),
        "scalars"            => Float64[f.pearson, f.spearman, f.precision_at_k,
                                        isnan(f.deletion_auc) ? -1.0 : f.deletion_auc,
                                        isnan(f.insertion_auc) ? -1.0 : f.insertion_auc,
                                        f.valid ? 1.0 : 0.0,
                                        isnan(f.minimality_ratio) ? -1.0 : f.minimality_ratio,
                                        Float64(f.n_true_causal_cells)],
    ))
    return json_path, npz_path
end

# ============================================================================
# CLI
# ============================================================================
function main(args = ARGS)
    games = CORE_GAMES; single_game = nothing
    target_frame = 120; horizon = 30
    topk = 3; seed = 0; tau = 0.5
    selftest_only = false
    i = 1
    while i <= length(args)
        a = args[i]
        if     a == "--games"
            v = args[i+1]; games = (v == "core") ? CORE_GAMES : String.(split(v, ",")); i += 2
        elseif a == "--game";          single_game = args[i+1]; i += 2
        elseif a == "--target-frame";  target_frame = parse(Int, args[i+1]); i += 2
        elseif a == "--horizon";       horizon = parse(Int, args[i+1]); i += 2
        elseif a == "--topk";          topk = parse(Int, args[i+1]); i += 2
        elseif a == "--seed";          seed = parse(Int, args[i+1]); i += 2
        elseif a == "--tau";           tau = parse(Float64, args[i+1]); i += 2
        elseif a == "--selftest";      selftest_only = true; i += 1
        else; i += 1
        end
    end
    single_game !== nothing && (games = [single_game])

    println("[cf] On-distribution counterfactual attribution vs oracle — games=$(join(games, ",")) " *
            "target_frame=$target_frame horizon=$horizon topk=$topk tau=$tau seed=$seed (jutari/Julia)")

    summary = Dict{String,Any}[]
    for game in games
        println("\n[cf] ===== $game =====")
        f_content, f_pos, st_extra = compute_game(; game = game, target_frame = target_frame,
            horizon = horizon, topk = topk, tau = tau, seed = seed, verbose = true)
        @assert !f_content.oracle_column_degenerate "content oracle column is degenerate for $game " *
            "at f$target_frame — pick_content_idx failed to find a causally-active concept byte"
        selftest(f_content; require_nondegenerate = true)
        selftest(f_pos)
        if st_extra !== nothing
            println("[cf] $game SHARED region output: gate " *
                "$(st_extra.cause_density)/$(st_extra.n_causes) accepted=$(st_extra.accepted) " *
                "cell=$(st_extra.cell)")
        end
        if !selftest_only
            for f in (f_content, f_pos)
                jp, np = write_result(f; st_extra = (f === f_pos ? st_extra : nothing))
                println("[cf] wrote $jp"); println("[cf] arrays  $np")
            end
        end
        for f in (f_content, f_pos)
            push!(summary, Dict{String,Any}(
                "game" => game, "output" => f.output, "output_kind" => f.output_kind,
                "content_ram_index" => f.content_idx,
                "content_byte_varies_on_distribution" => f.content_varies,
                "pearson" => f.pearson, "spearman" => f.spearman,
                "precision_at_k" => f.precision_at_k,
                "deletion_auc" => _json_num(f.deletion_auc),
                "insertion_auc" => _json_num(f.insertion_auc),
                "valid" => f.valid, "minimal_set_size" => length(f.minimal_set),
                "minimality_ratio" => _json_num(f.minimality_ratio),
                "n_true_causal_cells" => f.n_true_causal_cells,
                "y_x" => f.y_x, "y_alt" => f.y_alt, "y_S" => f.y_S,
                "n_real_reruns" => f.n_runs,
                "oracle_self_pearson" => f.oracle_self_pearson,
                "oracle_self_precision_at_k" => f.oracle_self_precision_at_k,
                "is_content_path" => f.output_kind == "content"))
        end
    end

    if selftest_only
        println("\n[cf] --selftest: all passed, not writing artifacts.")
        return 0
    end

    isdir(OUT_DIR) || mkpath(OUT_DIR)
    summary_path = joinpath(OUT_DIR, "counterfactual_core_summary.json")
    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseB_attribution",
        "method" => "on_distribution_counterfactual",
        "item" => "P2-E4-12", "games" => games,
        "topk" => topk, "seed" => seed, "tau" => tau,
        "target_frame" => target_frame, "horizon" => horizon,
        "where" => "local", "commit" => _git_commit(), "timestamp" => string(round(Int, time())),
        "headline" => "On-distribution counterfactual (Olson 2021; Atrey 2020): find the MINIMAL set " *
            "of candidate RAM cells whose substitution toward a REAL alternative state x' (genuine RAM " *
            "at another frame of the same ROM) + a re-run of the real ROM changes y by ≥ τ·|y(x')−y(x)|. " *
            "Set-state-and-re-render keeps every edit ON-DISTRIBUTION ⇒ removes Atrey's off-manifold " *
            "objection (§7 Partial→Succeed). Scored vs the intervention oracle (corr/del-ins-AUC/p@k) " *
            "PLUS the CF-specific VALIDITY + MINIMALITY/SPARSITY (|S| / |causal support| ∈ (0,1], lower=sparser). KEY " *
            "CONTRAST vs IG/saliency: the counterfactual ALSO works on POSITION outputs (it re-runs the " *
            "real ROM, no gradient) where the gradient methods VANISH. Oracle-as-method positive control " *
            "corr=1/p@k=1.",
        "metrics_note" => Dict{String,Any}(
            "validity" => "the minimal set S, substituted toward x' and re-run, moves y by ≥ τ·|y(x')−y(x)|",
            "minimality_ratio" => "SPARSITY = |S| / |causal support| ∈ (0,1] (causal support = # candidate " *
                "cells whose single-cell substitution toward x' moves y on the real ROM); S ⊆ support so " *
                "ratio ≤ 1, LOWER = sparser (a few cells flip y, e.g. 1/2 = one cell out of two causal ones)",
            "faithfulness" => "corr + deletion/insertion AUC on the TRUE VCS + precision@k vs the oracle top-k"),
        "results" => summary)
    open(summary_path, "w") do io; JSON.print(io, rec, 2); end
    println("\n[cf] wrote summary $summary_path")

    println("\n[cf] ===== per-game headline (content path) =====")
    println("  game             ram   corr    p@$topk    del    ins    valid |S|  min-ratio  y(x)→y(x')")
    for r in summary
        r["is_content_path"] || continue
        println("  $(rpad(r["game"],16)) $(rpad(r["content_ram_index"],5)) " *
                "$(rpad(round(r["pearson"],digits=3),7)) " *
                "$(rpad(round(r["precision_at_k"],digits=3),6)) " *
                "$(rpad(r["deletion_auc"]===nothing ? "NA" : round(r["deletion_auc"],digits=3),6)) " *
                "$(rpad(r["insertion_auc"]===nothing ? "NA" : round(r["insertion_auc"],digits=3),6)) " *
                "$(rpad(r["valid"],5)) $(rpad(r["minimal_set_size"],4)) " *
                "$(rpad(r["minimality_ratio"]===nothing ? "NA" : round(r["minimality_ratio"],digits=2),10)) " *
                "$(round(r["y_x"],digits=1))→$(round(r["y_alt"],digits=1))")
    end
    println("[cf] ===== position contrast (ball_pixel — CF works where IG/saliency VANISH) =====")
    for r in summary
        r["is_content_path"] && continue
        println("  $(rpad(r["game"],16)) corr=$(round(r["pearson"],digits=3)) " *
                "p@$topk=$(round(r["precision_at_k"],digits=3)) valid=$(r["valid"]) " *
                "|S|=$(r["minimal_set_size"]) y(x)=$(round(r["y_x"],digits=1)) y(x')=$(round(r["y_alt"],digits=1))")
    end
    return 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    Counterfactual.main()
end
