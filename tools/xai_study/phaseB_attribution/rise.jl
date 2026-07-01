# rise.jl — Phase-B attribution (P2-E4-9), JULIA path.
#
# RISE (Petsiuk, Das & Saenko, BMVC 2018) — RANDOMIZED-INPUT SAMPLING for
# Explanation — scored against the exact intervention oracle on the 6 CORE games
# (tools/xai_study/common/game_set.json). The 9th attribution method on the
# Phase-B leaderboard, built on the validated Phase-B foundation pinned by the IG
# pilot (P2-E4-0) + ig_baseline_sweep.jl, reusing the faithfulness CONTRACT
# verbatim (experiment_design.md §5 row "RISE (Petsiuk et al. 2018):
# randomized-mask saliency → corr + del/ins").
#
# METHOD — a BLACK-BOX, perturb-and-re-run saliency. RISE never differentiates: it
# probes the model with N RANDOM BINARY MASKS over the candidate cells and reads
# the output each time, then weights each mask by the masked output. Over the
# candidate RAM-cell universe (the SAME cells the oracle scores, so attribution
# lives on the same axis as the ground truth):
#
#   for n = 1..N:  draw m_n ∈ {0,1}^C,  P[m_n[i]=1] = p   (each cell KEPT w.p. p)
#                  ŷ_n = y( do(ram[i] := 0  for every i with m_n[i]=0) )   ← REAL re-run
#   importance(i) = (1 / (N·p)) · Σ_n ŷ_n · m_n[i]                         ← Petsiuk eq.
#
# i.e. cell i's saliency is the EXPECTATION of the output over masks that KEEP cell
# i, with the 1/(N·p) importance-sampling normalisation (Petsiuk §3.2). A cell that
# raises y whenever it is present gets a high score; an inert cell averages to the
# global mean. Each ŷ_n is a GENUINE re-run of the real ROM through the bit-exact
# oracle machinery (no surrogate, no world model, no gradient) — so RISE applies
# UNCHANGED to the §1 POSITION/INDEX output (ball_pixel) where the gradient methods
# VANISH (the §7 Partial→Succeed contrast it shares with Occlusion / the
# counterfactual). "Masking to 0" is RISE's standard occlusion baseline and is
# EXACTLY the oracle's own occlude! operator (RAM→0), so the masked re-runs are the
# same do()-interventions the oracle uses ⇒ apples-to-apples with del/ins.
#
# BUDGET (the §R-recorded knob) — N = 500 masks at keep-prob p = 0.5 (the paper's
# default mask density). RISE's variance ∝ 1/N, so we also report a CONVERGENCE
# SWEEP: the faithfulness (corr / precision@k / del-ins) recomputed at N/8, N/4,
# N/2, N from the SAME mask pool (an incremental prefix — no extra re-runs), so the
# record shows HOW the attribution scales with the mask count. The masks are drawn
# from a SEEDED RNG (recorded), so the run is reproducible.
#
# OUTPUT SELECTION (the §1 content-vs-position split, mirroring the siblings):
#   * HEADLINE  = a CONTENT output: the candidate CONCEPT BYTE the oracle ranks as
#     the most causally-active RAM cause (SAME pick as the IG/EG/saliency/CF
#     siblings, via pick_content_idx), read straight off RAM. RISE concentrates its
#     output-weighted mass on the cells that actually move that byte.
#   * CONTRAST  = a POSITION/INDEX output `ball_pixel`: RISE STILL produces a real
#     attribution here (it re-runs the real ROM and reads the real pixel), unlike
#     IG/saliency which vanish — the "perturbation methods survive position outputs"
#     contrast (§7), reported not asserted.
#
# REUSES the validated Phase-B foundation on main (NO emulator core touched):
#   * ig_baseline_sweep.jl — the multi-game env layer (load_env, boot_replay,
#     continue_from, fresh_baseline, assert_bit_exact, occlude!, oracle_abs_delta,
#     deletion_insertion_auc with a generic read_y, position_pixel_cell,
#     pick_content_idx, candidates_path_for, CORE_GAMES) — the SAME machinery the
#     IG/EG/saliency/CF siblings use, so the oracle column + del/ins curves are
#     apples-to-apples with them.
#   * pilot_ig_vs_oracle.jl — the SCORER (pearson/spearman/precision_at_k), the §R
#     writer helpers (_git_commit/_json_num/_trapz_unit), the harness
#     positive-control idea (oracle-as-method ⇒ corr=1/p@k=1).
#   * oracle_intervene.jl — Cause / build_pong_causes / candidate_ram_indices.
#   * jutari_oracle.jl — boot/replay/snapshot/intervene + the §R NPZ writer.
#
# Run (warm shared depot, primary's project) — DEFAULT = all 6 core games locally:
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseB_attribution/rise.jl --games core
# Flags: --games core|<g1,g2,...>  --game <g>
#        --target-frame N --horizon N --n-masks N --keep-prob P --topk K --seed S
#        --selftest
# Cluster-shardable (Slurm array via tools/cluster/xai_array_jl.sbatch):
#        --shard <i> --nshards <n> --shard-kind game   (shard i → the i-th core game)
#        --out-dir <dir>   (where the §R records land; default = ./out)
#        (--roms-dir / --where accepted-and-ignored: ROMs located internally.)
#
# Writes (SPEC §R; file_scope rise_* under the out dir):
#   <out>/rise_<game>_content.{json,npz}
#   <out>/rise_<game>_ball_pixel.{json,npz}
#   <out>/rise_core_summary.json

module RISE

using JSON
import Statistics
import Random

# the Phase-B foundation (env layer + scorer + del/ins + oracle |Δy|), identical to
# the IG/EG/saliency/CF siblings — REUSED, not re-implemented.
include(joinpath(@__DIR__, "ig_baseline_sweep.jl"))
using .IGBaselineSweep: CORE_GAMES, load_env, boot_replay, continue_from,
                        fresh_baseline, assert_bit_exact, occlude!,
                        oracle_abs_delta, deletion_insertion_auc, position_pixel_cell,
                        pick_content_idx, candidates_path_for
# The P2 SHARED TESTBED handles — reached through the SAME single ig_baseline_sweep
# include chain (which already includes shared_testbed_impl.jl ONCE), so
# build_shared_testbed operates on OUR own Cause/Snapshot types. RISE is
# INTERVENTION/MASK-based, so it needs only the shared gameplay STATE + the shared
# REGION read_y for the position output (no gradient, no sampler).
using .IGBaselineSweep: build_shared_testbed, SHARED_TESTBED,
                        ST_PREFIX, ST_HORIZON, ST_SEED, ST_GATE_K, ST_FLOOR,
                        settings_for, rom_path_for, run_intervention,
                        env_step!, soft_ram_peek
# NOTE: intervene_ram! + snapshot come in below via the JutariOracle line (same
# single include chain ⇒ same identity); do NOT re-import them here.
using .IGBaselineSweep.PilotIGvsOracle: pearson, spearman, precision_at_k,
                                        _git_commit, _json_num, _trapz_unit
using .IGBaselineSweep.PilotIGvsOracle.OracleIntervene: build_pong_causes, Cause,
                                                        candidate_ram_indices
using .IGBaselineSweep.PilotIGvsOracle.OracleIntervene.JutariOracle: Snapshot, snapshot,
                                                        intervene_ram!, write_npz, RAM_SIZE

const OUT_DIR = joinpath(@__DIR__, "out")

# ============================================================================
# The RISE machinery — all on the TRUE VCS (perturb-and-re-run; no gradient).
# ============================================================================
# We work over the UNIVERSE of candidate RAM cells (the oracle's candidate concept
# bytes — the SAME cells the oracle scores, so attribution lives on the same axis
# as the ground truth). RISE masks a subset to 0 (the standard occlusion baseline =
# the oracle's own occlude! operator) and re-runs the real ROM. Each call is a
# genuine re-run.

"""
    masked_read_y(checkpoint, actions, tf, hz, drop_idx, read_y) -> Float64

Re-run y under the RISE mask `do(ram[i] := 0 for i in drop_idx)`: deepcopy the
checkpoint, zero each DROPPED cell (the cells whose mask bit is 0), continue the
horizon on the real ROM, read y. `drop_idx` is a collection of 0-based RAM indices.
Empty ⇒ the intact y(x). Masking-to-0 is RISE's standard occlusion AND the oracle's
own occlude! (RAM→0), so these probes are the oracle's do()-interventions."""
function masked_read_y(checkpoint, actions, tf, hz, drop_idx, read_y)
    env = deepcopy(checkpoint)
    for i in drop_idx
        intervene_ram!(env, i, 0)
    end
    tail = Int.(actions[tf + 1 : tf + hz])
    for a in tail; IGBaselineSweep.env_step!(env, a); end
    return read_y(snapshot(env, length(tail)))
end

"""
    rise_sample(checkpoint, actions, tf, hz, cells, read_y; n_masks, keep_prob, rng)
        -> (masks::BitMatrix, ys::Vector{Float64})

Draw `n_masks` random binary masks over the candidate `cells` (each cell KEPT with
probability `keep_prob`) and, for each mask, RE-RUN the real ROM with the dropped
cells masked to 0, reading y. Returns the N×C mask matrix (rows = masks, columns =
cells; `true` = cell kept) and the N masked outputs. Every output is a genuine
emulator re-run — this is the entire compute budget."""
function rise_sample(checkpoint, actions, tf, hz, cells, read_y;
                     n_masks::Integer, keep_prob::Real, rng)
    C = length(cells)
    masks = falses(n_masks, C)
    ys = zeros(Float64, n_masks)
    for n in 1:n_masks
        # draw the mask: cell kept w.p. keep_prob (so it is PRESENT in the run);
        # the dropped cells (bit 0) are the ones we occlude to 0.
        keep = rand(rng, C) .< keep_prob
        masks[n, :] = keep
        drop = [cells[j] for j in 1:C if !keep[j]]
        ys[n] = masked_read_y(checkpoint, actions, tf, hz, drop, read_y)
    end
    return masks, ys
end

"""
    rise_importance(masks, ys; keep_prob) -> Vector{Float64}

The RISE saliency over the candidate cells from the mask/output pool (Petsiuk
§3.2): importance(i) = (1/(N·p)) · Σ_n y_n · m_n[i] — the output-weighted mean
mask, with the 1/(N·p) importance-sampling normalisation so the estimate is the
conditional expectation E[y | cell i present] (up to the constant p). N is the #
masks (rows of `masks`); p is `keep_prob`. Operates on a PREFIX of the pool when
`masks`/`ys` are sliced — that is the convergence sweep (no extra re-runs)."""
function rise_importance(masks::AbstractMatrix{Bool}, ys::AbstractVector{<:Real};
                         keep_prob::Real)
    N = size(masks, 1); C = size(masks, 2)
    N == 0 && return zeros(Float64, C)
    imp = zeros(Float64, C)
    @inbounds for j in 1:C
        acc = 0.0
        for n in 1:N
            masks[n, j] && (acc += ys[n])
        end
        imp[j] = acc / (N * keep_prob)
    end
    return imp
end

"""Map the per-cell RISE saliency onto the oracle's per-CAUSE axis: a RAM cause
touching candidate cell i gets cell i's RISE importance; a TIA/joystick cause gets
0 (the RISE universe is RAM cells — the method honestly assigns no mass off it).
We CENTER the per-cell importances first (subtract the mean over cells) so the map
measures DEVIATION from the global average output — the cells that genuinely raise
y stand out, mirroring RISE's heat-map normalisation; an inert cell → ~0."""
function rise_attr_per_cause(imp_cells::AbstractVector{<:Real}, cells, causes::Vector{Cause})
    centered = imp_cells .- Statistics.mean(imp_cells)
    by_idx = Dict(cells[j] => centered[j] for j in 1:length(cells))
    return [c.kind == "ram" ? abs(get(by_idx, c.index, 0.0)) : 0.0 for c in causes]
end

# ============================================================================
# Per-output result + the convergence sweep.
# ============================================================================
struct ConvergencePoint
    n_masks::Int
    pearson::Float64
    spearman::Float64
    precision_at_k::Float64
    deletion_auc::Float64
    insertion_auc::Float64
end

struct RISEResult
    game::String
    output::String
    output_kind::String                  # "content" | "position"
    target_frame::Int
    horizon::Int
    n_masks::Int                         # the BUDGET (full mask count)
    keep_prob::Float64
    topk::Int
    seed::Int
    content_idx::Int
    cause_names::Vector{String}
    candidate_cells::Vector{Int}
    oracle_abs_delta::Vector{Float64}
    rise_attr_per_cause::Vector{Float64} # the headline RISE map (at the full budget)
    rise_importance_cells::Vector{Float64}
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
    n_real_reruns::Int                   # = n_masks (every mask is one re-run)
    y_intact::Float64
    oracle_column_degenerate::Bool
    # harness positive control (oracle's OWN |Δy| as the method)
    oracle_self_pearson::Float64
    oracle_self_precision_at_k::Float64
    oracle_self_deletion_auc::Float64
    oracle_self_insertion_auc::Float64
end

"""Score one RISE map (built from N masks) against the oracle with the §5 metrics."""
function _score_map(attr, odelta, checkpoint, actions, tf, hz, causes, read_y, topk)
    pr  = pearson(attr, odelta)
    sp  = spearman(attr, odelta)
    pak = precision_at_k(attr, odelta, topk)
    order = sortperm(attr; rev = true)
    del_auc, ins_auc, dc, ic = deletion_insertion_auc(
        checkpoint, actions, tf, hz, causes, order, read_y)
    return pr, sp, pak, del_auc, ins_auc, dc, ic
end

"""Compute the RISE attribution + faithfulness + convergence sweep for one output
of one game. `read_y` reads the chosen output off a Snapshot; `output_kind` is
"content"/"position"; `cells` is the candidate RAM-cell universe (0-based)."""
function compute_one(; game, output_kind, content_idx = -1, position_cell = nothing,
                     checkpoint, actions, target_frame, horizon, causes, candidate_cells,
                     n_masks, keep_prob, topk, seed, verbose,
                     read_y_override = nothing, output_name_override = nothing)
    # reader for the oracle / mask re-runs / del-ins (reads the TRUE VCS state). In the
    # SHARED TESTBED the position output is the screen-buffer REGION (n_changed_px)
    # supplied by read_y_override (redesign Problem 4 — one shared output all methods
    # explain). RISE is intervention/mask-based, so no gradient path changes.
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

    # 2) intact y(x)
    base_snap = continue_from(checkpoint, Int.(actions[target_frame + 1 : target_frame + horizon]))
    y_intact = read_y(base_snap)

    # 3) RISE: draw N random masks over the candidate cells, re-run the real ROM for
    #    each (THE BUDGET — every mask is one genuine emulator re-run), collect y.
    verbose && println("[rise] $game '$output_name': sampling $n_masks random masks " *
                       "(keep_prob=$keep_prob) over $(length(candidate_cells)) candidate cells " *
                       "[$n_masks real re-runs]...")
    rng = Random.MersenneTwister(seed * 1_000_003 + 7919)  # deterministic, recorded seed
    masks, ys = rise_sample(checkpoint, actions, target_frame, horizon, candidate_cells, read_y;
                            n_masks = n_masks, keep_prob = keep_prob, rng = rng)

    # 4) the headline RISE map at the FULL budget + its faithfulness
    imp_full = rise_importance(masks, ys; keep_prob = keep_prob)
    attr_full = rise_attr_per_cause(imp_full, candidate_cells, causes)
    pr, sp, pak, del_auc, ins_auc, dc, ic =
        _score_map(attr_full, odelta, checkpoint, actions, target_frame, horizon, causes, read_y, topk)

    # 5) the CONVERGENCE sweep: recompute faithfulness from PREFIXES of the SAME
    #    mask pool (N/8, N/4, N/2, N) — no extra re-runs; shows how RISE scales with
    #    the mask count (its variance ∝ 1/N). del/ins use the prefix ranking.
    conv = ConvergencePoint[]
    for frac in (8, 4, 2, 1)
        nk = max(1, n_masks ÷ frac)
        impk = rise_importance(view(masks, 1:nk, :), view(ys, 1:nk); keep_prob = keep_prob)
        attrk = rise_attr_per_cause(impk, candidate_cells, causes)
        prk, spk, pakk, delk, insk, _, _ =
            _score_map(attrk, odelta, checkpoint, actions, target_frame, horizon, causes, read_y, topk)
        push!(conv, ConvergencePoint(nk, prk, spk, pakk, delk, insk))
        nk == n_masks && break    # avoid a duplicate at the full budget
    end

    # 6) harness positive control (oracle's OWN |Δy| as the candidate map)
    or_pr  = pearson(odelta, odelta); or_pak = precision_at_k(odelta, odelta, topk)
    or_order = sortperm(odelta; rev = true)
    or_del, or_ins, _, _ = deletion_insertion_auc(checkpoint, actions, target_frame,
                                                  horizon, causes, or_order, read_y)

    if verbose
        println("[rise]   faithfulness vs oracle @N=$n_masks: corr=$(round(pr,digits=4)) " *
                "spearman=$(round(sp,digits=4)) p@$topk=$(round(pak,digits=3)) " *
                "del=$(round(del_auc,digits=3)) ins=$(round(ins_auc,digits=3))")
        print("[rise]   convergence corr@N:")
        for c in conv; print(" $(c.n_masks)=$(round(c.pearson,digits=3))"); end
        println()
        println("[rise]   [harness] oracle-as-method: corr=$(round(or_pr,digits=3)) " *
                "p@$topk=$(round(or_pak,digits=3)) del=$(round(or_del,digits=3)) ins=$(round(or_ins,digits=3))" *
                (degenerate ? "  (oracle column flat at this state)" : ""))
    end

    return RISEResult(game, output_name, output_kind, target_frame, horizon, n_masks,
                      Float64(keep_prob), topk, seed, content_idx, [c.name for c in causes],
                      collect(candidate_cells), odelta, attr_full, imp_full,
                      pr, sp, pak, del_auc, ins_auc, dc, ic, conv, n_masks, y_intact,
                      degenerate, or_pr, or_pak, or_del, or_ins)
end

"""Drive both outputs for one game: assert bit-exact, build causes + candidate
cells, pick the content byte, locate the position pixel, run RISE for content +
position."""
function compute_game(; game, target_frame, horizon, n_masks, keep_prob, topk, seed, verbose)
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
        verbose && println("[rise] $game SHARED gameplay state: " *
            "cause_density=$(st.cause_density)/$(length(st.causes)) accepted=$(st.accepted) cell=$(st.cell)")

        actions = st.actions; checkpoint = st.checkpoint; causes = st.causes
        tf = st.prefix; hz = st.horizon
        # RISE samples over the candidate RAM-cell universe (the oracle's candidate
        # concept bytes) — the SAME cells for BOTH outputs; independent of the position
        # cell, so no candidate-cell substitution is needed for the region output.
        candidate_cells = sort(unique(st.cand_indices))
        content_idx, content_mv = pick_content_idx(checkpoint, actions, tf, hz, causes, st.cand_indices)
        verbose && println("[rise] $game content byte = RAM[$content_idx] (max oracle |Δself|=$(round(content_mv,digits=2)))")

        f_content = compute_one(; game = game, output_kind = "content", content_idx = content_idx,
            checkpoint = checkpoint, actions = actions, target_frame = tf, horizon = hz,
            causes = causes, candidate_cells = candidate_cells, n_masks = n_masks, keep_prob = keep_prob,
            topk = topk, seed = seed, verbose = verbose)
        f_pos = compute_one(; game = game, output_kind = "position",
            checkpoint = checkpoint, actions = actions, target_frame = tf, horizon = hz,
            causes = causes, candidate_cells = candidate_cells, n_masks = n_masks, keep_prob = keep_prob,
            topk = topk, seed = seed, verbose = verbose,
            read_y_override = st.read_y,
            output_name_override = "screen_region(n_changed_px)@r$(st.cell[1])c$(st.cell[2])")
        st_extra = (cause_density = st.cause_density, accepted = st.accepted,
                    n_causes = length(st.causes), cell = st.cell,
                    prefix = st.prefix, horizon = st.horizon, seed = st.seed)
        return f_content, f_pos, st_extra
    end

    total = target_frame + horizon
    actions = fill(0, total)
    verbose && println("[rise] $game: asserting bit-exactness (2 fresh boots+replays to f$total)...")
    assert_bit_exact(actions, total; game = game)

    cand = candidates_path_for(game)
    checkpoint = boot_replay(actions, target_frame; game = game)
    at_target = continue_from(checkpoint, Int[])
    causes = build_pong_causes(cand, at_target)
    candidate_cells = sort(unique(idx for (idx, _) in candidate_ram_indices(cand)))

    # content byte = the most causally-active candidate concept byte (SAME pick as
    # the IG/EG/saliency/CF siblings ⇒ the methods score the SAME content output).
    content_idx, content_mv = pick_content_idx(checkpoint, actions, target_frame, horizon, causes,
                                               [idx for (idx, _) in candidate_ram_indices(cand)])
    verbose && println("[rise] $game content byte = RAM[$content_idx] (max oracle |Δself|=$(round(content_mv,digits=2)))")

    base = continue_from(checkpoint, Int.(actions[target_frame + 1 : total]))
    pcell = position_pixel_cell(checkpoint, base.screen; horizon = horizon)

    f_content = compute_one(; game = game, output_kind = "content", content_idx = content_idx,
        checkpoint = checkpoint, actions = actions, target_frame = target_frame, horizon = horizon,
        causes = causes, candidate_cells = candidate_cells, n_masks = n_masks, keep_prob = keep_prob,
        topk = topk, seed = seed, verbose = verbose)
    f_pos = compute_one(; game = game, output_kind = "position", position_cell = pcell,
        checkpoint = checkpoint, actions = actions, target_frame = target_frame, horizon = horizon,
        causes = causes, candidate_cells = candidate_cells, n_masks = n_masks, keep_prob = keep_prob,
        topk = topk, seed = seed, verbose = verbose)
    return f_content, f_pos, nothing
end

# ============================================================================
# Self-check (DoD) — the scoring contract is sound; results are non-fabricated.
# ============================================================================
"""
    selftest(f::RISEResult; require_nondegenerate) -> Bool

Asserts the load-bearing claims of RISE + the shared contract:

  (HARNESS POSITIVE CONTROL) feeding the oracle's OWN |Δy| as the candidate map
    scores corr=1 (on a non-degenerate column) and precision@k=1 — the harness
    rewards a faithful map (else the harness is broken, not the method).

  (RISE CORRECTNESS)
    * attribution finite + non-negative (it is |centered output-weighted mean|);
      all AUCs ∈ [0,1] or NaN.
    * the budget is real: n_real_reruns == n_masks (every mask is one re-run).
    * the convergence sweep is monotone in N (the prefixes grow) and ends at the
      full budget — recorded, the headline being that faithfulness stabilises as N
      grows (reported, not asserted to be large).

  (POSITION OUTPUT — the §7 contrast) RISE still produces a real attribution on the
    position pixel (it re-runs the real ROM, no gradient) — recorded, not asserted
    to be large; the contrast with IG/saliency which vanish there.

Throws on a violation."""
function selftest(f::RISEResult; require_nondegenerate = false)
    @assert all(isfinite, f.rise_attr_per_cause) "non-finite RISE attribution ($(f.game)/$(f.output))"
    @assert all(>=(0.0), f.rise_attr_per_cause) "RISE attribution must be non-negative ($(f.game)/$(f.output))"
    for (nm, v) in (("deletion", f.deletion_auc), ("insertion", f.insertion_auc),
                    ("oracle_self_deletion", f.oracle_self_deletion_auc),
                    ("oracle_self_insertion", f.oracle_self_insertion_auc))
        @assert isnan(v) || (0.0 <= v <= 1.0 + 1e-9) "$nm AUC out of [0,1]: $v ($(f.game)/$(f.output))"
    end
    @assert f.n_real_reruns == f.n_masks "budget mismatch: n_real_reruns=$(f.n_real_reruns) != n_masks=$(f.n_masks)"
    @assert !isempty(f.convergence) "empty convergence sweep ($(f.game)/$(f.output))"
    @assert f.convergence[end].n_masks == f.n_masks "convergence sweep does not reach the full budget"
    for c in f.convergence
        @assert isnan(c.deletion_auc) || (0.0 <= c.deletion_auc <= 1.0 + 1e-9) "convergence del AUC out of range"
        @assert isnan(c.insertion_auc) || (0.0 <= c.insertion_auc <= 1.0 + 1e-9) "convergence ins AUC out of range"
    end

    if f.output_kind == "position"
        println("[rise] SELF-CHECK PASS (position '$(f.output)', $(f.game)): " *
                "RISE map finite (max|attr|=$(round(maximum(f.rise_attr_per_cause),sigdigits=3))) over " *
                "$(f.n_masks) masks; corr=$(round(f.pearson,digits=3)); " *
                "oracle-as-method corr=$(round(f.oracle_self_pearson,digits=3)) " *
                "(the §7 contrast: RISE re-runs the real ROM, so it works on a position output where " *
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
    conv_str = join(["$(c.n_masks)=$(round(c.pearson,digits=3))" for c in f.convergence], " ")
    println("[rise] SELF-CHECK PASS (content '$(f.output)', $(f.game)): " *
            "RISE map finite + non-negative over $(f.n_masks) masks (keep_prob=$(f.keep_prob)); " *
            "corr=$(round(f.pearson,digits=3)) p@$(f.topk)=$(round(f.precision_at_k,digits=3)) " *
            "del=$(round(f.deletion_auc,digits=3)) ins=$(round(f.insertion_auc,digits=3)); " *
            "convergence corr@N [$conv_str]; oracle-as-method corr=$(round(f.oracle_self_pearson,digits=3)).")
    return true
end

# ============================================================================
# Persist (SPEC §R) — JSON record + sibling .npz; file_scope rise_*.
# ============================================================================
function _output_note(f::RISEResult)
    if f.output_kind == "position"
        return "POSITION/INDEX output (ball_pixel) — the §7 CONTRAST: unlike IG/saliency (which " *
               "VANISH on position outputs because the pixel is a round/argmax sprite column with no " *
               "differentiable RAM path), RISE — a BLACK-BOX perturb-and-re-run method — STILL " *
               "produces a real attribution here. It masks candidate RAM cells to 0 and RE-RUNS the " *
               "real ROM, reading the real pixel; no differentiation needed. Reported, not asserted " *
               "to be large; the intervention oracle remains the sole truth."
    else
        conv_corr = [c.pearson for c in f.convergence]
        return "CONTENT output: RAM byte $(f.content_idx) (the most causally-active candidate concept " *
               "byte — SAME pick as the IG/EG/saliency/CF siblings). RISE (Petsiuk 2018) draws " *
               "$(f.n_masks) RANDOM BINARY MASKS (keep_prob=$(f.keep_prob)) over the candidate cells, " *
               "masks the dropped cells to 0, RE-RUNS the real ROM, and weights each mask by the " *
               "masked output: importance(i)=(1/(N·p))·Σ y·m[i]. Faithfulness corr=$(round(f.pearson,digits=3)) " *
               "@N=$(f.n_masks); convergence corr grows $(round(first(conv_corr),digits=3))→" *
               "$(round(last(conv_corr),digits=3)) as N goes $(f.convergence[1].n_masks)→$(f.n_masks) " *
               "(RISE variance ∝ 1/N). Every probe is a genuine emulator re-run (RAM→0 = the oracle's " *
               "own occlude!), so RISE is apples-to-apples with the del/ins curves."
    end
end

function write_result(f::RISEResult; out_dir = OUT_DIR, st_extra = nothing)
    isdir(out_dir) || mkpath(out_dir)
    tag = f.output_kind == "content" ? "content" : "ball_pixel"
    stem = "rise_$(f.game)_$(tag)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path  = joinpath(out_dir, stem * ".npz")

    conv_json = [Dict{String,Any}(
        "n_masks" => c.n_masks, "pearson_corr" => c.pearson, "spearman_corr" => c.spearman,
        "precision_at_k" => c.precision_at_k,
        "deletion_auc" => _json_num(c.deletion_auc), "insertion_auc" => _json_num(c.insertion_auc),
    ) for c in f.convergence]

    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseB_attribution", "method" => "rise",
        "game" => f.game, "state" => "f$(f.target_frame)+$(f.horizon)",
        "target_output" => f.output,
        # headline §R scalar = the RISE map's corr with the oracle (one comparable
        # number per method on the leaderboard, like the siblings).
        "metric_name" => "pearson_corr_with_oracle",
        "value" => f.pearson,
        "stderr" => nothing, "ci" => nothing, "n" => length(f.cause_names),
        "seed" => f.seed, "where" => "local", "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene@$(f.game)#$(f.output)",
        "timestamp" => string(round(Int, time())), "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            "substrate" => "jutari (Julia, HARD) — RISE (Petsiuk 2018) randomized-mask saliency: draw " *
                "N random binary masks over the candidate RAM cells, mask the dropped cells to 0, RE-RUN " *
                "the real ROM (bit-exact oracle machinery), weight each mask by the masked output. Every " *
                "probe is a genuine emulator re-run; no gradient, no surrogate.",
            "output_kind" => f.output_kind,
            "content_ram_index" => f.content_idx,
            # THE BUDGET (recorded per §R) — the perturb-and-re-run cost knob.
            "budget" => Dict{String,Any}(
                "n_masks" => f.n_masks,
                "keep_prob" => f.keep_prob,
                "n_candidate_cells" => length(f.candidate_cells),
                "n_real_reruns" => f.n_real_reruns,
                "rng" => "MersenneTwister(seed*1_000_003 + 7919), seed=$(f.seed)",
                "note" => "N=$(f.n_masks) masks at keep_prob=$(f.keep_prob) (Petsiuk's default mask " *
                    "density). Each mask = ONE real emulator re-run, so the cost is exactly N re-runs " *
                    "per output. RISE variance ∝ 1/N — see convergence_sweep."),
            "headline_metrics" => Dict{String,Any}(
                "pearson_corr" => f.pearson, "spearman_corr" => f.spearman,
                "precision_at_k" => f.precision_at_k, "topk" => f.topk,
                "deletion_auc" => _json_num(f.deletion_auc),
                "insertion_auc" => _json_num(f.insertion_auc)),
            # how faithfulness SCALES with the mask count (prefixes of one pool).
            "convergence_sweep" => Dict{String,Any}(
                "points" => conv_json,
                "interpretation" => "faithfulness (corr / precision@k / del-ins) recomputed from " *
                    "N/8, N/4, N/2, N masks of the SAME pool (no extra re-runs). RISE's Monte-Carlo " *
                    "variance is ∝ 1/N, so corr/p@k STABILISE as N grows; the spread between the " *
                    "smallest and full budget quantifies how many masks RISE needs on this output."),
            "harness_positive_control" => Dict{String,Any}(
                "method" => "oracle_abs_delta (the perfectly-faithful attribution)",
                "pearson_corr" => f.oracle_self_pearson,
                "precision_at_k" => f.oracle_self_precision_at_k,
                "deletion_auc" => _json_num(f.oracle_self_deletion_auc),
                "insertion_auc" => _json_num(f.oracle_self_insertion_auc),
                "oracle_column_degenerate" => f.oracle_column_degenerate,
                "interpretation" => "corr=1 & precision@k=1 (on a non-degenerate column) ⇒ the scoring " *
                    "harness rewards a faithful map; the RISE numbers above are then a true measurement, " *
                    "not an artefact of the metric."),
            "auc_note" => "deletion/insertion curves measured on the TRUE VCS by re-running the real ROM " *
                "with top-RISE causes occluded (deletion) / restored (insertion); every point is a genuine " *
                "emulator re-run, not a surrogate. NaN = a genuinely flat experiment.",
            "output_note" => _output_note(f),
            "cause_names" => f.cause_names,
            "rise_attr_per_cause" => Dict(f.cause_names[i] => f.rise_attr_per_cause[i] for i in 1:length(f.cause_names)),
            "oracle_abs_delta_per_cause" => Dict(f.cause_names[i] => f.oracle_abs_delta[i] for i in 1:length(f.cause_names)),
            "y_intact" => f.y_intact,
            "scales_to_cluster_via" =>
                "tools/cluster/xai_array_jl.sbatch (--shard i --nshards n --shard-kind game): one Slurm " *
                "array task per core game; or batched SOFT-STE masked re-runs over masks×cells×games on " *
                "GPU (the forward is bit-exact to this HARD map).",
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
        "oracle_abs_delta"       => f.oracle_abs_delta,
        "rise_attr_per_cause"    => f.rise_attr_per_cause,
        "rise_importance_cells"  => f.rise_importance_cells,
        "candidate_cells"        => Float64.(f.candidate_cells),
        "deletion_curve"         => f.del_curve,
        "insertion_curve"        => f.ins_curve,
        "convergence_n_masks"    => Float64[c.n_masks for c in f.convergence],
        "convergence_pearson"    => Float64[c.pearson for c in f.convergence],
        "convergence_spearman"   => Float64[c.spearman for c in f.convergence],
        "convergence_precision_at_k" => Float64[c.precision_at_k for c in f.convergence],
        "convergence_deletion_auc"   => Float64[isnan(c.deletion_auc) ? -1.0 : c.deletion_auc for c in f.convergence],
        "convergence_insertion_auc"  => Float64[isnan(c.insertion_auc) ? -1.0 : c.insertion_auc for c in f.convergence],
        "scalars"                => Float64[f.pearson, f.spearman, f.precision_at_k,
                                            isnan(f.deletion_auc) ? -1.0 : f.deletion_auc,
                                            isnan(f.insertion_auc) ? -1.0 : f.insertion_auc,
                                            Float64(f.n_masks), f.keep_prob],
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
        kind == "game" || error("rise.jl only shards by game (--shard-kind game), got $kind")
        n = nshards === nothing ? length(CORE_GAMES) : nshards
        # shard i (0-based) → every game at position i, i+n, i+2n, ... (round-robin),
        # so any nshards ≤ |core| picks the i-th core game and nshards=|core| is 1-per.
        sel = [CORE_GAMES[j] for j in (shard + 1):n:length(CORE_GAMES)]
        isempty(sel) && error("shard $shard of $n selects no core game (|core|=$(length(CORE_GAMES)))")
        return sel
    end
    return CORE_GAMES
end

function main(args = ARGS)
    single_game = nothing; games_arg = nothing
    target_frame = 120; horizon = 30
    n_masks = 500; keep_prob = 0.5
    topk = 3; seed = 0
    shard = nothing; nshards = nothing; shard_kind = nothing
    out_dir = OUT_DIR
    selftest_only = false
    i = 1
    while i <= length(args)
        a = args[i]
        if     a == "--games"
            v = args[i+1]; games_arg = (v == "core") ? CORE_GAMES : String.(split(v, ",")); i += 2
        elseif a == "--game";          single_game = args[i+1]; i += 2
        elseif a == "--target-frame";  target_frame = parse(Int, args[i+1]); i += 2
        elseif a == "--horizon";       horizon = parse(Int, args[i+1]); i += 2
        elseif a == "--n-masks";       n_masks = parse(Int, args[i+1]); i += 2
        elseif a == "--keep-prob";     keep_prob = parse(Float64, args[i+1]); i += 2
        elseif a == "--topk";          topk = parse(Int, args[i+1]); i += 2
        elseif a == "--seed";          seed = parse(Int, args[i+1]); i += 2
        elseif a == "--shard";         shard = parse(Int, args[i+1]); i += 2
        elseif a == "--nshards";       nshards = parse(Int, args[i+1]); i += 2
        elseif a == "--shard-kind";    shard_kind = args[i+1]; i += 2
        elseif a == "--out-dir";       out_dir = args[i+1]; i += 2
        elseif a == "--roms-dir";      i += 2            # accepted-and-ignored (ROMs located internally)
        elseif a == "--where";         i += 2            # accepted-and-ignored (record always says where=local/cluster via env)
        elseif a == "--selftest";      selftest_only = true; i += 1
        else; i += 1
        end
    end
    games = _resolve_games(; single_game = single_game, games_arg = games_arg,
                           shard = shard, nshards = nshards, shard_kind = shard_kind)

    println("[rise] RISE (Petsiuk 2018) randomized-mask saliency vs oracle — games=$(join(games, ",")) " *
            "target_frame=$target_frame horizon=$horizon n_masks=$n_masks keep_prob=$keep_prob " *
            "topk=$topk seed=$seed out_dir=$out_dir (jutari/Julia)")

    summary = Dict{String,Any}[]
    for game in games
        println("\n[rise] ===== $game =====")
        f_content, f_pos, st_extra = compute_game(; game = game, target_frame = target_frame,
            horizon = horizon, n_masks = n_masks, keep_prob = keep_prob,
            topk = topk, seed = seed, verbose = true)
        @assert !f_content.oracle_column_degenerate "content oracle column is degenerate for $game " *
            "at f$target_frame — pick_content_idx failed to find a causally-active concept byte"
        selftest(f_content; require_nondegenerate = true)
        selftest(f_pos)
        if st_extra !== nothing
            println("[rise] $game SHARED region output: gate " *
                "$(st_extra.cause_density)/$(st_extra.n_causes) accepted=$(st_extra.accepted) " *
                "cell=$(st_extra.cell)")
        end
        if !selftest_only
            for f in (f_content, f_pos)
                jp, np = write_result(f; out_dir = out_dir,
                    st_extra = (f === f_pos ? st_extra : nothing))
                println("[rise] wrote $jp"); println("[rise] arrays  $np")
            end
        end
        for f in (f_content, f_pos)
            push!(summary, Dict{String,Any}(
                "game" => game, "output" => f.output, "output_kind" => f.output_kind,
                "content_ram_index" => f.content_idx,
                "n_masks" => f.n_masks, "keep_prob" => f.keep_prob,
                "pearson" => f.pearson, "spearman" => f.spearman,
                "precision_at_k" => f.precision_at_k,
                "deletion_auc" => _json_num(f.deletion_auc),
                "insertion_auc" => _json_num(f.insertion_auc),
                "convergence_pearson" => [c.pearson for c in f.convergence],
                "convergence_n_masks" => [c.n_masks for c in f.convergence],
                "oracle_self_pearson" => f.oracle_self_pearson,
                "oracle_self_precision_at_k" => f.oracle_self_precision_at_k,
                "is_content_path" => f.output_kind == "content"))
        end
    end

    if selftest_only
        println("\n[rise] --selftest: all passed, not writing artifacts.")
        return 0
    end

    isdir(out_dir) || mkpath(out_dir)
    summary_path = joinpath(out_dir, "rise_core_summary.json")
    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseB_attribution", "method" => "rise",
        "item" => "P2-E4-9", "games" => games,
        "n_masks" => n_masks, "keep_prob" => keep_prob, "topk" => topk, "seed" => seed,
        "target_frame" => target_frame, "horizon" => horizon,
        "where" => "local", "commit" => _git_commit(), "timestamp" => string(round(Int, time())),
        "headline" => "RISE (Petsiuk, Das & Saenko 2018): randomized-mask saliency — draw N=$n_masks " *
            "random binary masks over the candidate RAM cells (keep_prob=$keep_prob), mask the dropped " *
            "cells to 0, RE-RUN the real ROM, and weight each mask by the masked output: " *
            "importance(i)=(1/(N·p))·Σ y·m[i]. A black-box perturb-and-re-run method (no gradient), " *
            "scored vs the intervention oracle (corr + del/ins AUC + precision@k). BUDGET = N masks = " *
            "exactly N real re-runs per output; the CONVERGENCE SWEEP (N/8…N) shows faithfulness " *
            "stabilising as N grows (RISE variance ∝ 1/N). KEY CONTRAST vs IG/saliency: RISE ALSO " *
            "works on POSITION outputs (it re-runs the real ROM) where the gradient methods VANISH. " *
            "Oracle-as-method positive control corr=1/p@k=1.",
        "budget_note" => Dict{String,Any}(
            "n_masks" => n_masks, "keep_prob" => keep_prob,
            "cost" => "exactly n_masks real emulator re-runs per output (2 outputs/game: content + position)",
            "convergence" => "faithfulness recomputed at N/8, N/4, N/2, N from the SAME mask pool — no extra re-runs"),
        "metrics_note" => Dict{String,Any}(
            "faithfulness" => "corr + deletion/insertion AUC on the TRUE VCS + precision@k vs the oracle top-k",
            "convergence" => "corr/p@k/del-ins vs the mask count N (how RISE scales with budget)"),
        "results" => summary)
    open(summary_path, "w") do io; JSON.print(io, rec, 2); end
    println("\n[rise] wrote summary $summary_path")

    println("\n[rise] ===== per-game headline (content path, N=$n_masks masks) =====")
    println("  game             ram   corr    spear   p@$topk    del    ins    conv-corr@N")
    for r in summary
        r["is_content_path"] || continue
        conv = join(["$(r["convergence_n_masks"][i])=$(round(r["convergence_pearson"][i],digits=2))"
                     for i in 1:length(r["convergence_pearson"])], " ")
        println("  $(rpad(r["game"],16)) $(rpad(r["content_ram_index"],5)) " *
                "$(rpad(round(r["pearson"],digits=3),7)) " *
                "$(rpad(round(r["spearman"],digits=3),7)) " *
                "$(rpad(round(r["precision_at_k"],digits=3),6)) " *
                "$(rpad(r["deletion_auc"]===nothing ? "NA" : round(r["deletion_auc"],digits=3),6)) " *
                "$(rpad(r["insertion_auc"]===nothing ? "NA" : round(r["insertion_auc"],digits=3),6)) " *
                "[$conv]")
    end
    println("[rise] ===== position contrast (ball_pixel — RISE works where IG/saliency VANISH) =====")
    for r in summary
        r["is_content_path"] && continue
        println("  $(rpad(r["game"],16)) corr=$(round(r["pearson"],digits=3)) " *
                "p@$topk=$(round(r["precision_at_k"],digits=3)) " *
                "del=$(r["deletion_auc"]===nothing ? "NA" : round(r["deletion_auc"],digits=3)) " *
                "ins=$(r["insertion_auc"]===nothing ? "NA" : round(r["insertion_auc"],digits=3))")
    end
    return 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    RISE.main()
end
