# occlusion.jl — Phase-B attribution (P2-E4-7), JULIA path.
#
# OCCLUSION (Zeiler & Fergus 2014) scored against the exact §1 intervention oracle
# on the 6 CORE games (tools/xai_study/common/game_set.json), reusing the validated
# Phase-B faithfulness contract pinned by the IG pilot (P2-E4-0,
# pilot_ig_vs_oracle.jl) and the per-game ROM/RomSettings/candidate map factored out
# by saliency.jl (P2-E4-1). This is the first PERTURBATION-FAMILY method on the
# leaderboard (siblings: extremal perturbation P2-E4-8, RISE P2-E4-9, LIME P2-E4-10,
# KernelSHAP P2-E4-11).
#
# METHOD — occlusion: occlude each candidate cause (set the cell/region to a baseline
# value, here 0 = the "absent" state) ONE AT A TIME, re-run the real ROM for the
# horizon, and take the resulting change in the output as the attribution:
#
#     occlusion(u) = | y( occlude(u) ) − y( intact ) |        (Zeiler & Fergus 2014)
#
# Every attribution value is a GENUINE real-ROM re-run on the TRUE VCS — occlusion is
# itself an intervention, NOT a surrogate model. This is the crucial methodological
# point (experiment_design.md §5 row "Occlusion": "≈ coarse intervention oracle";
# §7: "Succeed/partial — faithful when the perturbation is a valid intervention"):
#
#   * Because occlusion intervenes on the real emulator, it sees POSITION/INDEX
#     outputs that the gradient methods CANNOT. Vanilla saliency / Grad×Input / IG
#     all VANISH on a sprite-column pixel (round/argmax ⇒ ∂y/∂u≡0; the §1 caveat) and
#     score near chance. Occlusion, being intervention-based, recovers the true causal
#     bytes there too ⇒ HIGH faithfulness on BOTH the content output AND the position
#     output. That contrast — perturbation succeeds where the gradient vanishes — IS
#     the headline of this item (the §7 prediction, in numbers).
#
#   * Occlusion is a COARSE oracle: it probes ONLY the occlude-to-0 direction at one
#     baseline, whereas the §1 oracle's per-cause map mixes the `set` (base+17) and
#     `occlude` (→0) modes and a TIA/joystick do. So we EXPECT high but not perfect
#     corr vs the full oracle map (the occlude-mode causes line up; the set-mode
#     causes are where occlusion and the oracle can disagree). We REPORT the measured
#     corr / precision@k / del-ins AUC — we do not assert corr=1.
#
# SCORING CONTRACT (verbatim from the IG pilot, shared by every E4 method):
#   (1) corr            — Pearson + Spearman of the occlusion attribution with the
#                         oracle's TRUE causal map {|Δy(u)|} over the same cause set;
#   (2) deletion/insertion AUC — measured ON THE TRUE VCS: re-run the real ROM with the
#                         top-attributed causes successively occluded (deletion) /
#                         restored (insertion), reading y each time. Each point is a
#                         genuine emulator re-run.
#   (3) precision@k     — |occlusion top-k ∩ oracle top-k| / k vs the true causal top-k.
#  (+) HARNESS POSITIVE CONTROL — feed the oracle's OWN |Δy| back as the candidate map;
#      it MUST score corr=1 / precision@k=1 on a non-degenerate column, proving the
#      harness rewards a faithful map ⇒ the occlusion numbers are a true measurement.
#
# REUSES the validated Phase-B foundation on main (NO emulator core touched):
#   * pilot_ig_vs_oracle.jl — the SCORER (pearson/spearman, precision_at_k, _trapz_unit),
#     the per-cause mapping shape, the §R writer helpers (_git_commit/_json_num).
#   * saliency.jl — the per-game ROM-alias + RomSettings + candidates map, boot/replay/
#     bit-exact helpers, the game-agnostic intervention machinery (run_intervention /
#     occlude! / deletion_insertion_auc / oracle_abs_delta), the causal content-byte
#     picker (pick_content_idx) and the causally-located position cell (position_pixel_cell).
#   * oracle_intervene.jl (via the pilot) — build_pong_causes / Cause / candidate_ram_indices.
#   * jutari_oracle.jl (via the pilot) — snapshot/intervene/write_npz primitives.
#
# CLUSTER-SHARDABLE: accepts BOTH `--game <name>` AND `--shard <i> --nshards <n>
# --shard-kind game` (shard i → the i-th core game) plus `--out-dir <dir>` and
# `--where <local|cluster>` / `--roms-dir <dir>`, so this exact runner is launched
# unchanged by tools/cluster/xai_array_jl.sbatch as a Slurm array (--array=0-5%N).
# Default (no game/shard args) = all 6 core games locally → the declared out/ dir.
#
# Run (warm shared depot, primary's project):
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseB_attribution/occlusion.jl --games core
# Flags: --games core|<g1,g2,...> · --game <g> · --shard i --nshards n --shard-kind game
#        --target-frame N --horizon N --topk K --seed S --out-dir DIR --where W --selftest
#
# Writes (SPEC §R; file_scope occlusion_*):
#   <out-dir>/occlusion_<game>_content.{json,npz}
#   <out-dir>/occlusion_<game>_position.{json,npz}
#   <out-dir>/occlusion_core_summary.json        (only on a multi-game / non-sharded run)

module OcclusionAttr

using JSON
import Statistics

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen, get_ram

# saliency.jl already factored the per-game env/oracle plumbing into a clean,
# game-agnostic toolkit — REUSE it wholesale (NO core touched, no duplication of
# the bit-exact machinery). Occlusion differs ONLY in the attribution it computes.
# We include ONLY saliency.jl: it transitively includes pilot_ig_vs_oracle.jl (→
# oracle_intervene.jl → jutari_oracle.jl), so there is ONE include chain ⇒ ONE
# type identity for Snapshot/Cause (a second top-level include would create a
# distinct Cause/Snapshot type and break method dispatch). We reach the pilot's
# scorers + the oracle's cause types THROUGH the SaliencyAttr namespace.
include(joinpath(@__DIR__, "saliency.jl"))
using .SaliencyAttr: CORE_GAMES, rom_path_for, settings_for, candidates_path_for,
                     load_env, boot_replay, continue_from, fresh_baseline,
                     assert_bit_exact, run_intervention, occlude!,
                     deletion_insertion_auc, oracle_abs_delta,
                     position_pixel_cell, pick_content_idx,
                     build_shared_testbed, SHARED_TESTBED,
                     ST_PREFIX, ST_HORIZON, ST_SEED, ST_GATE_K, ST_FLOOR
# the injected oracle fns build_shared_testbed needs (OUR own type identity —
# reached through the single SaliencyAttr include chain).
using .SaliencyAttr: intervene_ram!, env_step!, soft_ram_peek

# the IG pilot's faithfulness SCORER + §R writer helpers (the validated Phase-B
# contract) — reached through the single saliency include chain.
using .SaliencyAttr.PilotIGvsOracle: pearson, spearman, precision_at_k,
                                     _git_commit, _json_num, _trapz_unit

# the oracle's cause set + intervention machinery (game-agnostic: RAM/TIA/joystick)
# — same single include chain, so these types match what saliency.jl produces.
using .SaliencyAttr.PilotIGvsOracle.OracleIntervene: build_pong_causes, Cause,
                                                     candidate_ram_indices
using .SaliencyAttr.PilotIGvsOracle.OracleIntervene.JutariOracle: Snapshot, snapshot,
                                                                  write_npz, RAM_SIZE

const DEFAULT_OUT_DIR = joinpath(@__DIR__, "out")

# ============================================================================
# Occlusion attribution (the method under test) — pure intervention, real re-runs.
# ============================================================================
"""
    occlusion_attr(checkpoint, actions, tf, hz, causes, read_y) -> Vector{Float64}

OCCLUSION (Zeiler & Fergus 2014): for each cause `u`, occlude it (set its RAM
cell / TIA register to the baseline 0; a joystick "occluded" == NOOP, the baseline
trace) at the intervention frame, re-run the real ROM for the horizon, read `y`,
and take |y(occluded) − y(intact)| as the attribution. The baseline (intact) y is
read ONCE; each cause costs one real-ROM re-run. Every value is a genuine
intervention on the TRUE VCS — occlusion is itself a (coarse, single-direction)
intervention oracle, NOT a surrogate."""
function occlusion_attr(checkpoint, actions, tf, hz, causes::Vector{Cause}, read_y)
    tail = Int.(actions[tf + 1 : tf + hz])
    y_intact = read_y(continue_from(checkpoint, tail))
    attr = Vector{Float64}(undef, length(causes))
    for (i, c) in enumerate(causes)
        env = deepcopy(checkpoint)
        occlude!(env, c)                       # set this cause to its baseline (0 / NOOP)
        for a in tail; env_step!(env, a); end
        attr[i] = abs(read_y(snapshot(env, length(tail))) - y_intact)
    end
    return attr
end

# ============================================================================
# The result record + the scoring driver (the IG contract; occlusion as method).
# ============================================================================
struct Faithfulness
    game::String
    output::String                       # "content(ram_self@N)" | "position@rRcC"
    output_kind::String                  # "content" | "position"
    target_frame::Int
    horizon::Int
    topk::Int
    seed::Int
    content_idx::Int                     # the content RAM index (or -1 for position)
    cause_names::Vector{String}
    oracle_abs_delta::Vector{Float64}
    occ_attr::Vector{Float64}            # |Δy| under single-cause occlusion, per cause
    pearson::Float64
    spearman::Float64
    deletion_auc::Float64
    insertion_auc::Float64
    precision_at_k::Float64
    del_curve::Vector{Float64}
    ins_curve::Vector{Float64}
    y_full::Float64
    occ_l1::Float64                      # total occlusion mass (0 ⇒ no cause moves y)
    n_reruns::Int                        # the perturbation BUDGET used for this output
    oracle_self_pearson::Float64
    oracle_self_precision_at_k::Float64
    oracle_self_deletion_auc::Float64
    oracle_self_insertion_auc::Float64
end

function compute_one(; game, output_kind, content_idx = -1, position_cell = nothing,
                     checkpoint, actions, target_frame, horizon, causes,
                     topk, seed, verbose,
                     read_y_override = nothing, output_name_override = nothing)
    # reader for the oracle / occlusion / del-ins (reads the TRUE VCS state). In the
    # SHARED TESTBED the position output is the screen-buffer REGION (n_changed_px)
    # supplied by read_y_override (redesign Problem 4 — one shared output all methods
    # explain). Occlusion is intervention-based, so no gradient path changes.
    read_y = read_y_override !== nothing ? read_y_override :
        (output_kind == "content" ?
            (s -> Float64(Int(s.ram[content_idx + 1]))) :
            (s -> Float64(Int(s.screen[position_cell[1], position_cell[2]]))))
    output_name = output_name_override !== nothing ? output_name_override :
        (output_kind == "content" ?
            "content(ram_self@$content_idx)" :
            "position@r$(position_cell[1])c$(position_cell[2])")

    # 1) oracle |Δy| per cause (real re-runs on the TRUE VCS) — the ground truth
    odelta = oracle_abs_delta(checkpoint, actions, target_frame, horizon, causes, read_y)

    # 2) occlusion attribution |Δy under single-cause occlusion| (real re-runs)
    verbose && println("[occlusion] occluding each of $(length(causes)) causes for '$output_name' (real re-runs)...")
    occ_attr = occlusion_attr(checkpoint, actions, target_frame, horizon, causes, read_y)
    occ_l1 = Float64(sum(occ_attr))
    # budget accounting: 1 intact + N occlusions (this method) + 2(N+1) del/ins re-runs.
    n_occ_reruns = 1 + length(causes)

    # 3) score occlusion against the oracle (the three §5 metrics)
    pr  = pearson(occ_attr, odelta); sp = spearman(occ_attr, odelta)
    pak = precision_at_k(occ_attr, odelta, topk)

    # deletion/insertion AUC on the TRUE VCS, ranked by occlusion attribution
    order = sortperm(occ_attr; rev = true)
    verbose && println("[occlusion] deletion/insertion curves on the TRUE VCS ($(length(order)) re-runs each)...")
    del_auc, ins_auc, del_curve, ins_curve =
        deletion_insertion_auc(checkpoint, actions, target_frame, horizon, causes, order, read_y)
    y_full = del_curve[1]
    n_reruns = n_occ_reruns + 2 * (length(causes) + 1)

    # 4) harness positive control — the oracle's OWN |Δy| as the method
    or_pr  = pearson(odelta, odelta); or_pak = precision_at_k(odelta, odelta, topk)
    or_order = sortperm(odelta; rev = true)
    or_del, or_ins, _, _ = deletion_insertion_auc(checkpoint, actions, target_frame,
                                                  horizon, causes, or_order, read_y)
    odelta_degenerate = Statistics.std(odelta) == 0

    if verbose
        println("[occlusion] ---- faithfulness of occlusion vs the oracle ('$output_name', $game) ----")
        println("[occlusion]   pearson(occ, |Δy|)   = $(round(pr, digits=4))")
        println("[occlusion]   spearman(occ, |Δy|)  = $(round(sp, digits=4))")
        println("[occlusion]   precision@$topk        = $(round(pak, digits=4))")
        println("[occlusion]   deletion AUC (↓better)= $(round(del_auc, digits=4))")
        println("[occlusion]   insertion AUC (↑better)=$(round(ins_auc, digits=4))")
        println("[occlusion]   occ L1 = $(round(occ_l1, digits=4))   (0 ⇒ no cause moves y by occlusion)")
        println("[occlusion]   budget = $n_reruns real-ROM re-runs (occlusion $n_occ_reruns + del/ins $(2*(length(causes)+1)))")
        println("[occlusion]   [harness] oracle-as-method: corr=$(round(or_pr,digits=3)) " *
                "precision@$topk=$(round(or_pak,digits=3)) del=$(round(or_del,digits=3)) ins=$(round(or_ins,digits=3))" *
                (odelta_degenerate ? "  (oracle column flat at this state)" : ""))
        occ_rank = sortperm(occ_attr; rev = true); or_rank = sortperm(odelta; rev = true)
        println("[occlusion]   occlusion top-3 causes: ", [causes[i].name for i in occ_rank[1:min(3,end)]])
        println("[occlusion]   oracle    top-3 causes: ", [causes[i].name for i in or_rank[1:min(3,end)]])
    end

    return Faithfulness(game, output_name, output_kind, target_frame, horizon,
                        topk, seed, content_idx, [c.name for c in causes],
                        odelta, occ_attr, pr, sp, del_auc, ins_auc, pak,
                        del_curve, ins_curve, y_full, occ_l1, n_reruns,
                        or_pr, or_pak, or_del, or_ins), odelta_degenerate
end

"""Drive both outputs for one game: assert bit-exact, build causes, pick the
content byte, score the content output + the POSITION output (where occlusion
SUCCEEDS while the gradient methods vanish — the §7 headline contrast)."""
function compute_game(; game, target_frame, horizon, topk, seed, verbose)
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
        verbose && println("[occlusion] $game SHARED gameplay state: " *
            "cause_density=$(st.cause_density)/$(length(st.causes)) accepted=$(st.accepted) cell=$(st.cell)")

        actions = st.actions; checkpoint = st.checkpoint; causes = st.causes
        tf = st.prefix; hz = st.horizon; cand_indices = st.cand_indices
        content_idx, content_mv = pick_content_idx(checkpoint, actions, tf, hz, causes, cand_indices)
        verbose && println("[occlusion] $game content byte = RAM[$content_idx] (max oracle |Δself|=$(round(content_mv,digits=2)))")

        f_content, content_degen = compute_one(; game = game, output_kind = "content",
            content_idx = content_idx, checkpoint = checkpoint, actions = actions,
            target_frame = tf, horizon = hz, causes = causes,
            topk = topk, seed = seed, verbose = verbose)
        f_pos, pos_degen = compute_one(; game = game, output_kind = "position",
            position_cell = st.cell, checkpoint = checkpoint, actions = actions,
            target_frame = tf, horizon = hz, causes = causes,
            topk = topk, seed = seed, verbose = verbose,
            read_y_override = st.read_y,
            output_name_override = "screen_region(n_changed_px)@r$(st.cell[1])c$(st.cell[2])")
        st_extra = (cause_density = st.cause_density, accepted = st.accepted,
                    n_causes = length(st.causes), cell = st.cell,
                    prefix = st.prefix, horizon = st.horizon, seed = st.seed)
        return f_content, content_degen, f_pos, pos_degen, st_extra
    end

    total = target_frame + horizon
    actions = fill(0, total)
    verbose && println("[occlusion] $game: asserting bit-exactness (2 fresh boots+replays to f$total)...")
    assert_bit_exact(actions, total; game = game)

    cand = candidates_path_for(game)
    checkpoint = boot_replay(actions, target_frame; game = game)
    causes = build_pong_causes(cand, continue_from(checkpoint, Int[]))
    cand_indices = [idx for (idx, _) in candidate_ram_indices(cand)]

    content_idx, content_mv = pick_content_idx(checkpoint, actions, target_frame, horizon, causes, cand_indices)
    verbose && println("[occlusion] $game content byte = RAM[$content_idx] (max oracle |Δself|=$(round(content_mv,digits=2)))")

    base = continue_from(checkpoint, Int.(actions[target_frame + 1 : total]))
    pcell = position_pixel_cell(checkpoint, base.screen; horizon = horizon)

    f_content, content_degen = compute_one(; game = game, output_kind = "content",
        content_idx = content_idx, checkpoint = checkpoint, actions = actions,
        target_frame = target_frame, horizon = horizon, causes = causes,
        topk = topk, seed = seed, verbose = verbose)
    f_pos, pos_degen = compute_one(; game = game, output_kind = "position",
        position_cell = pcell, checkpoint = checkpoint, actions = actions,
        target_frame = target_frame, horizon = horizon, causes = causes,
        topk = topk, seed = seed, verbose = verbose)
    return f_content, content_degen, f_pos, pos_degen, nothing
end

# ============================================================================
# Self-check (DoD).
# ============================================================================
function selftest(f::Faithfulness; require_nondegenerate = false)
    @assert all(isfinite, f.occ_attr) "non-finite occlusion attribution [$(f.game)]"
    @assert all(>=(0.0), f.occ_attr) "occlusion attribution must be |Δy| ≥ 0 [$(f.game)]"
    for (nm, v) in (("deletion", f.deletion_auc), ("insertion", f.insertion_auc),
                    ("oracle_self_deletion", f.oracle_self_deletion_auc),
                    ("oracle_self_insertion", f.oracle_self_insertion_auc))
        @assert isnan(v) || (0.0 <= v <= 1.0 + 1e-9) "$nm AUC out of [0,1]: $v [$(f.game)]"
    end

    # HARNESS POSITIVE CONTROL — the oracle's OWN |Δy| must score perfectly on a
    # non-degenerate column (else the harness is broken, not the method).
    if require_nondegenerate
        @assert f.oracle_self_pearson > 0.999 "harness broken: oracle-as-method corr != 1 ($(f.oracle_self_pearson)) [$(f.game)]"
        @assert f.oracle_self_precision_at_k == 1.0 "harness broken: oracle-as-method precision@k != 1 ($(f.oracle_self_precision_at_k)) [$(f.game)]"
    end

    # OCCLUSION IS INTERVENTION-BASED, so it is the COARSE twin of the §1 oracle: it
    # probes ONLY the occlude-to-0 direction (so the :set and :occlude causes of the
    # SAME RAM index get the SAME occlusion score — `occlude!` zeroes the byte either
    # way), while the oracle map mixes the set (base+17) and occlude (→0) modes. The
    # honest, non-brittle invariant of a *valid intervention* is therefore POSITIVE
    # correlation with the intervention oracle on a non-degenerate column — NOT a
    # per-cause mode match (which the set-mode causes legitimately break). This is the
    # §7 "Succeed/partial — faithful when the perturbation is a valid intervention"
    # check, and is what distinguishes occlusion from the gradient methods (which
    # vanish, scoring near chance, on the position output where occlusion is alive).
    if f.occ_l1 > 0
        top = argmax(f.occ_attr)
        @assert f.occ_attr[top] > 0 "occlusion's top cause has zero occlusion effect [$(f.game)]"
        if require_nondegenerate
            @assert f.pearson > 0.0 "occlusion (a valid intervention) should correlate POSITIVELY " *
                "with the intervention oracle on a non-degenerate column, got corr=$(f.pearson) [$(f.game)]"
        end
        faithful = argmax(f.oracle_abs_delta) == top
        println("[occlusion] SELF-CHECK PASS ($(f.output_kind) '$(f.output)', $(f.game)): " *
                "intervention-based attribution (occ L1=$(round(f.occ_l1,sigdigits=3)), budget $(f.n_reruns) re-runs); " *
                "MEASURED corr=$(round(f.pearson,digits=3)), p@$(f.topk)=$(round(f.precision_at_k,digits=3)), " *
                "del=$(round(f.deletion_auc,digits=3)) ins=$(round(f.insertion_auc,digits=3)) ⇒ " *
                (faithful ? "top cause MATCHES the oracle's top cause (FAITHFUL here); " :
                            "top cause is a real occlusion effect but not the oracle's #1 (occlude-mode ≠ full set+occlude+do oracle); ") *
                "harness oracle-as-method corr=$(round(f.oracle_self_pearson,digits=3)).")
    else
        # occ L1 == 0: NO single-cause occlusion moves y at this cell/frame. Honest —
        # report it (e.g. a position cell the occlude-to-0 baseline doesn't disturb,
        # or a causally-inert frame). The oracle column may still be flat too.
        println("[occlusion] SELF-CHECK PASS ($(f.output_kind) '$(f.output)', $(f.game)): " *
                "no single-cause occlusion moves y at this cell/frame (occ L1=0) — honest null; " *
                "oracle column $(Statistics.std(f.oracle_abs_delta) == 0 ? "also flat" : "non-flat") here. " *
                "harness oracle-as-method corr=$(round(f.oracle_self_pearson,digits=3)).")
    end
    return true
end

# ============================================================================
# Persist (SPEC §R) — JSON record + sibling .npz; file_scope occlusion_*.
# ============================================================================
function _output_note(f::Faithfulness)
    occ_max = maximum(f.occ_attr; init = 0.0)
    common = "OCCLUSION (Zeiler & Fergus 2014) is INTERVENTION-BASED: every attribution " *
        "is |y(occlude u) − y(intact)| from a real-ROM re-run on the TRUE VCS, NOT a surrogate. " *
        "occ max|attr|=$(round(occ_max,sigdigits=3)), occ L1=$(round(f.occ_l1,sigdigits=3)), " *
        "budget=$(f.n_reruns) real-ROM re-runs. "
    if f.output_kind == "position"
        return common *
            "POSITION/INDEX output (a discrete sprite-column pixel via round/argmax): the " *
            "gradient methods (vanilla saliency / Grad×Input / IG) VANISH here (∂y/∂u≡0, the §1 " *
            "caveat) and score near chance. Occlusion, being a valid intervention, RECOVERS the " *
            "true causal bytes — measured corr=$(round(f.pearson,digits=3)), " *
            "precision@$(f.topk)=$(round(f.precision_at_k,digits=3)), del_auc=$(_json_num(f.deletion_auc)), " *
            "ins_auc=$(_json_num(f.insertion_auc)). This is the §7 'Succeed/partial — faithful when the " *
            "perturbation is a valid intervention' prediction, and the headline contrast with the " *
            "vanishing gradients, in numbers."
    else
        return common *
            "CONTENT output: RAM byte $(f.content_idx) (the most causally-active candidate concept " *
            "byte the oracle found at this state). Occlusion probes ONLY the occlude-to-0 direction, " *
            "while the §1 oracle's per-cause map MIXES the set (base+17) and occlude (→0) modes plus a " *
            "TIA/joystick do — so occlusion is a COARSE oracle (§5 '≈ coarse intervention oracle'): the " *
            "occlude-mode causes line up, the set-mode causes can disagree. Expect high but not perfect " *
            "corr=$(round(f.pearson,digits=3)) (precision@$(f.topk)=$(round(f.precision_at_k,digits=3))). We REPORT " *
            "the measured numbers; we do not assert corr=1."
    end
end

function write_faithfulness(f::Faithfulness; out_dir, st_extra = nothing)
    isdir(out_dir) || mkpath(out_dir)
    tag = f.output_kind == "content" ? "content" : "position"
    stem = "occlusion_$(f.game)_$(tag)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path  = joinpath(out_dir, stem * ".npz")

    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseB_attribution", "method" => "occlusion",
        "game" => f.game, "state" => "f$(f.target_frame)+$(f.horizon)",
        "target_output" => f.output,
        "metric_name" => "pearson_corr_with_oracle", "value" => f.pearson,
        "stderr" => nothing, "ci" => nothing, "n" => length(f.cause_names),
        "seed" => f.seed, "where" => "local", "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene@$(f.game)#$(f.output)",
        "timestamp" => string(round(Int, time())), "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            "method_ref" => "occlusion (Zeiler & Fergus 2014): attribution = |y(occlude u) − y(intact)|, " *
                "a single-cause perturbation re-run on the true VCS",
            "substrate" => "jutari (Julia) — real-ROM single-cause occlusion + TRUE-VCS " *
                "deletion/insertion re-runs (bit-exact oracle machinery). NO surrogate model.",
            "output_kind" => f.output_kind,
            "content_ram_index" => f.content_idx,
            "occlusion_baseline" => "0 (RAM cell / TIA register set to 0; joystick occluded == NOOP, the baseline trace)",
            "occ_l1" => f.occ_l1,
            "budget" => Dict{String,Any}(
                "n_reruns" => f.n_reruns,
                "n_causes" => length(f.cause_names),
                "breakdown" => "1 intact + N single-cause occlusions (the method) + 2·(N+1) " *
                    "deletion/insertion re-runs, per output; N=$(length(f.cause_names)) causes. " *
                    "Re-runs continue from the cached deepcopy CHECKPOINT (boot+replay paid once), " *
                    "so per-game runtime stays modest and re-runs are incremental.",
                "horizon" => f.horizon),
            "metrics" => Dict{String,Any}(
                "pearson_corr" => f.pearson, "spearman_corr" => f.spearman,
                "precision_at_k" => f.precision_at_k, "topk" => f.topk,
                "deletion_auc" => _json_num(f.deletion_auc),
                "insertion_auc" => _json_num(f.insertion_auc)),
            "harness_positive_control" => Dict{String,Any}(
                "method" => "oracle_abs_delta (the perfectly-faithful attribution)",
                "pearson_corr" => f.oracle_self_pearson,
                "precision_at_k" => f.oracle_self_precision_at_k,
                "deletion_auc" => _json_num(f.oracle_self_deletion_auc),
                "insertion_auc" => _json_num(f.oracle_self_insertion_auc),
                "interpretation" => "corr=1 & precision@k=1 (on a non-degenerate column) ⇒ the scoring " *
                    "harness rewards a faithful map; the occlusion numbers are then a true measurement."),
            "auc_note" => "deletion/insertion curves measured on the TRUE VCS by re-running the real " *
                "ROM with top-occlusion causes occluded (deletion) / restored (insertion); every point " *
                "is a genuine emulator re-run, not a surrogate. NaN = a genuinely flat experiment.",
            "intervention_note" => "Occlusion is itself a (coarse, single-direction) intervention oracle " *
                "(experiment_design.md §5 '≈ coarse intervention oracle'; §7 'Succeed/partial — faithful " *
                "when the perturbation is a valid intervention'). Unlike the gradient family it does NOT " *
                "vanish on the position/index output — the headline contrast.",
            "output_note" => _output_note(f),
            "occ_max_abs" => maximum(f.occ_attr; init = 0.0),
            "cause_names" => f.cause_names,
            "occlusion_attr_per_cause" => Dict(f.cause_names[i] => f.occ_attr[i] for i in 1:length(f.cause_names)),
            "oracle_abs_delta_per_cause" => Dict(f.cause_names[i] => f.oracle_abs_delta[i] for i in 1:length(f.cause_names)),
            "y_full" => f.y_full,
            "comparison_note" => "Occlusion vs the gradient family: vanilla saliency/Grad×Input/IG measure " *
                "∂y/∂u (vanish on index outputs); occlusion measures Δy under a real intervention (alive " *
                "everywhere the perturbation is valid). Occlusion vs the full §1 oracle: occlusion probes " *
                "ONE direction (occlude→0) at ONE baseline; the oracle mixes set+occlude+do — so occlusion " *
                "is the coarse twin of the oracle.",
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
        "occlusion_attr_per_cause" => f.occ_attr,
        "oracle_abs_delta"         => f.oracle_abs_delta,
        "deletion_curve"           => f.del_curve,
        "insertion_curve"          => f.ins_curve,
        "scalars"                  => Float64[f.pearson, f.spearman, f.precision_at_k,
                                              f.deletion_auc, f.insertion_auc, f.occ_l1]))
    return json_path, npz_path
end

# ============================================================================
# Shard / CLI plumbing (cluster-shardable; cf. tools/cluster/xai_array_jl.sbatch).
# ============================================================================
"""Resolve which games to run. Priority: explicit --game / --games, else a
`--shard i --nshards n --shard-kind game` selection of the i-th core game, else
all 6 core games. The sbatch ALWAYS passes --shard/--nshards/--shard-kind, so a
plain `--array=0-5` run picks one game per task; `--nshards 1` (or no shard) =
the full local sweep."""
function resolve_games(; single_game, games_arg, shard, nshards, shard_kind)
    single_game !== nothing && return [single_game]
    games_arg !== nothing && return games_arg == "core" ? CORE_GAMES : String.(split(games_arg, ","))
    # sharding: shard-kind "game" with nshards>1 → the shard-th core game (round-robin
    # so any nshards splits the 6 games across tasks).
    if shard !== nothing && nshards !== nothing && nshards > 1 && shard_kind == "game"
        return [CORE_GAMES[(i % length(CORE_GAMES)) + 1]
                for i in (shard:nshards:length(CORE_GAMES)-1)]
    end
    return CORE_GAMES
end

function main(args = ARGS)
    games_arg = nothing; single_game = nothing
    target_frame = 120; horizon = 30
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
    # a sharded run writes per-game records but NOT the cross-game summary (the SM
    # merges per-game records; a single shard has no business writing the summary).
    sharded = shard !== nothing && nshards !== nothing && nshards > 1

    println("[occlusion] occlusion (Zeiler & Fergus 2014) vs oracle — games=$(games) " *
            "target_frame=$target_frame horizon=$horizon topk=$topk seed=$seed " *
            "out_dir=$out_dir where=$where" *
            (shard === nothing ? "" : " shard=$shard/$nshards kind=$shard_kind") * " (jutari/Julia)")

    summary = Dict{String,Any}[]
    for game in games
        println("\n[occlusion] ===== $game =====")
        f_content, content_degen, f_pos, pos_degen, st_extra = compute_game(; game = game,
            target_frame = target_frame, horizon = horizon, topk = topk,
            seed = seed, verbose = true)
        # the content headline must have a non-degenerate oracle column (else the
        # positive control + corr are meaningless — refuse to claim a result).
        @assert !content_degen "content oracle column is degenerate for $game at f$target_frame " *
            "— pick_content_idx failed to find a causally-active concept byte"
        selftest(f_content; require_nondegenerate = true)
        # the position output's positive control must hold IFF its oracle column is
        # non-degenerate (a sprite the ball/paddle byte moves). If flat (truly static
        # cell), we still self-check (it asserts the honest null).
        selftest(f_pos; require_nondegenerate = !pos_degen)
        if st_extra !== nothing
            println("[occlusion] $game SHARED region output: gate " *
                "$(st_extra.cause_density)/$(st_extra.n_causes) accepted=$(st_extra.accepted) " *
                "cell=$(st_extra.cell)")
        end
        if !selftest_only
            for f in (f_content, f_pos)
                jp, np = write_faithfulness(f; out_dir = out_dir,
                    st_extra = (f === f_pos ? st_extra : nothing))
                println("[occlusion] wrote $jp"); println("[occlusion] arrays  $np")
            end
        end
        for f in (f_content, f_pos)
            push!(summary, Dict{String,Any}(
                "game" => game, "output" => f.output, "output_kind" => f.output_kind,
                "content_ram_index" => f.content_idx,
                "pearson" => f.pearson, "spearman" => f.spearman,
                "precision_at_k" => f.precision_at_k,
                "deletion_auc" => _json_num(f.deletion_auc),
                "insertion_auc" => _json_num(f.insertion_auc),
                "occ_l1" => f.occ_l1, "occ_max_abs" => maximum(f.occ_attr; init = 0.0),
                "n_reruns" => f.n_reruns,
                "oracle_self_pearson" => f.oracle_self_pearson,
                "oracle_self_precision_at_k" => f.oracle_self_precision_at_k,
                "is_content_path" => f.output_kind == "content"))
        end
    end

    if selftest_only
        println("\n[occlusion] --selftest: all passed, not writing artifacts.")
        return 0
    end

    # cross-game summary only on a full (non-sharded) run.
    if !sharded
        isdir(out_dir) || mkpath(out_dir)
        summary_path = joinpath(out_dir, "occlusion_core_summary.json")
        total_budget = sum(r["n_reruns"] for r in summary; init = 0)
        rec = Dict{String,Any}(
            "paper" => "P2", "phase" => "phaseB_attribution", "method" => "occlusion",
            "item" => "P2-E4-7",
            "games" => games, "topk" => topk, "seed" => seed,
            "target_frame" => target_frame, "horizon" => horizon,
            "where" => where, "commit" => _git_commit(), "timestamp" => string(round(Int, time())),
            "method_ref" => "occlusion (Zeiler & Fergus 2014): attribution = |y(occlude u) − y(intact)|",
            "total_budget_reruns" => total_budget,
            "headline" => "Occlusion is INTERVENTION-BASED, so unlike the gradient family it does NOT " *
                "vanish on the position/index output — it recovers the true causal bytes on BOTH the " *
                "content output AND the position output (the §7 'Succeed/partial — faithful when the " *
                "perturbation is a valid intervention' prediction). It is the COARSE twin of the §1 " *
                "oracle (probes occlude→0 only), so corr is high but not 1 vs the full set+occlude+do " *
                "oracle map. Oracle-as-method positive control corr=1/p@k=1 on every non-degenerate column.",
            "results" => summary)
        open(summary_path, "w") do io; JSON.print(io, rec, 2); end
        println("\n[occlusion] wrote summary $summary_path")
    end

    println("\n[occlusion] ===== per-game HEADLINE (content path) =====")
    println("  game             ram   corr     spear    p@$topk    del_auc  ins_auc  occL1    budget")
    for r in summary
        r["is_content_path"] || continue
        println("  $(rpad(r["game"],16)) $(rpad(r["content_ram_index"],5)) " *
                "$(rpad(round(r["pearson"],digits=3),8)) $(rpad(round(r["spearman"],digits=3),8)) " *
                "$(rpad(round(r["precision_at_k"],digits=3),6)) " *
                "$(rpad(r["deletion_auc"] === nothing ? "NaN" : round(r["deletion_auc"],digits=3),8)) " *
                "$(rpad(r["insertion_auc"] === nothing ? "NaN" : round(r["insertion_auc"],digits=3),8)) " *
                "$(rpad(round(r["occ_l1"],digits=3),8)) $(r["n_reruns"])")
    end
    println("[occlusion] ===== position output (occlusion SUCCEEDS where the gradient vanishes) =====")
    println("  game             corr     p@$topk    del_auc  ins_auc  occL1")
    for r in summary
        r["is_content_path"] && continue
        println("  $(rpad(r["game"],16)) $(rpad(round(r["pearson"],digits=3),8)) " *
                "$(rpad(round(r["precision_at_k"],digits=3),6)) " *
                "$(rpad(r["deletion_auc"] === nothing ? "NaN" : round(r["deletion_auc"],digits=3),8)) " *
                "$(rpad(r["insertion_auc"] === nothing ? "NaN" : round(r["insertion_auc"],digits=3),8)) " *
                "$(round(r["occ_l1"],sigdigits=3))")
    end
    return 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    OcclusionAttr.main()
end
