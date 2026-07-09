# acdc.jl — Phase-C mechanistic interpretability (P2-E5-5), JULIA path
# (the jutari real-ROM substrate; jaxtari eager is ~205× slower — SCRUM §7).
#
# ACDC — Automatic Circuit DisCovery (Conmy et al. 2023, "Towards Automated
# Circuit Discovery for Mechanistic Interpretability") on the VCS, generalising the
# Phase-C foundation (activation_patching.jl P2-E5-1, das.jl P2-E5-2,
# A1_connectomics.jl P2-E3-1) from per-site/per-variable effects to the FULL
# data-flow GRAPH discovered automatically by iterative edge pruning, scored
# against the TRUE inter-cell data-flow (experiment_design.md §6 row "ACDC"; §7
# prediction: **Partial→Succeed** — the goal state IS our ground truth).
#
# ---------------------------------------------------------------------------
# What ACDC IS here (Conmy et al. 2023, adapted to the VCS known circuit):
#
#   * The "computational graph" is the program's inter-cell data-flow over the
#     candidate RAM cells (the E2-1 import / T3 candidates): nodes = candidate
#     cells, a directed EDGE i->j means cell i's value (at frame t) feeds the
#     computation of cell j (over the horizon). The CANDIDATE (superset) graph is
#     the dense off-diagonal graph — every ordered pair — which ACDC prunes.
#
#   * ACDC iterates over edges and ABLATES each one: it removes edge i->j by
#     RESAMPLING the source — overwriting cell i at frame t with a CORRUPTED value
#     (its value from a corrupted donor run where that genuinely diverges, else the
#     synthetic resample base+17 — a real, non-degenerate corrupted activation that
#     also matches the oracle's `set` cause so the discovery is directly comparable
#     to the exhaustive true graph) — then RE-RUNS the real ROM the horizon and
#     measures the change in the downstream readout y_j. This is the ACDC ablation:
#     resample-ablation of the parent activation (Conmy §3; the standard ACDC
#     corruption is a resampled/patched activation, not zero). NB: at the early,
#     pre-interactive boot frames the action-context donor does not diverge (input
#     is ignored on the title sequence), so the base+17 resample is what makes every
#     edge probe-able — we record the honest donor-divergence count separately.
#
#   * KEEP RULE (ACDC's threshold τ): an edge is part of the discovered circuit
#     iff removing it (ablating the parent) changes the child's output by MORE than
#     τ. Edges whose ablation barely moves the child (effect ≤ τ) are pruned —
#     they carry no information for that child. Sweeping τ from large→small
#     recovers more edges (ACDC's tau sweep / pareto frontier, Conmy §4): a large τ
#     keeps only the strongest edges (high precision, low recall); τ→0 keeps every
#     edge with any nonzero effect.
#
#   The DISCOVERED CIRCUIT at threshold τ = { i->j : effect(i->j) > τ }, where
#   effect(i->j) = | y_j(ablate i->j) − y_j(clean) |  — a genuine real-ROM re-run.
#
# ---------------------------------------------------------------------------
# What is the GROUND TRUTH (the TRUE data-flow circuit, T1/T2)?
#
# Reusing A1_connectomics.jl's exhaustive exact-intervention definition (the TRUE
# inter-cell data-flow graph from exhaustive pairwise interventions — the
# "connectome"): TRUE edge i->j iff intervening on cell i with the EXHAUSTIVE
# oracle do-set (occlude->0, set->base+17, set->base+37) and re-running the
# deterministic emulator for the FULL horizon makes cell j differ from the
# un-intervened baseline, for ANY perturbation value. Every Δ is a bit-exact
# real-ROM re-run (Paper-1 64/64), so a TRUE edge is a real data-flow edge — no
# world model assumed. The self-edge i->i is excluded (tautology).
#
# ---------------------------------------------------------------------------
# SCORING (experiment_design.md §6: "edge P/R + scrubbing-preserved performance
# vs true data-flow"):
#
#   For each τ in the sweep, the discovered edge set is scored against the TRUE
#   edge set over the OFF-DIAGONAL edges:
#       precision(τ) = |E_disc(τ) ∩ E_true| / |E_disc(τ)|
#       recall(τ)    = |E_disc(τ) ∩ E_true| / |E_true|
#       F1(τ)        = 2 P R / (P + R)
#   The τ sweep yields an ROC-style frontier (recall vs 1−precision and TPR vs
#   FPR); we report the AUC of the ROC, the best-F1 operating point, and the full
#   per-τ table.
#
#   SCRUBBING-PRESERVED PERFORMANCE (causal scrubbing of the discovered circuit,
#   experiment_design.md §6): take the discovered circuit at the best-F1 τ, then
#   ABLATE every edge NOT in the circuit (resample all non-circuit parents
#   simultaneously) and re-run; the fraction of the clean output preserved is the
#   scrubbing-preserved performance. A faithful circuit preserves the output even
#   when everything outside it is scrubbed (high preservation); scrubbing the TRUE
#   circuit's complement is the positive reference.
#
#   POSITIVE CONTROL (oracle-as-ACDC, mirroring A1/pilot_si): run the edge-keep
#   rule with the EXHAUSTIVE oracle do-set + full horizon and τ=0 (keep any edge
#   with a nonzero exact effect) — this IS the true-graph procedure, so it must
#   recover the TRUE graph exactly: P=R=F1=1. This proves the harness REWARDS a
#   perfectly-faithful discovery, so a sub-1 resample-ACDC F1 is a real
#   measurement of the cheap-ablation faithfulness gap, not a broken scorer.
#
# F/S/M triad (correctness triad, experiment_design.md §0), scored vs the oracle:
#   F (faithful)   — best-F1 of the discovered circuit's edges vs the true edges.
#   S (sufficient) — scrubbing-preserved performance: the discovered circuit alone
#                    (non-circuit edges scrubbed) reproduces the clean behaviour —
#                    the held-out "the circuit suffices" test, on the bit-exact
#                    re-run.
#   M (minimal)    — 1 − over-claim rate at the best-F1 τ: of the edges ACDC
#                    discovered, the fraction the oracle says do NOT exist (spurious
#                    wiring) — the method's tendency to keep edges that don't carry
#                    real data-flow.
#
# BUILDS ON the verified jutari foundation (NO emulator core touched):
#   * tools/xai_study/common/jutari_oracle.jl — boot/replay/snapshot/deepcopy
#     CHECKPOINT for incremental re-runs + intervene_ram! + the dependency-free §R
#     NPZ writer + RAM_SIZE.
#   * tools/xai_study/phaseA_kording/A1_connectomics.jl — the TRUE inter-cell
#     data-flow graph (exhaustive pairwise interventions) = the ground-truth circuit
#     this ACDC is scored against (we reuse its exact exhaustive definition).
#   * tools/xai_study/ground_truth/oracle_intervene.jl — the exact-intervention
#     causal-map reference whose Δ definition this reuses cell-to-cell.
#   * each game's tools/xai_study/t3/out/candidates_<game>.json — the candidate
#     cells (the graph nodes).
#
# Run (warm shared depot, primary's project) — SYNCHRONOUS, all 6 core games:
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseC_mechanistic/acdc.jl --games core
# Flags:
#   --games core|<g1,g2,...>   --game <g>
#   --target-frame N --horizon N
#   --selftest                 (positive control on the pilot game; writes nothing)
#   # cluster-shardable (tools/cluster/xai_array_jl.sbatch contract):
#   --shard i --nshards n --shard-kind game   (default kind=game; shard the game list)
#   --out-dir DIR              (default: this file's out/)
#   --roms-dir DIR             (extra ROM search root; defaults still apply)
#   --where local|cluster      (recorded in the §R record's "where")
#
# Writes (SPEC §R; file_scope acdc_* under out/):
#   tools/xai_study/phaseC_mechanistic/out/acdc_<game>.{json,npz}
#   tools/xai_study/phaseC_mechanistic/out/acdc_core_summary.json   (>1 game)

module ACDC

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen, get_ram
using JuTari.JoystickGames: MsPacmanRomSettings, QbertRomSettings

# the verified foundation (NO core touched): boot/replay/snapshot/checkpoint/
# intervene + the dependency-free §R NPZ writer + RAM_SIZE.
include(joinpath(@__DIR__, "..", "common", "jutari_oracle.jl"))
using .JutariOracle: write_npz, RAM_SIZE

# The §1 exact-intervention oracle (Cause / build_pong_causes / candidate_ram_indices /
# run_intervention / assert_bit_exact) — used ONLY to build the P2 SHARED gameplay
# state + its cause-density gate (below). ACDC keeps its OWN GameSpec/Snap/Candidate
# machinery; the oracle here supplies the shared action stream + the gate, not the
# circuit. Referenced as OracleIntervene.X (its internal JutariOracle submodule is a
# SEPARATE instance from the one included above — no type is mixed across them; that
# is why we do NOT alias-import it).
include(joinpath(@__DIR__, "..", "ground_truth", "oracle_intervene.jl"))
using JuTari.Diff: soft_ram_peek

# The P2 SHARED TESTBED (xai_paper/xai_2_interpretability/experiment_redesign.md):
# seeded random-action GAMEPLAY state + oracle cause-density gate. Included as a
# fragment (see its header for why not a module). Phase C is not a gradient method,
# so the sampler-on path does not apply — we consume the shared action STREAM +
# cause-density GATE only, and boot ACDC's OWN checkpoint from that stream so the
# discovery machinery is unchanged. Opt in with XAI_SHARED_TESTBED=1 (default on).
include(joinpath(@__DIR__, "..", "common", "shared_testbed_impl.jl"))
# the shared game-set + ROM-root resolver (XAI_LABELED / xai_resolve_games / xai_rom_roots).
include(joinpath(@__DIR__, "..", "common", "game_sets.jl"))

# shared triad helpers (the canonical M = |U*|/|U_hat|; here the EDGE-set analogue);
# guarded so a repeated include across sibling runners is a no-op.
isdefined(@__MODULE__, :TriadSM) ||
    include(joinpath(@__DIR__, "..", "common", "triad_sm.jl"))

import JSON

const OUT_DIR = joinpath(@__DIR__, "out")
const CORE_GAMES = ["pong", "breakout", "space_invaders", "seaquest", "ms_pacman", "qbert"]
const _PRIMARY_REPO = get(ENV, "XAI_PRIMARY_REPO",
                          "/Users/maier/Documents/code/UnderstandingVCS")

# shared-testbed switch + params (redesign protocol: prefix=90 gameplay, horizon=15).
const SHARED_TESTBED = get(ENV, "XAI_SHARED_TESTBED", "1") == "1"
const ST_PREFIX  = parse(Int, get(ENV, "XAI_ST_PREFIX", "90"))
const ST_HORIZON = parse(Int, get(ENV, "XAI_ST_HORIZON", "15"))
const ST_SEED    = parse(Int, get(ENV, "XAI_ST_SEED", "0"))
const ST_GATE_K  = parse(Int, get(ENV, "XAI_ST_GATE_K", "4"))
const ST_FLOOR   = parse(Float64, get(ENV, "XAI_ST_FLOOR", "0.5"))

# joystick action codes (oracle_intervene.jl: RIGHT=3; LEFT=4). The corrupted
# (donor) run for resample-ablation uses a different action context so source cells
# genuinely diverge from the clean (NOOP) run.
const ACT_NOOP    = 0
const ACT_CORRUPT = 3   # RIGHT — the ACDC "corrupted" trace for resample-ablation

# ===========================================================================
# Per-game ROM-basename + RomSettings map (self-contained; mirrors
# A1_connectomics.jl / activation_patching.jl so boot/render parity holds; NO
# emulator core touched). seaquest has no registered RomSettings → Generic.
# ===========================================================================
struct GameSpec
    name::String
    rom_basename::String
    settings::Function
    settings_name::String
end

const GAME_SPECS = Dict{String,GameSpec}(
    "pong"           => GameSpec("pong", "pong.bin",
                                 () -> JuTari.PaddleGames.PongRomSettings(), "PongRomSettings"),
    "breakout"       => GameSpec("breakout", "breakout.bin",
                                 () -> JuTari.PaddleGames.BreakoutRomSettings(), "BreakoutRomSettings"),
    "space_invaders" => GameSpec("space_invaders", "space_invaders.bin",
                                 () -> JuTari.SpaceInvadersRomSettings(), "SpaceInvadersRomSettings"),
    "seaquest"       => GameSpec("seaquest", "seaquest.bin",
                                 () -> JuTari.GenericRomSettings(), "GenericRomSettings"),
    "ms_pacman"      => GameSpec("ms_pacman", "mspacman.bin",
                                 () -> MsPacmanRomSettings(), "MsPacmanRomSettings"),
    "qbert"          => GameSpec("qbert", "qbert.bin",
                                 () -> QbertRomSettings(), "QbertRomSettings"),
)

# extra ROM search roots set at runtime by --roms-dir (cluster contract).
const _EXTRA_ROM_DIRS = String[]

# curated core aliases (xitari/roms uses e.g. mspacman.bin); every other labeled
# game's ROM basename IS its ALE-canonical name (== tools/rom_sweep/roms/<name>.bin).
const ROM_BASENAME = Dict(
    "pong" => "pong", "breakout" => "breakout",
    "space_invaders" => "space_invaders", "seaquest" => "seaquest",
    "ms_pacman" => "mspacman", "qbert" => "qbert")

"""The GameSpec for a game: the curated core spec if present, else a generic one for
any T3-labeled game (GenericRomSettings — boots fine; the screen scoreboard's Generic
fallback), so all 54 labeled games run. The ROM alias maps ms_pacman → mspacman."""
function spec_for(game::AbstractString)
    g = lowercase(string(game))
    haskey(GAME_SPECS, g) && return GAME_SPECS[g]
    stem = get(ROM_BASENAME, g, g)
    return GameSpec(g, stem * ".bin",
                    () -> JuTari.GenericRomSettings(), "GenericRomSettings")
end

"""Absolute ROM path for a game: search this worktree's xitari/roms, the primary
repo's xitari/roms, then any --roms-dir roots (the cluster ROM collection layout).
ROMs are gitignored — used in place, never committed."""
function rom_path_for(spec::GameSpec)
    # --roms-dir roots first (cluster ROM collection layout): try the spec basename
    # directly there. Then delegate to the shared root set (this worktree + primary
    # xitari/roms + the 54-ROM store tools/rom_sweep/roms + the collection), trying
    # BOTH the spec basename stem AND the raw ALE name, so all 54 labeled games resolve.
    for d in _EXTRA_ROM_DIRS
        p = joinpath(d, spec.rom_basename)
        isfile(p) && return p
    end
    stem = replace(spec.rom_basename, r"\.bin$" => "")
    names = unique([stem, lowercase(spec.name)])
    return xai_find_rom(names, xai_rom_roots(; primary_repo = _PRIMARY_REPO,
                                             extra = _EXTRA_ROM_DIRS))
end

"""A freshly-reset env with the xitari-parity boot (60 NOOP + 4 RESET)."""
function fresh_env(spec::GameSpec)
    rom = read(rom_path_for(spec))
    env = StellaEnvironment(rom, spec.settings())
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    return env
end

# --- snapshot / replay / intervene (same primitives as A1_connectomics.jl) ----
struct Snap
    ram::Vector{UInt8}
    screen::Matrix{UInt8}
end
snap(env) = Snap(copy(collect(get_ram(env))), Matrix{UInt8}(get_screen(env)))

"""Boot + step `actions[1:target_frame]`; return the env AT the target frame (the
reusable CHECKPOINT — boot+to-target paid once; deepcopy before mutating)."""
function boot_replay(spec::GameSpec, actions, target_frame)
    env = fresh_env(spec)
    for i in 1:target_frame
        env_step!(env, Int(actions[i]))
    end
    return env
end

"""do(ram[idx] := value) into RIOT RAM (0-based idx; the RAM vector is 1-indexed)."""
function intervene_ram!(env, idx::Integer, value::Integer)
    @assert 0 <= idx < RAM_SIZE "ram_index out of range: $idx"
    env.console.bus.ram[Int(idx) + 1] = UInt8(Int(value) & 0xFF)
    return env
end

"""deepcopy the checkpoint, step `tail`, snapshot."""
function continue_from(checkpoint, tail)
    env = deepcopy(checkpoint)
    for a in tail
        env_step!(env, Int(a))
    end
    return snap(env)
end

"""deepcopy the checkpoint, poke ram[idx]:=value, step `tail`, snapshot."""
function intervene_continue(checkpoint, idx::Integer, value::Integer, tail)
    env = deepcopy(checkpoint)
    intervene_ram!(env, idx, value)
    for a in tail
        env_step!(env, Int(a))
    end
    return snap(env)
end

"""deepcopy the checkpoint, poke a SET of cells {idx=>value}, step `tail`, snapshot.
Used for simultaneous resample-scrubbing of multiple parents."""
function intervene_set_continue(checkpoint, sets::Dict{Int,Int}, tail)
    env = deepcopy(checkpoint)
    for (idx, value) in sets
        intervene_ram!(env, idx, value)
    end
    for a in tail
        env_step!(env, Int(a))
    end
    return snap(env)
end

# ===========================================================================
# Candidate cells (E2-1 / T3 import) — the graph nodes.
# ===========================================================================
struct Candidate
    ram_index::Int
    concept::String
end

"""Read `candidates_<game>.json`, de-duplicated by ram_index, in file order."""
function load_candidates(game::AbstractString)
    rel = joinpath("tools", "xai_study", "t3", "out", "candidates_$(game).json")
    path = nothing
    here = normpath(joinpath(@__DIR__, "..", "..", ".."))
    for base in (here, _PRIMARY_REPO)
        p = joinpath(base, rel)
        if isfile(p); path = p; break; end
    end
    out = Candidate[]
    if path !== nothing
        data = JSON.parsefile(path)
        seen = Set{Int}()
        for c in get(data, "candidates", [])
            idx = Int(c["ram_index"])
            idx in seen && continue
            push!(seen, idx)
            concept = get(c, "concept", nothing)
            push!(out, Candidate(idx, concept === nothing ? "(unnamed)" : string(concept)))
        end
    end
    # non-core games have no T3 candidate file — fall back to the SAME generic
    # RAM-byte cause set the shared testbed's candidate_ram_indices(nothing) uses,
    # so all 54 labeled games get a bounded, uniform candidate cell set.
    if isempty(out)
        for (idx, concept) in ((13, "enemy_score"), (14, "player_score"),
                               (49, "ball_x"), (54, "ball_y"),
                               (51, "player_y"), (50, "enemy_y"))
            push!(out, Candidate(idx, concept))
        end
    end
    return out, path
end

# ===========================================================================
# The TRUE data-flow circuit (ground truth) — reused from A1_connectomics.jl's
# exhaustive exact-intervention definition (the "connectome").
# ===========================================================================
"""TRUE edge i -> j iff the EXHAUSTIVE oracle do-set on cell i (occlude->0,
set->base+17, set->base+37) changes cell j at the FULL horizon, for ANY value.
Computed by the bit-exact intervention oracle. (= A1_connectomics.true_dataflow_graph)"""
function true_dataflow_graph(checkpoint, tail, cands::Vector{Candidate}, at_target::Snap)
    n = length(cands)
    A = falses(n, n)
    base = continue_from(checkpoint, tail).ram
    for (i, c) in enumerate(cands)
        bval = Int(at_target.ram[c.ram_index + 1])
        changed = falses(n)
        for v in (0, (bval + 17) & 0xFF, (bval + 37) & 0xFF)
            v == bval && continue
            r = intervene_continue(checkpoint, c.ram_index, v, tail).ram
            for (j, cj) in enumerate(cands)
                if r[cj.ram_index + 1] != base[cj.ram_index + 1]
                    changed[j] = true
                end
            end
        end
        A[i, :] = changed
    end
    return A
end

# ===========================================================================
# ACDC edge-effect matrix (the method under test).
#
# For every candidate source cell i, ABLATE it by RESAMPLING — overwrite cell i at
# the target frame with its value from the CORRUPTED (donor) run — then re-run the
# horizon ONCE and read every child j. effect(i->j) = |y_j(ablate i) − y_j(clean)|.
# Because every source is ablated to a SINGLE corrupted value (the ACDC resample,
# not the exhaustive oracle do-set), one re-run per source covers all of that
# source's outgoing edges. The DISCOVERED circuit at threshold τ is then
# { i->j : effect(i->j) > τ } — pure thresholding, no extra re-runs in the τ sweep.
#
# This is the cheap-ablation faithfulness gap vs the exhaustive true graph: a
# single resample value can miss edges that fire only for a different value/sign
# (recall deficit) and can over-claim transient ones (precision deficit).
# ===========================================================================
"""The corrupted (resample) value for a source cell: its value from the corrupted
ACTION-context donor run IF that genuinely diverged from the clean run, else the
ACDC synthetic resample `base+17` (a real, non-degenerate corrupted activation —
the standard ACDC corruption is a resampled activation, not zero; base+17 also
matches the oracle's `set` cause, so the discovery is directly comparable to the
exhaustive true graph). Returns (value, diverged_in_donor)."""
function resample_value(clean_val::Int, donor_val::Int)
    if donor_val != clean_val
        return donor_val, true                 # genuine cross-run resample
    end
    return (clean_val + 17) & 0xFF, false      # synthetic corrupted activation
end

"""The ACDC edge-EFFECT matrix E[i,j] = |child_j(resample-ablate source_i) −
child_j(clean)|, one real-ROM re-run per source. The resample value comes from the
corrupted donor where it diverges, else base+17 (a non-degenerate corrupted
activation). Also returns per-source flags for whether the ablation actually
changed the source value (it always does here) and whether the donor diverged (an
honest cross-run audit)."""
function acdc_edge_effects(checkpoint, tail, cands::Vector{Candidate},
                           at_target::Snap, corrupt_at_target::Snap)
    n = length(cands)
    E = zeros(Float64, n, n)
    base = continue_from(checkpoint, tail).ram
    donor_diverged = falses(n)
    resampled = Vector{Int}(undef, n)          # the corrupted value used per source
    for (i, c) in enumerate(cands)
        clean_val = Int(at_target.ram[c.ram_index + 1])
        donor_val = Int(corrupt_at_target.ram[c.ram_index + 1])
        rv, dvg = resample_value(clean_val, donor_val)
        donor_diverged[i] = dvg
        resampled[i] = rv
        # resample-ablation: overwrite source i with its corrupted value, re-run.
        r = intervene_continue(checkpoint, c.ram_index, rv, tail).ram
        for (j, cj) in enumerate(cands)
            E[i, j] = abs(Float64(Int(r[cj.ram_index + 1])) -
                          Float64(Int(base[cj.ram_index + 1])))
        end
    end
    return E, donor_diverged, resampled
end

"""The discovered circuit at threshold τ: edge i->j iff effect(i->j) > τ. The
diagonal is forced off (self-edges are tautological and excluded from scoring)."""
function discovered_at(E::Matrix{Float64}, tau::Float64)
    n = size(E, 1)
    A = E .> tau
    for k in 1:n; A[k, k] = false; end
    return BitMatrix(A)
end

# ===========================================================================
# Scoring: edge P/R/F1 over off-diagonal edges, a τ sweep, ROC + AUC.
# ===========================================================================
struct EdgeScore
    n_true::Int
    n_disc::Int
    tp::Int
    fp::Int
    fn::Int
    precision::Float64
    recall::Float64
    f1::Float64
end

function score_edges(disc::BitMatrix, tru::BitMatrix)
    n = size(tru, 1)
    @assert size(disc) == size(tru) "adjacency shape mismatch"
    offdiag = trues(n, n); for k in 1:n; offdiag[k, k] = false; end
    D = disc .& offdiag
    T = tru .& offdiag
    tp = count(D .& T)
    fp = count(D .& .!T)
    fn = count(.!D .& T)
    n_true = count(T); n_disc = count(D)
    precision = n_disc == 0 ? (n_true == 0 ? 1.0 : 0.0) : tp / n_disc
    recall    = n_true == 0 ? 1.0 : tp / n_true
    f1 = (precision + recall) == 0 ? 0.0 : 2 * precision * recall / (precision + recall)
    return EdgeScore(n_true, n_disc, tp, fp, fn, precision, recall, f1)
end

"""Build a τ sweep from the distinct effect magnitudes (the ACDC pareto frontier).
Thresholds are placed just-below each distinct positive effect so that τ→0 keeps
all nonzero-effect edges and τ=∞ keeps none. Returns sorted ascending τ values."""
function tau_sweep(E::Matrix{Float64})
    n = size(E, 1)
    vals = Float64[]
    for i in 1:n, j in 1:n
        i == j && continue
        e = E[i, j]
        e > 0 && push!(vals, e)
    end
    uniq = sort!(unique(vals))
    # One threshold per distinct effect, placed as the MIDPOINT between consecutive
    # distinct effects (and between 0 and the smallest effect). `effect > τ` then
    # gives a proper threshold sweep: the smallest τ (midpoint of 0 and the min
    # effect) keeps EVERY nonzero-effect edge (and never a zero-effect one); each
    # larger τ drops the weakest surviving edge; a τ above the max keeps nothing.
    taus = Float64[]
    for k in 1:length(uniq)
        prev = k == 1 ? 0.0 : uniq[k - 1]
        push!(taus, (prev + uniq[k]) / 2)      # midpoint threshold
    end
    isempty(uniq) && push!(taus, 0.0)          # degenerate: no positive effects
    push!(taus, isempty(uniq) ? 1.0 : uniq[end] + 1.0)   # keeps nothing
    return sort!(unique(taus))
end

"""ROC AUC of the discovery as τ sweeps: TPR = recall, FPR = fp / (#negatives).
Trapezoidal AUC over the (FPR, TPR) points (ascending FPR)."""
function roc_auc(taus::Vector{Float64}, E::Matrix{Float64}, tru::BitMatrix)
    n = size(tru, 1)
    offdiag_count = n * (n - 1)
    n_pos = count(tru .& .!_eye(n))
    n_neg = offdiag_count - n_pos
    pts = Tuple{Float64,Float64}[]
    for tau in taus
        s = score_edges(discovered_at(E, tau), tru)
        tpr = n_pos == 0 ? 1.0 : s.tp / n_pos
        fpr = n_neg == 0 ? 0.0 : s.fp / n_neg
        push!(pts, (fpr, tpr))
    end
    # ensure the curve spans [0,0] and [1,1]
    push!(pts, (0.0, 0.0)); push!(pts, (1.0, 1.0))
    sort!(pts; by = p -> (p[1], p[2]))
    # trapezoid integral
    auc = 0.0
    for k in 2:length(pts)
        x0, y0 = pts[k - 1]; x1, y1 = pts[k]
        auc += (x1 - x0) * (y0 + y1) / 2
    end
    return clamp(auc, 0.0, 1.0)
end
_eye(n) = BitMatrix([i == j for i in 1:n, j in 1:n])

# ===========================================================================
# Scrubbing-preserved performance (causal scrubbing of the discovered circuit).
#
# Take the discovered circuit at the best-F1 τ. SCRUB every edge NOT in the circuit
# by resample-ablating every parent that has NO surviving outgoing circuit edge
# (its information is supposed to be irrelevant), then re-run. Preservation =
# fraction of candidate-cell readouts that match the clean baseline. A faithful
# circuit preserves the output when its complement is scrubbed.
# ===========================================================================
"""Scrubbing-preserved performance: resample-ablate all sources that are NOT part
of the discovered circuit (no outgoing circuit edge) — using the SAME corrupted
values as the edge-effect probe (`resampled`) — re-run, and report the fraction of
candidate-cell values unchanged vs the clean baseline."""
function scrubbing_preserved(checkpoint, tail, cands::Vector{Candidate},
                             at_target::Snap, resampled::Vector{Int},
                             circuit::BitMatrix)
    n = length(cands)
    base = continue_from(checkpoint, tail).ram
    # a source is "in the circuit" if it has any outgoing surviving edge.
    in_circuit_src = [any(circuit[i, :]) for i in 1:n]
    scrub = Dict{Int,Int}()
    for (i, c) in enumerate(cands)
        if !in_circuit_src[i]
            cv = resampled[i]
            cv != Int(at_target.ram[c.ram_index + 1]) && (scrub[c.ram_index] = cv)
        end
    end
    if isempty(scrub)
        return 1.0, 0                       # nothing to scrub → fully preserved
    end
    r = intervene_set_continue(checkpoint, scrub, tail).ram
    preserved = 0
    for cj in cands
        r[cj.ram_index + 1] == base[cj.ram_index + 1] && (preserved += 1)
    end
    return preserved / n, length(scrub)
end

# ===========================================================================
# F / S / M triad.
# ===========================================================================
struct Triad
    F::Float64
    S::Float64
    M::Float64
    F_note::String
    S_note::String
    M_note::String
end

# ===========================================================================
# Per-game ACDC result.
# ===========================================================================
struct GameResult
    game::String
    settings_name::String
    rom_basename::String
    candidates_path::Union{Nothing,String}
    target_frame::Int
    horizon::Int
    bit_exact::Bool
    cands::Vector{Candidate}
    E::Matrix{Float64}                 # ACDC edge-effect matrix
    true_graph::BitMatrix
    taus::Vector{Float64}
    sweep::Vector{EdgeScore}           # per-τ scores (same order as taus)
    best_tau::Float64
    best_score::EdgeScore
    roc_auc::Float64
    oracle_control::EdgeScore          # oracle-as-ACDC (exhaustive, τ=0) vs true
    scrub_preserved::Float64
    n_scrubbed::Int
    n_source_diverged::Int
    triad::Triad
    # SHARED-TESTBED provenance (redesign); "noop"/-1/false in the legacy path.
    state_kind::String             # "seeded_random_action_gameplay" | "noop"
    st_seed::Int
    st_prefix::Int
    cause_density::Int             # #causes above the floor at the shared output
    cause_density_accepted::Bool   # passed the cause-density gate?
    n_causes::Int
    shared_cell::Tuple{Int,Int}    # the shared screen-buffer output cell
end

# ACDC's candidates-path resolver, in the shape the shared testbed injects (a
# String->path-or-nothing map). Mirrors load_candidates' file search.
function _st_candidates_path_for(game::AbstractString)
    rel = joinpath("tools", "xai_study", "t3", "out", "candidates_$(game).json")
    here = normpath(joinpath(@__DIR__, "..", "..", ".."))
    for base in (here, _PRIMARY_REPO)
        p = joinpath(base, rel)
        isfile(p) && return p
    end
    return nothing
end

"""Build the P2 SHARED gameplay state + cause-density gate for `game` using the §1
oracle machinery (oracle_intervene.jl). Returns the substrate NamedTuple (we use
its `.actions` stream + `.cause_density`/`.accepted`/`.cell` gate). ACDC then boots
its OWN checkpoint from `st.actions` so the discovery algorithm is unchanged."""
function build_acdc_shared_state(game::AbstractString; verbose = false)
    O = OracleIntervene; J = OracleIntervene.JutariOracle
    return build_shared_testbed(game;
        settings_for = J.settings_for, rom_path_for = J.rom_path_for,
        candidates_path_for = _st_candidates_path_for,
        build_causes = O.build_pong_causes, candidate_ram_indices = O.candidate_ram_indices,
        continue_from = J.continue_from, snapshot = J.snapshot, env_step = J.env_step!,
        intervene_ram = J.intervene_ram!, boot_replay = J.boot_replay,
        run_intervention = O.run_intervention, soft_ram_peek = soft_ram_peek,
        prefix = ST_PREFIX, horizon = ST_HORIZON, seed = ST_SEED,
        k = ST_GATE_K, floor = ST_FLOOR, verbose = verbose,
        assert_bit_exact = O.assert_bit_exact)
end

function run_game(spec::GameSpec; target_frame = 30, horizon = 30, verbose = true)
    # SHARED-TESTBED (redesign): replace the all-NOOP clean / RIGHT-action corrupt
    # tapes with a seeded random-action GAMEPLAY state at f*=ST_PREFIX, gated by the
    # oracle cause-density gate. We take the shared action STREAM + the gate from the
    # substrate and boot ACDC's OWN clean checkpoint from that stream; the ACDC
    # discovery machinery (GameSpec/Snap/Candidate/resample) is UNCHANGED — only the
    # analysis state moves onto genuine input-driven gameplay. The CORRUPT (resample
    # donor) stream shares the gameplay prefix and diverges only from f* onward (a
    # RIGHT action from the analysis frame), so the donor is on-distribution.
    st = nothing
    if SHARED_TESTBED
        st = build_acdc_shared_state(spec.name; verbose = verbose)
        target_frame = st.prefix; horizon = st.horizon
        clean_actions   = Int.(st.actions)
        corrupt_actions = vcat(Int.(st.actions[1:target_frame]),
                               fill(ACT_CORRUPT, horizon))
        verbose && println("[acdc:$(spec.name)] SHARED gameplay state: cause_density=$(st.cause_density)/" *
            "$(length(st.causes)) accepted=$(st.accepted) cell=$(st.cell)")
    else
        total = target_frame + horizon
        clean_actions   = fill(ACT_NOOP, total)
        corrupt_actions = fill(ACT_CORRUPT, total)
    end
    total = target_frame + horizon
    tail = Int.(clean_actions[target_frame + 1 : total])

    # 1) bit-exact baseline guarantee — two fresh boots+replays must be identical.
    verbose && println("[acdc:$(spec.name)] asserting bit-exactness (2 fresh boots+replays to f$total)...")
    a = continue_from(boot_replay(spec, clean_actions, target_frame), tail)
    b = continue_from(boot_replay(spec, clean_actions, target_frame), tail)
    bit_exact = (a.ram == b.ram) && (a.screen == b.screen)
    bit_exact || error("[acdc:$(spec.name)] bit-exact re-run FAILED to f$total — refusing to score")
    verbose && println("[acdc:$(spec.name)] bit-exact re-run: PASS")

    # 2) ONE clean checkpoint at the intervention frame (boot+to-target paid once).
    checkpoint = boot_replay(spec, clean_actions, target_frame)
    at_target = continue_from(checkpoint, Int[])            # clean state AT frame t

    # 2b) the CORRUPTED (donor) run's state at the target frame — the resample source.
    corrupt_at_target = continue_from(boot_replay(spec, corrupt_actions, target_frame), Int[])

    cands, cand_path = load_candidates(spec.name)
    isempty(cands) && error("[acdc:$(spec.name)] no candidate cells (candidates_$(spec.name).json missing/empty)")
    verbose && println("[acdc:$(spec.name)] candidates: $(cand_path) ($(length(cands)) cells)")

    # 3) TRUE data-flow circuit (exhaustive exact oracle — A1's ground truth).
    verbose && println("[acdc:$(spec.name)] true data-flow circuit (exhaustive oracle, $(length(cands)) cells)...")
    true_graph = true_dataflow_graph(checkpoint, tail, cands, at_target)

    # 4) ACDC edge-effect matrix (resample-ablation, one re-run per source).
    verbose && println("[acdc:$(spec.name)] ACDC edge effects (resample-ablation, 1 re-run/source)...")
    E, donor_diverged, resampled = acdc_edge_effects(checkpoint, tail, cands, at_target, corrupt_at_target)
    n_source_diverged = count(donor_diverged)

    # 5) τ sweep (ROC frontier): score the discovered circuit at every threshold.
    taus = tau_sweep(E)
    sweep = [score_edges(discovered_at(E, t), true_graph) for t in taus]
    # best operating point = max F1 (ties → larger τ = sparser/more minimal circuit).
    best_k = 1; best_f1 = -1.0
    for (k, s) in enumerate(sweep)
        if s.f1 > best_f1 + 1e-12 || (abs(s.f1 - best_f1) <= 1e-12 && taus[k] > taus[best_k])
            best_f1 = s.f1; best_k = k
        end
    end
    best_tau = taus[best_k]; best_score = sweep[best_k]
    auc = roc_auc(taus, E, true_graph)

    # 6) POSITIVE CONTROL — oracle-as-ACDC: keep any edge with a nonzero EXHAUSTIVE
    #    exact effect (τ=0 on the exhaustive do-set) = the true-graph procedure
    #    itself. Must score P=R=F1=1.
    oracle_control = score_edges(true_graph, true_graph)

    # 7) scrubbing-preserved performance of the best-F1 discovered circuit.
    best_circuit = discovered_at(E, best_tau)
    scrub_preserved, n_scrubbed =
        scrubbing_preserved(checkpoint, tail, cands, at_target, resampled, best_circuit)

    # 8) triad. M standardized to the paper's definition (03_methods.tex sec:triad):
    #    M = |U*|/|U_hat|. For an EDGE/GRAPH method U* = the TRUE edges (the wiring
    #    that carries real data-flow) and U_hat = the edges the method NAMES (the
    #    discovered circuit at best-F1 τ) — the same |true edges|/|discovered edges|
    #    ratio path_patching.jl uses, clamped to (0,1]; null when 0 edges discovered
    #    (nothing named ⇒ M undefined). An over-claiming circuit that names spurious
    #    edges pays for every one of them.
    n_true_e = best_score.n_true; n_disc_e = best_score.n_disc
    M = n_disc_e == 0 ? NaN : min(1.0, n_true_e / n_disc_e)
    Mnote = n_disc_e == 0 ?
        "M undefined: ACDC named NO edge at best-F1 τ (|U_hat|=0)" :
        "|U*|=$n_true_e true edges / |U_hat|=$n_disc_e discovered edges (best-F1 τ)"
    triad = Triad(best_score.f1, scrub_preserved, M,
        "best-F1 of the ACDC-discovered circuit edges vs the exhaustive-oracle true data-flow ($(length(cands)) candidate cells)",
        "scrubbing-preserved performance: the discovered circuit alone (non-circuit parents resample-scrubbed) reproduces the clean candidate-cell readouts (fraction unchanged, bit-exact re-run)",
        Mnote)

    if verbose
        println("[acdc:$(spec.name)] ---- scores ----")
        println("[acdc:$(spec.name)]   nodes=$(length(cands))  true_edges=$(best_score.n_true)  " *
                "best τ=$(round(best_tau,digits=3))  disc_edges=$(best_score.n_disc)")
        println("[acdc:$(spec.name)]   best  P=$(round(best_score.precision,digits=3))  " *
                "R=$(round(best_score.recall,digits=3))  F1=$(round(best_score.f1,digits=3))  " *
                "ROC-AUC=$(round(auc,digits=3))")
        println("[acdc:$(spec.name)]   positive control (oracle-as-ACDC, exhaustive τ=0): " *
                "P=$(round(oracle_control.precision,digits=3)) R=$(round(oracle_control.recall,digits=3)) " *
                "F1=$(round(oracle_control.f1,digits=3))")
        println("[acdc:$(spec.name)]   scrubbing-preserved=$(round(scrub_preserved,digits=3)) " *
                "(scrubbed $n_scrubbed non-circuit parents)  source_diverged=$n_source_diverged/$(length(cands))")
        println("[acdc:$(spec.name)]   TRIAD: F=$(round(triad.F,digits=3))  " *
                "S=$(round(triad.S,digits=3))  M=$(round(triad.M,digits=3))")
    end

    return GameResult(spec.name, spec.settings_name, spec.rom_basename, cand_path,
                      target_frame, horizon, bit_exact, cands, E, true_graph,
                      taus, sweep, best_tau, best_score, auc, oracle_control,
                      scrub_preserved, n_scrubbed, n_source_diverged, triad,
                      st === nothing ? "noop" : "seeded_random_action_gameplay",
                      st === nothing ? -1 : st.seed,
                      st === nothing ? -1 : st.prefix,
                      st === nothing ? -1 : st.cause_density,
                      st === nothing ? false : st.accepted,
                      st === nothing ? 0 : length(st.causes),
                      st === nothing ? (-1, -1) : st.cell)
end

# ===========================================================================
# Persist (SPEC §R) — per-game record + combined summary; file_scope acdc_*.
# ===========================================================================
_git_commit() = try
    strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
catch
    "unknown"
end
_jnum(x::Real) = isfinite(x) ? Float64(x) : nothing

# adjacency -> list of [i,j] edge pairs (0-based candidate ordinals, off-diagonal)
function _edge_list(A::BitMatrix)
    n = size(A, 1)
    edges = Vector{Vector{Int}}()
    for i in 1:n, j in 1:n
        i == j && continue
        A[i, j] && push!(edges, [i - 1, j - 1])
    end
    return edges
end

function _game_record(r::GameResult; where_str::AbstractString)
    cell_names = ["RAM[$(c.ram_index)]:$(c.concept)" for c in r.cands]
    sweep_tbl = [Dict{String,Any}(
        "tau" => _jnum(r.taus[k]),
        "precision" => _jnum(r.sweep[k].precision),
        "recall" => _jnum(r.sweep[k].recall),
        "f1" => _jnum(r.sweep[k].f1),
        "n_discovered" => r.sweep[k].n_disc,
        "tp" => r.sweep[k].tp, "fp" => r.sweep[k].fp, "fn" => r.sweep[k].fn,
    ) for k in 1:length(r.taus)]
    Dict{String,Any}(
        "paper" => "P2",
        "phase" => "phaseC_mechanistic",
        "method" => "ACDC(automatic circuit discovery, resample-ablation edge pruning)",
        "game" => r.game,
        "state" => r.state_kind == "noop" ? "f$(r.target_frame)+$(r.horizon)" :
                   "gameplay(seed=$(r.st_seed),prefix=$(r.st_prefix))+$(r.horizon)",
        "target_output" => "auto-discovered inter-cell data-flow circuit (candidate RAM cells)",
        # headline §R scalar (value/metric_name): the discovered-circuit best-F1.
        "metric_name" => "discovered_circuit_best_F1_vs_true_dataflow",
        "value" => _jnum(r.triad.F),
        "stderr" => nothing,
        "ci" => nothing,
        "n" => length(r.cands),
        "seed" => 0,
        "where" => where_str,
        "commit" => _git_commit(),
        "oracle_ref" => "A1_connectomics/oracle_intervene@$(r.game) — exhaustive exact inter-cell data-flow (do-set 0/base+17/base+37, full horizon)",
        "timestamp" => string(round(Int, time())),
        "budget" => Dict{String,Any}(
            "rom_reruns_true_graph" => "≤ 3·N exhaustive interventions (N=$(length(r.cands)) cells × 3 do-values)",
            "rom_reruns_acdc" => "N resample-ablations (1 per source) + 1 corrupted boot + 1 scrub re-run + 2 bit-exact baselines",
            "tau_sweep" => "$(length(r.taus)) thresholds (pure thresholding of the cached effect matrix — NO extra re-runs)",
            "horizon_frames" => r.horizon,
            "note" => "paper-reasonable: the τ sweep adds zero re-runs (it thresholds " *
                "the cached effect matrix). Heavy cost is the N resample re-runs + the " *
                "exhaustive 3N ground-truth re-runs; re-runnable incrementally from the " *
                "cached deepcopy checkpoint.",
        ),
        "extra" => Dict{String,Any}(
            "substrate" => "jutari (Julia, HARD) — real-ROM bit-exact path; every edge " *
                "effect is an EXACT resample-ablation re-run on the true ROM.",
            "settings" => r.settings_name,
            "rom_basename" => r.rom_basename,
            "candidates_file" => r.candidates_path,
            "bit_exact_rerun" => r.bit_exact,
            "candidate_cells" => cell_names,
            "candidate_ram_indices" => [c.ram_index for c in r.cands],
            "ablation" => "resample (parent overwritten with its value from the " *
                "corrupted RIGHT-action donor run); standard ACDC corruption (Conmy " *
                "et al. 2023 §3), not zero.",
            "n_source_diverged" => r.n_source_diverged,
            "testbed" => Dict{String,Any}(
                "state_kind" => r.state_kind,
                "seed" => r.st_seed, "prefix" => r.st_prefix, "horizon" => r.horizon,
                "shared_output" => "screen_region(n_changed_px)@r$(r.shared_cell[1])c$(r.shared_cell[2])",
                "cause_density_above_floor" => r.cause_density,
                "cause_density_floor" => ST_FLOOR, "cause_density_gate_k" => ST_GATE_K,
                "cause_density_accepted" => r.cause_density_accepted, "n_causes" => r.n_causes,
                "note" => "P2 redesign: ACDC runs on a seeded random-action GAMEPLAY " *
                    "state (not the boot/attract NOOP tape), gated by the §1 oracle " *
                    "cause-density gate. The clean stream is the shared gameplay tape; " *
                    "the resample donor shares the gameplay prefix and diverges only " *
                    "from f* (a RIGHT action) so it is on-distribution. The ACDC " *
                    "discovery algorithm is unchanged; only the analysis state moves."),
            "circuit" => Dict{String,Any}(
                "n_nodes" => length(r.cands),
                "n_true_edges" => r.best_score.n_true,
                "best_tau" => _jnum(r.best_tau),
                "n_discovered_edges_at_best" => r.best_score.n_disc,
                "tp" => r.best_score.tp, "fp" => r.best_score.fp, "fn" => r.best_score.fn,
                "precision" => _jnum(r.best_score.precision),
                "recall" => _jnum(r.best_score.recall),
                "f1" => _jnum(r.best_score.f1),
                "roc_auc" => _jnum(r.roc_auc),
                "true_edges" => _edge_list(r.true_graph),
                "discovered_edges_at_best_tau" => _edge_list(discovered_at(r.E, r.best_tau)),
                "note" => "edges are 0-based candidate ordinals [i,j] = cell i -> cell j. " *
                    "TRUE = exhaustive oracle (do 0/base+17/base+37, full horizon, any " *
                    "value changes j); DISCOVERED = ACDC keep-rule (resample-ablation " *
                    "effect > τ). The τ sweep is the ROC frontier.",
            ),
            "tau_sweep" => sweep_tbl,
            "scrubbing" => Dict{String,Any}(
                "preserved_fraction" => _jnum(r.scrub_preserved),
                "n_scrubbed_parents" => r.n_scrubbed,
                "note" => "causal scrubbing of the best-F1 discovered circuit: " *
                    "resample-ablate every parent NOT in the circuit, re-run, report " *
                    "the fraction of candidate-cell readouts unchanged vs clean.",
            ),
            "positive_control" => Dict{String,Any}(
                "name" => "oracle-as-ACDC (exhaustive do-set, full horizon, τ=0)",
                "precision" => _jnum(r.oracle_control.precision),
                "recall" => _jnum(r.oracle_control.recall),
                "f1" => _jnum(r.oracle_control.f1),
                "note" => "the exhaustive keep-rule IS the true-graph procedure, so it " *
                    "scores F1=1 — the harness rewards a perfectly-faithful discovery, " *
                    "so the resample-ACDC F1<1 is a real cheap-ablation faithfulness gap.",
            ),
            "triad" => Dict{String,Any}(
                "F" => _jnum(r.triad.F), "F_note" => r.triad.F_note,
                "S" => _jnum(r.triad.S), "S_note" => r.triad.S_note,
                "M" => _jnum(r.triad.M), "M_note" => r.triad.M_note,
                "interpretation" => "ACDC's goal state IS our ground truth " *
                    "(experiment_design.md §7: Partial→Succeed). The discovery uses a " *
                    "cheap single-value resample-ablation, so it can miss edges that " *
                    "fire only for a different value (recall deficit) and keep transient " *
                    "ones (precision deficit) vs the exhaustive true data-flow.",
            ),
            "scales_to_cluster_via" =>
                "tools/cluster/xai_array_jl.sbatch (--shard/--nshards/--shard-kind game " *
                "--out-dir --roms-dir --where) — the full ACDC battery over the core + " *
                "breadth sets; forward bit-exact to this HARD path.",
        ),
    )
end

function _write_game_npz(r::GameResult, npz_path)
    n = length(r.taus)
    sweep_mat = zeros(Float64, n, 4)        # [tau, precision, recall, f1]
    for k in 1:n
        sweep_mat[k, :] = [r.taus[k], r.sweep[k].precision, r.sweep[k].recall, r.sweep[k].f1]
    end
    write_npz(npz_path, Dict(
        "candidate_ram_indices" => Int64[c.ram_index for c in r.cands],
        "edge_effects"          => r.E,                       # (n_cells, n_cells)
        "true_graph"            => UInt8.(r.true_graph),
        "discovered_at_best_tau" => UInt8.(discovered_at(r.E, r.best_tau)),
        "tau_sweep_tau_P_R_F1"  => sweep_mat,                 # (n_tau, 4)
        "best_PRF_AUC"          => Float64[r.best_score.precision, r.best_score.recall,
                                           r.best_score.f1, r.roc_auc],
        "scrub_preserved"       => Float64[r.scrub_preserved],
        "triad_FSM"             => Float64[r.triad.F, r.triad.S, r.triad.M],
    ))
end

function write_game_result(r::GameResult; out_dir = OUT_DIR, where_str = "local")
    isdir(out_dir) || mkpath(out_dir)
    jp = joinpath(out_dir, "acdc_$(r.game).json")
    np = joinpath(out_dir, "acdc_$(r.game).npz")
    rec = _game_record(r; where_str = where_str)
    rec["arrays"] = basename(np)
    open(jp, "w") do io; JSON.print(io, rec, 2); end
    _write_game_npz(r, np)
    return jp, np
end

function write_summary(results::Vector{GameResult}; out_dir = OUT_DIR, where_str = "local")
    isdir(out_dir) || mkpath(out_dir)
    path = joinpath(out_dir, "acdc_core_summary.json")
    per_game = Dict{String,Any}[]
    all_ctrl_ok = true
    for r in results
        all_ctrl_ok &= (r.oracle_control.f1 > 0.999)
        push!(per_game, Dict{String,Any}(
            "game" => r.game,
            "n_nodes" => length(r.cands),
            "n_true_edges" => r.best_score.n_true,
            "best_tau" => _jnum(r.best_tau),
            "best_precision" => _jnum(r.best_score.precision),
            "best_recall" => _jnum(r.best_score.recall),
            "best_f1" => _jnum(r.best_score.f1),
            "roc_auc" => _jnum(r.roc_auc),
            "scrub_preserved" => _jnum(r.scrub_preserved),
            "oracle_control_f1" => _jnum(r.oracle_control.f1),
            "F" => _jnum(r.triad.F), "S" => _jnum(r.triad.S), "M" => _jnum(r.triad.M),
        ))
    end
    mean(f) = isempty(results) ? 0.0 : sum(f, results) / length(results)
    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseC_mechanistic",
        "method" => "ACDC(automatic circuit discovery)", "scope" => "core ($(length(results)) games)",
        "where" => where_str, "commit" => _git_commit(),
        "timestamp" => string(round(Int, time())),
        "all_games_positive_control_passes" => all_ctrl_ok,
        "mean_best_f1" => _jnum(mean(r -> r.best_score.f1)),
        "mean_roc_auc" => _jnum(mean(r -> r.roc_auc)),
        "mean_scrub_preserved" => _jnum(mean(r -> r.scrub_preserved)),
        "mean_triad_F" => _jnum(mean(r -> r.triad.F)),
        "mean_triad_S" => _jnum(mean(r -> r.triad.S)),
        "mean_triad_M" => _jnum(mean(r -> r.triad.M)),
        "per_game" => per_game,
        "note" => "Phase-C ACDC (Conmy et al. 2023) over the core games; auto-discovered " *
            "data-flow circuit by resample-ablation edge pruning + τ sweep (ROC), scored " *
            "edge P/R/F1 + scrubbing-preserved performance vs the exhaustive true " *
            "data-flow (experiment_design.md §6/§7: Partial→Succeed — goal state = our " *
            "ground truth).",
    )
    open(path, "w") do io; JSON.print(io, rec, 2); end
    return path
end

# ===========================================================================
# Self-check (DoD) — positive control + scoring contract on the pilot game.
# ===========================================================================
"""
    selftest(r::GameResult) -> Bool

Asserts ACDC's load-bearing claims:

  (BIT-EXACT) the baseline re-run is byte-identical (every edge effect is a clean
    causal re-run on the real ROM).

  (GROUND-TRUTH ANCHORED) the true data-flow circuit has ≥1 off-diagonal edge
    (otherwise the discovery score is degenerate).

  (POSITIVE CONTROL) the oracle-as-ACDC (exhaustive keep-rule) scores F1=1 vs the
    true graph — the harness rewards a perfectly-faithful discovery.

  (τ MONOTONE) the discovered edge count is non-increasing as τ increases (the
    pruning is a proper threshold) and τ→0 recovers ≥ the best-τ recall.

  (METRIC RANGES) P,R,F1,ROC-AUC,F,S,M ∈ [0,1]; all finite.

Throws on a contract violation."""
function selftest(r::GameResult)
    @assert r.bit_exact "[acdc:$(r.game)] bit-exact baseline re-run failed"
    n = length(r.cands)
    offdiag_true = 0
    for i in 1:n, j in 1:n
        i != j && r.true_graph[i, j] && (offdiag_true += 1)
    end
    @assert offdiag_true >= 1 "[acdc:$(r.game)] true data-flow circuit has NO off-diagonal " *
        "edge at this state — uninformative; pick a livelier target frame"

    oc = r.oracle_control
    @assert (oc.precision > 0.999 && oc.recall > 0.999 && oc.f1 > 0.999) "" *
        "[acdc:$(r.game)] positive control broken: oracle-as-ACDC " *
        "P=$(oc.precision) R=$(oc.recall) F1=$(oc.f1) (expected 1/1/1)"

    # τ monotonicity: discovered-edge count must be non-increasing in τ.
    counts = [s.n_disc for s in r.sweep]
    for k in 2:length(counts)
        @assert counts[k] <= counts[k - 1] "[acdc:$(r.game)] discovered-edge count not " *
            "monotone non-increasing in τ ($(counts[k-1]) -> $(counts[k]) at τ step $k)"
    end

    for (nm, v) in (("P", r.best_score.precision), ("R", r.best_score.recall),
                    ("F1", r.best_score.f1), ("AUC", r.roc_auc),
                    ("S", r.triad.S), ("F", r.triad.F))
        @assert 0.0 - 1e-9 <= v <= 1.0 + 1e-9 "[acdc:$(r.game)] $nm out of [0,1]: $v"
    end
    # M = |U*|/|U_hat| ∈ (0,1], or NaN (undefined — 0 edges named); both are valid.
    @assert isnan(r.triad.M) || (0.0 - 1e-9 <= r.triad.M <= 1.0 + 1e-9) "[acdc:$(r.game)] M out of (0,1]: $(r.triad.M)"

    println("[acdc:$(r.game)] SELF-CHECK PASS:")
    println("[acdc:$(r.game)]   bit-exact baseline re-run: $(r.bit_exact)")
    println("[acdc:$(r.game)]   true edges = $offdiag_true (off-diagonal); positive control " *
            "F1 = $(round(oc.f1, digits = 3))")
    println("[acdc:$(r.game)]   discovered best F1 = $(round(r.best_score.f1, digits = 3))  " *
            "P = $(round(r.best_score.precision, digits = 3))  " *
            "R = $(round(r.best_score.recall, digits = 3))  ROC-AUC = $(round(r.roc_auc, digits = 3))")
    println("[acdc:$(r.game)]   scrubbing-preserved = $(round(r.scrub_preserved, digits = 3))")
    println("[acdc:$(r.game)]   TRIAD F=$(round(r.triad.F,digits=3)) " *
            "S=$(round(r.triad.S,digits=3)) M=$(round(r.triad.M,digits=3))")
    return true
end

# ===========================================================================
# Shard selection (cluster contract: --shard i --nshards n --shard-kind game).
# ===========================================================================
"""Select this shard's games from the full game list. shard-kind=game shards the
game list round-robin; any other kind is treated as a no-op (run all)."""
function select_shard(games::Vector{String}, shard::Int, nshards::Int, kind::AbstractString)
    (nshards <= 1 || lowercase(string(kind)) != "game") && return games
    sel = String[]
    for (k, g) in enumerate(games)
        ((k - 1) % nshards) == (shard % nshards) && push!(sel, g)
    end
    return sel
end

# ===========================================================================
# CLI
# ===========================================================================
function main(args = ARGS)
    games = copy(CORE_GAMES)
    target_frame = 30; horizon = 30
    selftest_only = false
    shard = 0; nshards = 1; shard_kind = "game"
    out_dir = OUT_DIR
    where_str = "local"
    i = 1
    while i <= length(args)
        a = args[i]
        if     a == "--games"
            games = xai_resolve_games(args[i+1], CORE_GAMES); i += 2
        elseif a == "--game";         games = [args[i+1]]; i += 2
        elseif a == "--target-frame"; target_frame = parse(Int, args[i+1]); i += 2
        elseif a == "--horizon";      horizon = parse(Int, args[i+1]); i += 2
        elseif a == "--selftest" || a == "--self-check"; selftest_only = true; i += 1
        elseif a == "--shard";        shard = parse(Int, args[i+1]); i += 2
        elseif a == "--nshards";      nshards = parse(Int, args[i+1]); i += 2
        elseif a == "--shard-kind";   shard_kind = args[i+1]; i += 2
        elseif a == "--out-dir";      out_dir = args[i+1]; i += 2
        elseif a == "--roms-dir";     push!(_EXTRA_ROM_DIRS, args[i+1]); i += 2
        elseif a == "--where";        where_str = args[i+1]; i += 2
        else; i += 1
        end
    end

    if selftest_only
        g = "space_invaders" in games ? "space_invaders" : games[1]
        println("[acdc] --selftest on $g (target_frame=$target_frame horizon=$horizon)")
        r = run_game(spec_for(g); target_frame = target_frame, horizon = horizon, verbose = true)
        selftest(r)
        println("[acdc] --selftest: passed, not writing artifacts.")
        return 0
    end

    games = select_shard(games, shard, nshards, shard_kind)
    println("[acdc] automatic circuit discovery over $(length(games)) game(s): " *
            "$(join(games, ",")) (target_frame=$target_frame horizon=$horizon, " *
            "shard=$shard/$nshards kind=$shard_kind, where=$where_str, out=$out_dir)")

    results = GameResult[]
    blockers = String[]
    for g in games
        println("\n========== $g ==========")
        try
            r = run_game(spec_for(g); target_frame = target_frame, horizon = horizon, verbose = true)
            selftest(r)
            jp, np = write_game_result(r; out_dir = out_dir, where_str = where_str)
            println("[$g] wrote $jp")
            println("[$g] arrays  $np")
            push!(results, r)
        catch e
            msg = sprint(showerror, e)
            println("[acdc] !! $g FAILED: $msg")
            push!(blockers, "$g: $msg")
        end
    end

    if isempty(results)
        println("[acdc] no games scored successfully; blockers: $blockers")
        return 1
    end

    if length(results) > 1
        sp = write_summary(results; out_dir = out_dir, where_str = where_str)
        println("\n[acdc] core summary -> $sp")
    end

    println("\n[acdc] ---- summary (discovered circuit vs exhaustive true data-flow) ----")
    for r in results
        println("[acdc]   $(rpad(r.game, 16)) nodes=$(length(r.cands)) " *
                "true=$(r.best_score.n_true) disc=$(r.best_score.n_disc) " *
                "P=$(round(r.best_score.precision,digits=2)) R=$(round(r.best_score.recall,digits=2)) " *
                "F1=$(round(r.best_score.f1,digits=2)) AUC=$(round(r.roc_auc,digits=2)) " *
                "| scrub=$(round(r.scrub_preserved,digits=2)) " *
                "| ctrl F1=$(round(r.oracle_control.f1,digits=2)) " *
                "| F=$(round(r.triad.F,digits=2)) S=$(round(r.triad.S,digits=2)) M=$(round(r.triad.M,digits=2))")
    end
    isempty(blockers) || println("[acdc] blockers: $blockers")
    return 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    ACDC.main()
end
