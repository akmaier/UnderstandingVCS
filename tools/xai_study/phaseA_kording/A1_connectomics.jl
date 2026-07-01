# A1_connectomics.jl — Phase-A analysis A1 (P2-E3-1), JULIA path.
#
# CONNECTOMICS / DATA-FLOW GRAPH RECOVERY on the VCS (experiment_design.md §4,
# row A1). The neuroscience analogy: reconstruct the "wiring diagram" of a system
# from interventions — perturb one unit, watch which other units respond, and from
# that build the directed influence graph. Here the "units" are the candidate RAM
# cells (E2-1 import) and the "wiring" is the program's inter-cell DATA-FLOW: which
# cell's value, when changed at frame t, causally changes which other cell's value
# downstream.
#
# Phase A is the *calibration baseline* of Paper 2 (experiment_design.md §4 Novelty
# note): classical neuroscience methods score LOW-to-PARTIAL faithfulness despite
# the system having rich, fully-known structure — the quantified-Kording lesson.
# A1 makes that concrete for connectomics: a CHEAP single-shot, single-step
# perturbation recovery (the classical method under test) is scored against the
# EXHAUSTIVE exact-intervention oracle's data-flow adjacency (the ground truth),
# and the gap is the method's faithfulness deficit.
#
# ---------------------------------------------------------------------------
# What is the GROUND TRUTH (the "true read/write graph", T2)?
#
# Following the validated Phase-A/-C methodology (pilot_si.jl, pilot_patch_sae.jl):
# the EXACT intervention oracle IS the ground-truth instrument. Every Δ is a real
# bit-exact re-run of the real ROM (Paper-1 64/64), so a measured "cell i causally
# changes cell j" is a TRUE data-flow edge — no world model assumed. We define the
# true directed data-flow graph over the candidate cells as:
#
#   TRUE edge  i -> j   iff   intervening on cell i (the EXHAUSTIVE do-set:
#                              occlude->0, set->base+17, set->base+37) and re-running
#                              the deterministic emulator for the full horizon makes
#                              cell j's value differ from the un-intervened baseline,
#                              for ANY of the perturbation values.
#
# This is exactly the exact-intervention oracle's causal map restricted to
# (candidate cell -> candidate cell) at-final-frame readouts. The self-edge i->i is
# excluded from scoring (it is a tautology: poking i changes i). It is the program's
# observed inter-variable adjacency over these cells at this state/horizon — the
# read/write data-flow the disassembly encodes, measured causally.
#
# ---------------------------------------------------------------------------
# The METHOD UNDER TEST (classical connectomics recovery):
#
#   RECOVERED edge  i -> j  iff   a SINGLE cheap perturbation of cell i
#                                 (do(i := base+17)) at frame t makes cell j differ
#                                 ONE STEP later (t+1) from baseline.
#
# This mirrors how connectomics is done in practice: a single, thin perturbation
# and a short-latency readout, rather than the oracle's exhaustive multi-value,
# full-horizon sweep. The single magnitude can miss edges whose effect needs a
# different sign/value to fire, and the one-step latency misses edges that take
# several frames to propagate — so the recovered graph generally DIFFERS from the
# true graph. That difference, scored, is the quantified-Kording faithfulness gap.
#
# ---------------------------------------------------------------------------
# SCORING (experiment_design.md §4 A1: precision/recall + graph-edit-distance):
#
#   precision = |E_rec ∩ E_true| / |E_rec|
#   recall    = |E_rec ∩ E_true| / |E_true|
#   F1        = 2 P R / (P + R)
#   GED       = |E_rec \ E_true| + |E_true \ E_rec|   (symmetric edge difference;
#               the standard graph-edit-distance for a FIXED node set — only
#               edge insertions/deletions, no node ops — = #false + #missed edges)
#   GED_norm  = GED / (N(N-1))   (normalised by the #possible directed edges)
#
#   POSITIVE CONTROL (oracle-as-method, as in pilot_si.jl): run the recovery with
#   the FULL oracle do-set + FULL horizon (i.e. the true-graph procedure itself) and
#   score it — it must yield P=R=F1=1, GED=0. This proves the harness REWARDS a
#   perfectly-faithful recovery, so a sub-1 classical F1 is a real measurement, not
#   a broken scorer.
#
# F/S/M triad (correctness triad, experiment_design.md §0), scored vs the oracle:
#   F (faithful)   — F1 of the recovered data-flow edges vs the true edges.
#   S (sufficient) — predict a HELD-OUT intervention's edge set: recover the graph
#                    with a DIFFERENT single perturbation value (do(i := base+37) —
#                    a value the recovery above never used) and measure how well the
#                    base+17 recovered graph predicts the base+37 true edges (edge
#                    agreement) — generalisation to an unseen intervention, scored on
#                    the bit-exact re-run, not on a method.
#   M (minimal)    — 1 − over-claim rate: of the edges the method asserts, the
#                    fraction the oracle says do NOT exist (spurious wiring) — the
#                    classical method's tendency to hallucinate connections.
#
# BUILDS ON the verified jutari foundation (NO emulator core touched):
#   * tools/xai_study/common/jutari_oracle.jl — boot/replay/snapshot/deepcopy
#     checkpoint/intervene_ram!/bit-exact baseline/dependency-free NPZ writer.
#     (The oracle's rom_path_for/settings_for are Pong/Breakout/SI-only, so this
#     runner carries a self-contained per-game ROM-basename + RomSettings map — it
#     does NOT modify the shared lib — and constructs the env via the public JuTari
#     API the lib itself uses.)
#   * tools/xai_study/ground_truth/oracle_intervene.jl — the causal-map reference
#     whose exact-intervention definition of Δ this A1 graph reuses cell-to-cell.
#   * each game's tools/xai_study/t3/out/candidates_<game>.json — the candidate
#     cells (the "units").
#
# Run (warm shared depot, primary's project):
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseA_kording/A1_connectomics.jl
# Flags: --games pong,breakout,...   --target-frame N   --horizon N
#        --selftest   (run the self-check on the pilot game, write nothing)
#        --game G      (single game; alias for --games G)
#
# Writes (SPEC §R; file_scope A1_*): one record per game +
#   tools/xai_study/phaseA_kording/out/A1_connectomics.{json,npz}   (combined)
#   tools/xai_study/phaseA_kording/out/A1_<game>.{json,npz}         (per-game)

module A1Connectomics

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_ram, get_screen
using JuTari.JoystickGames: MsPacmanRomSettings, QbertRomSettings

# the verified foundation (NO core touched) — reuse the dependency-free NPZ writer
# + the RAM_SIZE constant from the oracle helper.
include(joinpath(@__DIR__, "..", "common", "jutari_oracle.jl"))
using .JutariOracle: write_npz, RAM_SIZE

# The §1 exact-intervention oracle — used ONLY to build the P2 SHARED gameplay
# state + its cause-density gate (below). A1 keeps its OWN GameSpec/Snap/Candidate
# machinery; the oracle here supplies the shared action stream + gate, not the
# graph. Referenced as OracleIntervene.X / OracleIntervene.JutariOracle.X (NOT
# alias-imported, so its internal JutariOracle instance never clashes with the one
# included above — separate module instances; no type is mixed across them).
include(joinpath(@__DIR__, "..", "ground_truth", "oracle_intervene.jl"))
using JuTari.Diff: soft_ram_peek

import JSON

const OUT_DIR = joinpath(@__DIR__, "out")
const PRIMARY_REPO = get(ENV, "XAI_PRIMARY_REPO",
                         "/Users/maier/Documents/code/UnderstandingVCS")

# The P2 SHARED TESTBED (xai_paper/xai_2_interpretability/experiment_redesign.md):
# seeded random-action GAMEPLAY state + oracle cause-density gate. Included as a
# fragment (see its header for why not a module). Phase A is not a gradient method,
# so we consume the shared action STREAM + cause-density GATE only, and boot A1's
# OWN checkpoint from that stream so the graph machinery is unchanged. Opt in with
# XAI_SHARED_TESTBED=1 (default on).
include(joinpath(@__DIR__, "..", "common", "shared_testbed_impl.jl"))

# shared-testbed switch + params (redesign protocol: prefix=90 gameplay, horizon=15).
const SHARED_TESTBED = get(ENV, "XAI_SHARED_TESTBED", "1") == "1"
const ST_PREFIX  = parse(Int, get(ENV, "XAI_ST_PREFIX", "90"))
const ST_HORIZON = parse(Int, get(ENV, "XAI_ST_HORIZON", "15"))
const ST_SEED    = parse(Int, get(ENV, "XAI_ST_SEED", "0"))
const ST_GATE_K  = parse(Int, get(ENV, "XAI_ST_GATE_K", "4"))
const ST_FLOOR   = parse(Float64, get(ENV, "XAI_ST_FLOOR", "0.5"))

# ===========================================================================
# Per-game ROM-basename + RomSettings map (self-contained; mirrors the canonical
# tools/jutari_screen_dump.jl _SETTINGS_BY_BASENAME so boot/render parity holds).
# The ROM-on-disk basenames are the real filenames under xitari/roms/ (e.g.
# ms_pacman -> mspacman.bin). Seaquest has NO jutari RomSettings yet (its conformance
# fixture is rendered with Generic; SCRUM/test_screen_conformance §note), so it uses
# GenericRomSettings — recorded honestly per game.
# ===========================================================================
struct GameSpec
    name::String
    rom_basename::String
    settings::Function          # () -> RomSettings
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

const CORE_GAMES = ["pong", "breakout", "space_invaders", "seaquest", "ms_pacman", "qbert"]

"""Absolute ROM path for a game, searching this worktree's xitari/roms then the
primary repo (ROMs are gitignored — used in place)."""
function rom_path_for(spec::GameSpec)
    here = normpath(joinpath(@__DIR__, "..", "..", ".."))
    for base in (here, PRIMARY_REPO)
        p = joinpath(base, "xitari", "roms", spec.rom_basename)
        isfile(p) && return p
    end
    error("ROM not found for $(spec.name) (looked for $(spec.rom_basename) under " *
          "$(here) and $(PRIMARY_REPO))")
end

"""A freshly-reset env with the xitari-parity boot (60 NOOP + 4 RESET) and the
game's RomSettings."""
function fresh_env(spec::GameSpec)
    rom = read(rom_path_for(spec))
    env = StellaEnvironment(rom, spec.settings())
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    return env
end

# --- snapshot / replay / intervene (same primitives as jutari_oracle.jl) -----
struct Snap
    ram::Vector{UInt8}
    screen::Matrix{UInt8}
end
snap(env) = Snap(copy(collect(get_ram(env))), Matrix{UInt8}(get_screen(env)))

"""Boot + step `actions[1:target_frame]`; return the env AT the target frame (the
reusable checkpoint; deepcopy before mutating)."""
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

# ===========================================================================
# Candidate cells (E2-1 import) — the "units" the connectomics probes.
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
    for base in (here, PRIMARY_REPO)
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
    return out, path
end

# ===========================================================================
# Graph recovery + the ground-truth graph (both over the candidate cells).
# ===========================================================================
# An adjacency matrix A[i,j] = true means a directed data-flow edge i -> j (cell i
# causally influences cell j). The diagonal (self-edges) is excluded from scoring.

"""The TRUE data-flow graph (the ground-truth read/write adjacency over the
candidate cells): edge i -> j iff the EXHAUSTIVE oracle do-set on cell i
(occlude->0, set->base+17, set->base+37) changes cell j at the FULL horizon, for
ANY perturbation value. Computed by the bit-exact intervention oracle."""
function true_dataflow_graph(checkpoint, tail, cands::Vector{Candidate}, at_target::Snap)
    n = length(cands)
    A = falses(n, n)
    base = continue_from(checkpoint, tail).ram
    for (i, c) in enumerate(cands)
        bval = Int(at_target.ram[c.ram_index + 1])
        # any-effect over the exhaustive perturbation set
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

"""The RECOVERED data-flow graph (the classical connectomics method under test):
edge i -> j iff a SINGLE cheap perturbation do(i := base+`delta`) makes cell j
differ ONE STEP later (latency-`latency`, default 1 frame) from baseline. A thin,
single-shot, short-latency recovery — the realistic connectomics procedure."""
function recovered_dataflow_graph(checkpoint, tail, cands::Vector{Candidate},
                                  at_target::Snap; delta::Int = 17, latency::Int = 1)
    n = length(cands)
    A = falses(n, n)
    short_tail = Int.(tail[1:min(latency, length(tail))])
    base = continue_from(checkpoint, short_tail).ram
    for (i, c) in enumerate(cands)
        bval = Int(at_target.ram[c.ram_index + 1])
        v = (bval + delta) & 0xFF
        v == bval && continue        # degenerate perturbation (delta wraps to 0)
        r = intervene_continue(checkpoint, c.ram_index, v, short_tail).ram
        for (j, cj) in enumerate(cands)
            if r[cj.ram_index + 1] != base[cj.ram_index + 1]
                A[i, j] = true
            end
        end
    end
    return A
end

# --- scoring: precision / recall / F1 / GED over the off-diagonal edges ------
struct GraphScore
    n_nodes::Int
    n_true_edges::Int
    n_rec_edges::Int
    tp::Int
    fp::Int
    fn::Int
    precision::Float64
    recall::Float64
    f1::Float64
    ged::Int                # symmetric edge difference = fp + fn
    ged_norm::Float64       # ged / (n(n-1))
end

"""Score a recovered adjacency against the true adjacency over OFF-DIAGONAL edges
(self-edges excluded — they are tautological)."""
function score_graph(rec::BitMatrix, tru::BitMatrix)
    n = size(tru, 1)
    @assert size(rec) == size(tru) "adjacency shape mismatch"
    offdiag = trues(n, n)
    for k in 1:n; offdiag[k, k] = false; end
    R = rec .& offdiag
    T = tru .& offdiag
    tp = count(R .& T)
    fp = count(R .& .!T)
    fn = count(.!R .& T)
    n_true = count(T); n_rec = count(R)
    precision = n_rec == 0 ? (n_true == 0 ? 1.0 : 0.0) : tp / n_rec
    recall    = n_true == 0 ? 1.0 : tp / n_true
    f1 = (precision + recall) == 0 ? 0.0 : 2 * precision * recall / (precision + recall)
    ged = fp + fn
    denom = n * (n - 1)
    ged_norm = denom == 0 ? 0.0 : ged / denom
    return GraphScore(n, n_true, n_rec, tp, fp, fn, precision, recall, f1,
                      ged, ged_norm)
end

# ===========================================================================
# F / S / M triad (correctness triad, scored against the oracle).
# ===========================================================================
struct Triad
    F::Float64
    S::Float64
    M::Float64
    F_note::String
    S_note::String
    M_note::String
end

"""F = F1 of recovered vs true edges. S = generalisation to a HELD-OUT
intervention value: how well the base+17 recovered graph predicts the base+37 TRUE
edge set (edge-agreement = (TP+TN)/all off-diagonal). M = 1 − over-claim rate
(fraction of recovered edges the oracle says don't exist)."""
function score_triad(rec::BitMatrix, tru::BitMatrix, heldout_true::BitMatrix,
                     gs::GraphScore)
    F = gs.f1
    # S — predict the held-out (base+37) TRUE edges from the base+17 recovered
    #     graph: off-diagonal edge-agreement (accuracy). A faithful recovery
    #     generalises across the unseen intervention value.
    n = size(tru, 1)
    offdiag = trues(n, n); for k in 1:n; offdiag[k, k] = false; end
    R = rec .& offdiag
    H = heldout_true .& offdiag
    agree = count((R .& H) .| (.!R .& .!H .& offdiag))
    total = count(offdiag)
    S = total == 0 ? 1.0 : agree / total
    # M — minimality / no spurious wiring: 1 − (false edges / recovered edges)
    M = gs.n_rec_edges == 0 ? 1.0 : 1.0 - gs.fp / gs.n_rec_edges
    return Triad(F, S, M,
        "F1 of recovered data-flow edges vs the exact-oracle true edges ($(gs.n_nodes) candidate cells)",
        "off-diagonal edge-agreement between the base+17 recovered graph and the HELD-OUT base+37 true edge set (generalisation to an unseen intervention value, bit-exact re-run)",
        "1 − over-claim rate: fraction of recovered edges the oracle says are spurious (false wiring)")
end

# ===========================================================================
# Per-game run.
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
    true_graph::BitMatrix
    rec_graph::BitMatrix
    heldout_true_graph::BitMatrix
    oracle_method_graph::BitMatrix     # the positive control's recovered graph
    score::GraphScore
    oracle_control::GraphScore         # oracle-as-method vs true
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

# A1's candidates-path resolver, in the shape the shared testbed injects (a
# String->path-or-nothing map). Mirrors load_candidates' file search.
function _st_candidates_path_for(game::AbstractString)
    rel = joinpath("tools", "xai_study", "t3", "out", "candidates_$(game).json")
    here = normpath(joinpath(@__DIR__, "..", "..", ".."))
    for base in (here, PRIMARY_REPO)
        p = joinpath(base, rel)
        isfile(p) && return p
    end
    return nothing
end

"""Build the P2 SHARED gameplay state + cause-density gate for `game` using the §1
oracle machinery (oracle_intervene.jl). Returns the substrate NamedTuple (we use
its `.actions` stream + `.cause_density`/`.accepted`/`.cell` gate). A1 then boots
its OWN checkpoint from `st.actions` so the graph algorithm is unchanged."""
function build_a1_shared_state(game::AbstractString; verbose = false)
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

function run_game(game::AbstractString; target_frame = 30, horizon = 30, verbose = true)
    haskey(GAME_SPECS, game) || error("unknown game $game (have $(keys(GAME_SPECS)))")
    spec = GAME_SPECS[game]

    # SHARED-TESTBED (redesign): replace the all-NOOP boot/attract tape with a
    # seeded random-action GAMEPLAY state at f*=ST_PREFIX, gated by the oracle
    # cause-density gate. We take the shared action STREAM + the gate from the
    # substrate, then boot A1's OWN checkpoint from that stream so the graph
    # machinery (GameSpec/Snap/Candidate) is UNCHANGED — only the state moves.
    st = nothing
    if SHARED_TESTBED
        st = build_a1_shared_state(game; verbose = verbose)
        target_frame = st.prefix; horizon = st.horizon
        actions = st.actions
        verbose && println("[A1:$game] SHARED gameplay state: cause_density=$(st.cause_density)/" *
            "$(length(st.causes)) accepted=$(st.accepted) cell=$(st.cell)")
    else
        actions = fill(0, target_frame + horizon)   # all-NOOP deterministic trace
    end
    total = target_frame + horizon
    tail = Int.(actions[target_frame + 1 : total])

    # 1) bit-exact baseline guarantee — two fresh boots+replays must be identical.
    verbose && println("[A1:$game] asserting bit-exactness (2 fresh boots+replays to f$total)...")
    a = continue_from(boot_replay(spec, actions, target_frame), tail)
    b = continue_from(boot_replay(spec, actions, target_frame), tail)
    bit_exact = (a.ram == b.ram) && (a.screen == b.screen)
    bit_exact || error("[A1:$game] bit-exact re-run FAILED to f$total — refusing to score")
    verbose && println("[A1:$game] bit-exact re-run: PASS")

    # 2) ONE checkpoint at the intervention frame (boot+to-target paid once).
    checkpoint = boot_replay(spec, actions, target_frame)
    at_target = continue_from(checkpoint, Int[])      # state AT the target frame

    cands, cand_path = load_candidates(game)
    if isempty(cands)
        error("[A1:$game] no candidate cells found (candidates_$(game).json missing/empty)")
    end
    verbose && println("[A1:$game] candidates: $(cand_path) ($(length(cands)) cells)")

    # 3) the TRUE data-flow graph (exact oracle, exhaustive do-set, full horizon).
    verbose && println("[A1:$game] true data-flow graph (exact oracle, $(length(cands)) cells)...")
    true_graph = true_dataflow_graph(checkpoint, tail, cands, at_target)

    # 4) the RECOVERED graph (classical: single do(base+17), one-step latency).
    verbose && println("[A1:$game] recovered graph (classical: single perturb, 1-step latency)...")
    rec_graph = recovered_dataflow_graph(checkpoint, tail, cands, at_target;
                                         delta = 17, latency = 1)

    # 5) HELD-OUT true graph (exhaustive oracle on a DIFFERENT value, base+37 only)
    #    — the S probe: a value the recovery never used.
    heldout_true = heldout_true_graph(checkpoint, tail, cands, at_target; delta = 37)

    # 6) POSITIVE CONTROL — oracle-as-method: recover with the FULL oracle do-set +
    #    FULL horizon (the true-graph procedure). Must score P=R=F1=1, GED=0.
    oracle_method = true_dataflow_graph(checkpoint, tail, cands, at_target)

    score = score_graph(rec_graph, true_graph)
    oracle_control = score_graph(oracle_method, true_graph)
    triad = score_triad(rec_graph, true_graph, heldout_true, score)

    if verbose
        println("[A1:$game] ---- scores ----")
        println("[A1:$game]   nodes=$(score.n_nodes)  true_edges=$(score.n_true_edges)  " *
                "rec_edges=$(score.n_rec_edges)")
        println("[A1:$game]   P=$(round(score.precision,digits=3))  " *
                "R=$(round(score.recall,digits=3))  F1=$(round(score.f1,digits=3))  " *
                "GED=$(score.ged) (norm=$(round(score.ged_norm,digits=3)))")
        println("[A1:$game]   positive control (oracle-as-method): " *
                "P=$(round(oracle_control.precision,digits=3)) " *
                "R=$(round(oracle_control.recall,digits=3)) " *
                "F1=$(round(oracle_control.f1,digits=3)) GED=$(oracle_control.ged)")
        println("[A1:$game]   TRIAD: F=$(round(triad.F,digits=3))  " *
                "S=$(round(triad.S,digits=3))  M=$(round(triad.M,digits=3))")
    end

    return GameResult(game, spec.settings_name, spec.rom_basename, cand_path,
                      target_frame, horizon, bit_exact, cands, true_graph, rec_graph,
                      heldout_true, oracle_method, score, oracle_control, triad,
                      st === nothing ? "noop" : "seeded_random_action_gameplay",
                      st === nothing ? -1 : st.seed,
                      st === nothing ? -1 : st.prefix,
                      st === nothing ? -1 : st.cause_density,
                      st === nothing ? false : st.accepted,
                      st === nothing ? 0 : length(st.causes),
                      st === nothing ? (-1, -1) : st.cell)
end

"""The HELD-OUT true data-flow graph: edge i -> j iff a do(i := base+`delta`)
(default base+37) changes cell j at the full horizon. A single-value true graph
on a value the base+17 recovery never used (the S probe)."""
function heldout_true_graph(checkpoint, tail, cands::Vector{Candidate}, at_target::Snap;
                            delta::Int = 37)
    n = length(cands)
    A = falses(n, n)
    base = continue_from(checkpoint, tail).ram
    for (i, c) in enumerate(cands)
        bval = Int(at_target.ram[c.ram_index + 1])
        v = (bval + delta) & 0xFF
        v == bval && continue
        r = intervene_continue(checkpoint, c.ram_index, v, tail).ram
        for (j, cj) in enumerate(cands)
            if r[cj.ram_index + 1] != base[cj.ram_index + 1]
                A[i, j] = true
            end
        end
    end
    return A
end

# ===========================================================================
# Persist (SPEC §R) — combined record + per-game records; file_scope A1_*.
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

function _game_record(r::GameResult)
    cell_names = ["RAM[$(c.ram_index)]:$(c.concept)" for c in r.cands]
    Dict{String,Any}(
        "paper" => "P2",
        "phase" => "phaseA_kording",
        "method" => "A1_connectomics(single-shot perturbation graph recovery)",
        "game" => r.game,
        "state" => r.state_kind == "noop" ? "f$(r.target_frame)+$(r.horizon)" :
                   "gameplay(seed=$(r.st_seed),prefix=$(r.st_prefix))+$(r.horizon)",
        "target_output" => "inter-cell data-flow graph (candidate RAM cells)",
        # headline scalar (SPEC §R value/metric_name): the recovered-graph F1.
        "metric_name" => "dataflow_graph_F1_vs_oracle",
        "value" => _jnum(r.triad.F),
        "stderr" => nothing,
        "ci" => nothing,
        "n" => length(r.cands),
        "seed" => 0,
        "where" => "local",
        "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene@$(r.game)#cell-to-cell data-flow (exhaustive do-set, full horizon)",
        "timestamp" => string(round(Int, time())),
        "extra" => Dict{String,Any}(
            "substrate" => "jutari (Julia, HARD) — real-ROM bit-exact path; every " *
                "edge is an EXACT intervention re-run on the true ROM.",
            "settings" => r.settings_name,
            "rom_basename" => r.rom_basename,
            "candidates_file" => r.candidates_path,
            "bit_exact_rerun" => r.bit_exact,
            "candidate_cells" => cell_names,
            "candidate_ram_indices" => [c.ram_index for c in r.cands],
            "testbed" => Dict{String,Any}(
                "state_kind" => r.state_kind,
                "seed" => r.st_seed, "prefix" => r.st_prefix, "horizon" => r.horizon,
                "shared_output" => "screen_region(n_changed_px)@r$(r.shared_cell[1])c$(r.shared_cell[2])",
                "cause_density_above_floor" => r.cause_density,
                "cause_density_floor" => ST_FLOOR, "cause_density_gate_k" => ST_GATE_K,
                "cause_density_accepted" => r.cause_density_accepted, "n_causes" => r.n_causes,
                "note" => "P2 redesign: the connectomics recovery runs on a seeded " *
                    "random-action GAMEPLAY state (not the boot/attract NOOP tape), " *
                    "gated by the §1 oracle cause-density gate. A1's graph algorithm " *
                    "is unchanged; only the analysis state moves onto genuine " *
                    "input-driven gameplay."),
            "graph" => Dict{String,Any}(
                "n_nodes" => r.score.n_nodes,
                "n_true_edges" => r.score.n_true_edges,
                "n_recovered_edges" => r.score.n_rec_edges,
                "tp" => r.score.tp, "fp" => r.score.fp, "fn" => r.score.fn,
                "precision" => _jnum(r.score.precision),
                "recall" => _jnum(r.score.recall),
                "f1" => _jnum(r.score.f1),
                "graph_edit_distance" => r.score.ged,
                "graph_edit_distance_normalised" => _jnum(r.score.ged_norm),
                "true_edges" => _edge_list(r.true_graph),
                "recovered_edges" => _edge_list(r.rec_graph),
                "note" => "edges are 0-based candidate ordinals [i,j] meaning " *
                    "cell i -> cell j. TRUE = exhaustive oracle (do 0/base+17/" *
                    "base+37, full horizon, any value changes j); RECOVERED = " *
                    "classical single do(base+17), 1-step latency. GED = symmetric " *
                    "edge difference (fp+fn) over off-diagonal edges.",
            ),
            "positive_control" => Dict{String,Any}(
                "name" => "oracle-as-method (full do-set + full horizon)",
                "precision" => _jnum(r.oracle_control.precision),
                "recall" => _jnum(r.oracle_control.recall),
                "f1" => _jnum(r.oracle_control.f1),
                "graph_edit_distance" => r.oracle_control.ged,
                "note" => "the oracle's OWN recovery must score F1=1, GED=0 — the " *
                    "harness rewards a perfectly-faithful graph, so the classical " *
                    "F1<1 is a real measurement.",
            ),
            "triad" => Dict{String,Any}(
                "F" => _jnum(r.triad.F), "F_note" => r.triad.F_note,
                "S" => _jnum(r.triad.S), "S_note" => r.triad.S_note,
                "M" => _jnum(r.triad.M), "M_note" => r.triad.M_note,
                "interpretation" => "Phase A is the calibration baseline " *
                    "(experiment_design.md §4): classical connectomics scores " *
                    "LOW-to-PARTIAL faithfulness despite fully-known data-flow — the " *
                    "quantified-Kording lesson. A thin single-shot/short-latency " *
                    "perturbation misses long-latency / sign-dependent edges (recall " *
                    "deficit) and can over-claim transient ones (precision deficit).",
            ),
            "scales_to_cluster_via" =>
                "tools/cluster/xai_*.sbatch (E0-3) — the full A1..A8 battery over " *
                "the core + breadth sets; forward bit-exact to this HARD path.",
        ),
    )
end

function _write_game_npz(r::GameResult, npz_path)
    # adjacency matrices as UInt8 (0/1); per-edge confusion is recoverable.
    to_u8(A::BitMatrix) = UInt8.(A)
    write_npz(npz_path, Dict(
        "candidate_ram_indices" => Int64[c.ram_index for c in r.cands],
        "true_graph"            => to_u8(r.true_graph),
        "recovered_graph"       => to_u8(r.rec_graph),
        "heldout_true_graph"    => to_u8(r.heldout_true_graph),
        "oracle_method_graph"   => to_u8(r.oracle_method_graph),
        "scores"                => Float64[r.score.precision, r.score.recall, r.score.f1,
                                           Float64(r.score.ged), r.score.ged_norm],
        "triad_FSM"             => Float64[r.triad.F, r.triad.S, r.triad.M],
    ))
end

function write_results(results::Vector{GameResult}; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    written = String[]
    # per-game records
    per_game = Dict{String,Any}()
    for r in results
        rec = _game_record(r)
        jp = joinpath(out_dir, "A1_$(r.game).json")
        np = joinpath(out_dir, "A1_$(r.game).npz")
        rec["arrays"] = basename(np)
        open(jp, "w") do io; JSON.print(io, rec, 2); end
        _write_game_npz(r, np)
        push!(written, jp); push!(written, np)
        per_game[r.game] = rec
    end
    # combined record (a self-describing index over the per-game entries)
    f1s = Float64[r.triad.F for r in results]
    mean_f1 = isempty(f1s) ? 0.0 : sum(f1s) / length(f1s)
    combined = Dict{String,Any}(
        "paper" => "P2",
        "phase" => "phaseA_kording",
        "method" => "A1_connectomics(single-shot perturbation graph recovery)",
        "game" => "core_set",
        "state" => "f$(results[1].target_frame)+$(results[1].horizon)",
        "target_output" => "inter-cell data-flow graph (per game)",
        "metric_name" => "mean_dataflow_graph_F1_vs_oracle",
        "value" => _jnum(mean_f1),
        "stderr" => nothing,
        "ci" => nothing,
        "n" => length(results),
        "seed" => 0,
        "where" => "local",
        "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene (exact intervention data-flow per game)",
        "timestamp" => string(round(Int, time())),
        "arrays" => "A1_connectomics.npz",
        "games" => [r.game for r in results],
        "per_game" => per_game,
    )
    cjp = joinpath(out_dir, "A1_connectomics.json")
    open(cjp, "w") do io; JSON.print(io, combined, 2); end
    # combined npz: one stacked vector of per-game [P,R,F1,GED,F,S,M]
    n = length(results)
    M = zeros(Float64, n, 7)
    for (k, r) in enumerate(results)
        M[k, :] = [r.score.precision, r.score.recall, r.score.f1,
                   Float64(r.score.ged), r.triad.F, r.triad.S, r.triad.M]
    end
    cnp = joinpath(out_dir, "A1_connectomics.npz")
    write_npz(cnp, Dict(
        "per_game_PRF_GED_FSM" => M,                # (n_games, 7)
        "game_order" => UInt8.(collect(1:n)),       # ordinal index; names in JSON
    ))
    push!(written, cjp); push!(written, cnp)
    return written
end

# ===========================================================================
# Self-check (DoD) — the scoring contract is sound; results are non-fabricated.
# ===========================================================================
"""
    selftest(r::GameResult) -> Bool

Asserts the load-bearing claims of A1:

  (BIT-EXACT) the baseline re-run is byte-identical (every edge is a clean causal
    effect on the real ROM).

  (GROUND-TRUTH ANCHORED) the true data-flow graph has ≥1 off-diagonal edge —
    otherwise the graph-recovery score is degenerate and the state/horizon are
    uninformative.

  (POSITIVE CONTROL) the oracle-as-method recovery scores P=R=F1=1, GED=0 against
    the true graph — the harness rewards a perfectly-faithful graph (so a sub-1
    classical F1 is a real measurement, not a broken scorer).

  (METRIC RANGES) P,R,F1,S,M ∈ [0,1]; GED ≥ 0; all finite.

Throws on a contract violation."""
function selftest(r::GameResult)
    @assert r.bit_exact "[A1:$(r.game)] bit-exact baseline re-run failed"
    n = r.score.n_nodes
    offdiag_true = 0
    for i in 1:n, j in 1:n
        i != j && r.true_graph[i, j] && (offdiag_true += 1)
    end
    @assert offdiag_true >= 1 "[A1:$(r.game)] true data-flow graph has NO off-diagonal " *
        "edge at this state — uninformative; pick a livelier target frame"

    oc = r.oracle_control
    @assert (oc.precision > 0.999 && oc.recall > 0.999 && oc.f1 > 0.999 && oc.ged == 0) "" *
        "[A1:$(r.game)] positive control broken: oracle-as-method " *
        "P=$(oc.precision) R=$(oc.recall) F1=$(oc.f1) GED=$(oc.ged) (expected 1/1/1/0)"

    for (nm, v) in (("P", r.score.precision), ("R", r.score.recall), ("F1", r.score.f1),
                    ("S", r.triad.S), ("M", r.triad.M), ("F", r.triad.F))
        @assert 0.0 - 1e-9 <= v <= 1.0 + 1e-9 "[A1:$(r.game)] $nm out of [0,1]: $v"
    end
    @assert r.score.ged >= 0 "[A1:$(r.game)] GED negative"

    println("[A1:$(r.game)] SELF-CHECK PASS:")
    println("[A1:$(r.game)]   bit-exact baseline re-run: $(r.bit_exact)")
    println("[A1:$(r.game)]   true edges = $offdiag_true (off-diagonal); positive control " *
            "F1 = $(round(oc.f1, digits = 3)) GED = $(oc.ged)")
    println("[A1:$(r.game)]   recovered F1 = $(round(r.score.f1, digits = 3))  " *
            "P = $(round(r.score.precision, digits = 3))  R = $(round(r.score.recall, digits = 3))")
    println("[A1:$(r.game)]   TRIAD F=$(round(r.triad.F,digits=3)) " *
            "S=$(round(r.triad.S,digits=3)) M=$(round(r.triad.M,digits=3))")
    return true
end

# ===========================================================================
# CLI
# ===========================================================================
function main(args = ARGS)
    games = copy(CORE_GAMES)
    target_frame = 30; horizon = 30
    selftest_only = false
    i = 1
    while i <= length(args)
        a = args[i]
        if     a == "--games";        games = String.(split(args[i+1], ",")); i += 2
        elseif a == "--game";         games = [args[i+1]]; i += 2
        elseif a == "--target-frame"; target_frame = parse(Int, args[i+1]); i += 2
        elseif a == "--horizon";      horizon = parse(Int, args[i+1]); i += 2
        elseif a == "--selftest";     selftest_only = true; i += 1
        else; i += 1
        end
    end

    if selftest_only
        # self-check on the Phase-A pilot game (space_invaders) — fast + lively.
        g = "space_invaders" in games ? "space_invaders" : games[1]
        println("[A1] --selftest on $g (target_frame=$target_frame horizon=$horizon)")
        r = run_game(g; target_frame = target_frame, horizon = horizon, verbose = true)
        selftest(r)
        println("[A1] --selftest: passed, not writing artifacts.")
        return 0
    end

    println("[A1] connectomics / data-flow graph recovery over $(length(games)) games " *
            "(target_frame=$target_frame horizon=$horizon, jutari/Julia path)")
    results = GameResult[]
    blockers = String[]
    for g in games
        try
            r = run_game(g; target_frame = target_frame, horizon = horizon, verbose = true)
            selftest(r)
            push!(results, r)
        catch e
            msg = sprint(showerror, e)
            println("[A1] !! $g FAILED: $msg")
            push!(blockers, "$g: $msg")
        end
    end

    if isempty(results)
        println("[A1] no games scored successfully; blockers: $blockers")
        return 1
    end

    written = write_results(results)
    println("[A1] ---- summary (recovered single-shot graph vs exact-oracle data-flow) ----")
    for r in results
        println("[A1]   $(rpad(r.game, 16)) nodes=$(r.score.n_nodes) " *
                "true=$(r.score.n_true_edges) rec=$(r.score.n_rec_edges) " *
                "P=$(round(r.score.precision,digits=2)) R=$(round(r.score.recall,digits=2)) " *
                "F1=$(round(r.score.f1,digits=2)) GED=$(r.score.ged) " *
                "| ctrl F1=$(round(r.oracle_control.f1,digits=2)) " *
                "| F=$(round(r.triad.F,digits=2)) S=$(round(r.triad.S,digits=2)) M=$(round(r.triad.M,digits=2))")
    end
    isempty(blockers) || println("[A1] blockers: $blockers")
    println("[A1] wrote:")
    for w in written; println("[A1]   $w"); end
    return 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    A1Connectomics.main()
end
