# causal_scrubbing.jl — Phase-C mechanistic interpretability (P2-E5-8), JULIA path
# (the jutari real-ROM substrate; jaxtari eager is ~205× slower — SCRUM §7).
#
# METHOD: CAUSAL SCRUBBING (Chan et al. 2022). Given a HYPOTHESIZED CIRCUIT (a claim
# about which parts of the computation matter for a behaviour, and which are
# irrelevant), causal scrubbing RESAMPLE-ABLATES everything the hypothesis says is
# OUTSIDE the circuit — replaces those activations with values from a DIFFERENT input
# (a "source" run) — and checks whether the behaviour is PRESERVED. If the hypothesis
# is correct, the irrelevant parts genuinely don't matter, so resampling them leaves
# the behaviour unchanged (the loss is RECOVERED → the hypothesis PASSES). If the
# hypothesis is wrong (it claims relevant cells are irrelevant), resampling them
# destroys the behaviour (loss NOT recovered → the hypothesis FAILS).
# (experiment_design.md §6 row "Causal scrubbing": finding = hypothesis pass/fail;
# measured = scrubbing-preserved performance vs the true routine; §7 prediction:
# **Succeed** — "we *have* ground-truth hypotheses to scrub". T1 scoring, no T3.)
#
# ---------------------------------------------------------------------------
# THE GROUND-TRUTH ROUTINE (what makes this a *validation*, not a guess):
#   On the VCS the program's data-flow is KNOWN. We reuse the EXACT inter-cell
#   data-flow graph from A1_connectomics (the TRUE graph: edge i->j iff the
#   exhaustive oracle do-set {0, base+17, base+37} on cell i, re-run for the full
#   horizon, changes cell j — a genuine real-ROM intervention, not a world model).
#   For a chosen BEHAVIOUR (an OUTPUT cell = a sink of the true graph), the TRUE
#   CIRCUIT is the set of cells on a path TO the output: the output's causal
#   ANCESTORS in the true graph (transitive closure), plus the output itself. Every
#   other candidate cell is, by the ground truth, irrelevant to that output.
#
# THE TWO HYPOTHESES WE SCRUB (the §7 discrimination):
#   * TRUE hypothesis    = circuit := {output} ∪ ancestors(output) in the true graph.
#                          Scrubbing everything outside it should PRESERVE the output
#                          (PASS) — the irrelevant cells are genuinely irrelevant.
#   * WRONG hypothesis   = a SCRAMBLED circuit of the SAME SIZE that EXCLUDES a true
#                          ancestor (and includes a non-ancestor instead). Scrubbing
#                          now resample-ablates a cell that DOES drive the output, so
#                          the output breaks (FAIL). This is the negative control that
#                          proves the metric discriminates true from false circuits.
#
# SCRUBBING-PRESERVED PERFORMANCE (the §R headline metric):
#   resample-ablate every NON-circuit candidate cell at frame t (write the value that
#   cell holds in a genuinely different SOURCE run — the standard causal-scrubbing
#   resample ablation, not a zero), then re-run the REAL ROM for `horizon` frames and
#   compare the OUTPUT cell to its clean (un-scrubbed) value:
#       preserved = 1 if scrubbed_output == clean_output else 0   (per output)
#   averaged over the behaviour's output cells. PASS iff preserved == 1.0. We also
#   report a graded behaviour-similarity over ALL candidate cells + the full screen so
#   the failure magnitude is visible, and run the resample over SEVERAL source runs so
#   the preservation is not a single-source fluke.
#
# Why this SUCCEEDS on the VCS (the paper's point): causal scrubbing's premise — a
# faithful hypothesis survives resampling of everything outside the circuit — is
# directly TESTABLE here because the circuit is the program's known data-flow. We
# show the oracle circuit passes and a scrambled one fails, on a real ROM, bit-exact.
#
# No JuTari/jaxtari/xitari core is modified — pure tooling under tools/xai_study/.
# Reuses the validated foundations on main:
#   * oracle_intervene.jl  — candidate_ram_indices (the candidate cell set the oracle
#     scores) + the JutariOracle run helper (boot/replay/snapshot/deepcopy CHECKPOINT/
#     intervene_ram!/intervene_tia! + the dependency-free §R NPZ writer).
#   * the A1 true-graph procedure (exhaustive pairwise interventions = the
#     ground-truth circuit), recomputed here so the hypothesis is the *same* object
#     A1_connectomics scores against (NO emulator core touched).
#
# Run (warm shared depot, primary's project):
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseC_mechanistic/causal_scrubbing.jl --games core
# Flags: --games core|<g1,g2,...>  --game <g>
#        --target-frame N --horizon N --selftest
#        --shard i --nshards n --shard-kind game  --out-dir DIR  --roms-dir DIR
#        --where local|cluster      (cluster-shardable; runs unchanged under
#                                    tools/cluster/xai_array_jl.sbatch)
#        --n-sources K              (accepted for CLI compat; the resample
#                                    distribution is the oracle do-set per cell)
#
# Writes (SPEC §R; file_scope causal_scrubbing_* under out/):
#   tools/xai_study/phaseC_mechanistic/out/causal_scrubbing_<game>.{json,npz}
#   tools/xai_study/phaseC_mechanistic/out/causal_scrubbing_core_summary.json

module CausalScrubbing

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen, get_ram

# the oracle's candidate cause set (game-agnostic) + the verified jutari run helper
# (boot/replay/snapshot/intervene + the dependency-free NPZ writer). NO core touched.
include(joinpath(@__DIR__, "..", "ground_truth", "oracle_intervene.jl"))
using .OracleIntervene: candidate_ram_indices, build_pong_causes, run_intervention
using .OracleIntervene.JutariOracle: Snapshot, snapshot, intervene_ram!,
                                     intervene_tia!, write_npz, RAM_SIZE
using JuTari.Diff: soft_ram_peek

# The P2 SHARED TESTBED (xai_paper/xai_2_interpretability/experiment_redesign.md):
# seeded random-action GAMEPLAY state + oracle cause-density gate + shared
# screen-buffer REGION output. Included as a fragment (see its header) so
# build_shared_testbed operates on OUR own Cause/Snapshot types. Opt in with
# XAI_SHARED_TESTBED=1 (default on for the redesign re-run). Phase C is not a
# gradient method, so the sampler-on path does not apply — we consume the shared
# STATE + cause-density GATE + shared screen-buffer output only.
include(joinpath(@__DIR__, "..", "common", "shared_testbed_impl.jl"))
# the shared game-set + ROM-root resolver (XAI_LABELED / xai_resolve_games / xai_rom_roots).
include(joinpath(@__DIR__, "..", "common", "game_sets.jl"))

# shared triad helpers (jnum_or_null for JSON-null on undefined S/M); guarded.
isdefined(@__MODULE__, :TriadSM) ||
    include(joinpath(@__DIR__, "..", "common", "triad_sm.jl"))
using .TriadSM: jnum_or_null

const OUT_DIR = joinpath(@__DIR__, "out")
const CORE_GAMES = ["pong", "breakout", "space_invaders", "seaquest", "ms_pacman", "qbert"]
# shared-testbed switch + params (redesign protocol: prefix=90 gameplay, horizon=15).
const SHARED_TESTBED = get(ENV, "XAI_SHARED_TESTBED", "1") == "1"
const ST_PREFIX  = parse(Int, get(ENV, "XAI_ST_PREFIX", "90"))
const ST_HORIZON = parse(Int, get(ENV, "XAI_ST_HORIZON", "15"))
const ST_SEED    = parse(Int, get(ENV, "XAI_ST_SEED", "0"))
const ST_GATE_K  = parse(Int, get(ENV, "XAI_ST_GATE_K", "4"))
const ST_FLOOR   = parse(Float64, get(ENV, "XAI_ST_FLOOR", "0.5"))

# joystick action codes (oracle_intervene.jl: RIGHT=3; LEFT=4; FIRE=1). NOOP=0 = BASE.
const ACT_NOOP  = 0
const ACT_FIRE  = 1
const ACT_RIGHT = 3
const ACT_LEFT  = 4

const _PRIMARY_REPO = get(ENV, "XAI_PRIMARY_REPO", "/Users/maier/Documents/code/UnderstandingVCS")

# ============================================================================
# Per-game ROM-basename + RomSettings + ROM-root resolution (mirrors das.jl /
# A2_lesions.jl; NO emulator core touched). seaquest has no registered RomSettings
# yet → Generic (boots fine; the screen scoreboard's Generic fallback).
# ============================================================================
const ROM_BASENAME = Dict(
    "pong" => "pong", "breakout" => "breakout",
    "space_invaders" => "space_invaders", "seaquest" => "seaquest",
    "ms_pacman" => "mspacman", "qbert" => "qbert")

# the shared ROM-root set (this worktree + primary xitari/roms + the 54-ROM store
# tools/rom_sweep/roms + the collection), with an optional explicit --roms-dir
# prepended (cluster flat ROM dir), so all 54 labeled games resolve uniformly.
function _rom_roots(roms_dir)
    extra = roms_dir === nothing ? String[] : String[roms_dir]
    return xai_rom_roots(; primary_repo = _PRIMARY_REPO, extra = extra)
end

function rom_path_for(game::AbstractString; roms_dir = nothing)
    g = lowercase(string(game))
    stem = get(ROM_BASENAME, g, g)
    # try the mapped stem AND the raw ALE name across all roots.
    return xai_find_rom(unique([stem, g]), _rom_roots(roms_dir))
end

function settings_for(game::AbstractString)
    g = lowercase(string(game))
    g == "pong"           && return JuTari.PaddleGames.PongRomSettings()
    g == "breakout"       && return JuTari.PaddleGames.BreakoutRomSettings()
    g == "space_invaders" && return JuTari.SpaceInvadersRomSettings()
    g == "ms_pacman"      && return JuTari.JoystickGames.MsPacmanRomSettings()
    g == "qbert"          && return JuTari.JoystickGames.QbertRomSettings()
    return JuTari.GenericRomSettings()   # seaquest (no registered settings yet)
end

function candidates_path_for(game::AbstractString)
    rel = joinpath("tools", "xai_study", "t3", "out", "candidates_$(game).json")
    here = normpath(joinpath(@__DIR__, "..", "..", ".."))
    for base in (here, _PRIMARY_REPO)
        p = joinpath(base, rel)
        isfile(p) && return p
    end
    return nothing
end

# ============================================================================
# env construction + replay + checkpoint (mirrors jutari_oracle / das / A1)
# ============================================================================
function load_env(; game::AbstractString, roms_dir = nothing)
    rom = read(rom_path_for(game; roms_dir = roms_dir))
    env = StellaEnvironment(rom, settings_for(game))
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    return env
end

function boot_replay(actions, target_frame; game, roms_dir = nothing)
    env = load_env(; game = game, roms_dir = roms_dir)
    for i in 1:target_frame; env_step!(env, Int(actions[i])); end
    return env
end

function continue_from(checkpoint::StellaEnvironment, tail)
    env = deepcopy(checkpoint)
    for a in tail; env_step!(env, Int(a)); end
    return snapshot(env, length(tail))
end

function fresh_baseline(actions, total; game, roms_dir = nothing)
    env = load_env(; game = game, roms_dir = roms_dir)
    for i in 1:total; env_step!(env, Int(actions[i])); end
    return snapshot(env, Int(total))
end

"""Two fresh boots+replays must be byte-identical in RAM AND screen — the
load-bearing guarantee that makes every scrub Δ a clean causal effect (mirrors
OracleIntervene.assert_bit_exact)."""
function assert_bit_exact(actions, total; game, roms_dir = nothing)
    a = fresh_baseline(actions, total; game = game, roms_dir = roms_dir)
    b = fresh_baseline(actions, total; game = game, roms_dir = roms_dir)
    a.ram == b.ram || error("bit-exact RAM re-run FAILED for $game: " *
        "$(count(a.ram .!= b.ram))/$(length(a.ram)) bytes differ to f$total")
    a.screen == b.screen || error("bit-exact SCREEN re-run FAILED for $game: " *
        "$(count(a.screen .!= b.screen)) px differ to f$total")
    return true
end

# ============================================================================
# THE GROUND-TRUTH ROUTINE — the A1 TRUE inter-cell data-flow graph over the
# candidate cells (the SAME object A1_connectomics scores against): edge i->j iff
# the EXHAUSTIVE oracle do-set {0, base+17, base+37} on cell i, re-run for the FULL
# horizon, changes cell j (any value). Computed by the bit-exact intervention oracle.
# ============================================================================
"""Build the TRUE n×n data-flow adjacency (A[i,j] = cell i causally drives cell j)
over the candidate cells `cand_idx` (0-based RAM indices), at `base_ckpt`+`tail`.
This is exactly A1_connectomics.true_dataflow_graph (re-derived here so the scrubbed
circuit IS the ground-truth circuit; NO emulator core touched)."""
function true_dataflow_graph(base_ckpt, tail, cand_idx::Vector{Int}, at_target::Snapshot)
    n = length(cand_idx)
    A = falses(n, n)
    base = continue_from(base_ckpt, Int.(tail)).ram
    for (i, ci) in enumerate(cand_idx)
        bval = Int(at_target.ram[ci + 1])
        changed = falses(n)
        for v in (0, (bval + 17) & 0xFF, (bval + 37) & 0xFF)
            v == bval && continue
            env = deepcopy(base_ckpt)
            intervene_ram!(env, ci, v)
            for a in tail; env_step!(env, Int(a)); end
            r = snapshot(env, length(tail)).ram
            for (j, cj) in enumerate(cand_idx)
                r[cj + 1] != base[cj + 1] && (changed[j] = true)
            end
        end
        A[i, :] = changed
    end
    return A
end

"""Transitive-closure ancestor set of node `out` (1-based) in adjacency `A`
(A[i,j] = edge i->j): every node with a directed path to `out`. Excludes `out`."""
function ancestors(A::BitMatrix, out::Int)
    n = size(A, 1)
    anc = falses(n)
    frontier = [out]
    while !isempty(frontier)
        v = pop!(frontier)
        for i in 1:n
            if A[i, v] && !anc[i] && i != out
                anc[i] = true
                push!(frontier, i)
            end
        end
    end
    return findall(anc)
end

# ============================================================================
# The scrub operation (resample ablation). Causal scrubbing replaces every
# activation the hypothesis claims is OUTSIDE the circuit by a value the activation
# takes under a DIFFERENT, consistent condition (a resample), never a zero/mean
# ablation. On the VCS the principled resample distribution for a cell is the
# oracle's interchange value-set {0, base+17, base+37} — the SAME do-set A1 used to
# establish the ground-truth edges (so a cell that genuinely drives the output WILL
# move it under at least one resample value, exactly as the oracle measured). A
# hypothesis PASSES only if the behaviour is preserved under ALL such resamplings of
# the out-of-circuit cells (Chan et al.: faithfulness must hold for every consistent
# resample), so a wrong circuit that calls a live driver "irrelevant" is exposed.
# ============================================================================
"""The resample distribution for candidate cell `k` (1-based): the oracle do-set
{0, base+17, base+37} minus the base value (the values it takes under the
interchange conditions A1 used to establish the edges). Never empty (≥2 values)."""
function resample_values(at_target::Snapshot, cand_idx::Vector{Int}, k::Int)
    b = Int(at_target.ram[cand_idx[k] + 1])
    vals = Int[]
    for v in (0, (b + 17) & 0xFF, (b + 37) & 0xFF)
        v == b || push!(vals, v)
    end
    return isempty(vals) ? [(b + 1) & 0xFF] : vals
end

"""Scrub with a SPECIFIC resample assignment: for every candidate cell NOT in
`circuit` (a set of 1-based ordinals), write `assign[k]` into a deepcopy of
`base_ckpt`, run `tail`, snapshot. Circuit cells keep their BASE value (untouched).
`assign` is a Dict k=>value built from each cell's resample distribution."""
function scrub_assign(base_ckpt, tail, cand_idx::Vector{Int}, circuit::Set{Int},
                      assign::Dict{Int,Int})
    env = deepcopy(base_ckpt)
    for (k, ci) in enumerate(cand_idx)
        k in circuit && continue                 # in-circuit → keep base value
        intervene_ram!(env, ci, get(assign, k, Int(env.console.bus.ram[ci + 1])))
    end
    for a in tail; env_step!(env, Int(a)); end
    return snapshot(env, length(tail))
end

# ============================================================================
# Per-game result
# ============================================================================
struct CSResult
    game::String
    target_frame::Int
    horizon::Int
    n_candidates::Int
    cand_idx::Vector{Int}
    cand_concepts::Vector{String}
    n_outputs::Int
    output_ordinals::Vector{Int}        # 1-based ordinals (sinks of the true graph)
    true_circuit::Vector{Int}           # union of {outputs} ∪ ancestors (1-based)
    wrong_circuit::Vector{Int}          # scrambled, same size, drops a true ancestor
    n_true_trials::Int                  # resample trials scrubbed (TRUE hypothesis)
    n_wrong_trials::Int                 # resample trials scrubbed (WRONG hypothesis)
    bit_exact::Bool
    # the headline pass/fail metrics (mean over outputs × resample trials)
    true_preserved::Float64             # scrubbing-preserved perf, TRUE hypothesis
    wrong_preserved::Float64            # scrubbing-preserved perf, WRONG hypothesis
    true_pass::Bool
    wrong_pass::Bool
    discriminates::Bool                 # true_preserved > wrong_preserved
    # graded behaviour similarity (whole observable state) for context
    true_behaviour_sim::Float64         # frac of all candidate cells preserved (TRUE)
    wrong_behaviour_sim::Float64        # frac of all candidate cells preserved (WRONG)
    true_screen_sim::Float64            # frac of screen pixels preserved (TRUE)
    wrong_screen_sim::Float64           # frac of screen pixels preserved (WRONG)
    n_true_edges::Int
    # the per-output preservation matrices (outputs × sources), for the npz
    true_preserve_mat::Matrix{Float64}
    wrong_preserve_mat::Matrix{Float64}
    # SHARED-TESTBED provenance (redesign); all NaN/-1/"" in the legacy NOOP path.
    state_kind::String               # "seeded_random_action_gameplay" | "noop"
    st_seed::Int
    st_prefix::Int
    cause_density::Int               # #causes above the floor at the shared output
    cause_density_accepted::Bool     # passed the cause-density gate?
    n_causes::Int
    shared_cell::Tuple{Int,Int}      # the shared screen-buffer output cell
end

"""Pick the BEHAVIOUR output cells: the SINKS of the true graph that HAVE at least
one true ancestor (a non-trivial routine to scrub around). Falls back to the cells
with the most incoming true edges if no pure sink qualifies, then to the last
candidate cell, so every game yields a testable behaviour."""
function pick_outputs(A::BitMatrix)
    n = size(A, 1)
    indeg = [count(A[:, j]) - (A[j, j] ? 1 : 0) for j in 1:n]   # incoming, no self
    outdeg = [count(A[j, :]) - (A[j, j] ? 1 : 0) for j in 1:n]  # outgoing, no self
    # ideal output = has incoming true edges (driven) and is itself a sink (outdeg 0)
    outs = [j for j in 1:n if indeg[j] > 0 && outdeg[j] == 0]
    if isempty(outs)
        outs = [j for j in 1:n if indeg[j] > 0]                 # any driven cell
    end
    if isempty(outs)
        # degenerate: no edges at all → use the last candidate (behaviour = its value)
        outs = [n]
    end
    return sort(outs)
end

function run_game(; game, target_frame = 30, horizon = 30, n_sources = 3,
                  roms_dir = nothing, verbose = true)
    # SHARED-TESTBED (redesign): replace the all-NOOP boot/attract tape with a
    # seeded random-action GAMEPLAY state at f*=ST_PREFIX, gated by the oracle
    # cause-density gate. The checkpoint + action stream come from the shared
    # substrate so every method sits on the SAME state (P1, P4). The scrubbing
    # algorithm (hypotheses, resample ablation, scoring) is unchanged; only the
    # BASE state/action-stream/checkpoint it sits on moves.
    st = nothing
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
        target_frame = st.prefix; horizon = st.horizon
        total = st.total
        base_actions = st.actions
        bit_exact = true
        base_ckpt = st.checkpoint
        tail = base_actions[target_frame + 1 : total]
        clean = st.base                                   # the un-scrubbed behaviour
        at_target = st.at_target                          # state AT frame t
        verbose && println("[$game] SHARED gameplay state: cause_density=$(st.cause_density)/" *
            "$(length(st.causes)) accepted=$(st.accepted) cell=$(st.cell)")
        return _run_game_body(; game = game, target_frame = target_frame, horizon = horizon,
            total = total, base_actions = base_actions, base_ckpt = base_ckpt, tail = tail,
            clean = clean, at_target = at_target, bit_exact = bit_exact, st = st, verbose = verbose)
    end

    total = target_frame + horizon
    base_actions = fill(ACT_NOOP, total)

    # ---- 1) bit-exact guarantee (two fresh boots+replays, RAM AND screen) ----
    verbose && println("[$game] asserting bit-exactness (2 fresh replays to f$total)...")
    assert_bit_exact(base_actions, total; game = game, roms_dir = roms_dir)
    verbose && println("[$game] bit-exact re-run: PASS")
    bit_exact = true

    # ---- 2) BASE checkpoint at the target frame + clean continuation ----------
    base_ckpt = boot_replay(base_actions, target_frame; game = game, roms_dir = roms_dir)
    tail = base_actions[target_frame + 1 : total]
    clean = continue_from(base_ckpt, Int.(tail))          # the un-scrubbed behaviour
    at_target = continue_from(base_ckpt, Int[])           # state AT frame t
    return _run_game_body(; game = game, target_frame = target_frame, horizon = horizon,
        total = total, base_actions = base_actions, base_ckpt = base_ckpt, tail = tail,
        clean = clean, at_target = at_target, bit_exact = bit_exact, st = nothing, verbose = verbose)
end

"""The per-game body, shared by the legacy NOOP path and the SHARED gameplay-state
path (the ONLY difference is which BASE state/action-stream/checkpoint the scrubbing
algorithm sits on — the algorithm itself is unchanged). `st` carries the
shared-testbed provenance for the record, or `nothing` in the legacy path."""
function _run_game_body(; game, target_frame, horizon, total, base_actions, base_ckpt,
                        tail, clean, at_target, bit_exact, st, verbose)

    # ---- 3) candidate cells (the units the hypothesis is over) ----------------
    cand = candidates_path_for(game)
    cand_ic = candidate_ram_indices(cand)                 # [(idx, concept), ...]
    cand_idx = [idx for (idx, _) in cand_ic]
    cand_concepts = [c for (_, c) in cand_ic]
    n = length(cand_idx)
    n >= 2 || error("[$game] need >=2 candidate cells for a circuit; got $n")

    # ---- 4) the GROUND-TRUTH routine: the A1 true data-flow graph --------------
    verbose && println("[$game] computing the TRUE data-flow graph over $n candidate cells...")
    A = true_dataflow_graph(base_ckpt, Int.(tail), cand_idx, at_target)
    n_true_edges = count(A) - count(A[k, k] for k in 1:n)   # off-diagonal

    # ---- 5) the behaviour outputs + the TRUE / WRONG hypotheses ---------------
    outs = pick_outputs(A)
    # TRUE circuit = {outputs} ∪ all ancestors of any output (transitive closure).
    true_circuit = Set{Int}(outs)
    for o in outs
        for a in ancestors(A, o); push!(true_circuit, a); end
    end

    # The resample distribution per cell = the oracle do-set values (minus base).
    resamp = Dict{Int,Vector{Int}}(k => resample_values(at_target, cand_idx, k) for k in 1:n)
    clean_out = Int[Int(clean.ram[cand_idx[o] + 1]) for o in outs]

    # --- the scrub-and-read primitive: scrub everything outside `circuit` (all out-
    #     of-circuit cells at their FIRST resample value), but OVERRIDE cell `vary`
    #     to `val` (its specific resample value being tested); read the outputs.
    function scrub_outputs(circuit::Set{Int}; vary::Int = 0, val::Int = 0)
        assign = Dict{Int,Int}()
        for k in 1:n
            k in circuit && continue
            assign[k] = resamp[k][1]                 # baseline resample for the rest
        end
        vary != 0 && (assign[vary] = val)            # the value under test
        sc = scrub_assign(base_ckpt, Int.(tail), cand_idx, circuit, assign)
        return Int[Int(sc.ram[cand_idx[o] + 1]) for o in outs], sc
    end

    # --- scrubbing-preserved performance: a hypothesis PASSES iff the behaviour
    #     (every output) is preserved under EVERY resampling of the out-of-circuit
    #     cells (Chan et al.: faithfulness must hold over the whole resample
    #     distribution). We enumerate, for each out-of-circuit cell, every resample
    #     value (others held at their baseline resample) — the trial set that exposes
    #     any single live driver the hypothesis wrongly called irrelevant — plus the
    #     all-at-once combined resample. preserved = fraction of (trial × output)
    #     pairs whose output matches clean.
    function preserve_matrix(circuit::Set{Int})
        outside = [k for k in 1:n if !(k in circuit)]
        trials = Tuple{Int,Int}[]                    # (vary cell, value); (0,0)=combined
        for k in outside, v in resamp[k]
            push!(trials, (k, v))
        end
        push!(trials, (0, 0))                        # all-at-once baseline resample
        isempty(outside) && (trials = [(0, 0)])      # nothing to scrub → only combined
        m = zeros(Float64, length(outs), length(trials))
        for (ti, (vk, vv)) in enumerate(trials)
            o2, _ = scrub_outputs(circuit; vary = vk, val = vv)
            for oi in 1:length(outs)
                m[oi, ti] = (o2[oi] == clean_out[oi]) ? 1.0 : 0.0
            end
        end
        return m
    end

    # WRONG circuit = the TRUE circuit with a genuine, LIVE ancestor DROPPED (so the
    # scrub now resample-ablates a cell that DEMONSTRABLY drives an output) and a
    # non-circuit cell added to keep the size honest. "Live" = the oracle confirms a
    # resample value of that ancestor changes some output. This is the negative
    # control: a wrong hypothesis that calls a real driver irrelevant must FAIL.
    droppable = [a for a in true_circuit if !(a in outs)]   # never drop an output
    outside_t = [k for k in 1:n if !(k in true_circuit)]
    # a droppable ancestor is "live" iff some resample value of it (with all other
    # cells at their clean/base value) changes an output — the oracle's own edge test.
    function is_live(k)
        for v in resamp[k]
            env = deepcopy(base_ckpt)
            intervene_ram!(env, cand_idx[k], v)
            for a in Int.(tail); env_step!(env, a); end
            sc = snapshot(env, length(tail))
            any(Int(sc.ram[cand_idx[o] + 1]) != clean_out[oi] for (oi, o) in enumerate(outs)) && return true
        end
        return false
    end
    live_drop = [a for a in droppable if is_live(a)]
    wrong_circuit = copy(true_circuit)
    if !isempty(live_drop) && !isempty(outside_t)
        drop = minimum(live_drop); add = minimum(outside_t)
        delete!(wrong_circuit, drop); push!(wrong_circuit, add)
    elseif !isempty(droppable) && !isempty(outside_t)
        # no provably-live ancestor (e.g. quiescent in this window) → still drop one
        drop = minimum(droppable); add = minimum(outside_t)
        delete!(wrong_circuit, drop); push!(wrong_circuit, add)
    elseif !isempty(droppable)
        delete!(wrong_circuit, minimum(droppable))    # shrink (no outsider to add)
    else
        # circuit already spans every cell / no ancestors → wrong = only the outputs
        # (every real driver is now scrubbed): a clear FAIL when any edge is live.
        wrong_circuit = Set{Int}(outs)
    end

    # ---- 6) SCRUB + measure scrubbing-preserved performance (both hypotheses) ---
    verbose && println("[$game] scrubbing TRUE hypothesis (circuit=$(sort(collect(true_circuit)))) ...")
    true_mat = preserve_matrix(true_circuit)
    verbose && println("[$game] scrubbing WRONG hypothesis (circuit=$(sort(collect(wrong_circuit)))) ...")
    wrong_mat = preserve_matrix(wrong_circuit)

    # graded behaviour similarity (whole observable state) under the all-at-once scrub.
    function behaviour_sim(circuit::Set{Int})
        _, sc = scrub_outputs(circuit; vary = 0, val = 0)
        cell_sim = mean(Float64[sc.ram[cand_idx[k] + 1] == clean.ram[cand_idx[k] + 1]
                                for k in 1:n])
        screen_sim = mean(Float64.(vec(sc.screen) .== vec(clean.screen)))
        return cell_sim, screen_sim
    end
    true_cell_sim, true_scr_sim = behaviour_sim(true_circuit)
    wrong_cell_sim, wrong_scr_sim = behaviour_sim(wrong_circuit)

    true_preserved  = isempty(true_mat)  ? 1.0 : mean(vec(true_mat))
    wrong_preserved = isempty(wrong_mat) ? 1.0 : mean(vec(wrong_mat))
    true_pass  = (true_preserved == 1.0)
    wrong_pass = (wrong_preserved == 1.0)
    discriminates = (true_preserved > wrong_preserved)

    verbose && println("[$game] TRUE preserved=$(round(true_preserved,digits=3)) (pass=$true_pass) " *
                       "WRONG preserved=$(round(wrong_preserved,digits=3)) (pass=$wrong_pass) " *
                       "discriminates=$discriminates  true_edges=$n_true_edges " *
                       "outputs=$(length(outs)) resample_vals/cell~$(maximum(length.(values(resamp))))")

    return CSResult(game, target_frame, horizon, n, cand_idx, cand_concepts,
                    length(outs), outs, sort(collect(true_circuit)),
                    sort(collect(wrong_circuit)), size(true_mat, 2), size(wrong_mat, 2),
                    bit_exact, true_preserved, wrong_preserved, true_pass, wrong_pass,
                    discriminates, true_cell_sim, wrong_cell_sim, true_scr_sim,
                    wrong_scr_sim, n_true_edges, true_mat, wrong_mat,
                    st === nothing ? "noop" : "seeded_random_action_gameplay",
                    st === nothing ? -1 : st.seed,
                    st === nothing ? -1 : st.prefix,
                    st === nothing ? -1 : st.cause_density,
                    st === nothing ? false : st.accepted,
                    st === nothing ? 0 : length(st.causes),
                    st === nothing ? (-1, -1) : st.cell)
end

# tiny mean (no Statistics dep needed for these small vectors)
mean(v) = isempty(v) ? 0.0 : sum(v) / length(v)

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

function _budget(r::CSResult)
    return Dict{String,Any}(
        "target_frame" => r.target_frame,
        "horizon" => r.horizon,
        "n_candidate_cells" => r.n_candidates,
        "true_graph_doses_per_cell" => 3,        # {0, base+17, base+37} (A1 do-set)
        "n_outputs_scrubbed" => r.n_outputs,
        "resample_distribution" => "oracle do-set {0, base+17, base+37}\\base (the " *
            "values A1 used to establish the ground-truth edges; ≤2 values per cell)",
        "scrub_trials_per_hypothesis" => "Σ_{out-of-circuit cell} |resample(cell)| + 1 " *
            "(per-cell single resamples + 1 all-at-once); one full-ROM re-run/trial",
        "action_trace" => "all-NOOP base trace (deterministic, Paper-1 bit-exact)",
        "rationale" => "paper-reasonable causal-scrubbing budget: the TRUE circuit is " *
            "the A1 ground-truth data-flow (exhaustive do-set, full horizon, cached " *
            "boot+replay checkpoint → incremental re-runs). The scrub resample-ablates " *
            "every out-of-circuit cell with its oracle resample distribution (the do-" *
            "values that established the edges, so a live driver WILL move the output) " *
            "and re-runs the real ROM for `horizon` frames; a hypothesis passes only " *
            "if behaviour is preserved over the WHOLE resample distribution.",
    )
end

function write_game_result(r::CSResult; out_dir = OUT_DIR, where_str = "local")
    isdir(out_dir) || mkpath(out_dir)
    stem = "causal_scrubbing_$(r.game)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path  = joinpath(out_dir, stem * ".npz")

    out_names = [r.cand_concepts[o] == "" ? "ram@$(r.cand_idx[o])" :
                 "$(r.cand_concepts[o])@$(r.cand_idx[o])" for o in r.output_ordinals]

    rec = Dict{String,Any}(
        "paper" => "P2",
        "phase" => "phaseC_mechanistic",
        "method" => "causal_scrubbing",
        "game" => r.game,
        "state" => r.state_kind == "noop" ? "f$(r.target_frame)+$(r.horizon)" :
                   "gameplay(seed=$(r.st_seed),prefix=$(r.st_prefix))+$(r.horizon)",
        "target_output" => "behaviour = the true-graph sink cell(s): " * join(out_names, ", "),
        # headline §R scalar: scrubbing-preserved performance of the TRUE hypothesis
        # (1.0 = the ground-truth circuit's hypothesis PASSES — behaviour preserved).
        "metric_name" => "scrubbing_preserved_performance_true",
        "value" => r.true_preserved,
        "stderr" => nothing,
        "ci" => nothing,
        "n" => r.n_outputs * r.n_true_trials,
        "seed" => 0,
        "where" => where_str,
        "commit" => _git_commit(),
        "oracle_ref" => "A1_connectomics true data-flow @$(r.game) (P2-E3-1) — " *
                        "exhaustive pairwise interventions = the ground-truth circuit",
        "timestamp" => string(round(Int, time())),
        "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            # F∧S∧M triad (paper sec:triad). F = scrubbing-preserved performance of
            # the TRUE hypothesis (leaderboard-oriented value, UNCHANGED). S is the
            # SAME held-out sufficiency test the sibling ACDC uses: the hypothesised
            # circuit alone (non-circuit parents resample-scrubbed) reproduces the
            # clean behaviour under held-out resample trials ⇒ S = true_preserved. M =
            # |U*|/|U_named|: the minimal true circuit vs the whole candidate state
            # (the named circuit's parsimony against recording every cell).
            "triad" => let ncirc = length(r.true_circuit), ncand = r.n_candidates
                Dict{String,Any}(
                    "F" => jnum_or_null(clamp(r.true_preserved, 0.0, 1.0)),
                    "S" => jnum_or_null(clamp(r.true_preserved, 0.0, 1.0)),
                    "S_note" => "held-out scrubbing sufficiency: the circuit alone " *
                        "(non-circuit parents resample-scrubbed) reproduces the clean " *
                        "readouts over held-out resample trials (fraction preserved, bit-exact re-run)",
                    "M" => jnum_or_null(ncand == 0 ? nothing : min(1.0, ncirc / ncand)),
                    "M_note" => "|U*|=$ncirc minimal true-circuit cells / |U_named|=$ncand " *
                        "candidate cells (circuit parsimony vs recording the whole candidate state)",
                    "M_true_minimal_size" => ncirc, "M_named_size" => ncand,
                    "definition" => "F∧S∧M triad (03_methods.tex sec:triad): F = scrubbing-" *
                        "preserved performance (unchanged); S = held-out scrubbing sufficiency; " *
                        "M = |true circuit|/|candidate cells|.")
            end,
            "substrate" => "jutari (Julia, HARD) — real-ROM bit-exact path; every " *
                           "scrub is a genuine intervention + re-run on the true ROM.",
            "bit_exact_rerun" => r.bit_exact,
            # the headline causal-scrubbing pass/fail discrimination
            "scrubbing_preserved_performance_true" => r.true_preserved,
            "scrubbing_preserved_performance_wrong" => r.wrong_preserved,
            "true_hypothesis_pass" => r.true_pass,
            "wrong_hypothesis_pass" => r.wrong_pass,
            "discriminates_true_from_wrong" => r.discriminates,
            # graded behaviour similarity for context (whole observable state)
            "true_behaviour_cell_similarity" => r.true_behaviour_sim,
            "wrong_behaviour_cell_similarity" => r.wrong_behaviour_sim,
            "true_behaviour_screen_similarity" => r.true_screen_sim,
            "wrong_behaviour_screen_similarity" => r.wrong_screen_sim,
            # the circuits + outputs (1-based ordinals into candidate_ram_indices)
            "n_candidate_cells" => r.n_candidates,
            "candidate_ram_indices" => r.cand_idx,
            "candidate_concepts" => r.cand_concepts,
            "n_true_dataflow_edges" => r.n_true_edges,
            "output_ordinals_1based" => r.output_ordinals,
            "output_ram_indices" => [r.cand_idx[o] for o in r.output_ordinals],
            "true_circuit_ordinals_1based" => r.true_circuit,
            "true_circuit_ram_indices" => [r.cand_idx[k] for k in r.true_circuit],
            "wrong_circuit_ordinals_1based" => r.wrong_circuit,
            "wrong_circuit_ram_indices" => [r.cand_idx[k] for k in r.wrong_circuit],
            "n_scrub_trials_true" => r.n_true_trials,
            "n_scrub_trials_wrong" => r.n_wrong_trials,
            "budget" => _budget(r),
            "testbed" => Dict{String,Any}(
                "state_kind" => r.state_kind,
                "seed" => r.st_seed, "prefix" => r.st_prefix, "horizon" => r.horizon,
                "shared_output" => "screen_region(n_changed_px)@r$(r.shared_cell[1])c$(r.shared_cell[2])",
                "cause_density_above_floor" => r.cause_density,
                "cause_density_floor" => ST_FLOOR, "cause_density_gate_k" => ST_GATE_K,
                "cause_density_accepted" => r.cause_density_accepted, "n_causes" => r.n_causes,
                "note" => "P2 redesign: methods sit on a seeded random-action GAMEPLAY " *
                    "state (not the boot/attract NOOP tape), gated by the oracle " *
                    "cause-density gate (accept iff #causes above the floor >= k). The " *
                    "causal-scrubbing algorithm is unchanged; only the BASE state moves."),
            "note" =>
                "Causal scrubbing (Chan et al. 2022): given a HYPOTHESIZED CIRCUIT, " *
                "RESAMPLE-ABLATE every candidate cell OUTSIDE it (write a value it takes " *
                "under a different, consistent condition — here the oracle do-set " *
                "{0,base+17,base+37} that established the A1 edges) and re-run the REAL " *
                "ROM; the hypothesis PASSES iff the behaviour (the true-graph sink " *
                "cells) is PRESERVED over the WHOLE resample distribution. TRUE " *
                "hypothesis = the output's causal ancestors in the A1 ground-truth " *
                "data-flow ({outputs} ∪ transitive-closure ancestors) → it only scrubs " *
                "cells the oracle says are NOT ancestors, so behaviour is preserved " *
                "(PASS, preserved=1.0). WRONG hypothesis = the same-size circuit with a " *
                "genuine (live) ancestor DROPPED, so the scrub resample-ablates a cell " *
                "that DEMONSTRABLY drives the output → behaviour breaks at the live " *
                "resample value (FAIL, preserved<1.0). The gap (discriminates) is the " *
                "§6/§7 result: with ground-truth hypotheses to scrub, causal scrubbing " *
                "Succeeds on the VCS. scrubbing_preserved_performance = the mean exact- " *
                "match of the output cell over outputs × resample trials. T1 scoring " *
                "(no T3): the circuit is the program's known data-flow, not a naming.",
            "scales_to_cluster_via" =>
                "tools/cluster/xai_array_jl.sbatch (E0-3): --shard i --nshards n " *
                "--shard-kind game runs game i of the core set per array task; forward " *
                "bit-exact to this HARD map (SOFT-STE) for batched/larger resample pools.",
        ),
    )
    open(json_path, "w") do io
        write(io, _j(rec) * "\n")
    end

    write_npz(npz_path, Dict(
        "true_preserve_matrix"  => r.true_preserve_mat,    # (outputs × resample trials)
        "wrong_preserve_matrix" => r.wrong_preserve_mat,   # (outputs × resample trials)
        "output_ram_indices"    => Float64.([r.cand_idx[o] for o in r.output_ordinals]),
        "true_circuit_ram_indices"  => Float64.([r.cand_idx[k] for k in r.true_circuit]),
        "wrong_circuit_ram_indices" => Float64.([r.cand_idx[k] for k in r.wrong_circuit]),
        "candidate_ram_indices"     => Float64.(r.cand_idx),
    ))
    return json_path, npz_path
end

function write_summary(results::Vector{CSResult}; out_dir = OUT_DIR, where_str = "local")
    isdir(out_dir) || mkpath(out_dir)
    path = joinpath(out_dir, "causal_scrubbing_core_summary.json")
    per_game = Dict{String,Any}[]
    all_discriminate = true
    for r in results
        all_discriminate &= r.discriminates
        push!(per_game, Dict{String,Any}(
            "game" => r.game,
            "n_candidate_cells" => r.n_candidates,
            "n_true_dataflow_edges" => r.n_true_edges,
            "n_outputs" => r.n_outputs,
            "n_scrub_trials_true" => r.n_true_trials,
            "n_scrub_trials_wrong" => r.n_wrong_trials,
            "scrubbing_preserved_performance_true" => r.true_preserved,
            "scrubbing_preserved_performance_wrong" => r.wrong_preserved,
            "true_hypothesis_pass" => r.true_pass,
            "wrong_hypothesis_pass" => r.wrong_pass,
            "discriminates_true_from_wrong" => r.discriminates,
            "true_behaviour_screen_similarity" => r.true_screen_sim,
            "wrong_behaviour_screen_similarity" => r.wrong_screen_sim,
        ))
    end
    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseC_mechanistic",
        "method" => "causal_scrubbing", "scope" => "core (6 games)",
        "commit" => _git_commit(), "timestamp" => string(round(Int, time())),
        "where" => where_str,
        "all_games_discriminate_true_from_wrong" => all_discriminate,
        "mean_scrubbing_preserved_performance_true" =>
            mean([r.true_preserved for r in results]),
        "mean_scrubbing_preserved_performance_wrong" =>
            mean([r.wrong_preserved for r in results]),
        "n_true_hypothesis_pass" => count(r -> r.true_pass, results),
        "n_wrong_hypothesis_pass" => count(r -> r.wrong_pass, results),
        "per_game" => per_game,
        "note" => "Phase-C causal scrubbing over the 6 core games: a TRUE hypothesis " *
                  "(the A1 ground-truth data-flow circuit) PASSES (behaviour preserved " *
                  "under resample-ablation of everything outside it) while a WRONG " *
                  "(scrambled) hypothesis FAILS. discriminates = preserved_true > " *
                  "preserved_wrong (experiment_design.md §6/§7: causal scrubbing " *
                  "Succeeds — we have ground-truth hypotheses to scrub).",
    )
    open(path, "w") do io
        write(io, _j(rec) * "\n")
    end
    return path
end

# ============================================================================
# Self-check (DoD: the small test) — positive control on Pong:
#   1. the bit-exact re-run holds (precondition for trusting any Δ);
#   2. the TRUE hypothesis PASSES (scrubbing_preserved_performance_true == 1.0) —
#      the ground-truth circuit's irrelevant cells are genuinely irrelevant;
#   3. the WRONG hypothesis is WORSE (preserved_wrong < preserved_true) — the metric
#      DISCRIMINATES a true circuit from a scrambled one (no false pass);
#   4. there is a non-trivial routine to scrub (>=1 true data-flow edge), so the
#      discrimination is not vacuous.
# Exits nonzero (via `error`) on any failure.
# ============================================================================
function self_check(; game = "pong", target_frame = 30, horizon = 30, n_sources = 3,
                    roms_dir = nothing)
    println("[self-check] running causal scrubbing on $game (positive control)...")
    r = run_game(; game = game, target_frame = target_frame, horizon = horizon,
                 n_sources = n_sources, roms_dir = roms_dir, verbose = false)
    r.bit_exact || error("self-check FAIL: bit-exact re-run did not hold")
    r.n_true_edges >= 1 ||
        error("self-check FAIL: no true data-flow edges on $game — scrubbing is vacuous")
    r.true_preserved == 1.0 ||
        error("self-check FAIL: TRUE hypothesis preserved=$(r.true_preserved) (expected 1.0 PASS)")
    r.wrong_preserved < r.true_preserved ||
        error("self-check FAIL: WRONG preserved ($(r.wrong_preserved)) not < TRUE " *
              "($(r.true_preserved)) — metric does not discriminate")
    println("[self-check] PASS — bit-exact ✓, true_edges=$(r.n_true_edges) ✓, " *
            "TRUE preserved=1.0 (PASS) ✓, WRONG preserved=$(round(r.wrong_preserved,digits=3)) " *
            "< 1.0 (FAIL) ✓ — causal scrubbing discriminates the true routine.")
    return true
end

# ============================================================================
# cluster-shardable game selection (mirrors A2_lesions.shard_games)
# ============================================================================
function shard_games(games::Vector{String}, shard::Int, nshards::Int, shard_kind::String)
    if shard_kind == "game" && nshards > 1
        idx = shard + 1
        (1 <= idx <= length(games)) || return String[]
        return [games[idx]]
    end
    if nshards > 1
        return [games[k] for k in 1:length(games) if (k - 1) % nshards == shard]
    end
    return games
end

# ============================================================================
# CLI — single-game / --games / cluster-shardable (--shard --nshards --shard-kind).
# ============================================================================
function main(args = ARGS)
    games = copy(CORE_GAMES)
    target_frame = 30; horizon = 30; n_sources = 3
    out_dir = OUT_DIR; roms_dir = nothing; where_str = "local"
    shard = 0; nshards = 1; shard_kind = "game"
    do_self_check = false
    i = 1
    while i <= length(args)
        a = args[i]
        if     a == "--games"
            games = xai_resolve_games(args[i+1], CORE_GAMES); i += 2
        elseif a == "--game";         games = [args[i+1]]; i += 2
        elseif a == "--target-frame"; target_frame = parse(Int, args[i+1]); i += 2
        elseif a == "--horizon";      horizon = parse(Int, args[i+1]); i += 2
        elseif a == "--n-sources";    n_sources = parse(Int, args[i+1]); i += 2
        elseif a == "--out-dir";      out_dir = args[i+1]; i += 2
        elseif a == "--roms-dir";     roms_dir = args[i+1]; i += 2
        elseif a == "--where";        where_str = args[i+1]; i += 2
        elseif a == "--shard";        shard = parse(Int, args[i+1]); i += 2
        elseif a == "--nshards";      nshards = parse(Int, args[i+1]); i += 2
        elseif a == "--shard-kind";   shard_kind = args[i+1]; i += 2
        elseif a == "--selftest" || a == "--self-check"; do_self_check = true; i += 1
        else; i += 1
        end
    end

    if do_self_check
        self_check(; target_frame = target_frame, horizon = horizon,
                   n_sources = n_sources, roms_dir = roms_dir)
        return nothing
    end

    sel = shard_games(games, shard, nshards, shard_kind)
    println("[cs] causal scrubbing — games=$(join(sel, ",")) " *
            "target_frame=$target_frame horizon=$horizon " *
            "resample=oracle-do-set/cell " *
            "(shard $shard/$nshards kind=$shard_kind) → out=$out_dir where=$where_str")
    if isempty(sel)
        println("[cs] shard $shard/$nshards (kind=$shard_kind) selected NO games — nothing to do.")
        return nothing
    end

    results = CSResult[]; blockers = Tuple{String,String}[]
    for g in sel
        println("\n========== $g ==========")
        try
            r = run_game(; game = g, target_frame = target_frame, horizon = horizon,
                         n_sources = n_sources, roms_dir = roms_dir, verbose = true)
            jp, np = write_game_result(r; out_dir = out_dir, where_str = where_str)
            println("[$g] TRUE preserved=$(round(r.true_preserved,digits=3)) (pass=$(r.true_pass)) " *
                    "WRONG preserved=$(round(r.wrong_preserved,digits=3)) (pass=$(r.wrong_pass)) " *
                    "discriminates=$(r.discriminates)")
            println("[$g] wrote $jp")
            println("[$g] arrays  $np")
            push!(results, r)
        catch err
            msg = first(split(sprint(showerror, err), '\n'))
            println("[cs] !! game $g FAILED (scoring the rest, not fabricating): $msg")
            push!(blockers, (g, msg))
        end
    end

    # combined record only for an unsharded local run that covered the set in one
    # process; a sharded array task writes only its per-game record (SM merges).
    if nshards <= 1 && length(results) > 1
        sp = write_summary(results; out_dir = out_dir, where_str = where_str)
        println("\n[cs] core summary -> $sp")
    end

    println("\n[cs] ==== headline (causal scrubbing: hypothesis pass/fail vs true routine) ====")
    for r in results
        println("    $(rpad(r.game, 16)) true_edges=$(rpad(r.n_true_edges,3)) " *
                "TRUE_preserved=$(round(r.true_preserved,digits=3)) (pass=$(r.true_pass)) " *
                "WRONG_preserved=$(round(r.wrong_preserved,digits=3)) (pass=$(r.wrong_pass)) " *
                "discriminates=$(r.discriminates)")
    end
    if !isempty(blockers)
        println("[cs] blockers:")
        for (g, m) in blockers; println("    $g: $m"); end
    end
    return results
end

end # module

# run as a script (not when `include`d by a test)
if abspath(PROGRAM_FILE) == @__FILE__
    CausalScrubbing.main()
end
