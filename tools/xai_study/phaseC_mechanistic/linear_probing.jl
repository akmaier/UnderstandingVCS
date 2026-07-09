# linear_probing.jl — Phase-C mechanistic interpretability (P2-E5-9), JULIA path
# (the jutari real-ROM substrate; jaxtari eager is ~205× slower — SCRUM §7).
#
# METHOD: LINEAR PROBING + CONTROL TASKS (Alain & Bengio 2017; Hewitt & Liang
# 2019) over the recorded VCS state trajectory, scored against the EXACT
# intervention oracle (experiment_design.md §6 row "Linear probing + control
# tasks"; §7 prediction: **Partial / misleading — present ≠ used, the
# tuning-curve trap**).
#
# ---------------------------------------------------------------------------
# WHAT A LINEAR PROBE IS HERE (experiment_design.md §6, §2):
#   the VCS state trajectory is the "activations" — a (T, 128) tape of the RIOT
#   RAM over T frames of a real ROM (the recorder common/jutari_record.jl). A
#   known game CONCEPT (ball_x, player_score, lives, ...) lives in a named RAM
#   cell c (the T3 candidate map). A linear probe asks: can a *linear* readout of
#   the REST of the state (all cells EXCEPT c, the holdout) decode the concept's
#   value? If yes, the concept is linearly DECODABLE — i.e. PRESENT in the
#   representation. (We hold out cell c so the probe can't trivially copy the
#   answer; this is the real, non-tautological probing question — is the concept
#   *redundantly* encoded / linearly recoverable from the surrounding state.)
#
# CONTROL TASK (Hewitt & Liang 2019 — the load-bearing methodological control):
#   a high probe accuracy can come from the PROBE's capacity rather than real
#   structure in the representation. The control task fixes this: keep the SAME
#   inputs and the SAME label distribution, but assign each distinct concept VALUE
#   a FIXED RANDOM label (a structure-free target the probe must MEMORISE through
#   the inputs). The same-capacity linear probe is trained on it. SELECTIVITY =
#   probe accuracy − control accuracy. High selectivity ⇒ the probe's accuracy
#   reflects real linear structure in the state, not probe expressivity. (Hewitt
#   & Liang's headline: a probe that scores high on BOTH tasks is "just memorising"
#   — selectivity, not raw accuracy, is the honest decodability signal.)
#
# THE KEY "PRESENT vs USED" GAP (experiment_design.md §6 thesis tie-in, §7
# "present ≠ used — the tuning-curve trap"):
#   probing measures whether a concept is PRESENT (linearly decodable). It does
#   NOT measure whether the program actually USES that cell to compute its output.
#   The EXACT intervention oracle does: the Phase-C interchange/DAS run (das.jl,
#   P2-E5-2) flags cells whose interchange is CLOBBERED within the horizon
#   ("transient_downstream_cells" — the program re-derives the cell next frame, so
#   intervening on it has NO downstream causal effect → present but NOT used). We
#   cross-reference the two: a cell that is DECODABLE (high probe acc + high
#   selectivity) yet flagged NOT-CAUSALLY-USED by the oracle is a
#   DECODABLE-BUT-NOT-CAUSAL cell — a concrete, counted instance of the
#   present-vs-used gap. This is exactly the misleadingness §7 predicts: probing
#   would "find" the concept and a naive reading would claim the model uses it,
#   while the intervention oracle (strictly stronger) shows it does not drive the
#   output over the horizon.
#
# Why this is the ONE method that NEEDS LABELS: every other Phase-C method scores
# against the exact patch / true data-flow (T1/T2). Probing decodes a NAMED
# concept, so it requires the T3 concept↔cell labels — supplied by the per-game
# T3 candidate map (tools/xai_study/t3/out/candidates_<game>.json).
#
# No JuTari/jaxtari/xitari core is modified — pure tooling under tools/xai_study/.
# Reuses the validated foundations on main:
#   * common/jutari_record.jl — the per-frame RAM-tape trajectory recorder.
#   * ground_truth/oracle_intervene.jl — candidate_ram_indices (the labelled
#     concept↔cell map the probe decodes).
#   * the per-game ROM-alias + RomSettings + candidates map (mirrors das.jl /
#     activation_patching.jl / ig_baseline_sweep.jl — NO emulator core touched).
#   * das.jl's out/das_<game>.json "transient_downstream_cells" — the oracle's
#     not-causally-used flag for the present-vs-used cross-reference.
#
# Run (warm shared depot, primary's project):
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseC_mechanistic/linear_probing.jl --games core
# Flags: --games core|<g1,g2,...>  --game <g>
#        --frames N --selftest
#
# Writes (SPEC §R; file_scope linear_probing_* under out/):
#   tools/xai_study/phaseC_mechanistic/out/linear_probing_<game>.{json,npz}
#   tools/xai_study/phaseC_mechanistic/out/linear_probing_core_summary.json

module LinearProbing

using LinearAlgebra
using Random
using Statistics

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_ram

# the labelled concept↔cell candidate map + the dependency-free §R NPZ writer +
# RAM_SIZE (the verified Paper-1 path). record_ram_tape below replicates the common
# recorder's (jutari_record.jl) deterministic per-frame RAM stacking, with the
# per-game ROM alias added so every core game resolves.
include(joinpath(@__DIR__, "..", "ground_truth", "oracle_intervene.jl"))
using .OracleIntervene: candidate_ram_indices
using .OracleIntervene.JutariOracle: write_npz, RAM_SIZE
using JuTari.Diff: soft_ram_peek

# The P2 SHARED TESTBED (xai_paper/xai_2_interpretability/experiment_redesign.md):
# seeded random-action GAMEPLAY state + oracle cause-density gate. Included as a
# fragment (see its header for why not a module) so build_shared_testbed operates
# on OUR own Cause/Snapshot types. Probing is not a gradient method, so the
# sampler-on path does not apply — we consume the shared action STREAM + the
# cause-density GATE, and SEED the recorded RAM trajectory from that gameplay
# stream (feed st.actions for the first prefix+horizon frames, then NOOP-continue
# if more frames are needed) so the probe algorithm is UNCHANGED — only the state
# the trajectory sits on moves. Opt in with XAI_SHARED_TESTBED=1 (default on).
include(joinpath(@__DIR__, "..", "common", "shared_testbed_impl.jl"))
# the shared game-set + ROM-root resolver (XAI_LABELED / xai_resolve_games / xai_rom_roots).
include(joinpath(@__DIR__, "..", "common", "game_sets.jl"))

# shared triad helpers (jnum_or_null for JSON-null on undefined S/M); guarded.
isdefined(@__MODULE__, :TriadSM) ||
    include(joinpath(@__DIR__, "..", "common", "triad_sm.jl"))
using .TriadSM: jnum_or_null, minimality_score

const OUT_DIR = joinpath(@__DIR__, "out")
const CORE_GAMES = ["pong", "breakout", "space_invaders", "seaquest", "ms_pacman", "qbert"]
const _PRIMARY_REPO = get(ENV, "XAI_PRIMARY_REPO", "/Users/maier/Documents/code/UnderstandingVCS")
# shared-testbed switch + params (redesign protocol: prefix=90 gameplay, horizon=15).
const SHARED_TESTBED = get(ENV, "XAI_SHARED_TESTBED", "1") == "1"
const ST_PREFIX  = parse(Int, get(ENV, "XAI_ST_PREFIX", "90"))
const ST_HORIZON = parse(Int, get(ENV, "XAI_ST_HORIZON", "15"))
const ST_SEED    = parse(Int, get(ENV, "XAI_ST_SEED", "0"))
const ST_GATE_K  = parse(Int, get(ENV, "XAI_ST_GATE_K", "4"))
const ST_FLOOR   = parse(Float64, get(ENV, "XAI_ST_FLOOR", "0.5"))

# ============================================================================
# Per-game ROM + RomSettings resolution — mirrors das.jl / activation_patching.jl
# / ig_baseline_sweep.jl EXACTLY (the shared per-game map). The common-recorder's
# JutariOracle.rom_path_for resolves the game name VERBATIM (no alias), so
# ms_pacman → mspacman.bin is unresolved there; we replicate the alias map here
# (NO emulator core / shared-tool file touched — in file_scope). seaquest has no
# registered RomSettings → Generic (boots fine; the screen scoreboard's fallback).
# ============================================================================
const ROM_BASENAME = Dict(
    "pong" => "pong", "breakout" => "breakout",
    "space_invaders" => "space_invaders", "seaquest" => "seaquest",
    "ms_pacman" => "mspacman", "qbert" => "qbert")

function rom_path_for(game::AbstractString)
    g = lowercase(string(game))
    stem = get(ROM_BASENAME, g, g)
    # search xitari/roms + the 54-ROM store tools/rom_sweep/roms (ALE names), trying
    # the mapped stem AND the raw ALE name, so all labeled games resolve uniformly.
    return xai_find_rom(unique([stem, g]), xai_rom_roots(; primary_repo = _PRIMARY_REPO))
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

"""Boot a game (xitari-parity boot via the per-game RomSettings) and record the
per-frame 128-byte RIOT RAM tape over `actions`. Returns a (T, 128) UInt8 matrix.
Mirrors das.jl's load_env + jutari_record's per-frame stacking, but with the
per-game ROM alias so ALL core games (incl. ms_pacman) resolve — keeps the
recorder's deterministic-replay semantics (the verified Paper-1 bit-exact path)."""
function record_ram_tape(game::AbstractString, actions::AbstractVector{<:Integer})
    rom = read(rom_path_for(game))
    env = StellaEnvironment(rom, settings_for(game))
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    T = length(actions)
    tape = Matrix{UInt8}(undef, T, RAM_SIZE)
    for t in 1:T
        env_step!(env, Int(actions[t]))
        tape[t, :] = UInt8.(collect(get_ram(env)))
    end
    return tape
end

# joystick action codes (oracle_intervene.jl: RIGHT=3; LEFT=4). A VARIED action
# stream (so candidate concept cells take multiple values across the trajectory →
# non-degenerate probe labels). For paddle games RIGHT/LEFT move the paddle; for
# joystick games they move the player — both produce real state variation.
const ACT_NOOP  = 0
const ACT_RIGHT = 3
const ACT_LEFT  = 4
const ACT_DOWN  = 5
const ACT_UP    = 2

"""A SEEDED pseudo-random varied action stream of length `n` (drawn from a small
input set). A *deterministic periodic* trace makes the whole RAM tape a single
low-dimensional periodic orbit — every cell becomes a near-perfect linear function
of the frame phase, so a linear probe decodes ANY target (the real concept AND the
random control) and selectivity collapses to ~0. A seeded random stream gives a
richer, less-collinear state distribution so the control task is genuinely hard for
a linear probe while the real structured concept stays decodable — the regime
Hewitt & Liang's selectivity is designed for. Seeded ⇒ the trajectory is still
bit-exact reproducible run-to-run (same seed → same actions → same tape)."""
function varied_actions(n::Integer; seed::Integer = 0)
    rng = MersenneTwister(seed)
    choices = [ACT_NOOP, ACT_RIGHT, ACT_LEFT, ACT_DOWN, ACT_UP, ACT_RIGHT, ACT_LEFT]
    return [choices[rand(rng, 1:length(choices))] for _ in 1:Int(n)]
end

# ============================================================================
# Candidate concept resolution (mirrors das.jl / oracle_intervene candidates path).
# ============================================================================
function candidates_path_for(game::AbstractString)
    rel = joinpath("tools", "xai_study", "t3", "out", "candidates_$(game).json")
    here = normpath(joinpath(@__DIR__, "..", "..", ".."))
    for base in (here, _PRIMARY_REPO)
        p = joinpath(base, rel)
        isfile(p) && return p
    end
    return nothing
end

"""Read the das.jl oracle output for `game` and return the set of 0-based RAM
indices flagged as NOT-CAUSALLY-USED over the interchange horizon
(transient_downstream_cells: the program re-derives the cell within the horizon,
so an interchange on it has no downstream self-effect → present but not used).
Returns an empty set if the das output is absent (cross-reference is then n/a)."""
function oracle_transient_cells(game::AbstractString)
    out = Set{Int}()
    for base in (normpath(joinpath(@__DIR__, "..", "..", "..")), _PRIMARY_REPO)
        p = joinpath(base, "tools", "xai_study", "phaseC_mechanistic", "out", "das_$(game).json")
        isfile(p) || continue
        txt = read(p, String)
        # the das var names embed the cell index as "...@<idx>"; the transient list
        # holds those names. Pull the bracketed array then parse the @<idx> tails.
        m = match(r"\"transient_downstream_cells\"\s*:\s*\[(.*?)\]", txt)
        m === nothing && return out
        for vm in eachmatch(r"@(\d+)", m.captures[1])
            push!(out, parse(Int, vm.captures[1]))
        end
        return out
    end
    return out
end

# ============================================================================
# Quantise a continuous RAM-cell value series into a small class label set, so a
# linear probe (least-squares one-vs-rest) has well-defined accuracy. We bin the
# observed value range of the cell into up to K equal-width bins over its OBSERVED
# range (only non-degenerate cells — >1 distinct value — are probeable).
# ============================================================================
const N_BINS = 4

"""Map a vector of UInt8 cell values to integer class labels by equal-width
binning over the observed [min,max] range, then REMAP the occupied bins to a
contiguous 1:n_classes label set (so `labels` indexes a dense 1:n_classes range —
empty bins are dropped). Returns (labels, n_classes). A cell with a single
observed value is degenerate (n_classes==1) and is skipped."""
function quantise(values::AbstractVector{<:Real}; k::Integer = N_BINS)
    lo, hi = extrema(values)
    if hi == lo
        return (fill(1, length(values)), 1)
    end
    width = (hi - lo) / k
    raw = [min(k, 1 + floor(Int, (v - lo) / width)) for v in values]
    # remap occupied bins → dense 1:n_classes so label values are contiguous
    occ = sort(unique(raw))
    remap = Dict(b => i for (i, b) in enumerate(occ))
    labels = [remap[r] for r in raw]
    return (labels, length(occ))
end

# ============================================================================
# Linear probe: one-vs-rest least-squares (ridge-regularised) classifier on the
# state tape. X = (n_samples, n_features) with a bias column; Y = one-hot of the
# class labels. Solve W = (XᵀX + λI)⁻¹ XᵀY (the closed-form linear probe — Alain &
# Bengio's "linear classifier probe"); predict argmax. Pure LinearAlgebra, no pip.
# ============================================================================
const RIDGE_LAMBDA = 1e-2

"""Fit a least-squares one-vs-rest linear classifier on (Xtr, ytr) and return its
accuracy on (Xte, yte). `ytr`/`yte` are integer labels in 1:nclass. X already
includes a bias column. Ridge term λ stabilises the normal equations (and gives
the probe a *fixed, bounded* capacity, so probe vs control is apples-to-apples)."""
function probe_accuracy(Xtr, ytr, Xte, yte, nclass; lambda = RIDGE_LAMBDA)
    n, d = size(Xtr)
    # one-hot targets
    Ytr = zeros(Float64, n, nclass)
    for i in 1:n
        Ytr[i, ytr[i]] = 1.0
    end
    A = Xtr' * Xtr + lambda * Matrix{Float64}(I, d, d)
    W = A \ (Xtr' * Ytr)               # (d, nclass)
    scores = Xte * W                   # (n_te, nclass)
    preds = [argmax(@view scores[i, :]) for i in 1:size(scores, 1)]
    return count(preds .== yte) / length(yte)
end

# ============================================================================
# Per-cell probing result
# ============================================================================
struct CellProbe
    cell::Int                 # 0-based RAM index
    concept::String
    n_classes::Int            # distinct quantised labels observed
    probe_acc::Float64        # real-task held-out accuracy
    control_acc::Float64      # control-task (random-label) held-out accuracy
    selectivity::Float64      # probe_acc − control_acc
    majority_acc::Float64     # the trivial majority-class baseline (sanity floor)
    decodable::Bool           # probe_acc high AND selectivity high → PRESENT
    not_causally_used::Bool   # oracle (das) flags the cell transient → NOT USED
    decodable_not_causal::Bool# the present-vs-used cell: decodable AND not used
end

struct GameProbeResult
    game::String
    frames::Int
    n_features::Int
    cells::Vector{CellProbe}
    # headlines
    mean_probe_acc::Float64
    mean_control_acc::Float64
    mean_selectivity::Float64
    n_probeable::Int               # cells with >1 class (a real probe task)
    n_decodable::Int               # decodable (present) cells
    n_not_causally_used::Int       # cells the oracle flags transient (present≠used)
    n_decodable_not_causal::Int    # THE present-vs-used count
    oracle_xref_available::Bool    # das_<game>.json found?
    # T3 concept-label provenance: true iff a verified candidates_<game>.json exists.
    # false ⇒ the concept↔cell labels are the generic fallback set (the probe still
    # runs — it decodes each fallback cell — but the concept SEMANTICS are unverified).
    concepts_verified::Bool
    # SHARED-TESTBED provenance (redesign); all "noop"/-1/false in the legacy path.
    state_kind::String             # "seeded_random_action_gameplay" | "varied_action"
    st_seed::Int
    st_prefix::Int
    st_horizon::Int
    cause_density::Int             # #causes above the floor at the shared output
    cause_density_accepted::Bool   # passed the cause-density gate?
    n_causes::Int
    shared_cell::Tuple{Int,Int}    # the shared screen-buffer output cell
end

# decodability thresholds: a concept is "present/decodable" if the linear probe
# beats both the majority floor AND has real linear structure (selectivity).
const DECODE_ACC_MIN = 0.5         # probe must clear chance-ish (4-bin)
const DECODE_SELECTIVITY_MIN = 0.1 # and show real structure over the control

# default 600 frames: with 127 input features the probe needs MANY more train
# samples than features (600×0.7 = 420 ≫ 127) so the linear system is well-
# determined and CANNOT trivially memorise the random control task — the regime
# in which selectivity (probe − control) is meaningful (Hewitt & Liang 2019).
# Stays well inside the Paper-1 conformance horizon (the core games are 64/64
# long-horizon bit-exact; the recorder replays deterministically).
"""Build the P2 SHARED gameplay state + cause-density gate for `game` using the §1
oracle machinery (oracle_intervene.jl). Returns the substrate NamedTuple; probing
uses its `.actions` gameplay stream to SEED the recorded trajectory and threads its
`.cause_density`/`.accepted`/`.cell` gate into the record. The probe algorithm is
unchanged — only the state the trajectory sits on moves."""
function build_linear_probing_shared_state(game::AbstractString; verbose = false)
    O = OracleIntervene; J = OracleIntervene.JutariOracle
    return build_shared_testbed(game;
        settings_for = J.settings_for, rom_path_for = J.rom_path_for,
        candidates_path_for = candidates_path_for,
        build_causes = O.build_pong_causes, candidate_ram_indices = O.candidate_ram_indices,
        continue_from = J.continue_from, snapshot = J.snapshot, env_step = J.env_step!,
        intervene_ram = J.intervene_ram!, boot_replay = J.boot_replay,
        run_intervention = O.run_intervention, soft_ram_peek = soft_ram_peek,
        prefix = ST_PREFIX, horizon = ST_HORIZON, seed = ST_SEED,
        k = ST_GATE_K, floor = ST_FLOOR, verbose = verbose,
        assert_bit_exact = O.assert_bit_exact)
end

function run_game(; game::AbstractString, frames::Integer = 600,
                  train_frac = 0.7, seed::Integer = 0, verbose = true)
    # ---- 0) SHARED-TESTBED (redesign): seed the recorded trajectory from the
    # seeded random-action GAMEPLAY stream (gated by the oracle cause-density gate)
    # instead of the local varied-action stream. We feed st.actions for the first
    # prefix+horizon frames, then NOOP-continue to reach `frames` if the probe needs
    # MORE samples than the gameplay window (the probe needs many samples ≫ features;
    # the tail is a NOOP continuation FROM the gameplay state, still on-distribution
    # and deterministic). The probe algorithm below is unchanged.
    st = nothing
    if SHARED_TESTBED
        st = build_linear_probing_shared_state(game; verbose = verbose)
        verbose && println("[$game] SHARED gameplay state: cause_density=$(st.cause_density)/" *
            "$(length(st.causes)) accepted=$(st.accepted) cell=$(st.cell)")
    end

    # ---- 1) record the state trajectory under the (shared or varied) action stream
    # record_ram_tape mirrors the common recorder's deterministic per-frame RAM
    # stacking (jutari_record), with the per-game ROM alias so ms_pacman et al.
    # resolve (the common recorder resolves names verbatim).
    acts = if st !== nothing
        # gameplay stream (prefix+horizon) then NOOP-continued to `frames`.
        gp = Int.(st.actions)
        Int(frames) <= length(gp) ? gp[1:Int(frames)] :
            vcat(gp, fill(ACT_NOOP, Int(frames) - length(gp)))
    else
        varied_actions(frames; seed = seed)
    end
    verbose && println("[$game] recording $frames-frame RAM trajectory " *
        (st === nothing ? "(seeded varied actions)..." :
         "(gameplay seed=$(st.seed) prefix=$(st.prefix)+$(st.horizon), NOOP-continued)..."))
    tape = Float64.(record_ram_tape(game, acts))    # (T, 128)
    T, ncell = size(tape)
    @assert ncell == RAM_SIZE "expected $(RAM_SIZE)-cell RAM tape, got $ncell"

    # ---- 2) the labelled concept↔cell map (the probe TARGETS) -----------------
    cand = candidates_path_for(game)
    cand_ic = candidate_ram_indices(cand)        # [(idx, concept), ...] de-duplicated
    transient = oracle_transient_cells(game)     # oracle's not-causally-used set
    xref_available = !isempty(transient) ||
        isfile(joinpath(_PRIMARY_REPO, "tools", "xai_study", "phaseC_mechanistic",
                        "out", "das_$(game).json")) ||
        isfile(joinpath(normpath(joinpath(@__DIR__, "..", "..", "..")),
                        "tools", "xai_study", "phaseC_mechanistic", "out",
                        "das_$(game).json"))

    # ---- 3) train/test split over frames (deterministic shuffle) --------------
    rng = MersenneTwister(seed)
    perm = randperm(rng, T)
    ntr = max(2, round(Int, train_frac * T))
    tr_idx = perm[1:ntr]
    te_idx = perm[ntr + 1:end]
    isempty(te_idx) && (te_idx = tr_idx)          # tiny T fallback

    # standardised feature matrix (z-score per column over the TRAIN rows so the
    # probe is scale-invariant; constant columns → zero, harmless). Bias column appended.
    function feature_matrix(holdout_cell::Int)
        keep = [c for c in 1:ncell if c != holdout_cell + 1]   # 0-based → 1-based, drop the target
        Xraw = tape[:, keep]
        mu = vec(sum(@view Xraw[tr_idx, :]; dims = 1) ./ length(tr_idx))
        sd = vec(sqrt.(max.(1e-12, sum((@view(Xraw[tr_idx, :]) .- mu') .^ 2; dims = 1) ./ length(tr_idx))))
        Xz = (Xraw .- mu') ./ sd'
        return hcat(Xz, ones(Float64, T))          # + bias column
    end

    # ---- 4) probe + control per candidate concept cell ------------------------
    cells = CellProbe[]
    rng_ctrl = MersenneTwister(seed + 1)
    for (idx, concept) in cand_ic
        0 <= idx < ncell || continue
        vals = tape[:, idx + 1]
        labels, nclass = quantise(vals)
        if nclass <= 1
            # degenerate (cell constant over the trajectory) — not a probe task.
            push!(cells, CellProbe(idx, string(concept), nclass,
                                   NaN, NaN, NaN, NaN,
                                   false, idx in transient, false))
            continue
        end
        X = feature_matrix(idx)
        Xtr, Xte = X[tr_idx, :], X[te_idx, :]
        ytr, yte = labels[tr_idx], labels[te_idx]

        # majority-class floor (sanity baseline)
        counts = zeros(Int, nclass)
        for y in ytr; counts[y] += 1; end
        maj_class = argmax(counts)
        majority_acc = count(yte .== maj_class) / length(yte)

        # REAL probe accuracy
        pacc = probe_accuracy(Xtr, ytr, Xte, yte, nclass)

        # CONTROL task (Hewitt & Liang 2019): assign each distinct "word type" — here
        # each distinct RAW cell value (the finest type granularity, many types) — a
        # FIXED RANDOM label in 1:nclass, then probe THAT. Same inputs, same label
        # cardinality, but a STRUCTURE-FREE target: the probe must MEMORISE a random
        # lookup over many value-types through the (continuous) state features. A
        # linear probe with bounded capacity cannot fit an arbitrary many-type random
        # map, so control accuracy stays near chance while the real (binned, linearly
        # structured) concept task is decodable → selectivity = probe − control
        # separates real linear structure from probe capacity. (Keying on the raw
        # value, not the 4-bin label, is what makes the control genuinely hard — a
        # control over only ~4 bins is trivially memorisable, masking selectivity.)
        rawvals = vals                                  # the raw cell value series
        uniq = sort(unique(rawvals))
        ctrl_label_for = Dict(u => rand(rng_ctrl, 1:nclass) for u in uniq)
        ctrl_labels = [ctrl_label_for[v] for v in rawvals]
        # ensure the control target is non-degenerate (>=2 classes); if it collapsed,
        # round-robin distinct values across the nclass labels instead.
        if length(unique(ctrl_labels)) <= 1
            ctrl_label_for = Dict(uniq[i] => mod1(i, nclass) for i in 1:length(uniq))
            ctrl_labels = [ctrl_label_for[v] for v in rawvals]
        end
        cacc = probe_accuracy(Xtr, ctrl_labels[tr_idx], Xte, ctrl_labels[te_idx], nclass)

        selectivity = pacc - cacc
        decodable = (pacc >= DECODE_ACC_MIN) && (selectivity >= DECODE_SELECTIVITY_MIN)
        not_used = idx in transient
        push!(cells, CellProbe(idx, string(concept), nclass,
                               pacc, cacc, selectivity, majority_acc,
                               decodable, not_used, decodable && not_used))
        verbose && println("  cell $(rpad(idx, 3)) $(rpad(string(concept), 22)) " *
                           "k=$nclass probe=$(rpad(round(pacc, digits = 3), 5)) " *
                           "ctrl=$(rpad(round(cacc, digits = 3), 5)) " *
                           "sel=$(rpad(round(selectivity, digits = 3), 6)) " *
                           "decodable=$(decodable ? "Y" : "·") " *
                           "not_used=$(not_used ? "Y" : "·")" *
                           (decodable && not_used ? "  <- PRESENT≠USED" : ""))
    end

    probeable = [c for c in cells if c.n_classes > 1]
    np = length(probeable)
    mean_probe = np == 0 ? 0.0 : sum(c.probe_acc for c in probeable) / np
    mean_ctrl  = np == 0 ? 0.0 : sum(c.control_acc for c in probeable) / np
    mean_sel   = np == 0 ? 0.0 : sum(c.selectivity for c in probeable) / np
    n_decodable = count(c -> c.decodable, cells)
    n_not_used = count(c -> c.not_causally_used, cells)
    n_dnc = count(c -> c.decodable_not_causal, cells)

    verbose && println("[$game] probeable=$np/$(length(cells)) " *
                       "mean_probe=$(round(mean_probe, digits = 3)) " *
                       "mean_ctrl=$(round(mean_ctrl, digits = 3)) " *
                       "mean_selectivity=$(round(mean_sel, digits = 3)) " *
                       "decodable=$n_decodable not_used=$n_not_used " *
                       "DECODABLE_NOT_CAUSAL=$n_dnc " *
                       "(oracle_xref=$(xref_available ? "yes" : "no"))")

    return GameProbeResult(game, Int(frames), ncell - 1, cells,
                           mean_probe, mean_ctrl, mean_sel,
                           np, n_decodable, n_not_used, n_dnc, xref_available,
                           cand !== nothing,
                           st === nothing ? "varied_action" : "seeded_random_action_gameplay",
                           st === nothing ? -1 : st.seed,
                           st === nothing ? -1 : st.prefix,
                           st === nothing ? -1 : st.horizon,
                           st === nothing ? -1 : st.cause_density,
                           st === nothing ? false : st.accepted,
                           st === nothing ? -1 : length(st.causes),
                           st === nothing ? (-1, -1) : st.cell)
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

function _cell_dict(c::CellProbe)
    return Dict{String,Any}(
        "cell" => c.cell, "concept" => c.concept, "n_classes" => c.n_classes,
        "probe_acc" => c.probe_acc, "control_acc" => c.control_acc,
        "selectivity" => c.selectivity, "majority_acc" => c.majority_acc,
        "decodable" => c.decodable, "not_causally_used" => c.not_causally_used,
        "decodable_not_causal" => c.decodable_not_causal)
end

function write_game_result(r::GameProbeResult; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    stem = "linear_probing_$(r.game)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path  = joinpath(out_dir, stem * ".npz")

    # §3 (sec:triad) causal-agreement F for the probe: the point-biserial CORRELATION
    # between the probe's per-cell claim (selectivity) and the oracle's per-cell causal
    # indicator (1 = causally used, 0 = transient/not-used). This is exactly what sec:triad
    # defines F to be — the correlation between the attribution and the true causal effects,
    # NOT a set-overlap precision. A low/zero value means the probe's selectivity does not
    # track causal use: the present-but-unused finding. Undefined (null) without the das
    # cross-reference, or when the causal indicator has no contrast (all cells causal, or
    # all transient) so a correlation is not defined. `_selmag` is the named-strength used
    # for the standardised minimality M.
    _sel = Float64[ c.selectivity for c in r.cells if c.n_classes > 1 && isfinite(c.selectivity) ]
    _eff = Float64[ (c.not_causally_used ? 0.0 : 1.0) for c in r.cells
                    if c.n_classes > 1 && isfinite(c.selectivity) ]
    _selmag = Float64[ max(0.0, x) for x in _sel ]
    _Fcausal = (!r.oracle_xref_available || length(_sel) < 2 ||
                Statistics.std(_sel) == 0.0 || Statistics.std(_eff) == 0.0) ? nothing :
               (let cc = Statistics.cor(_sel, _eff); isfinite(cc) ? cc : nothing end)

    rec = Dict{String,Any}(
        "paper" => "P2",
        "phase" => "phaseC_mechanistic",
        "method" => "linear_probing_control_tasks",
        "game" => r.game,
        "state" => r.state_kind == "varied_action" ?
                   "f1..f$(r.frames) (varied-action trajectory)" :
                   "gameplay(seed=$(r.st_seed),prefix=$(r.st_prefix))+$(r.st_horizon), " *
                   "NOOP-continued to f$(r.frames)",
        "target_output" => "concept decodability from VCS state (labelled RAM cells)",
        # headline §R scalar: the CAUSAL-agreement F = the point-biserial correlation of
        # per-cell selectivity (the probe's claim) with the oracle's causally-used indicator
        # (sec:triad's ρ between attribution and true causal effect). Low/zero = selectivity
        # does not track causal use (the present-but-unused finding). This replaces
        # mean_selectivity (decodability), which is kept under extra.mean_selectivity.
        # Undefined (null) without the das cross-reference / without causal contrast.
        "metric_name" => "probe_selectivity_vs_causal_use_correlation",
        "value" => _Fcausal,
        "stderr" => nothing,
        "ci" => nothing,
        "n" => r.n_probeable,
        "seed" => 0,
        "where" => "local",
        "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene/das@$(r.game) — transient cells = not-causally-used",
        "timestamp" => string(round(Int, time())),
        "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            # F∧S∧M triad (paper sec:triad). F is now the CAUSAL-agreement score: the
            # fraction of the probe's DECODABLE cells that the exact oracle confirms are
            # causally USED = |decodable∧causally-used|/|decodable| (a precision of
            # decodability w.r.t. causal use). Low = the present-but-unused finding
            # (probing "finds" concepts the program does not causally use over the
            # horizon). This replaces mean_selectivity (kept under extra.mean_selectivity),
            # which is decodability, NOT causal agreement. F is defined only when the
            # oracle cross-reference (das transient cells) is available; else null.
            # M is STANDARDISED to |U*|/|U_hat| via TriadSM over per-cell vectors:
            # attr = per-cell selectivity strength (the probe's claim), odelta = per-cell
            # oracle causal effect (1 if the cell is causally used, 0 if transient). S is
            # UNDEFINED: a probe reads out PRESENCE, not a held-out do(u) output prediction.
            "triad" => let
                # standardised M = |U*|/|U_hat| (oracle movers / probe-named cells). Only
                # meaningful when the oracle xref supplies causal effects; else undefined.
                Mv, ustar, uhat, mnote =
                    (r.oracle_xref_available && !isempty(_selmag)) ? minimality_score(_selmag, _eff) :
                    (nothing, 0, 0, "M undefined: no oracle causal cross-reference (das transient cells) available")
                Dict{String,Any}(
                    "F" => jnum_or_null(_Fcausal),
                    "F_note" => "causal-agreement F = point-biserial ρ(per-cell selectivity, " *
                        "oracle causally-used indicator) over $(length(_sel)) probeable cells " *
                        "(sec:triad's correlation between attribution and true causal effect). " *
                        "Low/zero = selectivity does not track causal use (present-but-unused). " *
                        "Undefined without the das cross-reference or without causal contrast.",
                    "S" => nothing,
                    "S_note" => "S undefined: a linear probe reads out whether a concept is " *
                        "PRESENT (decodable), not whether it is causally USED; it emits no " *
                        "held-out do(u) output prediction (present≠used, Hewitt & Liang 2019)",
                    "M" => jnum_or_null(Mv),
                    "M_note" => mnote,
                    "M_true_minimal_size" => ustar, "M_named_size" => uhat,
                    "definition" => "F∧S∧M triad (03_methods.tex sec:triad): F = point-biserial " *
                        "correlation of probe selectivity with the oracle causally-used indicator; " *
                        "M = |U*|/|U_hat| (oracle causally-used cells / probe-named cells " *
                        "by selectivity); S undefined for a probe (presence, not causal use).")
            end,
            "substrate" => "jutari (Julia, HARD) — real-ROM bit-exact path",
            "n_features" => r.n_features,
            "n_bins" => N_BINS,
            "ridge_lambda" => RIDGE_LAMBDA,
            "decode_acc_min" => DECODE_ACC_MIN,
            "decode_selectivity_min" => DECODE_SELECTIVITY_MIN,
            "mean_probe_acc" => r.mean_probe_acc,
            "mean_control_acc" => r.mean_control_acc,
            "mean_selectivity" => r.mean_selectivity,
            "n_probeable" => r.n_probeable,
            "n_decodable" => r.n_decodable,
            "n_not_causally_used" => r.n_not_causally_used,
            # THE present-vs-used headline: cells decodable (present) yet flagged
            # not-causally-used by the exact intervention oracle.
            "n_decodable_not_causal" => r.n_decodable_not_causal,
            "oracle_xref_available" => r.oracle_xref_available,
            # T3 concept-label provenance. On a non-core game with no verified
            # candidates_<game>.json the concept↔cell labels are the generic fallback
            # set: the probe still runs, but the concept semantics are unverified, so
            # the concept-alignment reading is recorded n/a-flagged rather than crashing.
            "concepts_verified" => r.concepts_verified,
            "concept_alignment_metric" => r.concepts_verified ? "verified_T3" : "n/a (generic fallback labels)",
            "testbed" => Dict{String,Any}(
                "state_kind" => r.state_kind,
                "seed" => r.st_seed, "prefix" => r.st_prefix, "horizon" => r.st_horizon,
                "frames" => r.frames,
                "shared_output" => r.shared_cell == (-1, -1) ? "n/a" :
                    "screen_region(n_changed_px)@r$(r.shared_cell[1])c$(r.shared_cell[2])",
                "cause_density_above_floor" => r.cause_density,
                "cause_density_floor" => ST_FLOOR, "cause_density_gate_k" => ST_GATE_K,
                "cause_density_accepted" => r.cause_density_accepted, "n_causes" => r.n_causes,
                "note" => "P2 redesign: the recorded state trajectory is SEEDED from a " *
                    "seeded random-action GAMEPLAY stream (not the local varied-action " *
                    "tape), gated by the oracle cause-density gate (accept iff #causes " *
                    "above the floor >= k). st.actions supplies the first prefix+horizon " *
                    "frames, then the trajectory is NOOP-continued to `frames` (the probe " *
                    "needs many samples ≫ features). The probing algorithm is unchanged."),
            "cells" => [_cell_dict(c) for c in r.cells],
            "note" =>
                "Linear probing + control tasks (Alain & Bengio 2017; Hewitt & " *
                "Liang 2019) over the recorded VCS state trajectory. For each " *
                "labelled concept cell c: a closed-form one-vs-rest ridge LINEAR " *
                "probe decodes c's (4-bin) value from the REST of the RAM state " *
                "(cell c held out) → probe_acc (PRESENT/decodable). The CONTROL " *
                "task keeps the same inputs but assigns each value a fixed random " *
                "label; selectivity = probe_acc − control_acc is the honest " *
                "decodability signal (a probe high on both is just memorising). " *
                "PRESENT vs USED: a cell DECODABLE (probe_acc>=$(DECODE_ACC_MIN) " *
                "& selectivity>=$(DECODE_SELECTIVITY_MIN)) yet flagged " *
                "not-causally-used by the exact intervention oracle (das.jl " *
                "transient_downstream_cells — re-derived within the horizon, no " *
                "downstream effect) is a DECODABLE-BUT-NOT-CAUSAL cell: a counted " *
                "instance of the tuning-curve trap (§7 'Partial / misleading'). " *
                "The intervention oracle (E2/E5-2) is strictly STRONGER than " *
                "probing — present ≠ used.",
            "scales_to_cluster_via" =>
                "tools/cluster/xai_*.sbatch (E0-3) + jaxtari SOFT-STE GPU batches " *
                "(forward bit-exact to this HARD trajectory) for probing over " *
                "state × concepts × games + nonlinear/MLP probe ablations.",
        ),
    )
    open(json_path, "w") do io
        write(io, _j(rec) * "\n")
    end

    # sibling .npz arrays (SPEC §R): a (n_cells, 6) matrix of the per-cell metrics.
    n = length(r.cells)
    M = zeros(Float64, n, 6)
    cell_idx = zeros(Float64, n)
    for (i, c) in enumerate(r.cells)
        cell_idx[i] = c.cell
        M[i, 1] = c.probe_acc
        M[i, 2] = c.control_acc
        M[i, 3] = c.selectivity
        M[i, 4] = c.majority_acc
        M[i, 5] = c.decodable ? 1.0 : 0.0
        M[i, 6] = c.decodable_not_causal ? 1.0 : 0.0
    end
    write_npz(npz_path, Dict(
        "cell_index"         => cell_idx,                       # (n,) 0-based RAM idx
        # columns: probe_acc, control_acc, selectivity, majority_acc, decodable, decodable_not_causal
        "metrics"            => M,
        "n_classes"          => Float64[c.n_classes for c in r.cells],
        "not_causally_used"  => Float64[c.not_causally_used ? 1.0 : 0.0 for c in r.cells],
    ))
    return json_path, npz_path
end

function write_summary(results::Vector{GameProbeResult}; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    path = joinpath(out_dir, "linear_probing_core_summary.json")
    per_game = Dict{String,Any}[]
    for r in results
        push!(per_game, Dict{String,Any}(
            "game" => r.game,
            "n_probeable" => r.n_probeable,
            "mean_probe_acc" => r.mean_probe_acc,
            "mean_control_acc" => r.mean_control_acc,
            "mean_selectivity" => r.mean_selectivity,
            "n_decodable" => r.n_decodable,
            "n_not_causally_used" => r.n_not_causally_used,
            "n_decodable_not_causal" => r.n_decodable_not_causal,
            "oracle_xref_available" => r.oracle_xref_available,
        ))
    end
    total_dnc = sum(r.n_decodable_not_causal for r in results)
    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseC_mechanistic",
        "method" => "linear_probing_control_tasks", "scope" => "core (6 games)",
        "commit" => _git_commit(), "timestamp" => string(round(Int, time())),
        "mean_probe_acc" => sum(r.mean_probe_acc for r in results) / length(results),
        "mean_control_acc" => sum(r.mean_control_acc for r in results) / length(results),
        "mean_selectivity" => sum(r.mean_selectivity for r in results) / length(results),
        "total_decodable" => sum(r.n_decodable for r in results),
        "total_not_causally_used" => sum(r.n_not_causally_used for r in results),
        # the program-level present-vs-used headline across the core set
        "total_decodable_not_causal" => total_dnc,
        "per_game" => per_game,
        "note" => "Phase-C linear probing + control tasks over the 6 core games; " *
                  "probe accuracy AND selectivity (probe − control) for concept " *
                  "decodability, cross-referenced to the exact intervention " *
                  "oracle (das transient cells) to count DECODABLE-BUT-NOT-CAUSAL " *
                  "cells — the present-vs-used gap (experiment_design.md §6/§7: " *
                  "'Partial / misleading — present ≠ used, the tuning-curve trap').",
    )
    open(path, "w") do io
        write(io, _j(rec) * "\n")
    end
    return path
end

# ============================================================================
# Self-check (DoD: the small test) — positive control on Pong:
#   1. the trajectory records a non-degenerate, multi-valued tape (probeable
#      cells exist) — the precondition for any probe;
#   2. the POSITIVE CONTROL: at least one concept cell is linearly DECODABLE with
#      POSITIVE selectivity (probe meaningfully > control) — probing works at all
#      on this representation;
#   3. the control task is a genuine baseline: mean control accuracy is strictly
#      below mean probe accuracy (selectivity > 0) — the metric DISCRIMINATES
#      real structure from probe capacity;
#   4. probe accuracy >= the majority-class floor on the decodable cells — the
#      probe is not worse than trivial.
# Exits nonzero (via `error`) on any failure.
# ============================================================================
function self_check(; game = "pong", frames = 600)
    println("[self-check] running linear probing on $game (positive control)...")
    r = run_game(; game = game, frames = frames, verbose = false)
    r.n_probeable >= 1 ||
        error("self-check FAIL: no probeable (multi-valued) concept cell — degenerate trajectory")
    any(c -> c.decodable, r.cells) ||
        error("self-check FAIL: NO concept cell is linearly decodable with selectivity — " *
              "probing finds nothing (expected at least one present concept on Pong)")
    r.mean_selectivity > 0.0 ||
        error("self-check FAIL: mean selectivity = $(r.mean_selectivity) not > 0 — " *
              "control task does not discriminate (probe == control)")
    # probe must beat the majority floor on the decodable cells
    dec = [c for c in r.cells if c.decodable]
    all(c -> c.probe_acc >= c.majority_acc - 1e-9, dec) ||
        error("self-check FAIL: a decodable cell's probe is below the majority floor")
    println("[self-check] PASS — probeable=$(r.n_probeable) ✓, " *
            "≥1 decodable cell ✓, mean_selectivity=$(round(r.mean_selectivity, digits = 3)) > 0 ✓, " *
            "probe≥majority on decodable ✓; " *
            "decodable=$(r.n_decodable) not_used=$(r.n_not_causally_used) " *
            "DECODABLE_NOT_CAUSAL=$(r.n_decodable_not_causal) " *
            "(present≠used; oracle_xref=$(r.oracle_xref_available ? "yes" : "no"))")
    return true
end

# ============================================================================
# CLI
# ============================================================================
function main(args = ARGS)
    games = CORE_GAMES
    frames = 600
    do_self_check = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--games"
            games = xai_resolve_games(args[i + 1], CORE_GAMES); i += 2
        elseif a == "--game"; games = [args[i + 1]]; i += 2
        elseif a == "--frames"; frames = parse(Int, args[i + 1]); i += 2
        elseif a == "--selftest" || a == "--self-check"; do_self_check = true; i += 1
        else; i += 1
        end
    end

    if do_self_check
        self_check(; frames = frames)
        return nothing
    end

    println("[linear_probing] games=$(join(games, ",")) frames=$frames (jutari/Julia)")
    results = GameProbeResult[]
    na_records = Tuple{String,String}[]
    for g in games
        println("\n========== $g ==========")
        try
            r = run_game(; game = g, frames = frames, verbose = true)
            jp, np = write_game_result(r)
            println("[$g] mean_selectivity=$(round(r.mean_selectivity, digits = 3)) " *
                    "decodable=$(r.n_decodable) not_used=$(r.n_not_causally_used) " *
                    "DECODABLE_NOT_CAUSAL=$(r.n_decodable_not_causal)")
            println("[$g] wrote $jp")
            println("[$g] arrays  $np")
            push!(results, r)
        catch e
            # A game DEGENERATE/STATIC at the shared gameplay state (or a bit-exact
            # re-run failure) records an n/a and is SKIPPED — it does NOT abort the
            # whole battery. Many non-core games are static at prefix=90; flag for a
            # per-game livelier prefix later.
            msg = first(split(sprint(showerror, e), '\n'))
            println("[linear_probing] !! $g SKIPPED (n/a): $msg")
            push!(na_records, (g, msg))
        end
    end
    isempty(na_records) || println("\n[linear_probing] $(length(na_records)) game(s) n/a " *
        "(degenerate/static at the shared state): $(join([g for (g, _) in na_records], ", "))")

    if length(results) > 1
        sp = write_summary(results)
        println("\n[linear_probing] core summary -> $sp")
    end

    println("\n[linear_probing] headline (probe acc / selectivity / present≠used, all games):")
    for r in results
        println("    $(rpad(r.game, 16)) " *
                "probe=$(rpad(round(r.mean_probe_acc, digits = 3), 5)) " *
                "ctrl=$(rpad(round(r.mean_control_acc, digits = 3), 5)) " *
                "sel=$(rpad(round(r.mean_selectivity, digits = 3), 6)) " *
                "decodable=$(rpad(r.n_decodable, 2)) " *
                "not_used=$(rpad(r.n_not_causally_used, 2)) " *
                "DECODABLE_NOT_CAUSAL=$(r.n_decodable_not_causal)")
    end
    return results
end

end # module

# run as a script (not when `include`d by a test)
if abspath(PROGRAM_FILE) == @__FILE__
    LinearProbing.main()
end
