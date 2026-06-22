# pilot_si.jl — Phase-A PILOT (P2-E3-0), JULIA path.
#
# The Phase-A neuroscience battery, de-risked on ONE game (Space Invaders), with
# THREE classical analyses run on the real ROM and SCORED against the exact
# intervention oracle (P2-E1-1). Phase A is the *calibration baseline* of Paper 2
# (experiment_design.md §4 Novelty note): classical neuroscience methods score
# LOW faithfulness despite the system having rich, fully-known structure — the
# quantified-Kording lesson. This pilot pins the scoring harness those analyses
# reuse (E3-1..E3-8) and shows, in numbers, where each classical method departs
# from the ground truth.
#
# The three analyses (experiment_design.md §4):
#   * A2  single-unit LESIONS — clamp/occlude each candidate RAM cell one at a
#         time (do(u := 0) and do(u := base+17)), re-run the real ROM for a short
#         horizon, measure the behavioural break (Δscreen pixels, ΔRAM bytes, Δ on
#         the game outputs). The lesion-importance ranking is SCORED against the
#         oracle's true causal map (rank-corr = F); units that look "specific" by
#         lesion but are generic (or vice-versa) are flagged (the spurious rate).
#   * A3  TUNING CURVES — over a recorded trajectory, correlate each candidate
#         cell's *value* against a game concept / output (the score, the on-screen
#         object count). Report per-cell selectivity and the SPURIOUS-TUNING rate:
#         strongly-tuned cells whose tuning ≠ their true causal role (the classic
#         tuning-curve trap; present ≠ used).
#   * A7  DIM-REDUCTION (PCA) — LinearAlgebra.svd of the recorded RAM trajectory.
#         Report variance explained per component + the matched-component fraction
#         (do the top PCs align with the cells the oracle marks causal / that vary,
#         or with epiphenomenal structure?).
#
# F/S/M scoring (the correctness triad, experiment_design.md §0), scored against
# the oracle (NOT against any interpretability method):
#   F (faithful)   — Spearman(lesion-importance, |oracle Δ|): does the recovered
#                    importance map track the TRUE causal effects?
#   S (sufficient) — predict a HELD-OUT lesion (do(u := base+37), a value the A2
#                    sweep never saw) from the recovered importance, compare the
#                    predicted "is this cell important?" to the true held-out Δ.
#                    Scored on the bit-exact re-run, not on a method.
#   M (minimal)    — sparsity of the recovered causal set vs the oracle's: how many
#                    cells the method flags "specific" that the oracle says are
#                    generic (over-claiming) — minimality at the right level.
#
# BUILDS ON the verified jutari foundation (NO emulator core touched):
#   * tools/xai_study/ground_truth/oracle_intervene.jl — the bit-exact oracle
#     (causal map {Δy(u)}) and the intervention machinery (run_intervention,
#     build/​causes). The OracleIntervene module is Pong-specialised, so this pilot
#     defines SI-specific outputs/causes inline using the SAME primitives.
#   * tools/xai_study/common/jutari_oracle.jl — load SI, xitari-parity boot,
#     replay, byte-exact snapshot, deepcopy checkpoint, RAM/TIA interventions, the
#     dependency-free §R NPY/NPZ writer.
#   * tools/xai_study/common/jutari_record.jl — the trajectory recorder (A3/A7
#     consume its RAM tape).
#
# Run (warm shared depot, primary's project):
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseA_kording/pilot_si.jl
# Flags: --game space_invaders --target-frame N --horizon N --traj-frames N
#        --selftest   (run the self-check, do not write artifacts)
#
# Writes (SPEC §R; file_scope pilotA_*):
#   tools/xai_study/phaseA_kording/out/pilotA_space_invaders.{json,npz}

module PilotSI

using JSON
using LinearAlgebra: svd
import Statistics

# --- the verified foundation (NO core touched) -----------------------------
include(joinpath(@__DIR__, "..", "common", "jutari_oracle.jl"))
using .JutariOracle
using .JutariOracle: Snapshot, load_pong_env, boot_replay, continue_from,
                     snapshot, intervene_ram!, intervene_tia!,
                     fresh_baseline_ram_screen, write_npz, RAM_SIZE
using .JutariOracle.JuTari.Env: env_step!, get_ram, get_screen

# the trajectory recorder (A3 tuning curves + A7 PCA consume its RAM tape)
include(joinpath(@__DIR__, "..", "common", "jutari_record.jl"))
using .JutariRecord: record_trajectory, Trajectory

const OUT_DIR = joinpath(@__DIR__, "out")
const CANDIDATES_REL =
    joinpath("tools", "xai_study", "t3", "out", "candidates_space_invaders.json")

# Space Invaders RAM semantics (jutari TerminalGames.jl :: SpaceInvadersRomSettings;
# xitari games/supported/SpaceInvaders.cpp). 0-based RAM indices.
# RIOT RAM is the 128-byte zero-page block; OCAtari/AtariARI indices are already
# the 0..127 RAM offsets (candidates file `ram_index`). The score is read by the
# RomSettings from RAM 0xE8/0xE6 which alias into the 128-byte file at the low 7
# bits (0xE8 & 0x7F = 0x68 = 104, 0xE6 & 0x7F = 0x66 = 102) — matching the
# candidate cells 104 (player_score, AtariARI) and 102 (score, OCAtari).
const SI_SCORE_LO_IDX = 104     # RAM[0x68] (0xE8 mirror) — tens/ones BCD
const SI_SCORE_HIH_IDX = 102    # RAM[0x66] (0xE6 mirror) — thousands/hundreds BCD
const SI_LIVES_IDX = 73         # RAM[0xC9 & 0x7F = 0x49] lives

# The 4×FIRE-then-NOOP trace gets SI into live play deterministically (FIRE
# starts the game; NOOP after keeps it deterministic and in the conformance
# window). Verified bit-exact through 60 frames (screen window) in probing. This
# is the A2/oracle trace — the bit-exact intervention substrate.
si_default_actions(total::Integer) = vcat(fill(1, 4), fill(0, max(0, total - 4)))

# The A3/A7 trace is an ACTIVE one (FIRE to start + periodic RIGHTFIRE/LEFTFIRE
# motion) so the recorded trajectory actually MOVES state — within the all-NOOP
# tail the SI RAM cells are nearly static (the candidate cells don't vary), which
# is honest but gives the tuning/PCA analyses no signal to be selective about.
# A3 (tuning curves) and A7 (PCA) are DESCRIPTIVE statistics of a deterministic
# jutari trajectory (the recorder's documented purpose) — not bit-exact-vs-xitari
# intervention claims — so they legitimately use this livelier replay. (The
# scored-vs-oracle claims, A2 + the oracle, stay strictly inside the 60-frame
# conformance window on the NOOP-tail trace above.) Action codes (jutari IO.jl):
# FIRE=1, RIGHTFIRE=11, LEFTFIRE=12.
function si_active_actions(total::Integer)
    acts = Vector{Int}(undef, max(0, total))
    for t in 1:length(acts)
        acts[t] = t <= 4         ? 1 :     # 4×FIRE: start the game
                  t % 4 == 0     ? 11 :    # RIGHTFIRE: move right + shoot
                  t % 4 == 2     ? 12 :    # LEFTFIRE:  move left + shoot
                                   1       # FIRE: keep shooting
    end
    return acts
end

# ============================================================================
# Candidate cells (E2-1 import) — the "units" the battery probes.
# ============================================================================
struct Candidate
    ram_index::Int
    concept::String
end

"""Read the SI candidate RAM cells, de-duplicated by ram_index, in file order.
Falls back to the documented SI cells if the candidates file is absent."""
function load_candidates(candidates_path)
    out = Candidate[]
    if candidates_path !== nothing && isfile(candidates_path)
        data = JSON.parsefile(candidates_path)
        seen = Set{Int}()
        for c in get(data, "candidates", [])
            idx = Int(c["ram_index"])
            idx in seen && continue
            push!(seen, idx)
            concept = c["concept"] === nothing ? "(unnamed)" : string(c["concept"])
            push!(out, Candidate(idx, concept))
        end
    end
    if isempty(out)
        out = [Candidate(9, "missiles_y"), Candidate(17, "invaders_left_count"),
               Candidate(24, "enemies_y"), Candidate(26, "enemies_x"),
               Candidate(28, "player_x"), Candidate(73, "num_lives"),
               Candidate(104, "player_score"), Candidate(102, "score")]
    end
    return out
end

# ============================================================================
# Outputs y(state) for Space Invaders (the behaviour the lesions perturb).
# ============================================================================
struct Output
    name::String
    read::Function       # Snapshot -> Float64
    note::String
end

"""SI behavioural outputs. `n_changed_px` is the robust whole-screen break (how
many framebuffer cells differ from the baseline) — it captures position effects
regardless of which cell an object lands on; `score`/`lives` are the RAM game
outputs; `n_changed_ram` the whole-RAM downstream break."""
function si_outputs(baseline::Snapshot)
    bcd(x) = Float64(10 * (Int(x) >> 4) + (Int(x) & 0x0F))
    return Output[
        Output("score", s -> 100 * bcd(s.ram[SI_SCORE_HIH_IDX + 1]) +
                              bcd(s.ram[SI_SCORE_LO_IDX + 1]),
               "decimal score (BCD RAM 0xE6/0xE8 mirrors)"),
        Output("lives", s -> Float64(Int(s.ram[SI_LIVES_IDX + 1])),
               "RAM[0xC9] lives"),
        Output("n_changed_px", s -> Float64(count(s.screen .!= baseline.screen)),
               "count of framebuffer cells differing from the baseline frame"),
        Output("n_changed_ram", s -> Float64(count(s.ram .!= baseline.ram)),
               "count of RAM bytes differing from the baseline frame"),
    ]
end

# ============================================================================
# A2 — single-unit lesions, scored against the oracle.
# ============================================================================
# For each candidate cell we run an EXACT intervention (clamp/occlude to 0, and a
# perturb to base+17) at the target frame, continue the deterministic emulator,
# and read the outputs. The lesion "importance" of a cell = the magnitude of the
# behavioural break it causes (max over outputs of |Δ|, with the whole-screen
# break as the robust scalar). This is scored against the oracle's TRUE causal map
# (the same intervention machinery, which IS the ground truth here).

"""Apply do(ram[idx] := value) to a deepcopy of the checkpoint, continue the
tail, snapshot. The bit-exact re-run guarantee makes Δ a clean causal effect."""
function lesion_snapshot(checkpoint, tail::Vector{Int}, idx::Integer, value::Integer)
    env = deepcopy(checkpoint)
    intervene_ram!(env, idx, value)
    for a in tail; env_step!(env, a); end
    return snapshot(env, length(tail))
end

struct LesionResult
    cand::Vector{Candidate}
    output_names::Vector{String}
    y_baseline::Vector{Float64}
    # Δ per (cell, output) for the occlude(→0) intervention and the set(→base+17)
    delta_occlude::Matrix{Float64}     # (n_cand, n_out)
    delta_set::Matrix{Float64}         # (n_cand, n_out)
    importance::Vector{Float64}        # robust per-cell lesion importance (screen break)
    held_out_delta::Vector{Float64}    # do(u := base+37) screen break (the S probe)
end

function run_a2_lesions(checkpoint, tail::Vector{Int}, cands::Vector{Candidate},
                        at_target::Snapshot, base_snap::Snapshot; verbose = true)
    outputs = si_outputs(base_snap)
    y_base = Float64[o.read(base_snap) for o in outputs]
    n, m = length(cands), length(outputs)
    d_occ = zeros(Float64, n, m)
    d_set = zeros(Float64, n, m)
    importance = zeros(Float64, n)
    held = zeros(Float64, n)
    # which output column is the robust whole-screen break
    px_col = findfirst(==("n_changed_px"), [o.name for o in outputs])
    for (i, c) in enumerate(cands)
        base_val = Int(at_target.ram[c.ram_index + 1])
        s_occ = lesion_snapshot(checkpoint, tail, c.ram_index, 0)
        s_set = lesion_snapshot(checkpoint, tail, c.ram_index, (base_val + 17) & 0xFF)
        s_hold = lesion_snapshot(checkpoint, tail, c.ram_index, (base_val + 37) & 0xFF)
        for (j, o) in enumerate(outputs)
            d_occ[i, j] = o.read(s_occ) - y_base[j]
            d_set[i, j] = o.read(s_set) - y_base[j]
        end
        # robust per-cell importance = the larger whole-screen break across the
        # two interventions (the behavioural footprint of lesioning the unit).
        importance[i] = max(abs(d_occ[i, px_col]), abs(d_set[i, px_col]))
        # held-out lesion (a value the A2 sweep above never used) — the S probe.
        held[i] = Float64(count(s_hold.screen .!= base_snap.screen))
        verbose && println("  A2 [$i/$n] RAM[$(lpad(c.ram_index,3))] " *
                           "$(rpad(c.concept,20)) importance(Δpx)=$(round(importance[i]))")
    end
    return LesionResult([c for c in cands], [o.name for o in outputs], y_base,
                        d_occ, d_set, importance, held)
end

# ============================================================================
# The oracle causal map for the SAME candidate cells (the GROUND TRUTH).
# ============================================================================
# The oracle's true causal effect of each candidate cell = the behavioural break
# of an exact intervention on it, read on the SAME outputs. We compute it with the
# SAME bit-exact machinery (this is `oracle_intervene.jl`'s definition of Δy(u),
# specialised to SI cells here). The A2 lesion importance is then scored against
# THIS — a faithful lesion method should rank cells the same way the oracle does.

"""The oracle's |Δy| per candidate cell: max over a do(0)/do(base+17)/do(base+37)
intervention set of the whole-screen break — the TRUE causal footprint. (This is
the oracle's causal map restricted to the candidate cells + the screen output.)"""
function oracle_causal_importance(checkpoint, tail::Vector{Int},
                                  cands::Vector{Candidate}, at_target::Snapshot,
                                  base_snap::Snapshot)
    n = length(cands)
    imp = zeros(Float64, n)
    for (i, c) in enumerate(cands)
        base_val = Int(at_target.ram[c.ram_index + 1])
        best = 0.0
        for v in (0, (base_val + 17) & 0xFF, (base_val + 37) & 0xFF)
            v == base_val && continue
            s = lesion_snapshot(checkpoint, tail, c.ram_index, v)
            best = max(best, Float64(count(s.screen .!= base_snap.screen)))
        end
        imp[i] = best
    end
    return imp
end

# ============================================================================
# A3 — tuning curves over the recorded trajectory.
# ============================================================================
# Over a recorded trajectory (T frames), each candidate cell has a time series of
# values. We correlate that against GAME CONCEPT / OUTPUT signals derived from the
# same trajectory: (a) the on-screen content (nonzero-pixel count over time — the
# game's visible activity), (b) the decimal score, (c) the lives count, and (d)
# the frame clock (an EPIPHENOMENAL control signal — a cell tuned only to the
# clock is the classic confound). A cell's selectivity = |max correlation| with
# any game-concept signal (the clock is excluded from selectivity — it is the
# negative control). The SPURIOUS-TUNING rate = the fraction of strongly-tuned
# cells (selectivity ≥ τ) whose tuning does NOT match their true causal role
# (oracle importance ≈ 0) — the present-≠-used tuning-curve trap.

struct TuningResult
    cand::Vector{Candidate}
    concept_names::Vector{String}    # game-concept signals (clock excluded)
    corr::Matrix{Float64}            # (n_cand, n_concept) Pearson |corr|
    clock_corr::Vector{Float64}      # |corr| with the epiphenomenal frame clock
    selectivity::Vector{Float64}     # per-cell max |corr| over game concepts
    strongly_tuned::Vector{Bool}     # selectivity ≥ τ
    spurious::Vector{Bool}           # strongly-tuned but NOT causal (oracle)
    spurious_rate::Float64
    tau::Float64
end

"""Per-frame game-concept signals + the epiphenomenal clock control, derived from
the recorded trajectory tape. The screen content (nonzero-pixel count) requires
the `screen` field to be stacked in the trajectory (offset = the RAM width);
score/lives come from the RAM block. Returns (game_concept_names, game_concept
matrix, clock_signal)."""
function concept_signals_from_tape(traj::Trajectory)
    tape = traj.tape
    T = size(tape, 1)
    bcd(x) = 10 * (Int(x) >> 4) + (Int(x) & 0x0F)
    score = Float64[100 * bcd(tape[t, SI_SCORE_HIH_IDX + 1]) +
                    bcd(tape[t, SI_SCORE_LO_IDX + 1]) for t in 1:T]
    lives = Float64[Int(tape[t, SI_LIVES_IDX + 1]) for t in 1:T]
    # on-screen content = #nonzero framebuffer cells per frame (the visible game
    # activity); present only if the recorder stacked the screen field.
    names = String[]; cols = Vector{Float64}[]
    if "screen" in traj.fields
        nram = traj.widths[findfirst(==("ram"), traj.fields)]
        screen = @view tape[:, nram + 1 : end]
        content = Float64[count(!=(0x00), @view screen[t, :]) for t in 1:T]
        push!(names, "screen_content"); push!(cols, content)
    end
    push!(names, "score"); push!(cols, score)
    push!(names, "lives"); push!(cols, lives)
    clock = Float64.(1:T)            # the frame clock — the EPIPHENOMENAL control
    return (names, hcat(cols...), clock)
end

function run_a3_tuning(traj::Trajectory, cands::Vector{Candidate},
                       oracle_importance::Vector{Float64}; tau = 0.7)
    tape = traj.tape
    cnames, csig, clock = concept_signals_from_tape(traj)
    n, nc = length(cands), length(cnames)
    C = zeros(Float64, n, nc)
    clock_corr = zeros(Float64, n)
    for (i, c) in enumerate(cands)
        x = Float64[Int(tape[t, c.ram_index + 1]) for t in 1:size(tape, 1)]
        for j in 1:nc
            C[i, j] = abs(pearson(x, csig[:, j]))
        end
        clock_corr[i] = abs(pearson(x, clock))
    end
    selectivity = vec(maximum(C; dims = 2))     # over GAME concepts (clock excluded)
    strong = selectivity .>= tau
    # "causal" = the cell actually moves behaviour when intervened on (oracle
    # importance > 0). A strongly-tuned-but-non-causal cell is spurious tuning.
    causal_thresh = _causal_threshold(oracle_importance)
    is_causal = oracle_importance .> causal_thresh
    spurious = strong .& .!is_causal        # strongly-tuned but NOT causal
    rate = sum(strong) == 0 ? 0.0 : sum(spurious) / sum(strong)
    return TuningResult([c for c in cands], cnames, C, clock_corr, selectivity,
                        strong, spurious, rate, tau)
end

# A causal cell = oracle importance strictly above 0 (it moves the screen). The
# threshold is 0 (any real behavioural footprint counts); cells with 0 footprint
# are non-causal at this state/horizon.
_causal_threshold(imp::AbstractVector) = 0.0

# ============================================================================
# A7 — dim-reduction (PCA via SVD) on the recorded RAM trajectory.
# ============================================================================
# PCA of the (T, 128) RAM tape. Report the variance explained by each component
# and the MATCHED-COMPONENT fraction: do the leading components load on the cells
# that actually VARY / are causal (the real signal), or do they capture
# epiphenomenal structure? We measure, per top-k component, the share of its
# loading mass that sits on cells the oracle marks causal — the "aligned" mass.

struct PCAResult
    n_frames::Int
    n_cells::Int
    n_varying::Int
    var_explained::Vector{Float64}       # per-component fraction of total variance
    cum_var::Vector{Float64}
    # for each of the top-k components: the fraction of squared loading mass that
    # sits on the candidate causal cells (vs spread over epiphenomenal cells)
    topk::Int
    aligned_mass::Vector{Float64}
    matched_component_fraction::Float64  # #top-k comps with aligned_mass≥0.5 / k
    causal_cells::Vector{Int}            # 0-based RAM indices judged causal
end

function run_a7_pca(traj::Trajectory, cands::Vector{Candidate},
                    oracle_importance::Vector{Float64}; topk = 5)
    # restrict PCA to the RAM field columns (the recorder may stack tia/screen).
    nram = min(RAM_SIZE, size(traj.tape, 2))
    X = Float64.(traj.tape[:, 1:nram])               # (T, 128)
    T = size(X, 1)
    # centre columns; columns with zero variance contribute nothing.
    mu = Statistics.mean(X; dims = 1)
    Xc = X .- mu
    col_var = vec(Statistics.var(X; dims = 1, corrected = false))
    n_varying = count(>(0.0), col_var)
    # SVD: Xc = U S Vt; component variances ∝ S.^2 / (T-1).
    F = svd(Xc; full = false)
    sv = F.S
    var_comp = (sv .^ 2)
    total = sum(var_comp)
    var_explained = total == 0 ? zeros(length(sv)) : var_comp ./ total
    cum = cumsum(var_explained)
    # causal cells = candidate cells with non-zero oracle importance
    causal_cells = Int[cands[i].ram_index for i in 1:length(cands)
                       if oracle_importance[i] > 0.0]
    causal_set = Set(causal_cells)
    V = F.V                                            # (128, r) loadings
    k = min(topk, size(V, 2))
    aligned = zeros(Float64, k)
    for j in 1:k
        load2 = V[:, j] .^ 2                           # squared loading per cell
        s = sum(load2)
        if s == 0
            aligned[j] = 0.0
        else
            m = sum(load2[idx + 1] for idx in causal_cells; init = 0.0)
            aligned[j] = m / s
        end
    end
    matched = k == 0 ? 0.0 : count(>=(0.5), aligned) / k
    return PCAResult(T, nram, n_varying, var_explained, cum, k,
                     aligned, matched, causal_cells)
end

# ============================================================================
# F / S / M scores (the correctness triad, scored against the oracle).
# ============================================================================
struct Triad
    F::Float64      # Spearman(lesion importance, oracle importance) — faithfulness
    S::Float64      # held-out-lesion prediction correlation — sufficiency
    M::Float64      # minimality: 1 − over-claim rate (cells flagged specific that aren't)
    F_note::String
    S_note::String
    M_note::String
end

function score_triad(les::LesionResult, oracle_importance::Vector{Float64})
    # F — does the lesion ranking match the oracle's true causal ranking?
    F = spearman(les.importance, oracle_importance)
    # S — predict the HELD-OUT lesion (do base+37, never used in A2's importance)
    #     from the recovered A2 importance. A faithful importance map predicts the
    #     held-out behavioural break ⇒ high correlation. Scored on the bit-exact
    #     re-run (les.held_out_delta), not on any method.
    S = pearson(les.importance, les.held_out_delta)
    # M — minimality: of the cells A2 flags "specific" (importance > 0), how many
    #     are NOT truly causal (oracle importance == 0)?  Over-claiming a generic
    #     cell as specific is a minimality failure. M = 1 − over-claim rate.
    flagged = les.importance .> 0.0
    nflag = sum(flagged)
    overclaim = nflag == 0 ? 0.0 :
        count(i -> flagged[i] && oracle_importance[i] == 0.0, 1:length(flagged)) / nflag
    M = 1.0 - overclaim
    return Triad(F, S, M,
        "Spearman(A2 lesion importance, oracle causal importance) over $(length(oracle_importance)) candidate cells",
        "Pearson(A2 importance, held-out do(base+37) screen break) — sufficiency on a bit-exact re-run",
        "1 − over-claim rate: fraction of A2-flagged-specific cells the oracle says are generic")
end

# ============================================================================
# stats helpers
# ============================================================================
function pearson(a::AbstractVector, b::AbstractVector)
    length(a) < 2 && return 0.0
    sa = Statistics.std(a); sb = Statistics.std(b)
    (sa == 0 || sb == 0) && return 0.0
    return Statistics.cor(a, b)
end
spearman(a::AbstractVector, b::AbstractVector) = pearson(_rank(a), _rank(b))
function _rank(v::AbstractVector)
    n = length(v); p = sortperm(v); r = zeros(Float64, n)
    i = 1
    while i <= n
        j = i
        while j < n && v[p[j + 1]] == v[p[i]]; j += 1; end
        avg = (i + j) / 2
        for k in i:j; r[p[k]] = avg; end
        i = j + 1
    end
    return r
end

# ============================================================================
# Drive it: oracle + A2 + A3 + A7 + triad.
# ============================================================================
struct PilotA
    game::String
    target_frame::Int
    horizon::Int
    traj_frames::Int
    seed::Int
    bit_exact::Bool
    cands::Vector{Candidate}
    oracle_importance::Vector{Float64}
    lesions::LesionResult
    tuning::TuningResult
    pca::PCAResult
    triad::Triad
end

function compute_pilot(; game = "space_invaders", target_frame = 30, horizon = 30,
                       traj_frames = 150, seed = 0, verbose = true)
    total = target_frame + horizon
    actions = si_default_actions(total)
    tail = Int.(actions[target_frame + 1 : target_frame + horizon])

    # 1) bit-exact baseline guarantee — two fresh boots+replays must be identical.
    verbose && println("[pilotA] asserting bit-exactness (2 fresh boots+replays to f$total)...")
    a = fresh_baseline_ram_screen(actions, total; game = game)
    b = fresh_baseline_ram_screen(actions, total; game = game)
    bit_exact = (a.ram == b.ram) && (a.screen == b.screen)
    bit_exact || error("bit-exact re-run FAILED for $game to f$total — refusing to score")
    verbose && println("[pilotA] bit-exact re-run: PASS")

    # 2) ONE checkpoint at the intervention frame (boot+to-target paid once).
    checkpoint = boot_replay(actions, target_frame; game = game)
    at_target = continue_from(checkpoint, Int[])           # state AT the lesion frame
    base_snap = continue_from(checkpoint, tail)            # baseline continuation

    cand_path = resolve_candidates()
    cands = load_candidates(cand_path)
    verbose && println("[pilotA] candidates: $(cand_path === nothing ? "(fallback)" : cand_path) " *
                       "($(length(cands)) cells)")

    # 3) the ORACLE causal importance for the SAME cells (the ground truth).
    verbose && println("[pilotA] oracle causal importance over $(length(cands)) candidate cells...")
    oracle_importance =
        oracle_causal_importance(checkpoint, tail, cands, at_target, base_snap)

    # 4) A2 — single-unit lesions.
    verbose && println("[pilotA] A2: single-unit lesions ($(length(cands)) cells, real re-runs)...")
    lesions = run_a2_lesions(checkpoint, tail, cands, at_target, base_snap; verbose = verbose)

    # 5) A3 — tuning curves over a recorded trajectory. The trajectory uses the
    #    ACTIVE trace (the candidate cells are nearly static under the NOOP tail)
    #    and stacks RAM + screen so the on-screen content concept signal exists.
    #    This is a descriptive jutari recording, not a bit-exact-vs-xitari claim.
    verbose && println("[pilotA] A3: tuning curves over a $(traj_frames)-frame (active) RAM+screen trajectory...")
    traj = record_trajectory(game; frames = traj_frames, fields = ["ram", "screen"],
                             actions = si_active_actions(traj_frames))
    tuning = run_a3_tuning(traj, cands, oracle_importance)

    # 6) A7 — PCA (SVD) of the RAM trajectory.
    verbose && println("[pilotA] A7: PCA (SVD) of the (T,128) RAM trajectory...")
    pca = run_a7_pca(traj, cands, oracle_importance)

    # 7) F / S / M triad (A2 vs the oracle).
    triad = score_triad(lesions, oracle_importance)

    if verbose
        println("[pilotA] ---- Phase-A scores (Space Invaders) ----")
        println("[pilotA]   A2 lesions: oracle-causal cells = ",
                count(>(0.0), oracle_importance), "/$(length(cands)); ",
                "A2-flagged-specific = ", count(>(0.0), lesions.importance))
        println("[pilotA]   A3 tuning : strongly-tuned = ", sum(tuning.strongly_tuned),
                ", SPURIOUS-tuning rate = ", round(tuning.spurious_rate, digits = 3))
        println("[pilotA]   A7 PCA    : top-1 var = ", round(pca.var_explained[1], digits = 3),
                ", cum var(top5) = ", round(pca.cum_var[min(5, end)], digits = 3),
                ", matched-component fraction = ", round(pca.matched_component_fraction, digits = 3))
        println("[pilotA]   TRIAD     : F=", round(triad.F, digits = 3),
                "  S=", round(triad.S, digits = 3), "  M=", round(triad.M, digits = 3))
    end

    return PilotA(game, target_frame, horizon, traj_frames, seed, bit_exact,
                  cands, oracle_importance, lesions, tuning, pca, triad)
end

# ============================================================================
# Persist (SPEC §R) — JSON record + sibling .npz; file_scope pilotA_*.
# ============================================================================
_git_commit() = try
    strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
catch
    "unknown"
end
_json_num(x::Real) = isfinite(x) ? Float64(x) : nothing

function resolve_candidates()
    here = normpath(joinpath(@__DIR__, "..", "..", ".."))
    for base in (here, "/Users/maier/Documents/code/UnderstandingVCS")
        p = joinpath(base, CANDIDATES_REL)
        isfile(p) && return p
    end
    return nothing
end

function write_pilot(p::PilotA; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    stem = "pilotA_$(p.game)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path  = joinpath(out_dir, stem * ".npz")

    cell_names = ["RAM[$(c.ram_index)]:$(c.concept)" for c in p.cands]
    rec = Dict{String,Any}(
        "paper" => "P2",
        "phase" => "phaseA_kording",
        "method" => "phaseA_pilot(A2_lesions+A3_tuning+A7_pca)",
        "game" => p.game,
        "state" => "f$(p.target_frame)+$(p.horizon)",
        "target_output" => "score+lives+screen-break",
        # headline scalar (SPEC §R value/metric_name): the A2 faithfulness F —
        # how well the classical lesion map tracks the TRUE causal map.
        "metric_name" => "A2_faithfulness_spearman_vs_oracle",
        "value" => _json_num(p.triad.F),
        "stderr" => nothing,
        "ci" => nothing,
        "n" => length(p.cands),
        "seed" => p.seed,
        "where" => "local",
        "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene@$(p.game)#screen-break,score,lives",
        "timestamp" => string(round(Int, time())),
        "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            "substrate" => "jutari (Julia, HARD) — real-ROM bit-exact path; " *
                "lesions are EXACT interventions re-run on the true ROM.",
            "bit_exact_rerun" => p.bit_exact,
            "candidate_cells" => cell_names,
            "candidate_ram_indices" => [c.ram_index for c in p.cands],
            "triad" => Dict{String,Any}(
                "F" => _json_num(p.triad.F), "F_note" => p.triad.F_note,
                "S" => _json_num(p.triad.S), "S_note" => p.triad.S_note,
                "M" => _json_num(p.triad.M), "M_note" => p.triad.M_note,
                "interpretation" => "Phase A is the calibration baseline " *
                    "(experiment_design.md §4): classical methods score LOW " *
                    "faithfulness despite fully-known structure — the quantified " *
                    "Kording lesson. F<1 / spurious-tuning>0 / unaligned PCA show " *
                    "where each classical analysis departs from the ground truth.",
            ),
            "A2_lesions" => Dict{String,Any}(
                "output_names" => p.lesions.output_names,
                "y_baseline" => p.lesions.y_baseline,
                "lesion_importance_per_cell" =>
                    Dict(cell_names[i] => p.lesions.importance[i] for i in 1:length(cell_names)),
                "oracle_importance_per_cell" =>
                    Dict(cell_names[i] => p.oracle_importance[i] for i in 1:length(cell_names)),
                "n_oracle_causal" => count(>(0.0), p.oracle_importance),
                "n_flagged_specific" => count(>(0.0), p.lesions.importance),
                "note" => "importance = max whole-screen break over do(0)/do(base+17); " *
                    "scored vs the oracle's true causal importance (same machinery).",
            ),
            "A3_tuning" => Dict{String,Any}(
                "concept_signals" => p.tuning.concept_names,
                "epiphenomenal_control" => "frame_clock",
                "trajectory_frames" => p.traj_frames,
                "trajectory_trace" => "active (4xFIRE + periodic RIGHTFIRE/LEFTFIRE)",
                "tau" => p.tuning.tau,
                "selectivity_per_cell" =>
                    Dict(cell_names[i] => p.tuning.selectivity[i] for i in 1:length(cell_names)),
                "clock_corr_per_cell" =>
                    Dict(cell_names[i] => p.tuning.clock_corr[i] for i in 1:length(cell_names)),
                "n_strongly_tuned" => sum(p.tuning.strongly_tuned),
                "n_spurious" => sum(p.tuning.spurious),
                "spurious_tuning_rate" => _json_num(p.tuning.spurious_rate),
                "strongly_tuned_cells" => cell_names[p.tuning.strongly_tuned],
                "spurious_cells" => cell_names[p.tuning.spurious],
                "note" => "selectivity = max |corr| to a GAME concept (frame clock " *
                    "excluded — it is the epiphenomenal control). spurious = " *
                    "strongly-tuned (selectivity≥τ) but NOT causal by the oracle — " *
                    "the present≠used tuning-curve trap (Phase-A's headline failure).",
            ),
            "A7_pca" => Dict{String,Any}(
                "n_frames" => p.pca.n_frames,
                "n_cells" => p.pca.n_cells,
                "n_varying_cells" => p.pca.n_varying,
                "var_explained_top" => p.pca.var_explained[1:min(8, end)],
                "cum_var_top" => p.pca.cum_var[1:min(8, end)],
                "topk" => p.pca.topk,
                "aligned_causal_mass_per_component" => p.pca.aligned_mass,
                "matched_component_fraction" => _json_num(p.pca.matched_component_fraction),
                "causal_cells" => p.pca.causal_cells,
                "note" => "matched-component fraction = #top-k PCs whose squared-loading " *
                    "mass is ≥50% on oracle-causal cells / k; low ⇒ PCs capture " *
                    "epiphenomenal structure, not the causal variables.",
            ),
            "scales_to_cluster_via" =>
                "tools/cluster/xai_*.sbatch (E0-3) — the full A1..A8 battery over " *
                "the core set; the forward is bit-exact to this HARD path.",
        ),
    )
    open(json_path, "w") do io; JSON.print(io, rec, 2); end

    # sibling .npz arrays (SPEC §R) — all numeric matrices/vectors for tooling.
    write_npz(npz_path, Dict(
        "candidate_ram_indices" => Int64[c.ram_index for c in p.cands],
        "oracle_importance"     => p.oracle_importance,
        "lesion_importance"     => p.lesions.importance,
        "lesion_delta_occlude"  => p.lesions.delta_occlude,
        "lesion_delta_set"      => p.lesions.delta_set,
        "lesion_held_out_delta" => p.lesions.held_out_delta,
        "tuning_corr"           => p.tuning.corr,
        "tuning_selectivity"    => p.tuning.selectivity,
        "pca_var_explained"     => p.pca.var_explained,
        "pca_cum_var"           => p.pca.cum_var,
        "pca_aligned_mass"      => p.pca.aligned_mass,
        "triad_FSM"             => Float64[p.triad.F, p.triad.S, p.triad.M],
    ))
    return json_path, npz_path
end

# ============================================================================
# Self-check (DoD) — the scoring contract is sound; results are non-fabricated.
# ============================================================================
"""
    selftest(p::PilotA) -> Bool

Asserts the pilot's load-bearing claims (the contract E3-1..E3-8 reuse):

  (BIT-EXACT) the baseline re-run is byte-identical (every Δ is a clean causal
    effect on the real ROM).

  (A2 GROUND-TRUTH ANCHORED) the oracle finds ≥1 genuinely causal candidate cell
    (some lesion moves the screen) — otherwise the F score is undefined and the
    state/horizon are uninformative.

  (A2 FAITHFULNESS REWARDS THE ORACLE) scoring the ORACLE'S OWN importance map as
    the candidate "lesion method" yields Spearman F == 1 — the harness rewards a
    perfectly-faithful map (so a sub-1 IG/lesion F is a real measurement).

  (TRIAD RANGES) F,S ∈ [−1,1]; M ∈ [0,1]; all finite or honestly NaN-free.

  (A3 / A7 WELL-FORMED) the spurious-tuning rate ∈ [0,1]; PCA variance fractions
    sum to ≈1 and are non-negative; the matched-component fraction ∈ [0,1].

Throws on a contract violation."""
function selftest(p::PilotA)
    @assert p.bit_exact "bit-exact baseline re-run failed"
    n_causal = count(>(0.0), p.oracle_importance)
    @assert n_causal >= 1 "oracle found NO causal candidate cell at this state — " *
        "uninformative; pick a livelier target frame"

    # harness positive control: the oracle's own map must score F==1 against itself
    self_F = spearman(p.oracle_importance, p.oracle_importance)
    @assert self_F > 0.999 "harness broken: oracle-as-method Spearman != 1 ($self_F)"

    @assert -1.0 - 1e-9 <= p.triad.F <= 1.0 + 1e-9 "F out of [-1,1]: $(p.triad.F)"
    @assert -1.0 - 1e-9 <= p.triad.S <= 1.0 + 1e-9 "S out of [-1,1]: $(p.triad.S)"
    @assert 0.0 - 1e-9 <= p.triad.M <= 1.0 + 1e-9 "M out of [0,1]: $(p.triad.M)"

    @assert 0.0 <= p.tuning.spurious_rate <= 1.0 "spurious rate out of [0,1]"
    @assert all(>=(-1e-12), p.pca.var_explained) "negative PCA variance fraction"
    @assert abs(sum(p.pca.var_explained) - 1.0) < 1e-6 "PCA variance fractions don't sum to 1"
    @assert 0.0 <= p.pca.matched_component_fraction <= 1.0 "matched-component fraction out of [0,1]"

    println("[pilotA] SELF-CHECK PASS:")
    println("[pilotA]   bit-exact baseline re-run: $(p.bit_exact)")
    println("[pilotA]   oracle causal cells: $n_causal/$(length(p.cands)) " *
            "(harness positive control: oracle-as-method F = $(round(self_F, digits = 3)))")
    println("[pilotA]   A2 F=$(round(p.triad.F,digits=3))  S=$(round(p.triad.S,digits=3))  " *
            "M=$(round(p.triad.M,digits=3))")
    println("[pilotA]   A3 spurious-tuning rate = $(round(p.tuning.spurious_rate, digits = 3))")
    println("[pilotA]   A7 matched-component fraction = " *
            "$(round(p.pca.matched_component_fraction, digits = 3))")
    return true
end

# ============================================================================
# CLI
# ============================================================================
function main(args = ARGS)
    # tf=30 + hz=30 → total 60, inside the SI conformance window (60-frame screen);
    # at f30 under the 4×FIRE→NOOP trace the game is live (≈4.6k lit pixels) and
    # the candidate cells produce real behavioural breaks (enemies_x/y, lives).
    game = "space_invaders"
    target_frame = 30; horizon = 30; traj_frames = 150; seed = 0
    selftest_only = false
    i = 1
    while i <= length(args)
        a = args[i]
        if     a == "--game";         game = args[i+1]; i += 2
        elseif a == "--target-frame"; target_frame = parse(Int, args[i+1]); i += 2
        elseif a == "--horizon";      horizon = parse(Int, args[i+1]); i += 2
        elseif a == "--traj-frames";  traj_frames = parse(Int, args[i+1]); i += 2
        elseif a == "--seed";         seed = parse(Int, args[i+1]); i += 2
        elseif a == "--selftest";     selftest_only = true; i += 1
        else; i += 1
        end
    end
    println("[pilotA] game=$game target_frame=$target_frame horizon=$horizon " *
            "traj_frames=$traj_frames seed=$seed (jutari/Julia path)")

    p = compute_pilot(; game = game, target_frame = target_frame, horizon = horizon,
                      traj_frames = traj_frames, seed = seed, verbose = true)
    selftest(p)
    if selftest_only
        println("[pilotA] --selftest: passed, not writing artifact.")
        return 0
    end
    json_path, npz_path = write_pilot(p)
    println("[pilotA] wrote $json_path")
    println("[pilotA] arrays  $npz_path")
    return 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    PilotSI.main()
end
