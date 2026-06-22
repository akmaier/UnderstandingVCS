# A7_dimred.jl — Phase-A A7: dim-reduction (NMF / PCA) latent components
# (P2-E3-7), the JULIA path, run on jutari over the 6 core games.
#
# Phase A is Paper 2's *calibration baseline* (experiment_design.md §4, Novelty
# note): classical neuroscience analyses score LOW faithfulness despite the system
# having rich, fully-known structure — the quantified-Kording lesson. A7 is the
# dim-reduction member of the battery (the population-analysis staple — PCA / NMF
# of a recorded state tensor, then "do the latent components correspond to
# anything?"):
#
#   * Over a recorded deterministic jutari trajectory, take the (T, 128) RAM tape.
#   * PCA via SVD of the centred tape — variance explained per component.
#   * NMF (RAM bytes are NON-NEGATIVE, so the tape is a valid NMF input) of the raw
#     tape via dependency-free multiplicative updates (Lee & Seung 2001) — report
#     the reconstruction error (relative Frobenius) per rank.
#   * The MATCHED-COMPONENT FRACTION vs KNOWN signals (experiment_design.md §4 A7:
#     "matched-component fraction vs known signals (clock, R/W, vsync)") + the
#     optional game-variable matching: for each latent component, is its loading
#     concentrated on a SINGLE known variable (a candidate cell / a T3 concept
#     family / a derived known signal — the frame clock, the per-cell write
#     activity, the vsync-period phase), or is it a smear over many cells? A
#     component "matches" a known variable iff its top-loading mass is dominated by
#     ONE variable above a purity threshold. matched-component fraction =
#     #matched-components / #components — for both PCA and NMF, the head result the
#     prompt asks: the pilot found PCA matched-fraction ≈ 0; does NMF do better?
#
# F/S/M scoring (the correctness triad, experiment_design.md §0), scored against the
# exact intervention ORACLE (NOT against any interpretability method). A7's "claim"
# (where dim-reduction yields an importance/recovery claim) is a per-cell RECOVERY
# score: how strongly the LEADING latent components capture each candidate cell
# (its summed squared loading over the top-k components — high ⇒ the cell is a
# dominant axis of variation the method surfaces). The oracle's TRUE importance is
# the causal footprint of intervening on that cell:
#
#   oracle importance(i) = max whole-screen break of an EXACT do(cell_i := 0) /
#       do(cell_i := base+17) intervention re-run on the real ROM (the §1 oracle
#       restricted to the candidate cell + the screen output — the SAME map
#       pilot_si.jl / A2 score against).
#
#   F (faithful)   — Spearman(per-cell PCA recovery score, oracle importance) over
#                    the candidate cells: do the leading components surface the cells
#                    that are actually CAUSAL, or epiphenomenal high-variance cells?
#   S (sufficient) — predict a HELD-OUT lesion (do(cell_i := base+37), a value the
#                    importance sweep never used) from the recovery score; Pearson
#                    on the bit-exact re-run.
#   M (minimal)    — 1 − over-claim rate: of the cells A7 flags "captured" (recovery
#                    score ≥ τ-quantile, i.e. on a leading component), how many are
#                    NOT truly causal (oracle importance == 0)? Surfacing a
#                    high-variance-but-non-causal cell is the dim-reduction trap
#                    (variance ≠ causal role — present ≠ used).
#
# Positive control (as in pilot_si.jl): scoring the ORACLE'S OWN importance map as
# the candidate "method" must yield Spearman F == 1 — the harness rewards a
# perfectly-faithful map, so a sub-1 PCA/NMF recovery F is a real measurement.
#
# BUILDS ON the verified jutari foundation (NO emulator core touched):
#   * tools/xai_study/common/jutari_oracle.jl — load/boot/replay/snapshot/intervene,
#     bit-exact baseline guarantee, the dependency-free §R NPY/NPZ writer.
#   * tools/xai_study/common/jutari_record.jl — record_trajectory → (T,n) tape.
#   * tools/xai_study/ground_truth/oracle_intervene.jl — the causal-map reference
#     (this file specialises its Δy(u) machinery to per-candidate-cell importance).
#   * tools/xai_study/phaseA_kording/pilot_si.jl — the Phase-A pilot TEMPLATE
#     (PCA matched-component fraction + F/S/M + positive control + §R record shape).
#     A7 EXTENDS the pilot's single-game PCA with (i) NMF, (ii) the 6-game core run,
#     (iii) the matched-component fraction vs known signals (clock/R-W/vsync).
#   * tools/xai_study/phaseA_kording/A4_correlations.jl — the multi-game driver
#     PATTERN (per-game RomSettings, ROM-alias resolution, active/dir traces, the
#     GAME_CFG live-play map, per-game §R record + a combined index). A7 mirrors it
#     so the two share live-play tuning, staying inside A7's own file_scope.
#
# Per-game RomSettings: the shared jutari_oracle.settings_for() only knows the 3
# pilot games, so A7 supplies its own per-game settings map (matching the canonical
# tools/jutari_screen_dump.jl map) and builds envs directly via the same JuTari
# primitives — keeping everything inside A7's file_scope (it must not edit the
# shared helper). seaquest legitimately uses GenericRomSettings (Paper-1 64/64 with
# Generic).
#
# Run (warm shared depot, primary's project):
#   XAI_PRIMARY_REPO=/Users/maier/Documents/code/UnderstandingVCS \
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseA_kording/A7_dimred.jl
# Flags: --games pong,breakout,...  --target-frame N --horizon N --traj-frames N
#        --topk K  --nmf-rank R  --purity F
#        --selftest (run the self-check on the first game, write nothing)
#
# Writes (SPEC §R; file_scope A7_*): one record per game + a combined index
#   tools/xai_study/phaseA_kording/out/A7_<game>.{json,npz}
#   tools/xai_study/phaseA_kording/out/A7_dimred_all.json

module A7DimRed

using JSON
using LinearAlgebra: svd, norm
import Statistics
import Random

# --- the verified foundation (NO core touched) -----------------------------
include(joinpath(@__DIR__, "..", "common", "jutari_oracle.jl"))
using .JutariOracle
using .JutariOracle: Snapshot, snapshot, intervene_ram!, RAM_SIZE, write_npz,
                     rom_path_for
using .JutariOracle.JuTari: StellaEnvironment
using .JutariOracle.JuTari.Env: env_reset!, env_step!, get_ram

# the trajectory recorder lineage (A7 reuses its recording PATTERN but re-implements
# the loop, a7_record_ram, so each game's proper RomSettings are used — the shared
# recorder hard-codes settings_for). The include keeps the dependency explicit.
include(joinpath(@__DIR__, "..", "common", "jutari_record.jl"))

const OUT_DIR = joinpath(@__DIR__, "out")
const CORE_GAMES = ["pong", "breakout", "space_invaders",
                    "seaquest", "ms_pacman", "qbert"]

# Per-game live-play configuration — SHARED with A4 (A4_correlations.jl GAME_CFG):
# the candidate cells must actually MOVE for the trajectory to carry variation the
# dim-reduction can decompose; each game reaches live play at a different point.
# `trace` selects the trajectory input style ("fire" = FIRE+RIGHT/LEFTFIRE active
# trace; "dir" = full UP/DOWN/LEFT/RIGHT maze trace, where FIRE is a no-op).
# `in_window` flags whether the SCORED oracle frame stays inside the strict Paper-1
# 60-frame screen conformance window; games scored beyond it are HONEST descriptive
# jutari results (bit-exact jutari re-run still asserted), annotated in §R.
struct GameCfg
    target_frame::Int
    horizon::Int
    traj_frames::Int
    trace::String       # "fire" | "dir"
    in_window::Bool
end
const GAME_CFG = Dict{String,GameCfg}(
    "pong"           => GameCfg(60,  30, 150, "fire", true),
    "breakout"       => GameCfg(60,  30, 150, "fire", true),
    "space_invaders" => GameCfg(120, 30, 150, "fire", false),
    "seaquest"       => GameCfg(120, 30, 150, "fire", false),
    "ms_pacman"      => GameCfg(330, 30, 400, "dir",  false),
    "qbert"          => GameCfg(60,  30, 200, "fire", true),
)
game_cfg(game::AbstractString) =
    get(GAME_CFG, lowercase(string(game)), GameCfg(60, 30, 150, "fire", true))

# ============================================================================
# Per-game RomSettings (A7-owned; mirrors tools/jutari_screen_dump.jl's canonical
# map). seaquest -> Generic (Paper-1 64/64 with Generic). Stays in A7's file_scope.
# ============================================================================
function a7_settings_for(game::AbstractString)
    g = lowercase(string(game))
    JT = JutariOracle.JuTari
    g == "pong"           && return JT.PaddleGames.PongRomSettings()
    g == "breakout"       && return JT.PaddleGames.BreakoutRomSettings()
    g == "space_invaders" && return JT.SpaceInvadersRomSettings()
    g == "ms_pacman"      && return JT.JoystickGames.MsPacmanRomSettings()
    g == "qbert"          && return JT.JoystickGames.QbertRomSettings()
    return JT.GenericRomSettings()     # seaquest + any other
end

# A few ROM files don't match their canonical name; mirror the alias map so A7
# resolves them in this worktree's xitari/roms (or the primary's). A7-owned.
const _ROM_ALIASES = Dict("ms_pacman" => ["ms_pacman", "mspacman"],
                          "beam_rider" => ["beam_rider", "beamrider"])

function a7_rom_path(game::AbstractString)
    try
        return rom_path_for(game)
    catch
    end
    here = normpath(joinpath(@__DIR__, "..", "..", ".."))
    bases = String[here, get(ENV, "XAI_PRIMARY_REPO", ""),
                   "/Users/maier/Documents/code/UnderstandingVCS"]
    names = get(_ROM_ALIASES, lowercase(string(game)), [string(game)])
    for base in bases
        base == "" && continue
        for nm in names
            p = joinpath(base, "xitari", "roms", nm * ".bin")
            isfile(p) && return p
        end
    end
    error("ROM not found for game=$game (tried $(names) under $(filter(!isempty, bases)))")
end

"""A freshly-reset env for `game` with the A7 per-game settings + the xitari-parity
boot (60 NOOP + 4 RESET) — the SAME boot jutari_oracle uses."""
function a7_load_env(game::AbstractString)
    rom = read(a7_rom_path(game))
    env = StellaEnvironment(rom, a7_settings_for(game))
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    return env
end

"""Boot + step actions[1:target_frame]; return the env AT the intervention frame
(deepcopy it for a reusable checkpoint). Mirrors jutari_oracle.boot_replay with the
A7 per-game settings."""
function a7_boot_replay(game, actions::AbstractVector{<:Integer}, target_frame::Integer)
    env = a7_load_env(game)
    for i in 1:target_frame; env_step!(env, Int(actions[i])); end
    return env
end

"""deepcopy the checkpoint, step the tail, snapshot (byte-exact continuation)."""
function a7_continue_from(checkpoint, tail::AbstractVector{<:Integer})
    env = deepcopy(checkpoint)
    for a in tail; env_step!(env, Int(a)); end
    return snapshot(env, length(tail))
end

"""A full from-scratch replay (boot included) of actions[1:total] — the bit-exact
baseline guarantee (two fresh runs must be byte-identical)."""
function a7_fresh_baseline(game, actions::AbstractVector{<:Integer}, total::Integer)
    env = a7_load_env(game)
    for i in 1:total; env_step!(env, Int(actions[i])); end
    return snapshot(env, Int(total))
end

"""Record a (frames, 128) RAM tape on `game` with A7's per-game settings.
Re-implements record_trajectory's loop with a7_load_env so the proper RomSettings
are used (the shared recorder hard-codes settings_for)."""
function a7_record_ram(game::AbstractString, frames::Integer,
                       actions::AbstractVector{<:Integer})
    @assert length(actions) >= frames "actions shorter than frames"
    env = a7_load_env(game)
    tape = Matrix{UInt8}(undef, Int(frames), RAM_SIZE)
    for t in 1:frames
        env_step!(env, Int(actions[t]))
        tape[t, :] = UInt8.(collect(get_ram(env)))
    end
    return tape
end

# ============================================================================
# Action traces (shared shape with A4).
# ============================================================================
# Action codes (jutari src/io/IO.jl): NOOP=0 FIRE=1 UP=2 RIGHT=3 LEFT=4 DOWN=5
# RIGHTFIRE=11 LEFTFIRE=12.
#
# The ORACLE / scored trace is the deterministic FIRE-then-NOOP start — it stays
# inside the conformance window and the bit-exact re-run is byte-identical.
oracle_actions(total::Integer) = vcat(fill(1, 4), fill(0, max(0, total - 4)))

# The A7 (descriptive) dim-reduction trace is ACTIVE (FIRE + periodic directional
# fire) so the candidate cells actually MOVE (an all-NOOP tail leaves most cells
# static, giving PCA/NMF no variance to decompose). The dim-reduction is a
# DESCRIPTIVE statistic of a deterministic jutari trajectory (the recorder's
# documented purpose) — not a bit-exact-vs-xitari claim — so it legitimately uses
# the livelier replay. The scored-vs-oracle claims (F/S/M, the oracle importance)
# stay on the oracle trace, strictly inside the conformance window.
function active_actions(total::Integer)
    acts = Vector{Int}(undef, max(0, total))
    for t in 1:length(acts)
        acts[t] = t <= 4      ? 1  :
                  t % 4 == 0  ? 11 :
                  t % 4 == 2  ? 12 :
                                1
    end
    return acts
end

# Full directional maze trace for games where FIRE is a no-op (Ms. Pac-Man).
function dir_actions(total::Integer)
    acts = Vector{Int}(undef, max(0, total))
    cyc = (3, 2, 4, 5)              # RIGHT, UP, LEFT, DOWN
    for t in 1:length(acts)
        acts[t] = t <= 8 ? 1 : cyc[((t - 9) % 4) + 1]
    end
    return acts
end

trace_actions(kind::AbstractString, total::Integer) =
    kind == "dir" ? dir_actions(total) : active_actions(total)

# ============================================================================
# Candidate cells (E2-1 import) — the "units" / known variables A7 matches against.
# ============================================================================
struct Candidate
    ram_index::Int
    concept::String
    family::String        # the true-variable-group label (concept family)
end

"""The concept *family* = the true variable group a cell belongs to (same
normalisation as A4 so e.g. enemy_x/enemy_y/enemy.xy → `enemy`, player_score/score
→ `score`). Cells sharing a family are ONE known game variable — a latent component
that loads on exactly one family is "matched" to a known game variable."""
function concept_family(concept::AbstractString)
    s = lowercase(strip(string(concept)))
    isempty(s) && return "(unnamed)"
    s = split(s, '.')[1]
    s = replace(s, r"^enemy_(sue|inky|pinky|blinky)" => "enemy")
    s = replace(s, r"^(blinky|inky|pinky|sue)" => "enemy")
    s = replace(s, r"^enemy[0-9]+" => "enemy")
    s = replace(s, r"^g[0-9]+$" => "enemy")
    s = replace(s, r"^ghosts?" => "enemy")
    for suf in ("_position_x", "_position_y", "_direction", "_meter_value",
                "_value", "_count", "_amount", "_column", "_map", "_bit_map",
                "_collected", "_eaten", "_x", "_y", "_xy", "_wh", "_orientation",
                "_prev_y", "_missile_x", "_missiles_x", "_missile_direction")
        endswith(s, suf) && (s = s[1:end-length(suf)])
    end
    s = replace(s, r"_[0-9]+$" => "")
    s = replace(s, r"[0-9]+$" => "")
    s = rstrip(s, '_')
    s in ("num_lives", "n_lives", "last_lives") && (s = "lives")
    s in ("player_score",) && (s = "score")
    s in ("enemies",) && (s = "enemy")
    s == "" && (s = "(unnamed)")
    return s
end

"""Read the candidate RAM cells for `game`, de-duplicated by ram_index in file
order, each tagged with its true-variable-group family."""
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
            push!(out, Candidate(idx, concept, concept_family(concept)))
        end
    end
    return out
end

# ============================================================================
# PCA (SVD) of the (T, 128) RAM trajectory.
# ============================================================================
struct PCAResult
    n_frames::Int
    n_cells::Int
    n_varying::Int
    var_explained::Vector{Float64}   # per-component fraction of total variance
    cum_var::Vector{Float64}
    V::Matrix{Float64}               # (128, r) loadings (right singular vectors)
    mu::Vector{Float64}              # column means (centering)
end

"""PCA via SVD of the centred (T,128) tape. Returns per-component variance fractions
and the loading matrix V (right singular vectors; V[:,j] is component j's loading
over the 128 RAM cells)."""
function run_pca(tape::AbstractMatrix)
    nram = min(RAM_SIZE, size(tape, 2))
    X = Float64.(tape[:, 1:nram])                 # (T, 128)
    T = size(X, 1)
    mu = vec(Statistics.mean(X; dims = 1))
    Xc = X .- reshape(mu, 1, :)
    col_var = vec(Statistics.var(X; dims = 1, corrected = false))
    n_varying = count(>(0.0), col_var)
    F = svd(Xc; full = false)
    sv = F.S
    var_comp = sv .^ 2
    total = sum(var_comp)
    var_explained = total == 0 ? zeros(length(sv)) : var_comp ./ total
    cum = cumsum(var_explained)
    return PCAResult(T, nram, n_varying, var_explained, cum, Matrix(F.V), mu)
end

# ============================================================================
# NMF of the (T, 128) RAM trajectory (RAM bytes are NON-NEGATIVE → valid input).
# ============================================================================
# Lee & Seung (2001) multiplicative-update NMF, dependency-free (no package added to
# the shared jutari env). X ≈ W H with W (T×r) ≥ 0, H (r×128) ≥ 0 (the r non-negative
# "parts" — each H row is a latent component's loading over the 128 RAM cells).
struct NMFResult
    rank::Int
    n_iters::Int
    W::Matrix{Float64}               # (T, r) activations
    H::Matrix{Float64}               # (r, 128) loadings (non-negative parts)
    recon_rel_err::Float64           # ||X - WH||_F / ||X||_F at the chosen rank
    recon_rel_err_by_rank::Vector{Float64}  # rel-err using top-1..rank components
    component_energy::Vector{Float64}  # per-component ||W[:,k]||·||H[k,:]|| share
end

"""Lee & Seung multiplicative-update NMF of a non-negative X (T×n) into W (T×r) and
H (r×n). Deterministic given `seed`. Returns W, H and the relative Frobenius
reconstruction error. ε guards the divisions; constant-zero columns stay zero."""
function _nmf(X::AbstractMatrix{<:Real}, r::Integer; iters = 300, seed = 0, ε = 1e-9)
    T, n = size(X)
    Xf = Float64.(X)
    rng = Random.MersenneTwister(seed)
    # init in [0,1)·scale so the parts can grow into the data magnitude.
    scale = max(1.0, Statistics.mean(Xf) + ε)
    W = scale .* (rand(rng, T, r) .+ ε)
    H = scale .* (rand(rng, r, n) .+ ε)
    for _ in 1:iters
        # H <- H .* (Wᵀ X) ./ (Wᵀ W H)
        WtX = W' * Xf
        WtWH = (W' * W) * H .+ ε
        H .= H .* (WtX ./ WtWH)
        # W <- W .* (X Hᵀ) ./ (W H Hᵀ)
        XHt = Xf * H'
        WHHt = W * (H * H') .+ ε
        W .= W .* (XHt ./ WHHt)
    end
    return W, H
end

"""Run NMF at the requested rank, plus a reconstruction-vs-rank curve (rel error
using the top-1..rank components, ordered by component energy). component_energy[k]
= the share of the rank-r reconstruction Frobenius mass carried by component k."""
function run_nmf(tape::AbstractMatrix, rank::Integer; iters = 300, seed = 0)
    nram = min(RAM_SIZE, size(tape, 2))
    X = Float64.(tape[:, 1:nram])                 # (T, 128), already non-negative
    Xnorm = norm(X)
    r = min(rank, size(X, 1), nram)
    r = max(r, 1)
    W, H = _nmf(X, r; iters = iters, seed = seed)
    R = W * H
    rel = Xnorm == 0 ? 0.0 : norm(X .- R) / Xnorm
    # per-component energy = ||W[:,k] H[k,:]||_F (a rank-1 outer product's mass)
    energy = Float64[norm(W[:, k]) * norm(H[k, :]) for k in 1:r]
    ord = sortperm(energy; rev = true)
    # reconstruction-vs-rank: cumulative top-m components (by energy)
    rel_by_rank = zeros(Float64, r)
    Rm = zeros(Float64, size(X))
    for (m, k) in enumerate(ord)
        Rm .+= W[:, k] * H[k, :]'
        rel_by_rank[m] = Xnorm == 0 ? 0.0 : norm(X .- Rm) / Xnorm
    end
    # reorder H / energy by descending energy so component 1 is the most important
    H = H[ord, :]
    energy = energy[ord]
    W = W[:, ord]
    eshare = sum(energy) == 0 ? zeros(r) : energy ./ sum(energy)
    return NMFResult(r, iters, W, H, rel, rel_by_rank, eshare)
end

# ============================================================================
# Known signals (experiment_design.md §4 A7: "vs known signals (clock, R/W, vsync)")
# ============================================================================
# Beyond the game variables (candidate concept families), A7 builds the structural
# known signals a dim-reduction *should* surface if it recovered the chip's bookkeeping:
#   * CLOCK — a monotone frame counter (the dominant epiphenomenal drift).
#   * R/W (write activity) — per-cell, how often the cell's value changes frame-to-
#     frame (a proxy for "this cell is being written"); the population R/W signal is
#     the per-frame count of changed cells.
#   * VSYNC phase — the periodic ~per-frame oscillation (a sinusoid at the dominant
#     auto-correlation period of the population write-activity) — the chip's frame
#     cadence.
# A latent component "matches a known signal" iff its per-frame ACTIVATION time
# series correlates strongly (|corr| ≥ purity) with one of these — i.e. the
# component IS the clock / the write cadence rather than a game variable.
struct KnownSignals
    names::Vector{String}            # ["clock", "rw_activity", "vsync_phase"]
    sig::Matrix{Float64}             # (T, 3) per-frame signals
end

function known_signals(tape::AbstractMatrix)
    T = size(tape, 1)
    X = Float64.(tape)
    clock = Float64.(1:T)
    # population write activity: #cells whose value changed vs the previous frame.
    rw = zeros(Float64, T)
    for t in 2:T
        rw[t] = count(j -> X[t, j] != X[t-1, j], 1:size(X, 2))
    end
    rw[1] = T >= 2 ? rw[2] : 0.0
    # vsync phase: a sinusoid at the dominant period of the (mean-removed) write
    # activity (its strongest auto-correlation lag in 2..T÷2); fall back to period 2.
    period = _dominant_period(rw)
    vsync = Float64[sin(2π * (t - 1) / period) for t in 1:T]
    return KnownSignals(["clock", "rw_activity", "vsync_phase"],
                        hcat(clock, rw, vsync))
end

"""The lag (2..T÷2) of maximum auto-correlation of x (mean-removed) — the dominant
period. Returns 2 if x is constant or too short."""
function _dominant_period(x::AbstractVector)
    T = length(x)
    T < 4 && return 2.0
    xc = x .- Statistics.mean(x)
    s = sum(xc .^ 2)
    s == 0 && return 2.0
    best_lag = 2; best = -Inf
    for lag in 2:(T ÷ 2)
        ac = sum(xc[1:T-lag] .* xc[1+lag:T]) / s
        if ac > best; best = ac; best_lag = lag; end
    end
    return Float64(best_lag)
end

# ============================================================================
# Matched-component fraction (the A7 head metric).
# ============================================================================
# For each latent component (PCA loading V[:,j] over 128 cells, or NMF loading
# H[j,:]) we ask: is it CONCENTRATED on a single known variable?
#   (a) GAME-VARIABLE match — the squared-loading mass over candidate cells is
#       dominated by ONE concept family above `purity` (the leading family's share
#       of the component's mass that lies on candidate cells), AND the candidate
#       cells carry a real share (≥ candidate_floor) of the WHOLE component mass.
#   (b) KNOWN-SIGNAL match — the component's per-frame ACTIVATION (the score time
#       series) correlates |·| ≥ purity with the clock / R-W activity / vsync phase.
# A component is "matched" if (a) OR (b). matched-component fraction =
# #matched / #components over the top-k. This is the pilot's matched-component
# fraction (which used only a binary causal-cell mass test) generalised to the §4
# "vs known signals" spec + a single-variable purity test.
struct MatchResult
    method::String                   # "pca" | "nmf"
    topk::Int
    # per-component, the matched known variable (or "" / "(none)") + its purity
    matched_label::Vector{String}
    matched_purity::Vector{Float64}
    matched_kind::Vector{String}     # "game_var" | "known_signal" | "(none)"
    matched::Vector{Bool}
    matched_component_fraction::Float64
    # breakdown for the §R record
    n_matched_game_var::Int
    n_matched_known_signal::Int
end

"""Per-frame activation (score) time series of component j for the method.
For PCA: the projection of the centred tape onto V[:,j]. For NMF: W[:,j]."""
function component_activations(method::AbstractString, tape::AbstractMatrix,
                              pca::PCAResult, nmf::NMFResult)
    nram = min(RAM_SIZE, size(tape, 2))
    if method == "pca"
        X = Float64.(tape[:, 1:nram]) .- reshape(pca.mu, 1, :)
        return X * pca.V                                # (T, r) scores
    else
        return nmf.W                                   # (T, r) activations
    end
end

"""The component's loading over the 128 cells (squared, so it's a non-negative mass).
PCA: V[:,j].^2. NMF: H[j,:].^2 (already non-negative; square for a comparable mass)."""
function component_loading_mass(method::AbstractString, j::Integer,
                                pca::PCAResult, nmf::NMFResult)
    if method == "pca"
        return pca.V[:, j] .^ 2
    else
        return nmf.H[j, :] .^ 2
    end
end

function match_components(method::AbstractString, tape::AbstractMatrix,
                          pca::PCAResult, nmf::NMFResult, cands::Vector{Candidate},
                          known::KnownSignals; topk = 5, purity = 0.5,
                          candidate_floor = 0.25)
    acts = component_activations(method, tape, pca, nmf)   # (T, r)
    r = size(acts, 2)
    k = min(topk, r)
    # candidate cell → family index
    fams = unique([c.family for c in cands])
    fam_of = Dict(c.ram_index => c.family for c in cands)
    cand_idx0 = [c.ram_index for c in cands]              # 0-based RAM indices

    labels = String[]; purities = Float64[]; kinds = String[]; matched = Bool[]
    n_gv = 0; n_ks = 0
    for j in 1:k
        load2 = component_loading_mass(method, j, pca, nmf)   # (128,) mass
        totmass = sum(load2)
        # (a) game-variable match: of the component's mass that lies on candidate
        # cells, is it dominated by ONE family? And do candidate cells carry a real
        # share of the WHOLE component mass?
        cand_mass = sum(load2[idx + 1] for idx in cand_idx0; init = 0.0)
        cand_share = totmass == 0 ? 0.0 : cand_mass / totmass
        fam_mass = Dict(f => 0.0 for f in fams)
        for idx in cand_idx0
            fam_mass[fam_of[idx]] += load2[idx + 1]
        end
        best_fam = ""; best_fam_mass = 0.0
        for (f, m) in fam_mass
            if m > best_fam_mass; best_fam_mass = m; best_fam = f; end
        end
        fam_purity = cand_mass == 0 ? 0.0 : best_fam_mass / cand_mass
        game_match = (cand_share >= candidate_floor) && (fam_purity >= purity)

        # (b) known-signal match: does the component activation track clock/R-W/vsync?
        a = acts[:, j]
        best_sig = ""; best_sig_corr = 0.0
        for (si, nm) in enumerate(known.names)
            c = abs(_pearson(a, known.sig[:, si]))
            if c > best_sig_corr; best_sig_corr = c; best_sig = nm; end
        end
        signal_match = best_sig_corr >= purity

        if game_match && fam_purity >= best_sig_corr
            push!(labels, best_fam); push!(purities, fam_purity)
            push!(kinds, "game_var"); push!(matched, true); n_gv += 1
        elseif signal_match
            push!(labels, best_sig); push!(purities, best_sig_corr)
            push!(kinds, "known_signal"); push!(matched, true); n_ks += 1
        elseif game_match
            push!(labels, best_fam); push!(purities, fam_purity)
            push!(kinds, "game_var"); push!(matched, true); n_gv += 1
        else
            # report the closest miss for transparency
            if best_sig_corr >= cand_share * (best_fam == "" ? 0 : 1) && best_sig_corr > 0
                push!(labels, "(none; closest=$(best_sig)@$(round(best_sig_corr,digits=2)))")
            else
                push!(labels, "(none; cand_share=$(round(cand_share,digits=2)))")
            end
            push!(purities, max(fam_purity * (cand_share >= candidate_floor ? 1 : 0), best_sig_corr))
            push!(kinds, "(none)"); push!(matched, false)
        end
    end
    frac = k == 0 ? 0.0 : count(matched) / k
    return MatchResult(method, k, labels, purities, kinds, matched, frac, n_gv, n_ks)
end

# ============================================================================
# Per-cell RECOVERY score (the A7 importance/recovery CLAIM, for F/S/M).
# ============================================================================
# A dim-reduction surfaces a cell iff the LEADING components load on it. The per-cell
# recovery score = summed squared loading over the top-k components, variance-weighted
# (PCA: weighted by var_explained; NMF: by component energy). High ⇒ the method
# presents this cell as a dominant axis of variation. A FAITHFUL dim-reduction would
# rank the oracle-CAUSAL cells high; the trap is surfacing high-variance epiphenomenal
# cells (the clock-driven counters) instead — present (high variance) ≠ used (causal).
function recovery_scores(method::AbstractString, cands::Vector{Candidate},
                         pca::PCAResult, nmf::NMFResult; topk = 5)
    n = length(cands)
    score = zeros(Float64, n)
    if method == "pca"
        k = min(topk, length(pca.var_explained))
        for (i, c) in enumerate(cands)
            s = 0.0
            for j in 1:k
                s += pca.var_explained[j] * (pca.V[c.ram_index + 1, j]^2)
            end
            score[i] = s
        end
    else
        k = min(topk, nmf.rank)
        for (i, c) in enumerate(cands)
            s = 0.0
            for j in 1:k
                s += nmf.component_energy[j] * (nmf.H[j, c.ram_index + 1]^2)
            end
            score[i] = s
        end
    end
    return score
end

# ============================================================================
# The ORACLE per-cell importance (the TRUE causal footprint — the F ground truth).
# ============================================================================
struct OracleImportance
    importance::Vector{Float64}      # max whole-screen break under do(0)/do(+17)
    held_out::Vector{Float64}        # do(base+37) screen break (the S probe)
end

"""The oracle's |Δy| per candidate cell = max whole-screen break of an exact
do(0)/do(base+17) intervention (the TRUE causal footprint, same as pilot_si.jl /
A2). held_out = the do(base+37) break (a value the importance sweep never used)."""
function oracle_importance(checkpoint, tail::Vector{Int}, cands::Vector{Candidate},
                           at_target::Snapshot, base_snap::Snapshot; verbose = true)
    n = length(cands)
    imp = zeros(Float64, n)
    held = zeros(Float64, n)
    function lesion(idx, value)
        env = deepcopy(checkpoint)
        intervene_ram!(env, idx, value)
        for a in tail; env_step!(env, a); end
        return snapshot(env, length(tail))
    end
    for (i, c) in enumerate(cands)
        bv = Int(at_target.ram[c.ram_index + 1])
        s0  = lesion(c.ram_index, 0)
        s17 = lesion(c.ram_index, (bv + 17) & 0xFF)
        s37 = lesion(c.ram_index, (bv + 37) & 0xFF)
        imp[i]  = max(Float64(count(s0.screen  .!= base_snap.screen)),
                      Float64(count(s17.screen .!= base_snap.screen)))
        held[i] = Float64(count(s37.screen .!= base_snap.screen))
        verbose && println("  A7-oracle [$i/$n] RAM[$(lpad(c.ram_index,3))] " *
                           "$(rpad(c.concept,18)) importance(Δpx)=$(round(imp[i]))")
    end
    return OracleImportance(imp, held)
end

# ============================================================================
# F / S / M (the correctness triad, scored against the oracle importance).
# ============================================================================
struct Triad
    F::Float64; S::Float64; M::Float64
    F_note::String; S_note::String; M_note::String
    method::String
end

function _pearson(a::AbstractVector, b::AbstractVector)
    length(a) < 2 && return 0.0
    sa = Statistics.std(a); sb = Statistics.std(b)
    (sa == 0 || sb == 0) && return 0.0
    c = Statistics.cor(a, b)
    return isfinite(c) ? c : 0.0
end
_spearman(a::AbstractVector, b::AbstractVector) = _pearson(_rank(a), _rank(b))
function _rank(v::AbstractVector)
    n = length(v); p = sortperm(v); r = zeros(Float64, n); i = 1
    while i <= n
        j = i
        while j < n && v[p[j+1]] == v[p[i]]; j += 1; end
        avg = (i + j) / 2
        for k in i:j; r[p[k]] = avg; end
        i = j + 1
    end
    return r
end

"""Score the per-cell recovery map against the oracle importance.
  F — Spearman(recovery score, oracle importance): do leading components surface
      the CAUSAL cells?
  S — Pearson(recovery score, held-out do(base+37) break): predict an unseen lesion.
  M — 1 − over-claim rate: of cells the method flags "captured" (recovery ≥ the
      top-quantile, i.e. on a leading component), how many are NOT causal."""
function score_triad(method::AbstractString, recovery::Vector{Float64},
                     oracle::OracleImportance; flag_quantile = 0.5)
    F = _spearman(recovery, oracle.importance)
    S = _pearson(recovery, oracle.held_out)
    # "captured" = recovery score in the upper portion (a leading-component cell).
    # threshold = the flag_quantile-quantile of the non-zero recovery scores (so a
    # cell the method genuinely surfaces). If all recovery is 0, nothing is flagged.
    pos = recovery[recovery .> 0.0]
    thr = isempty(pos) ? Inf : _quantile(pos, flag_quantile)
    flagged = recovery .>= thr
    nflag = sum(flagged)
    overclaim = nflag == 0 ? 0.0 :
        count(i -> flagged[i] && oracle.importance[i] == 0.0, 1:length(flagged)) / nflag
    M = 1.0 - overclaim
    return Triad(F, S, M,
        "Spearman($(method) per-cell recovery score, oracle causal importance) over candidate cells",
        "Pearson($(method) recovery, held-out do(base+37) screen break) — sufficiency on the bit-exact re-run",
        "1 − over-claim rate: fraction of $(method)-surfaced cells (recovery ≥ q$(flag_quantile)) the oracle says are non-causal (variance≠causal trap)",
        method)
end

function _quantile(v::AbstractVector, q::Real)
    isempty(v) && return 0.0
    s = sort(collect(Float64.(v)))
    n = length(s)
    n == 1 && return s[1]
    pos = clamp(q, 0.0, 1.0) * (n - 1) + 1
    lo = floor(Int, pos); hi = ceil(Int, pos)
    lo == hi && return s[lo]
    return s[lo] + (pos - lo) * (s[hi] - s[lo])
end

# ============================================================================
# Drive one game.
# ============================================================================
struct GameResult
    game::String
    target_frame::Int
    horizon::Int
    traj_frames::Int
    trace::String
    in_window::Bool
    seed::Int
    bit_exact::Bool
    cands::Vector{Candidate}
    pca::PCAResult
    nmf::NMFResult
    pca_match::MatchResult
    nmf_match::MatchResult
    pca_recovery::Vector{Float64}
    nmf_recovery::Vector{Float64}
    oracle::OracleImportance
    pca_triad::Triad
    nmf_triad::Triad
    topk::Int
    purity::Float64
end

function compute_game(game::AbstractString; target_frame = nothing, horizon = nothing,
                      traj_frames = nothing, topk = 5, nmf_rank = 8, nmf_iters = 300,
                      purity = 0.5, seed = 0, verbose = true)
    cfg = game_cfg(game)
    target_frame = target_frame === nothing ? cfg.target_frame : target_frame
    horizon      = horizon      === nothing ? cfg.horizon      : horizon
    traj_frames  = traj_frames  === nothing ? cfg.traj_frames  : traj_frames
    trace        = cfg.trace
    in_window    = cfg.in_window
    total = target_frame + horizon
    oacts = oracle_actions(total)
    tail = Int.(oacts[target_frame + 1 : target_frame + horizon])

    # 1) bit-exact baseline — two fresh boots+replays must be byte-identical.
    verbose && println("[A7:$game] asserting bit-exactness (2 fresh boots+replays to f$total)...")
    a = a7_fresh_baseline(game, oacts, total)
    b = a7_fresh_baseline(game, oacts, total)
    bit_exact = (a.ram == b.ram) && (a.screen == b.screen)
    bit_exact || error("bit-exact re-run FAILED for $game to f$total — refusing to score")
    verbose && println("[A7:$game] bit-exact re-run: PASS")

    # 2) one checkpoint at the intervention frame (boot+to-target paid once).
    checkpoint = a7_boot_replay(game, oacts, target_frame)
    at_target = a7_continue_from(checkpoint, Int[])
    base_snap = a7_continue_from(checkpoint, tail)

    cand_path = resolve_candidates(game)
    cands = load_candidates(cand_path)
    isempty(cands) && error("no candidate cells for $game (candidates file missing/empty: $cand_path)")
    verbose && println("[A7:$game] candidates: $(cand_path) ($(length(cands)) cells, " *
                       "$(length(unique([c.family for c in cands]))) known game variables)")

    # 3) descriptive dim-reduction over an ACTIVE trajectory (cells must move).
    verbose && println("[A7:$game] recording $(traj_frames)-frame active ($(trace)) RAM trajectory...")
    tape = a7_record_ram(game, traj_frames, trace_actions(trace, traj_frames))

    verbose && println("[A7:$game] PCA (SVD) + NMF (rank $(nmf_rank), $(nmf_iters) iters) of the (T,128) tape...")
    pca = run_pca(tape)
    nmf = run_nmf(tape, nmf_rank; iters = nmf_iters, seed = seed)
    known = known_signals(tape)

    # 4) matched-component fraction vs known variables (game vars + clock/R-W/vsync).
    pca_match = match_components("pca", tape, pca, nmf, cands, known; topk = topk, purity = purity)
    nmf_match = match_components("nmf", tape, pca, nmf, cands, known; topk = topk, purity = purity)

    # 5) per-cell recovery score (the importance/recovery claim) for F/S/M.
    pca_recovery = recovery_scores("pca", cands, pca, nmf; topk = topk)
    nmf_recovery = recovery_scores("nmf", cands, pca, nmf; topk = topk)

    # 6) the ORACLE per-cell importance (TRUE causal footprint — the F ground truth).
    verbose && println("[A7:$game] oracle per-cell causal importance over $(length(cands)) cells...")
    oracle = oracle_importance(checkpoint, tail, cands, at_target, base_snap; verbose = verbose)

    # 7) F / S / M triad for BOTH methods (recovery map vs the oracle importance).
    pca_triad = score_triad("pca", pca_recovery, oracle)
    nmf_triad = score_triad("nmf", nmf_recovery, oracle)

    if verbose
        println("[A7:$game] ---- A7 scores ----")
        println("[A7:$game]   PCA top1 var=$(round(pca.var_explained[1],digits=3)) " *
                "cum(top5)=$(round(pca.cum_var[min(5,end)],digits=3)) " *
                "matched-frac=$(round(pca_match.matched_component_fraction,digits=3)) " *
                "(gv=$(pca_match.n_matched_game_var) sig=$(pca_match.n_matched_known_signal))")
        println("[A7:$game]   NMF rank=$(nmf.rank) rel-err=$(round(nmf.recon_rel_err,digits=3)) " *
                "matched-frac=$(round(nmf_match.matched_component_fraction,digits=3)) " *
                "(gv=$(nmf_match.n_matched_game_var) sig=$(nmf_match.n_matched_known_signal))")
        println("[A7:$game]   PCA TRIAD F=$(round(pca_triad.F,digits=3)) S=$(round(pca_triad.S,digits=3)) M=$(round(pca_triad.M,digits=3))")
        println("[A7:$game]   NMF TRIAD F=$(round(nmf_triad.F,digits=3)) S=$(round(nmf_triad.S,digits=3)) M=$(round(nmf_triad.M,digits=3))")
    end

    return GameResult(game, target_frame, horizon, traj_frames, trace, in_window,
                      seed, bit_exact, cands, pca, nmf, pca_match, nmf_match,
                      pca_recovery, nmf_recovery, oracle, pca_triad, nmf_triad,
                      topk, purity)
end

# ============================================================================
# Self-check (DoD) — the scoring contract is sound; results are non-fabricated.
# ============================================================================
"""
    selftest(r::GameResult) -> Bool

Asserts A7's load-bearing claims (the same contract shape as pilot_si.selftest):

  (BIT-EXACT) the baseline re-run is byte-identical (every importance Δ is a clean
    causal effect on the real ROM).

  (ORACLE GROUND-TRUTH ANCHORED) the oracle finds ≥1 genuinely causal candidate
    cell (some lesion moves the screen) — else F is uninformative.

  (POSITIVE CONTROL) scoring the ORACLE'S OWN importance map as the candidate
    method yields Spearman F == 1 — the harness rewards a perfectly-faithful map,
    so a sub-1 PCA/NMF recovery F is a real measurement.

  (PCA WELL-FORMED) variance fractions ≥ 0 and sum to ≈1.

  (NMF WELL-FORMED) W,H ≥ 0; reconstruction rel-error ∈ [0, ~1+ε]; the recon-vs-rank
    curve is non-increasing (more components never reconstruct worse).

  (MATCHED-FRACTION RANGES) ∈ [0,1] for both methods.

  (TRIAD RANGES) F,S ∈ [−1,1]; M ∈ [0,1]; finite.

Throws on a contract violation."""
function selftest(r::GameResult)
    @assert r.bit_exact "bit-exact baseline re-run failed for $(r.game)"
    n_causal = count(>(0.0), r.oracle.importance)
    @assert n_causal >= 1 "oracle found NO causal candidate cell for $(r.game) at this " *
        "state — uninformative; pick a livelier target frame"
    self_F = _spearman(r.oracle.importance, r.oracle.importance)
    @assert self_F > 0.999 "harness broken: oracle-as-method Spearman != 1 ($self_F) for $(r.game)"

    @assert all(>=(-1e-12), r.pca.var_explained) "negative PCA variance fraction"
    @assert abs(sum(r.pca.var_explained) - 1.0) < 1e-6 "PCA variance fractions don't sum to 1"

    @assert all(>=(-1e-9), r.nmf.W) "NMF W has a negative entry"
    @assert all(>=(-1e-9), r.nmf.H) "NMF H has a negative entry"
    @assert 0.0 - 1e-9 <= r.nmf.recon_rel_err <= 1.0 + 1e-6 "NMF rel-error out of [0,1]: $(r.nmf.recon_rel_err)"
    for m in 2:length(r.nmf.recon_rel_err_by_rank)
        @assert r.nmf.recon_rel_err_by_rank[m] <= r.nmf.recon_rel_err_by_rank[m-1] + 1e-6 "NMF recon-vs-rank not non-increasing at $m"
    end

    @assert 0.0 <= r.pca_match.matched_component_fraction <= 1.0 "PCA matched-fraction out of [0,1]"
    @assert 0.0 <= r.nmf_match.matched_component_fraction <= 1.0 "NMF matched-fraction out of [0,1]"

    for tr in (r.pca_triad, r.nmf_triad)
        @assert -1.0 - 1e-9 <= tr.F <= 1.0 + 1e-9 "$(tr.method) F out of [-1,1]: $(tr.F)"
        @assert -1.0 - 1e-9 <= tr.S <= 1.0 + 1e-9 "$(tr.method) S out of [-1,1]: $(tr.S)"
        @assert  0.0 - 1e-9 <= tr.M <= 1.0 + 1e-9 "$(tr.method) M out of [0,1]: $(tr.M)"
    end

    println("[A7:$(r.game)] SELF-CHECK PASS:")
    println("[A7:$(r.game)]   bit-exact baseline re-run: $(r.bit_exact)")
    println("[A7:$(r.game)]   oracle causal cells: $n_causal/$(length(r.cands)) " *
            "(positive control F = $(round(self_F,digits=3)))")
    println("[A7:$(r.game)]   PCA matched-frac=$(round(r.pca_match.matched_component_fraction,digits=3)) " *
            "F=$(round(r.pca_triad.F,digits=3)) S=$(round(r.pca_triad.S,digits=3)) M=$(round(r.pca_triad.M,digits=3))")
    println("[A7:$(r.game)]   NMF matched-frac=$(round(r.nmf_match.matched_component_fraction,digits=3)) " *
            "rel-err=$(round(r.nmf.recon_rel_err,digits=3)) " *
            "F=$(round(r.nmf_triad.F,digits=3)) S=$(round(r.nmf_triad.S,digits=3)) M=$(round(r.nmf_triad.M,digits=3))")
    return true
end

# ============================================================================
# Persist (SPEC §R) — JSON record + sibling .npz; file_scope A7_*.
# ============================================================================
_git_commit() = try
    strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
catch
    "unknown"
end
_json_num(x::Real) = isfinite(x) ? Float64(x) : nothing

const CANDIDATES_DIR_REL = joinpath("tools", "xai_study", "t3", "out")
function resolve_candidates(game::AbstractString)
    here = normpath(joinpath(@__DIR__, "..", "..", ".."))
    for base in (here, get(ENV, "XAI_PRIMARY_REPO", ""),
                 "/Users/maier/Documents/code/UnderstandingVCS")
        base == "" && continue
        p = joinpath(base, CANDIDATES_DIR_REL, "candidates_$(game).json")
        isfile(p) && return p
    end
    return nothing
end

_match_record(m::MatchResult) = Dict{String,Any}(
    "method" => m.method,
    "topk" => m.topk,
    "matched_component_fraction" => _json_num(m.matched_component_fraction),
    "n_matched_game_var" => m.n_matched_game_var,
    "n_matched_known_signal" => m.n_matched_known_signal,
    "per_component_matched_label" => m.matched_label,
    "per_component_matched_purity" => [_json_num(x) for x in m.matched_purity],
    "per_component_matched_kind" => m.matched_kind,
)

function write_game(r::GameResult; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    stem = "A7_$(r.game)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path  = joinpath(out_dir, stem * ".npz")

    cell_names = ["RAM[$(c.ram_index)]:$(c.concept)" for c in r.cands]
    families = [c.family for c in r.cands]
    rec = Dict{String,Any}(
        "paper" => "P2",
        "phase" => "phaseA_kording",
        "method" => "A7_dim_reduction(NMF+PCA)",
        "game" => r.game,
        "state" => "f$(r.target_frame)+$(r.horizon)",
        "target_output" => "latent-components",
        # headline scalar (SPEC §R value/metric_name): the NMF matched-component
        # fraction (the prompt's question — does NMF beat the pilot's PCA ≈0?).
        "metric_name" => "A7_nmf_matched_component_fraction_vs_known_vars",
        "value" => _json_num(r.nmf_match.matched_component_fraction),
        "stderr" => nothing,
        "ci" => nothing,
        "n" => length(r.cands),
        "seed" => r.seed,
        "where" => "local",
        "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene@$(r.game)#per-cell-causal-importance(screen-break)",
        "timestamp" => string(round(Int, time())),
        "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            "substrate" => "jutari (Julia, HARD) — real-ROM bit-exact path; the " *
                "importance oracle uses EXACT interventions re-run on the true ROM.",
            "bit_exact_rerun" => r.bit_exact,
            "trajectory_trace" => r.trace == "dir" ?
                "directional maze trace (FIRE warmup + cyclic UP/DOWN/LEFT/RIGHT)" :
                "active trace (4×FIRE + periodic RIGHTFIRE/LEFTFIRE)",
            "trajectory_frames" => r.traj_frames,
            "scored_in_conformance_window" => r.in_window,
            "conformance_note" => r.in_window ?
                "scored oracle frame f$(r.target_frame)+$(r.horizon) is inside the " *
                "Paper-1 60-frame screen conformance window (jutari↔xitari bit-exact)." :
                "scored oracle frame f$(r.target_frame)+$(r.horizon) is BEYOND the strict " *
                "60-frame screen conformance window — the game only reaches live play " *
                "later. HONEST descriptive jutari result (the bit-exact jutari re-run IS " *
                "asserted), not an xitari-parity claim; A is the calibration baseline.",
            "candidate_cells" => cell_names,
            "candidate_ram_indices" => [c.ram_index for c in r.cands],
            "candidate_concept_family" => families,
            "n_known_game_variables" => length(unique(families)),
            "topk_components" => r.topk,
            "purity_threshold" => r.purity,
            # ---- PCA ----
            "pca" => Dict{String,Any}(
                "n_frames" => r.pca.n_frames,
                "n_cells" => r.pca.n_cells,
                "n_varying_cells" => r.pca.n_varying,
                "var_explained_top" => r.pca.var_explained[1:min(8, end)],
                "cum_var_top" => r.pca.cum_var[1:min(8, end)],
                "matched_components" => _match_record(r.pca_match),
                "note" => "PCA (SVD of the centred (T,128) tape). matched-component " *
                    "fraction = #top-k PCs concentrated on ONE known variable " *
                    "(a concept family, OR clock/R-W/vsync) / k. The pilot found ≈0 — " *
                    "PCs mix epiphenomenal high-variance structure across cells.",
            ),
            # ---- NMF (the extension over the pilot) ----
            "nmf" => Dict{String,Any}(
                "rank" => r.nmf.rank,
                "n_iters" => r.nmf.n_iters,
                "recon_rel_frobenius_err" => _json_num(r.nmf.recon_rel_err),
                "recon_rel_err_by_rank" => [_json_num(x) for x in r.nmf.recon_rel_err_by_rank],
                "component_energy_share" => [_json_num(x) for x in r.nmf.component_energy],
                "matched_components" => _match_record(r.nmf_match),
                "note" => "NMF (Lee & Seung multiplicative updates; RAM bytes are " *
                    "non-negative). matched-component fraction tests whether the " *
                    "non-negative PARTS align with single known variables better than " *
                    "PCA's signed components do — the prompt's NMF-vs-PCA question.",
            ),
            # ---- F/S/M (BOTH methods, recovery map vs the oracle) ----
            "triad_pca" => Dict{String,Any}(
                "F" => _json_num(r.pca_triad.F), "F_note" => r.pca_triad.F_note,
                "S" => _json_num(r.pca_triad.S), "S_note" => r.pca_triad.S_note,
                "M" => _json_num(r.pca_triad.M), "M_note" => r.pca_triad.M_note,
            ),
            "triad_nmf" => Dict{String,Any}(
                "F" => _json_num(r.nmf_triad.F), "F_note" => r.nmf_triad.F_note,
                "S" => _json_num(r.nmf_triad.S), "S_note" => r.nmf_triad.S_note,
                "M" => _json_num(r.nmf_triad.M), "M_note" => r.nmf_triad.M_note,
                "interpretation" => "Phase A is the calibration baseline " *
                    "(experiment_design.md §4): dim-reduction surfaces high-VARIANCE " *
                    "axes, which are NOT the causal cells (variance ≠ causal role — " *
                    "present ≠ used). F<1 + low matched-component fraction quantify the " *
                    "departure from the true causal structure.",
            ),
            "oracle_importance" => Dict{String,Any}(
                "importance_per_cell" =>
                    Dict(cell_names[i] => r.oracle.importance[i] for i in 1:length(cell_names)),
                "n_causal_cells" => count(>(0.0), r.oracle.importance),
                "note" => "oracle importance(i) = max whole-screen break of an EXACT " *
                    "do(cell_i:=0)/do(cell_i:=base+17) intervention re-run on the real " *
                    "ROM (the §1 oracle restricted to the candidate cell + screen). " *
                    "The SAME map A2 / pilot_si.jl score against — the TRUE causal role.",
            ),
            "scales_to_cluster_via" =>
                "tools/cluster/xai_*.sbatch (E0-3) — the full A1..A8 battery over the " *
                "core+breadth set; the forward is bit-exact to this HARD path.",
        ),
    )
    open(json_path, "w") do io; JSON.print(io, rec, 2); end

    write_npz(npz_path, Dict(
        "candidate_ram_indices" => Int64[c.ram_index for c in r.cands],
        "pca_var_explained"     => r.pca.var_explained,
        "pca_cum_var"           => r.pca.cum_var,
        "pca_loadings"          => r.pca.V,                 # (128, r)
        "nmf_H_loadings"        => r.nmf.H,                 # (rank, 128)
        "nmf_W_activations"     => r.nmf.W,                 # (T, rank)
        "nmf_recon_rel_err_by_rank" => r.nmf.recon_rel_err_by_rank,
        "nmf_component_energy"  => r.nmf.component_energy,
        "pca_matched_bool"      => Int64[r.pca_match.matched[j] ? 1 : 0 for j in 1:length(r.pca_match.matched)],
        "nmf_matched_bool"      => Int64[r.nmf_match.matched[j] ? 1 : 0 for j in 1:length(r.nmf_match.matched)],
        "pca_recovery"          => r.pca_recovery,
        "nmf_recovery"          => r.nmf_recovery,
        "oracle_importance"     => r.oracle.importance,
        "oracle_held_out"       => r.oracle.held_out,
        # [pca_F,pca_S,pca_M, nmf_F,nmf_S,nmf_M, pca_matched_frac, nmf_matched_frac]
        "scores"                => Float64[r.pca_triad.F, r.pca_triad.S, r.pca_triad.M,
                                           r.nmf_triad.F, r.nmf_triad.S, r.nmf_triad.M,
                                           r.pca_match.matched_component_fraction,
                                           r.nmf_match.matched_component_fraction],
    ))
    return json_path, npz_path
end

# combined index (a small read-only roll-up; not a §R record itself but handy for
# the leaderboard / a quick PCA-vs-NMF comparison across the core set).
function write_index(results::Vector{GameResult}; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    path = joinpath(out_dir, "A7_dimred_all.json")
    rows = Dict{String,Any}[]
    for r in results
        push!(rows, Dict{String,Any}(
            "game" => r.game,
            "state" => "f$(r.target_frame)+$(r.horizon)",
            "in_window" => r.in_window,
            "n_cands" => length(r.cands),
            "n_known_game_vars" => length(unique([c.family for c in r.cands])),
            "n_oracle_causal" => count(>(0.0), r.oracle.importance),
            "pca_top1_var" => _json_num(r.pca.var_explained[1]),
            "pca_cum_var_top5" => _json_num(r.pca.cum_var[min(5, end)]),
            "pca_matched_frac" => _json_num(r.pca_match.matched_component_fraction),
            "nmf_rank" => r.nmf.rank,
            "nmf_recon_rel_err" => _json_num(r.nmf.recon_rel_err),
            "nmf_matched_frac" => _json_num(r.nmf_match.matched_component_fraction),
            "pca_FSM" => [_json_num(r.pca_triad.F), _json_num(r.pca_triad.S), _json_num(r.pca_triad.M)],
            "nmf_FSM" => [_json_num(r.nmf_triad.F), _json_num(r.nmf_triad.S), _json_num(r.nmf_triad.M)],
        ))
    end
    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseA_kording", "method" => "A7_dim_reduction(NMF+PCA)",
        "commit" => _git_commit(), "timestamp" => string(round(Int, time())),
        "n_games" => length(results),
        "headline" => "PCA vs NMF matched-component fraction (does NMF beat the pilot's " *
            "PCA ≈ 0?) + per-cell recovery F/S/M vs the exact intervention oracle.",
        "games" => rows,
    )
    open(path, "w") do io; JSON.print(io, rec, 2); end
    return path
end

# ============================================================================
# CLI
# ============================================================================
function main(args = ARGS)
    games = copy(CORE_GAMES)
    target_frame = nothing; horizon = nothing; traj_frames = nothing
    topk = 5; nmf_rank = 8; nmf_iters = 300; purity = 0.5; seed = 0
    selftest_only = false
    i = 1
    while i <= length(args)
        a = args[i]
        if     a == "--games";        games = String.(split(args[i+1], ",")); i += 2
        elseif a == "--game";         games = [args[i+1]]; i += 2
        elseif a == "--target-frame"; target_frame = parse(Int, args[i+1]); i += 2
        elseif a == "--horizon";      horizon = parse(Int, args[i+1]); i += 2
        elseif a == "--traj-frames";  traj_frames = parse(Int, args[i+1]); i += 2
        elseif a == "--topk";         topk = parse(Int, args[i+1]); i += 2
        elseif a == "--nmf-rank";     nmf_rank = parse(Int, args[i+1]); i += 2
        elseif a == "--nmf-iters";    nmf_iters = parse(Int, args[i+1]); i += 2
        elseif a == "--purity";       purity = parse(Float64, args[i+1]); i += 2
        elseif a == "--seed";         seed = parse(Int, args[i+1]); i += 2
        elseif a == "--selftest";     selftest_only = true; i += 1
        else; i += 1
        end
    end
    println("[A7] games=$(join(games, ",")) target_frame=$target_frame horizon=$horizon " *
            "traj_frames=$traj_frames topk=$topk nmf_rank=$nmf_rank purity=$purity " *
            "seed=$seed (jutari/Julia path)")

    if selftest_only
        g = games[1]
        r = compute_game(g; target_frame = target_frame, horizon = horizon,
                         traj_frames = traj_frames, topk = topk, nmf_rank = nmf_rank,
                         nmf_iters = nmf_iters, purity = purity, seed = seed)
        selftest(r)
        println("[A7] --selftest: passed on $g, not writing artifact.")
        return 0
    end

    ok = String[]; failed = Tuple{String,String}[]; results = GameResult[]
    for g in games
        try
            r = compute_game(g; target_frame = target_frame, horizon = horizon,
                             traj_frames = traj_frames, topk = topk, nmf_rank = nmf_rank,
                             nmf_iters = nmf_iters, purity = purity, seed = seed)
            selftest(r)
            json_path, npz_path = write_game(r)
            println("[A7] wrote $json_path")
            println("[A7] arrays  $npz_path")
            push!(ok, g); push!(results, r)
        catch err
            msg = sprint(showerror, err)
            println("[A7] !! game $g FAILED (scoring the rest, not fabricating): " *
                    first(split(msg, '\n')))
            push!(failed, (g, first(split(msg, '\n'))))
        end
    end
    if !isempty(results)
        idx = write_index(results)
        println("[A7] wrote index $idx")
    end
    println("[A7] ==== summary: $(length(ok))/$(length(games)) games scored ====")
    for g in ok; println("[A7]   OK   $g"); end
    for (g, m) in failed; println("[A7]   FAIL $g — $m"); end
    return 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    A7DimRed.main()
end
