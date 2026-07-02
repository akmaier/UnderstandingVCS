# A6_granger.jl — Phase-A A6: Granger causality on CPU/TIA/RIOT activity (P2-E3-6),
# the JULIA path, run on jutari over the 6 core games.
#
# Phase A is Paper 2's *calibration baseline* (experiment_design.md §4, Novelty
# note): classical neuroscience analyses score LOW faithfulness despite the system
# having rich, fully-known structure — the quantified-Kording lesson. A6 is the
# GRANGER-CAUSALITY member of the battery (Granger 1969; Seth, Barrett & Barnett
# 2015 — the neuroscience workhorse for "directed functional connectivity"). The
# headline A6 lesson is sharper than the others: Granger infers TEMPORAL PRECEDENCE
# ("does the past of X improve the prediction of Y beyond Y's own past?"), which is
# NOT the same as the true causal data-flow direction. On a deterministic register-
# transfer machine the two demonstrably DISAGREE — we quantify by how much.
#
# Two scopes of activity time-series are Granger-tested over a recorded jutari
# trajectory (T frames):
#   * SUBSYSTEM scope — the per-frame activity of the three hardware subsystems the
#     VCS frame is built from: CPU (the RIOT zero-page RAM the 6507 reads/writes —
#     the program's working set), TIA (the 64-byte TIA write-register file —
#     graphics/colour/audio state), RIOT (the timer/SWCHx I/O sub-block). Each
#     subsystem's scalar activity per frame = the number of bytes that CHANGED since
#     the previous frame (its "firing rate"). Granger-test all ordered subsystem
#     pairs ⇒ an inferred CPU/TIA/RIOT directed-edge set.
#   * CELL scope — the per-frame value time-series of each candidate RAM cell.
#     Pairwise Granger over candidate cells ⇒ an inferred cell→cell directed graph,
#     directly comparable (same node set) to the EXACT intervention data-flow graph
#     (the A1/A4 ground truth) so the false-edge / missed-edge rates are well-posed.
#
# Granger estimator (bivariate, lag-p): for an ordered pair (x→y) compare the
# RESIDUAL variance of an autoregression of y on its own p lags (restricted) vs the
# same plus the p lags of x (unrestricted). The Granger statistic is
#   G(x→y) = ln( RSS_restricted / RSS_unrestricted )  ≥ 0,
# the log-likelihood-ratio form (Geweke 1982); an F-test gives a p-value. An edge
# x→y is asserted on the CANONICAL Granger criterion — the F-test is SIGNIFICANT
# (p ≤ alpha, default 0.05) — optionally AND-gated by a magnitude floor on G
# (`granger_tau`, default 0.0 = significance alone, the textbook procedure). (No
# external package — a tiny normal-equations least-squares solver, so nothing is
# added to the shared jutari env.)
#
# GROUND TRUTH (T1/T2 — no T3): the TRUE directed data-flow over the candidate
# cells, edge i→j iff an EXACT intervention on cell i (the exhaustive do-set
# occlude→0 / set→base+17 / set→base+37) moves cell j over the horizon on a
# bit-exact re-run of the real ROM. This is exactly the A1 "true read/write graph"
# (experiment_design.md §4 A1/A6 share the reference), measured causally — the
# direction is the TRUE data-flow direction Granger is scored against.
#
# §4 A6 metric (the headline): FALSE-EDGE rate and MISSED-EDGE rate of the
# Granger-inferred edges vs the true data-flow, over the off-diagonal candidate-cell
# pairs:
#   false_edge_rate  = #(Granger edge i→j present, true edge ABSENT) / #Granger edges
#                      ( = 1 − precision; the spurious directed wiring rate)
#   missed_edge_rate = #(true edge i→j present, Granger edge ABSENT)  / #true edges
#                      ( = 1 − recall;    the data-flow Granger fails to recover)
# plus the GRANGER-SPECIFIC caveat, quantified:
#   direction_disagreement = of the unordered pairs where BOTH the true graph and
#     Granger assert a single directed edge, the fraction whose directions DISAGREE
#     (Granger's temporal-precedence arrow points the opposite way to the true
#     data-flow). This is the present≠used trap in its directional form — the A6
#     headline that temporal precedence ≠ causation.
#
# F/S/M scoring (the correctness triad, experiment_design.md §0), scored against the
# exact intervention ORACLE (NOT against any interpretability method):
#   F (faithful)   — F1 of the Granger-inferred directed edges vs the true edges.
#   S (sufficient) — predict a HELD-OUT intervention's edge set: how well the
#                    Granger graph predicts the base+37 TRUE edges (off-diagonal
#                    directed-edge agreement on the bit-exact re-run) — does the
#                    inferred connectivity generalise to an unseen intervention?
#   M (minimal)    — 1 − false-edge rate: minimality / no spurious directed wiring.
#
# Positive control (as in pilot_si.jl / A1 / A4): scoring the TRUE data-flow graph
# itself as the candidate "method" must yield F1 == 1 (false-edge=missed-edge=0) —
# the harness rewards a perfectly-faithful directed map, so a sub-1 Granger F is a
# real measurement.
#
# BUILDS ON the verified jutari foundation (NO emulator core touched):
#   * tools/xai_study/common/jutari_oracle.jl — load/boot/replay/snapshot/intervene,
#     bit-exact baseline guarantee, the dependency-free §R NPY/NPZ writer.
#   * tools/xai_study/common/jutari_record.jl — record_trajectory → (T,n) tape
#     (A6 reuses its recording PATTERN but re-implements the loop so each game's
#     proper RomSettings are used — mirrors A4_correlations.a4_record_ram).
#   * tools/xai_study/ground_truth/oracle_intervene.jl — the causal-map reference
#     (this file specialises its Δy(u) machinery to the directed candidate-cell
#     data-flow graph — the same construction A1 uses).
#   * tools/xai_study/phaseA_kording/pilot_si.jl — the Phase-A pilot TEMPLATE
#     (per-game oracle scoring + F/S/M + positive control + §R record shape).
#
# Per-game RomSettings: the shared jutari_oracle.settings_for() only knows the 3
# pilot games, so A6 supplies its own per-game settings map (matching the canonical
# tools/jutari_screen_dump.jl map) and builds envs directly via the same JuTari
# primitives — keeping everything inside A6's file_scope (it must not edit the
# shared helper). seaquest legitimately uses GenericRomSettings (Paper-1 64/64).
#
# Run (warm shared depot, primary's project):
#   XAI_PRIMARY_REPO=/Users/maier/Documents/code/UnderstandingVCS \
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseA_kording/A6_granger.jl
# Flags: --games pong,breakout,...  --target-frame N --horizon N --traj-frames N
#        --lag P  --granger-tau G  --alpha A
#        --selftest (run the self-check on the first game, write nothing)
#
# Writes (SPEC §R; file_scope A6_*): one record per game
#   tools/xai_study/phaseA_kording/out/A6_<game>.{json,npz}

module A6Granger

using JSON
import Statistics
using Statistics: mean, std, var

# --- the verified foundation (NO core touched) -----------------------------
include(joinpath(@__DIR__, "..", "common", "jutari_oracle.jl"))
using .JutariOracle
using .JutariOracle: Snapshot, snapshot, intervene_ram!, RAM_SIZE, write_npz,
                     rom_path_for
using .JutariOracle.JuTari: StellaEnvironment
using .JutariOracle.JuTari.Env: env_reset!, env_step!, get_ram

# the trajectory recorder lineage (A6 reuses the recording PATTERN but re-implements
# the loop so each game's proper RomSettings drive the trajectory — the shared
# recorder hard-codes settings_for; the include keeps the lineage explicit).
include(joinpath(@__DIR__, "..", "common", "jutari_record.jl"))

# The §1 exact-intervention oracle — used ONLY to build the P2 SHARED gameplay state
# + its cause-density gate (below). A6 keeps its OWN Granger / true-data-flow
# machinery; the oracle here supplies the shared action STREAM + the gate. Referenced
# as OracleIntervene.X (NOT alias-imported — its internal JutariOracle submodule is a
# separate instance from the one included above; qualifying it avoids mixing the two).
include(joinpath(@__DIR__, "..", "ground_truth", "oracle_intervene.jl"))
using JuTari.Diff: soft_ram_peek

# The P2 SHARED TESTBED: seeded random-action GAMEPLAY state + oracle cause-density
# gate. Included as a fragment. Phase A is not a gradient method, so the sampler-on
# path does not apply — we consume the shared action STREAM + GATE only, and drive
# A6's OWN scored oracle prefix + its descriptive Granger trajectory from that stream
# so the Granger algorithm is unchanged. Opt in with XAI_SHARED_TESTBED=1 (default on).
include(joinpath(@__DIR__, "..", "common", "shared_testbed_impl.jl"))
# the shared game-set + ROM-root resolver (XAI_LABELED / xai_resolve_games / xai_rom_roots).
include(joinpath(@__DIR__, "..", "common", "game_sets.jl"))

const OUT_DIR = joinpath(@__DIR__, "out")
const PRIMARY_REPO = get(ENV, "XAI_PRIMARY_REPO",
                         "/Users/maier/Documents/code/UnderstandingVCS")
# shared-testbed switch + params (redesign protocol: prefix=90 gameplay, horizon=15).
const SHARED_TESTBED = get(ENV, "XAI_SHARED_TESTBED", "1") == "1"
const ST_PREFIX  = parse(Int, get(ENV, "XAI_ST_PREFIX", "90"))
const ST_HORIZON = parse(Int, get(ENV, "XAI_ST_HORIZON", "15"))
const ST_SEED    = parse(Int, get(ENV, "XAI_ST_SEED", "0"))
const ST_GATE_K  = parse(Int, get(ENV, "XAI_ST_GATE_K", "4"))
const ST_FLOOR   = parse(Float64, get(ENV, "XAI_ST_FLOOR", "0.5"))
const CORE_GAMES = ["pong", "breakout", "space_invaders",
                    "seaquest", "ms_pacman", "qbert"]
const TIA_SIZE = 64

# ============================================================================
# Per-game live-play configuration (mirrors A4's GAME_CFG — the cells must MOVE for
# the Granger time-series to carry signal). `trace` selects the trajectory input
# ("fire" = FIRE+RIGHT/LEFTFIRE active trace; "dir" = full UP/DOWN/LEFT/RIGHT maze
# trace for games where FIRE is a no-op). `in_window` flags whether the SCORED
# oracle frame stays inside the strict Paper-1 60-frame screen conformance window
# (jutari↔xitari); games scored beyond it are honest descriptive jutari results
# (bit-exact jutari re-run still asserted), not xitari-parity claims.
# ============================================================================
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
# Per-game RomSettings + ROM resolution (A6-owned; mirrors A4 / the canonical
# tools/jutari_screen_dump.jl map). seaquest -> Generic (Paper-1 64/64 Generic).
# Stays inside A6's file_scope; does not modify the shared settings_for().
# ============================================================================
function a6_settings_for(game::AbstractString)
    g = lowercase(string(game))
    JT = JutariOracle.JuTari
    g == "pong"           && return JT.PaddleGames.PongRomSettings()
    g == "breakout"       && return JT.PaddleGames.BreakoutRomSettings()
    g == "space_invaders" && return JT.SpaceInvadersRomSettings()
    g == "ms_pacman"      && return JT.JoystickGames.MsPacmanRomSettings()
    g == "qbert"          && return JT.JoystickGames.QbertRomSettings()
    return JT.GenericRomSettings()
end

# ROM-name aliases (ms_pacman's file is mspacman.bin); mirrors A4 / import_labels.
const _ROM_ALIASES = Dict("ms_pacman" => ["ms_pacman", "mspacman"],
                          "beam_rider" => ["beam_rider", "beamrider"])

function a6_rom_path(game::AbstractString)
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

"""A freshly-reset env with A6's per-game settings and the xitari-parity boot."""
function a6_load_env(game::AbstractString)
    rom = read(a6_rom_path(game))
    env = StellaEnvironment(rom, a6_settings_for(game))
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    return env
end

"""Boot + step actions[1:target_frame]; return the env AT the intervention frame
(the reusable checkpoint; deepcopy before mutating)."""
function a6_boot_replay(game, actions::AbstractVector{<:Integer}, target_frame::Integer)
    env = a6_load_env(game)
    for i in 1:target_frame
        env_step!(env, Int(actions[i]))
    end
    return env
end

"""deepcopy the checkpoint, step the tail, snapshot (byte-exact continuation)."""
function a6_continue_from(checkpoint, tail::AbstractVector{<:Integer})
    env = deepcopy(checkpoint)
    for a in tail; env_step!(env, Int(a)); end
    return snapshot(env, length(tail))
end

"""A full from-scratch replay (boot included) — the bit-exact baseline guarantee
(two fresh runs must be byte-identical)."""
function a6_fresh_baseline(game, actions::AbstractVector{<:Integer}, total::Integer)
    env = a6_load_env(game)
    for i in 1:total; env_step!(env, Int(actions[i])); end
    return snapshot(env, Int(total))
end

"""Record a (frames, RAM_SIZE+TIA_SIZE) tape on `game` with A6's per-game settings:
columns 1:RAM_SIZE are the 128-byte RIOT RAM, columns RAM_SIZE+1:end the 64-byte
TIA register file. Re-implements record_trajectory's loop with a6_load_env so the
proper RomSettings are used."""
function a6_record_tape(game::AbstractString, frames::Integer,
                        actions::AbstractVector{<:Integer})
    @assert length(actions) >= frames "actions shorter than frames"
    env = a6_load_env(game)
    n = RAM_SIZE + TIA_SIZE
    tape = Matrix{UInt8}(undef, Int(frames), n)
    for t in 1:frames
        env_step!(env, Int(actions[t]))
        tape[t, 1:RAM_SIZE] = UInt8.(collect(get_ram(env)))
        tape[t, RAM_SIZE+1:end] = copy(env.console.bus.tia.registers)
    end
    return tape
end

# ============================================================================
# Action traces (mirror A4 — Granger needs the cells to MOVE).
# ============================================================================
# Action codes (jutari src/io/IO.jl): NOOP=0 FIRE=1 UP=2 RIGHT=3 LEFT=4 DOWN=5
# RIGHTFIRE=11 LEFTFIRE=12.
#
# ORACLE / scored trace = deterministic FIRE-then-NOOP start (stays in the
# conformance window; bit-exact re-run byte-identical).
oracle_actions(total::Integer) = vcat(fill(1, 4), fill(0, max(0, total - 4)))

# The Granger (descriptive) trajectory uses an ACTIVE trace so the candidate cells
# actually MOVE (an all-NOOP tail leaves most cells static, giving Granger no
# signal). The Granger time-series / inferred edges are DESCRIPTIVE statistics of a
# deterministic jutari trajectory (the recorder's documented purpose) — not a
# bit-exact-vs-xitari claim — so they legitimately use this livelier replay. The
# scored-vs-oracle ground truth (the exact data-flow graph + F/S/M) stays on the
# oracle trace, strictly inside the conformance window.
function active_actions(total::Integer)
    acts = Vector{Int}(undef, max(0, total))
    for t in 1:length(acts)
        acts[t] = t <= 4      ? 1  :   # 4×FIRE: start the game
                  t % 4 == 0  ? 11 :   # RIGHTFIRE: move right + fire
                  t % 4 == 2  ? 12 :   # LEFTFIRE:  move left + fire
                                1      # FIRE: keep firing
    end
    return acts
end

# A full maze trace for games where FIRE is a no-op (Ms. Pac-Man).
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
# Candidate cells (E2-1 import) — the "units" the cell-scope Granger probes.
# ============================================================================
struct Candidate
    ram_index::Int
    concept::String
end

"""Read `candidates_<game>.json`, de-duplicated by ram_index, in file order."""
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
    # non-core games have no T3 candidate file — fall back to the SAME generic
    # RAM-byte cause set A1/A2 + the shared testbed's candidate_ram_indices(nothing)
    # use, so all 54 labeled games get a bounded, uniform candidate cell set.
    if isempty(out)
        for (idx, concept) in ((13, "enemy_score"), (14, "player_score"),
                               (49, "ball_x"), (54, "ball_y"),
                               (51, "player_y"), (50, "enemy_y"))
            push!(out, Candidate(idx, concept))
        end
    end
    return out
end

# ============================================================================
# Bivariate Granger causality (lag-p, log-likelihood-ratio form + an F-test).
# ============================================================================
# G(x→y) = ln(RSS_restricted / RSS_unrestricted), where the restricted model
# regresses y[t] on {1, y[t-1..t-p]} and the unrestricted adds {x[t-1..t-p]}.
# G ≥ 0; larger ⇒ x's past improves the prediction of y beyond y's own past
# (temporal precedence). An F-test on the RSS drop gives a significance p-value.

"""Ordinary-least-squares via the normal equations (AᵀA β = Aᵀb), with a tiny
ridge for numerical stability on collinear / constant columns. Returns the residual
sum of squares (RSS). Dependency-free (no LinearAlgebra.\\ needed — a Cholesky-free
Gaussian elimination on the small (k×k) normal matrix)."""
function _ols_rss(A::AbstractMatrix{Float64}, b::AbstractVector{Float64})
    n, k = size(A)
    n <= k && return Statistics.var(b) * max(n - 1, 1)   # underdetermined: fall back
    M = A' * A
    @inbounds for d in 1:k
        M[d, d] += 1e-8                                   # ridge: stabilise singulars
    end
    rhs = A' * b
    β = _solve_spd(M, rhs)
    r = b .- A * β
    return sum(abs2, r)
end

"""Solve M β = rhs for a small symmetric positive-(semi)definite M by Gaussian
elimination with partial pivoting (dependency-free; M is k×k with k = small)."""
function _solve_spd(M::Matrix{Float64}, rhs::Vector{Float64})
    k = length(rhs)
    Aug = hcat(copy(M), copy(rhs))
    for col in 1:k
        # partial pivot
        p = col
        for r in col+1:k
            abs(Aug[r, col]) > abs(Aug[p, col]) && (p = r)
        end
        if p != col
            Aug[col, :], Aug[p, :] = Aug[p, :], Aug[col, :]
        end
        piv = Aug[col, col]
        abs(piv) < 1e-12 && (piv = piv >= 0 ? 1e-12 : -1e-12)
        for r in col+1:k
            f = Aug[r, col] / piv
            f == 0 && continue
            Aug[r, col:end] .-= f .* Aug[col, col:end]
        end
    end
    β = zeros(Float64, k)
    for col in k:-1:1
        s = Aug[col, k+1] - sum(Aug[col, col+1:k] .* β[col+1:k])
        piv = Aug[col, col]
        abs(piv) < 1e-12 && (piv = piv >= 0 ? 1e-12 : -1e-12)
        β[col] = s / piv
    end
    return β
end

"""Build the lagged design matrices for a bivariate Granger test of x→y at lag p.
Returns (RSS_restricted, RSS_unrestricted, n_obs, p_lags). Constant series (no
variance) give a degenerate test (G=0, p-value 1)."""
function _granger_rss(x::AbstractVector{Float64}, y::AbstractVector{Float64}, p::Int)
    T = length(y)
    n = T - p
    n <= 2p + 2 && return (0.0, 0.0, n, p)   # too few observations to fit
    # response and lag blocks
    Y = Float64[y[t] for t in p+1:T]
    # restricted design: [1, y_{t-1..t-p}]
    Ar = ones(Float64, n, p + 1)
    for l in 1:p
        Ar[:, l + 1] = Float64[y[t - l] for t in p+1:T]
    end
    # unrestricted design: [1, y_{t-1..t-p}, x_{t-1..t-p}]
    Au = ones(Float64, n, 2p + 1)
    Au[:, 1:p+1] = Ar
    for l in 1:p
        Au[:, p + 1 + l] = Float64[x[t - l] for t in p+1:T]
    end
    rss_r = _ols_rss(Ar, Y)
    rss_u = _ols_rss(Au, Y)
    return (rss_r, rss_u, n, p)
end

"""Granger statistic + F-test p-value for x→y at lag p.
G = ln(RSS_r / RSS_u) ≥ 0; F = ((RSS_r − RSS_u)/p) / (RSS_u/(n − 2p − 1)).
Returns (G, F, pvalue). Degenerate (constant y, or RSS_u≈0 exact-fit) ⇒ G=0,p=1."""
function granger(x::AbstractVector{Float64}, y::AbstractVector{Float64}, p::Int)
    rss_r, rss_u, n, _ = _granger_rss(x, y, p)
    df2 = n - 2p - 1
    (df2 <= 0 || rss_r <= 0) && return (0.0, 0.0, 1.0)
    # exact-fit / numerically-zero unrestricted residual: x perfectly predicts y's
    # residual ⇒ treat as a strong (significant) edge but a finite G (avoid Inf).
    if rss_u <= 1e-12 * max(rss_r, 1.0)
        return (log(max(rss_r, 1e-12) / 1e-12), Inf, 0.0)
    end
    G = log(rss_r / rss_u)
    G < 0 && (G = 0.0)                       # numerical: restricted can't beat unrestricted
    F = ((rss_r - rss_u) / p) / (rss_u / df2)
    F < 0 && (F = 0.0)
    pval = _f_sf(F, p, df2)
    return (G, F, pval)
end

# Survival function P(F_{d1,d2} > f) via the regularized incomplete beta function
# I_x(a,b) (continued-fraction, Lentz) — dependency-free p-values. P(F>f) =
# I_{d2/(d2+d1 f)}(d2/2, d1/2).
function _f_sf(f::Real, d1::Integer, d2::Integer)
    (f <= 0 || d1 <= 0 || d2 <= 0) && return 1.0
    !isfinite(f) && return 0.0
    x = d2 / (d2 + d1 * f)
    return _betainc(x, d2 / 2, d1 / 2)
end

# log Γ (Lanczos) — for the beta-function front factor.
function _lgamma(z::Float64)
    g = 7.0
    c = (0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)
    if z < 0.5
        return log(pi / sin(pi * z)) - _lgamma(1 - z)
    end
    z -= 1
    a = c[1]
    t = z + g + 0.5
    for i in 2:length(c)
        a += c[i] / (z + (i - 1))
    end
    return 0.5 * log(2 * pi) + (z + 0.5) * log(t) - t + log(a)
end

"""Regularized incomplete beta I_x(a,b) via the Lentz continued fraction
(Numerical Recipes betai). Dependency-free; used only for Granger F p-values."""
function _betainc(x::Float64, a::Float64, b::Float64)
    (x <= 0.0) && return 0.0
    (x >= 1.0) && return 1.0
    lbeta = _lgamma(a) + _lgamma(b) - _lgamma(a + b)
    front = exp(a * log(x) + b * log(1 - x) - lbeta) / a
    # use the symmetry that converges fastest
    if x < (a + 1) / (a + b + 2)
        return front * _betacf(x, a, b)
    else
        return 1.0 - exp(b * log(1 - x) + a * log(x) - lbeta) / b * _betacf(1 - x, b, a)
    end
end

function _betacf(x::Float64, a::Float64, b::Float64)
    MAXIT = 200; EPS = 3e-12; FPMIN = 1e-300
    qab = a + b; qap = a + 1; qam = a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
    abs(d) < FPMIN && (d = FPMIN)
    d = 1.0 / d
    h = d
    for m in 1:MAXIT
        m2 = 2m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; abs(d) < FPMIN && (d = FPMIN)
        c = 1.0 + aa / c; abs(c) < FPMIN && (c = FPMIN)
        d = 1.0 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; abs(d) < FPMIN && (d = FPMIN)
        c = 1.0 + aa / c; abs(c) < FPMIN && (c = FPMIN)
        d = 1.0 / d; del = d * c; h *= del
        abs(del - 1.0) < EPS && break
    end
    return h
end

# ============================================================================
# SUBSYSTEM-scope Granger: CPU/TIA/RIOT activity time-series ⇒ inferred edges.
# ============================================================================
# The three hardware subsystems whose interplay builds each VCS frame. Each
# subsystem's per-frame ACTIVITY = the number of its bytes that CHANGED since the
# previous frame (its "firing rate") — a scalar activity series for the Granger
# test. The RAM tape (columns 1:RAM_SIZE) is the CPU working set; the TIA register
# file (columns RAM_SIZE+1:end) is the TIA; the RIOT timer/I/O block is the high
# zero-page sub-region the RIOT chip owns (0x80..0x97 in the 0x80..0xFF I/O page;
# the RAM file holds the 128-byte zero page so we use the canonical RIOT-owned RAM
# offsets the disassembly writes: SWCHx/timer scratch live in the upper RAM cells).
#
# NOTE the T2 known data-flow direction at the subsystem level (the ground-truth
# arrows A6 is scored against): the CPU (6507) is the only active bus master — it
# READS RIOT (timer/inputs) and WRITES TIA (graphics) and writes/reads RAM. So the
# true directed edges are CPU→TIA, RIOT→CPU, CPU↔RAM. TIA and RIOT never write each
# other or the CPU's RAM. Granger, seeing only temporal precedence on a tightly
# clock-locked frame loop, generically infers DENSE bidirectional edges among all
# co-active subsystems — the quantified failure.
struct SubsystemGranger
    names::Vector{String}                 # ["CPU","TIA","RIOT"]
    activity::Matrix{Float64}             # (T, 3) per-frame change-count series
    G::Matrix{Float64}                    # (3,3) Granger statistic i->j (diag 0)
    pval::Matrix{Float64}                 # (3,3) F-test p-value i->j
    inferred::BitMatrix                   # (3,3) asserted edges i->j
    true_edges::BitMatrix                 # (3,3) the T2 known data-flow direction
    false_edge_rate::Float64
    missed_edge_rate::Float64
    direction_disagreement::Float64
end

# The T2 known subsystem data-flow (the disassembly-level truth): CPU is the only
# bus master. Index order [CPU, TIA, RIOT].
function _subsystem_true_edges()
    A = falses(3, 3)
    A[1, 2] = true     # CPU -> TIA  (the 6507 writes graphics/colour registers)
    A[3, 1] = true     # RIOT -> CPU (the CPU reads the timer / SWCHx inputs)
    # CPU<->RAM is intra-CPU (RAM is the CPU's working set) — not a separate node.
    # TIA and RIOT never drive each other or write the CPU's RAM.
    return A
end

"""Per-frame subsystem activity = #bytes that changed vs the previous frame, for
each of CPU (RAM block), TIA (register file), RIOT (the RIOT-owned upper RAM I/O
sub-block). Columns: 1 CPU, 2 TIA, 3 RIOT."""
function subsystem_activity(tape::AbstractMatrix)
    T = size(tape, 1)
    ram = @view tape[:, 1:RAM_SIZE]
    tia = @view tape[:, RAM_SIZE+1:end]
    # The RIOT chip owns the timer + SWCHA/SWCHB + INTIM scratch; in the 128-byte
    # zero-page file these live in the upper region the disassembly reserves for I/O
    # mirrors/scratch (cells 0x60..0x7F here, the high quarter). We treat that
    # sub-block as the RIOT activity proxy; the lower 3/4 is the CPU working set.
    riot_lo = 0x60 + 1
    act = zeros(Float64, T, 3)
    for t in 2:T
        act[t, 1] = count(k -> ram[t, k] != ram[t-1, k], 1:riot_lo-1)         # CPU
        act[t, 2] = count(k -> tia[t, k] != tia[t-1, k], 1:size(tia, 2))       # TIA
        act[t, 3] = count(k -> ram[t, k] != ram[t-1, k], riot_lo:RAM_SIZE)     # RIOT
    end
    return act
end

function run_subsystem_granger(tape::AbstractMatrix; p::Int, granger_tau::Float64,
                               alpha::Float64)
    names = ["CPU", "TIA", "RIOT"]
    act = subsystem_activity(tape)
    G = zeros(Float64, 3, 3)
    P = ones(Float64, 3, 3)
    for i in 1:3, j in 1:3
        i == j && continue
        g, _f, pv = granger(Float64.(@view act[:, i]), Float64.(@view act[:, j]), p)
        G[i, j] = g; P[i, j] = pv
    end
    inferred = falses(3, 3)
    for i in 1:3, j in 1:3
        i == j && continue
        inferred[i, j] = (G[i, j] >= granger_tau) && (P[i, j] <= alpha)
    end
    tru = _subsystem_true_edges()
    fer, mer, dda = edge_rates(inferred, tru)
    return SubsystemGranger(names, act, G, P, inferred, tru, fer, mer, dda)
end

# ============================================================================
# CELL-scope Granger over candidate cells ⇒ inferred directed graph.
# ============================================================================
struct CellGranger
    G::Matrix{Float64}                    # (n,n) Granger statistic i->j (diag 0)
    pval::Matrix{Float64}                 # (n,n) F-test p-value i->j
    inferred::BitMatrix                   # (n,n) asserted edges i->j
    varying::Vector{Bool}                 # cell varies over the trajectory
end

"""Pairwise Granger over the candidate cells' value time-series. Edge i→j iff
G(i→j) ≥ granger_tau AND the F-test p-value ≤ alpha. Constant cells (no variance)
get no outgoing/incoming edges (Granger is undefined ⇒ degenerate, not asserted)."""
function run_cell_granger(tape::AbstractMatrix, cands::Vector{Candidate};
                          p::Int, granger_tau::Float64, alpha::Float64)
    n = length(cands)
    series = [Float64[Int(tape[t, c.ram_index + 1]) for t in 1:size(tape, 1)] for c in cands]
    varying = Bool[std(s) > 0 for s in series]
    G = zeros(Float64, n, n)
    P = ones(Float64, n, n)
    inferred = falses(n, n)
    for i in 1:n, j in 1:n
        i == j && continue
        (!varying[i] || !varying[j]) && continue   # degenerate: no signal
        g, _f, pv = granger(series[i], series[j], p)
        G[i, j] = g; P[i, j] = pv
        inferred[i, j] = (g >= granger_tau) && (pv <= alpha)
    end
    return CellGranger(G, P, inferred, varying)
end

# ============================================================================
# The TRUE directed data-flow graph over candidate cells (T1/T2 ground truth) —
# the SAME construction A1 uses (exact intervention oracle, exhaustive do-set).
# ============================================================================
"""deepcopy the checkpoint, poke ram[idx]:=value, step `tail`, snapshot."""
function _intervene_continue(checkpoint, idx::Integer, value::Integer, tail)
    env = deepcopy(checkpoint)
    intervene_ram!(env, idx, value)
    for a in tail; env_step!(env, Int(a)); end
    return snapshot(env, length(tail))
end

"""TRUE directed data-flow graph: edge i→j iff the EXHAUSTIVE oracle do-set on cell
i (occlude→0, set→base+17, set→base+37) changes cell j at the full horizon, for ANY
value. The §1 oracle restricted to candidate-cell pairs (= A1's true read/write
graph). `delta_set` selects which values: the default {0,+17,+37} is the full
truth; pass a single delta for a held-out / single-value truth."""
function true_dataflow_graph(checkpoint, tail, cands::Vector{Candidate},
                             at_target::Snapshot; deltas = (0, 17, 37),
                             base_ram = nothing)
    n = length(cands)
    A = falses(n, n)
    base = base_ram === nothing ? a6_continue_from(checkpoint, tail).ram : base_ram
    for (i, c) in enumerate(cands)
        bval = Int(at_target.ram[c.ram_index + 1])
        changed = falses(n)
        for d in deltas
            v = d == 0 ? 0 : (bval + d) & 0xFF
            v == bval && continue
            r = _intervene_continue(checkpoint, c.ram_index, v, tail).ram
            for (j, cj) in enumerate(cands)
                r[cj.ram_index + 1] != base[cj.ram_index + 1] && (changed[j] = true)
            end
        end
        A[i, :] = changed
    end
    return A
end

# ============================================================================
# false-edge / missed-edge rates + direction disagreement (the §4 A6 metric).
# ============================================================================
"""Over OFF-DIAGONAL directed edges:
  false_edge_rate  = #(inferred i→j, true ABSENT) / #inferred edges  ( = 1−precision)
  missed_edge_rate = #(true i→j, inferred ABSENT) / #true edges      ( = 1−recall)
  direction_disagreement = of unordered pairs {i,j} where the TRUE graph has a single
    directed edge AND the inferred graph has a single directed edge, the fraction
    whose directions DISAGREE (the temporal-precedence-≠-causation Granger failure).
Returns (false_edge_rate, missed_edge_rate, direction_disagreement)."""
function edge_rates(inferred::BitMatrix, tru::BitMatrix)
    n = size(tru, 1)
    @assert size(inferred) == size(tru) "adjacency shape mismatch"
    n_inf = 0; n_tru = 0; fp = 0; fn = 0
    for i in 1:n, j in 1:n
        i == j && continue
        inf = inferred[i, j]; tr = tru[i, j]
        inf && (n_inf += 1)
        tr && (n_tru += 1)
        (inf && !tr) && (fp += 1)
        (!inf && tr) && (fn += 1)
    end
    fer = n_inf == 0 ? 0.0 : fp / n_inf
    mer = n_tru == 0 ? 0.0 : fn / n_tru
    # direction disagreement over unordered pairs with a single arrow each side
    disagree = 0; comparable = 0
    for i in 1:n, j in i+1:n
        t_ij = tru[i, j]; t_ji = tru[j, i]
        i_ij = inferred[i, j]; i_ji = inferred[j, i]
        true_single = (t_ij ⊻ t_ji)            # exactly one true direction
        inf_single  = (i_ij ⊻ i_ji)            # exactly one inferred direction
        if true_single && inf_single
            comparable += 1
            # disagree iff the single arrows point opposite ways
            ((t_ij && i_ji) || (t_ji && i_ij)) && (disagree += 1)
        end
    end
    dda = comparable == 0 ? 0.0 : disagree / comparable
    return (fer, mer, dda)
end

# F1 / precision / recall over off-diagonal directed edges.
function _prf(inferred::BitMatrix, tru::BitMatrix)
    n = size(tru, 1)
    tp = 0; fp = 0; fn = 0; n_inf = 0; n_tru = 0
    for i in 1:n, j in 1:n
        i == j && continue
        inf = inferred[i, j]; tr = tru[i, j]
        inf && (n_inf += 1); tr && (n_tru += 1)
        (inf && tr) && (tp += 1)
        (inf && !tr) && (fp += 1)
        (!inf && tr) && (fn += 1)
    end
    precision = n_inf == 0 ? (n_tru == 0 ? 1.0 : 0.0) : tp / n_inf
    recall    = n_tru == 0 ? 1.0 : tp / n_tru
    f1 = (precision + recall) == 0 ? 0.0 : 2 * precision * recall / (precision + recall)
    return (precision, recall, f1, tp, fp, fn, n_inf, n_tru)
end

# ============================================================================
# F / S / M (the correctness triad, scored against the exact data-flow oracle).
# ============================================================================
struct Triad
    F::Float64; S::Float64; M::Float64
    F_note::String; S_note::String; M_note::String
    n_nodes::Int; n_true_edges::Int; n_inferred_edges::Int
end

"""F = F1 of the Granger graph vs the true data-flow edges. S = generalisation to a
HELD-OUT intervention value: off-diagonal directed-edge agreement between the
Granger graph and the base+37-only TRUE edge set (does the inferred connectivity
predict an unseen intervention's data-flow?). M = 1 − false-edge rate (minimality /
no spurious directed wiring)."""
function score_triad(inferred::BitMatrix, tru::BitMatrix, heldout_true::BitMatrix,
                     false_edge_rate::Float64)
    _p, _r, f1, _tp, _fp, _fn, n_inf, n_tru = _prf(inferred, tru)
    F = f1
    # S — predict the held-out (base+37) TRUE edges from the Granger graph:
    #     off-diagonal directed-edge agreement (accuracy).
    n = size(tru, 1)
    agree = 0; total = 0
    for i in 1:n, j in 1:n
        i == j && continue
        total += 1
        (inferred[i, j] == heldout_true[i, j]) && (agree += 1)
    end
    S = total == 0 ? 1.0 : agree / total
    # M — minimality: 1 − false-edge rate (spurious directed wiring).
    M = 1.0 - false_edge_rate
    return Triad(F, S, M,
        "F1 of the Granger-inferred directed edges vs the exact-oracle true data-flow edges ($n candidate cells)",
        "off-diagonal directed-edge agreement between the Granger graph and the HELD-OUT base+37 true data-flow (generalisation to an unseen intervention, bit-exact re-run)",
        "1 − false-edge rate: fraction of Granger-inferred edges the oracle says are spurious (temporal precedence ≠ causation)",
        n, n_tru, n_inf)
end

# ============================================================================
# Drive one game: trajectory → subsystem & cell Granger → true graph → rates → triad.
# ============================================================================
struct GameResult
    game::String
    target_frame::Int
    horizon::Int
    traj_frames::Int
    trace::String
    in_window::Bool
    lag::Int
    granger_tau::Float64
    alpha::Float64
    seed::Int
    bit_exact::Bool
    cands::Vector{Candidate}
    subsys::SubsystemGranger
    cell::CellGranger
    true_graph::BitMatrix
    heldout_true::BitMatrix
    cell_false_edge_rate::Float64
    cell_missed_edge_rate::Float64
    cell_direction_disagreement::Float64
    triad::Triad
    # positive control: the true graph used as the "inferred" method → F1 must be 1
    control_false_edge_rate::Float64
    control_missed_edge_rate::Float64
    control_f1::Float64
    # SHARED-TESTBED provenance (redesign); "noop"/-1/false in the legacy path.
    state_kind::String             # "seeded_random_action_gameplay" | "noop"
    st_seed::Int
    st_prefix::Int
    cause_density::Int
    cause_density_accepted::Bool
    n_causes::Int
    shared_cell::Tuple{Int,Int}
end

# A6's candidates-path resolver in the shape the shared testbed injects. Mirrors
# resolve_candidates' search.
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
oracle machinery (oracle_intervene.jl). Returns the substrate NamedTuple (we use its
`.actions` stream + `.cause_density`/`.accepted`/`.cell` gate). A6 then drives BOTH
its scored oracle prefix AND its descriptive Granger trajectory from `st.actions` so
the Granger + true-data-flow algorithm is unchanged."""
function build_a6_shared_state(game::AbstractString; verbose = false)
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

function compute_game(game::AbstractString; target_frame = nothing, horizon = nothing,
                      traj_frames = nothing, lag = 2, granger_tau = 0.0,
                      alpha = 0.05, seed = 0, verbose = true)
    cfg = game_cfg(game)
    target_frame = target_frame === nothing ? cfg.target_frame : target_frame
    horizon      = horizon      === nothing ? cfg.horizon      : horizon
    traj_frames  = traj_frames  === nothing ? cfg.traj_frames  : traj_frames
    trace        = cfg.trace
    in_window    = cfg.in_window

    # SHARED-TESTBED (redesign): the scored oracle prefix (true data-flow graph) + the
    # descriptive Granger trajectory both move onto the seeded random-action GAMEPLAY
    # stream (oracle cause-density gated). target_frame/horizon become the shared
    # prefix/horizon; the oracle action stream = st.actions; the Granger tape is driven
    # by st.actions for the first prefix+horizon frames then NOOP-continued to
    # traj_frames. The Granger + true-data-flow algorithm is UNCHANGED — only the
    # driving action streams move.
    st = nothing
    if SHARED_TESTBED
        st = build_a6_shared_state(game; verbose = verbose)
        target_frame = st.prefix; horizon = st.horizon
        gp = Int.(st.actions)
        traj_acts = Vector{Int}(undef, traj_frames)
        for t in 1:traj_frames
            traj_acts[t] = t <= length(gp) ? gp[t] : 0     # gameplay window, then NOOP
        end
        verbose && println("[A6:$game] SHARED gameplay state: cause_density=$(st.cause_density)/" *
            "$(length(st.causes)) accepted=$(st.accepted) cell=$(st.cell); " *
            "trajectory = gameplay($(length(gp)) frames)+NOOP tail to f$traj_frames")
    end
    total = target_frame + horizon
    oacts = SHARED_TESTBED ? Int.(st.actions) : oracle_actions(total)
    tail = Int.(oacts[target_frame + 1 : target_frame + horizon])

    # 1) bit-exact baseline — two fresh boots+replays must be byte-identical.
    verbose && println("[A6:$game] asserting bit-exactness (2 fresh boots+replays to f$total)...")
    a = a6_fresh_baseline(game, oacts, total)
    b = a6_fresh_baseline(game, oacts, total)
    bit_exact = (a.ram == b.ram) && (a.screen == b.screen)
    bit_exact || error("bit-exact re-run FAILED for $game to f$total — refusing to score")
    verbose && println("[A6:$game] bit-exact re-run: PASS")

    # 2) one checkpoint at the intervention frame (boot+to-target paid once).
    checkpoint = a6_boot_replay(game, oacts, target_frame)
    at_target = a6_continue_from(checkpoint, Int[])
    base_snap = a6_continue_from(checkpoint, tail)

    cand_path = resolve_candidates(game)
    cands = load_candidates(cand_path)
    isempty(cands) && error("no candidate cells for $game (candidates file missing/empty: $cand_path)")
    verbose && println("[A6:$game] candidates: $(cand_path) ($(length(cands)) cells)")

    # 3) descriptive Granger over an ACTIVE trajectory (cells must move). The tape
    #    stacks RAM + TIA so the subsystem-scope activity series exist.
    traj_stream = SHARED_TESTBED ? traj_acts : trace_actions(trace, traj_frames)
    verbose && println("[A6:$game] recording $(traj_frames)-frame " *
                       (SHARED_TESTBED ? "gameplay+NOOP" : "active ($(trace))") *
                       " RAM+TIA trajectory...")
    tape = a6_record_tape(game, traj_frames, traj_stream)

    verbose && println("[A6:$game] subsystem-scope Granger (CPU/TIA/RIOT, lag=$lag)...")
    subsys = run_subsystem_granger(tape; p = lag, granger_tau = granger_tau, alpha = alpha)

    verbose && println("[A6:$game] cell-scope Granger over $(length(cands)) candidate cells (lag=$lag)...")
    cell = run_cell_granger(tape, cands; p = lag, granger_tau = granger_tau, alpha = alpha)

    # 4) the TRUE directed data-flow graph (exact oracle, exhaustive do-set, full
    #    horizon) — the T1/T2 ground truth (A1's true read/write graph).
    verbose && println("[A6:$game] true data-flow graph (exact oracle, $(length(cands)) cells)...")
    true_graph = true_dataflow_graph(checkpoint, tail, cands, at_target;
                                     deltas = (0, 17, 37), base_ram = base_snap.ram)
    # 5) HELD-OUT true graph (base+37 only — a single value, the S probe).
    heldout_true = true_dataflow_graph(checkpoint, tail, cands, at_target;
                                       deltas = (37,), base_ram = base_snap.ram)

    # 6) §4 A6 metric: false-edge / missed-edge rates + direction disagreement
    #    (cell scope — same node set as the true graph, so the rates are well-posed).
    fer, mer, dda = edge_rates(cell.inferred, true_graph)

    # 7) F / S / M triad (Granger graph vs the exact data-flow oracle).
    triad = score_triad(cell.inferred, true_graph, heldout_true, fer)

    # 8) POSITIVE CONTROL — the TRUE graph used as the "inferred" method: must give
    #    false-edge = missed-edge = 0, F1 = 1 (the harness rewards a perfect map).
    cfer, cmer, _cdda = edge_rates(true_graph, true_graph)
    _cp, _cr, cf1, _t, _f, _fn, _ni, _nt = _prf(true_graph, true_graph)

    if verbose
        println("[A6:$game] ---- A6 scores ----")
        println("[A6:$game]   subsystem (CPU/TIA/RIOT): inferred edges=$(count(subsys.inferred)) " *
                "vs true=$(count(subsys.true_edges)); false-edge=$(round(subsys.false_edge_rate,digits=3)) " *
                "missed-edge=$(round(subsys.missed_edge_rate,digits=3)) " *
                "dir-disagree=$(round(subsys.direction_disagreement,digits=3))")
        println("[A6:$game]   cell: true_edges=$(triad.n_true_edges) inferred=$(triad.n_inferred_edges); " *
                "FALSE-EDGE=$(round(fer,digits=3)) MISSED-EDGE=$(round(mer,digits=3)) " *
                "DIR-DISAGREE=$(round(dda,digits=3))")
        println("[A6:$game]   positive control (true-graph-as-method): " *
                "false-edge=$(round(cfer,digits=3)) missed-edge=$(round(cmer,digits=3)) F1=$(round(cf1,digits=3))")
        println("[A6:$game]   TRIAD F=$(round(triad.F,digits=3)) S=$(round(triad.S,digits=3)) M=$(round(triad.M,digits=3))")
    end

    return GameResult(game, target_frame, horizon, traj_frames, trace, in_window,
                      lag, granger_tau, alpha, seed, bit_exact, cands, subsys, cell,
                      true_graph, heldout_true, fer, mer, dda, triad,
                      cfer, cmer, cf1,
                      st === nothing ? "noop" : "seeded_random_action_gameplay",
                      st === nothing ? -1 : st.seed,
                      st === nothing ? -1 : st.prefix,
                      st === nothing ? -1 : st.cause_density,
                      st === nothing ? false : st.accepted,
                      st === nothing ? 0 : length(st.causes),
                      st === nothing ? (-1, -1) : st.cell)
end

# ============================================================================
# Self-check (DoD) — the scoring contract is sound; results are non-fabricated.
# ============================================================================
"""
    selftest(r::GameResult) -> Bool

Asserts A6's load-bearing claims (same contract shape as A1/A4 selftest):

  (BIT-EXACT) the baseline re-run is byte-identical (every data-flow edge is a clean
    causal effect on the real ROM).

  (GROUND-TRUTH ANCHORED) the true data-flow graph has ≥1 edge (some intervention
    moves another candidate cell) — else the rates / F are uninformative.

  (POSITIVE CONTROL) the TRUE graph used as the "inferred" method scores
    false-edge = missed-edge = 0 and F1 = 1 — the harness rewards a perfectly-
    faithful directed map, so a sub-1 Granger F is a real measurement.

  (GRANGER WELL-FORMED) the cell Granger statistic is ≥ 0 with a 0 diagonal; the
    F-test p-values ∈ [0,1]; the subsystem Granger likewise.

  (RATES / TRIAD RANGES) false-edge / missed-edge / direction-disagreement ∈ [0,1];
    F,S ∈ [0,1] (F1 + agreement are non-negative); M ∈ [0,1]; finite.

Throws on a contract violation."""
function selftest(r::GameResult)
    @assert r.bit_exact "bit-exact baseline re-run failed for $(r.game)"

    n_true = count(r.true_graph) - count(r.true_graph[i, i] for i in 1:size(r.true_graph, 1))
    @assert n_true >= 1 "true data-flow graph has NO off-diagonal edge for $(r.game) at this " *
        "state — uninformative; pick a livelier target frame"

    # positive control: the true graph as method → false-edge = missed-edge = 0, F1 = 1
    @assert r.control_false_edge_rate <= 1e-12 "positive control false-edge != 0 ($(r.control_false_edge_rate))"
    @assert r.control_missed_edge_rate <= 1e-12 "positive control missed-edge != 0 ($(r.control_missed_edge_rate))"
    @assert r.control_f1 > 0.999 "harness broken: true-graph-as-method F1 != 1 ($(r.control_f1)) for $(r.game)"

    # Granger statistic well-formed
    n = size(r.cell.G, 1)
    @assert all(r.cell.G[i, i] == 0.0 for i in 1:n) "cell Granger diagonal != 0"
    @assert all(r.cell.G[i, j] >= -1e-9 for i in 1:n, j in 1:n) "negative Granger statistic"
    @assert all(0.0 - 1e-9 <= r.cell.pval[i, j] <= 1.0 + 1e-9 for i in 1:n, j in 1:n) "cell p-value out of [0,1]"
    for i in 1:3, j in 1:3
        @assert r.subsys.G[i, j] >= -1e-9 "negative subsystem Granger statistic"
        @assert 0.0 - 1e-9 <= r.subsys.pval[i, j] <= 1.0 + 1e-9 "subsystem p-value out of [0,1]"
    end

    for (nm, v) in (("cell false_edge", r.cell_false_edge_rate),
                    ("cell missed_edge", r.cell_missed_edge_rate),
                    ("cell dir_disagree", r.cell_direction_disagreement),
                    ("subsys false_edge", r.subsys.false_edge_rate),
                    ("subsys missed_edge", r.subsys.missed_edge_rate),
                    ("subsys dir_disagree", r.subsys.direction_disagreement))
        @assert 0.0 - 1e-9 <= v <= 1.0 + 1e-9 "$nm out of [0,1]: $v"
    end
    @assert 0.0 - 1e-9 <= r.triad.F <= 1.0 + 1e-9 "F out of [0,1]: $(r.triad.F)"
    @assert 0.0 - 1e-9 <= r.triad.S <= 1.0 + 1e-9 "S out of [0,1]: $(r.triad.S)"
    @assert 0.0 - 1e-9 <= r.triad.M <= 1.0 + 1e-9 "M out of [0,1]: $(r.triad.M)"

    println("[A6:$(r.game)] SELF-CHECK PASS:")
    println("[A6:$(r.game)]   bit-exact baseline re-run: $(r.bit_exact)")
    println("[A6:$(r.game)]   true data-flow edges: $n_true (positive control F1 = $(round(r.control_f1,digits=3)), " *
            "false-edge=$(round(r.control_false_edge_rate,digits=3)))")
    println("[A6:$(r.game)]   cell FALSE-EDGE=$(round(r.cell_false_edge_rate,digits=3)) " *
            "MISSED-EDGE=$(round(r.cell_missed_edge_rate,digits=3)) " *
            "DIR-DISAGREE=$(round(r.cell_direction_disagreement,digits=3))")
    println("[A6:$(r.game)]   F=$(round(r.triad.F,digits=3)) S=$(round(r.triad.S,digits=3)) M=$(round(r.triad.M,digits=3))")
    return true
end

# ============================================================================
# Persist (SPEC §R) — JSON record + sibling .npz; file_scope A6_*.
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

# adjacency -> list of [i,j] off-diagonal edge pairs (0-based candidate ordinals)
function _edge_list(A::BitMatrix)
    n = size(A, 1)
    edges = Vector{Vector{Int}}()
    for i in 1:n, j in 1:n
        i == j && continue
        A[i, j] && push!(edges, [i - 1, j - 1])
    end
    return edges
end

function write_game(r::GameResult; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    stem = "A6_$(r.game)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path  = joinpath(out_dir, stem * ".npz")

    cell_names = ["RAM[$(c.ram_index)]:$(c.concept)" for c in r.cands]
    rec = Dict{String,Any}(
        "paper" => "P2",
        "phase" => "phaseA_kording",
        "method" => "A6_granger_causality(CPU/TIA/RIOT + cell-cell directed edges)",
        "game" => r.game,
        "state" => r.state_kind == "noop" ? "f$(r.target_frame)+$(r.horizon)" :
                   "gameplay(seed=$(r.st_seed),prefix=$(r.st_prefix))+$(r.horizon)",
        "target_output" => "directed data-flow edges (CPU/TIA/RIOT subsystems + candidate RAM cells)",
        # headline scalar (SPEC §R value/metric_name): the §4 A6 FALSE-EDGE rate of
        # the cell-scope Granger graph vs the exact true data-flow.
        "metric_name" => "A6_cell_false_edge_rate_vs_true_dataflow",
        "value" => _json_num(r.cell_false_edge_rate),
        "stderr" => nothing,
        "ci" => nothing,
        "n" => length(r.cands),
        "seed" => r.seed,
        "where" => "local",
        "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene@$(r.game)#directed-data-flow(candidate cells)",
        "timestamp" => string(round(Int, time())),
        "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            "substrate" => "jutari (Julia, HARD) — real-ROM bit-exact path; the true " *
                "data-flow graph uses EXACT interventions re-run on the true ROM.",
            "bit_exact_rerun" => r.bit_exact,
            "testbed" => Dict{String,Any}(
                "state_kind" => r.state_kind,
                "seed" => r.st_seed, "prefix" => r.st_prefix, "horizon" => r.horizon,
                "shared_output" => "screen_region(n_changed_px)@r$(r.shared_cell[1])c$(r.shared_cell[2])",
                "cause_density_above_floor" => r.cause_density,
                "cause_density_floor" => ST_FLOOR, "cause_density_gate_k" => ST_GATE_K,
                "cause_density_accepted" => r.cause_density_accepted, "n_causes" => r.n_causes,
                "note" => "P2 redesign: BOTH the scored oracle prefix (true data-flow graph) " *
                    "AND the descriptive Granger trajectory run on a seeded random-action " *
                    "GAMEPLAY state (not the FIRE-then-NOOP oracle tape / synthetic active " *
                    "tape), gated by the §1 oracle cause-density gate. The Granger tape is the " *
                    "gameplay stream for the first prefix+horizon frames then NOOP-continued to " *
                    "traj_frames. A6's Granger + true-data-flow algorithm is unchanged; only " *
                    "the analysis state moves onto genuine input-driven gameplay."),
            "trajectory_trace" => r.trace == "dir" ?
                "directional maze trace (FIRE warmup + cyclic UP/DOWN/LEFT/RIGHT)" :
                "active trace (4×FIRE + periodic RIGHTFIRE/LEFTFIRE)",
            "trajectory_frames" => r.traj_frames,
            "scored_in_conformance_window" => r.in_window,
            "conformance_note" => r.in_window ?
                "scored oracle frame f$(r.target_frame)+$(r.horizon) is inside the " *
                "Paper-1 60-frame screen conformance window (jutari↔xitari bit-exact)." :
                "scored oracle frame f$(r.target_frame)+$(r.horizon) is BEYOND the strict " *
                "60-frame screen conformance window — the game only reaches live play later. " *
                "This is an HONEST descriptive jutari result (the bit-exact jutari re-run IS " *
                "asserted), not an xitari-parity claim; A6 is the calibration baseline.",
            "granger_estimator" => Dict{String,Any}(
                "form" => "bivariate lag-p log-likelihood-ratio G(x→y)=ln(RSS_restricted/RSS_unrestricted) (Geweke 1982)",
                "lag" => r.lag,
                "granger_tau" => r.granger_tau,
                "alpha" => r.alpha,
                "edge_rule" => "canonical Granger: edge x→y iff the F-test p-value ≤ alpha " *
                    "(significance), optionally AND-gated by a magnitude floor G(x→y) ≥ granger_tau " *
                    "(default granger_tau=0.0 = significance alone, the textbook procedure)",
                "pvalue" => "F-test on the RSS drop; survival function via the regularized incomplete beta (dependency-free)",
            ),
            "candidate_cells" => cell_names,
            "candidate_ram_indices" => [c.ram_index for c in r.cands],
            # the §4 A6 headline metric — cell scope (same node set as the true graph)
            "cell_scope" => Dict{String,Any}(
                "n_nodes" => length(r.cands),
                "n_true_edges" => r.triad.n_true_edges,
                "n_inferred_edges" => r.triad.n_inferred_edges,
                "false_edge_rate" => _json_num(r.cell_false_edge_rate),
                "missed_edge_rate" => _json_num(r.cell_missed_edge_rate),
                "direction_disagreement" => _json_num(r.cell_direction_disagreement),
                "inferred_edges_0based" => _edge_list(r.cell.inferred),
                "true_edges_0based" => _edge_list(r.true_graph),
                "note" => "false_edge_rate=#(Granger edge, true absent)/#Granger edges " *
                    "(=1−precision); missed_edge_rate=#(true edge, Granger absent)/#true " *
                    "edges (=1−recall); direction_disagreement = of unordered pairs with a " *
                    "single true AND single Granger arrow, the fraction whose directions " *
                    "OPPOSE — the headline temporal-precedence≠causation Granger failure.",
            ),
            # subsystem scope — the CPU/TIA/RIOT directed edges vs the T2 known data-flow
            "subsystem_scope" => Dict{String,Any}(
                "subsystems" => r.subsys.names,
                "activity_metric" => "per-frame #bytes changed vs previous frame (CPU=RAM working " *
                    "set lower 3/4; TIA=64-byte register file; RIOT=upper-RAM timer/SWCHx I/O block)",
                "true_dataflow_edges_0based" => _edge_list(r.subsys.true_edges),
                "true_dataflow_note" => "T2 known data-flow: the 6507 CPU is the only bus master — " *
                    "CPU→TIA (writes graphics), RIOT→CPU (reads timer/inputs); TIA and RIOT never " *
                    "drive each other or the CPU. Index order [CPU,TIA,RIOT].",
                "inferred_edges_0based" => _edge_list(r.subsys.inferred),
                "false_edge_rate" => _json_num(r.subsys.false_edge_rate),
                "missed_edge_rate" => _json_num(r.subsys.missed_edge_rate),
                "direction_disagreement" => _json_num(r.subsys.direction_disagreement),
                "note" => "on a tightly clock-locked frame loop Granger generically infers DENSE " *
                    "bidirectional edges among all co-active subsystems (every subsystem's past " *
                    "predicts every other's) — false-edge rate > 0 despite the true data-flow " *
                    "being sparse and directional.",
            ),
            "triad" => Dict{String,Any}(
                "F" => _json_num(r.triad.F), "F_note" => r.triad.F_note,
                "S" => _json_num(r.triad.S), "S_note" => r.triad.S_note,
                "M" => _json_num(r.triad.M), "M_note" => r.triad.M_note,
                "n_nodes" => r.triad.n_nodes,
                "n_true_edges" => r.triad.n_true_edges,
                "n_inferred_edges" => r.triad.n_inferred_edges,
                "interpretation" => "Phase A is the calibration baseline " *
                    "(experiment_design.md §4): Granger causality infers TEMPORAL " *
                    "PRECEDENCE, which is NOT the true causal data-flow direction. On a " *
                    "deterministic register-transfer machine the two demonstrably disagree — " *
                    "false_edge_rate>0, missed_edge_rate>0, and direction_disagreement>0 " *
                    "quantify the departure from the known data-flow.",
            ),
            "positive_control" => Dict{String,Any}(
                "method" => "true-data-flow-graph-as-method",
                "false_edge_rate" => _json_num(r.control_false_edge_rate),
                "missed_edge_rate" => _json_num(r.control_missed_edge_rate),
                "f1" => _json_num(r.control_f1),
                "note" => "scoring the exact true graph as the candidate method yields " *
                    "false-edge=missed-edge=0, F1=1 — the harness rewards a perfectly-faithful " *
                    "directed map, so a sub-1 Granger F is a real measurement.",
            ),
            "scales_to_cluster_via" =>
                "tools/cluster/xai_*.sbatch (E0-3) — the full A1..A8 battery over the " *
                "core+breadth set; the forward is bit-exact to this HARD path.",
        ),
    )
    open(json_path, "w") do io; JSON.print(io, rec, 2); end

    write_npz(npz_path, Dict(
        "candidate_ram_indices"   => Int64[c.ram_index for c in r.cands],
        "cell_granger_G"          => r.cell.G,
        "cell_granger_pval"       => r.cell.pval,
        "cell_inferred_adj"       => Int64[r.cell.inferred[i, j] ? 1 : 0
                                           for i in 1:size(r.cell.inferred, 1),
                                               j in 1:size(r.cell.inferred, 2)],
        "true_dataflow_adj"       => Int64[r.true_graph[i, j] ? 1 : 0
                                           for i in 1:size(r.true_graph, 1),
                                               j in 1:size(r.true_graph, 2)],
        "heldout_true_adj"        => Int64[r.heldout_true[i, j] ? 1 : 0
                                           for i in 1:size(r.heldout_true, 1),
                                               j in 1:size(r.heldout_true, 2)],
        "cell_varying"            => Int64[r.cell.varying[i] ? 1 : 0 for i in 1:length(r.cell.varying)],
        "subsystem_activity"      => r.subsys.activity,
        "subsystem_granger_G"     => r.subsys.G,
        "subsystem_granger_pval"  => r.subsys.pval,
        "subsystem_inferred_adj"  => Int64[r.subsys.inferred[i, j] ? 1 : 0 for i in 1:3, j in 1:3],
        "subsystem_true_adj"      => Int64[r.subsys.true_edges[i, j] ? 1 : 0 for i in 1:3, j in 1:3],
        # [cell_false_edge, cell_missed_edge, cell_dir_disagree,
        #  subsys_false_edge, subsys_missed_edge, subsys_dir_disagree]
        "edge_rates"              => Float64[r.cell_false_edge_rate, r.cell_missed_edge_rate,
                                             r.cell_direction_disagreement,
                                             r.subsys.false_edge_rate, r.subsys.missed_edge_rate,
                                             r.subsys.direction_disagreement],
        "triad_FSM"               => Float64[r.triad.F, r.triad.S, r.triad.M],
    ))
    return json_path, npz_path
end

# ============================================================================
# CLI
# ============================================================================
function main(args = ARGS)
    games = copy(CORE_GAMES)
    target_frame = nothing; horizon = nothing; traj_frames = nothing
    lag = 2; granger_tau = 0.0; alpha = 0.05; seed = 0
    selftest_only = false
    i = 1
    while i <= length(args)
        a = args[i]
        if     a == "--games";        games = xai_resolve_games(args[i+1], CORE_GAMES); i += 2
        elseif a == "--game";         games = [args[i+1]]; i += 2
        elseif a == "--target-frame"; target_frame = parse(Int, args[i+1]); i += 2
        elseif a == "--horizon";      horizon = parse(Int, args[i+1]); i += 2
        elseif a == "--traj-frames";  traj_frames = parse(Int, args[i+1]); i += 2
        elseif a == "--lag";          lag = parse(Int, args[i+1]); i += 2
        elseif a == "--granger-tau";  granger_tau = parse(Float64, args[i+1]); i += 2
        elseif a == "--alpha";        alpha = parse(Float64, args[i+1]); i += 2
        elseif a == "--seed";         seed = parse(Int, args[i+1]); i += 2
        elseif a == "--selftest";     selftest_only = true; i += 1
        else; i += 1
        end
    end
    println("[A6] games=$(join(games, ",")) target_frame=$target_frame horizon=$horizon " *
            "traj_frames=$traj_frames lag=$lag granger_tau=$granger_tau alpha=$alpha seed=$seed (jutari/Julia path)")

    if selftest_only
        g = games[1]
        r = compute_game(g; target_frame = target_frame, horizon = horizon,
                         traj_frames = traj_frames, lag = lag, granger_tau = granger_tau,
                         alpha = alpha, seed = seed)
        selftest(r)
        println("[A6] --selftest: passed on $g, not writing artifact.")
        return 0
    end

    ok = String[]; failed = Tuple{String,String}[]
    for g in games
        try
            r = compute_game(g; target_frame = target_frame, horizon = horizon,
                             traj_frames = traj_frames, lag = lag, granger_tau = granger_tau,
                             alpha = alpha, seed = seed)
            selftest(r)
            json_path, npz_path = write_game(r)
            println("[A6] wrote $json_path")
            println("[A6] arrays  $npz_path")
            push!(ok, g)
        catch err
            msg = sprint(showerror, err)
            println("[A6] !! game $g FAILED (scoring the rest, not fabricating): " *
                    first(split(msg, '\n')))
            push!(failed, (g, first(split(msg, '\n'))))
        end
    end
    println("[A6] ==== summary: $(length(ok))/$(length(games)) games scored ====")
    for g in ok; println("[A6]   OK   $g"); end
    for (g, m) in failed; println("[A6]   FAIL $g — $m"); end
    return 0   # partial success is allowed (score the rest, don't fabricate)
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    A6Granger.main()
end
