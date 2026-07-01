# dictionaries.jl — Phase-C E5-7: NMF / PCA (+ a small SAE) DICTIONARIES over the
# recorded VCS state trajectory, the JULIA path (the jutari real-ROM substrate;
# jaxtari eager is ~205× slower — SCRUM §7). Runs on the 6 core games + a
# component-count (rank) sweep, and adds the Phase-C novelty over the Phase-A A7
# dim-reduction: a CAUSAL-USE test — ablate a learned dictionary atom, reconstruct
# the state, write it back into the real ROM and RE-RUN via the exact intervention
# oracle, and ask "does this dictionary atom causally drive the output?"
#
# ---------------------------------------------------------------------------
# What E5-7 IS (experiment_design.md §6, row "NMF/PCA dictionaries"; SPEC §E5):
#   The state trajectory is the "activations"; the program's data-flow is the
#   "circuit" — both KNOWN here. Dictionary learning (PCA — signed; NMF — the
#   non-negative parts; and a tiny SAE for contrast) over the recorded (T,128) RAM
#   tape yields latent components ("dictionary atoms"). Two scores:
#
#   (1) MATCHED-COMPONENT FRACTION vs the known variables. For each atom: is its
#       loading CONCENTRATED on a single known variable — a candidate concept
#       family (T3, game-variable matching) OR a structural known signal (T2:
#       the frame clock, per-cell write activity / R-W, the vsync phase)? An atom
#       is "matched" iff its loading mass is dominated by ONE known variable above
#       a purity threshold OR its per-frame activation tracks one of the structural
#       signals. matched fraction = #matched atoms / #atoms. Reported for PCA, NMF
#       AND the SAE (the prompt's "contrast NMF vs PCA vs the SAE").
#
#   (2) CAUSAL-USE test (the Phase-C contribution over A7's correlational match).
#       A "dictionary atom" is just a direction in state space; correlational
#       matching shows it is PRESENT, not that it is USED. The causal-use test:
#         * project a clean state-at-frame-t onto the dictionary → activations a;
#         * ABLATE atom k (a := a with a[k]=0), reconstruct x̂ = decode(a);
#         * round/clamp x̂ to bytes and WRITE it into a deepcopy of the real env at
#           frame t, then RE-RUN the ROM `horizon` steps (the §1 intervention
#           oracle, a bit-exact re-run);
#         * the causal effect of atom k = whole-screen break vs the
#           reconstruct-without-ablation control (so the metric isolates the ATOM,
#           not the lossy reconstruction itself).
#       per-atom causal effect → does any atom causally move the output? A FAITHFUL
#       dictionary would have its matched atoms be the causally-active ones; the
#       trap is high-variance atoms that are PRESENT but causally inert (variance ≠
#       causal role — the quantified-Kording / present-≠-used lesson, now at the
#       feature/dictionary level). Positive control: ablating ALL atoms (zeroing
#       the whole varying subspace) must move the screen — else the test is inert.
#
# F/S/M triad (experiment_design.md §0), scored against the EXACT oracle (NOT
# against any interpretability method):
#   F (faithful)   — Spearman(per-atom |matched-purity|·is-matched, per-atom causal
#                    effect): are the atoms that MATCH a known variable the ones that
#                    are causally USED? (do matched ⇒ causal.)
#   S (sufficient) — Pearson(per-atom causal effect, a HELD-OUT causal probe: the
#                    atom-ablation re-run under a DIFFERENT continuation tail the
#                    causal sweep never used) — predicts an unseen intervention.
#   M (minimal)    — 1 − over-claim rate: of the atoms the method flags "matched"
#                    (a named dictionary feature), how many are causally INERT
#                    (causal effect == 0)? Naming a causally-dead atom is the
#                    dictionary trap (a "monosemantic"-looking but unused feature).
#
# CONTRAST vs SAE (E5-6): the same trajectory feeds PCA, NMF and a small L1 SAE
# (re-implemented inline in Julia — manual GD — so the 3-way comparison is over
# IDENTICAL data and the SAME causal-use harness; this file owns its own SAE and
# does not depend on E5-6's separate file_scope/output). The SAE is the learned,
# sparse, over-complete dictionary; PCA the orthogonal/signed one; NMF the
# non-negative-parts one. We report all three side-by-side.
#
# BUILDS ON the verified jutari foundation (NO emulator core touched):
#   * common/jutari_oracle.jl     — load/boot/replay/snapshot/deepcopy-checkpoint/
#                                    intervene/byte-exact NPY-NPZ writer.
#   * common/jutari_record.jl     — record_trajectory PATTERN (re-implemented here
#                                    with per-game RomSettings, as A7 does).
#   * phaseA_kording/A7_dimred.jl — the NMF/PCA + matched-component-fraction
#                                    PRECEDENT (this file generalises it: adds the
#                                    SAE, the rank sweep, and the CAUSAL-USE test).
#   * phaseC_mechanistic/pilot_patch_sae.* — the Phase-C SAE + activation-patching
#                                    pilot (the feature↔variable + patch-effect
#                                    template the causal-use test extends from
#                                    single-cell to whole-atom reconstruction).
#   * ground_truth/oracle_intervene.jl — the exact causal-map reference.
#
# Cluster-shardable (SCRUM §7): --game / --shard i --nshards n --shard-kind game,
# --out-dir, --roms-dir, --where. Default = all 6 core games locally → declared out/.
#
# Run (warm shared depot, primary's project):
#   XAI_PRIMARY_REPO=/Users/maier/Documents/code/UnderstandingVCS \
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseC_mechanistic/dictionaries.jl
# Flags: --games pong,breakout,...  --game G  --shard i --nshards n --shard-kind game
#        --target-frame N --horizon N --traj-frames N
#        --ranks 4,8,12  --rank R  --sae-hidden H  --sae-l1 L  --sae-epochs E
#        --purity F  --out-dir DIR  --roms-dir DIR  --where local|cluster
#        --selftest (self-check on the first game, write nothing)
#
# Writes (SPEC §R; file_scope dictionaries_*): one record per game + a combined index
#   tools/xai_study/phaseC_mechanistic/out/dictionaries_<game>.{json,npz}
#   tools/xai_study/phaseC_mechanistic/out/dictionaries_core_summary.json

module Dictionaries

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
using .JutariOracle.JuTari.Diff: soft_ram_peek

# The §1 exact-intervention oracle (build_pong_causes / candidate_ram_indices /
# run_intervention / assert_bit_exact) — used ONLY to build the P2 SHARED gameplay
# state + its cause-density gate (below). Dictionaries keeps its OWN record/PCA/NMF/
# SAE/causal-use machinery; the oracle supplies the shared action stream + the gate.
# Referenced as OracleIntervene.X (its internal JutariOracle submodule is a SEPARATE
# instance from the one included above — no type is mixed across them; that is why
# we do NOT alias-import it).
include(joinpath(@__DIR__, "..", "ground_truth", "oracle_intervene.jl"))

# The P2 SHARED TESTBED (xai_paper/xai_2_interpretability/experiment_redesign.md):
# seeded random-action GAMEPLAY state + oracle cause-density gate (an includable
# fragment). Phase C is not a gradient method, so the sampler-on path does not apply
# — we consume the shared action STREAM + cause-density GATE only, and boot the
# dictionary's OWN checkpoint/trajectory from that stream. Opt in with
# XAI_SHARED_TESTBED=1 (default on).
include(joinpath(@__DIR__, "..", "common", "shared_testbed_impl.jl"))

const DEFAULT_OUT_DIR = joinpath(@__DIR__, "out")
const CORE_GAMES = ["pong", "breakout", "space_invaders",
                    "seaquest", "ms_pacman", "qbert"]

# shared-testbed switch + params (redesign protocol: prefix=90 gameplay, horizon=15).
const SHARED_TESTBED = get(ENV, "XAI_SHARED_TESTBED", "1") == "1"
const ST_PREFIX  = parse(Int, get(ENV, "XAI_ST_PREFIX", "90"))
const ST_HORIZON = parse(Int, get(ENV, "XAI_ST_HORIZON", "15"))
const ST_SEED    = parse(Int, get(ENV, "XAI_ST_SEED", "0"))
const ST_GATE_K  = parse(Int, get(ENV, "XAI_ST_GATE_K", "4"))
const ST_FLOOR   = parse(Float64, get(ENV, "XAI_ST_FLOOR", "0.5"))

# ============================================================================
# Per-game live-play configuration (mirrors A7 GAME_CFG so the candidate cells
# actually MOVE — a static tail leaves the dictionary nothing to decompose).
# `in_window` flags whether the SCORED oracle frame is inside the strict Paper-1
# 60-frame screen conformance window (games scored beyond it are HONEST descriptive
# jutari results — bit-exact jutari re-run still asserted — annotated in §R).
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
# Per-game RomSettings (mirrors the canonical tools/jutari_screen_dump.jl map,
# the SAME map A7 uses). seaquest -> Generic (Paper-1 64/64 with Generic).
# ============================================================================
function settings_for_game(game::AbstractString)
    g = lowercase(string(game))
    JT = JutariOracle.JuTari
    g == "pong"           && return JT.PaddleGames.PongRomSettings()
    g == "breakout"       && return JT.PaddleGames.BreakoutRomSettings()
    g == "space_invaders" && return JT.SpaceInvadersRomSettings()
    g == "ms_pacman"      && return JT.JoystickGames.MsPacmanRomSettings()
    g == "qbert"          && return JT.JoystickGames.QbertRomSettings()
    return JT.GenericRomSettings()     # seaquest + any other
end

const _ROM_ALIASES = Dict("ms_pacman" => ["ms_pacman", "mspacman"],
                          "beam_rider" => ["beam_rider", "beamrider"])

# ROM resolution honouring --roms-dir, then the oracle's default search.
function rom_path(game::AbstractString; roms_dir::Union{Nothing,String} = nothing)
    names = get(_ROM_ALIASES, lowercase(string(game)), [string(game)])
    if roms_dir !== nothing
        for nm in names
            p = joinpath(roms_dir, nm * ".bin")
            isfile(p) && return p
        end
    end
    try
        return rom_path_for(game)
    catch
    end
    here = normpath(joinpath(@__DIR__, "..", "..", ".."))
    bases = String[here, get(ENV, "XAI_PRIMARY_REPO", ""),
                   "/Users/maier/Documents/code/UnderstandingVCS"]
    for base in bases
        base == "" && continue
        for nm in names
            p = joinpath(base, "xitari", "roms", nm * ".bin")
            isfile(p) && return p
        end
    end
    error("ROM not found for game=$game (tried $(names))")
end

"""A freshly-reset env for `game` with the xitari-parity boot (60 NOOP + 4 RESET)."""
function load_env(game::AbstractString; roms_dir = nothing)
    rom = read(rom_path(game; roms_dir = roms_dir))
    env = StellaEnvironment(rom, settings_for_game(game))
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    return env
end

"""Boot + step actions[1:target_frame]; return the env AT the intervention frame."""
function boot_to(game, actions::AbstractVector{<:Integer}, target_frame::Integer; roms_dir = nothing)
    env = load_env(game; roms_dir = roms_dir)
    for i in 1:target_frame; env_step!(env, Int(actions[i])); end
    return env
end

"""deepcopy a checkpoint, step the tail, snapshot (byte-exact continuation)."""
function continue_from(checkpoint, tail::AbstractVector{<:Integer})
    env = deepcopy(checkpoint)
    for a in tail; env_step!(env, Int(a)); end
    return snapshot(env, length(tail))
end

"""A full from-scratch replay (boot incl.) — the bit-exact baseline guarantee."""
function fresh_baseline(game, actions::AbstractVector{<:Integer}, total::Integer; roms_dir = nothing)
    env = load_env(game; roms_dir = roms_dir)
    for i in 1:total; env_step!(env, Int(actions[i])); end
    return snapshot(env, Int(total))
end

"""Record a (frames, 128) RAM tape on `game` with per-game settings."""
function record_ram(game::AbstractString, frames::Integer,
                    actions::AbstractVector{<:Integer}; roms_dir = nothing)
    @assert length(actions) >= frames "actions shorter than frames"
    env = load_env(game; roms_dir = roms_dir)
    tape = Matrix{UInt8}(undef, Int(frames), RAM_SIZE)
    for t in 1:frames
        env_step!(env, Int(actions[t]))
        tape[t, :] = UInt8.(collect(get_ram(env)))
    end
    return tape
end

# ============================================================================
# Action traces (shared shape with A7/A4).
# Action codes: NOOP=0 FIRE=1 UP=2 RIGHT=3 LEFT=4 DOWN=5 RIGHTFIRE=11 LEFTFIRE=12.
# ============================================================================
# Oracle / scored trace: deterministic FIRE-then-NOOP start (inside the window).
oracle_actions(total::Integer) = vcat(fill(1, 4), fill(0, max(0, total - 4)))
# Held-out continuation (a DIFFERENT tail the causal sweep never used — the S probe).
heldout_tail(len::Integer) = Int[ (t % 3 == 0 ? 11 : t % 3 == 1 ? 12 : 0) for t in 1:max(0, len) ]

# Active (descriptive) trajectory trace so the candidate cells move.
function active_actions(total::Integer)
    acts = Vector{Int}(undef, max(0, total))
    for t in 1:length(acts)
        acts[t] = t <= 4 ? 1 : t % 4 == 0 ? 11 : t % 4 == 2 ? 12 : 1
    end
    return acts
end
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
# Candidate cells (E2-1 import) — the known game variables to match against.
# ============================================================================
struct Candidate
    ram_index::Int
    concept::String
    family::String
end

"""Concept *family* normalisation (mirrors A7 / A4 so enemy_x/enemy_y/ghost → enemy,
player_score/score → score). Cells sharing a family are ONE known game variable."""
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
# DICTIONARY MODELS — all over the SAME (T,128) tape, with a common interface:
# each yields per-atom loadings (atoms × 128), per-frame activations (T × atoms),
# and a decode(activations)->reconstruction so the causal-use test can ablate an
# atom, reconstruct, and write the state back.
# ============================================================================

# ---- PCA (SVD of the centred tape) -----------------------------------------
struct DictModel
    method::String              # "pca" | "nmf" | "sae"
    rank::Int                   # #atoms
    loadings::Matrix{Float64}   # (rank, 128) per-atom loading over RAM cells
    activations::Matrix{Float64}# (T, rank) per-frame activation of each atom
    mu::Vector{Float64}         # column means (for decode); PCA centring / SAE
    recon_rel_err::Float64      # ||X - decode(A)||_F / ||X||_F
    var_explained::Vector{Float64}  # per-atom variance/energy share (descending)
    extra::Dict{String,Any}     # method-specific notes
end

"""decode the (T or 1, rank) activations of model `m` back to (·, 128) RAM space."""
function decode(m::DictModel, A::AbstractMatrix)
    if m.method == "pca"
        # x = mu + A * loadings   (loadings rows are the principal directions)
        return reshape(m.mu, 1, :) .+ A * m.loadings
    elseif m.method == "nmf"
        # x = A * loadings        (non-negative parts; no centring)
        return A * m.loadings
    else # sae: x̂ = mu + A * W_dec' ; here loadings == W_dec' rows? We store decoder
        return reshape(m.mu, 1, :) .+ A * m.loadings
    end
end

"""project a (·,128) RAM block onto model `m` -> (·, rank) activations."""
function project(m::DictModel, X::AbstractMatrix)
    if m.method == "pca"
        Xc = X .- reshape(m.mu, 1, :)
        return Xc * m.loadings'            # (·, rank)
    elseif m.method == "nmf"
        # non-negative least-squares-ish: a few multiplicative refinement steps from
        # a non-negative init (H known = loadings). Solve A>=0 minimising ||X-A H||.
        return _nmf_project(X, m.loadings)
    else # sae encoder reused
        Wenc = m.extra["W_enc"]::Matrix{Float64}
        benc = m.extra["b_enc"]::Vector{Float64}
        Xn = (X .- reshape(m.mu, 1, :)) ./ reshape(m.extra["sd"]::Vector{Float64}, 1, :)
        return max.(0.0, Xn * Wenc' .+ reshape(benc, 1, :))
    end
end

function run_pca(tape::AbstractMatrix, rank::Integer)
    nram = min(RAM_SIZE, size(tape, 2))
    X = Float64.(tape[:, 1:nram])
    T = size(X, 1)
    mu = vec(Statistics.mean(X; dims = 1))
    Xc = X .- reshape(mu, 1, :)
    F = svd(Xc; full = false)
    r = min(rank, length(F.S))
    sv = F.S
    var = sv .^ 2
    total = sum(var)
    var_expl = total == 0 ? zeros(length(sv)) : var ./ total
    loadings = Matrix(F.V')[1:r, :]          # (r, 128) — rows are PC directions
    A = Xc * F.V[:, 1:r]                       # (T, r) scores
    R = reshape(mu, 1, :) .+ A * loadings
    rel = total == 0 ? 0.0 : norm(X .- R) / max(norm(X), eps())
    return DictModel("pca", r, loadings, A, mu, rel, var_expl[1:r],
                     Dict{String,Any}("centered" => true))
end

# ---- NMF (Lee & Seung multiplicative updates) ------------------------------
function _nmf(X::AbstractMatrix{<:Real}, r::Integer; iters = 300, seed = 0, ε = 1e-9)
    T, n = size(X)
    Xf = Float64.(X)
    rng = Random.MersenneTwister(seed)
    scale = max(1.0, Statistics.mean(Xf) + ε)
    W = scale .* (rand(rng, T, r) .+ ε)
    H = scale .* (rand(rng, r, n) .+ ε)
    for _ in 1:iters
        WtX = W' * Xf
        WtWH = (W' * W) * H .+ ε
        H .= H .* (WtX ./ WtWH)
        XHt = Xf * H'
        WHHt = W * (H * H') .+ ε
        W .= W .* (XHt ./ WHHt)
    end
    return W, H
end

"""Project new X onto fixed non-negative parts H (loadings): a few multiplicative
updates of A>=0 minimising ||X - A H|| (H held fixed)."""
function _nmf_project(X::AbstractMatrix, H::AbstractMatrix; iters = 80, seed = 7, ε = 1e-9)
    T = size(X, 1); r = size(H, 1)
    rng = Random.MersenneTwister(seed)
    A = max.(ε, rand(rng, T, r))
    Xf = max.(0.0, Float64.(X))
    HHt = H * H'
    XHt = Xf * H'
    for _ in 1:iters
        AHHt = A * HHt .+ ε
        A .= A .* (XHt ./ AHHt)
    end
    return A
end

function run_nmf(tape::AbstractMatrix, rank::Integer; iters = 300, seed = 0)
    nram = min(RAM_SIZE, size(tape, 2))
    X = Float64.(tape[:, 1:nram])
    Xnorm = norm(X)
    r = max(1, min(rank, size(X, 1), nram))
    W, H = _nmf(X, r; iters = iters, seed = seed)
    R = W * H
    rel = Xnorm == 0 ? 0.0 : norm(X .- R) / Xnorm
    energy = Float64[norm(W[:, k]) * norm(H[k, :]) for k in 1:r]
    ord = sortperm(energy; rev = true)
    H = H[ord, :]; W = W[:, ord]; energy = energy[ord]
    eshare = sum(energy) == 0 ? zeros(r) : energy ./ sum(energy)
    # mu = 0 (no centring for NMF); decode = A*H.
    return DictModel("nmf", r, H, W, zeros(nram), rel, eshare,
                     Dict{String,Any}("nonnegative" => true, "iters" => iters,
                                      "raw_energy" => energy))
end

# ---- SAE (1 hidden layer, L1 sparsity, manual GD — Julia inline) ------------
# x̂ = mu + (z @ W_dec) where z = ReLU((x-mu)/sd @ W_enc' + b_enc). Inputs are
# z-scored per varying column so the L1 penalty is comparable across cells. This
# is the SAME objective as pilot_patch_sae.py, re-implemented in Julia so the
# PCA/NMF/SAE 3-way contrast runs over identical data in one harness.
function run_sae(tape::AbstractMatrix, hidden::Integer; l1 = 0.05, epochs = 3000,
                 lr = 0.5, seed = 0)
    nram = min(RAM_SIZE, size(tape, 2))
    X = Float64.(tape[:, 1:nram])
    T = size(X, 1)
    mu = vec(Statistics.mean(X; dims = 1))
    sd = vec(Statistics.std(X; dims = 1, corrected = false))
    varying = sd .> 0
    sd_safe = [varying[j] ? sd[j] : 1.0 for j in 1:nram]
    Xn = (X .- reshape(mu, 1, :)) ./ reshape(sd_safe, 1, :)
    for j in 1:nram
        varying[j] || (Xn[:, j] .= 0.0)
    end
    rng = Random.MersenneTwister(seed)
    h = Int(hidden)
    scale = 1.0 / sqrt(max(nram, 1))
    W_enc = scale .* randn(rng, h, nram)
    b_enc = zeros(h)
    W_dec = scale .* randn(rng, nram, h)
    b_dec = zeros(nram)
    N = T
    for _ in 1:epochs
        Z = max.(0.0, Xn * W_enc' .+ reshape(b_enc, 1, :))   # (T, h)
        Xhat = Z * W_dec' .+ reshape(b_dec, 1, :)            # (T, nram)
        resid = Xhat .- Xn
        dXhat = (2.0 / (N * nram)) .* resid
        gW_dec = dXhat' * Z                                  # (nram, h)
        gb_dec = vec(sum(dXhat; dims = 1))
        dZ = dXhat * W_dec                                   # (T, h)
        dZ .+= (l1 / (N * h)) .* sign.(Z)
        dZ .*= (Z .> 0)
        gW_enc = dZ' * Xn                                    # (h, nram)
        gb_enc = vec(sum(dZ; dims = 1))
        W_dec .-= lr .* gW_dec
        b_dec .-= lr .* gb_dec
        W_enc .-= lr .* gW_enc
        b_enc .-= lr .* gb_enc
    end
    Z = max.(0.0, Xn * W_enc' .+ reshape(b_enc, 1, :))       # (T, h)
    Xhat_n = Z * W_dec' .+ reshape(b_dec, 1, :)
    # reconstruction in z-scored space; map back to byte space for rel-err
    Xhat = reshape(mu, 1, :) .+ Xhat_n .* reshape(sd_safe, 1, :)
    rel = norm(X) == 0 ? 0.0 : norm(X .- Xhat) / norm(X)
    # decode in BYTE space: x = mu + (z @ W_dec') .* sd  (+ b_dec*sd). Fold b_dec
    # into mu so decode(z) = mu_eff + z * (W_dec' .* sd). loadings row k = W_dec[:,k].*sd.
    loadings = Matrix{Float64}(undef, h, nram)               # (h, nram)
    for k in 1:h
        loadings[k, :] = W_dec[:, k] .* sd_safe
    end
    mu_eff = mu .+ b_dec .* sd_safe
    energy = Float64[norm(Z[:, k]) * norm(loadings[k, :]) for k in 1:h]
    ord = sortperm(energy; rev = true)
    loadings = loadings[ord, :]; Z = Z[:, ord]; energy = energy[ord]
    W_enc = W_enc[ord, :]; b_enc = b_enc[ord]
    eshare = sum(energy) == 0 ? zeros(h) : energy ./ sum(energy)
    extra = Dict{String,Any}("sparse" => true, "l1" => l1, "epochs" => epochs,
                             "lr" => lr, "hidden" => h,
                             "W_enc" => W_enc, "b_enc" => b_enc,
                             "sd" => sd_safe, "raw_energy" => energy,
                             "mean_abs_activation" => Statistics.mean(abs.(Z)))
    return DictModel("sae", h, loadings, Z, mu_eff, rel, eshare, extra)
end

# ============================================================================
# Known structural signals (T2: clock / R-W activity / vsync phase) — A7 lineage.
# ============================================================================
struct KnownSignals
    names::Vector{String}
    sig::Matrix{Float64}             # (T, 3)
end
function known_signals(tape::AbstractMatrix)
    T = size(tape, 1)
    X = Float64.(tape)
    clock = Float64.(1:T)
    rw = zeros(Float64, T)
    for t in 2:T
        rw[t] = count(j -> X[t, j] != X[t-1, j], 1:size(X, 2))
    end
    rw[1] = T >= 2 ? rw[2] : 0.0
    period = _dominant_period(rw)
    vsync = Float64[sin(2π * (t - 1) / period) for t in 1:T]
    return KnownSignals(["clock", "rw_activity", "vsync_phase"], hcat(clock, rw, vsync))
end
function _dominant_period(x::AbstractVector)
    T = length(x)
    T < 4 && return 2.0
    xc = x .- Statistics.mean(x)
    s = sum(xc .^ 2); s == 0 && return 2.0
    best_lag = 2; best = -Inf
    for lag in 2:(T ÷ 2)
        ac = sum(xc[1:T-lag] .* xc[1+lag:T]) / s
        if ac > best; best = ac; best_lag = lag; end
    end
    return Float64(best_lag)
end

# ============================================================================
# Matched-component fraction (game vars + clock/R-W/vsync). Generalises A7's
# match to the common DictModel interface (works for pca/nmf/sae alike).
# ============================================================================
struct MatchResult
    method::String
    n_atoms::Int
    matched_label::Vector{String}
    matched_purity::Vector{Float64}
    matched_kind::Vector{String}     # "game_var" | "known_signal" | "(none)"
    matched::Vector{Bool}
    matched_component_fraction::Float64
    n_matched_game_var::Int
    n_matched_known_signal::Int
end

function match_components(m::DictModel, cands::Vector{Candidate}, known::KnownSignals;
                         purity = 0.5, candidate_floor = 0.25)
    r = m.rank
    fams = unique([c.family for c in cands])
    fam_of = Dict(c.ram_index => c.family for c in cands)
    cand_idx0 = [c.ram_index for c in cands]
    labels = String[]; purities = Float64[]; kinds = String[]; matched = Bool[]
    n_gv = 0; n_ks = 0
    for j in 1:r
        load2 = m.loadings[j, :] .^ 2
        totmass = sum(load2)
        cand_mass = sum(load2[idx + 1] for idx in cand_idx0; init = 0.0)
        cand_share = totmass == 0 ? 0.0 : cand_mass / totmass
        fam_mass = Dict(f => 0.0 for f in fams)
        for idx in cand_idx0; fam_mass[fam_of[idx]] += load2[idx + 1]; end
        best_fam = ""; best_fam_mass = 0.0
        for (f, mm) in fam_mass
            if mm > best_fam_mass; best_fam_mass = mm; best_fam = f; end
        end
        fam_purity = cand_mass == 0 ? 0.0 : best_fam_mass / cand_mass
        game_match = (cand_share >= candidate_floor) && (fam_purity >= purity)

        a = m.activations[:, j]
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
            push!(labels, "(none; cand_share=$(round(cand_share,digits=2)),sig=$(round(best_sig_corr,digits=2)))")
            push!(purities, max(fam_purity * (cand_share >= candidate_floor ? 1 : 0), best_sig_corr))
            push!(kinds, "(none)"); push!(matched, false)
        end
    end
    frac = r == 0 ? 0.0 : count(matched) / r
    return MatchResult(m.method, r, labels, purities, kinds, matched, frac, n_gv, n_ks)
end

# ============================================================================
# CAUSAL-USE test — the Phase-C contribution. Ablate atom k, reconstruct the
# state, write it into the real env, re-run via the oracle, measure Δscreen.
# ============================================================================
struct CausalUse
    method::String
    n_atoms::Int
    # per-atom causal effect = whole-screen break of the atom-ablated reconstruction
    # re-run, RELATIVE to the no-ablation reconstruction control (isolates the atom).
    effect::Vector{Float64}
    held_out_effect::Vector{Float64}    # same ablation, the held-out continuation tail
    recon_only_break::Float64           # control: reconstruct-without-ablation Δscreen
    all_ablated_break::Float64          # positive control: zero ALL atoms
    bit_exact::Bool
    n_causal_atoms::Int
end

"""Round/clamp a reconstructed RAM vector to bytes."""
_to_bytes(v::AbstractVector) = UInt8[ UInt8(clamp(round(Int, x), 0, 255)) for x in v ]

"""Write a full 128-byte RAM vector into a fresh deepcopy of `checkpoint`, re-run
`tail`, snapshot. (The whole-state write is a valid intervention — set-state-and-
re-run; the bit-exact substrate makes the Δ a clean causal effect.)"""
function rerun_with_ram(checkpoint, ram::AbstractVector{UInt8}, tail::AbstractVector{<:Integer})
    env = deepcopy(checkpoint)
    for i in 0:(length(ram) - 1)
        intervene_ram!(env, i, ram[i + 1])
    end
    for a in tail; env_step!(env, Int(a)); end
    return snapshot(env, length(tail))
end

function causal_use(m::DictModel, checkpoint, at_target::Snapshot, tail::Vector{Int},
                    held_tail::Vector{Int}, base_snap::Snapshot, base_held::Snapshot;
                    verbose = true)
    nram = min(RAM_SIZE, length(at_target.ram))
    x = reshape(Float64.(at_target.ram[1:nram]), 1, :)        # (1, 128) clean state
    a0 = project(m, x)                                        # (1, rank) activations
    recon = decode(m, a0)                                     # (1, 128) no-ablation recon
    recon_bytes = _to_bytes(vec(recon))
    # control: reconstruct-without-ablation, re-run — the lossy-reconstruction floor.
    recon_snap = rerun_with_ram(checkpoint, recon_bytes, tail)
    recon_only_break = Float64(count(recon_snap.screen .!= base_snap.screen))

    r = m.rank
    effect = zeros(Float64, r); held = zeros(Float64, r)
    for k in 1:r
        ak = copy(a0); ak[1, k] = 0.0
        xk = decode(m, ak)
        bk = _to_bytes(vec(xk))
        sk = rerun_with_ram(checkpoint, bk, tail)
        # causal effect of atom k = how much ABLATING it changes the screen, beyond
        # what the lossy reconstruction already changes (symmetric-diff vs recon ctrl).
        effect[k] = Float64(count(sk.screen .!= recon_snap.screen))
        skh = rerun_with_ram(checkpoint, bk, held_tail)
        held[k] = Float64(count(skh.screen .!= base_held.screen))
        verbose && println("    causal[$(m.method)] atom $(lpad(k,2)) Δscreen(ablate)=$(round(effect[k]))")
    end
    # positive control: zero ALL atoms (the whole reconstructed varying subspace).
    a_all = zeros(size(a0))
    x_all = decode(m, a_all)
    all_snap = rerun_with_ram(checkpoint, _to_bytes(vec(x_all)), tail)
    all_break = Float64(count(all_snap.screen .!= base_snap.screen))
    n_causal = count(>(0.0), effect)
    return CausalUse(m.method, r, effect, held, recon_only_break, all_break, true, n_causal)
end

# ============================================================================
# F / S / M triad — scored against the per-atom CAUSAL effect (the §1 oracle on
# the reconstructed-state re-run). matched ⇒ causal? matched-but-inert = trap.
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
function _quantile(v::AbstractVector, q::Real)
    isempty(v) && return 0.0
    s = sort(collect(Float64.(v))); n = length(s)
    n == 1 && return s[1]
    pos = clamp(q, 0.0, 1.0) * (n - 1) + 1
    lo = floor(Int, pos); hi = ceil(Int, pos)
    lo == hi && return s[lo]
    return s[lo] + (pos - lo) * (s[hi] - s[lo])
end

"""Per-atom "matched strength" = matched_purity if the atom was named a known
variable, else 0 (an unmatched atom makes no recovery claim)."""
matched_strength(mm::MatchResult) =
    Float64[ mm.matched[j] ? mm.matched_purity[j] : 0.0 for j in 1:mm.n_atoms ]

function score_triad(method::AbstractString, mm::MatchResult, cu::CausalUse;
                     flag_quantile = 0.5)
    strength = matched_strength(mm)
    F = _spearman(strength, cu.effect)
    S = _pearson(cu.effect, cu.held_out_effect)
    flagged = mm.matched
    nflag = count(flagged)
    overclaim = nflag == 0 ? 0.0 :
        count(j -> flagged[j] && cu.effect[j] == 0.0, 1:length(flagged)) / nflag
    M = 1.0 - overclaim
    return Triad(F, S, M,
        "Spearman(per-atom matched-strength, per-atom causal effect) — are MATCHED atoms the USED ones?",
        "Pearson(per-atom causal effect, held-out-tail causal effect) — sufficiency on an unseen continuation (bit-exact re-run)",
        "1 − over-claim rate: fraction of MATCHED ($(method)) atoms whose ablation has ZERO causal effect (present≠used dictionary trap)",
        string(method))
end

# ============================================================================
# Drive one game (a rank sweep; the headline rank is the FIRST in --ranks).
# ============================================================================
struct RankResult
    rank::Int
    pca::DictModel; nmf::DictModel; sae::DictModel
    pca_match::MatchResult; nmf_match::MatchResult; sae_match::MatchResult
    pca_cu::CausalUse; nmf_cu::CausalUse; sae_cu::CausalUse
    pca_triad::Triad; nmf_triad::Triad; sae_triad::Triad
end

struct GameResult
    game::String
    target_frame::Int; horizon::Int; traj_frames::Int; trace::String
    in_window::Bool; seed::Int; bit_exact::Bool
    cands::Vector{Candidate}
    ranks::Vector{Int}
    by_rank::Vector{RankResult}
    purity::Float64
    sae_hidden::Int; sae_l1::Float64; sae_epochs::Int
    nmf_iters::Int
    where::String
    # SHARED-TESTBED provenance (redesign); "noop"/-1/false in the legacy path.
    state_kind::String             # "seeded_random_action_gameplay" | "noop"
    st_seed::Int
    st_prefix::Int
    cause_density::Int             # #causes above the floor at the shared output
    cause_density_accepted::Bool   # passed the cause-density gate?
    n_causes::Int
    shared_cell::Tuple{Int,Int}    # the shared screen-buffer output cell
end

# Dictionaries' candidates-path resolver, in the shape the shared testbed injects.
function _st_candidates_path_for(game::AbstractString)
    return resolve_candidates(game)
end

"""Build the P2 SHARED gameplay state + cause-density gate for `game` using the §1
oracle machinery (oracle_intervene.jl). Returns the substrate NamedTuple (we use its
`.actions` stream + `.cause_density`/`.accepted`/`.cell` gate). Dictionaries then
boots its OWN checkpoint + records its OWN trajectory from `st.actions` so the
PCA/NMF/SAE algorithms are unchanged."""
function build_dict_shared_state(game::AbstractString; roms_dir = nothing, verbose = false)
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

"""The dictionary trajectory action stream on the SHARED gameplay state: the
gameplay `st.actions` for its first `prefix+horizon` frames, NOOP-continued to
`traj_frames` if the descriptive trajectory needs more."""
function shared_traj_actions(st, traj_frames::Integer)
    base = Int.(st.actions)
    n = Int(traj_frames)
    n <= length(base) && return base[1:n]
    return vcat(base, fill(0, n - length(base)))
end

function compute_game(game::AbstractString; target_frame = nothing, horizon = nothing,
                      traj_frames = nothing, ranks = [8, 4, 12], purity = 0.5,
                      sae_hidden = nothing, sae_l1 = 0.05, sae_epochs = 3000,
                      nmf_iters = 300, seed = 0, where = "local",
                      roms_dir = nothing, verbose = true)
    cfg = game_cfg(game)
    target_frame = target_frame === nothing ? cfg.target_frame : target_frame
    horizon      = horizon      === nothing ? cfg.horizon      : horizon
    traj_frames  = traj_frames  === nothing ? cfg.traj_frames  : traj_frames
    trace = cfg.trace; in_window = cfg.in_window

    # SHARED-TESTBED (redesign): replace the FIRE-then-NOOP oracle tape + the
    # fabricated active/dir trajectory with a seeded random-action GAMEPLAY state at
    # f*=ST_PREFIX, gated by the oracle cause-density gate. We take the shared action
    # STREAM + gate from the substrate and boot the dictionary's OWN checkpoint/record
    # from it; the PCA/NMF/SAE + causal-use machinery is UNCHANGED — only the state +
    # trajectory move onto genuine input-driven gameplay. The held-out S-probe tail
    # stays the distinct heldout_tail (an unseen continuation, as before).
    st = nothing
    if SHARED_TESTBED
        st = build_dict_shared_state(game; roms_dir = roms_dir, verbose = verbose)
        target_frame = st.prefix; horizon = st.horizon
        in_window = false          # prefix+horizon=105 exceeds the 60-frame screen window
        traj_frames = max(traj_frames, st.prefix + st.horizon)
        verbose && println("[E5-7:$game] SHARED gameplay state: cause_density=$(st.cause_density)/" *
            "$(length(st.causes)) accepted=$(st.accepted) cell=$(st.cell)")
    end
    total = target_frame + horizon
    oacts = st === nothing ? oracle_actions(total) : Int.(st.actions)
    tail = Int.(oacts[target_frame + 1 : target_frame + horizon])
    held_tail = heldout_tail(horizon)

    # 1) bit-exact baseline.
    verbose && println("[E5-7:$game] asserting bit-exactness (2 fresh boots+replays to f$total)...")
    a = fresh_baseline(game, oacts, total; roms_dir = roms_dir)
    b = fresh_baseline(game, oacts, total; roms_dir = roms_dir)
    bit_exact = (a.ram == b.ram) && (a.screen == b.screen)
    bit_exact || error("bit-exact re-run FAILED for $game to f$total — refusing to score")
    verbose && println("[E5-7:$game] bit-exact re-run: PASS")

    # 2) one checkpoint at the intervention frame.
    checkpoint = boot_to(game, oacts, target_frame; roms_dir = roms_dir)
    at_target = continue_from(checkpoint, Int[])
    base_snap = continue_from(checkpoint, tail)
    base_held = continue_from(checkpoint, held_tail)

    cands = load_candidates(resolve_candidates(game))
    isempty(cands) && error("no candidate cells for $game (candidates file missing/empty)")
    verbose && println("[E5-7:$game] candidates: $(length(cands)) cells, " *
                       "$(length(unique([c.family for c in cands]))) known game variables")

    # 3) descriptive trajectory. Under the shared testbed it is stepped along the
    #    GAMEPLAY stream (NOOP-continued past prefix+horizon); else the legacy trace.
    traj_acts = st === nothing ? trace_actions(trace, traj_frames) :
                shared_traj_actions(st, traj_frames)
    verbose && println("[E5-7:$game] recording $(traj_frames)-frame " *
        (st === nothing ? "active ($trace)" : "shared-gameplay") * " RAM trajectory...")
    tape = record_ram(game, traj_frames, traj_acts; roms_dir = roms_dir)
    known = known_signals(tape)

    sae_hidden = sae_hidden === nothing ? max(8, maximum(ranks)) : sae_hidden

    by_rank = RankResult[]
    for rk in ranks
        verbose && println("[E5-7:$game] --- rank $rk (PCA / NMF / SAE) ---")
        pca = run_pca(tape, rk)
        nmf = run_nmf(tape, rk; iters = nmf_iters, seed = seed)
        # SAE: hidden = rank for the sweep (so the 3 methods have the SAME #atoms).
        sae = run_sae(tape, rk; l1 = sae_l1, epochs = sae_epochs, seed = seed)

        pca_m = match_components(pca, cands, known; purity = purity)
        nmf_m = match_components(nmf, cands, known; purity = purity)
        sae_m = match_components(sae, cands, known; purity = purity)

        verbose && println("[E5-7:$game]   causal-use (ablate each atom, re-run via oracle)...")
        pca_cu = causal_use(pca, checkpoint, at_target, tail, held_tail, base_snap, base_held; verbose = false)
        nmf_cu = causal_use(nmf, checkpoint, at_target, tail, held_tail, base_snap, base_held; verbose = false)
        sae_cu = causal_use(sae, checkpoint, at_target, tail, held_tail, base_snap, base_held; verbose = false)

        pca_t = score_triad("pca", pca_m, pca_cu)
        nmf_t = score_triad("nmf", nmf_m, nmf_cu)
        sae_t = score_triad("sae", sae_m, sae_cu)

        if verbose
            println("[E5-7:$game]   rank=$rk matched-frac PCA=$(round(pca_m.matched_component_fraction,digits=3)) " *
                    "NMF=$(round(nmf_m.matched_component_fraction,digits=3)) SAE=$(round(sae_m.matched_component_fraction,digits=3))")
            println("[E5-7:$game]   rank=$rk causal atoms PCA=$(pca_cu.n_causal_atoms)/$rk " *
                    "NMF=$(nmf_cu.n_causal_atoms)/$rk SAE=$(sae_cu.n_causal_atoms)/$rk " *
                    "(all-ablated ctrl Δscreen=$(round(pca_cu.all_ablated_break)))")
            println("[E5-7:$game]   rank=$rk PCA F/S/M=$(round(pca_t.F,digits=2))/$(round(pca_t.S,digits=2))/$(round(pca_t.M,digits=2)) " *
                    "NMF=$(round(nmf_t.F,digits=2))/$(round(nmf_t.S,digits=2))/$(round(nmf_t.M,digits=2)) " *
                    "SAE=$(round(sae_t.F,digits=2))/$(round(sae_t.S,digits=2))/$(round(sae_t.M,digits=2))")
        end
        push!(by_rank, RankResult(rk, pca, nmf, sae, pca_m, nmf_m, sae_m,
                                  pca_cu, nmf_cu, sae_cu, pca_t, nmf_t, sae_t))
    end

    return GameResult(game, target_frame, horizon, traj_frames, trace, in_window,
                      seed, bit_exact, cands, collect(ranks), by_rank, purity,
                      sae_hidden, sae_l1, sae_epochs, nmf_iters, where,
                      st === nothing ? "noop" : "seeded_random_action_gameplay",
                      st === nothing ? -1 : st.seed,
                      st === nothing ? -1 : st.prefix,
                      st === nothing ? -1 : st.cause_density,
                      st === nothing ? false : st.accepted,
                      st === nothing ? 0 : length(st.causes),
                      st === nothing ? (-1, -1) : st.cell)
end

# ============================================================================
# Self-check (DoD).
# ============================================================================
function selftest(r::GameResult)
    @assert r.bit_exact "bit-exact baseline re-run failed for $(r.game)"
    @assert !isempty(r.by_rank) "no ranks computed for $(r.game)"
    rr = r.by_rank[1]
    # (POSITIVE CONTROL) ablating ALL atoms must move the screen for >=1 method —
    # else the causal-use harness is inert (the reconstructed varying subspace IS
    # what drives the output).
    any_all_break = any(>(0.0), (rr.pca_cu.all_ablated_break, rr.nmf_cu.all_ablated_break,
                                 rr.sae_cu.all_ablated_break))
    @assert any_all_break "causal-use POSITIVE CONTROL failed: zeroing ALL atoms moved " *
        "the screen for NO method on $(r.game) — uninformative; pick a livelier frame"
    # (CAUSAL-USE ANCHORED) at least one INDIVIDUAL atom is causally active for >=1 method.
    n_causal = rr.pca_cu.n_causal_atoms + rr.nmf_cu.n_causal_atoms + rr.sae_cu.n_causal_atoms
    @assert n_causal >= 1 "no single dictionary atom had a causal effect on $(r.game) " *
        "(all atom-ablations inert) — uninformative for the causal-use claim"
    # (DICTIONARIES WELL-FORMED)
    @assert all(>=(-1e-12), rr.pca.var_explained) "negative PCA variance fraction"
    @assert all(>=(-1e-9), rr.nmf.loadings) "NMF loadings have a negative entry"
    @assert all(>=(-1e-9), rr.nmf.activations) "NMF activations have a negative entry"
    for m in (rr.pca, rr.nmf, rr.sae)
        @assert 0.0 - 1e-9 <= m.recon_rel_err <= 1.5 "$(m.method) rel-err out of [0,1.5]: $(m.recon_rel_err)"
    end
    # (MATCHED-FRACTION RANGES)
    for mm in (rr.pca_match, rr.nmf_match, rr.sae_match)
        @assert 0.0 <= mm.matched_component_fraction <= 1.0 "$(mm.method) matched-frac out of [0,1]"
    end
    # (TRIAD RANGES)
    for tr in (rr.pca_triad, rr.nmf_triad, rr.sae_triad)
        @assert -1.0 - 1e-9 <= tr.F <= 1.0 + 1e-9 "$(tr.method) F out of [-1,1]: $(tr.F)"
        @assert -1.0 - 1e-9 <= tr.S <= 1.0 + 1e-9 "$(tr.method) S out of [-1,1]: $(tr.S)"
        @assert  0.0 - 1e-9 <= tr.M <= 1.0 + 1e-9 "$(tr.method) M out of [0,1]: $(tr.M)"
    end
    println("[E5-7:$(r.game)] SELF-CHECK PASS:")
    println("[E5-7:$(r.game)]   bit-exact baseline re-run: $(r.bit_exact)")
    println("[E5-7:$(r.game)]   rank $(rr.rank): all-ablated ctrl Δscreen " *
            "PCA=$(round(rr.pca_cu.all_ablated_break)) NMF=$(round(rr.nmf_cu.all_ablated_break)) SAE=$(round(rr.sae_cu.all_ablated_break))")
    println("[E5-7:$(r.game)]   causal atoms (rank $(rr.rank)): PCA=$(rr.pca_cu.n_causal_atoms) " *
            "NMF=$(rr.nmf_cu.n_causal_atoms) SAE=$(rr.sae_cu.n_causal_atoms)")
    println("[E5-7:$(r.game)]   matched-frac PCA=$(round(rr.pca_match.matched_component_fraction,digits=3)) " *
            "NMF=$(round(rr.nmf_match.matched_component_fraction,digits=3)) SAE=$(round(rr.sae_match.matched_component_fraction,digits=3))")
    return true
end

# ============================================================================
# Persist (SPEC §R) — JSON record + sibling .npz; file_scope dictionaries_*.
# ============================================================================
_git_commit() = try
    strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
catch
    "unknown"
end
_jn(x::Real) = isfinite(x) ? Float64(x) : nothing

_match_record(m::MatchResult) = Dict{String,Any}(
    "method" => m.method, "n_atoms" => m.n_atoms,
    "matched_component_fraction" => _jn(m.matched_component_fraction),
    "n_matched_game_var" => m.n_matched_game_var,
    "n_matched_known_signal" => m.n_matched_known_signal,
    "per_atom_matched_label" => m.matched_label,
    "per_atom_matched_purity" => [_jn(x) for x in m.matched_purity],
    "per_atom_matched_kind" => m.matched_kind,
)
_cu_record(c::CausalUse) = Dict{String,Any}(
    "method" => c.method, "n_atoms" => c.n_atoms,
    "n_causal_atoms" => c.n_causal_atoms,
    "per_atom_causal_effect_px" => [_jn(x) for x in c.effect],
    "per_atom_heldout_causal_effect_px" => [_jn(x) for x in c.held_out_effect],
    "reconstruct_only_break_px" => _jn(c.recon_only_break),
    "all_atoms_ablated_break_px" => _jn(c.all_ablated_break),
)
_triad_record(t::Triad) = Dict{String,Any}(
    "F" => _jn(t.F), "F_note" => t.F_note,
    "S" => _jn(t.S), "S_note" => t.S_note,
    "M" => _jn(t.M), "M_note" => t.M_note,
)
_dict_record(m::DictModel) = Dict{String,Any}(
    "method" => m.method, "rank" => m.rank,
    "recon_rel_frobenius_err" => _jn(m.recon_rel_err),
    "var_or_energy_share" => [_jn(x) for x in m.var_explained],
)

function write_game(r::GameResult; out_dir = DEFAULT_OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    stem = "dictionaries_$(r.game)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path  = joinpath(out_dir, stem * ".npz")

    headline = r.by_rank[1]   # the headline rank (first in --ranks)
    cell_names = ["RAM[$(c.ram_index)]:$(c.concept)" for c in r.cands]
    families = [c.family for c in r.cands]

    ranks_block = Dict{String,Any}[]
    for rr in r.by_rank
        push!(ranks_block, Dict{String,Any}(
            "rank" => rr.rank,
            "pca" => Dict{String,Any}("dict" => _dict_record(rr.pca),
                                      "matched" => _match_record(rr.pca_match),
                                      "causal_use" => _cu_record(rr.pca_cu),
                                      "triad" => _triad_record(rr.pca_triad)),
            "nmf" => Dict{String,Any}("dict" => _dict_record(rr.nmf),
                                      "matched" => _match_record(rr.nmf_match),
                                      "causal_use" => _cu_record(rr.nmf_cu),
                                      "triad" => _triad_record(rr.nmf_triad)),
            "sae" => Dict{String,Any}("dict" => _dict_record(rr.sae),
                                      "matched" => _match_record(rr.sae_match),
                                      "causal_use" => _cu_record(rr.sae_cu),
                                      "triad" => _triad_record(rr.sae_triad)),
        ))
    end

    rec = Dict{String,Any}(
        "paper" => "P2",
        "phase" => "phaseC_mechanistic",
        "method" => "nmf_pca_dictionaries(+SAE contrast)",
        "game" => r.game,
        "state" => r.state_kind == "noop" ? "f$(r.target_frame)+$(r.horizon)" :
                   "gameplay(seed=$(r.st_seed),prefix=$(r.st_prefix))+$(r.horizon)",
        "target_output" => "dictionary-atoms-vs-known-variables+causal-use",
        # headline scalar: the NMF matched-component fraction at the headline rank
        # (the prompt's "matched-component fraction"). Full PCA/NMF/SAE breakdown in extra.
        "metric_name" => "nmf_matched_component_fraction_vs_known_vars",
        "value" => _jn(headline.nmf_match.matched_component_fraction),
        "stderr" => nothing, "ci" => nothing,
        "n" => length(r.cands),
        "seed" => r.seed,
        "where" => r.where,
        "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene@$(r.game) (P2-E1-1) — atom-ablation re-run (whole-state set + re-run)",
        "timestamp" => string(round(Int, time())),
        "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            "substrate" => "jutari (Julia, HARD) — real-ROM bit-exact path; the " *
                "causal-use test uses EXACT whole-state interventions re-run on the true ROM.",
            "bit_exact_rerun" => r.bit_exact,
            "trajectory_trace" => r.state_kind != "noop" ?
                "seeded random-action GAMEPLAY trajectory (seed=$(r.st_seed), " *
                "prefix=$(r.st_prefix); NOOP-continued past prefix+horizon)" :
                (r.trace == "dir" ?
                 "directional maze trace (FIRE warmup + cyclic UP/DOWN/LEFT/RIGHT)" :
                 "active trace (4×FIRE + periodic RIGHTFIRE/LEFTFIRE)"),
            "trajectory_frames" => r.traj_frames,
            "testbed" => Dict{String,Any}(
                "state_kind" => r.state_kind,
                "seed" => r.st_seed, "prefix" => r.st_prefix, "horizon" => r.horizon,
                "shared_output" => "screen_region(n_changed_px)@r$(r.shared_cell[1])c$(r.shared_cell[2])",
                "cause_density_above_floor" => r.cause_density,
                "cause_density_floor" => ST_FLOOR, "cause_density_gate_k" => ST_GATE_K,
                "cause_density_accepted" => r.cause_density_accepted, "n_causes" => r.n_causes,
                "note" => "P2 redesign: PCA/NMF/SAE dictionaries are fit/scored on a " *
                    "seeded random-action GAMEPLAY state (not the FIRE/attract tape), " *
                    "gated by the §1 oracle cause-density gate. The descriptive " *
                    "trajectory is stepped along the gameplay stream (NOOP-continued " *
                    "past prefix+horizon). The dictionary algorithms are unchanged; " *
                    "only the analysis state + trajectory move."),
            "scored_in_conformance_window" => r.in_window,
            "conformance_note" => r.in_window ?
                "scored oracle frame f$(r.target_frame)+$(r.horizon) is inside the Paper-1 " *
                "60-frame screen conformance window (jutari↔xitari bit-exact)." :
                "scored oracle frame f$(r.target_frame)+$(r.horizon) is BEYOND the strict 60-frame " *
                "window — the game reaches live play later. HONEST descriptive jutari result " *
                "(bit-exact jutari re-run IS asserted), not an xitari-parity claim.",
            "candidate_cells" => cell_names,
            "candidate_ram_indices" => [c.ram_index for c in r.cands],
            "candidate_concept_family" => families,
            "n_known_game_variables" => length(unique(families)),
            "ranks_swept" => r.ranks,
            "headline_rank" => headline.rank,
            "purity_threshold" => r.purity,
            "sae_config" => Dict{String,Any}("hidden_at_headline" => headline.rank,
                "l1" => r.sae_l1, "epochs" => r.sae_epochs,
                "note" => "SAE hidden size = rank at each sweep point (same #atoms as PCA/NMF)"),
            "nmf_iters" => r.nmf_iters,
            "by_rank" => ranks_block,
            "headline_summary" => Dict{String,Any}(
                "rank" => headline.rank,
                "matched_component_fraction" => Dict{String,Any}(
                    "pca" => _jn(headline.pca_match.matched_component_fraction),
                    "nmf" => _jn(headline.nmf_match.matched_component_fraction),
                    "sae" => _jn(headline.sae_match.matched_component_fraction)),
                "n_causal_atoms" => Dict{String,Any}(
                    "pca" => headline.pca_cu.n_causal_atoms,
                    "nmf" => headline.nmf_cu.n_causal_atoms,
                    "sae" => headline.sae_cu.n_causal_atoms),
                "all_atoms_ablated_break_px" => Dict{String,Any}(
                    "pca" => _jn(headline.pca_cu.all_ablated_break),
                    "nmf" => _jn(headline.nmf_cu.all_ablated_break),
                    "sae" => _jn(headline.sae_cu.all_ablated_break)),
                "recon_rel_err" => Dict{String,Any}(
                    "pca" => _jn(headline.pca.recon_rel_err),
                    "nmf" => _jn(headline.nmf.recon_rel_err),
                    "sae" => _jn(headline.sae.recon_rel_err)),
                "triad" => Dict{String,Any}(
                    "pca" => _triad_record(headline.pca_triad),
                    "nmf" => _triad_record(headline.nmf_triad),
                    "sae" => _triad_record(headline.sae_triad)),
            ),
            "causal_use_note" =>
                "CAUSAL-USE test = ablate a dictionary atom (zero its activation), " *
                "reconstruct the 128-byte state, WRITE it into the real env at frame t, " *
                "RE-RUN the ROM via the §1 oracle, and measure the whole-screen break " *
                "RELATIVE to the reconstruct-WITHOUT-ablation control (isolates the atom, " *
                "not the lossy reconstruction). Asks: does a dictionary atom CAUSALLY " *
                "drive the output? Generalises A7's correlational matched-component " *
                "fraction with causal scoring (present≠used at the feature level).",
            "interpretation" =>
                "Phase-C dictionary learning: PCA (orthogonal/signed), NMF (non-negative " *
                "parts), SAE (sparse/over-complete). matched-component fraction quantifies " *
                "how well each atom aligns to ONE known variable; the causal-use test " *
                "quantifies how many atoms actually MOVE the output. A high matched " *
                "fraction with few causal atoms is the dictionary trap — atoms named after " *
                "variables that the program does not USE through that direction.",
            "scales_to_cluster_via" =>
                "tools/cluster/xai_*.sbatch (E0-3) — full dictionary learning + causal-use " *
                "sweeps over ranks × games; the forward is bit-exact to this HARD path.",
        ),
    )
    open(json_path, "w") do io; JSON.print(io, rec, 2); end

    # NPZ: the headline-rank arrays for downstream figures.
    h = headline
    write_npz(npz_path, Dict(
        "candidate_ram_indices" => Int64[c.ram_index for c in r.cands],
        "headline_rank" => Int64[h.rank],
        "ranks_swept" => Int64.(r.ranks),
        "pca_loadings" => h.pca.loadings,                 # (rank, 128)
        "nmf_loadings" => h.nmf.loadings,
        "sae_loadings" => h.sae.loadings,
        "pca_activations" => h.pca.activations,           # (T, rank)
        "nmf_activations" => h.nmf.activations,
        "sae_activations" => h.sae.activations,
        "pca_causal_effect" => h.pca_cu.effect,           # (rank,)
        "nmf_causal_effect" => h.nmf_cu.effect,
        "sae_causal_effect" => h.sae_cu.effect,
        "pca_heldout_effect" => h.pca_cu.held_out_effect,
        "nmf_heldout_effect" => h.nmf_cu.held_out_effect,
        "sae_heldout_effect" => h.sae_cu.held_out_effect,
        "pca_matched_bool" => Int64[h.pca_match.matched[j] ? 1 : 0 for j in 1:h.pca_match.n_atoms],
        "nmf_matched_bool" => Int64[h.nmf_match.matched[j] ? 1 : 0 for j in 1:h.nmf_match.n_atoms],
        "sae_matched_bool" => Int64[h.sae_match.matched[j] ? 1 : 0 for j in 1:h.sae_match.n_atoms],
        # [matched_frac pca,nmf,sae ; F pca,nmf,sae ; S ... ; M ...]
        "scores" => Float64[
            h.pca_match.matched_component_fraction, h.nmf_match.matched_component_fraction, h.sae_match.matched_component_fraction,
            h.pca_triad.F, h.nmf_triad.F, h.sae_triad.F,
            h.pca_triad.S, h.nmf_triad.S, h.sae_triad.S,
            h.pca_triad.M, h.nmf_triad.M, h.sae_triad.M],
    ))
    return json_path, npz_path
end

function write_summary(results::Vector{GameResult}; out_dir = DEFAULT_OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    path = joinpath(out_dir, "dictionaries_core_summary.json")
    rows = Dict{String,Any}[]
    for r in results
        h = r.by_rank[1]
        push!(rows, Dict{String,Any}(
            "game" => r.game, "state" => "f$(r.target_frame)+$(r.horizon)",
            "in_window" => r.in_window, "headline_rank" => h.rank,
            "n_cands" => length(r.cands),
            "n_known_game_vars" => length(unique([c.family for c in r.cands])),
            "matched_frac_pca" => _jn(h.pca_match.matched_component_fraction),
            "matched_frac_nmf" => _jn(h.nmf_match.matched_component_fraction),
            "matched_frac_sae" => _jn(h.sae_match.matched_component_fraction),
            "n_causal_atoms_pca" => h.pca_cu.n_causal_atoms,
            "n_causal_atoms_nmf" => h.nmf_cu.n_causal_atoms,
            "n_causal_atoms_sae" => h.sae_cu.n_causal_atoms,
            "all_ablated_break_px" => _jn(h.pca_cu.all_ablated_break),
            "pca_FSM" => [_jn(h.pca_triad.F), _jn(h.pca_triad.S), _jn(h.pca_triad.M)],
            "nmf_FSM" => [_jn(h.nmf_triad.F), _jn(h.nmf_triad.S), _jn(h.nmf_triad.M)],
            "sae_FSM" => [_jn(h.sae_triad.F), _jn(h.sae_triad.S), _jn(h.sae_triad.M)],
        ))
    end
    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseC_mechanistic",
        "method" => "nmf_pca_dictionaries(+SAE contrast)",
        "commit" => _git_commit(), "timestamp" => string(round(Int, time())),
        "n_games" => length(results),
        "headline" => "PCA vs NMF vs SAE: matched-component fraction vs known variables " *
            "AND the causal-use test (ablate an atom, re-run via the oracle — does it " *
            "drive the output?). Generalises A7 with causal scoring.",
        "games" => rows,
    )
    open(path, "w") do io; JSON.print(io, rec, 2); end
    return path
end

# ============================================================================
# Sharding helper (SCRUM §7 cluster-shardability).
# ============================================================================
function shard_games(games::Vector{String}, shard::Int, nshards::Int, kind::String)
    (nshards <= 1) && return games
    kind == "game" || error("--shard-kind must be 'game' (got $kind)")
    return [games[i] for i in 1:length(games) if (i - 1) % nshards == shard]
end

# ============================================================================
# CLI
# ============================================================================
function _parse_ranks(s::AbstractString)
    return [parse(Int, strip(x)) for x in split(s, ",") if !isempty(strip(x))]
end

function main(args = ARGS)
    games = copy(CORE_GAMES)
    target_frame = nothing; horizon = nothing; traj_frames = nothing
    ranks = [8, 4, 12]; purity = 0.5
    sae_hidden = nothing; sae_l1 = 0.05; sae_epochs = 3000; nmf_iters = 300; seed = 0
    out_dir = DEFAULT_OUT_DIR; roms_dir = nothing; where = "local"
    shard = 0; nshards = 1; shard_kind = "game"
    selftest_only = false
    i = 1
    while i <= length(args)
        a = args[i]
        if     a == "--games";        games = String.(split(args[i+1], ",")); i += 2
        elseif a == "--game";         games = [args[i+1]]; i += 2
        elseif a == "--shard";        shard = parse(Int, args[i+1]); i += 2
        elseif a == "--nshards";      nshards = parse(Int, args[i+1]); i += 2
        elseif a == "--shard-kind";   shard_kind = args[i+1]; i += 2
        elseif a == "--target-frame"; target_frame = parse(Int, args[i+1]); i += 2
        elseif a == "--horizon";      horizon = parse(Int, args[i+1]); i += 2
        elseif a == "--traj-frames";  traj_frames = parse(Int, args[i+1]); i += 2
        elseif a == "--ranks";        ranks = _parse_ranks(args[i+1]); i += 2
        elseif a == "--rank";         ranks = [parse(Int, args[i+1])]; i += 2
        elseif a == "--sae-hidden";   sae_hidden = parse(Int, args[i+1]); i += 2
        elseif a == "--sae-l1";       sae_l1 = parse(Float64, args[i+1]); i += 2
        elseif a == "--sae-epochs";   sae_epochs = parse(Int, args[i+1]); i += 2
        elseif a == "--nmf-iters";    nmf_iters = parse(Int, args[i+1]); i += 2
        elseif a == "--purity";       purity = parse(Float64, args[i+1]); i += 2
        elseif a == "--seed";         seed = parse(Int, args[i+1]); i += 2
        elseif a == "--out-dir";      out_dir = args[i+1]; i += 2
        elseif a == "--roms-dir";     roms_dir = args[i+1]; i += 2
        elseif a == "--where";        where = args[i+1]; i += 2
        elseif a == "--selftest";     selftest_only = true; i += 1
        else; i += 1
        end
    end
    games = shard_games(games, shard, nshards, shard_kind)
    println("[E5-7] games=$(join(games, ",")) ranks=$(join(ranks, ",")) purity=$purity " *
            "sae_l1=$sae_l1 sae_epochs=$sae_epochs nmf_iters=$nmf_iters seed=$seed " *
            "where=$where shard=$shard/$nshards (jutari/Julia path)")

    if selftest_only
        isempty(games) && error("no games in this shard")
        g = games[1]
        r = compute_game(g; target_frame = target_frame, horizon = horizon,
                         traj_frames = traj_frames, ranks = ranks, purity = purity,
                         sae_hidden = sae_hidden, sae_l1 = sae_l1, sae_epochs = sae_epochs,
                         nmf_iters = nmf_iters, seed = seed, where = where, roms_dir = roms_dir)
        selftest(r)
        println("[E5-7] --selftest: passed on $g, not writing artifact.")
        return 0
    end

    ok = String[]; failed = Tuple{String,String}[]; results = GameResult[]
    for g in games
        try
            r = compute_game(g; target_frame = target_frame, horizon = horizon,
                             traj_frames = traj_frames, ranks = ranks, purity = purity,
                             sae_hidden = sae_hidden, sae_l1 = sae_l1, sae_epochs = sae_epochs,
                             nmf_iters = nmf_iters, seed = seed, where = where, roms_dir = roms_dir)
            selftest(r)
            json_path, npz_path = write_game(r; out_dir = out_dir)
            println("[E5-7] wrote $json_path")
            println("[E5-7] arrays  $npz_path")
            push!(ok, g); push!(results, r)
        catch err
            msg = sprint(showerror, err)
            println("[E5-7] !! game $g FAILED (scoring the rest, not fabricating): " *
                    first(split(msg, '\n')))
            push!(failed, (g, first(split(msg, '\n'))))
        end
    end
    if !isempty(results)
        sp = write_summary(results; out_dir = out_dir)
        println("[E5-7] wrote summary $sp")
    end
    println("[E5-7] ==== summary: $(length(ok))/$(length(games)) games scored ====")
    for g in ok; println("[E5-7]   OK   $g"); end
    for (g, m) in failed; println("[E5-7]   FAIL $g — $m"); end
    return 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    Dictionaries.main()
end
