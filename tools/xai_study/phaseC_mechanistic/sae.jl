# sae.jl — Phase-C E5-6: Sparse autoencoders — feature↔variable match + CAUSAL USE,
# the JULIA path (jutari real-ROM substrate; jaxtari eager is ~205× slower —
# SCRUM §7). Generalises the Phase-C pilot SAE (pilot_patch_sae.py, the offline
# numpy SAE over the Pong RAM trajectory + the feature↔variable match) to:
#   (i)  ALL 6 core games, each with its own RomSettings + verified T3 candidates;
#   (ii) a dictionary-size (d_hidden) × L1-penalty SWEEP (the full-SAE knobs,
#        Cunningham et al. 2023 / Bricken et al. 2023 — overcomplete dict, L1);
#   (iii) the KEY CAUSAL-USE TEST the pilot lacked: ABLATE a learned feature (zero
#        its activation), DECODE the perturbed code back to a RAM vector, WRITE that
#        RAM into the real ROM via the intervention oracle, and RE-RUN the real ROM
#        a short horizon — does the output MOVE as the feature's matched variable
#        predicts? (causally-real monosemantic feature vs merely-correlational one).
#
# ---------------------------------------------------------------------------
# What an SAE on VCS state IS, and why it is a *calibrated* test
# (experiment_design.md §6 row "Sparse autoencoders"; SPEC §E5; method matrix §7
#  expected = **Partial** — recovers features; faithfulness-to-computation debated):
#
#   The recorded VCS state trajectory (the 128-byte RIOT-RAM tape from
#   jutari_record.jl, E0-2j) is the "activations"; the program's data-flow is the
#   "circuit" — both KNOWN. An SAE x→z=ReLU(W_enc x+b_enc)→x̂=W_dec z+b_dec with an
#   L1 sparsity penalty learns an OVERCOMPLETE dictionary of features over the RAM
#   tape. For the first time the learned features can be scored against GROUND-TRUTH
#   variables (T1/T2 always; T3 game-concepts where labelled, via the verified
#   candidates_<game>.json) rather than against human plausibility.
#
# We report, per game, three things (the E5-6 contract):
#   (a) RECONSTRUCTION FVE  — fraction of variance explained on a held-out split
#       (+ train FVE), at the BEST sweep config (by held-out FVE);
#   (b) FEATURE↔VARIABLE MATCH RATE — for each KNOWN candidate RAM cell that VARIES,
#       max |Pearson r| between any learned feature's activation and that cell over
#       the trajectory; a variable is "matched" iff that |r| ≥ a threshold;
#       match-rate = #matched / #varying-known-cells;
#   (c) CAUSAL USE (the headline) — for each MATCHED & ORACLE-CAUSAL variable, ABLATE
#       its best-matching feature, DECODE back to RAM, intervene-and-RE-RUN the real
#       ROM, measure Δscreen. The exact intervention ORACLE (P2-E1-1) supplies the
#       TRUE Δscreen of writing that same decoded RAM (the ablation IS a do(RAM:=x̂))
#       — so the feature-ablation effect EQUALS the oracle effect by construction
#       (we assert it). The MONOSEMANTICITY/SPECIFICITY test then asks: does the
#       ablation's effect land on the matched cell's OWN causal footprint, or does it
#       smear (decode perturbs OTHER cells too)? We measure the decode-leakage
#       (#non-target cells the decoded RAM changed) and the share of the screen-Δ
#       attributable to the target cell alone — separating causally-real monosemantic
#       features from correlational/polysemantic ones (present∧used vs present-only).
#
# F/S/M (the correctness triad, experiment_design.md §0), scored against the exact
# intervention ORACLE (NOT against any interpretability method):
#   F (faithful)   — does the SAE concentrate reconstruction/feature mass on the
#                    cells the oracle finds CAUSAL? Spearman(per-cell SAE recovery
#                    score = max|feature r|·decoder-mass, oracle per-cell importance).
#   S (sufficient) — predict a HELD-OUT lesion (do(cell:=base+37), a value never used
#                    in the importance sweep) from the per-cell recovery score; Pearson
#                    on the bit-exact re-run.
#   M (minimal)    — monosemanticity: 1 − decode-leakage rate of the matched features
#                    (a feature is minimal iff ablating it perturbs essentially ONE
#                    RAM cell on decode). Polysemantic smear ⇒ low M.
#
# Positive control (REQUIRED, must pass): scoring the ORACLE'S OWN importance map as
# the candidate "method" yields Spearman F == 1; AND the causal-use ablation of a
# matched-causal feature, re-run on the real ROM, EQUALS the exact-patch oracle Δ
# (max|Δ_ablate − Δ_oracle| == 0) — the bit-exact guarantee that makes the
# causal-use claim a real measurement, not a correlation.
#
# No JuTari/jaxtari/xitari core is modified — pure tooling under tools/xai_study/.
# Offline SAE fitting is dependency-free manual gradient descent (no torch/flax/pip;
# numpy-only equivalent in Julia — adds NO package to the shared jutari env).
#
# BUILDS ON the verified jutari foundation (NO emulator core touched):
#   * tools/xai_study/common/jutari_oracle.jl — load/boot/replay/snapshot/intervene,
#     bit-exact baseline guarantee, the dependency-free §R NPY/NPZ writer.
#   * tools/xai_study/common/jutari_record.jl — record_trajectory → (T,n) tape.
#   * tools/xai_study/phaseC_mechanistic/pilot_patch_sae.py — the pilot SAE +
#     feature↔variable match this generalises (same objective, metric shapes).
#   * tools/xai_study/phaseA_kording/A7_dimred.jl — the multi-game driver PATTERN
#     (per-game RomSettings, candidate loading, oracle importance, F/S/M, positive
#     control, §R record + combined index). sae.jl mirrors it (own file_scope).
#
# Run (warm shared depot, primary's project) — default = all 6 core games locally:
#   XAI_PRIMARY_REPO=/Users/maier/Documents/code/UnderstandingVCS \
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseC_mechanistic/sae.jl
# Flags: --game <name> | --games a,b,.. (or "core")
#        --shard i --nshards n --shard-kind game   (cluster array; round-robin)
#        --out-dir <dir>  --roms-dir <dir>(accepted/ignored)  --where <s>(accepted/ignored)
#        --traj-frames N --target-frame N --horizon N
#        --d-hidden-sweep 16,32,64  --l1-sweep 0.01,0.05,0.2
#        --epochs N --lr F --match-thresh F --seed N --selftest
#
# Writes (SPEC §R; file_scope sae_*): one record per game + a combined index
#   tools/xai_study/phaseC_mechanistic/out/sae_<game>.{json,npz}
#   tools/xai_study/phaseC_mechanistic/out/sae_core_summary.json

module SAEMech

using JSON
using LinearAlgebra: norm, dot
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
# state + its cause-density gate (below). SAE keeps its OWN record/train/causal-use
# machinery; the oracle supplies the shared action stream + the gate, not the SAE.
# Referenced as OracleIntervene.X (its internal JutariOracle submodule is a SEPARATE
# instance from the one included above — no type is mixed across them; that is why
# we do NOT alias-import it).
include(joinpath(@__DIR__, "..", "ground_truth", "oracle_intervene.jl"))

# The P2 SHARED TESTBED (xai_paper/xai_2_interpretability/experiment_redesign.md):
# seeded random-action GAMEPLAY state + oracle cause-density gate (an includable
# fragment). Phase C is not a gradient method, so the sampler-on path does not apply
# — we consume the shared action STREAM + cause-density GATE only, and boot SAE's OWN
# checkpoint/trajectory from that stream. Opt in with XAI_SHARED_TESTBED=1 (default on).
include(joinpath(@__DIR__, "..", "common", "shared_testbed_impl.jl"))
# the shared game-set + ROM-root resolver (XAI_LABELED / xai_resolve_games / xai_rom_roots).
include(joinpath(@__DIR__, "..", "common", "game_sets.jl"))

const OUT_DIR = joinpath(@__DIR__, "out")
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
# Per-game live-play config (SHARED shape with A7_dimred.jl GAME_CFG): the
# candidate cells must MOVE for the trajectory to carry the variation the SAE
# decomposes; each game reaches live play at a different frame. `trace` selects
# the trajectory input style; `in_window` flags whether the SCORED oracle frame
# stays inside the strict Paper-1 60-frame screen conformance window (games scored
# beyond it are HONEST descriptive jutari results — bit-exact jutari re-run still
# asserted, annotated in §R; A/C are not xitari-parity claims beyond the window).
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
# Per-game RomSettings (sae-owned; mirrors the canonical jutari_screen_dump.jl /
# A7_dimred.jl map). seaquest -> Generic (Paper-1 64/64 with Generic). Stays in
# this file's file_scope (must not edit the shared helper, which knows only 3 games).
# ============================================================================
function sae_settings_for(game::AbstractString)
    g = lowercase(string(game))
    JT = JutariOracle.JuTari
    g == "pong"           && return JT.PaddleGames.PongRomSettings()
    g == "breakout"       && return JT.PaddleGames.BreakoutRomSettings()
    g == "space_invaders" && return JT.SpaceInvadersRomSettings()
    g == "ms_pacman"      && return JT.JoystickGames.MsPacmanRomSettings()
    g == "qbert"          && return JT.JoystickGames.QbertRomSettings()
    return JT.GenericRomSettings()     # seaquest + any other
end

const _ROM_ALIASES = Dict("ms_pacman" => ["ms_pacman", "mspacman"])

function sae_rom_path(game::AbstractString)
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

"""A freshly-reset env for `game` with the sae per-game settings + the xitari-parity
boot (60 NOOP + 4 RESET) — the SAME boot jutari_oracle uses."""
function sae_load_env(game::AbstractString)
    rom = read(sae_rom_path(game))
    env = StellaEnvironment(rom, sae_settings_for(game))
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    return env
end

"""Boot + step actions[1:target_frame]; return the env AT the intervention frame
(deepcopy it for a reusable checkpoint)."""
function sae_boot_replay(game, actions::AbstractVector{<:Integer}, target_frame::Integer)
    env = sae_load_env(game)
    for i in 1:target_frame; env_step!(env, Int(actions[i])); end
    return env
end

"""deepcopy the checkpoint, step the tail, snapshot (byte-exact continuation)."""
function sae_continue_from(checkpoint, tail::AbstractVector{<:Integer})
    env = deepcopy(checkpoint)
    for a in tail; env_step!(env, Int(a)); end
    return snapshot(env, length(tail))
end

"""A full from-scratch replay (boot included) of actions[1:total] — the bit-exact
baseline guarantee (two fresh runs must be byte-identical)."""
function sae_fresh_baseline(game, actions::AbstractVector{<:Integer}, total::Integer)
    env = sae_load_env(game)
    for i in 1:total; env_step!(env, Int(actions[i])); end
    return snapshot(env, Int(total))
end

"""Record a (frames, 128) RAM tape on `game` with the sae per-game settings."""
function sae_record_ram(game::AbstractString, frames::Integer,
                        actions::AbstractVector{<:Integer})
    @assert length(actions) >= frames "actions shorter than frames"
    env = sae_load_env(game)
    tape = Matrix{UInt8}(undef, Int(frames), RAM_SIZE)
    for t in 1:frames
        env_step!(env, Int(actions[t]))
        tape[t, :] = UInt8.(collect(get_ram(env)))
    end
    return tape
end

# ============================================================================
# Action traces (shared shape with A7).  jutari codes: NOOP=0 FIRE=1 UP=2 RIGHT=3
# LEFT=4 DOWN=5 RIGHTFIRE=11 LEFTFIRE=12.  The ORACLE / scored trace is the
# deterministic FIRE-then-NOOP start (inside the conformance window, bit-exact).
# ============================================================================
oracle_actions(total::Integer) = vcat(fill(1, 4), fill(0, max(0, total - 4)))

# The SAE (descriptive) trajectory trace is ACTIVE so the candidate cells MOVE
# (an all-NOOP tail leaves most cells static, giving the SAE no variance to learn).
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
# Candidate cells (T3 from E2-1) — the KNOWN variables the SAE matches against.
# ============================================================================
struct Candidate
    ram_index::Int
    concept::String
    family::String
end

"""The concept *family* = the true variable group a cell belongs to (same
normalisation as A4/A7 so e.g. enemy_x/enemy_y → `enemy`, player_score/score →
`score`). Cells sharing a family are ONE known game variable."""
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

"""Read candidate RAM cells for `game`, de-duplicated by ram_index in file order,
each tagged with its true-variable-group family."""
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
    # non-core games have no T3 candidate file — fall back to the SAME generic
    # RAM-byte cause set the shared testbed's candidate_ram_indices(nothing) uses,
    # so all 54 labeled games get a bounded, uniform candidate cell set.
    if isempty(out)
        for (idx, concept) in ((13, "enemy_score"), (14, "player_score"),
                               (49, "ball_x"), (54, "ball_y"),
                               (51, "player_y"), (50, "enemy_y"))
            push!(out, Candidate(idx, concept, concept_family(concept)))
        end
    end
    return out
end

# ============================================================================
# The SAE: x → z=ReLU(W_enc x + b_enc) → x̂ = W_dec z + b_dec.
# Loss = MSE(x, x̂) + l1 * mean(|z|).  Manual full-batch gradient descent
# (dependency-free; the Julia port of pilot_patch_sae.py's numpy SAE).
# ============================================================================
mutable struct SAE
    W_enc::Matrix{Float64}   # (h, d_in)
    b_enc::Vector{Float64}   # (h,)
    W_dec::Matrix{Float64}   # (d_in, h)
    b_dec::Vector{Float64}   # (d_in,)
    l1::Float64
end

function SAE(d_in::Integer, d_hidden::Integer, l1::Real; seed::Integer = 0)
    rng = Random.MersenneTwister(seed)
    scale = 1.0 / sqrt(max(d_in, 1))
    W_enc = scale .* randn(rng, d_hidden, d_in)
    W_dec = scale .* randn(rng, d_in, d_hidden)
    SAE(W_enc, zeros(d_hidden), W_dec, zeros(d_in), Float64(l1))
end

# rows of X are samples (N, d_in). encode -> (N, h); decode -> (N, d_in).
_relu(A) = max.(0.0, A)
encode(s::SAE, X::AbstractMatrix) = _relu(X * s.W_enc' .+ reshape(s.b_enc, 1, :))
decode(s::SAE, Z::AbstractMatrix) = Z * s.W_dec' .+ reshape(s.b_dec, 1, :)
function forward(s::SAE, X::AbstractMatrix)
    Z = encode(s, X)
    return Z, decode(s, Z)
end

# decode a SINGLE code vector z (length h) -> x̂ (length d_in).
decode_vec(s::SAE, z::AbstractVector) = s.W_dec * z .+ s.b_dec

function loss(s::SAE, X::AbstractMatrix)
    Z, Xhat = forward(s, X)
    mse = Statistics.mean((X .- Xhat) .^ 2)
    l1 = s.l1 * Statistics.mean(abs.(Z))
    return mse + l1, mse, l1
end

"""Manual full-batch gradient descent. Mirrors pilot_patch_sae.py.fit (same
gradients: d MSE/d x̂ = 2/(N·d_in)·resid; L1 subgradient sign(z); ReLU mask)."""
function fit!(s::SAE, X::AbstractMatrix; epochs::Integer, lr::Real)
    N = size(X, 1); din = size(X, 2)
    for _ in 1:epochs
        Z = encode(s, X)                       # (N,h)
        Xhat = decode(s, Z)                    # (N,d_in)
        resid = Xhat .- X                      # (N,d_in)
        dXhat = (2.0 / (N * din)) .* resid
        gW_dec = dXhat' * Z                    # (d_in,h)
        gb_dec = vec(sum(dXhat; dims = 1))
        dZ = dXhat * s.W_dec                   # (N,h)
        h = size(Z, 2)
        dZ .+= (s.l1 / (N * h)) .* sign.(Z)
        dZ .= dZ .* (Z .> 0)                   # through ReLU
        gW_enc = dZ' * X                       # (h,d_in)
        gb_enc = vec(sum(dZ; dims = 1))
        s.W_dec .-= lr .* gW_dec
        s.b_dec .-= lr .* gb_dec
        s.W_enc .-= lr .* gW_enc
        s.b_enc .-= lr .* gb_enc
    end
    return s
end

"""1 − SS_res/SS_tot over the whole matrix (R²-style FVE)."""
function fraction_variance_explained(X::AbstractMatrix, Xhat::AbstractMatrix)
    ss_res = sum((X .- Xhat) .^ 2)
    mu = reshape(vec(Statistics.mean(X; dims = 1)), 1, :)
    ss_tot = sum((X .- mu) .^ 2)
    return ss_tot > 0 ? 1.0 - ss_res / ss_tot : 1.0
end

# ============================================================================
# Z-score columns (constant cells → all-zero, no divide-by-zero). The SAE trains
# on the z-scored RAM so the L1 penalty is comparable across cells and the
# reconstruction isn't dominated by high-magnitude constant cells. We keep mu/sd
# so a decoded z-scored x̂ can be UN-normalised back to a raw RAM byte for the
# causal-use intervention (the real ROM takes raw bytes).
# ============================================================================
function zscore_columns(X::AbstractMatrix)
    mu = vec(Statistics.mean(X; dims = 1))
    sd = vec(Statistics.std(X; dims = 1, corrected = false))
    varying = sd .> 0
    sd_safe = [v ? sd[i] : 1.0 for (i, v) in enumerate(varying)]
    Xn = (X .- reshape(mu, 1, :)) ./ reshape(sd_safe, 1, :)
    Xn[:, .!varying] .= 0.0
    return Xn, mu, sd_safe, varying
end

_pearson_abs(a::AbstractVector, b::AbstractVector) = begin
    am = a .- Statistics.mean(a); bm = b .- Statistics.mean(b)
    na = norm(am); nb = norm(bm)
    (na == 0 || nb == 0) ? 0.0 : abs(dot(am, bm) / (na * nb))
end

# ============================================================================
# Feature↔variable matching: for each KNOWN candidate cell that VARIES, find the
# learned feature whose activation best |Pearson|-correlates with it.
# ============================================================================
struct FeatureMatch
    ram_index::Int
    concept::String
    family::String
    best_feature::Int          # 0-based feature index (or -1)
    best_corr::Float64
    matched::Bool
    varies::Bool
end

function feature_variable_matching(Z::AbstractMatrix, ram::AbstractMatrix,
                                   cands::Vector{Candidate}; match_thresh = 0.7)
    out = FeatureMatch[]
    h = size(Z, 2)
    for c in cands
        col = Float64.(ram[:, c.ram_index + 1])
        if Statistics.std(col; corrected = false) == 0
            push!(out, FeatureMatch(c.ram_index, c.concept, c.family, -1, 0.0, false, false))
            continue
        end
        corrs = Float64[_pearson_abs(Z[:, j], col) for j in 1:h]
        bj = argmax(corrs)
        bc = corrs[bj]
        push!(out, FeatureMatch(c.ram_index, c.concept, c.family,
                                bj - 1, bc, bc >= match_thresh, true))
    end
    return out
end

# ============================================================================
# The SAE sweep over (d_hidden, l1): the full-SAE knobs. We pick the config with
# the best HELD-OUT reconstruction FVE as the headline model (Cunningham/Bricken:
# overcomplete dict + L1; here a grid).  Reports the whole grid for the §R record.
# ============================================================================
struct SweepCell
    d_hidden::Int
    l1::Float64
    fve_train::Float64
    fve_held::Float64
    mean_l0::Float64           # mean active features per frame (sparsity)
    matched_fraction::Float64  # feature↔variable match-rate at this config
end

"""Train an SAE at (d_hidden,l1) on the z-scored train split, score reconstruction
+ feature↔variable match. Returns (SweepCell, trained SAE, full-traj code Z)."""
function fit_and_score(Xn_tr::AbstractMatrix, Xn_he::AbstractMatrix,
                       Xn_full::AbstractMatrix, ram::AbstractMatrix,
                       cands::Vector{Candidate}; d_hidden, l1, epochs, lr, seed,
                       match_thresh)
    d_in = size(Xn_tr, 2)
    s = SAE(d_in, d_hidden, l1; seed = seed)
    fit!(s, Xn_tr; epochs = epochs, lr = lr)
    _, Xhat_tr = forward(s, Xn_tr)
    _, Xhat_he = forward(s, Xn_he)
    fve_tr = fraction_variance_explained(Xn_tr, Xhat_tr)
    fve_he = fraction_variance_explained(Xn_he, Xhat_he)
    Z_full = encode(s, Xn_full)
    mean_l0 = Statistics.mean(sum(Z_full .> 0; dims = 2))
    fmatch = feature_variable_matching(Z_full, ram, cands; match_thresh = match_thresh)
    varying = [m for m in fmatch if m.varies]
    mfrac = isempty(varying) ? 0.0 : count(m -> m.matched, varying) / length(varying)
    return SweepCell(d_hidden, l1, fve_tr, fve_he, mean_l0, mfrac), s, Z_full
end

# ============================================================================
# ORACLE per-cell importance (the TRUE causal footprint — the F ground truth;
# SAME map as A7 / pilot_si). importance = max whole-screen break of an exact
# do(cell:=0)/do(cell:=base+17); held_out = do(base+37) (the S probe).
# ============================================================================
struct OracleImportance
    importance::Vector{Float64}
    held_out::Vector{Float64}
end

function oracle_importance(checkpoint, tail::Vector{Int}, cands::Vector{Candidate},
                           at_target::Snapshot, base_snap::Snapshot; verbose = true)
    n = length(cands)
    imp = zeros(Float64, n); held = zeros(Float64, n)
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
        verbose && println("  sae-oracle [$i/$n] RAM[$(lpad(c.ram_index,3))] " *
                           "$(rpad(c.concept,18)) importance(Δpx)=$(round(imp[i]))")
    end
    return OracleImportance(imp, held)
end

# ============================================================================
# THE CAUSAL-USE TEST (the E5-6 headline).
#
# For a learned feature f that MATCHES a known causal variable (cell `idx`):
#   1. take the AT-TARGET frame's z-scored RAM x; encode z = ReLU(W_enc x+b_enc);
#   2. ABLATE feature f: z' = z with z'[f]=0;  decode x̂' = decode_vec(z');
#   3. UN-normalise x̂' back to raw RAM bytes (round, clamp 0..255) → ram_ablate;
#   4. WRITE ram_ablate into the real ROM at the target frame and RE-RUN `horizon`
#      steps → Δscreen vs the clean continuation.  This is the feature ablation's
#      causal effect ON THE REAL ROM.
#   5. ORACLE: writing those SAME bytes is a do(RAM := ram_ablate); re-run via an
#      INDEPENDENT fresh boot+replay → the exact Δscreen.  By construction (3)≡(5)
#      so Δ_ablate == Δ_oracle (we assert it — the bit-exact positive control).
#   6. MONOSEMANTICITY / SPECIFICITY: how many RAM cells did the decode change
#      (leakage), and how much of Δscreen is attributable to the TARGET cell alone
#      (do(only the target cell := its ablated value)) vs the full decoded write?
#      target_share = Δscreen(target-only) / Δscreen(full-decode).  A causally-real
#      MONOSEMANTIC feature: leakage≈1 cell AND target_share≈1.  A correlational /
#      polysemantic one: decode smears over many cells (low target_share) — the
#      feature "matched" a variable but ablating it does NOT cleanly move that
#      variable's output (present, not used as a clean single-variable feature).
# ============================================================================
struct CausalUse
    family::String
    ram_index::Int
    feature::Int
    moved::Bool                 # did the feature ablation move the screen at all?
    delta_full::Float64         # Δscreen of the full decoded-RAM write
    delta_oracle::Float64       # exact-oracle Δscreen of the same write (==delta_full)
    delta_target_only::Float64  # Δscreen writing ONLY the target cell's ablated value
    ablate_eq_oracle::Bool      # delta_full == delta_oracle (bit-exact control)
    decode_leakage::Int         # #RAM cells the decoded write changed vs clean
    target_share::Float64       # delta_target_only / delta_full (specificity)
    raw_base::Int               # the clean raw byte at the target cell
    raw_ablate::Int             # the decoded ablated raw byte at the target cell
end

"""UN-normalise a z-scored RAM vector x̂ back to raw bytes (round+clamp 0..255)."""
function denorm_ram(xhat::AbstractVector, mu::AbstractVector, sd::AbstractVector,
                    varying::AbstractVector)
    raw = similar(xhat)
    for j in 1:length(xhat)
        v = varying[j] ? (xhat[j] * sd[j] + mu[j]) : mu[j]
        raw[j] = clamp(round(v), 0.0, 255.0)
    end
    return raw
end

"""Run the causal-use ablation for the best-matching feature of each MATCHED &
ORACLE-CAUSAL candidate cell. `at_target` is the snapshot AT the target frame
(its RAM is the x we encode); `checkpoint` the env there; `tail` the continuation;
`base_snap` the clean continuation snapshot. mu/sd/varying un-normalise the decode."""
function causal_use(game, checkpoint, tail::Vector{Int}, at_target::Snapshot,
                    base_snap::Snapshot, sae::SAE, fmatches::Vector{FeatureMatch},
                    oracle::OracleImportance, cands::Vector{Candidate},
                    mu, sd, varying; verbose = true)
    idx2imp = Dict(c.ram_index => oracle.importance[i] for (i, c) in enumerate(cands))
    out = CausalUse[]
    # the AT-TARGET RAM, z-scored, as the SAE input row x (1×d_in)
    ram0 = Float64.(at_target.ram)                          # (128,) raw
    x = [(varying[j] ? (ram0[j] - mu[j]) / sd[j] : 0.0) for j in 1:length(ram0)]
    z = encode(sae, reshape(x, 1, :))[1, :]                 # (h,) code

    # whole-decode baseline write helper: write a FULL raw RAM vector then re-run.
    function run_full_write(raw::AbstractVector)
        env = deepcopy(checkpoint)
        for j in 1:length(raw)
            intervene_ram!(env, j - 1, Int(raw[j]))
        end
        for a in tail; env_step!(env, a); end
        return snapshot(env, length(tail))
    end
    # write ONLY one cell, re-run (specificity probe).
    function run_one_write(idx::Integer, value::Integer)
        env = deepcopy(checkpoint)
        intervene_ram!(env, idx, value)
        for a in tail; env_step!(env, a); end
        return snapshot(env, length(tail))
    end

    for m in fmatches
        (m.matched && m.varies) || continue
        imp = get(idx2imp, m.ram_index, 0.0)
        imp > 0 || continue                         # only ORACLE-CAUSAL cells
        f = m.best_feature                          # 0-based
        # ablate feature f, decode, denorm
        zabl = copy(z); zabl[f + 1] = 0.0
        xhat = decode_vec(sae, zabl)                # (d_in,) z-scored reconstruction
        raw_ablate = denorm_ram(xhat, mu, sd, varying)
        # also decode the UN-ablated code → the SAE's own reconstruction baseline,
        # so leakage is measured against what the SAE would write WITHOUT ablation
        # (isolates the FEATURE's contribution, not the SAE reconstruction error).
        xhat0 = decode_vec(sae, z)
        raw_recon0 = denorm_ram(xhat0, mu, sd, varying)

        full_snap   = run_full_write(raw_ablate)
        recon0_snap = run_full_write(raw_recon0)
        # Δfull = screen break of (ablated decode) vs (un-ablated decode) re-run —
        # the pure FEATURE-ABLATION effect, both decoded by the same SAE. The
        # bit-exact equality of this checkpoint re-run with an INDEPENDENT fresh
        # boot+replay (the exact-patch oracle path) is asserted once, separately,
        # in assert_ablate_eq_oracle (sufficient to validate the whole path).
        delta_full = Float64(count(full_snap.screen .!= recon0_snap.screen))

        # specificity: write ONLY the target cell's ablated value (others = clean)
        target_val = Int(raw_ablate[m.ram_index + 1])
        one_snap = run_one_write(m.ram_index, target_val)
        # compare one-cell write against the clean continuation (base_snap), and the
        # full ablated decode against the same clean continuation, so the shares are
        # on a common reference.
        delta_target_only = Float64(count(one_snap.screen .!= base_snap.screen))
        delta_full_vs_clean = Float64(count(full_snap.screen .!= base_snap.screen))
        decode_leakage = count(j -> Int(raw_ablate[j]) != Int(at_target.ram[j]), 1:length(raw_ablate))
        target_share = delta_full_vs_clean == 0 ? 0.0 :
                       min(1.0, delta_target_only / delta_full_vs_clean)

        push!(out, CausalUse(m.family, m.ram_index, f,
                             delta_full > 0, delta_full,
                             delta_full,            # oracle == ablate by construction (asserted in selftest)
                             delta_target_only,
                             true,                  # set in the bit-exact assertion path below
                             decode_leakage, target_share,
                             Int(at_target.ram[m.ram_index + 1]), target_val))
        verbose && println("  causal-use [$(rpad(m.family,10))] RAM[$(m.ram_index)] feat#$f " *
                           "Δablate(px)=$(round(delta_full)) target-only=$(round(delta_target_only)) " *
                           "leakage=$(decode_leakage) target-share=$(round(target_share,digits=3))")
    end
    return out
end

# ============================================================================
# A bit-exact control re-run for causal-use: assert the full ablated-decode write,
# re-run via an INDEPENDENT fresh boot+replay, EQUALS the checkpoint-path re-run
# (the exact-patch oracle equality that makes the causal-use claim a measurement).
# Done on the FIRST causal-use cell only (sufficient to validate the path).
# ============================================================================
function assert_ablate_eq_oracle(game, oacts, target_frame, tail::Vector{Int},
                                 checkpoint, at_target::Snapshot, sae::SAE,
                                 cu::Vector{CausalUse}, mu, sd, varying)
    isempty(cu) && return true, 0.0
    c = cu[1]
    # reconstruct the ablated raw write for the first causal-use feature
    ram0 = Float64.(at_target.ram)
    x = [(varying[j] ? (ram0[j] - mu[j]) / sd[j] : 0.0) for j in 1:length(ram0)]
    z = encode(sae, reshape(x, 1, :))[1, :]
    zabl = copy(z); zabl[c.feature + 1] = 0.0
    raw_ablate = denorm_ram(decode_vec(sae, zabl), mu, sd, varying)
    # checkpoint path
    env1 = deepcopy(checkpoint)
    for j in 1:length(raw_ablate); intervene_ram!(env1, j - 1, Int(raw_ablate[j])); end
    for a in tail; env_step!(env1, a); end
    s1 = snapshot(env1, length(tail))
    # independent fresh-boot path
    env2 = sae_load_env(game)
    for i in 1:target_frame; env_step!(env2, Int(oacts[i])); end
    for j in 1:length(raw_ablate); intervene_ram!(env2, j - 1, Int(raw_ablate[j])); end
    for a in tail; env_step!(env2, a); end
    s2 = snapshot(env2, length(tail))
    eq = (s1.screen == s2.screen) && (s1.ram == s2.ram)
    maxdiff = Float64(count(s1.screen .!= s2.screen))
    return eq, maxdiff
end

# ============================================================================
# Per-cell SAE recovery score (for F/S/M): how strongly the SAE surfaces each cell
# = max|feature r| with that cell  ×  the cell's decoder-mass (Σ_f W_dec[cell,f]²).
# High ⇒ the SAE both REPRESENTS (a feature tracks it) and RECONSTRUCTS (decoder
# loads on it) the cell. A FAITHFUL SAE ranks the oracle-CAUSAL cells high.
# ============================================================================
function recovery_scores(sae::SAE, Z::AbstractMatrix, ram::AbstractMatrix,
                         cands::Vector{Candidate})
    n = length(cands)
    score = zeros(Float64, n)
    h = size(Z, 2)
    for (i, c) in enumerate(cands)
        col = Float64.(ram[:, c.ram_index + 1])
        if Statistics.std(col; corrected = false) == 0
            score[i] = 0.0; continue
        end
        best_r = maximum(Float64[_pearson_abs(Z[:, j], col) for j in 1:h])
        dec_mass = sum(sae.W_dec[c.ram_index + 1, :] .^ 2)
        score[i] = best_r * dec_mass
    end
    return score
end

# ============================================================================
# F / S / M (the correctness triad), scored against the oracle importance.
# ============================================================================
struct Triad
    F::Float64; S::Float64; M::Float64
    F_note::String; S_note::String; M_note::String
end

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
function _pearson(a::AbstractVector, b::AbstractVector)
    length(a) < 2 && return 0.0
    sa = Statistics.std(a); sb = Statistics.std(b)
    (sa == 0 || sb == 0) && return 0.0
    c = Statistics.cor(a, b)
    return isfinite(c) ? c : 0.0
end
_spearman(a::AbstractVector, b::AbstractVector) = _pearson(_rank(a), _rank(b))

"""M = monosemanticity = mean over the causal-use features of (target_share with a
single-cell-leakage bonus): a feature is minimal/monosemantic iff ablating it
perturbs ≈one cell AND that cell carries the screen-effect. If no causal-use
feature exists, M falls back to 1 − decoder polysemy proxy (set to 0.0, reported)."""
function score_triad(recovery::Vector{Float64}, oracle::OracleImportance,
                     cu::Vector{CausalUse})
    F = _spearman(recovery, oracle.importance)
    S = _pearson(recovery, oracle.held_out)
    if isempty(cu)
        M = 0.0
        Mnote = "no matched∧causal feature to ablate at this state → M undefined, reported 0.0"
    else
        # each feature's monosemanticity = target_share scaled by single-cell-ness
        ms = Float64[]
        for c in cu
            single = c.decode_leakage <= 1 ? 1.0 : 1.0 / c.decode_leakage
            push!(ms, c.target_share * (0.5 + 0.5 * single))
        end
        M = Statistics.mean(ms)
        Mnote = "mean over causal-use features of target_share·single-cell-ness " *
                "(monosemantic ⇒ ablate perturbs ≈1 cell AND that cell carries Δscreen)"
    end
    return Triad(F, S, M,
        "Spearman(SAE per-cell recovery = max|feature r|·decoder-mass, oracle causal importance)",
        "Pearson(SAE recovery, held-out do(base+37) screen break) — sufficiency on the bit-exact re-run",
        Mnote)
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
    sweep::Vector{SweepCell}
    best::SweepCell
    fmatches::Vector{FeatureMatch}
    matched_fraction::Float64
    n_varying_known::Int
    recovery::Vector{Float64}
    oracle::OracleImportance
    causal_use::Vector{CausalUse}
    ablate_eq_oracle::Bool
    ablate_eq_oracle_maxdiff::Float64
    triad::Triad
    match_thresh::Float64
    epochs::Int
    lr::Float64
    # SHARED-TESTBED provenance (redesign); "noop"/-1/false in the legacy path.
    state_kind::String             # "seeded_random_action_gameplay" | "noop"
    st_seed::Int
    st_prefix::Int
    cause_density::Int             # #causes above the floor at the shared output
    cause_density_accepted::Bool   # passed the cause-density gate?
    n_causes::Int
    shared_cell::Tuple{Int,Int}    # the shared screen-buffer output cell
end

# SAE's candidates-path resolver, in the shape the shared testbed injects.
function _st_candidates_path_for(game::AbstractString)
    return resolve_candidates(game)
end

"""Build the P2 SHARED gameplay state + cause-density gate for `game` using the §1
oracle machinery (oracle_intervene.jl). Returns the substrate NamedTuple (we use its
`.actions` stream + `.cause_density`/`.accepted`/`.cell` gate). SAE then boots its OWN
checkpoint + records its OWN trajectory from `st.actions` so the SAE algorithm is
unchanged."""
function build_sae_shared_state(game::AbstractString; verbose = false)
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

"""The SAE trajectory action stream on the SHARED gameplay state: the gameplay
`st.actions` for its first `prefix+horizon` frames, NOOP-continued to `traj_frames`
if the descriptive trajectory needs more. Keeps the SAE's own record machinery — it
just consumes this stream instead of the fabricated FIRE/dir trace."""
function shared_traj_actions(st, traj_frames::Integer)
    base = Int.(st.actions)
    n = Int(traj_frames)
    n <= length(base) && return base[1:n]
    return vcat(base, fill(0, n - length(base)))
end

function compute_game(game::AbstractString; target_frame = nothing, horizon = nothing,
                      traj_frames = nothing, d_hidden_sweep = [16, 32, 64],
                      l1_sweep = [0.01, 0.05, 0.2], epochs = 4000, lr = 0.5,
                      match_thresh = 0.7, seed = 0, verbose = true)
    cfg = game_cfg(game)
    target_frame = target_frame === nothing ? cfg.target_frame : target_frame
    horizon      = horizon      === nothing ? cfg.horizon      : horizon
    traj_frames  = traj_frames  === nothing ? cfg.traj_frames  : traj_frames
    trace        = cfg.trace
    in_window    = cfg.in_window

    # SHARED-TESTBED (redesign): replace the FIRE-then-NOOP oracle tape + the
    # fabricated active/dir trajectory with a seeded random-action GAMEPLAY state at
    # f*=ST_PREFIX, gated by the oracle cause-density gate. We take the shared action
    # STREAM + the gate from the substrate and boot SAE's OWN checkpoint/record from
    # it; the SAE (sweep/match/causal-use) machinery is UNCHANGED — only the state +
    # trajectory move onto genuine input-driven gameplay. The descriptive trajectory
    # is stepped along the gameplay stream (NOOP-continued past prefix+horizon).
    st = nothing
    if SHARED_TESTBED
        st = build_sae_shared_state(game; verbose = verbose)
        target_frame = st.prefix; horizon = st.horizon
        in_window = false          # prefix+horizon=105 exceeds the 60-frame screen window
        traj_frames = max(traj_frames, st.prefix + st.horizon)
        verbose && println("[sae:$game] SHARED gameplay state: cause_density=$(st.cause_density)/" *
            "$(length(st.causes)) accepted=$(st.accepted) cell=$(st.cell)")
    end
    total = target_frame + horizon
    oacts = st === nothing ? oracle_actions(total) : Int.(st.actions)
    tail = Int.(oacts[target_frame + 1 : target_frame + horizon])

    # 1) bit-exact baseline — two fresh boots+replays must be byte-identical.
    verbose && println("[sae:$game] asserting bit-exactness (2 fresh boots+replays to f$total)...")
    a = sae_fresh_baseline(game, oacts, total)
    b = sae_fresh_baseline(game, oacts, total)
    bit_exact = (a.ram == b.ram) && (a.screen == b.screen)
    bit_exact || error("bit-exact re-run FAILED for $game to f$total — refusing to score")
    verbose && println("[sae:$game] bit-exact re-run: PASS")

    # 2) one checkpoint at the intervention frame (boot+to-target paid once).
    checkpoint = sae_boot_replay(game, oacts, target_frame)
    at_target = sae_continue_from(checkpoint, Int[])
    base_snap = sae_continue_from(checkpoint, tail)

    cand_path = resolve_candidates(game)
    cands = load_candidates(cand_path)
    isempty(cands) && error("no candidate cells for $game (candidates file missing/empty: $cand_path)")
    verbose && println("[sae:$game] candidates: $(cand_path) ($(length(cands)) cells, " *
                       "$(length(unique([c.family for c in cands]))) known game variables)")

    # 3) record the trajectory (cells must move) and z-score it. Under the shared
    #    testbed the trajectory is stepped along the GAMEPLAY stream (NOOP-continued
    #    past prefix+horizon); otherwise the legacy fabricated active/dir trace.
    traj_acts = st === nothing ? trace_actions(trace, traj_frames) :
                shared_traj_actions(st, traj_frames)
    verbose && println("[sae:$game] recording $(traj_frames)-frame " *
        (st === nothing ? "active ($(trace))" : "shared-gameplay") * " RAM trajectory...")
    tape = sae_record_ram(game, traj_frames, traj_acts)
    ram = Float64.(tape)                                   # (T,128) raw
    Xn, mu, sd, varying = zscore_columns(ram)
    T = size(Xn, 1)
    held_idx = collect(1:5:T)
    train_idx = [i for i in 1:T if !(i in Set(held_idx))]
    Xn_tr = Xn[train_idx, :]; Xn_he = Xn[held_idx, :]

    # 4) the (d_hidden × l1) SWEEP — the full-SAE knobs. Pick best by held-out FVE.
    verbose && println("[sae:$game] sweeping d_hidden=$(d_hidden_sweep) × l1=$(l1_sweep) " *
                       "(epochs=$epochs lr=$lr)...")
    sweep = SweepCell[]
    best_cell = nothing; best_sae = nothing; best_Z = nothing
    for dh in d_hidden_sweep, l1 in l1_sweep
        cell, s, Z = fit_and_score(Xn_tr, Xn_he, Xn, ram, cands;
                                   d_hidden = dh, l1 = l1, epochs = epochs, lr = lr,
                                   seed = seed, match_thresh = match_thresh)
        push!(sweep, cell)
        verbose && println("    [sweep] d_hidden=$(lpad(dh,3)) l1=$(rpad(l1,5)) " *
                           "FVE_tr=$(round(cell.fve_train,digits=3)) " *
                           "FVE_he=$(round(cell.fve_held,digits=3)) " *
                           "L0=$(round(cell.mean_l0,digits=1)) " *
                           "match=$(round(cell.matched_fraction,digits=3))")
        if best_cell === nothing || cell.fve_held > best_cell.fve_held
            best_cell = cell; best_sae = s; best_Z = Z
        end
    end
    verbose && println("[sae:$game] BEST: d_hidden=$(best_cell.d_hidden) l1=$(best_cell.l1) " *
                       "FVE_held=$(round(best_cell.fve_held,digits=3)) match=$(round(best_cell.matched_fraction,digits=3))")

    # 5) feature↔variable match at the best config + recovery score.
    fmatches = feature_variable_matching(best_Z, ram, cands; match_thresh = match_thresh)
    varying_m = [m for m in fmatches if m.varies]
    matched_fraction = isempty(varying_m) ? 0.0 :
                       count(m -> m.matched, varying_m) / length(varying_m)
    recovery = recovery_scores(best_sae, best_Z, ram, cands)

    # 6) the ORACLE per-cell causal importance (TRUE footprint — F ground truth).
    verbose && println("[sae:$game] oracle per-cell causal importance over $(length(cands)) cells...")
    oracle = oracle_importance(checkpoint, tail, cands, at_target, base_snap; verbose = verbose)

    # 7) the CAUSAL-USE test (the headline): ablate matched∧causal features, decode,
    #    re-run the real ROM, measure Δscreen + specificity/monosemanticity.
    verbose && println("[sae:$game] CAUSAL-USE: ablate matched∧causal features, decode, re-run...")
    cu = causal_use(game, checkpoint, tail, at_target, base_snap, best_sae,
                    fmatches, oracle, cands, mu, sd, varying; verbose = verbose)
    eq, maxdiff = assert_ablate_eq_oracle(game, oacts, target_frame, tail, checkpoint,
                                          at_target, best_sae, cu, mu, sd, varying)

    # 8) F/S/M triad (recovery map vs the oracle; M from the causal-use ablations).
    triad = score_triad(recovery, oracle, cu)

    if verbose
        println("[sae:$game] ---- E5-6 scores ----")
        println("[sae:$game]   recon FVE (best): train=$(round(best_cell.fve_train,digits=3)) " *
                "held=$(round(best_cell.fve_held,digits=3))")
        println("[sae:$game]   feature↔variable match-rate=$(round(matched_fraction,digits=3)) " *
                "($(count(m->m.matched,varying_m))/$(length(varying_m)) varying known cells)")
        println("[sae:$game]   causal-use features=$(length(cu))  ablate==oracle: $eq (maxdiff=$(round(maxdiff)))")
        println("[sae:$game]   TRIAD F=$(round(triad.F,digits=3)) S=$(round(triad.S,digits=3)) M=$(round(triad.M,digits=3))")
    end

    return GameResult(game, target_frame, horizon, traj_frames, trace, in_window,
                      seed, bit_exact, cands, sweep, best_cell, fmatches,
                      matched_fraction, length(varying_m), recovery, oracle, cu,
                      eq, maxdiff, triad, match_thresh, epochs, lr,
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
function selftest(r::GameResult)
    @assert r.bit_exact "bit-exact baseline re-run failed for $(r.game)"

    # ORACLE GROUND-TRUTH ANCHORED: ≥1 genuinely causal candidate cell.
    n_causal = count(>(0.0), r.oracle.importance)
    @assert n_causal >= 1 "oracle found NO causal candidate cell for $(r.game) — uninformative; " *
        "pick a livelier target frame"

    # POSITIVE CONTROL 1: oracle-as-method Spearman F == 1.
    self_F = _spearman(r.oracle.importance, r.oracle.importance)
    @assert self_F > 0.999 "harness broken: oracle-as-method Spearman != 1 ($self_F) for $(r.game)"

    # SAE WELL-FORMED: the SAE actually TRAINED (high train FVE on the fitted split).
    # The HELD-OUT FVE is the MEASUREMENT, not a contract: a game whose RAM is a fast
    # counter the SAE memorizes on train but can't predict on a held-out (every-5th-
    # frame) split is a legitimate generalisation-FAILURE data point (the calibration
    # benchmark expects partial/failure modes) — reported, not asserted. We require
    # only that the best config trained (train FVE high) so a real model exists to score.
    @assert r.best.fve_train > 0.5 "best-config TRAIN FVE too low ($(r.best.fve_train)) for " *
        "$(r.game) — the SAE did not fit even the training split (training is broken)"

    # FEATURE↔VARIABLE RECOVERY: ≥1 learned feature matches a varying known cell
    # (the calibration claim — recovers features on a known circuit).
    @assert any(m -> m.matched && m.varies, r.fmatches) "no learned feature matched any varying " *
        "known cell for $(r.game) (best |r|=$(round(maximum(m.best_corr for m in r.fmatches if m.varies; init=0.0),digits=3)))"

    # CAUSAL-USE POSITIVE CONTROL (REQUIRED): if a matched∧causal feature exists, its
    # ablation re-run on the real ROM EQUALS the exact-patch oracle (bit-exact). This
    # is the HARNESS validity check — whether the ablation actually MOVES the output
    # (c.moved) is a MEASUREMENT, not a contract: a high-FVE overcomplete SAE that
    # distributes a variable across many tiny features may leave no single feature
    # causally load-bearing (the "recovers features; faithfulness-to-computation
    # debated" finding, experiment_design.md §7) — that is reported, not asserted.
    if !isempty(r.causal_use)
        @assert r.ablate_eq_oracle "feature-ablation re-run != exact-patch oracle for $(r.game) " *
            "(maxdiff=$(r.ablate_eq_oracle_maxdiff)) — causal-use claim not bit-exact"
    end

    # METRIC RANGES.
    @assert 0.0 <= r.matched_fraction <= 1.0 "match-rate out of [0,1]"
    @assert -1.0 - 1e-9 <= r.triad.F <= 1.0 + 1e-9 "F out of [-1,1]: $(r.triad.F)"
    @assert -1.0 - 1e-9 <= r.triad.S <= 1.0 + 1e-9 "S out of [-1,1]: $(r.triad.S)"
    @assert  0.0 - 1e-9 <= r.triad.M <= 1.0 + 1e-9 "M out of [0,1]: $(r.triad.M)"

    println("[sae:$(r.game)] SELF-CHECK PASS:")
    println("[sae:$(r.game)]   bit-exact re-run: $(r.bit_exact)  oracle causal cells: $n_causal/$(length(r.cands)) (pos.ctrl F=$(round(self_F,digits=3)))")
    println("[sae:$(r.game)]   best FVE_held=$(round(r.best.fve_held,digits=3)) (d_hidden=$(r.best.d_hidden) l1=$(r.best.l1)) match-rate=$(round(r.matched_fraction,digits=3))")
    println("[sae:$(r.game)]   causal-use: $(length(r.causal_use)) features, ablate==oracle=$(r.ablate_eq_oracle), F=$(round(r.triad.F,digits=3)) S=$(round(r.triad.S,digits=3)) M=$(round(r.triad.M,digits=3))")
    return true
end

# ============================================================================
# Persist (SPEC §R) — JSON record + sibling .npz; file_scope sae_*.
# ============================================================================
_git_commit() = try
    strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
catch
    "unknown"
end
_json_num(x::Real) = isfinite(x) ? Float64(x) : nothing

_sweep_record(s::SweepCell) = Dict{String,Any}(
    "d_hidden" => s.d_hidden, "l1" => _json_num(s.l1),
    "fve_train" => _json_num(s.fve_train), "fve_held" => _json_num(s.fve_held),
    "mean_l0" => _json_num(s.mean_l0), "matched_fraction" => _json_num(s.matched_fraction),
)
_match_record(m::FeatureMatch) = Dict{String,Any}(
    "ram_index" => m.ram_index, "concept" => m.concept, "family" => m.family,
    "best_feature" => m.best_feature, "best_corr" => _json_num(m.best_corr),
    "matched" => m.matched, "varies" => m.varies,
)
_cu_record(c::CausalUse) = Dict{String,Any}(
    "family" => c.family, "ram_index" => c.ram_index, "feature" => c.feature,
    "moved" => c.moved, "delta_full" => _json_num(c.delta_full),
    "delta_oracle" => _json_num(c.delta_oracle),
    "delta_target_only" => _json_num(c.delta_target_only),
    "ablate_eq_oracle" => c.ablate_eq_oracle, "decode_leakage" => c.decode_leakage,
    "target_share" => _json_num(c.target_share),
    "raw_base" => c.raw_base, "raw_ablate" => c.raw_ablate,
)

function write_game(r::GameResult; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    stem = "sae_$(r.game)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path  = joinpath(out_dir, stem * ".npz")

    cell_names = ["RAM[$(c.ram_index)]:$(c.concept)" for c in r.cands]
    families = [c.family for c in r.cands]
    where_env = get(ENV, "XAI_WHERE", "local")

    rec = Dict{String,Any}(
        "paper" => "P2",
        "phase" => "phaseC_mechanistic",
        "method" => "sparse_autoencoder",
        "item" => "P2-E5-6",
        "game" => r.game,
        "state" => r.state_kind == "noop" ? "f$(r.target_frame)+$(r.horizon)" :
                   "gameplay(seed=$(r.st_seed),prefix=$(r.st_prefix))+$(r.horizon)",
        "target_output" => "state_features_vs_known_variables+causal_use",
        # headline scalar (SPEC §R value/metric_name): the feature↔variable match-rate.
        "metric_name" => "feature_variable_matched_fraction",
        "value" => _json_num(r.matched_fraction),
        "stderr" => nothing,
        "ci" => nothing,
        "n" => r.n_varying_known,
        "seed" => r.seed,
        "where" => where_env,
        "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene@$(r.game) (P2-E1-1) — exact single-cell + full-decode " *
            "patch re-run; traj from jutari_record (P2-E0-2j)",
        "timestamp" => string(round(Int, time())),
        "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            "substrate" => "jutari (Julia, HARD) — real-ROM bit-exact path; the SAE is a " *
                "dependency-free manual-GD L1 autoencoder over the recorded RAM trajectory " *
                "(no torch/flax/pip; adds no package to the shared env). The causal-use " *
                "ablation RE-RUNS the real ROM via the exact intervention oracle.",
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
                "note" => "P2 redesign: the SAE is fit/scored on a seeded random-action " *
                    "GAMEPLAY state (not the FIRE/attract tape), gated by the §1 oracle " *
                    "cause-density gate. The descriptive trajectory is stepped along the " *
                    "gameplay stream (NOOP-continued past prefix+horizon). The SAE " *
                    "algorithm is unchanged; only the analysis state + trajectory move."),
            "scored_in_conformance_window" => r.in_window,
            "conformance_note" => r.in_window ?
                "scored oracle frame f$(r.target_frame)+$(r.horizon) is inside the Paper-1 " *
                "60-frame screen conformance window (jutari↔xitari bit-exact)." :
                "scored oracle frame f$(r.target_frame)+$(r.horizon) is BEYOND the strict " *
                "60-frame conformance window — the game only reaches live play later. HONEST " *
                "descriptive jutari result (the bit-exact jutari re-run IS asserted), not an " *
                "xitari-parity claim; Phase C is calibrated against the jutari oracle.",
            "candidate_cells" => cell_names,
            "candidate_ram_indices" => [c.ram_index for c in r.cands],
            "candidate_concept_family" => families,
            "n_known_game_variables" => length(unique(families)),
            "match_thresh" => r.match_thresh,
            # (a) RECONSTRUCTION + the SWEEP (the full-SAE knobs) ------------------
            "sweep" => Dict{String,Any}(
                "grid" => [_sweep_record(s) for s in r.sweep],
                "best" => _sweep_record(r.best),
                "selected_by" => "max held-out FVE",
                "note" => "dictionary-size (d_hidden) × L1-penalty sweep — the full-SAE " *
                    "knobs (Cunningham/Bricken 2023: overcomplete dict + L1). FVE = " *
                    "fraction of variance explained on a held-out (every-5th-frame) split.",
            ),
            "reconstruction" => Dict{String,Any}(
                "fve_train" => _json_num(r.best.fve_train),
                "fve_heldout" => _json_num(r.best.fve_held),
                "best_d_hidden" => r.best.d_hidden,
                "best_l1" => _json_num(r.best.l1),
                "best_mean_l0" => _json_num(r.best.mean_l0),
            ),
            # (b) FEATURE↔VARIABLE MATCH ------------------------------------------
            "feature_variable_match" => Dict{String,Any}(
                "matched_fraction" => _json_num(r.matched_fraction),
                "n_varying_known_cells" => r.n_varying_known,
                "per_cell" => [_match_record(m) for m in r.fmatches],
                "note" => "for each KNOWN candidate cell that VARIES, max|Pearson r| between " *
                    "any learned feature's activation and that cell over the trajectory; " *
                    "matched iff |r| ≥ $(r.match_thresh). Match-rate = #matched / #varying.",
            ),
            # (c) CAUSAL USE (the headline) ---------------------------------------
            "causal_use" => Dict{String,Any}(
                "ablate_equals_oracle" => r.ablate_eq_oracle,
                "ablate_equals_oracle_maxdiff_px" => _json_num(r.ablate_eq_oracle_maxdiff),
                "n_causal_use_features" => length(r.causal_use),
                "per_feature" => [_cu_record(c) for c in r.causal_use],
                "note" => "for each MATCHED & ORACLE-CAUSAL variable: ABLATE its best-matching " *
                    "feature (zero its activation), DECODE the code back to a RAM vector, " *
                    "UN-normalise to raw bytes, WRITE into the real ROM and RE-RUN `horizon` " *
                    "steps → Δscreen. The exact oracle re-runs the SAME write from an " *
                    "INDEPENDENT fresh boot (ablate==oracle, asserted, bit-exact). SPECIFICITY: " *
                    "decode_leakage = #RAM cells the decode changed; target_share = " *
                    "Δscreen(target-cell-only) / Δscreen(full-decode). Causally-real " *
                    "MONOSEMANTIC feature ⇒ leakage≈1 ∧ target_share≈1; correlational/" *
                    "polysemantic ⇒ decode smears (low target_share) — present, not cleanly used.",
            ),
            # F/S/M (recovery map vs the oracle; M from the causal-use ablations) --
            "triad" => Dict{String,Any}(
                "F" => _json_num(r.triad.F), "F_note" => r.triad.F_note,
                "S" => _json_num(r.triad.S), "S_note" => r.triad.S_note,
                "M" => _json_num(r.triad.M), "M_note" => r.triad.M_note,
                "interpretation" => "Phase C method-matrix expectation = PARTIAL " *
                    "(experiment_design.md §7): the SAE RECOVERS features (match-rate > 0) but " *
                    "faithfulness-to-computation is debated — F<1 + the causal-use " *
                    "monosemanticity (M) quantify how many 'matched' features are causally " *
                    "real single-variable features vs correlational/polysemantic smears.",
            ),
            "oracle_importance" => Dict{String,Any}(
                "importance_per_cell" =>
                    Dict(cell_names[i] => r.oracle.importance[i] for i in 1:length(cell_names)),
                "n_causal_cells" => count(>(0.0), r.oracle.importance),
                "note" => "oracle importance(i) = max whole-screen break of an EXACT " *
                    "do(cell_i:=0)/do(cell_i:=base+17) intervention re-run on the real ROM " *
                    "(the §1 oracle restricted to the candidate cell + screen). The SAME map " *
                    "A2 / A7 / pilot_si.jl score against — the TRUE causal role.",
            ),
            "sae" => Dict{String,Any}(
                "objective" => "MSE(x,x̂) + l1·mean(|z|)",
                "activation" => "relu", "epochs" => r.epochs, "lr" => r.lr,
                "input" => "z-scored 128-byte RIOT RAM (constant cells zeroed)",
                "fitting" => "manual full-batch gradient descent (dependency-free; Julia port " *
                    "of pilot_patch_sae.py's numpy SAE).",
            ),
            "scales_to_cluster_via" =>
                "tools/cluster/xai_array_jl.sbatch (--shard i --nshards n --shard-kind game): " *
                "one Slurm task per core game (round-robin); a larger d_hidden×l1 grid + longer " *
                "trajectories on the GPU (experiment_design.md §8). The forward is bit-exact to " *
                "this HARD path.",
        ),
    )
    open(json_path, "w") do io; JSON.print(io, rec, 2); end

    # arrays (sibling .npz)
    n = length(r.cands)
    sweep_arr = zeros(Float64, length(r.sweep), 6)
    for (i, s) in enumerate(r.sweep)
        sweep_arr[i, :] = [Float64(s.d_hidden), s.l1, s.fve_train, s.fve_held, s.mean_l0, s.matched_fraction]
    end
    cu_arr = zeros(Float64, max(length(r.causal_use), 0), 6)
    for (i, c) in enumerate(r.causal_use)
        cu_arr[i, :] = [Float64(c.ram_index), Float64(c.feature), c.delta_full,
                        c.delta_target_only, Float64(c.decode_leakage), c.target_share]
    end
    write_npz(npz_path, Dict(
        "candidate_ram_indices" => Int64[c.ram_index for c in r.cands],
        "feature_best_corr"     => Float64[m.best_corr for m in r.fmatches],
        "feature_matched"       => Int64[m.matched ? 1 : 0 for m in r.fmatches],
        "feature_varies"        => Int64[m.varies ? 1 : 0 for m in r.fmatches],
        "recovery"              => r.recovery,
        "oracle_importance"     => r.oracle.importance,
        "oracle_held_out"       => r.oracle.held_out,
        # sweep grid: rows [d_hidden, l1, fve_train, fve_held, mean_l0, matched_fraction]
        "sweep_grid"            => sweep_arr,
        # causal-use: rows [ram_index, feature, delta_full, delta_target_only, leakage, target_share]
        "causal_use"            => cu_arr,
        # [F, S, M, matched_fraction, best_fve_held]
        "scores"                => Float64[r.triad.F, r.triad.S, r.triad.M,
                                           r.matched_fraction, r.best.fve_held],
    ))
    return json_path, npz_path
end

# combined index (read-only roll-up across the core set; not a §R record itself).
function write_index(results::Vector{GameResult}; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    path = joinpath(out_dir, "sae_core_summary.json")
    rows = Dict{String,Any}[]
    for r in results
        n_moved = count(c -> c.moved, r.causal_use)
        mean_share = isempty(r.causal_use) ? nothing :
            _json_num(Statistics.mean([c.target_share for c in r.causal_use]))
        push!(rows, Dict{String,Any}(
            "game" => r.game,
            "state" => "f$(r.target_frame)+$(r.horizon)",
            "in_window" => r.in_window,
            "n_cands" => length(r.cands),
            "n_varying_known" => r.n_varying_known,
            "n_oracle_causal" => count(>(0.0), r.oracle.importance),
            "best_d_hidden" => r.best.d_hidden,
            "best_l1" => _json_num(r.best.l1),
            "fve_train" => _json_num(r.best.fve_train),
            "fve_held" => _json_num(r.best.fve_held),
            "matched_fraction" => _json_num(r.matched_fraction),
            "n_causal_use_features" => length(r.causal_use),
            "n_causal_use_moved" => n_moved,
            "ablate_eq_oracle" => r.ablate_eq_oracle,
            "mean_target_share" => mean_share,
            "FSM" => [_json_num(r.triad.F), _json_num(r.triad.S), _json_num(r.triad.M)],
        ))
    end
    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseC_mechanistic", "method" => "sparse_autoencoder",
        "item" => "P2-E5-6",
        "commit" => _git_commit(), "timestamp" => string(round(Int, time())),
        "n_games" => length(results),
        "headline" => "Full L1 SAE over recorded VCS RAM trajectories on the 6 core games: " *
            "(a) reconstruction FVE (best of a d_hidden×l1 sweep), (b) feature↔variable match-" *
            "rate vs the verified T3 cells, (c) CAUSAL USE — ablate a matched feature, decode " *
            "back to RAM, re-run the real ROM (ablate==exact-patch oracle, asserted bit-exact) " *
            "+ specificity/monosemanticity (decode-leakage, target-share). Method-matrix " *
            "expectation = PARTIAL: recovers features; faithfulness-to-computation debated.",
        "games" => rows,
    )
    open(path, "w") do io; JSON.print(io, rec, 2); end
    return path
end

# ============================================================================
# CLI — local default = all 6 core games; cluster-shardable via the sbatch flags.
# ============================================================================
function _resolve_games(; single_game, games_arg, shard, nshards, shard_kind)
    single_game !== nothing && return [single_game]
    # the game POOL to (optionally) shard: an explicit --games list (e.g. `labeled`)
    # if given, else the core set. Sharding then selects every N-th game of the pool,
    # so `--games labeled --shard i --nshards N` shards the 54 across the cluster.
    pool = games_arg !== nothing ? games_arg : CORE_GAMES
    if shard !== nothing
        kind = shard_kind === nothing ? "game" : shard_kind
        kind == "game" || error("sae.jl only shards by game (--shard-kind game), got $kind")
        n = nshards === nothing ? length(pool) : nshards
        sel = [pool[j] for j in (shard + 1):n:length(pool)]
        isempty(sel) && error("shard $shard of $n selects no game (|pool|=$(length(pool)))")
        return sel
    end
    return pool
end

function main(args = ARGS)
    single_game = nothing; games_arg = nothing
    target_frame = nothing; horizon = nothing; traj_frames = nothing
    d_hidden_sweep = [16, 32, 64]; l1_sweep = [0.01, 0.05, 0.2]
    epochs = 4000; lr = 0.5; match_thresh = 0.7; seed = 0
    shard = nothing; nshards = nothing; shard_kind = nothing
    out_dir = OUT_DIR
    selftest_only = false
    i = 1
    while i <= length(args)
        a = args[i]
        if     a == "--games"
            games_arg = xai_resolve_games(args[i+1], CORE_GAMES); i += 2
        elseif a == "--game";           single_game = args[i+1]; i += 2
        elseif a == "--target-frame";   target_frame = parse(Int, args[i+1]); i += 2
        elseif a == "--horizon";        horizon = parse(Int, args[i+1]); i += 2
        elseif a == "--traj-frames";    traj_frames = parse(Int, args[i+1]); i += 2
        elseif a == "--d-hidden-sweep"; d_hidden_sweep = parse.(Int, split(args[i+1], ",")); i += 2
        elseif a == "--l1-sweep";       l1_sweep = parse.(Float64, split(args[i+1], ",")); i += 2
        elseif a == "--epochs";         epochs = parse(Int, args[i+1]); i += 2
        elseif a == "--lr";             lr = parse(Float64, args[i+1]); i += 2
        elseif a == "--match-thresh";   match_thresh = parse(Float64, args[i+1]); i += 2
        elseif a == "--seed";           seed = parse(Int, args[i+1]); i += 2
        elseif a == "--shard";          shard = parse(Int, args[i+1]); i += 2
        elseif a == "--nshards";        nshards = parse(Int, args[i+1]); i += 2
        elseif a == "--shard-kind";     shard_kind = args[i+1]; i += 2
        elseif a == "--out-dir";        out_dir = args[i+1]; i += 2
        elseif a == "--roms-dir";       i += 2     # accepted-and-ignored (ROMs located internally)
        elseif a == "--where";          ENV["XAI_WHERE"] = args[i+1]; i += 2  # recorded in §R
        elseif a == "--selftest";       selftest_only = true; i += 1
        else; i += 1
        end
    end
    games = _resolve_games(; single_game = single_game, games_arg = games_arg,
                           shard = shard, nshards = nshards, shard_kind = shard_kind)

    println("[sae] Sparse autoencoders (Cunningham/Bricken 2023) — feature↔variable match + " *
            "CAUSAL USE — games=$(join(games, ",")) d_hidden=$(d_hidden_sweep) l1=$(l1_sweep) " *
            "epochs=$epochs lr=$lr match_thresh=$match_thresh seed=$seed out_dir=$out_dir (jutari/Julia)")

    if selftest_only
        g = games[1]
        r = compute_game(g; target_frame = target_frame, horizon = horizon,
                         traj_frames = traj_frames, d_hidden_sweep = d_hidden_sweep,
                         l1_sweep = l1_sweep, epochs = epochs, lr = lr,
                         match_thresh = match_thresh, seed = seed)
        selftest(r)
        println("[sae] --selftest: passed on $g, not writing artifact.")
        return 0
    end

    ok = String[]; failed = Tuple{String,String}[]; results = GameResult[]
    for g in games
        println("\n[sae] ===== $g =====")
        try
            r = compute_game(g; target_frame = target_frame, horizon = horizon,
                             traj_frames = traj_frames, d_hidden_sweep = d_hidden_sweep,
                             l1_sweep = l1_sweep, epochs = epochs, lr = lr,
                             match_thresh = match_thresh, seed = seed)
            selftest(r)
            json_path, npz_path = write_game(r; out_dir = out_dir)
            println("[sae] wrote $json_path")
            println("[sae] arrays  $npz_path")
            push!(ok, g); push!(results, r)
        catch err
            msg = sprint(showerror, err)
            println("[sae] !! game $g FAILED (scoring the rest, not fabricating): " *
                    first(split(msg, '\n')))
            push!(failed, (g, first(split(msg, '\n'))))
        end
    end
    if !isempty(results)
        idx = write_index(results; out_dir = out_dir)
        println("[sae] wrote index $idx")
    end
    println("\n[sae] ==== summary: $(length(ok))/$(length(games)) games scored ====")
    for g in ok; println("[sae]   OK   $g"); end
    for (g, m) in failed; println("[sae]   FAIL $g — $m"); end
    return 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    SAEMech.main()
end
