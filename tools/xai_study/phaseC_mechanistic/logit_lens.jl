# logit_lens.jl — Phase-C mechanistic interpretability (P2-E5-10), JULIA path (the
# jutari real-ROM substrate; jaxtari eager is ~205× slower — SCRUM §7).
#
# METHOD: LOGIT LENS / TUNED LENS (nostalgebraist 2020; Belrose et al. 2023) over
# the VCS state trajectory, scored against the EXACT intervention oracle
# (experiment_design.md §6 row "Logit / tuned lens"; §7: a probing/readout method
# whose headline danger is "present ≠ used"). T1/T2 scoring; no T3.
#
# ---------------------------------------------------------------------------
# What a LOGIT / TUNED LENS is here (the LLM analogy, made exact on the VCS):
#   In a transformer the logit lens decodes the FINAL output (the logits) from an
#   INTERMEDIATE residual stream by applying the *unembedding* directly to a hidden
#   state at an earlier layer; the TUNED lens (Belrose 2023) replaces the frozen
#   unembedding with a per-layer learned AFFINE readout `\hat y_ℓ = A_ℓ h_ℓ + b_ℓ`
#   fit so the early state decodes to the final output, and asks how early the
#   answer is "already there".
#
#   On the VCS the residual stream is the RAM TAPE and "layers" are the FRAMES of
#   the horizon: h_t = the 128-byte RAM at step t. The FINAL output we decode is a
#   target variable V's value at the END of the horizon (a RAM cell the program
#   reads V from — its T1/T2 ground-truth site). The lens asks: from the
#   INTERMEDIATE state at an earlier step t, how well can we read out V's FINAL
#   value?
#       * LOGIT lens (nostalgebraist, the frozen/identity readout) = read the cell
#         DIRECTLY at step t (\hat y_t = h_t[site_V]); the trivial "unembedding".
#       * TUNED lens (Belrose, the learned per-step affine readout) = fit
#         A_t, b_t by LEAST SQUARES over a set of trajectories so A_t h_t + b_t
#         best predicts V's FINAL value; ridge-regularised, per step.
#
# THE TRUE INTERMEDIATE VALUE (what makes the VCS special):
#   On the VCS the TRUE intermediate value of V at step t is KNOWN — it is the
#   actual content of V's site at step t along the real (bit-exact) trajectory. So
#   we don't just measure "does the lens predict the final output"; we measure
#   READOUT FIDELITY vs the TRUE INTERMEDIATE VALUE the program actually computes,
#   and — the discriminating VCS metric — we split predictive readouts into:
#       FAITHFUL   — the readout site is on V's TRUE causal path (intervening on it
#                    changes V's final output, per the exact oracle), AND the lens
#                    tracks the true intermediate it carries; vs
#       SPURIOUS   — the lens predicts V's final value from a site that is NOT
#                    causal for V (intervention shows zero effect) — a correlation
#                    the lens latched onto, the "present ≠ used" tuning-curve trap.
#   The intervention oracle (P2-E1-1) supplies the gold causal mask: site s is
#   causal-for-V iff do(s := s+17) at step t changes V's final value on the
#   bit-exact re-run.
#
# WHAT WE SCORE (DoD: readout fidelity vs the true intermediate value):
#   For each target variable V and each intermediate step t:
#   (1) LOGIT-lens fidelity  — corr / R² of \hat y_t = h_t[site_V] vs V's TRUE
#       intermediate value at t (the cell's own content), AND vs V's FINAL value
#       (the "answer is already there" curve). On the VCS the logit lens reading
#       the TRUE site is fidelity 1.0 to the true intermediate by construction
#       (it IS the cell) — the POSITIVE CONTROL that the readout point is correct.
#   (2) TUNED-lens fidelity  — R² of the fitted affine readout A_t h_t + b_t (over
#       held-out trajectories, train/test split) vs V's FINAL value, per step.
#       The tuned lens can decode the final value EARLIER than the logit lens
#       (Belrose's finding) because it linearly combines other cells — but on the
#       VCS we can ask whether that early decodability is FAITHFUL (it uses the
#       true causal predecessors) or SPURIOUS (it rides a correlated clock/counter).
#   (3) FAITHFUL-vs-SPURIOUS decomposition — for the tuned lens at the EARLIEST
#       step it decodes V (R² ≥ τ), measure the fraction of its readout WEIGHT
#       (|A_t| mass) that sits on TRUE-causal sites vs non-causal sites, using the
#       oracle causal mask. Report `faithful_weight_fraction`; high = the lens
#       reads the mechanism, low = it reads a spurious correlate. This is the
#       quantified "present ≠ used" story for the logit/tuned lens.
#
# Why this is the right Phase-C framing (the paper's point): the logit/tuned lens
# is a READOUT (a probe in disguise). On a normal network you cannot tell whether
# an early-decodable signal is the mechanism or a correlate. On the VCS the oracle
# gives the true intermediate AND the true causal mask, so we can score the lens'
# fidelity to the COMPUTATION, not just its predictiveness — exactly the
# faithfulness-vs-plausibility axis the paper benchmarks.
#
# No JuTari/jaxtari/xitari core is modified — pure tooling under tools/xai_study/.
# Reuses the validated foundations on main:
#   * jutari_record.jl — the per-frame state-trajectory recorder (E0-2j).
#   * jutari_oracle.jl — boot/replay/snapshot/deepcopy-checkpoint/intervene_ram! +
#     the dependency-free §R NPZ writer.
#   * oracle_intervene.jl — candidate_ram_indices (the game-agnostic candidate set
#     + its T3 concept labels), the cell↔variable ground-truth alignment.
#   * the per-game ROM-alias + RomSettings + candidates map (mirrors das.jl /
#     activation_patching.jl / ig_baseline_sweep.jl — NO emulator core touched).
#
# Run (warm shared depot, primary's project):
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseC_mechanistic/logit_lens.jl --games core
# Flags: --games core|<g1,g2,...>  --game <g>
#        --target-frame N --horizon N --n-traj K --selftest
#
# Writes (SPEC §R; file_scope logit_lens_* under out/):
#   tools/xai_study/phaseC_mechanistic/out/logit_lens_<game>.{json,npz}
#   tools/xai_study/phaseC_mechanistic/out/logit_lens_core_summary.json

module LogitLens

using LinearAlgebra
using Random
using Statistics

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen, get_ram

# the oracle's candidate cause set (game-agnostic) + the verified jutari run helper
# (boot/replay/snapshot/intervene + NPZ writer). We reuse the SAME resolution path
# as das.jl so the candidates/RomSettings/ROM-alias map is shared, not re-derived.
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

const OUT_DIR = joinpath(@__DIR__, "out")
const CORE_GAMES = ["pong", "breakout", "space_invaders", "seaquest", "ms_pacman", "qbert"]
# shared-testbed switch + params (redesign protocol: prefix=90 gameplay, horizon=15).
const SHARED_TESTBED = get(ENV, "XAI_SHARED_TESTBED", "1") == "1"
const ST_PREFIX  = parse(Int, get(ENV, "XAI_ST_PREFIX", "90"))
const ST_HORIZON = parse(Int, get(ENV, "XAI_ST_HORIZON", "15"))
const ST_SEED    = parse(Int, get(ENV, "XAI_ST_SEED", "0"))
const ST_GATE_K  = parse(Int, get(ENV, "XAI_ST_GATE_K", "4"))
const ST_FLOOR   = parse(Float64, get(ENV, "XAI_ST_FLOOR", "0.5"))

# joystick action codes (oracle_intervene.jl: RIGHT=3; LEFT=4). NOOP=0 is BASE.
const ACT_NOOP  = 0
const ACT_RIGHT = 3
const ACT_LEFT  = 4

# the directed-intervention magnitude used to build the oracle causal mask (mirrors
# the oracle's `set` cause base+17 and das.jl's directed source).
const SET_DELTA = 17

# ============================================================================
# Per-game ROM + RomSettings + candidates resolution (mirrors das.jl exactly; NO
# emulator core touched). seaquest has no registered RomSettings yet → Generic.
# ============================================================================
const ROM_BASENAME = Dict(
    "pong" => "pong", "breakout" => "breakout",
    "space_invaders" => "space_invaders", "seaquest" => "seaquest",
    "ms_pacman" => "mspacman", "qbert" => "qbert")

const _PRIMARY_REPO = get(ENV, "XAI_PRIMARY_REPO", "/Users/maier/Documents/code/UnderstandingVCS")

function rom_path_for(game::AbstractString)
    stem = get(ROM_BASENAME, lowercase(string(game)), lowercase(string(game)))
    here = normpath(joinpath(@__DIR__, "..", "..", ".."))
    for base in (here, _PRIMARY_REPO)
        p = joinpath(base, "xitari", "roms", stem * ".bin")
        isfile(p) && return p
    end
    error("ROM not found for game=$game (looked under $here and $_PRIMARY_REPO)")
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
# env construction + replay + checkpoint (mirrors jutari_oracle / das.jl)
# ============================================================================
function load_env(; game::AbstractString)
    rom = read(rom_path_for(game))
    env = StellaEnvironment(rom, settings_for(game))
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    return env
end

function boot_replay(actions, target_frame; game)
    env = load_env(; game = game)
    for i in 1:target_frame; env_step!(env, Int(actions[i])); end
    return env
end

# a Snapshot-returning continuation (mirrors das.jl / activation_patching.jl) — used
# by the shared-testbed substrate; not part of the lens algorithm proper.
function continue_from(checkpoint::StellaEnvironment, tail)
    env = deepcopy(checkpoint)
    for a in tail; env_step!(env, Int(a)); end
    return snapshot(env, length(tail))
end

"""Record the full RAM tape over `horizon` steps starting from a deepcopy of the
`base_ckpt`, applying `tail` actions. Returns a (horizon+1, RAM_SIZE) Float64
matrix whose row 1 is the state AT the checkpoint (step 0) and row t+1 is the
state after t tail actions. This is the "residual stream over layers/frames"."""
function record_ram_tape(base_ckpt::StellaEnvironment, tail::AbstractVector{<:Integer})
    env = deepcopy(base_ckpt)
    H = length(tail)
    tape = Matrix{Float64}(undef, H + 1, RAM_SIZE)
    tape[1, :] = Float64.(Int.(collect(get_ram(env))))
    for t in 1:H
        env_step!(env, Int(tail[t]))
        tape[t + 1, :] = Float64.(Int.(collect(get_ram(env))))
    end
    return tape   # (H+1, 128); row t (1-based) = state after t-1 tail steps
end

"""Assert two fresh boots+replays are byte-identical in RAM AND screen — the
load-bearing guarantee that the recorded tapes (and the oracle mask built from
the same path) are clean causal objects (mirrors das.assert_bit_exact)."""
function assert_bit_exact(actions, total; game)
    a = boot_replay(actions, total; game = game)
    b = boot_replay(actions, total; game = game)
    ra = collect(get_ram(a)); rb = collect(get_ram(b))
    ra == rb || error("bit-exact RAM re-run FAILED for $game: " *
        "$(count(ra .!= rb))/$(length(ra)) bytes differ to f$total")
    sa = Matrix{UInt8}(get_screen(a)); sb = Matrix{UInt8}(get_screen(b))
    sa == sb || error("bit-exact SCREEN re-run FAILED for $game: " *
        "$(count(sa .!= sb)) px differ to f$total")
    return true
end

# ============================================================================
# The set of trajectories the tuned lens is FIT over. We need a SET of runs so the
# least-squares readout has data — a single deterministic NOOP run gives one point
# per step. We build K trajectories by perturbing the BASE state at the checkpoint
# (a small directed shift on each candidate cell in turn, plus a few joystick
# contexts), each then re-run forward for the horizon. Every trajectory is a
# bit-exact real-ROM run from a fully-specified state, so the recorded tapes are
# genuine program states (not synthetic noise) — the regression is over real
# computations of the VCS, which is the point of the known-circuit testbed.
# ============================================================================
struct TrajSet
    H::Int                     # horizon
    sites::Vector{Int}         # the candidate RAM sites (0-based), aligned to vars
    concepts::Vector{String}   # T3 concept label per site (may be empty string)
    tapes::Vector{Matrix{Float64}}  # K tapes, each (H+1, RAM_SIZE)
    K::Int
end

function build_trajset(base_ckpt::StellaEnvironment, src_ckpt::StellaEnvironment,
                       tail::AbstractVector{<:Integer}, sites::Vector{Int},
                       concepts::Vector{String}; n_traj::Int = 24)
    H = length(tail)
    tapes = Matrix{Float64}[]
    # trajectory 0: the clean BASE run (the running example / oracle baseline).
    push!(tapes, record_ram_tape(base_ckpt, tail))
    # the SOURCE run (a genuinely different joystick context) — a second organic run.
    push!(tapes, record_ram_tape(src_ckpt, tail))
    base_ram = Int.(collect(get_ram(base_ckpt)))
    # the rest: perturb ONE candidate cell at the checkpoint by a per-trajectory
    # offset, re-run. Cycling sites × a few offsets gives K diverse real runs whose
    # tapes vary in the program's own variables (so the regression is conditioned).
    offsets = (SET_DELTA, 2 * SET_DELTA, 64, 128, 200)
    k = 1
    while length(tapes) < n_traj && !isempty(sites)
        site = sites[mod1(k, length(sites))]
        off  = offsets[mod1(fld(k - 1, length(sites)) + 1, length(offsets))]
        env = deepcopy(base_ckpt)
        intervene_ram!(env, site, (base_ram[site + 1] + off) & 0xFF)
        push!(tapes, record_ram_tape(env, tail))
        k += 1
        k > 4 * n_traj && break   # safety
    end
    return TrajSet(H, sites, concepts, tapes, length(tapes))
end

# ============================================================================
# Least-squares readouts (LinearAlgebra; no shared-venv pip).
# ============================================================================
"""Ridge least-squares: solve (XᵀX + λI) w = Xᵀy for w (X already has a bias
column). Returns the weight vector w (length = ncols of X)."""
function ridge_fit(X::Matrix{Float64}, y::Vector{Float64}; λ::Float64 = 1e-3)
    p = size(X, 2)
    A = X' * X + λ * Matrix{Float64}(I, p, p)
    b = X' * y
    return A \ b
end

"""R² (coefficient of determination) of predictions `ŷ` vs targets `y`. R²=1 is
perfect; R²≤0 means worse than predicting the mean. Constant-target guard: if y is
constant, R²=1.0 iff predictions match it exactly, else 0.0."""
function r2(y::Vector{Float64}, ŷ::Vector{Float64})
    ss_res = sum((y .- ŷ) .^ 2)
    ȳ = mean(y)
    ss_tot = sum((y .- ȳ) .^ 2)
    if ss_tot == 0.0
        return ss_res == 0.0 ? 1.0 : 0.0
    end
    return 1.0 - ss_res / ss_tot
end

"""Pearson correlation, NaN-safe (returns 0.0 if either side is constant)."""
function pearson(a::Vector{Float64}, b::Vector{Float64})
    (length(a) < 2) && return 0.0
    sa = std(a); sb = std(b)
    (sa == 0.0 || sb == 0.0) && return 0.0
    return cov(a, b) / (sa * sb)
end

# ============================================================================
# Per-game result
# ============================================================================
struct LensResult
    game::String
    target_frame::Int
    horizon::Int
    K::Int                          # number of trajectories fit over
    n_train::Int
    n_test::Int
    sites::Vector{Int}              # candidate RAM sites (0-based), the target vars
    concepts::Vector{String}
    var_names::Vector{String}
    bit_exact::Bool
    # per (variable, step) arrays — rows = variables, cols = steps 0..H
    logit_fid_true::Matrix{Float64}    # logit-lens R² vs TRUE intermediate value
    logit_fid_final::Matrix{Float64}   # logit-lens R² vs FINAL value
    tuned_fid_final::Matrix{Float64}   # tuned-lens (held-out) R² vs FINAL value
    earliest_logit::Vector{Int}        # earliest step the logit lens decodes (R²≥τ), -1 if none
    earliest_tuned::Vector{Int}        # earliest step the tuned lens decodes (R²≥τ), -1 if none
    faithful_weight_frac::Vector{Float64}  # tuned-lens faithful-weight fraction at earliest step
    causal_mask::Matrix{Bool}          # (var, site): is site causal for var's final value? (oracle)
    best_tuned_per_var::Vector{Float64}  # per-var peak (over steps) held-out tuned R² (clipped≥0)
    # headline scalars
    logit_true_fidelity::Float64       # mean logit-lens fidelity to TRUE intermediate (positive control)
    mean_tuned_final_r2::Float64       # mean tuned-lens final R² across vars/steps (RAW, can be <0)
    median_tuned_final_r2::Float64     # median (robust) held-out tuned R² across vars/steps
    mean_best_tuned_r2::Float64        # mean of per-var peak held-out tuned R² (the headline: peak decodability)
    mean_faithful_weight_frac::Float64 # mean faithful-weight fraction (low ⇒ spurious readout)
    tuned_beats_logit_earlier::Float64 # frac of vars the tuned lens decodes strictly earlier
    n_vars::Int
    decode_threshold::Float64
    # SHARED-TESTBED provenance (redesign); all NaN/-1/"" in the legacy NOOP path.
    state_kind::String                 # "seeded_random_action_gameplay" | "noop"
    st_seed::Int
    st_prefix::Int
    cause_density::Int                 # #causes above the floor at the shared output
    cause_density_accepted::Bool       # passed the cause-density gate?
    n_causes::Int
    shared_cell::Tuple{Int,Int}        # the shared screen-buffer output cell
end

"""Build the oracle CAUSAL MASK: for each candidate variable V (its site) and each
candidate site s, is s causal for V's FINAL value? do(s := s+17) at the checkpoint,
re-run the horizon, check whether V's final cell value changed. Bit-exact, so a
nonzero change is a true causal edge. (var, site) Bool matrix."""
function build_causal_mask(base_ckpt::StellaEnvironment, tail::AbstractVector{<:Integer},
                           sites::Vector{Int})
    nsite = length(sites)
    base_final = Int.(collect(get_ram(_run_tail(deepcopy(base_ckpt), tail))))
    base_ram = Int.(collect(get_ram(base_ckpt)))
    mask = falses(nsite, nsite)   # mask[v, s] : site s causal for var v's final value
    for (si, s) in enumerate(sites)
        env = deepcopy(base_ckpt)
        intervene_ram!(env, s, (base_ram[s + 1] + SET_DELTA) & 0xFF)
        fin = Int.(collect(get_ram(_run_tail(env, tail))))
        for (vi, v) in enumerate(sites)
            mask[vi, si] = fin[v + 1] != base_final[v + 1]
        end
    end
    return mask
end

function _run_tail(env::StellaEnvironment, tail::AbstractVector{<:Integer})
    for a in tail; env_step!(env, Int(a)); end
    return env
end

function run_game(; game, target_frame = 120, horizon = 30, n_traj = 48,
                  decode_threshold = 0.9, train_frac = 0.7, seed = 0, verbose = true)
    # SHARED-TESTBED (redesign): replace the all-NOOP boot/attract tape with a
    # seeded random-action GAMEPLAY state at f*=ST_PREFIX, gated by the oracle
    # cause-density gate. BASE + SOURCE checkpoints, the tail, and the trajectory
    # set all come from the shared substrate so every method sits on the SAME state
    # (P1, P4). The lens algorithm (trajectory set, causal mask, ridge fits) is
    # unchanged; only the state/action-stream it sits on moves.
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
        # SOURCE run: share the gameplay prefix, diverge only at the analysis frame
        # with a RIGHT joystick action (an on-distribution genuinely different state).
        pre = target_frame > 0 ? Int.(base_actions[1:target_frame - 1]) : Int[]
        src_ckpt = boot_replay(vcat(pre, ACT_RIGHT), target_frame; game = game)
        verbose && println("[$game] SHARED gameplay state: cause_density=$(st.cause_density)/" *
            "$(length(st.causes)) accepted=$(st.accepted) cell=$(st.cell)")
        return _run_game_body(; game = game, target_frame = target_frame, horizon = horizon,
            base_ckpt = base_ckpt, src_ckpt = src_ckpt, tail = tail, n_traj = n_traj,
            decode_threshold = decode_threshold, train_frac = train_frac, seed = seed,
            bit_exact = bit_exact, st = st, verbose = verbose)
    end

    total = target_frame + horizon
    base_actions = fill(ACT_NOOP, total)

    # ---- 1) bit-exact guarantee (RAM AND screen, two fresh boots+replays) ------
    verbose && println("[$game] asserting bit-exactness (2 fresh replays to f$total)...")
    assert_bit_exact(base_actions, total; game = game)
    verbose && println("[$game] bit-exact re-run: PASS")
    bit_exact = true

    # ---- 2) BASE + SOURCE checkpoints at the target frame ----------------------
    base_ckpt = boot_replay(base_actions, target_frame; game = game)
    src_actions = fill(ACT_RIGHT, total)
    src_ckpt = boot_replay(src_actions, target_frame; game = game)
    tail = base_actions[target_frame + 1 : total]   # H NOOPs
    return _run_game_body(; game = game, target_frame = target_frame, horizon = horizon,
        base_ckpt = base_ckpt, src_ckpt = src_ckpt, tail = tail, n_traj = n_traj,
        decode_threshold = decode_threshold, train_frac = train_frac, seed = seed,
        bit_exact = bit_exact, st = nothing, verbose = verbose)
end

"""The per-game body, shared by the legacy NOOP path and the SHARED gameplay-state
path (the ONLY difference is which state/action-stream/checkpoints the lens
algorithm sits on — the algorithm itself is unchanged). `st` carries the
shared-testbed provenance for the record, or `nothing` in the legacy path."""
function _run_game_body(; game, target_frame, horizon, base_ckpt, src_ckpt, tail,
                        n_traj, decode_threshold, train_frac, seed, bit_exact, st, verbose)

    # ---- 3) candidate sites + their T3 concept labels (the target variables) ---
    cand = candidates_path_for(game)
    cand_ic = candidate_ram_indices(cand)            # [(idx, concept), ...]
    sites = [idx for (idx, _) in cand_ic]
    concepts = [string(c) for (_, c) in cand_ic]
    var_names = [isempty(concepts[i]) ? "ram@$(sites[i])" : "$(concepts[i])@$(sites[i])"
                 for i in eachindex(sites)]
    nvar = length(sites)
    H = length(tail)

    # ---- 4) trajectory SET the tuned lens is fit over (real-ROM runs) ----------
    ts = build_trajset(base_ckpt, src_ckpt, tail, sites, concepts; n_traj = n_traj)
    K = ts.K
    verbose && println("[$game] recorded K=$K trajectories × (H+1=$(H+1)) steps × $RAM_SIZE cells")

    # train/test split over trajectories (held-out — the tuned lens must generalise)
    rng = MersenneTwister(seed)
    perm = randperm(rng, K)
    n_train = max(2, round(Int, train_frac * K))
    n_train = min(n_train, K - 1)            # keep at least 1 test trajectory
    train_idx = perm[1:n_train]
    test_idx  = perm[n_train + 1 : end]
    n_test = length(test_idx)

    # ---- 5) the oracle causal mask (true data-flow into each var's final value) -
    causal_mask = build_causal_mask(base_ckpt, tail, sites)

    # ---- 6) per-variable, per-step lens fidelity -------------------------------
    nstep = H + 1   # steps 0..H
    logit_fid_true  = fill(NaN, nvar, nstep)
    logit_fid_final = fill(NaN, nvar, nstep)
    tuned_fid_final = fill(NaN, nvar, nstep)
    earliest_logit  = fill(-1, nvar)
    earliest_tuned  = fill(-1, nvar)
    faithful_weight_frac = fill(NaN, nvar)

    # FINAL value of variable v = its own cell at the LAST recorded step (row H+1),
    # per trajectory. The lens decodes THIS from the intermediate state at step t.
    final_vals = [ [ ts.tapes[k][H + 1, sites[v] + 1] for k in 1:K ] for v in 1:nvar ]

    for v in 1:nvar
        site_col = sites[v] + 1
        # the TRUE intermediate value of v at each step = its own cell at that step.
        for t in 1:nstep
            # ---- LOGIT LENS (identity readout: read the cell directly at step t) --
            # \hat y_t = h_t[site_v]. Across trajectories at fixed step t.
            ŷ_logit = [ ts.tapes[k][t, site_col] for k in 1:K ]
            true_int = ŷ_logit                          # logit lens reads the true cell ⇒ IS the true intermediate
            fin = final_vals[v]
            # fidelity to TRUE intermediate: trivially 1.0 (the readout IS the cell) —
            # the positive control that the readout point/site is correct.
            logit_fid_true[v, t]  = r2(true_int, ŷ_logit)
            # fidelity to FINAL value: how early the cell already equals its final value.
            logit_fid_final[v, t] = r2(fin, ŷ_logit)

            # ---- TUNED LENS (learned per-step affine readout, held-out) ----------
            # fit A_t, b_t on TRAIN trajectories: A_t h_t + b_t ≈ final value of v.
            Xtr = hcat([ts.tapes[k][t, :] for k in train_idx]...)'  # (n_train, RAM_SIZE)
            Xtr = Matrix{Float64}(Xtr)
            Xtr_b = hcat(Xtr, ones(size(Xtr, 1)))                   # bias column
            ytr = Float64[ final_vals[v][k] for k in train_idx ]
            w = ridge_fit(Xtr_b, ytr)
            # evaluate on HELD-OUT test trajectories
            Xte = hcat([ts.tapes[k][t, :] for k in test_idx]...)'
            Xte = Matrix{Float64}(Xte)
            Xte_b = hcat(Xte, ones(size(Xte, 1)))
            yte = Float64[ final_vals[v][k] for k in test_idx ]
            ŷ_tuned = Xte_b * w
            tuned_fid_final[v, t] = r2(yte, ŷ_tuned)

            # earliest decoding steps (first t with R² ≥ threshold)
            if earliest_logit[v] == -1 && logit_fid_final[v, t] >= decode_threshold
                earliest_logit[v] = t - 1   # step index 0..H
            end
            if earliest_tuned[v] == -1 && tuned_fid_final[v, t] >= decode_threshold
                earliest_tuned[v] = t - 1
                # faithful-weight fraction at the earliest tuned-decode step: the share
                # of |A_t| weight mass on TRUE-causal sites (oracle mask) vs all sites.
                wmag = abs.(w[1:RAM_SIZE])
                causal_sites = [sites[s] + 1 for s in 1:nvar if causal_mask[v, s]]
                total_mass = sum(wmag) + 1e-12
                faithful_mass = isempty(causal_sites) ? 0.0 : sum(wmag[causal_sites])
                faithful_weight_frac[v] = faithful_mass / total_mass
            end
        end
    end

    # ---- 7) headline scalars ---------------------------------------------------
    logit_true_fidelity = mean(filter(isfinite, vec(logit_fid_true)))
    # the tuned-lens held-out R² is the WRONG thing to summarise with a plain mean:
    # a few catastrophic early-step extrapolations on held-out trajectories (R²≪0,
    # the high-dim ridge readout for a not-yet-determined variable) dominate it. The
    # standard tuned-lens convention is to FLOOR R² at 0 (a readout worse than the
    # mean baseline has 0 skill, not −800), then summarise by the PEAK (over steps)
    # per variable (= "is this variable ever linearly readable") + the MEDIAN
    # (robust to the outliers). We keep the RAW mean too, for full transparency.
    finite_tuned = filter(isfinite, vec(tuned_fid_final))
    mean_tuned_final_r2 = isempty(finite_tuned) ? NaN : mean(finite_tuned)            # RAW (can be ≪0)
    median_tuned_final_r2 = isempty(finite_tuned) ? NaN : median(finite_tuned)        # robust
    # per-variable peak held-out R², clipped at 0 (skill floor) — the headline.
    best_tuned_per_var = fill(NaN, nvar)
    for v in 1:nvar
        row = filter(isfinite, tuned_fid_final[v, :])
        best_tuned_per_var[v] = isempty(row) ? NaN : max(0.0, maximum(row))
    end
    finite_best = filter(isfinite, best_tuned_per_var)
    mean_best_tuned_r2 = isempty(finite_best) ? NaN : mean(finite_best)
    finite_fwf = filter(isfinite, faithful_weight_frac)
    mean_faithful_weight_frac = isempty(finite_fwf) ? NaN : mean(finite_fwf)
    # fraction of vars the tuned lens decodes strictly EARLIER than the logit lens
    earlier = 0; comparable = 0
    for v in 1:nvar
        (earliest_tuned[v] == -1 && earliest_logit[v] == -1) && continue
        comparable += 1
        et = earliest_tuned[v] == -1 ? H + 1 : earliest_tuned[v]
        el = earliest_logit[v] == -1 ? H + 1 : earliest_logit[v]
        et < el && (earlier += 1)
    end
    tuned_beats_logit_earlier = comparable == 0 ? 0.0 : earlier / comparable

    verbose && println("[$game] logit-lens fidelity(TRUE intermediate)=" *
                       "$(round(logit_true_fidelity, digits = 4)) (pos.control) | " *
                       "mean PEAK tuned-lens R²=$(round(mean_best_tuned_r2, digits = 3)) " *
                       "(median final R²=$(round(median_tuned_final_r2, digits = 3))) | " *
                       "mean faithful-weight frac=$(round(mean_faithful_weight_frac, digits = 3)) | " *
                       "tuned earlier than logit=$(round(tuned_beats_logit_earlier, digits = 3))")

    return LensResult(game, target_frame, horizon, K, n_train, n_test,
                      sites, concepts, var_names, bit_exact,
                      logit_fid_true, logit_fid_final, tuned_fid_final,
                      earliest_logit, earliest_tuned, faithful_weight_frac,
                      causal_mask, best_tuned_per_var,
                      logit_true_fidelity, mean_tuned_final_r2,
                      median_tuned_final_r2, mean_best_tuned_r2,
                      mean_faithful_weight_frac, tuned_beats_logit_earlier,
                      nvar, decode_threshold,
                      st === nothing ? "noop" : "seeded_random_action_gameplay",
                      st === nothing ? -1 : st.seed,
                      st === nothing ? -1 : st.prefix,
                      st === nothing ? -1 : st.cause_density,
                      st === nothing ? false : st.accepted,
                      st === nothing ? 0 : length(st.causes),
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

function write_game_result(r::LensResult; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    stem = "logit_lens_$(r.game)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path  = joinpath(out_dir, stem * ".npz")

    var_meta = Dict{String,Any}[]
    for v in 1:r.n_vars
        push!(var_meta, Dict{String,Any}(
            "name" => r.var_names[v],
            "site" => r.sites[v],
            "concept" => r.concepts[v],
            "earliest_logit_step" => r.earliest_logit[v],
            "earliest_tuned_step" => r.earliest_tuned[v],
            "faithful_weight_fraction" => (isfinite(r.faithful_weight_frac[v]) ?
                                           r.faithful_weight_frac[v] : nothing),
            "best_tuned_r2_heldout" => (isfinite(r.best_tuned_per_var[v]) ?
                                        r.best_tuned_per_var[v] : nothing),
            "n_causal_sites" => count(r.causal_mask[v, :]),
        ))
    end

    rec = Dict{String,Any}(
        "paper" => "P2",
        "phase" => "phaseC_mechanistic",
        "method" => "logit_tuned_lens",
        "game" => r.game,
        "state" => r.state_kind == "noop" ? "f$(r.target_frame)+$(r.horizon)" :
                   "gameplay(seed=$(r.st_seed),prefix=$(r.st_prefix))+$(r.horizon)",
        "target_output" => "per-stage readout of intermediate VCS state -> a variable's final value",
        # headline §R scalar: logit-lens fidelity to the TRUE intermediate value
        # (the DoD metric). =1.0 by construction on the VCS (the readout IS the
        # cell) — the positive control that the readout point is the true site.
        "metric_name" => "logit_lens_fidelity_true_intermediate",
        "value" => r.logit_true_fidelity,
        "stderr" => nothing,
        "ci" => nothing,
        "n" => r.n_vars,
        "seed" => 0,
        "where" => "local",
        "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene@$(r.game) (P2-E1-1) — exact causal mask + true intermediate",
        "timestamp" => string(round(Int, time())),
        "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            "substrate" => "jutari (Julia, HARD) — real-ROM bit-exact path",
            "variables" => var_meta,
            "bit_exact_rerun" => r.bit_exact,
            "n_trajectories" => r.K,
            "n_train_trajectories" => r.n_train,
            "n_test_trajectories" => r.n_test,
            "decode_threshold_r2" => r.decode_threshold,
            "n_variables" => r.n_vars,
            # the headline lens metrics
            "logit_lens_fidelity_true_intermediate" => r.logit_true_fidelity,
            # PEAK (over steps) held-out tuned R², clipped at 0, averaged over vars —
            # the robust headline of "how well the learned readout can ever decode".
            "mean_best_tuned_lens_r2_heldout" => (isfinite(r.mean_best_tuned_r2) ?
                                                  r.mean_best_tuned_r2 : nothing),
            "median_tuned_lens_final_r2_heldout" => (isfinite(r.median_tuned_final_r2) ?
                                                     r.median_tuned_final_r2 : nothing),
            "mean_tuned_lens_final_r2_heldout_raw" => (isfinite(r.mean_tuned_final_r2) ?
                                                       r.mean_tuned_final_r2 : nothing),
            "mean_faithful_weight_fraction" => (isfinite(r.mean_faithful_weight_frac) ?
                                                r.mean_faithful_weight_frac : nothing),
            "tuned_decodes_earlier_than_logit_fraction" => r.tuned_beats_logit_earlier,
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
                    "logit/tuned-lens algorithm is unchanged; only the BASE + SOURCE " *
                    "state (both on the shared gameplay prefix) moves."),
            "note" =>
                "Logit / tuned lens (nostalgebraist 2020; Belrose et al. 2023) over the " *
                "VCS RAM trajectory: the residual stream = the 128-byte RAM tape, the " *
                "\"layers\" = the frames of the horizon. We decode each candidate " *
                "variable V's FINAL value from the INTERMEDIATE state at earlier step t. " *
                "LOGIT lens = the frozen/identity readout (read V's cell directly at " *
                "step t): its fidelity to the TRUE intermediate value is 1.0 by " *
                "construction (the readout IS the cell) — the positive control. TUNED " *
                "lens = a per-step affine readout A_t h_t + b_t fit by ridge " *
                "least-squares over a SET of real-ROM trajectories (held-out " *
                "train/test split). mean_best_tuned_lens_r2_heldout = the per-variable " *
                "PEAK (over steps) held-out R² (clipped at the 0 skill floor) averaged " *
                "over variables — the robust headline of decodability; the plain mean " *
                "(mean_tuned_lens_final_r2_heldout_raw) is reported too but is dominated " *
                "by a few catastrophic early-step extrapolations (a not-yet-determined " *
                "variable's high-dim ridge readout) so the MEDIAN is the robust per-step " *
                "summary. The VCS-specific " *
                "discriminator: at the earliest step the tuned lens decodes V, " *
                "faithful_weight_fraction = the share of |A_t| weight mass on sites " *
                "the EXACT oracle says are CAUSAL for V's final value vs non-causal " *
                "sites. High ⇒ the lens reads the true mechanism; low ⇒ it rides a " *
                "spurious correlate (the \"present != used\" tuning-curve trap, " *
                "quantified). The oracle causal mask is do(site:=site+17) at the " *
                "checkpoint + bit-exact re-run.",
            "scales_to_cluster_via" =>
                "tools/cluster/xai_*.sbatch (E0-3) + jaxtari SOFT-STE GPU batches " *
                "(forward bit-exact to this HARD path) for many-trajectory tuned-lens " *
                "fits over state × variables × games.",
        ),
    )
    open(json_path, "w") do io
        write(io, _j(rec) * "\n")
    end

    write_npz(npz_path, Dict(
        "logit_fidelity_true_intermediate" => r.logit_fid_true,   # (var, step)
        "logit_fidelity_final"             => r.logit_fid_final,  # (var, step)
        "tuned_fidelity_final_heldout"     => r.tuned_fid_final,  # (var, step)
        "earliest_logit_step"  => Float64.(r.earliest_logit),
        "earliest_tuned_step"  => Float64.(r.earliest_tuned),
        "faithful_weight_fraction" => r.faithful_weight_frac,
        "best_tuned_r2_per_var"    => r.best_tuned_per_var,       # per-var peak (clipped≥0)
        "causal_mask"          => Float64.(r.causal_mask),        # (var, site)
        "sites"                => Float64.(r.sites),
    ))
    return json_path, npz_path
end

function write_summary(results::Vector{LensResult}; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    path = joinpath(out_dir, "logit_lens_core_summary.json")
    per_game = Dict{String,Any}[]
    all_pos_control = true
    for r in results
        all_pos_control &= (r.logit_true_fidelity == 1.0)
        push!(per_game, Dict{String,Any}(
            "game" => r.game,
            "n_variables" => r.n_vars,
            "n_trajectories" => r.K,
            "logit_lens_fidelity_true_intermediate" => r.logit_true_fidelity,
            "mean_best_tuned_lens_r2_heldout" => (isfinite(r.mean_best_tuned_r2) ?
                                                  r.mean_best_tuned_r2 : nothing),
            "median_tuned_lens_final_r2_heldout" => (isfinite(r.median_tuned_final_r2) ?
                                                     r.median_tuned_final_r2 : nothing),
            "mean_faithful_weight_fraction" => (isfinite(r.mean_faithful_weight_frac) ?
                                                r.mean_faithful_weight_frac : nothing),
            "tuned_decodes_earlier_than_logit_fraction" => r.tuned_beats_logit_earlier,
        ))
    end
    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseC_mechanistic",
        "method" => "logit_tuned_lens", "scope" => "core (6 games)",
        "commit" => _git_commit(), "timestamp" => string(round(Int, time())),
        "all_games_logit_true_fidelity_unit" => all_pos_control,
        "mean_logit_lens_fidelity_true_intermediate" =>
            sum(r.logit_true_fidelity for r in results) / length(results),
        "mean_best_tuned_lens_r2_heldout" =>
            (let xs = filter(isfinite, [r.mean_best_tuned_r2 for r in results])
                 isempty(xs) ? nothing : sum(xs) / length(xs)
             end),
        "mean_median_tuned_lens_final_r2_heldout" =>
            (let xs = filter(isfinite, [r.median_tuned_final_r2 for r in results])
                 isempty(xs) ? nothing : sum(xs) / length(xs)
             end),
        "mean_faithful_weight_fraction" =>
            (let xs = filter(isfinite, [r.mean_faithful_weight_frac for r in results])
                 isempty(xs) ? nothing : sum(xs) / length(xs)
             end),
        "mean_tuned_decodes_earlier_fraction" =>
            sum(r.tuned_beats_logit_earlier for r in results) / length(results),
        "per_game" => per_game,
        "note" => "Phase-C logit/tuned lens over the 6 core games; readout fidelity " *
                  "vs the TRUE intermediate value (logit lens = positive control, " *
                  "fidelity 1.0) + the tuned-lens held-out final-value decodability and " *
                  "its faithful-weight fraction vs the exact oracle causal mask " *
                  "(faithful vs spuriously-predictive readout — experiment_design.md " *
                  "§6 row \"Logit / tuned lens\", the \"present != used\" gap).",
    )
    open(path, "w") do io
        write(io, _j(rec) * "\n")
    end
    return path
end

# ============================================================================
# Self-check (DoD: the small test) — positive control on Pong:
#   1. the bit-exact re-run holds (precondition for trusting every tape + the mask);
#   2. the LOGIT-lens fidelity to the TRUE intermediate value is exactly 1.0 — the
#      readout point is the true site (the cell IS its own intermediate value);
#   3. the TUNED lens decodes at least one variable's final value on held-out
#      trajectories (mean tuned final R² is finite and the best var R² is high) —
#      the learned readout is non-degenerate;
#   4. the oracle causal mask is non-trivial (at least one true causal edge) — the
#      faithful-vs-spurious decomposition has something to measure;
#   5. faithful_weight_fraction is in [0,1] wherever defined (a valid share).
# Exits nonzero (via `error`) on any failure.
# ============================================================================
function self_check(; game = "pong", target_frame = 120, horizon = 30, n_traj = 48)
    println("[self-check] running logit/tuned lens on $game (positive control)...")
    r = run_game(; game = game, target_frame = target_frame, horizon = horizon,
                 n_traj = n_traj, verbose = false)
    r.bit_exact || error("self-check FAIL: bit-exact re-run did not hold")
    r.logit_true_fidelity == 1.0 ||
        error("self-check FAIL: logit-lens fidelity to TRUE intermediate = " *
              "$(r.logit_true_fidelity) (expected 1.0 — the readout must BE the cell)")
    isfinite(r.mean_best_tuned_r2) ||
        error("self-check FAIL: mean peak tuned-lens R² is not finite (degenerate fit)")
    best_tuned = maximum(filter(isfinite, vec(r.tuned_fid_final)))
    best_tuned >= 0.5 ||
        error("self-check FAIL: best tuned-lens final R² = $(round(best_tuned,digits=3)) " *
              "(< 0.5 — the learned readout decodes nothing)")
    any(r.causal_mask) ||
        error("self-check FAIL: oracle causal mask is empty — no causal edge to score")
    for v in 1:r.n_vars
        if isfinite(r.faithful_weight_frac[v])
            (0.0 <= r.faithful_weight_frac[v] <= 1.0 + 1e-9) ||
                error("self-check FAIL: faithful_weight_fraction[$v]=" *
                      "$(r.faithful_weight_frac[v]) out of [0,1]")
        end
    end
    println("[self-check] PASS — bit-exact ✓, logit-lens fidelity(TRUE)=1.0 ✓, " *
            "best tuned final R²=$(round(best_tuned,digits=3)) ✓, " *
            "causal edges=$(count(r.causal_mask)) ✓, " *
            "mean faithful-weight frac=$(round(r.mean_faithful_weight_frac,digits=3)), " *
            "tuned earlier=$(round(r.tuned_beats_logit_earlier,digits=3))")
    return true
end

# ============================================================================
# CLI
# ============================================================================
function main(args = ARGS)
    games = CORE_GAMES
    target_frame = 120; horizon = 30; n_traj = 48
    do_self_check = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--games"
            v = args[i + 1]; i += 2
            games = lowercase(v) == "core" ? CORE_GAMES : String.(split(v, ","))
        elseif a == "--game"; games = [args[i + 1]]; i += 2
        elseif a == "--target-frame"; target_frame = parse(Int, args[i + 1]); i += 2
        elseif a == "--horizon"; horizon = parse(Int, args[i + 1]); i += 2
        elseif a == "--n-traj"; n_traj = parse(Int, args[i + 1]); i += 2
        elseif a == "--selftest" || a == "--self-check"; do_self_check = true; i += 1
        else; i += 1
        end
    end

    if do_self_check
        self_check(; target_frame = target_frame, horizon = horizon, n_traj = n_traj)
        return nothing
    end

    println("[logit_lens] games=$(join(games, ",")) " *
            "target_frame=$target_frame horizon=$horizon n_traj=$n_traj (jutari/Julia)")
    results = LensResult[]
    for g in games
        println("\n========== $g ==========")
        r = run_game(; game = g, target_frame = target_frame, horizon = horizon,
                     n_traj = n_traj, verbose = true)
        jp, np = write_game_result(r)
        println("[$g] logit-lens fidelity(TRUE)=$(round(r.logit_true_fidelity,digits=4)) " *
                "peak tuned R²=$(round(r.mean_best_tuned_r2,digits=3)) " *
                "faithful-weight=$(round(r.mean_faithful_weight_frac,digits=3)) " *
                "tuned-earlier=$(round(r.tuned_beats_logit_earlier,digits=3))")
        println("[$g] wrote $jp")
        println("[$g] arrays  $np")
        push!(results, r)
    end

    if length(results) > 1
        sp = write_summary(results)
        println("\n[logit_lens] core summary -> $sp")
    end

    println("\n[logit_lens] headline (logit/tuned lens — readout fidelity vs true intermediate):")
    for r in results
        println("    $(rpad(r.game, 16)) " *
                "logit_fid(TRUE)=$(round(r.logit_true_fidelity,digits=3)) " *
                "peak_tuned_R²=$(round(r.mean_best_tuned_r2,digits=3)) " *
                "median_tuned_R²=$(round(r.median_tuned_final_r2,digits=3)) " *
                "faithful_wt=$(round(r.mean_faithful_weight_frac,digits=3)) " *
                "tuned_earlier=$(round(r.tuned_beats_logit_earlier,digits=3)) " *
                "K=$(r.K) vars=$(r.n_vars)")
    end
    return results
end

end # module

# run as a script (not when `include`d by a test)
if abspath(PROGRAM_FILE) == @__FILE__
    LogitLens.main()
end
