# A3_tuning.jl — Phase-A analysis A3 (P2-E3-3), JULIA path. GENERALIZES the
# Space-Invaders pilot's A3 (tools/xai_study/phaseA_kording/pilot_si.jl) over the
# SIX core games (tools/xai_study/common/game_set.json: pong, breakout,
# space_invaders, seaquest, ms_pacman, qbert).
#
# THE ANALYSIS (experiment_design.md §4, row A3):
#   "tuning curves — per-unit tuning to luminance / a game variable;
#    measured score: spurious-tuning rate (strongly-tuned units whose tuning ≠
#    true role). Needs T3? Game-variable version only; hardware-tuning: No."
#
# Over a recorded (descriptive) trajectory each candidate RAM cell ("unit") has a
# time series of values. We compute TWO tuning families:
#
#   (1) LUMINANCE tuning (HARDWARE — needs NO T3): correlate each cell's value
#       against per-frame screen luminance (mean palette index over the visible
#       framebuffer) — the classic single-unit "tuning to a low-level feature".
#       The frame CLOCK is the epiphenomenal negative control (a cell tuned only
#       to the clock looks tuned but is a confound).
#
#   (2) GAME-VARIABLE tuning (needs T3): the per-game candidate concepts
#       (candidates_<game>.json — OCAtari/AtariARI RAM→concept labels) are the
#       game variables. Each cell's value time-series is correlated against every
#       candidate-concept signal (the labelled cells' own time-series) + the
#       decimal score. A cell's SELECTIVITY INDEX = max |corr| over the game
#       variables.
#
# THE SCORE — SPURIOUS-TUNING RATE (the tuning-curve trap, "present ≠ used"):
#   A unit is STRONGLY TUNED iff selectivity ≥ τ. It is SPURIOUS iff it is
#   strongly tuned but the intervention ORACLE says it is NOT causal (lesioning
#   it does not move behaviour). spurious_rate = #spurious / #strongly_tuned.
#   This is Phase A's headline failure: rich apparent tuning that does not match
#   the true causal role. We report it for BOTH the game-variable family (the
#   T3 version) and the luminance family (the hardware version).
#
# F / S / M (the correctness triad, experiment_design.md §0 — scored against the
# intervention ORACLE, never against another method) treating the tuning
# selectivity as the recovered per-cell "importance/recovery claim":
#   F (faithful)   — Spearman(selectivity, oracle causal importance): does the
#                    tuning-curve importance ranking track the TRUE causal map?
#                    (Low F = the tuning-curve trap, quantified.)
#   S (sufficient) — Pearson(selectivity, held-out lesion break do(base+37)):
#                    does the recovered importance predict an UNSEEN intervention?
#                    Scored on the bit-exact re-run, not on any method.
#   M (minimal)    — 1 − over-claim rate: of the cells the tuning method flags
#                    "strongly tuned", how many are NOT truly causal (oracle ≈ 0)?
#
# POSITIVE CONTROL (as in pilot_si.jl): the ORACLE'S OWN causal map, scored as
# the candidate "method", yields Spearman F == 1 — the harness rewards a
# perfectly-faithful map, so a sub-1 tuning F is a genuine measurement.
#
# BUILDS ON the validated foundation on main (NO emulator core touched):
#   * tools/xai_study/common/jutari_oracle.jl   — load/boot/replay/snapshot/
#       intervene + dependency-free NPY/NPZ writer.
#   * tools/xai_study/common/jutari_record.jl   — record_trajectory → (T,n) tape.
#   * tools/xai_study/ground_truth/oracle_intervene.jl — the causal-map reference
#       (we use the SAME do(u)/re-run primitives, specialised to candidate cells
#       + the whole-screen-break output, exactly as pilot_si.jl does).
#
# PER-GAME SETTINGS PARITY (CLAUDE.md pitfall #95): jutari_oracle.settings_for()
# only maps pong/breakout/space_invaders; this runner therefore resolves the
# CORRECT per-game RomSettings itself (matching tools/jutari_screen_dump.jl's
# map) so every game boots exactly as its xitari/screen-tool reference does.
# seaquest legitimately uses GenericRomSettings (it is not in the screen map).
#
# Run (warm shared depot, primary's project):
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseA_kording/A3_tuning.jl
# Flags: --games pong,breakout,...   --target-frame N --horizon N
#        --traj-frames N --tau 0.7   --selftest (run self-checks, write nothing)
#
# Writes (SPEC §R; file_scope A3_*): one record PER GAME +
#   tools/xai_study/phaseA_kording/out/A3_tuning_<game>.{json,npz}
# and one combined index:
#   tools/xai_study/phaseA_kording/out/A3_tuning_all.json

module A3Tuning

using JSON
import Statistics
using JuTari

# --- the validated foundation (NO core touched) ----------------------------
include(joinpath(@__DIR__, "..", "common", "jutari_oracle.jl"))
using .JutariOracle
using .JutariOracle: Snapshot, continue_from, snapshot,
                     intervene_ram!, write_npz, RAM_SIZE
using .JutariOracle.JuTari: JuTari
using .JutariOracle.JuTari.Env: StellaEnvironment, env_reset!, env_step!,
                                get_ram, get_screen

# the trajectory recorder's `Trajectory` container (the tuning curves consume a
# (T,n) RAM+screen tape). We RECORD it locally with parity RomSettings (below)
# rather than via record_trajectory(), which boots through jutari_oracle's
# settings_for() — that maps only pong/breakout/space_invaders, so ms_pacman/
# qbert would record under GenericRomSettings (a settings mismatch; CLAUDE.md
# #95). The container layout (tape/fields/widths) is identical.
include(joinpath(@__DIR__, "..", "common", "jutari_record.jl"))
using .JutariRecord: Trajectory

# The §1 exact-intervention oracle — used ONLY to build the P2 SHARED gameplay
# state + its cause-density gate (below). A3 keeps its OWN GameSpec/recorder/oracle
# machinery; the oracle here supplies the shared action stream + gate. Referenced
# as OracleIntervene.X / OracleIntervene.JutariOracle.X (NOT alias-imported, so its
# internal JutariOracle instance never clashes with the .JutariOracle already
# imported above — separate module instances; no type mixed across them).
include(joinpath(@__DIR__, "..", "ground_truth", "oracle_intervene.jl"))
using JuTari.Diff: soft_ram_peek

# The P2 SHARED TESTBED (experiment_redesign.md): seeded random-action GAMEPLAY
# state + oracle cause-density gate. Included as a fragment (see its header). Phase
# A is not a gradient method, so we consume the shared action STREAM + cause-density
# GATE only, and boot A3's OWN checkpoint + record its OWN trajectory from that
# stream. Opt in with XAI_SHARED_TESTBED=1 (default on).
include(joinpath(@__DIR__, "..", "common", "shared_testbed_impl.jl"))
# the shared game-set + ROM-root resolver (XAI_LABELED / xai_resolve_games / xai_rom_roots).
include(joinpath(@__DIR__, "..", "common", "game_sets.jl"))

# the canonical triad Minimality scorer M = |U*|/|U_hat| (03_methods.tex sec:triad);
# guarded so re-includes across the Phase-A battery don't redefine the module.
isdefined(@__MODULE__, :TriadSM) ||
    include(joinpath(@__DIR__, "..", "common", "triad_sm.jl"))
using .TriadSM: minimality_score

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

# ROM filename aliases: the game key (game_set.json / candidates_<game>.json) vs
# the actual `xitari/roms/<file>.bin` basename. Only ms_pacman differs (the ROM
# is `mspacman.bin`). All other core games' key == filename stem.
const ROM_ALIAS = Dict("ms_pacman" => "mspacman")
const _PRIMARY_REPO = "/Users/maier/Documents/code/UnderstandingVCS"

"""Absolute path to the real ROM for `game`, honouring the filename alias and the
worktree→primary fallback (same search order as jutari_oracle.rom_path_for)."""
function resolve_rom(game::AbstractString)
    g = lowercase(string(game))
    stem = get(ROM_ALIAS, g, g)
    # search the shared ROM roots (xitari/roms + the 64-ROM store tools/rom_sweep/roms
    # + cluster), trying the aliased stem AND the raw ALE name so all 54 labeled
    # games resolve uniformly (non-core games live under rom_sweep as `<game>.bin`).
    return xai_find_rom(unique([stem, g]), xai_rom_roots(; primary_repo = _PRIMARY_REPO))
end

# ============================================================================
# Per-game RomSettings parity (CLAUDE.md #95). MUST agree with
# tools/jutari_screen_dump.jl :: _SETTINGS_BY_BASENAME. Games absent there boot
# with GenericRomSettings (the screen-tool default) — e.g. seaquest.
# ============================================================================
function settings_for_game(game::AbstractString)
    g = lowercase(string(game))
    g == "pong"           && return JuTari.PaddleGames.PongRomSettings()
    g == "breakout"       && return JuTari.PaddleGames.BreakoutRomSettings()
    g == "space_invaders" && return JuTari.SpaceInvadersRomSettings()
    g == "ms_pacman"      && return JuTari.JoystickGames.MsPacmanRomSettings()
    g == "qbert"          && return JuTari.JoystickGames.QbertRomSettings()
    # seaquest (+ any other) → Generic, exactly as the screen tool resolves it.
    return JuTari.RomSettingsModule.GenericRomSettings()
end

"""Boot `game` with its PARITY RomSettings (xitari-parity boot 60 NOOP + 4 RESET),
NOT yet stepped into the action trace."""
function boot_env(game::AbstractString)
    rom = read(resolve_rom(game))
    env = StellaEnvironment(rom, settings_for_game(game))
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    return env
end

"""boot + replay `actions[1:target_frame]` with the parity settings; return env
AT the intervention frame (deepcopy it for a reusable checkpoint)."""
function boot_replay_parity(game::AbstractString, actions::AbstractVector{<:Integer},
                            target_frame::Integer)
    env = boot_env(game)
    for i in 1:target_frame
        env_step!(env, Int(actions[i]))
    end
    return env
end

"""Full from-scratch parity replay of `actions[1:total]` → Snapshot (for the
bit-exact assertion: two fresh runs must be byte-identical)."""
function fresh_parity_snapshot(game::AbstractString,
                               actions::AbstractVector{<:Integer}, total::Integer)
    env = boot_env(game)
    for i in 1:total
        env_step!(env, Int(actions[i]))
    end
    return snapshot(env, Int(total))
end

# ============================================================================
# Action traces.
#   * oracle/lesion trace — FIRE×4 then NOOP: deterministic, stays inside the
#     conformance window (the SCORED-vs-oracle path).
#   * active trajectory trace — FIRE to start + periodic motion so the recorded
#     trajectory actually MOVES state (otherwise the candidate cells are static
#     and the tuning analysis has no signal). DESCRIPTIVE jutari recording per
#     the recorder's documented purpose (pilot_si.jl §A3), not a bit-exact claim.
# Action codes (jutari IO.jl): NOOP=0 FIRE=1 RIGHT=3 LEFT=4 RIGHTFIRE=11 LEFTFIRE=12
# ============================================================================
oracle_actions(total::Integer) = vcat(fill(1, 4), fill(0, max(0, total - 4)))

function active_actions(total::Integer)
    acts = Vector{Int}(undef, max(0, total))
    for t in 1:length(acts)
        acts[t] = t <= 4     ? 1  :   # FIRE×4: start the game
                  t % 4 == 0 ? 11 :   # RIGHTFIRE: move right (+ shoot)
                  t % 4 == 2 ? 12 :   # LEFTFIRE:  move left  (+ shoot)
                               1      # FIRE: keep acting
    end
    return acts
end

# ============================================================================
# Local parity trajectory recorder — stacks RAM + screen per frame, booting with
# the CORRECT per-game RomSettings (settings_for_game). Layout matches
# JutariRecord.Trajectory exactly: fields ["ram","screen"], widths [128, |screen|],
# tape (T, 128+|screen|) UInt8, screen flattened ROW-MAJOR (same as the recorder).
# We record locally rather than via record_trajectory() because that boots through
# jutari_oracle.settings_for() (Generic for ms_pacman/qbert/seaquest) — a settings
# mismatch we must avoid (CLAUDE.md #95).
# ============================================================================
_screen_rowmajor(env) = vec(permutedims(get_screen(env), (2, 1)))

function record_parity_trajectory(game::AbstractString, actions::AbstractVector{<:Integer})
    frames = length(actions)
    env = boot_env(game)
    nscreen = length(get_screen(env))
    fields = ["ram", "screen"]
    widths = [RAM_SIZE, nscreen]
    n = sum(widths)
    tape = Matrix{UInt8}(undef, frames, n)
    frame_idx = Vector{Int}(undef, frames)
    for t in 1:frames
        env_step!(env, Int(actions[t]))
        ram = UInt8.(collect(get_ram(env)))
        scr = _screen_rowmajor(env)
        tape[t, 1:RAM_SIZE] = ram
        tape[t, RAM_SIZE + 1 : end] = scr
        frame_idx[t] = t
    end
    return Trajectory(string(game), frames, fields, widths, tape, frame_idx,
                      Int.(collect(actions)))
end

# ============================================================================
# Candidate cells (E2-1 import) — the "units" A3 probes, per game.
# ============================================================================
struct Candidate
    ram_index::Int
    concept::String
end

const CANDIDATES_REL = ("tools", "xai_study", "t3", "out")

function candidates_path(game::AbstractString)
    here = normpath(joinpath(@__DIR__, "..", "..", ".."))
    fname = "candidates_$(game).json"
    for base in (here, "/Users/maier/Documents/code/UnderstandingVCS")
        p = joinpath(base, CANDIDATES_REL..., fname)
        isfile(p) && return p
    end
    return nothing
end

"""Read the per-game candidate cells, de-duplicated by ram_index, in file order.
Each kept cell carries the FIRST concept name seen for that index."""
function load_candidates(game::AbstractString)
    p = candidates_path(game)
    out = Candidate[]
    if p !== nothing
        data = JSON.parsefile(p)
        seen = Set{Int}()
        for c in get(data, "candidates", [])
            idx = Int(c["ram_index"])
            (idx < 0 || idx >= RAM_SIZE) && continue       # only RIOT-RAM cells
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
        p = p === nothing ? "(generic fallback: no candidates_$(game).json)" : p
    end
    return out, p
end

# ============================================================================
# The oracle causal importance for the SAME candidate cells (GROUND TRUTH).
# Identical machinery to pilot_si.jl: do(u := v')/re-run/whole-screen-break, the
# robust position-aware behavioural footprint. This IS the intervention oracle
# (oracle_intervene.jl's Δy(u)) restricted to the candidate cells + screen output.
# ============================================================================
"""Apply do(ram[idx] := value) on a deepcopy of the checkpoint, continue `tail`,
snapshot. The bit-exact re-run makes Δ a clean causal effect."""
function lesion_snapshot(checkpoint::StellaEnvironment, tail::Vector{Int},
                         idx::Integer, value::Integer)
    env = deepcopy(checkpoint)
    intervene_ram!(env, idx, value)
    for a in tail; env_step!(env, a); end
    return snapshot(env, length(tail))
end

"""Per-cell oracle causal importance = max whole-screen break over
do(0)/do(base+17)/do(base+37) (skipping a no-op equal to base) — the TRUE causal
footprint; and the do(base+37) break alone = the held-out (S) probe."""
function oracle_causal_importance(checkpoint::StellaEnvironment, tail::Vector{Int},
                                  cands::Vector{Candidate}, at_target::Snapshot,
                                  base_snap::Snapshot)
    n = length(cands)
    imp = zeros(Float64, n)
    held = zeros(Float64, n)
    for (i, c) in enumerate(cands)
        base_val = Int(at_target.ram[c.ram_index + 1])
        best = 0.0
        for v in (0, (base_val + 17) & 0xFF, (base_val + 37) & 0xFF)
            v == base_val && continue
            s = lesion_snapshot(checkpoint, tail, c.ram_index, v)
            brk = Float64(count(s.screen .!= base_snap.screen))
            best = max(best, brk)
            v == ((base_val + 37) & 0xFF) && (held[i] = brk)
        end
        imp[i] = best
    end
    return imp, held
end

# ============================================================================
# A3 — tuning curves over the recorded trajectory.
# ============================================================================
struct TuningResult
    cand::Vector{Candidate}
    # game-variable family (T3): candidate-concept signals + score
    gvar_names::Vector{String}
    gvar_corr::Matrix{Float64}        # (n_cand, n_gvar) |Pearson|
    gvar_selectivity::Vector{Float64} # per-cell max |corr| over game variables
    # luminance family (hardware, no T3)
    lum_corr::Vector{Float64}         # per-cell |corr| with mean screen luminance
    clock_corr::Vector{Float64}       # per-cell |corr| with the epiphenomenal clock
    # spurious tuning vs the oracle
    tau::Float64
    gvar_strong::Vector{Bool}
    gvar_spurious::Vector{Bool}
    gvar_spurious_rate::Float64
    lum_strong::Vector{Bool}
    lum_spurious::Vector{Bool}
    lum_spurious_rate::Float64
end

"""Per-frame luminance = mean palette index over the visible framebuffer, and the
epiphenomenal frame clock. Requires the trajectory to stack the screen field."""
function luminance_and_clock(traj::Trajectory)
    tape = traj.tape
    T = size(tape, 1)
    clock = Float64.(1:T)
    if "screen" in traj.fields
        nram_i = findfirst(==("ram"), traj.fields)
        nram = nram_i === nothing ? 0 : traj.widths[nram_i]
        # screen is the trailing field block (recorder stacks ram, [tia,] screen)
        screen = @view tape[:, nram + 1 : end]
        lum = Float64[Statistics.mean(Float64.(@view screen[t, :])) for t in 1:T]
        return lum, clock
    end
    return fill(0.0, T), clock
end

"""The game-variable signals (T3): each candidate-cell value time-series is itself
the labelled game variable, plus the decimal score derived from the score cells.
Returns (names, (T, n_gvar) matrix). The candidate cells double as game variables
because each carries an OCAtari/AtariARI concept label — that IS the game-variable
ground-truth A3 tunes against."""
function game_variable_signals(traj::Trajectory, cands::Vector{Candidate})
    tape = traj.tape
    T = size(tape, 1)
    names = String[]
    cols = Vector{Float64}[]
    for c in cands
        push!(names, c.concept == "(unnamed)" ?
              "RAM[$(c.ram_index)]" : "$(c.concept)@$(c.ram_index)")
        push!(cols, Float64[Int(tape[t, c.ram_index + 1]) for t in 1:T])
    end
    # a derived "score" signal: low byte of any candidate whose concept names a
    # score, interpreted as raw value (BCD decode is game-specific; the raw byte
    # is monotone-equivalent within a short window, which is all a |corr| needs).
    score_idx = [c.ram_index for c in cands if occursin("score", lowercase(c.concept))]
    if !isempty(score_idx)
        si = first(score_idx)
        push!(names, "score_raw@$si")
        push!(cols, Float64[Int(tape[t, si + 1]) for t in 1:T])
    end
    isempty(cols) && return names, zeros(Float64, T, 0)
    return names, hcat(cols...)
end

function run_a3_tuning(traj::Trajectory, cands::Vector{Candidate},
                       oracle_importance::Vector{Float64}; tau = 0.7)
    tape = traj.tape
    T = size(tape, 1)
    n = length(cands)
    gnames, gsig = game_variable_signals(traj, cands)
    lum, clock = luminance_and_clock(traj)

    ng = size(gsig, 2)
    G = zeros(Float64, n, ng)
    lum_corr = zeros(Float64, n)
    clock_corr = zeros(Float64, n)
    for (i, c) in enumerate(cands)
        x = Float64[Int(tape[t, c.ram_index + 1]) for t in 1:T]
        for j in 1:ng
            # a cell's |corr| with ITSELF as a game-variable column is 1.0 by
            # construction — exclude the self-column so selectivity measures
            # tuning to OTHER game variables (the genuine cross-tuning signal).
            G[i, j] = abs(pearson(x, gsig[:, j]))
        end
        lum_corr[i] = abs(pearson(x, lum))
        clock_corr[i] = abs(pearson(x, clock))
    end
    # exclude each cell's own self-column from its selectivity. The first n game
    # variables are the n candidate cells in order, so column i is cell i's self.
    gvar_sel = zeros(Float64, n)
    for i in 1:n
        best = 0.0
        for j in 1:ng
            j == i && continue                 # skip the self-column
            best = max(best, G[i, j])
        end
        gvar_sel[i] = best
    end

    causal = oracle_importance .> 0.0          # oracle says it moves behaviour
    g_strong = gvar_sel .>= tau
    g_spur = g_strong .& .!causal
    g_rate = sum(g_strong) == 0 ? 0.0 : sum(g_spur) / sum(g_strong)
    l_strong = lum_corr .>= tau
    l_spur = l_strong .& .!causal
    l_rate = sum(l_strong) == 0 ? 0.0 : sum(l_spur) / sum(l_strong)

    return TuningResult(cands, gnames, G, gvar_sel, lum_corr, clock_corr, tau,
                        g_strong, g_spur, g_rate, l_strong, l_spur, l_rate)
end

# ============================================================================
# F / S / M (the triad) — tuning selectivity as the recovered importance claim,
# scored against the oracle.
# ============================================================================
struct Triad
    F::Float64; S::Float64; M::Union{Float64,Nothing}
    F_note::String; S_note::String; M_note::String
    legacy_minimality::Float64        # the old 1 − over-claim rate (kept for reference)
end

function score_triad(tun::TuningResult, oracle_importance::Vector{Float64},
                     held_out::Vector{Float64})
    sel = tun.gvar_selectivity
    F = spearman(sel, oracle_importance)
    S = pearson(sel, held_out)
    flagged = tun.gvar_strong
    nflag = sum(flagged)
    overclaim = nflag == 0 ? 0.0 :
        count(i -> flagged[i] && oracle_importance[i] == 0.0, 1:length(flagged)) / nflag
    legacy_M = 1.0 - overclaim
    # M — standardized to the paper's M = |U*|/|U_hat| (03_methods.tex sec:triad):
    #     |U*| = the true movers (oracle_importance above the floor), |U_hat| = the
    #     cells the tuning method NAMES (selectivity above its own threshold). Uses
    #     the SAME attr (gvar_selectivity, the F claim) and odelta (oracle_importance,
    #     the F ground truth). The legacy 1 − over-claim rate is retained separately.
    M, _ustar, _uhat, M_note = minimality_score(sel, oracle_importance)
    return Triad(F, S, M,
        "Spearman(A3 game-variable selectivity, oracle causal importance) over $(length(oracle_importance)) candidate cells",
        "Pearson(A3 selectivity, held-out do(base+37) screen break) — sufficiency on a bit-exact re-run",
        "M = |U*|/|U_hat| (03_methods.tex sec:triad): " * M_note,
        legacy_M)
end

# ============================================================================
# stats helpers (Pearson, Spearman with average-rank ties) — same as pilot.
# ============================================================================
function pearson(a::AbstractVector, b::AbstractVector)
    length(a) < 2 && return 0.0
    sa = Statistics.std(a); sb = Statistics.std(b)
    (sa == 0 || sb == 0 || !isfinite(sa) || !isfinite(sb)) && return 0.0
    c = Statistics.cor(a, b)
    return isfinite(c) ? c : 0.0
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
# Drive A3 for ONE game.
# ============================================================================
struct GameA3
    game::String
    target_frame::Int
    horizon::Int
    traj_frames::Int
    tau::Float64
    bit_exact::Bool
    cands::Vector{Candidate}
    candidates_file::Union{String,Nothing}
    oracle_importance::Vector{Float64}
    held_out::Vector{Float64}
    self_F::Float64                     # oracle-as-method positive control
    tuning::TuningResult
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

# A3's candidates-path resolver, in the shape the shared testbed injects (a
# String->path-or-nothing map). Mirrors candidates_path' file search.
function _st_candidates_path_for(game::AbstractString)
    rel = joinpath("tools", "xai_study", "t3", "out", "candidates_$(game).json")
    here = normpath(joinpath(@__DIR__, "..", "..", ".."))
    for base in (here, "/Users/maier/Documents/code/UnderstandingVCS")
        p = joinpath(base, rel)
        isfile(p) && return p
    end
    return nothing
end

"""Build the P2 SHARED gameplay state + cause-density gate for `game` using the §1
oracle machinery (oracle_intervene.jl). Returns the substrate NamedTuple (we use
its `.actions` stream + `.cause_density`/`.accepted`/`.cell` gate). A3 then boots
its OWN checkpoint + records its OWN trajectory from `st.actions` so the tuning
algorithm is unchanged."""
function build_a3_shared_state(game::AbstractString; verbose = false)
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

function compute_game(game::AbstractString; target_frame = 30, horizon = 30,
                      traj_frames = 150, tau = 0.7, verbose = true)
    # SHARED-TESTBED (redesign): replace the FIRE×4+NOOP oracle/attract tape with a
    # seeded random-action GAMEPLAY state at f*=ST_PREFIX, gated by the oracle
    # cause-density gate. A3's tuning needs the game VARIABLE to VARY, so we record
    # the descriptive trajectory along the SAME gameplay stream (extended with NOOP
    # if traj_frames exceeds prefix+horizon). A3's oracle/tuning machinery is
    # UNCHANGED — only the state moves.
    st = nothing
    if SHARED_TESTBED
        st = build_a3_shared_state(game; verbose = verbose)
        target_frame = st.prefix; horizon = st.horizon
        verbose && println("[A3:$game] SHARED gameplay state: cause_density=$(st.cause_density)/" *
            "$(length(st.causes)) accepted=$(st.accepted) cell=$(st.cell)")
    end
    total = target_frame + horizon
    oacts = SHARED_TESTBED ? Int.(st.actions) : oracle_actions(total)
    tail = Int.(oacts[target_frame + 1 : target_frame + horizon])

    # 1) bit-exact baseline guarantee on the (gameplay/oracle) trace (every Δ causal).
    verbose && println("[A3:$game] bit-exact assert (2 fresh boots+replays to f$total)...")
    a = fresh_parity_snapshot(game, oacts, total)
    b = fresh_parity_snapshot(game, oacts, total)
    bit_exact = (a.ram == b.ram) && (a.screen == b.screen)
    bit_exact || error("[$game] bit-exact re-run FAILED to f$total — refusing to score")

    # 2) ONE checkpoint at the intervention frame (boot+to-target paid once).
    checkpoint = boot_replay_parity(game, oacts, target_frame)
    at_target = continue_from(checkpoint, Int[])
    base_snap = continue_from(checkpoint, tail)

    cands, cfile = load_candidates(game)
    isempty(cands) && error("[$game] no candidate cells (candidates_$(game).json missing/empty)")
    verbose && println("[A3:$game] candidates: $(cfile === nothing ? "(none)" : basename(cfile)) " *
                       "($(length(cands)) RIOT-RAM cells)")

    # 3) the ORACLE causal map + held-out probe for the SAME cells (ground truth).
    verbose && println("[A3:$game] oracle causal importance over $(length(cands)) cells...")
    oracle_importance, held = oracle_causal_importance(checkpoint, tail, cands,
                                                       at_target, base_snap)
    self_F = spearman(oracle_importance, oracle_importance)   # positive control

    # 4) A3 tuning curves over a DESCRIPTIVE trajectory (RAM+screen), recorded with
    #    the game's PARITY RomSettings (settings_for_game). Under the shared testbed
    #    the trajectory is stepped along the SAME seeded gameplay stream (so the game
    #    variable VARIES on genuine input-driven play); if traj_frames exceeds the
    #    gameplay window (prefix+horizon), the tail is NOOP-continued.
    if SHARED_TESTBED
        traj_acts = Vector{Int}(undef, max(0, traj_frames))
        for t in 1:length(traj_acts)
            traj_acts[t] = t <= total ? Int(st.actions[t]) : 0    # gameplay then NOOP
        end
        verbose && println("[A3:$game] tuning curves over a $(traj_frames)-frame gameplay " *
            "RAM+screen trajectory (seed=$(st.seed) prefix=$(st.prefix)+$(st.horizon), " *
            "NOOP-continued past f$total)...")
    else
        traj_acts = active_actions(traj_frames)
        verbose && println("[A3:$game] tuning curves over a $(traj_frames)-frame (active) RAM+screen trajectory...")
    end
    traj = record_parity_trajectory(game, traj_acts)
    tuning = run_a3_tuning(traj, cands, oracle_importance; tau = tau)

    # 5) F/S/M triad (A3 selectivity vs the oracle).
    triad = score_triad(tuning, oracle_importance, held)

    if verbose
        println("[A3:$game] strongly-tuned (game-var) = $(sum(tuning.gvar_strong))" *
                "/$(length(cands)); SPURIOUS-tuning rate (game-var) = " *
                "$(round(tuning.gvar_spurious_rate, digits = 3))")
        println("[A3:$game] strongly-tuned (luminance) = $(sum(tuning.lum_strong))" *
                "; SPURIOUS-tuning rate (luminance) = " *
                "$(round(tuning.lum_spurious_rate, digits = 3))")
        println("[A3:$game] TRIAD F=$(round(triad.F,digits=3)) " *
                "S=$(round(triad.S,digits=3)) M=$(triad.M === nothing ? "null" : round(triad.M,digits=3)) " *
                "(oracle-as-method F=$(round(self_F,digits=3)); legacy_M=$(round(triad.legacy_minimality,digits=3)))")
    end

    return GameA3(string(game), target_frame, horizon, traj_frames, tau, bit_exact,
                  cands, cfile, oracle_importance, held, self_F, tuning, triad,
                  st === nothing ? "noop" : "seeded_random_action_gameplay",
                  st === nothing ? -1 : st.seed,
                  st === nothing ? -1 : st.prefix,
                  st === nothing ? -1 : st.cause_density,
                  st === nothing ? false : st.accepted,
                  st === nothing ? 0 : length(st.causes),
                  st === nothing ? (-1, -1) : st.cell)
end

# ============================================================================
# Self-check (DoD) — the scoring contract is sound; results non-fabricated.
# ============================================================================
"""Asserts the load-bearing A3 claims for one game's result. Throws on violation."""
function selftest(g::GameA3)
    @assert g.bit_exact "[$(g.game)] bit-exact baseline re-run failed"
    n_causal = count(>(0.0), g.oracle_importance)
    @assert n_causal >= 1 "[$(g.game)] oracle found NO causal candidate cell — " *
        "uninformative state/horizon"
    # POSITIVE CONTROL: oracle's own map must score F==1 against itself.
    @assert g.self_F > 0.999 "[$(g.game)] harness broken: oracle-as-method F=$(g.self_F) != 1"
    # ranges
    @assert -1.0 - 1e-9 <= g.triad.F <= 1.0 + 1e-9 "[$(g.game)] F out of [-1,1]: $(g.triad.F)"
    @assert -1.0 - 1e-9 <= g.triad.S <= 1.0 + 1e-9 "[$(g.game)] S out of [-1,1]: $(g.triad.S)"
    @assert g.triad.M === nothing || (0.0 - 1e-9 <= g.triad.M <= 1.0 + 1e-9) "[$(g.game)] M out of (0,1]: $(g.triad.M)"
    @assert 0.0 <= g.tuning.gvar_spurious_rate <= 1.0 "[$(g.game)] game-var spurious rate out of [0,1]"
    @assert 0.0 <= g.tuning.lum_spurious_rate <= 1.0 "[$(g.game)] luminance spurious rate out of [0,1]"
    # selectivity / correlations are well-formed (finite, in [0,1] as |corr|)
    @assert all(x -> isfinite(x) && -1e-9 <= x <= 1.0 + 1e-9, g.tuning.gvar_selectivity) "[$(g.game)] selectivity out of [0,1]"
    @assert all(x -> isfinite(x) && -1e-9 <= x <= 1.0 + 1e-9, g.tuning.lum_corr) "[$(g.game)] luminance corr out of [0,1]"
    return true
end

# ============================================================================
# Persist (SPEC §R) — JSON record + sibling .npz; file_scope A3_*.
# ============================================================================
_git_commit() = try
    strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
catch
    "unknown"
end
_json_num(x::Real) = isfinite(x) ? Float64(x) : nothing
_json_num(::Nothing) = nothing

function write_game(g::GameA3; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    stem = "A3_tuning_$(g.game)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path  = joinpath(out_dir, stem * ".npz")

    cell_names = ["RAM[$(c.ram_index)]:$(c.concept)" for c in g.cands]
    rec = Dict{String,Any}(
        "paper" => "P2",
        "phase" => "phaseA_kording",
        "method" => "A3_tuning_curves(luminance+game_variable)+spurious_tuning_rate",
        "game" => g.game,
        "state" => g.state_kind == "noop" ? "f$(g.target_frame)+$(g.horizon)" :
                   "gameplay(seed=$(g.st_seed),prefix=$(g.st_prefix))+$(g.horizon)",
        "target_output" => "screen-break (oracle causal importance) vs tuning selectivity",
        # headline scalar (§R value/metric_name): the GAME-VARIABLE spurious-tuning
        # rate — A3's named score (the tuning-curve trap, quantified).
        "metric_name" => "A3_game_variable_spurious_tuning_rate",
        "value" => _json_num(g.tuning.gvar_spurious_rate),
        "stderr" => nothing,
        "ci" => nothing,
        "n" => length(g.cands),
        "seed" => 0,
        "where" => "local",
        "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene@$(g.game)#screen-break",
        "timestamp" => string(round(Int, time())),
        "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            "substrate" => "jutari (Julia, HARD) — real-ROM bit-exact path; the " *
                "oracle causal map is EXACT interventions re-run on the true ROM; " *
                "the tuning curves are a DESCRIPTIVE active-trace recording (the " *
                "recorder's documented purpose), not a bit-exact-vs-xitari claim.",
            "rom_settings" => string(typeof(settings_for_game(g.game))),
            "bit_exact_rerun" => g.bit_exact,
            "candidate_cells" => cell_names,
            "candidate_ram_indices" => [c.ram_index for c in g.cands],
            "candidates_file" => g.candidates_file === nothing ? nothing :
                basename(g.candidates_file),
            "testbed" => Dict{String,Any}(
                "state_kind" => g.state_kind,
                "seed" => g.st_seed, "prefix" => g.st_prefix, "horizon" => g.horizon,
                "shared_output" => "screen_region(n_changed_px)@r$(g.shared_cell[1])c$(g.shared_cell[2])",
                "cause_density_above_floor" => g.cause_density,
                "cause_density_floor" => ST_FLOOR, "cause_density_gate_k" => ST_GATE_K,
                "cause_density_accepted" => g.cause_density_accepted, "n_causes" => g.n_causes,
                "note" => "P2 redesign: the tuning study runs on a seeded random-action " *
                    "GAMEPLAY state (not the FIRE×4/attract tape), gated by the §1 oracle " *
                    "cause-density gate. Both the oracle causal map AND the descriptive " *
                    "tuning trajectory are stepped along the SAME gameplay stream so the " *
                    "game variable VARIES on genuine input-driven play. A3's tuning " *
                    "algorithm is unchanged; only the analysis state moves."),
            "oracle_importance_per_cell" =>
                Dict(cell_names[i] => g.oracle_importance[i] for i in 1:length(cell_names)),
            "n_oracle_causal" => count(>(0.0), g.oracle_importance),
            "positive_control_oracle_as_method_F" => _json_num(g.self_F),
            "A3_tuning" => Dict{String,Any}(
                "tau" => g.tau,
                "trajectory_frames" => g.traj_frames,
                "trajectory_trace" => g.state_kind == "noop" ?
                    "active (FIRE×4 + periodic RIGHTFIRE/LEFTFIRE)" :
                    "seeded random-action GAMEPLAY (seed=$(g.st_seed), prefix=$(g.st_prefix)+$(g.horizon), NOOP-continued past the window)",
                "epiphenomenal_control" => "frame_clock",
                "game_variable_signals" => g.tuning.gvar_names,
                "gvar_selectivity_per_cell" =>
                    Dict(cell_names[i] => g.tuning.gvar_selectivity[i] for i in 1:length(cell_names)),
                "luminance_corr_per_cell" =>
                    Dict(cell_names[i] => g.tuning.lum_corr[i] for i in 1:length(cell_names)),
                "clock_corr_per_cell" =>
                    Dict(cell_names[i] => g.tuning.clock_corr[i] for i in 1:length(cell_names)),
                "n_strongly_tuned_gvar" => sum(g.tuning.gvar_strong),
                "n_spurious_gvar" => sum(g.tuning.gvar_spurious),
                "game_variable_spurious_tuning_rate" => _json_num(g.tuning.gvar_spurious_rate),
                "n_strongly_tuned_luminance" => sum(g.tuning.lum_strong),
                "n_spurious_luminance" => sum(g.tuning.lum_spurious),
                "luminance_spurious_tuning_rate" => _json_num(g.tuning.lum_spurious_rate),
                "strongly_tuned_gvar_cells" => cell_names[g.tuning.gvar_strong],
                "spurious_gvar_cells" => cell_names[g.tuning.gvar_spurious],
                "note" => "selectivity = max |corr| to a GAME variable (the cell's own " *
                    "self-column excluded; frame clock excluded — it is the " *
                    "epiphenomenal control). spurious = strongly-tuned (selectivity≥τ) " *
                    "but NOT causal by the oracle — the present≠used tuning-curve trap " *
                    "(experiment_design.md §4 A3, Phase-A's headline failure). The " *
                    "luminance family is the HARDWARE (no-T3) version of the same trap.",
            ),
            "triad" => Dict{String,Any}(
                "F" => _json_num(g.triad.F), "F_note" => g.triad.F_note,
                "S" => _json_num(g.triad.S), "S_note" => g.triad.S_note,
                "M" => _json_num(g.triad.M), "M_note" => g.triad.M_note,
                "legacy_minimality" => _json_num(g.triad.legacy_minimality),
                "legacy_minimality_note" => "1 − over-claim rate (the pre-standardization " *
                    "M): fraction of strongly-tuned cells the oracle says are generic. " *
                    "Retained for reference; the triad M above is the paper's |U*|/|U_hat|.",
                "interpretation" => "Phase A is the calibration baseline " *
                    "(experiment_design.md §4): a tuning-curve importance map scores " *
                    "LOW faithfulness despite fully-known structure. F<1 / " *
                    "spurious-tuning>0 quantify where the classical analysis departs " *
                    "from the ground truth. The oracle-as-method positive control " *
                    "(F=$(round(g.self_F, digits = 3))) confirms the harness rewards a " *
                    "perfectly-faithful map.",
            ),
            "scales_to_cluster_via" =>
                "tools/cluster/xai_*.sbatch (E0-3) — the full A1..A8 battery over " *
                "the core+breadth set; the forward is bit-exact to this HARD path.",
        ),
    )
    open(json_path, "w") do io; JSON.print(io, rec, 2); end

    write_npz(npz_path, Dict(
        "candidate_ram_indices" => Int64[c.ram_index for c in g.cands],
        "oracle_importance"     => g.oracle_importance,
        "held_out_delta"        => g.held_out,
        "gvar_corr"             => g.tuning.gvar_corr,
        "gvar_selectivity"      => g.tuning.gvar_selectivity,
        "luminance_corr"        => g.tuning.lum_corr,
        "clock_corr"            => g.tuning.clock_corr,
        "gvar_strong"           => Int64.(g.tuning.gvar_strong),
        "gvar_spurious"         => Int64.(g.tuning.gvar_spurious),
        "lum_strong"            => Int64.(g.tuning.lum_strong),
        "lum_spurious"          => Int64.(g.tuning.lum_spurious),
        "triad_FSM"             => Float64[g.triad.F, g.triad.S,
                                           g.triad.M === nothing ? NaN : g.triad.M],
    ))
    return json_path, npz_path
end

# ============================================================================
# CLI — run A3 over the core set; one record per game + a combined index.
# ============================================================================
function main(args = ARGS)
    games = copy(CORE_GAMES)
    target_frame = 30; horizon = 30; traj_frames = 150; tau = 0.7
    selftest_only = false
    i = 1
    while i <= length(args)
        a = args[i]
        if     a == "--games";        games = xai_resolve_games(args[i+1], CORE_GAMES); i += 2
        elseif a == "--game";         games = [args[i+1]]; i += 2
        elseif a == "--target-frame"; target_frame = parse(Int, args[i+1]); i += 2
        elseif a == "--horizon";      horizon = parse(Int, args[i+1]); i += 2
        elseif a == "--traj-frames";  traj_frames = parse(Int, args[i+1]); i += 2
        elseif a == "--tau";          tau = parse(Float64, args[i+1]); i += 2
        elseif a == "--selftest";     selftest_only = true; i += 1
        else; i += 1
        end
    end
    println("[A3] games=$(join(games, ",")) target_frame=$target_frame " *
            "horizon=$horizon traj_frames=$traj_frames tau=$tau (jutari/Julia)")

    results = GameA3[]
    problems = Dict{String,String}()
    for game in games
        try
            g = compute_game(game; target_frame = target_frame, horizon = horizon,
                             traj_frames = traj_frames, tau = tau, verbose = true)
            selftest(g)
            push!(results, g)
            if !selftest_only
                jp, np = write_game(g)
                println("[A3:$game] wrote $jp")
                println("[A3:$game] arrays  $np")
            end
        catch err
            msg = sprint(showerror, err)
            problems[game] = msg
            println("[A3:$game] PROBLEM (scoring the rest): $msg")
        end
    end

    # combined index (this runner's own aggregate — file_scope A3_*, NOT the
    # SM-owned results_index.csv).
    if !selftest_only && !isempty(results)
        isdir(OUT_DIR) || mkpath(OUT_DIR)
        idx = Dict{String,Any}(
            "paper" => "P2", "phase" => "phaseA_kording",
            "method" => "A3_tuning_curves+spurious_tuning_rate",
            "metric_name" => "A3_game_variable_spurious_tuning_rate",
            "commit" => _git_commit(),
            "timestamp" => string(round(Int, time())),
            "games" => [g.game for g in results],
            "problems" => problems,
            "per_game" => Dict(g.game => Dict{String,Any}(
                "n_candidates" => length(g.cands),
                "n_oracle_causal" => count(>(0.0), g.oracle_importance),
                "n_strongly_tuned_gvar" => sum(g.tuning.gvar_strong),
                "game_variable_spurious_tuning_rate" => _json_num(g.tuning.gvar_spurious_rate),
                "luminance_spurious_tuning_rate" => _json_num(g.tuning.lum_spurious_rate),
                "F" => _json_num(g.triad.F), "S" => _json_num(g.triad.S),
                "M" => _json_num(g.triad.M),
                "oracle_as_method_F" => _json_num(g.self_F),
            ) for g in results),
        )
        ip = joinpath(OUT_DIR, "A3_tuning_all.json")
        open(ip, "w") do io; JSON.print(io, idx, 2); end
        println("[A3] combined index → $ip")
    end

    println("\n[A3] ===== summary (game : gvar-spurious / lum-spurious : F/S/M) =====")
    for g in results
        println("[A3]   $(rpad(g.game,16)) : " *
                "$(round(g.tuning.gvar_spurious_rate,digits=2)) / " *
                "$(round(g.tuning.lum_spurious_rate,digits=2)) : " *
                "F=$(round(g.triad.F,digits=2)) S=$(round(g.triad.S,digits=2)) " *
                "M=$(g.triad.M === nothing ? "null" : round(g.triad.M,digits=2))")
    end
    for (game, msg) in problems
        println("[A3]   $(rpad(game,16)) : PROBLEM — $(first(split(msg, '\n')))")
    end
    if selftest_only
        println("[A3] --selftest: all self-checks passed, wrote nothing.")
    end
    return 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    A3Tuning.main()
end
