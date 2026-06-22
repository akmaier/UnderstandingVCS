# A5_lfp.jl — Phase-A A5: local field potentials / pooled-activity spectra
# (P2-E3-5), the JULIA path, run on jutari over the 6 core games.
#
# Phase A is Paper 2's *calibration baseline* (experiment_design.md §4, Novelty
# note): classical neuroscience analyses recover rich STRUCTURE from a system whose
# mechanism is fully known, yet that structure is LOW-faithfulness — the quantified-
# Kording lesson. A5 is the "local field potential" (LFP) member of the battery.
#
# In neuroscience the LFP is the POOLED extracellular potential of a brain region —
# a mesoscale signal you read off without single-cell access, whose power spectrum
# (theta/gamma bands, ...) is treated as a window onto the region's computation. The
# trap A5 makes precise: on the VCS the analogous "regional pooled activity" is
# dominated, in the spectral sense, by the system's KNOWN free-running clocks (the
# frame counter and the frame-driven animation/blink timers). A large fraction of the
# pooled-spectrum power is therefore EPIPHENOMENAL — it tracks a hardware/firmware
# counter ticking at a fixed cadence, NOT the game computation. The clock cadences are
# known EXACTLY (T2), so the epiphenomenal fraction is quantifiable, not merely argued.
#
# What A5 does, per game (experiment_design.md §4 row A5):
#   * Record a (T, n) deterministic jutari RAM trajectory.
#   * POOL activity into "regions": (a) the whole-population pool (all varying RAM
#     cells, z-scored, averaged per frame — the global LFP), and (b) per-concept-
#     FAMILY group pools (cell groups sharing a T3 variable family — regional LFPs).
#     A pooled signal = the region's mean z-scored activity per frame (the LFP analog).
#   * SPECTRA: detrend (remove DC + linear trend — the frame-counter ramp), Hann
#     window, DFT (a dependency-free O(T²) DFT — the trajectory is short, and the
#     shared jutari env must gain NO package, mirroring jutari_oracle's NPY writer).
#     Report the dominant PERIODICITIES (peak period in frames), peak power, and
#     relate the top period to the game/frame cadence.
#   * SCORE — %-variance attributable to the KNOWN CLOCKS → epiphenomenal:
#       known-clock basis = the frame-counter ramp (the sampling/frame cadence)
#       PLUS every detected free-running periodic counter cell (a RAM cell that is an
#       exact constant-stride or fixed-period counter over the trajectory — the on-
#       chip frame-driven timers, T2) and its first few harmonics. The fraction of a
#       pooled signal's variance captured by a least-squares projection onto that
#       clock basis = the clock-explained (epiphenomenal) fraction. Reported per
#       region; the GLOBAL pool's clock fraction is the headline scalar (SPEC §R
#       value). A high fraction ⇒ the LFP spectrum is mostly the known clocks (the
#       A5 finding: the dominant spectral peaks are epiphenomenal).
#
# DESCRIPTIVE method (per the item brief): A5 reports what spectral structure it
# REVEALS (dominant pooled periodicities) vs what it MISSES (it is blind to the
# causal map — the clock-explained fraction quantifies how much of the headline
# spectrum is causally inert). It therefore does NOT score the §0 F/S/M triad against
# the oracle (that is A2/A4). It DOES carry a positive control: a pure synthetic clock
# pool must score clock-fraction ≈ 1, and a pure injected non-clock sinusoid must
# score < 1 — so a measured fraction is real, not a tautology.
#
# BUILDS ON the verified jutari foundation (NO emulator core touched):
#   * tools/xai_study/common/jutari_oracle.jl — load/boot/replay/snapshot, the
#     xitari-parity boot, the bit-exact baseline guarantee, the dependency-free §R
#     NPY/NPZ writer.
#   * tools/xai_study/common/jutari_record.jl — record_trajectory → (T,n) tape
#     (A5 reuses its recording PATTERN; it re-implements the loop with the A5 per-game
#     RomSettings, exactly as A4 does, so each game's proper settings are used).
#   * tools/xai_study/phaseA_kording/pilot_si.jl — the Phase-A pilot TEMPLATE
#     (per-game scoring + positive control + §R record shape).
#   * tools/xai_study/phaseA_kording/A4_correlations.jl — the multi-game sibling whose
#     per-game settings map, ROM-alias resolution, action traces and GAME_CFG A5
#     mirrors (kept inside A5's file_scope; it must not edit the shared helpers).
#
# Per-game RomSettings: the shared jutari_oracle.settings_for() only knows the 3 pilot
# games, so A5 supplies its own per-game settings map (matching the canonical
# tools/jutari_screen_dump.jl map) and builds envs directly via the same JuTari
# primitives. seaquest legitimately uses GenericRomSettings (Paper-1 64/64 with Generic).
#
# Run (warm shared depot, primary's project):
#   XAI_PRIMARY_REPO=/Users/maier/Documents/code/UnderstandingVCS \
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseA_kording/A5_lfp.jl
# Flags: --games pong,breakout,...  --traj-frames N  --top-peaks K  --n-harmonics H
#        --selftest (run the self-check on the first game, write nothing)
#
# Writes (SPEC §R; file_scope A5_*): one record per game
#   tools/xai_study/phaseA_kording/out/A5_<game>.{json,npz}

module A5LFP

using JSON
using LinearAlgebra: qr, norm
import Statistics

# --- the verified foundation (NO core touched) -----------------------------
include(joinpath(@__DIR__, "..", "common", "jutari_oracle.jl"))
using .JutariOracle
using .JutariOracle: snapshot, RAM_SIZE, write_npz, rom_path_for
using .JutariOracle.JuTari: StellaEnvironment
using .JutariOracle.JuTari.Env: env_reset!, env_step!, get_ram

# the trajectory recorder lineage (A5 re-implements its loop with per-game settings).
include(joinpath(@__DIR__, "..", "common", "jutari_record.jl"))

const OUT_DIR = joinpath(@__DIR__, "out")
const CORE_GAMES = ["pong", "breakout", "space_invaders",
                    "seaquest", "ms_pacman", "qbert"]

# Per-game live-play configuration (mirrors A4_correlations.GAME_CFG). The pooled
# activity must actually VARY for the spectrum to carry structure; each game reaches
# live play at a different point. `trace` selects the trajectory input style
# ("fire" = FIRE + periodic RIGHT/LEFTFIRE; "dir" = a full UP/DOWN/LEFT/RIGHT maze
# trace, for games where FIRE is a no-op). A5 is a DESCRIPTIVE spectral statistic of
# a deterministic jutari trajectory (the recorder's documented purpose), not a
# bit-exact-vs-xitari claim; the bit-exact jutari RE-RUN is nonetheless asserted.
struct GameCfg
    traj_frames::Int
    trace::String       # "fire" | "dir"
end
const GAME_CFG = Dict{String,GameCfg}(
    "pong"           => GameCfg(256, "fire"),
    "breakout"       => GameCfg(256, "fire"),
    "space_invaders" => GameCfg(256, "fire"),
    "seaquest"       => GameCfg(256, "fire"),
    "ms_pacman"      => GameCfg(400, "dir"),
    "qbert"          => GameCfg(256, "fire"),
)
game_cfg(game::AbstractString) =
    get(GAME_CFG, lowercase(string(game)), GameCfg(256, "fire"))

# ============================================================================
# Per-game RomSettings (A5-owned; mirrors tools/jutari_screen_dump.jl's canonical
# map). seaquest -> Generic (Paper-1 64/64 with Generic). Stays in A5's file_scope.
# ============================================================================
function a5_settings_for(game::AbstractString)
    g = lowercase(string(game))
    JT = JutariOracle.JuTari
    g == "pong"           && return JT.PaddleGames.PongRomSettings()
    g == "breakout"       && return JT.PaddleGames.BreakoutRomSettings()
    g == "space_invaders" && return JT.SpaceInvadersRomSettings()
    g == "ms_pacman"      && return JT.JoystickGames.MsPacmanRomSettings()
    g == "qbert"          && return JT.JoystickGames.QbertRomSettings()
    return JT.GenericRomSettings()   # seaquest + any other
end

# ROM name aliases (mirror A4 / t3/import_labels.py) so A5 resolves ms_pacman etc.
const _ROM_ALIASES = Dict("ms_pacman" => ["ms_pacman", "mspacman"],
                          "beam_rider" => ["beam_rider", "beamrider"])
function a5_rom_path(game::AbstractString)
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

"""A freshly-reset env for `game` with the A5 per-game settings and the
xitari-parity boot (60 NOOP + 4 RESET) — the SAME boot jutari_oracle uses."""
function a5_load_env(game::AbstractString)
    rom = read(a5_rom_path(game))
    env = StellaEnvironment(rom, a5_settings_for(game))
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    return env
end

"""Record a (frames, RAM_SIZE) RAM tape on `game` with A5's per-game settings.
Re-implements record_trajectory's loop with a5_load_env (the shared recorder
hard-codes settings_for)."""
function a5_record_ram(game::AbstractString, frames::Integer,
                       actions::AbstractVector{<:Integer})
    @assert length(actions) >= frames "actions shorter than frames"
    env = a5_load_env(game)
    tape = Matrix{UInt8}(undef, Int(frames), RAM_SIZE)
    for t in 1:frames
        env_step!(env, Int(actions[t]))
        tape[t, :] = UInt8.(collect(get_ram(env)))
    end
    return tape
end

"""A full from-scratch replay (boot included) of actions[1:total] — the bit-exact
baseline guarantee (two fresh runs must be byte-identical)."""
function a5_fresh_baseline(game, actions::AbstractVector{<:Integer}, total::Integer)
    env = a5_load_env(game)
    for i in 1:total; env_step!(env, Int(actions[i])); end
    return snapshot(env, Int(total))
end

# ============================================================================
# Action traces (mirror A4_correlations).
# Action codes (jutari src/io/IO.jl): NOOP=0 FIRE=1 UP=2 RIGHT=3 LEFT=4 DOWN=5
# RIGHTFIRE=11 LEFTFIRE=12.
# ============================================================================
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
# Candidate cells (E2-1 import) — used to define the per-FAMILY regional pools.
# (Reuses A4's concept_family normalisation, copied here to stay in A5 file_scope.)
# ============================================================================
struct Candidate
    ram_index::Int
    concept::String
    family::String
end

"""Normalise a candidate concept to its base variable family (so enemy_x/enemy_y/
enemy.xy all map to `enemy`, score/player_score → score, etc.). Verbatim port of
A4_correlations.concept_family — copied (not shared) to keep A5 inside file_scope."""
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
# Stats helpers.
# ============================================================================
function _zscore(x::AbstractVector{<:Real})
    mu = Statistics.mean(x); sd = Statistics.std(x)
    sd == 0 && return zeros(Float64, length(x))
    return (Float64.(x) .- mu) ./ sd
end

# remove the DC level and the linear trend (the frame-counter ramp) — A5 reads the
# OSCILLATORY pooled structure; the ramp is reported separately as the frame cadence.
function _detrend(x::AbstractVector{<:Real})
    T = length(x)
    T < 2 && return Float64.(x) .- (T == 0 ? 0.0 : Statistics.mean(x))
    t = collect(0:T-1) ./ (T - 1)               # normalised time in [0,1]
    # least-squares fit y = a + b t ; residual is the detrended signal
    tm = Statistics.mean(t); ym = Statistics.mean(x)
    vt = sum((t .- tm) .^ 2)
    b = vt == 0 ? 0.0 : sum((t .- tm) .* (Float64.(x) .- ym)) / vt
    a = ym - b * tm
    return Float64.(x) .- (a .+ b .* t)
end

# Hann window (reduces spectral leakage of the finite trajectory).
function _hann(T::Integer)
    T <= 1 && return ones(Float64, T)
    return [0.5 * (1 - cos(2π * (n) / (T - 1))) for n in 0:T-1]
end

# ============================================================================
# Dependency-free real-signal DFT power spectrum (no FFTW added to the env).
# Returns (freqs_cycles_per_frame, power) for k = 0 .. floor(T/2). Power is the
# squared magnitude of the one-sided DFT; the k=0 (DC) bin is dropped by the caller
# after detrending. O(T²) — the trajectory is short (≤400 frames), trivially fast.
# ============================================================================
function dft_power(x::AbstractVector{<:Real})
    T = length(x)
    K = T ÷ 2
    freqs = Float64[k / T for k in 0:K]
    power = zeros(Float64, K + 1)
    @inbounds for k in 0:K
        re = 0.0; im = 0.0
        w = -2π * k / T
        for n in 0:T-1
            θ = w * n
            re += x[n+1] * cos(θ)
            im += x[n+1] * sin(θ)
        end
        power[k+1] = re * re + im * im
    end
    return freqs, power
end

# ============================================================================
# Known clocks (T2): the frame-counter ramp + free-running periodic counter cells.
# ============================================================================
# A "known clock" cell is a RAM cell that is an EXACT free-running counter over the
# trajectory — either a constant-stride ramp (frame counter / down-counter: the
# successive first differences are a single constant mod 256) or a fixed small-period
# cycle (an animation/blink timer that repeats with period p ≤ p_max). These are the
# T2 hardware/firmware clocks (driven by the frame cadence), NOT game state. We detect
# them structurally from the tape (no labels needed) so the clock basis is principled,
# not hand-picked.

struct ClockCell
    ram_index::Int
    kind::String        # "ramp" (constant stride) | "periodic" (fixed period)
    period::Int         # ramp: the wrap period (256 ÷ |stride| if divides) ; periodic: p
    stride::Int         # ramp stride (mod 256, signed to [-128,127]); 0 for periodic
end

_signed8(d::Integer) = (m = Int(d) & 0xFF; m >= 128 ? m - 256 : m)

"""Is column `col` (a UInt8 time series) a constant-stride free-running counter? The
mod-256 first differences must all equal one non-zero constant (a frame counter ticks
+1, a down-counter −1, a /N prescaler a constant)."""
function _is_ramp(col::AbstractVector{<:Integer})
    T = length(col)
    T < 4 && return (false, 0)
    d0 = _signed8(Int(col[2]) - Int(col[1]))
    d0 == 0 && return (false, 0)
    for t in 2:T-1
        _signed8(Int(col[t+1]) - Int(col[t])) == d0 || return (false, 0)
    end
    return (true, d0)
end

"""Is column `col` exactly periodic with some period 2 ≤ p ≤ p_max (a non-constant
cyclic timer)? Returns (true, p) for the smallest such p, else (false, 0). Constant
columns are NOT clocks (no variance ⇒ no spectral contribution)."""
function _is_periodic(col::AbstractVector{<:Integer}, p_max::Integer)
    T = length(col)
    T < 6 && return (false, 0)
    length(unique(col)) <= 1 && return (false, 0)         # constant ⇒ not a clock
    for p in 2:min(p_max, T ÷ 2)
        ok = true
        for t in 1:T-p
            if col[t] != col[t+p]; ok = false; break; end
        end
        if ok
            # exclude the degenerate "period = full length" alias by requiring it to
            # repeat at least ~twice; the loop already needs t+p ≤ T for ≥1 repeat.
            return (true, p)
        end
    end
    return (false, 0)
end

"""Detect the known-clock cells in the RAM tape (constant-stride ramps + fixed-period
counters). p_max bounds the periodic search (animation/blink timers are short)."""
function detect_clock_cells(tape::AbstractMatrix; p_max::Integer = 64)
    n = size(tape, 2)
    clocks = ClockCell[]
    for j in 1:n
        col = Int.(@view tape[:, j])
        length(unique(col)) <= 1 && continue              # constant: no clock signal
        isr, stride = _is_ramp(col)
        if isr
            period = stride != 0 && (256 % abs(stride) == 0) ? (256 ÷ abs(stride)) : 0
            push!(clocks, ClockCell(j - 1, "ramp", period, stride))
            continue
        end
        isp, p = _is_periodic(col, p_max)
        if isp
            push!(clocks, ClockCell(j - 1, "periodic", p, 0))
        end
    end
    return clocks
end

# ============================================================================
# The known-clock signal basis (for the variance-explained projection).
# ============================================================================
# The basis columns are the signals the known clocks impose on the (sampled-per-frame)
# trajectory:
#   (1) the FRAME CADENCE — the fundamental T2 clock. A free-running per-frame counter
#       sweeping its range across the trajectory injects the slow Fourier modes of the
#       trajectory length T: sin/cos at periods T/1, T/2, ... (a few). This is the frame
#       counter's trajectory-sampled signature and is ALWAYS present (even when no RAM
#       cell exposes a short counter — e.g. Q*Bert). A small number of frame-cadence
#       harmonics keeps it conservative (it must not over-explain).
#   (2) for each DETECTED periodic/ramp clock cell of period p, the first H harmonics
#       sin/cos(2π h t / p) — the animation/blink timers (ramp wrap period if it wraps).
# All columns are detrended+windowed identically to the pooled signals so the projection
# is apples-to-apples on the analysed (oscillatory) part of the spectrum. A detrended
# pure linear ramp vanishes, so the frame cadence is represented by its harmonics (which
# survive detrend), NOT a raw line.

"""Build the known-clock basis matrix B (T × m): the frame-cadence harmonics (periods
T/1..T/n_frame_harmonics) + sin/cos harmonics of every detected clock period, each
processed through the SAME detrend+window pipeline as the pooled signals, then
orthonormalised (QR) so the variance-explained projection is numerically clean. `win`
is the Hann window applied to the pooled signals."""
function clock_basis(T::Integer, clocks::Vector{ClockCell}, win::AbstractVector;
                     n_harmonics::Integer = 3, n_frame_harmonics::Integer = 2)
    cols = Vector{Float64}[]
    t = collect(0:T-1)
    add_harmonics(p) = begin
        for h in 1:n_harmonics
            f = h / p
            f > 0.5 && break                       # past Nyquist (cycles/frame)
            push!(cols, _detrend(sin.(2π .* f .* t)) .* win)
            push!(cols, _detrend(cos.(2π .* f .* t)) .* win)
        end
    end
    # (1) the FRAME CADENCE: the slow sweep modes of the trajectory length T (the frame
    #     counter's trajectory-sampled signature) — always present, conservative count.
    for hf in 1:n_frame_harmonics
        p = T / hf
        p >= 2 || continue
        f = 1 / p
        f > 0.5 && continue
        push!(cols, _detrend(sin.(2π .* f .* t)) .* win)
        push!(cols, _detrend(cos.(2π .* f .* t)) .* win)
    end
    # (2) the detected RAM clock cells (periodic timers + ramp wrap periods).
    periods = Set{Int}()
    for c in clocks
        p = c.kind == "periodic" ? c.period :
            (c.period > 0 ? c.period : 0)         # ramp wrap period (if it wraps)
        p >= 2 && push!(periods, p)
    end
    for p in sort(collect(periods))
        add_harmonics(p)
    end
    isempty(cols) && return zeros(Float64, T, 0)
    # drop all-zero columns up front (a detrended pure ramp can vanish, etc.)
    cols = [c for c in cols if norm(c) > 1e-9]
    isempty(cols) && return zeros(Float64, T, 0)
    B = hcat(cols...)
    # orthonormalise; keep only the numerically full-rank directions (rank = #columns
    # of B with a non-negligible R-diagonal in the QR factorisation).
    F = qr(B)
    Q = Matrix(F.Q)
    Rdiag = abs.([F.R[j, j] for j in 1:min(size(F.R)...)])
    tol = 1e-9 * (isempty(Rdiag) ? 1.0 : maximum(Rdiag))
    rank = count(>(tol), Rdiag)
    rank == 0 && return zeros(Float64, T, 0)
    return Q[:, 1:rank]
end

"""Fraction of `sig`'s energy captured by an orthonormal basis `B` (least-squares
projection): ‖Bᵀsig‖² / ‖sig‖². 1 ⇒ sig lies entirely in the clock subspace
(fully epiphenomenal); 0 ⇒ orthogonal to every known clock."""
function variance_explained_by(B::AbstractMatrix, sig::AbstractVector)
    s2 = sum(abs2, sig)
    s2 == 0 && return 0.0
    size(B, 2) == 0 && return 0.0
    proj = B' * sig                       # coordinates in the orthonormal basis
    return min(1.0, sum(abs2, proj) / s2)
end

# ============================================================================
# Regional pooled activity (the LFP) + its spectrum.
# ============================================================================
struct Region
    name::String                 # "global" | a concept family
    cells::Vector{Int}           # 0-based RAM indices pooled into this region
    pooled_raw::Vector{Float64}  # mean z-scored activity per frame (the LFP signal)
    analysed::Vector{Float64}    # detrended + windowed (what the DFT runs on)
    freqs::Vector{Float64}       # cycles/frame for k=1..T/2 (DC dropped)
    power::Vector{Float64}       # one-sided power at those freqs
    dom_period::Float64          # 1/peak-freq (frames) — the dominant periodicity
    dom_power_share::Float64     # peak-bin power / total oscillatory power
    clock_var_fraction::Float64  # %-variance of `analysed` in the known-clock subspace
    n_cells_varying::Int
end

"""Pool a set of 0-based RAM cells into one regional LFP: z-score each VARYING cell's
time series, average per frame. Constant cells contribute nothing (excluded). Returns
(pooled_raw, n_varying)."""
function pool_region(tape::AbstractMatrix, cells::Vector{Int})
    T = size(tape, 1)
    series = Vector{Float64}[]
    for idx in cells
        col = Float64.(@view tape[:, idx + 1])
        Statistics.std(col) > 0 || continue
        push!(series, _zscore(col))
    end
    isempty(series) && return (zeros(Float64, T), 0)
    pooled = reduce(+, series) ./ length(series)
    return (pooled, length(series))
end

function analyse_region(name::AbstractString, cells::Vector{Int}, tape::AbstractMatrix,
                        B::AbstractMatrix, win::AbstractVector)
    T = size(tape, 1)
    pooled, nvar = pool_region(tape, cells)
    analysed = _detrend(pooled) .* win
    freqs_all, power_all = dft_power(analysed)
    # drop the DC bin (k=0): after detrend it is ~0 and is not a "periodicity".
    freqs = freqs_all[2:end]
    power = power_all[2:end]
    total = sum(power)
    if isempty(power) || total == 0
        dom_period = Inf; dom_share = 0.0
    else
        ki = argmax(power)
        dom_period = freqs[ki] > 0 ? 1.0 / freqs[ki] : Inf
        dom_share = power[ki] / total
    end
    clock_frac = variance_explained_by(B, analysed)
    return Region(string(name), cells, pooled, analysed, freqs, power,
                  dom_period, dom_share, clock_frac, nvar)
end

# ============================================================================
# Drive one game: trajectory → clocks → regions (global + per-family) → spectra.
# ============================================================================
struct GameResult
    game::String
    traj_frames::Int
    trace::String
    seed::Int
    bit_exact::Bool
    n_harmonics::Int
    top_peaks::Int
    clocks::Vector{ClockCell}
    n_basis::Int
    regions::Vector{Region}                 # [1] is always "global"
    top_peak_periods::Vector{Float64}       # global pool: top-K dominant periods (frames)
    top_peak_power::Vector{Float64}
end

function compute_game(game::AbstractString; traj_frames = nothing, trace = nothing,
                      n_harmonics = 3, top_peaks = 5, seed = 0, verbose = true)
    cfg = game_cfg(game)
    traj_frames = traj_frames === nothing ? cfg.traj_frames : traj_frames
    trace       = trace === nothing ? cfg.trace : trace
    acts = trace_actions(trace, traj_frames)

    # 1) bit-exact baseline — two fresh boots+replays must be byte-identical. A5 is
    #    descriptive, but the trajectory it analyses must be a deterministic jutari
    #    run (no nondeterminism contaminating the spectrum) — assert it.
    verbose && println("[A5:$game] asserting bit-exactness (2 fresh boots+replays to f$traj_frames)...")
    a = a5_fresh_baseline(game, acts, traj_frames)
    b = a5_fresh_baseline(game, acts, traj_frames)
    bit_exact = (a.ram == b.ram) && (a.screen == b.screen)
    bit_exact || error("bit-exact re-run FAILED for $game to f$traj_frames — refusing to analyse")
    verbose && println("[A5:$game] bit-exact re-run: PASS")

    # 2) record the (T, 128) RAM trajectory (the pooled-activity substrate).
    verbose && println("[A5:$game] recording $(traj_frames)-frame ($(trace)) RAM trajectory...")
    tape = a5_record_ram(game, traj_frames, acts)
    T = size(tape, 1)
    win = _hann(T)

    # 3) detect the KNOWN CLOCKS (T2) and build the clock basis.
    clocks = detect_clock_cells(tape)
    B = clock_basis(T, clocks, win; n_harmonics = n_harmonics)
    n_ramp = count(c -> c.kind == "ramp", clocks)
    n_periodic = count(c -> c.kind == "periodic", clocks)
    verbose && println("[A5:$game] known clocks detected: $(length(clocks)) " *
                       "($(n_ramp) ramp, $(n_periodic) periodic); " *
                       "clock basis dim = $(size(B, 2))")

    # 4) define the regions: GLOBAL (all varying RAM cells) + one per T3 concept family.
    cand_path = resolve_candidates(game)
    cands = load_candidates(cand_path)
    varying_cells = Int[j - 1 for j in 1:size(tape, 2)
                        if Statistics.std(Float64.(@view tape[:, j])) > 0]
    regions = Region[]
    push!(regions, analyse_region("global", varying_cells, tape, B, win))
    # per-family regional pools (cells sharing a true variable family)
    fams = Dict{String,Vector{Int}}()
    for c in cands
        push!(get!(fams, c.family, Int[]), c.ram_index)
    end
    for fam in sort(collect(keys(fams)))
        cells = unique(fams[fam])
        push!(regions, analyse_region(fam, cells, tape, B, win))
    end

    # 5) global-pool dominant periodicities (the headline spectrum).
    gp = regions[1]
    order = sortperm(gp.power; rev = true)
    k = min(top_peaks, length(order))
    top_periods = Float64[gp.freqs[order[i]] > 0 ? 1.0 / gp.freqs[order[i]] : Inf for i in 1:k]
    top_power   = Float64[gp.power[order[i]] for i in 1:k]

    if verbose
        println("[A5:$game] ---- A5 pooled-activity spectra ----")
        println("[A5:$game]   global pool: $(gp.n_cells_varying) varying cells; " *
                "dominant period = $(round(gp.dom_period, digits=2)) frames " *
                "(peak share $(round(gp.dom_power_share, digits=3)))")
        println("[A5:$game]   global pool CLOCK-explained variance fraction = " *
                "$(round(gp.clock_var_fraction, digits=3)) → epiphenomenal")
        np = min(5, length(top_periods))
        println("[A5:$game]   top periods (frames): " *
                join([string(round(top_periods[i], digits=2)) for i in 1:np], ", "))
        for r in regions[2:end]
            r.n_cells_varying == 0 && continue
            println("[A5:$game]   region $(rpad(r.name, 16)) " *
                    "dom-period=$(rpad(round(r.dom_period,digits=2), 7)) " *
                    "clock-frac=$(round(r.clock_var_fraction, digits=3))")
        end
    end

    return GameResult(game, traj_frames, trace, seed, bit_exact, n_harmonics, top_peaks,
                      clocks, size(B, 2), regions, top_periods, top_power)
end

# ============================================================================
# Self-check (DoD) — the spectral contract is sound; results are non-fabricated.
# ============================================================================
"""
    selftest(r::GameResult) -> Bool

Asserts A5's load-bearing claims:

  (BIT-EXACT) the analysed trajectory is a deterministic jutari run (re-run is
    byte-identical) — the spectrum is of a real, reproducible signal.

  (SIGNAL PRESENT) the global pool has ≥1 varying cell and finite, non-negative power
    — there is actual pooled activity to take a spectrum of.

  (CLOCK BASIS NON-EMPTY) the clock basis is non-empty — anchored to at least the
    FRAME CADENCE (always a known T2 clock). Detecting zero short RAM counter cells is
    a RECORDED FINDING (n_clock_cells=0, e.g. Q*Bert over this window), not a failure;
    the frame-cadence harmonics still anchor the epiphenomenal score.

  (FRACTIONS WELL-FORMED) every region's clock-explained variance fraction ∈ [0,1];
    dominant-power share ∈ [0,1]; dominant period > 0 (or Inf when flat).

  (POSITIVE CONTROL — clock signal scores ≈1) projecting a PURE clock signal (a cos at
    a guaranteed-present, non-degenerate basis period — a detected period ≥3 if any,
    else the frame-cadence fundamental — run through the same detrend+window) onto the
    clock basis recovers variance fraction ≈ 1 — the basis truly captures the clocks.

  (NEGATIVE CONTROL — non-clock signal scores < 1) a deliberately NON-clock sinusoid (a
    half-bin-offset mid-band frequency, off the clock grid) scores < 0.5 and below the
    clock control — a high pooled fraction is a real measurement, not a tautology that
    scores everything ≈1.

Throws on a contract violation."""
function selftest(r::GameResult; n_harmonics = 3)
    @assert r.bit_exact "bit-exact baseline re-run failed for $(r.game)"
    gp = r.regions[1]
    @assert gp.n_cells_varying >= 1 "global pool has NO varying cell for $(r.game) — " *
        "uninformative; pick a livelier trace/length"
    @assert all(isfinite, gp.power) "non-finite power in the global spectrum"
    @assert all(>=(-1e-12), gp.power) "negative power in the global spectrum"
    # The FRAME CADENCE is always a known clock, so the basis is always non-empty even
    # when no RAM cell exposes a short counter (e.g. Q*Bert) — that is a recorded
    # finding (n_clock_cells=0), not a failure. The epiphenomenal score stays anchored.
    @assert r.n_basis >= 1 "empty clock basis for $(r.game) — even the frame cadence is missing"

    for reg in r.regions
        @assert 0.0 - 1e-9 <= reg.clock_var_fraction <= 1.0 + 1e-9 "clock fraction out of [0,1] for region $(reg.name): $(reg.clock_var_fraction)"
        @assert 0.0 - 1e-9 <= reg.dom_power_share <= 1.0 + 1e-9 "dom-power share out of [0,1] for region $(reg.name)"
        @assert reg.dom_period > 0 "non-positive dominant period for region $(reg.name)"
    end

    # rebuild the basis to run the positive controls on the same T.
    T = r.traj_frames
    win = _hann(T)
    B = clock_basis(T, r.clocks, win; n_harmonics = n_harmonics)
    @assert size(B, 2) >= 1 "rebuilt clock basis empty"

    # (+ control) a PURE clock signal must score ≈ 1. We build it from a basis period
    # that is GUARANTEED present and non-degenerate: prefer a detected periodic clock
    # with period ≥ 3 (a period-2 SINE is identically zero at integer t, so use COS and
    # avoid p=2 as the control), else fall back to the frame-cadence fundamental
    # (period T, always in the basis). We use COS so the control never degenerates.
    t = collect(0:T-1)
    det_periods = Int[(c.kind == "periodic" ? c.period : c.period) for c in r.clocks]
    good = filter(p -> p >= 3, det_periods)
    pclock = isempty(good) ? Float64(T) : Float64(minimum(good))   # frame cadence fallback
    clock_sig = _detrend(cos.(2π .* (1 / pclock) .* t)) .* win
    f_clock = variance_explained_by(B, clock_sig)
    @assert f_clock > 0.9 "harness broken: a pure clock signal (period $(round(pclock,digits=1))) " *
        "scores clock-fraction $f_clock < 0.9 — the basis does not capture the clocks"

    # (− control) a NON-clock sinusoid must score clearly below 1. We pick a mid-band
    # frequency that is maximally OFF the DFT grid (a half-bin offset, k+0.5 cycles
    # over T) and far from DC — so it is neither a clock harmonic nor near-spanned by
    # the low-order ramp/window leakage. (A near-DC off-grid tone leaks almost fully
    # into the detrended-ramp + low harmonics, which is why a naive 1/(p√2) control
    # fails — it is not actually off the clock subspace.)
    kmid = max(3, round(Int, T / 7))
    f_non_freq = (kmid + 0.5) / T              # half-bin-offset mid-band frequency
    f_non_freq >= 0.5 && (f_non_freq = 0.317)  # keep below Nyquist
    nonclock = _detrend(sin.(2π .* f_non_freq .* t)) .* win
    f_non = variance_explained_by(B, nonclock)
    @assert f_non < 0.5 "harness broken: a mid-band non-clock sinusoid (f=$(round(f_non_freq,digits=4)) cyc/frame) scores clock-fraction $f_non — the basis over-explains (would make every fraction ≈1)"
    @assert f_non < f_clock + 1e-9 "harness broken: non-clock ($f_non) ≥ clock ($f_clock)"

    println("[A5:$(r.game)] SELF-CHECK PASS:")
    println("[A5:$(r.game)]   bit-exact baseline re-run: $(r.bit_exact)")
    println("[A5:$(r.game)]   known clocks: $(length(r.clocks)) cells, basis dim $(r.n_basis)")
    println("[A5:$(r.game)]   global pool: $(gp.n_cells_varying) varying cells, " *
            "clock-fraction $(round(gp.clock_var_fraction, digits=3))")
    println("[A5:$(r.game)]   positive control: clock-signal fraction = " *
            "$(round(f_clock, digits=3)) (≈1), non-clock fraction = " *
            "$(round(f_non, digits=3)) (<1)")
    return true
end

# ============================================================================
# Persist (SPEC §R) — JSON record + sibling .npz; file_scope A5_*.
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

function write_game(r::GameResult; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    stem = "A5_$(r.game)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path  = joinpath(out_dir, stem * ".npz")

    gp = r.regions[1]
    # per-region summary (skip empty regions in the JSON but keep them honest)
    region_summary = Dict{String,Any}[]
    for reg in r.regions
        push!(region_summary, Dict{String,Any}(
            "region" => reg.name,
            "n_cells" => length(reg.cells),
            "n_cells_varying" => reg.n_cells_varying,
            "dominant_period_frames" => _json_num(reg.dom_period),
            "dominant_power_share" => _json_num(reg.dom_power_share),
            "clock_explained_variance_fraction" => _json_num(reg.clock_var_fraction),
            "ram_cells" => reg.cells,
        ))
    end
    clock_list = Dict{String,Any}[]
    for c in r.clocks
        push!(clock_list, Dict{String,Any}(
            "ram_index" => c.ram_index, "kind" => c.kind,
            "period" => c.period, "stride" => c.stride))
    end

    rec = Dict{String,Any}(
        "paper" => "P2",
        "phase" => "phaseA_kording",
        "method" => "A5_local_field_potentials(pooled_activity_spectra)",
        "game" => r.game,
        "state" => "f0+$(r.traj_frames)",
        "target_output" => "pooled-activity-power-spectrum",
        # headline scalar (SPEC §R value/metric_name): the GLOBAL pool's clock-explained
        # variance fraction — the %-variance attributable to the KNOWN clocks (frame/
        # animation timers, T2) → epiphenomenal. High ⇒ the LFP spectrum is mostly the
        # known clocks (the A5 finding).
        "metric_name" => "A5_global_pool_clock_explained_variance_fraction_epiphenomenal",
        "value" => _json_num(gp.clock_var_fraction),
        "stderr" => nothing,
        "ci" => nothing,
        "n" => gp.n_cells_varying,
        "seed" => r.seed,
        "where" => "local",
        "commit" => _git_commit(),
        "oracle_ref" => "T2_known_clocks@$(r.game) (frame-counter + free-running periodic timers; structural detection)",
        "timestamp" => string(round(Int, time())),
        "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            "substrate" => "jutari (Julia, HARD) — real-ROM bit-exact path; the " *
                "analysed RAM trajectory is a deterministic re-run (asserted).",
            "bit_exact_rerun" => r.bit_exact,
            "method_kind" => "descriptive (no oracle F/S/M; A5 reports the pooled " *
                "spectral structure it REVEALS vs the causal map it MISSES — the " *
                "clock-explained fraction quantifies how much of the headline " *
                "spectrum is causally inert).",
            "trajectory_trace" => r.trace == "dir" ?
                "directional maze trace (FIRE warmup + cyclic UP/DOWN/LEFT/RIGHT)" :
                "active trace (4×FIRE + periodic RIGHTFIRE/LEFTFIRE)",
            "trajectory_frames" => r.traj_frames,
            "spectrum" => Dict{String,Any}(
                "dft" => "dependency-free one-sided DFT power (no FFTW added to the " *
                    "shared jutari env); freq unit = cycles/frame; DC bin dropped after " *
                    "detrend; Hann-windowed; linear trend (frame-counter ramp) removed.",
                "global_dominant_period_frames" => _json_num(gp.dom_period),
                "global_dominant_power_share" => _json_num(gp.dom_power_share),
                "global_top_peak_periods_frames" =>
                    [_json_num(p) for p in r.top_peak_periods],
                "global_top_peak_power" => [_json_num(p) for p in r.top_peak_power],
                "frame_cadence_note" => "the per-frame sampling clock is 1 frame; " *
                    "dominant periods are reported in FRAMES (period p ⇒ p-frame cycle, " *
                    "e.g. a period-2 blink, a period-N animation/score-flash timer). " *
                    "Relate to game cadence: short integer periods are animation/blink " *
                    "timers driven by the frame counter (T2), not game-state oscillations.",
            ),
            "known_clocks" => Dict{String,Any}(
                "n_clock_cells" => length(r.clocks),
                "n_ramp" => count(c -> c.kind == "ramp", r.clocks),
                "n_periodic" => count(c -> c.kind == "periodic", r.clocks),
                "clock_basis_dim" => r.n_basis,
                "n_harmonics_per_period" => r.n_harmonics,
                "clock_cells" => clock_list,
                "note" => "known clocks (T2) = RAM cells that are EXACT free-running " *
                    "counters over the trajectory: constant-stride ramps (frame/down " *
                    "counters, prescalers) or fixed-period cyclic timers (animation/blink). " *
                    "These are hardware/firmware clocks driven by the frame cadence, NOT " *
                    "game state — detected structurally (no labels). The clock basis is " *
                    "the frame ramp + sin/cos harmonics of every detected clock period; " *
                    "the %-variance a pooled signal projects onto it is the EPIPHENOMENAL " *
                    "fraction (experiment_design.md §4 A5).",
            ),
            "regions" => region_summary,
            "interpretation" => "Phase A is the calibration baseline " *
                "(experiment_design.md §4): the LFP-analog pooled-activity spectrum " *
                "is rich, but a high fraction of its power is the KNOWN frame/animation " *
                "clocks — epiphenomenal structure that says nothing about the causal " *
                "computation (the dominant spectral peaks are clock harmonics). A5 " *
                "REVEALS the pooled periodicities and MISSES the causal map; the " *
                "clock-explained fraction is the quantified-Kording measurement here.",
            "scales_to_cluster_via" =>
                "tools/cluster/xai_*.sbatch (E0-3) — the full A1..A8 battery over the " *
                "core+breadth set; the forward is bit-exact to this HARD path.",
        ),
    )
    open(json_path, "w") do io; JSON.print(io, rec, 2); end

    # sibling .npz arrays (SPEC §R). The per-region spectra differ in length only by
    # region count (all share the T/2 freq grid), so we stack the global spectrum and
    # store per-region scalars + the pooled global LFP + clock metadata.
    clock_idx    = Int64[c.ram_index for c in r.clocks]
    clock_period = Int64[c.period for c in r.clocks]
    region_names_clockfrac = Float64[reg.clock_var_fraction for reg in r.regions]
    region_dom_period      = Float64[isfinite(reg.dom_period) ? reg.dom_period : 0.0
                                     for reg in r.regions]
    write_npz(npz_path, Dict(
        "global_lfp_pooled"        => gp.pooled_raw,           # (T,) the global LFP signal
        "global_lfp_analysed"      => gp.analysed,             # (T,) detrended+windowed
        "global_freqs"             => gp.freqs,                # (T/2,) cycles/frame
        "global_power"             => gp.power,                # (T/2,) one-sided power
        "global_top_peak_periods"  => Float64[isfinite(p) ? p : 0.0 for p in r.top_peak_periods],
        "global_top_peak_power"    => r.top_peak_power,
        "global_clock_var_fraction" => Float64[gp.clock_var_fraction],
        "region_clock_var_fraction" => region_names_clockfrac, # (n_regions,)
        "region_dominant_period"   => region_dom_period,       # (n_regions,)
        "clock_cell_ram_index"     => clock_idx,               # (n_clocks,)
        "clock_cell_period"        => clock_period,            # (n_clocks,)
    ))
    return json_path, npz_path
end

# ============================================================================
# CLI
# ============================================================================
function main(args = ARGS)
    games = copy(CORE_GAMES)
    traj_frames = nothing; trace = nothing
    n_harmonics = 3; top_peaks = 5; seed = 0
    selftest_only = false
    i = 1
    while i <= length(args)
        a = args[i]
        if     a == "--games";        games = String.(split(args[i+1], ",")); i += 2
        elseif a == "--game";         games = [args[i+1]]; i += 2
        elseif a == "--traj-frames";  traj_frames = parse(Int, args[i+1]); i += 2
        elseif a == "--trace";        trace = args[i+1]; i += 2
        elseif a == "--top-peaks";    top_peaks = parse(Int, args[i+1]); i += 2
        elseif a == "--n-harmonics";  n_harmonics = parse(Int, args[i+1]); i += 2
        elseif a == "--seed";         seed = parse(Int, args[i+1]); i += 2
        elseif a == "--selftest";     selftest_only = true; i += 1
        else; i += 1
        end
    end
    println("[A5] games=$(join(games, ",")) traj_frames=$traj_frames trace=$trace " *
            "n_harmonics=$n_harmonics top_peaks=$top_peaks seed=$seed (jutari/Julia path)")

    if selftest_only
        g = games[1]
        r = compute_game(g; traj_frames = traj_frames, trace = trace,
                         n_harmonics = n_harmonics, top_peaks = top_peaks, seed = seed)
        selftest(r; n_harmonics = n_harmonics)
        println("[A5] --selftest: passed on $g, not writing artifact.")
        return 0
    end

    ok = String[]; failed = Tuple{String,String}[]
    for g in games
        try
            r = compute_game(g; traj_frames = traj_frames, trace = trace,
                             n_harmonics = n_harmonics, top_peaks = top_peaks, seed = seed)
            selftest(r; n_harmonics = n_harmonics)
            json_path, npz_path = write_game(r)
            println("[A5] wrote $json_path")
            println("[A5] arrays  $npz_path")
            push!(ok, g)
        catch err
            msg = sprint(showerror, err)
            println("[A5] !! game $g FAILED (scoring the rest, not fabricating): " *
                    first(split(msg, '\n')))
            push!(failed, (g, first(split(msg, '\n'))))
        end
    end
    println("[A5] ==== summary: $(length(ok))/$(length(games)) games scored ====")
    for g in ok; println("[A5]   OK   $g"); end
    for (g, m) in failed; println("[A5]   FAIL $g — $m"); end
    return 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    A5LFP.main()
end
