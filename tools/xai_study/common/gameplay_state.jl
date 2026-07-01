# gameplay_state.jl — the SHARED Phase-A/B/C testbed for the P2 experiment
# redesign (xai_paper/xai_2_interpretability/experiment_redesign.md).
#
# This module is the ONE place the redesign's testbed decisions live, so every
# Phase-B/A/C runner (and the audit site) can share ONE state + ONE output +
# ONE sampler path per game. It fixes the five diagnosed problems by providing:
#
#   1. SHARED gameplay state (redesign §Problem 1 / protocol 1): a seed=0
#      deterministic RANDOM-ACTION action stream (NOT the pure-NOOP tape) rolled
#      to a fixed analysis frame `f*`, so the game is in genuine input-driven
#      gameplay. `gameplay_actions(...)` builds the stream; `boot_gameplay(...)`
#      returns the checkpoint AT f*. The state is shared across all methods.
#
#   2. ORACLE CAUSE-DENSITY gate (§Problem 1 / protocol 2): `cause_density(...)`
#      runs the §1 intervention oracle at the candidate state and counts causes
#      with |Δy| above a floor; `accept_state(...)` rejects states below a
#      threshold K (kills the seaquest-style 2/48 near-flat oracle column at the
#      source). Reported + auditable.
#
#   3. ONE SHARED explained output per game (§Problem 4 / protocol 3): a
#      SCREEN-BUFFER target — a named pixel/region located causally once per game,
#      that ALL methods explain (replaces the per-method ram[54] vs score@ram[49]
#      drift). `shared_output(...)` returns the (cell, reader) once.
#
#   4. SAMPLER-ON gradient (§Problem 2 / protocol 4): `sampler_position_read(...)`
#      routes a gradient through the emulator's bilinear (triangular-kernel)
#      sampler so ∂pixel/∂(position byte) is REAL, not the naive zero. Same
#      mechanism as tools/xai_si_gradient/si_joystick_gradient.jl. The naive
#      (vanishing) handle `position_read_zero` is kept for the side-by-side.
#
#   5. SCREEN-BUFFER gradient (§Problem 3 / protocol 5): the sampler read IS the
#      direct ∂output/∂screen-pixel path for a screen-buffer output; combined with
#      `screen_pixel_wrt_cause(...)` it gives the image-domain ∂screen-pixel/∂cause
#      map through the differentiable renderer path.
#
#   6. Expected-Gradients reference pool from the GAMEPLAY trajectory
#      (§Experiments-to-repeat row EG): `gameplay_reference_pool(...)` records the
#      per-frame RAM tapes along the SAME random-action stream, so the causal byte
#      VARIES across the pool ((x−x0)≠0) and EG is not zeroed by a constant byte.
#
# UNCHANGED (do-not-drift, per redesign §"What stays unchanged"): the §1 oracle
# instrument (oracle_intervene.jl machinery), the F∧S∧M triad, the 6 core games,
# seed=0 determinism, the shared-testbed principle. This module only changes the
# STATES + the SHARED OUTPUT + turns the sampler on; it does NOT touch the
# emulator core or the scoring metrics.
#
# NB (conformance horizon): Paper-1 guarantees bit-exactness only within ≈30-frame
# RAM / 60-frame screen of a fixed action stream. `gameplay_actions` therefore
# keeps the ANALYSIS window (`prefix` + `horizon`) inside the validated screen
# window and every state is re-asserted bit-exact (`assert_bit_exact`) by the
# caller before it is trusted.

module GameplayState

import Random
import Statistics

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen, get_ram
using JuTari.Diff: soft_ram_peek

# The §1 oracle machinery (env/boot/snapshot/intervene + Cause + candidate set).
# oracle_intervene.jl includes jutari_oracle.jl, so we get the whole harness.
include(joinpath(@__DIR__, "..", "ground_truth", "oracle_intervene.jl"))
using .OracleIntervene: Cause, build_pong_causes, candidate_ram_indices,
                        run_intervention, assert_bit_exact, BALL_X_IDX, BALL_Y_IDX
using .OracleIntervene.JutariOracle: Snapshot, snapshot, continue_from,
                                     intervene_ram!, intervene_tia!, RAM_SIZE,
                                     boot_replay

export CORE_GAMES, gameplay_actions, boot_gameplay, cause_density, accept_state,
       shared_output, position_read_zero, sampler_position_read, find_position_byte,
       pick_position_cause, geom_at, is_position_concept,
       screen_pixel_wrt_cause, gameplay_reference_pool, SharedState, build_shared_state

const CORE_GAMES = ["pong", "breakout", "space_invaders", "seaquest", "ms_pacman", "qbert"]

# ============================================================================
# Per-game ROM + RomSettings resolution. The oracle harness's settings_for only
# knows pong/breakout/space_invaders/Generic; the Phase-B runners carry the full
# core-6 map (mspacman alias + Joystick settings). We resolve ROMs/settings here
# with the SAME map the Phase-B runners use so this module can boot all 6 core
# games identically (CLAUDE.md rule #2: harness parity across tools).
# ============================================================================
const ROM_BASENAME = Dict(
    "pong" => "pong", "breakout" => "breakout",
    "space_invaders" => "space_invaders", "seaquest" => "seaquest",
    "ms_pacman" => "mspacman", "qbert" => "qbert")

const _PRIMARY_REPO = get(ENV, "XAI_PRIMARY_REPO",
                          "/Users/maier/Documents/code/UnderstandingVCS")

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

function load_env(; game::AbstractString)
    rom = read(rom_path_for(game))
    env = StellaEnvironment(rom, settings_for(game))
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    return env
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
# (1) SHARED GAMEPLAY STATE — a seeded random-action stream.
# ============================================================================
"""
    gameplay_actions(game; prefix, horizon, seed, n_actions) -> Vector{Int}

The redesign's shared analysis action stream (protocol 1): a seed=0 deterministic
random-action trajectory of length `prefix + horizon`. The first `prefix` frames
drive the game OUT of boot/attract into genuine input-driven gameplay; the trailing
`horizon` frames are the oracle roll-forward window (kept fixed so causes are scored
over play, not a frozen NOOP window).

`n_actions` = the game's discrete action count (default 18, the ALE full set). Every
action is `rand(0:n_actions-1)`. seed=0 ⇒ fully deterministic (the shared-testbed
principle). The whole stream is re-asserted bit-exact by the caller before use.

CONFORMANCE HORIZON: `prefix + horizon` should stay inside Paper-1's validated
window (≈60-frame screen). The default prefix=45, horizon=15 keeps the total at 60.
Boot/attract is ≤ the first few frames of the prefix, so boot/attract < 10% of the
analysed play window for games that leave attract on random input."""
function gameplay_actions(game::AbstractString; prefix::Integer = 45,
                          horizon::Integer = 15, seed::Integer = 0,
                          n_actions::Integer = 18)
    total = Int(prefix) + Int(horizon)
    rng = Random.MersenneTwister(Int(seed))
    return [rand(rng, 0:(Int(n_actions) - 1)) for _ in 1:total]
end

"""
    boot_gameplay(game, actions, prefix) -> StellaEnvironment

Boot (60 NOOP + 4 RESET) then replay `actions[1:prefix]` — the checkpoint AT the
analysis frame f*=prefix. Reuses the oracle's boot_replay so boot parity is
identical to the §1 instrument."""
boot_gameplay(game::AbstractString, actions::AbstractVector{<:Integer}, prefix::Integer) =
    boot_replay(actions, Int(prefix); game = game)

# ============================================================================
# (2) ORACLE CAUSE-DENSITY GATE.
# ============================================================================
"""
    cause_density(checkpoint, actions, prefix, horizon, causes, read_y; floor) -> (n, deltas)

Run the §1 intervention oracle at the candidate state for reader `read_y` and count
how many causes move the output by more than `floor` in absolute value (the
non-degenerate cause-set size). Returns (count, per-cause |Δy|). This is the metric
the acceptance gate thresholds on."""
function cause_density(checkpoint, actions, prefix::Integer, horizon::Integer,
                       causes::Vector{Cause}, read_y; floor::Real = 0.5)
    base = continue_from(checkpoint, Int.(actions[prefix + 1 : prefix + horizon]))
    y0 = read_y(base)
    deltas = Float64[]
    for c in causes
        snap = run_intervention(checkpoint, actions, Int(prefix), Int(horizon), c)
        push!(deltas, abs(read_y(snap) - y0))
    end
    n = count(>(Float64(floor)), deltas)
    return n, deltas
end

"""
    accept_state(n_causes; k) -> Bool

The state-acceptance filter: accept a (game, state) only if the oracle's
non-degenerate cause-set has ≥ `k` causes above the floor. `k=4` by default (well
above the seaquest 2/48 degeneracy the redesign flags; a rich, non-degenerate
cause-set). Returns true to ACCEPT."""
accept_state(n_causes::Integer; k::Integer = 4) = n_causes >= Int(k)

# ============================================================================
# (3) ONE SHARED explained output per game — a SCREEN-BUFFER target.
# ============================================================================
"""
    shared_output(checkpoint, base_screen; horizon) -> (cell, read_y)

The single, well-specified explained output ALL methods share (protocol 3): a
SCREEN-BUFFER REGION target, scored once and shared. Two coupled handles are
returned:

  * `cell` — a framebuffer cell located CAUSALLY: the first cell that changes when a
    sprite-position byte is perturbed at the checkpoint (so it sits on a moving
    sprite/ball). This is the point the SAMPLER slides the sprite through (the
    ∂output/∂screen-pixel handle for the gradient family).
  * `read_y(::Snapshot)::Float64` — the oracle-scored output = the COUNT of
    framebuffer cells that differ from the baseline frame (`n_changed_px`, the
    screen-buffer REGION output the §1 oracle already defines). A region is used
    (not a single pixel) because a single pixel's oracle column is inherently
    sparse — only a cause that repaints that exact cell moves it — which starves the
    cause-density gate; the region is dense, still a pure screen-buffer target, and
    directly plottable (redesign §Problem 4: "a specified pixel/region").

Fixed BEFORE any method runs; the oracle's true-causal region is computed once at
this output and shared. (Same causal locator as the oracle's ball_pixel_cell / the
Phase-B position_pixel_cell for `cell`; the region reader is the oracle's
`n_changed_px` output.)"""
function shared_output(checkpoint, base_screen::AbstractMatrix; horizon::Integer)
    function perturbed(idx, d)
        env = deepcopy(checkpoint)
        b = Int(env.console.bus.ram[idx + 1])
        intervene_ram!(env, idx, (b + d) & 0xFF)
        for _ in 1:horizon; env_step!(env, 0); end
        snapshot(env, Int(horizon)).screen
    end
    sx = perturbed(BALL_X_IDX, 24)
    sy = perturbed(BALL_Y_IDX, 24)
    changed = (base_screen .!= sx) .| (base_screen .!= sy)
    cell = if any(changed)
        ci = findfirst(changed); (ci[1], ci[2])
    else
        h, w = size(base_screen); (max(1, h ÷ 2), max(1, w ÷ 2))
    end
    # the oracle-scored SCREEN-BUFFER REGION output: how many framebuffer cells the
    # intervention repaints vs the baseline frame (the §1 `n_changed_px` output).
    read_y = s -> Float64(count(s.screen .!= base_screen))
    return cell, read_y
end

# ============================================================================
# (4) SAMPLER-ON gradient family — the bilinear sampler surrogate. Verbatim in
#     spirit from tools/xai_si_gradient/si_joystick_gradient.jl and the keystone
#     runner sampler_on/run_sampler_faithfulness.jl.
# ============================================================================
tri(t) = max(0f0, 1f0 - abs(t))

# candidate-cause selection knobs for the sampler position byte (P2-redesign fix;
# mirrors common/shared_testbed_impl.jl): the sampler position byte is chosen from
# the OVERLAP of the game's SCORED candidate cause set AND bytes that translate a
# MOVING sprite, so its gradient is BOTH nonzero AND scorable against the oracle.
const LOCAL_RATIO = 8.0

"""Is `concept` a sprite-POSITION concept (an x/y coordinate the sampler can
slide)? Matches the candidate-file concept vocabulary (…_x/…_y/.xy/column/
direction)."""
function is_position_concept(concept::AbstractString)
    c = lowercase(strip(String(concept)))
    isempty(c) && return false
    for pat in ("_x", "_y", ".xy", "._xy", ".x", ".y", "position", "column", "direction")
        occursin(pat, c) && return true
    end
    return false
end

"""The NAIVE position handle — the §1 vanishing (Prop. prop:zero): a framebuffer
pixel is a sprite column placed by round/argmax, so ∂pixel/∂ram ≡ 0. Returned
through a 0-coefficient sum so Zygote yields an all-zero gradient (not `nothing`)."""
position_read_zero(ram::AbstractVector{<:Real}) = 0f0 * sum(Float32.(ram))

"""
    find_position_byte(checkpoint, base_screen, cell; horizon) -> geom | nothing

The sprite-position RAM byte driving the shared-output pixel + its footprint, found
by REAL interventions at the checkpoint: perturb each generic sprite-position byte
(BALL_X_IDX / BALL_Y_IDX), pick the one moving the most cells, and take the local
footprint around `cell`. `geom = (pidx, base_val, offs, py0, px0)`; `nothing` for a
static frame (no position to restore ⇒ sampler gradient stays 0, the honest
outcome)."""
function find_position_byte(checkpoint, base_screen::AbstractMatrix, cell; horizon::Integer)
    function perturbed_screen(idx, d)
        env = deepcopy(checkpoint)
        b = Int(env.console.bus.ram[idx + 1])
        intervene_ram!(env, idx, (b + d) & 0xFF)
        for _ in 1:horizon; env_step!(env, 0); end
        snapshot(env, Int(horizon)).screen
    end
    best = nothing; best_n = 0
    for idx in (BALL_X_IDX, BALL_Y_IDX)
        sp = perturbed_screen(idx, 24)
        n = count(base_screen .!= sp)
        if n > best_n; best_n = n; best = idx; end
    end
    best === nothing && return nothing
    env = deepcopy(checkpoint)
    base_val = Int(env.console.bus.ram[best + 1])
    sp = perturbed_screen(best, 24)
    changed = base_screen .!= sp
    H, W = size(base_screen)
    r0 = max(1, cell[1] - 8);  r1 = min(H, cell[1] + 8)
    c0 = max(1, cell[2] - 16); c1 = min(W, cell[2] + 16)
    footprint = Tuple{Int,Int}[]
    for r in r0:r1, c in c0:c1
        changed[r, c] && push!(footprint, (r, c))
    end
    isempty(footprint) && return nothing
    py0 = minimum(first.(footprint)); px0 = minimum(last.(footprint))
    offs = Tuple((r - py0, c - px0) for (r, c) in footprint)
    return (best, base_val, offs, py0, px0)
end

"""Build the sampler geometry for a KNOWN position byte `pidx` at its `cell`: poke
+24, take the local changed footprint around the cell, record the sprite offsets.
`nothing` if the local window is empty."""
function geom_at(checkpoint, base_screen::AbstractMatrix, pidx::Integer, cell; horizon::Integer)
    env = deepcopy(checkpoint)
    base_val = Int(env.console.bus.ram[pidx + 1])
    intervene_ram!(env, pidx, (base_val + 24) & 0xFF)
    for _ in 1:horizon; env_step!(env, 0); end
    sp = snapshot(env, Int(horizon)).screen
    changed = base_screen .!= sp
    H, W = size(base_screen)
    r0 = max(1, cell[1] - 8);  r1 = min(H, cell[1] + 8)
    c0 = max(1, cell[2] - 16); c1 = min(W, cell[2] + 16)
    footprint = Tuple{Int,Int}[]
    for r in r0:r1, c in c0:c1
        changed[r, c] && push!(footprint, (r, c))
    end
    isempty(footprint) && return nothing
    py0 = minimum(first.(footprint)); px0 = minimum(last.(footprint))
    offs = Tuple((r - py0, c - px0) for (r, c) in footprint)
    return (pidx, base_val, offs, py0, px0)
end

"""
    pick_position_cause(checkpoint, base_screen, causes, cand_concepts; horizon) -> (geom, cell)

THE P2-redesign fix (mirrors shared_testbed_impl._st_pick_position_cause): choose
the sampler position byte from the OVERLAP of the SCORED candidate cause set AND
bytes that translate a MOVING, LOCALIZED sprite, and locate the shared-output cell
ON that sprite. Ranked by local footprint; global-repaint bytes rejected
(full ≤ LOCAL_RATIO×local). Falls back to the generic ball_x/ball_y locator when no
candidate qualifies."""
function pick_position_cause(checkpoint, base_screen::AbstractMatrix,
                            causes::Vector{Cause}, cand_concepts::AbstractDict; horizon::Integer)
    function perturbed_screen(idx, d)
        env = deepcopy(checkpoint)
        b = Int(env.console.bus.ram[idx + 1])
        intervene_ram!(env, idx, (b + d) & 0xFF)
        for _ in 1:horizon; env_step!(env, 0); end
        snapshot(env, Int(horizon)).screen
    end
    H, W = size(base_screen)
    function local_fp(changed, cell)
        r0 = max(1, cell[1] - 8);  r1 = min(H, cell[1] + 8)
        c0 = max(1, cell[2] - 16); c1 = min(W, cell[2] + 16)
        n = 0
        for r in r0:r1, c in c0:c1; changed[r, c] && (n += 1); end
        return n
    end
    scored_idx = Int[]
    for c in causes
        c.kind == "ram" || continue
        c.index in scored_idx || push!(scored_idx, c.index)
    end
    best = nothing
    for idx in scored_idx
        is_position_concept(get(cand_concepts, idx, "")) || continue
        sp = perturbed_screen(idx, 24)
        changed = base_screen .!= sp
        full = count(changed)
        full == 0 && continue
        ci = findfirst(changed); cell = (ci[1], ci[2])
        lf = local_fp(changed, cell)
        lf == 0 && continue
        full <= LOCAL_RATIO * lf || continue
        if best === nothing || lf > best[3]
            best = (idx, cell, lf)
        end
    end
    if best !== nothing
        pidx, cell, _ = best
        geom = geom_at(checkpoint, base_screen, pidx, cell; horizon = horizon)
        geom !== nothing && return geom, cell
    end
    sx = perturbed_screen(BALL_X_IDX, 24)
    sy = perturbed_screen(BALL_Y_IDX, 24)
    changed = (base_screen .!= sx) .| (base_screen .!= sy)
    cell = if any(changed)
        ci = findfirst(changed); (ci[1], ci[2])
    else
        (max(1, H ÷ 2), max(1, W ÷ 2))
    end
    geom = find_position_byte(checkpoint, base_screen, cell; horizon = horizon)
    return geom, cell
end

"""
    sampler_position_read(ram, geom, cell; scale) -> Float32

THE SAMPLER-ON screen-buffer output as a differentiable function of the FULL RAM
vector: the triangular-kernel occupancy of the shared-output `cell` for a sprite
whose column is a CONTINUOUS affine function of RAM byte `geom.pidx` (read through
`soft_ram_peek`). ∂(this)/∂ram[pidx] ≠ 0 — the RESTORED position gradient — and
∂/∂(other bytes)=0. This IS the direct ∂output/∂screen-pixel path for the
screen-buffer target (protocol 5)."""
function sampler_position_read(ram::AbstractVector{<:Real}, geom, cell; scale::Real = 1.0)
    pidx, base_val, offs, py0, px0 = geom
    px = Float32(px0) + Float32(scale) * (soft_ram_peek(ram, pidx) - Float32(base_val))
    rr = Float32(cell[1]); cc = Float32(cell[2])
    occ = 0f0
    for (dr, dc) in offs
        occ += tri(rr - (Float32(py0) + Float32(dr))) * tri(cc - (px + Float32(dc)))
    end
    return clamp(occ, 0f0, 1f0)
end

# ============================================================================
# (5) SCREEN-BUFFER gradient: ∂screen-pixel/∂cause (the image-domain attribution
#     of a single cause onto the picture — the Paper-1 cannon-figure object).
# ============================================================================
"""
    screen_pixel_wrt_cause(ram, geom, cell_grid; scale) -> Matrix{Float32}

The image-domain map ∂(soft occupancy over a grid of cells)/∂ram[pidx] for the
sampler sprite — i.e. how a unit change of the position cause repaints the picture,
evaluated on `cell_grid = (rows, cols)`. This is the direct ∂screen-pixel/∂cause
gradient through the differentiable sampler (protocol 5 / redesign Problem 3), and
is exactly the object Paper-1's fig:xai(b) plots. Column-only sprites give a band on
the sprite edges (the cannon-edge saliency)."""
function screen_pixel_wrt_cause(ram::AbstractVector{<:Real}, geom, cell_grid; scale::Real = 1.0)
    rows, cols = cell_grid
    pidx = geom[1]
    G = Matrix{Float32}(undef, length(rows), length(cols))
    # finite-difference over the position byte through the sampler occupancy — a
    # cheap, robust ∂pixel/∂cause per cell (Zygote per-cell is overkill; the sampler
    # is piecewise-linear so central differences are exact away from the kinks).
    ε = 0.5f0
    ramp = Float32.(collect(ram)); ramp[pidx + 1] += ε
    ramm = Float32.(collect(ram)); ramm[pidx + 1] -= ε
    for (i, r) in enumerate(rows), (j, c) in enumerate(cols)
        yp = sampler_position_read(ramp, geom, (r, c); scale = scale)
        ym = sampler_position_read(ramm, geom, (r, c); scale = scale)
        G[i, j] = (yp - ym) / (2ε)
    end
    return G
end

# ============================================================================
# (6) Expected-Gradients reference pool from the GAMEPLAY trajectory.
# ============================================================================
"""
    gameplay_reference_pool(game, actions, prefix) -> Matrix{Float32} (prefix × 128)

The EG background distribution D drawn from the SAME seeded random-action gameplay
stream (redesign: replace the NOOP tape so the causal byte VARIES). Each row is the
genuine RAM tape at a frame along the gameplay prefix — a reachable, on-distribution
state whose position/score bytes differ frame-to-frame, so EG's (x−x0) factor is
non-zero and EG is not artificially zeroed."""
function gameplay_reference_pool(game::AbstractString,
                                 actions::AbstractVector{<:Integer}, prefix::Integer)
    n = max(1, Int(prefix))
    env = load_env(; game = game)
    D = Matrix{Float32}(undef, n, RAM_SIZE)
    for t in 1:n
        env_step!(env, Int(actions[t]))
        D[t, :] = Float32.(collect(get_ram(env)))
    end
    return D
end

# ============================================================================
# The one-call testbed: build the shared state + gate it + fix the output.
# ============================================================================
struct SharedState
    game::String
    prefix::Int
    horizon::Int
    seed::Int
    actions::Vector{Int}
    checkpoint::StellaEnvironment
    causes::Vector{Cause}
    cand_indices::Vector{Int}
    cell::Tuple{Int,Int}                 # the shared screen-buffer output cell
    read_y::Function                     # Snapshot -> Float64 at `cell`
    cause_density::Int                   # #causes above the floor (gate metric)
    deltas::Vector{Float64}              # per-cause oracle |Δy| at the shared output
    accepted::Bool                       # passed the cause-density gate?
    geom::Any                            # sampler geometry (or nothing)
    ram_now::Vector{Float32}             # RAM at f* (the gradient endpoint x)
    ref_pool::Matrix{Float32}            # the gameplay EG reference pool
end

"""
    build_shared_state(game; prefix, horizon, seed, k, floor, verbose) -> SharedState

Assemble the redesigned SHARED testbed for one game: seeded random-action stream →
bit-exact assert → checkpoint at f* → causes → shared screen-buffer output →
cause-density gate → sampler geometry → gameplay EG reference pool. Everything a
Phase-B method needs, computed ONCE and shared."""
function build_shared_state(game::AbstractString; prefix::Integer = 45,
                            horizon::Integer = 15, seed::Integer = 0,
                            k::Integer = 4, floor::Real = 0.5, verbose::Bool = false)
    actions = gameplay_actions(game; prefix = prefix, horizon = horizon, seed = seed)
    total = prefix + horizon
    verbose && println("[gs] $game: asserting bit-exactness of the gameplay stream to f$total ...")
    assert_bit_exact(actions, total; game = game)

    cand = candidates_path_for(game)
    checkpoint = boot_gameplay(game, actions, prefix)
    at_target = continue_from(checkpoint, Int[])
    causes = build_pong_causes(cand, at_target)
    cand_indices = [idx for (idx, _) in candidate_ram_indices(cand)]

    base = continue_from(checkpoint, Int.(actions[prefix + 1 : total]))
    # P2-redesign fix: locate the shared-output CELL + the sampler position byte from
    # the game's OWN scored candidate cause set (a moving, localized sprite), so the
    # sampler gradient is BOTH nonzero AND scorable against the oracle. `shared_output`
    # (the generic-poke locator) is kept as the fallback inside pick_position_cause.
    cand_concepts = Dict{Int,String}()
    for (idx, concept) in candidate_ram_indices(cand)
        haskey(cand_concepts, idx) || (cand_concepts[idx] = string(concept))
    end
    geom, cell = pick_position_cause(checkpoint, base.screen, causes, cand_concepts; horizon = horizon)
    base_screen = base.screen
    read_y = s -> Float64(count(s.screen .!= base_screen))

    ncauses, deltas = cause_density(checkpoint, actions, prefix, horizon, causes, read_y;
                                    floor = floor)
    accepted = accept_state(ncauses; k = k)
    ram_now = Float32.(collect(at_target.ram))
    ref_pool = gameplay_reference_pool(game, actions, prefix)

    if verbose
        gtxt = geom === nothing ? "none (static frame)" :
               "RAM[$(geom[1])] base=$(geom[2]) footprint=$(length(geom[3]))px"
        println("[gs] $game: shared output = screen cell $cell; " *
                "cause-density = $ncauses/$(length(causes)) above floor=$floor " *
                "(gate k=$k ⇒ $(accepted ? "ACCEPT" : "REJECT")); position byte = $gtxt")
    end
    return SharedState(game, prefix, horizon, seed, actions, checkpoint, causes,
                       cand_indices, cell, read_y, ncauses, deltas, accepted, geom,
                       ram_now, ref_pool)
end

end # module
