# shared_testbed_impl.jl — the P2 experiment-redesign SHARED TESTBED as an
# INCLUDABLE fragment (NOT a module), so every Phase-B runner can build the ONE
# shared gameplay state + shared screen-buffer output + sampler-on position path
# operating on ITS OWN already-included oracle machinery.
#
# WHY a fragment, not `using .GameplayState`: common/gameplay_state.jl includes
# oracle_intervene.jl and so defines its OWN `OracleIntervene.Cause` type. A
# runner already includes oracle_intervene.jl (via pilot_ig_vs_oracle.jl or
# ig_baseline_sweep.jl) and builds causes with a DIFFERENT `Cause` instance. Two
# distinct module instances ⇒ two distinct `Cause` types that would not unify.
# To keep everything in the runner's OWN types, this fragment injects the
# runner's own oracle functions/types as ARGUMENTS (dependency injection), and is
# `include`d directly into the runner's module — the functions it defines close
# over nothing external.
#
# It is the exact same logic validated in common/gameplay_state.jl +
# sampler_on/smoke_gameplay_redesign.jl (seed=0 random-action gameplay stream;
# oracle cause-density gate k>=4 at |Δy|>0.5; shared screen-buffer REGION output
# `n_changed_px`; bilinear-sampler position read; gameplay EG reference pool).
#
# Contract — a runner includes this fragment ONCE, then calls
#   ts = build_shared_testbed(game;
#          settings_for=settings_for, rom_path_for=rom_path_for,
#          candidates_path_for=candidates_path_for,
#          build_causes=build_pong_causes, candidate_ram_indices=candidate_ram_indices,
#          continue_from=continue_from, snapshot=snapshot, env_step=env_step!,
#          intervene_ram=intervene_ram!, boot_replay=boot_replay,
#          run_intervention=run_intervention, soft_ram_peek=soft_ram_peek,
#          prefix=90, horizon=15, seed=0, k=4, floor=0.5)
# and gets a NamedTuple with every handle its compute_game needs (all typed in
# the runner's own module). See build_shared_testbed's docstring.

import Random as _ST_Random

# generic sprite-position RAM bytes used to LOCATE the moving-sprite output cell
# and the sampler position byte (Pong ball_x/ball_y; used generically as a
# sprite-x/sprite-y poke — the same constants oracle_intervene.jl exports).
const _ST_BALL_X_IDX = 49      # RAM[$31]
const _ST_BALL_Y_IDX = 54      # RAM[$36]

_st_tri(t) = max(0f0, 1f0 - abs(t))

"""The seeded random-action gameplay stream (redesign protocol 1): a seed=0
deterministic trajectory of length prefix+horizon. n_actions=18 (ALE full set)."""
function shared_gameplay_actions(; prefix::Integer = 90, horizon::Integer = 15,
                                 seed::Integer = 0, n_actions::Integer = 18)
    total = Int(prefix) + Int(horizon)
    rng = _ST_Random.MersenneTwister(Int(seed))
    return [rand(rng, 0:(Int(n_actions) - 1)) for _ in 1:total]
end

"""The NAIVE (vanishing) position handle — ∂pixel/∂ram ≡ 0 (Prop. prop:zero),
returned through a 0-coefficient sum so Zygote yields all-zeros (not `nothing`)."""
shared_position_read_zero(ram::AbstractVector{<:Real}) = 0f0 * sum(Float32.(ram))

"""
    build_shared_testbed(game; <injected oracle fns>, prefix, horizon, seed, k, floor)
      -> NamedTuple

Assemble the ONE shared testbed for `game` (redesign §"the redesigned protocol"):
seeded random-action stream → bit-exact-asserted checkpoint at f*=prefix →
runner's causes → shared screen-buffer REGION output (`n_changed_px`) →
cause-density gate → sampler geometry → gameplay EG reference pool.

Injected (all the runner's OWN symbols, so returned handles are runner-typed):
  * `settings_for`, `rom_path_for`, `candidates_path_for` — the runner's per-game map
  * `build_causes(cand_path, at_target_snapshot)` — the runner's build_pong_causes
  * `candidate_ram_indices(cand_path)` — the runner's candidate index reader
  * `continue_from(checkpoint, tail)::Snapshot`, `snapshot(env, frame)::Snapshot`
  * `env_step(env, action)`, `intervene_ram(env, idx, val)`
  * `boot_replay(actions, target_frame; game)` — the runner's boot+replay
  * `run_intervention(checkpoint, actions, tf, hz, cause)::Snapshot`
  * `soft_ram_peek(ram, idx)` — JuTari.Diff one-hot content read (for the sampler)
  * `load_env(game)` OPTIONAL — for the EG reference pool; if `nothing`, the pool
    is stepped via boot_replay+continue.

Returns a NamedTuple:
  game, prefix, horizon, seed, actions, total,
  checkpoint, at_target, base, causes, cand_indices,
  cell,            # (r,c) shared screen-buffer output cell (a moving-sprite pixel)
  read_y,          # Snapshot -> Float64  = n_changed_px region output (SHARED)
  cause_density, deltas, accepted,   # the gate (count above floor, per-cause |Δy|, k-pass)
  geom,            # sampler geometry (pidx,base_val,offs,py0,px0) or nothing (static)
  ram_now,         # Float32 RAM at f* (the gradient endpoint x)
  sampler_read,    # ram -> Float32 sampler-on position read (or zero-read if geom===nothing)
  position_read_zero,   # ram -> Float32 naive vanishing read
  ref_pool         # (prefix × 128) Float32 gameplay EG reference pool
"""
function build_shared_testbed(game::AbstractString;
        settings_for, rom_path_for, candidates_path_for,
        build_causes, candidate_ram_indices,
        continue_from, snapshot, env_step, intervene_ram,
        boot_replay, run_intervention, soft_ram_peek,
        load_env = nothing,
        prefix::Integer = 90, horizon::Integer = 15, seed::Integer = 0,
        k::Integer = 4, floor::Real = 0.5, verbose::Bool = false,
        assert_bit_exact = nothing)

    actions = shared_gameplay_actions(; prefix = prefix, horizon = horizon, seed = seed)
    total = Int(prefix) + Int(horizon)

    # bit-exact assertion of the exact stream (redesign validity gate). The runner
    # passes its own assert_bit_exact; if none given we skip (the runner already
    # asserts in compute_game). Prefer to assert here so the shared state is trusted.
    if assert_bit_exact !== nothing
        verbose && println("[ts] $game: asserting bit-exactness of the gameplay stream to f$total ...")
        assert_bit_exact(actions, total; game = game)
    end

    cand = candidates_path_for(game)
    checkpoint = boot_replay(actions, Int(prefix); game = game)
    at_target = continue_from(checkpoint, Int[])
    causes = build_causes(cand, at_target)
    cand_indices = [idx for (idx, _) in candidate_ram_indices(cand)]

    base = continue_from(checkpoint, Int.(actions[prefix + 1 : total]))
    base_screen = base.screen

    # --- (3) shared screen-buffer output: the moving-sprite cell + region read ---
    function _perturbed_screen(idx, d)
        env = deepcopy(checkpoint)
        b = Int(env.console.bus.ram[idx + 1])
        intervene_ram(env, idx, (b + d) & 0xFF)
        for _ in 1:horizon; env_step(env, 0); end
        snapshot(env, Int(horizon)).screen
    end
    sx = _perturbed_screen(_ST_BALL_X_IDX, 24)
    sy = _perturbed_screen(_ST_BALL_Y_IDX, 24)
    changed = (base_screen .!= sx) .| (base_screen .!= sy)
    cell = if any(changed)
        ci = findfirst(changed); (ci[1], ci[2])
    else
        h, w = size(base_screen); (max(1, h ÷ 2), max(1, w ÷ 2))
    end
    # the SHARED oracle output: n_changed_px region vs the baseline frame.
    read_y = s -> Float64(count(s.screen .!= base_screen))

    # --- (2) cause-density gate at the shared output ----------------------------
    y0 = read_y(base)
    deltas = Float64[]
    for c in causes
        snap = run_intervention(checkpoint, actions, Int(prefix), Int(horizon), c)
        push!(deltas, abs(read_y(snap) - y0))
    end
    cause_density = count(>(Float64(floor)), deltas)
    accepted = cause_density >= Int(k)

    # --- (4) sampler geometry: the sprite-position byte + its local footprint ----
    geom = _st_find_position_byte(checkpoint, base_screen, cell;
                                  horizon = horizon, env_step = env_step,
                                  intervene_ram = intervene_ram, snapshot = snapshot)

    ram_now = Float32.(collect(at_target.ram))
    sampler_read = geom === nothing ? shared_position_read_zero :
        (ram -> _st_sampler_position_read(ram, geom, cell; soft_ram_peek = soft_ram_peek))

    # --- (6) gameplay EG reference pool -----------------------------------------
    ref_pool = _st_reference_pool(game, actions, Int(prefix);
                                  load_env = load_env, boot_replay = boot_replay,
                                  snapshot = snapshot)

    if verbose
        gtxt = geom === nothing ? "none (static frame)" :
               "RAM[$(geom[1])] base=$(geom[2]) footprint=$(length(geom[3]))px"
        println("[ts] $game: shared output = screen cell $cell; " *
                "cause-density = $cause_density/$(length(causes)) above floor=$floor " *
                "(gate k=$k ⇒ $(accepted ? "ACCEPT" : "REJECT")); position byte = $gtxt")
    end

    return (game = String(game), prefix = Int(prefix), horizon = Int(horizon),
            seed = Int(seed), actions = actions, total = total,
            checkpoint = checkpoint, at_target = at_target, base = base,
            causes = causes, cand_indices = cand_indices, cell = cell, read_y = read_y,
            cause_density = cause_density, deltas = deltas, accepted = accepted,
            geom = geom, ram_now = ram_now, sampler_read = sampler_read,
            position_read_zero = shared_position_read_zero, ref_pool = ref_pool)
end

"""Find the sprite-position RAM byte driving the shared-output cell + its local
footprint (redesign protocol 4). Returns (pidx,base_val,offs,py0,px0) or nothing
(static frame ⇒ sampler gradient stays 0, the honest outcome)."""
function _st_find_position_byte(checkpoint, base_screen::AbstractMatrix, cell;
                                horizon::Integer, env_step, intervene_ram, snapshot)
    function perturbed_screen(idx, d)
        env = deepcopy(checkpoint)
        b = Int(env.console.bus.ram[idx + 1])
        intervene_ram(env, idx, (b + d) & 0xFF)
        for _ in 1:horizon; env_step(env, 0); end
        snapshot(env, Int(horizon)).screen
    end
    best = nothing; best_n = 0
    for idx in (_ST_BALL_X_IDX, _ST_BALL_Y_IDX)
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

"""The sampler-on screen-buffer output as a differentiable fn of the FULL RAM
vector — triangular-kernel occupancy of `cell` for a sprite whose column is a
continuous affine fn of RAM byte geom.pidx (read via soft_ram_peek). ∂/∂ram[pidx]≠0
(the RESTORED position gradient); ∂/∂(other bytes)=0. redesign protocol 5."""
function _st_sampler_position_read(ram::AbstractVector{<:Real}, geom, cell;
                                   soft_ram_peek, scale::Real = 1.0)
    pidx, base_val, offs, py0, px0 = geom
    px = Float32(px0) + Float32(scale) * (soft_ram_peek(ram, pidx) - Float32(base_val))
    rr = Float32(cell[1]); cc = Float32(cell[2])
    occ = 0f0
    for (dr, dc) in offs
        occ += _st_tri(rr - (Float32(py0) + Float32(dr))) * _st_tri(cc - (px + Float32(dc)))
    end
    return clamp(occ, 0f0, 1f0)
end

"""The image-domain ∂screen-pixel/∂cause map over a grid of cells (protocol 5 /
Problem 3) — how a unit change of the position cause repaints the picture. Central
differences over the position byte through the piecewise-linear sampler."""
function screen_pixel_wrt_cause(ram::AbstractVector{<:Real}, geom, cell_grid;
                                soft_ram_peek, scale::Real = 1.0)
    rows, cols = cell_grid
    pidx = geom[1]
    G = Matrix{Float32}(undef, length(rows), length(cols))
    ε = 0.5f0
    ramp = Float32.(collect(ram)); ramp[pidx + 1] += ε
    ramm = Float32.(collect(ram)); ramm[pidx + 1] -= ε
    for (i, r) in enumerate(rows), (j, c) in enumerate(cols)
        yp = _st_sampler_position_read(ramp, geom, (r, c); soft_ram_peek = soft_ram_peek, scale = scale)
        ym = _st_sampler_position_read(ramm, geom, (r, c); soft_ram_peek = soft_ram_peek, scale = scale)
        G[i, j] = (yp - ym) / (2ε)
    end
    return G
end

"""The EG background pool drawn from the SAME seeded random-action gameplay stream
(redesign EG row): each row = the RAM tape at a frame along the prefix, so the
causal byte VARIES ((x−x0)≠0) and EG is not zeroed by a constant byte. Built via
`boot_replay` so no extra step handle is needed; prefix is small (<=90) and this
runs once per game."""
function _st_reference_pool(game::AbstractString, actions::AbstractVector{<:Integer},
                            prefix::Integer; load_env = nothing, boot_replay, snapshot)
    n = max(1, Int(prefix))
    D = Matrix{Float32}(undef, n, 128)
    for t in 1:n
        env = boot_replay(actions, t; game = game)
        D[t, :] = Float32.(collect(env.console.bus.ram))
    end
    return D
end
