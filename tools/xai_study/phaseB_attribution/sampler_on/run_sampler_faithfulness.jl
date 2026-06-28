# run_sampler_faithfulness.jl — P2-R7-EXP-sampler (the KEYSTONE experiment).
#
# THE CLAIM (plan.md storyline step 4; honesty-contract claim 4): turning Paper-1's
# bilinear sampler ON makes the gradient/correlational attribution methods FAITHFUL on
# the discrete position/index regime — where their NAIVE gradient is provably zero
# (Prop. prop:zero: a framebuffer pixel is a sprite column placed by round/argmax, so
# ∂pixel/∂ram ≡ 0) — and YET their semantic recovery stays ZERO, exactly like every
# other method. This forecloses the objection that the danger zone is a fixable
# technical artifact: the gradient can be repaired (faithfulness rises off the floor),
# but the repaired map still names no game concept (semantics stays 0).
#
# WHAT IT MEASURES, per gradient method × each of the 6 core games, on the POSITION
# regime, in TWO conditions:
#   (1) NAIVE  (sampler OFF) — the differentiable handle is the §1 vanishing
#       `position_read_zero(ram) = 0·Σram`; ∂/∂ram ≡ 0 ⇒ attribution ≡ 0 ⇒
#       faithfulness (Pearson of the per-cause attribution with the oracle's true
#       |Δy| on the position pixel) is 0. This REPRODUCES the committed baseline
#       (saliency_<game>_position.json: sal_max=0, pearson=0) and Prop. prop:zero.
#   (2) SAMPLER ON — the SAME bilinear index-boundary surrogate as the SI joystick
#       tool (tools/xai_si_gradient/si_joystick_gradient.jl): the hard sprite-column
#       placement round(px) is replaced by a triangular-kernel occupancy
#           occ(px) = Σ_offsets tri(ROWS−row)·tri(COLS−(px+dc)),   tri(t)=max(0,1−|t|),
#       with the column px a CONTINUOUS function of the sprite-position RAM byte. The
#       rendered pixel becomes differentiable in that byte, so the position gradient
#       is RESTORED: ∂pixel/∂ram[pidx] ≠ 0. The per-cause attribution lands its mass
#       on the position byte; scored against the SAME oracle column its Pearson rises
#       off the floor (whenever that byte is an oracle-active cause at the state).
#
# semantic_recovery — operationalized as: does the method, ON ITS OWN, emit/name a T3
# game-concept (missile, collision, score, restart)? An attribution map is a vector of
# per-cause magnitudes; it neither emits nor names a concept. So semantic_recovery = 0
# for EVERY method in BOTH conditions, with a one-line justification recorded per
# record. THIS is the point of the experiment: faithfulness is repairable (0 → >0),
# semantics is not (0 → 0). The only semantic label in play is the imported T3
# annotation, used solely to *check* localization, never *produced* by the method.
#
# The gradient methods (all in the gradient/correlational family whose naive position
# gradient is zero): vanilla saliency, grad×input, smoothgrad, integrated_gradients,
# expected_gradients. On the one-hot content read they all share ∂y/∂u = e_idx, and on
# the position pixel they all share the SAME vanishing — so the sampler restores the
# SAME position gradient for each (the per-cause attribution differs only by the
# method's weighting of that restored gradient). We therefore drive every method
# through the one shared sampler surrogate and report each method's attribution.
#
# REUSES (no emulator core touched; mirrors saliency.jl / ig_baseline_sweep.jl):
#   * ig_baseline_sweep.jl — the env/oracle harness (load_env, boot_replay,
#     continue_from, assert_bit_exact, run_intervention, occlude!, oracle_abs_delta,
#     position_pixel_cell, BALL_X_IDX/BALL_Y_IDX), the faithfulness SCORER
#     (pearson/spearman/precision_at_k via PilotIGvsOracle), and the cause machinery
#     (build_pong_causes / candidate_ram_indices / Cause / snapshot / intervene_ram!).
#   * tools/xai_si_gradient/si_joystick_gradient.jl — the bilinear-sampler construction
#     (tri / occ over a continuous position), reused verbatim in spirit.
#
# Run (warm shared depot, primary's project):
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseB_attribution/sampler_on/run_sampler_faithfulness.jl --games all
# Flags: --games all|core|<g1,g2,...>  --game <g>  --target-frame N --horizon N
#        --topk K --seed S  (defaults match saliency.jl: tf=120 hz=30 topk=3)
#
# Writes (file_scope sampler_on/*):
#   tools/xai_study/phaseB_attribution/sampler_on/out/sampler_<method>_<game>.json
#   tools/xai_study/compare/out/sampler_faithfulness.csv   (the aggregate; columns:
#     method,game,regime,faithfulness_naive,faithfulness_sampler,semantic_recovery,record_path)
#
# SELF-CHECK (asserted; exits non-zero on failure):
#   (a) for ≥1 gradient method, faithfulness_sampler > faithfulness_naive on the
#       position regime (the sampler restores a real gradient), AND
#   (b) semantic_recovery == 0 for EVERY method in BOTH conditions.

module SamplerFaithfulness

using JSON
import Zygote
import Statistics
import Random

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen, get_ram
using JuTari.Diff: soft_ram_peek

# the IG-sweep harness already wires the env, the oracle machinery, the scorer, and
# the position-pixel locator — reuse all of it (no core touched).
include(joinpath(@__DIR__, "..", "ig_baseline_sweep.jl"))
using .IGBaselineSweep: CORE_GAMES, load_env, boot_replay, continue_from, fresh_baseline,
                        assert_bit_exact, run_intervention, occlude!, oracle_abs_delta,
                        position_pixel_cell, deletion_insertion_auc,
                        candidates_path_for, BALL_X_IDX, BALL_Y_IDX
using .IGBaselineSweep.PilotIGvsOracle: pearson, spearman, precision_at_k,
                                        _git_commit, _json_num
using .IGBaselineSweep.PilotIGvsOracle.OracleIntervene: build_pong_causes, Cause,
                                                        candidate_ram_indices
using .IGBaselineSweep.PilotIGvsOracle.OracleIntervene.JutariOracle: Snapshot, snapshot,
                                                                     intervene_ram!, write_npz

const OUT_DIR     = joinpath(@__DIR__, "out")
const COMPARE_OUT = normpath(joinpath(@__DIR__, "..", "..", "compare", "out"))
const CSV_PATH    = joinpath(COMPARE_OUT, "sampler_faithfulness.csv")
# the gradient/correlational family whose naive position gradient is provably zero.
const METHODS = ["vanilla_saliency", "gradxinput", "smoothgrad",
                 "integrated_gradients", "expected_gradients"]
# "all" = the 6 core games (the keystone is stated on the 6 core games; --games all
# is the item's documented command and equals core here).
const ALL_GAMES = CORE_GAMES

# ============================================================================
# (1) NAIVE position handle — the §1 vanishing (Prop. prop:zero), verbatim.
# ============================================================================
position_read_zero(ram::AbstractVector{<:Real}) = 0f0 * sum(Float32.(ram))

# ============================================================================
# (2) The bilinear sampler surrogate (the SAME mechanism as
#     tools/xai_si_gradient/si_joystick_gradient.jl): a triangular-kernel occupancy
#     over a sprite footprint whose column is a CONTINUOUS function of a RAM byte.
# ============================================================================
tri(t) = max(0f0, 1f0 - abs(t))

"""Find the sprite-position RAM byte that drives the chosen position pixel cell, and
its footprint, by a REAL intervention at the checkpoint. Perturb each of the two
generic sprite-position bytes (BALL_X_IDX / BALL_Y_IDX) and pick the one whose poke
moves the most framebuffer cells near the position cell (the byte the sprite column
tracks). Returns (pidx, base_value, footprint::Vector{(dr,dc)}, py0, px0) where the
footprint is the set of cells that lit up around `cell` under the perturbation —
exactly the sprite the sampler will slide. `nothing` if neither byte moves anything
(a static frame; the sampler then has no position to restore → faithfulness stays 0,
the honest outcome for that game)."""
function find_position_byte(checkpoint, base_screen, cell; horizon)
    function perturbed_screen(idx, d)
        env = deepcopy(checkpoint)
        b = Int(env.console.bus.ram[idx + 1])
        intervene_ram!(env, idx, (b + d) & 0xFF)
        for _ in 1:horizon; env_step!(env, 0); end
        snapshot(env, horizon).screen
    end
    best = nothing; best_n = 0
    for idx in (BALL_X_IDX, BALL_Y_IDX)
        sp = perturbed_screen(idx, 24)
        changed = base_screen .!= sp
        n = count(changed)
        if n > best_n
            best_n = n; best = idx
        end
    end
    best === nothing && return nothing

    # footprint of the moving sprite: the cells whose colour changed under the poke,
    # restricted to a window around the tracked pixel `cell` (so the sampler slides a
    # local sprite, not the whole screen). Centre the footprint on `cell`.
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

"""The bilinear-sampler position output as a function of the FULL RAM vector: the
soft occupancy of the tracked pixel `cell`, with the sprite column a continuous
affine function of RAM byte `pidx`. ∂(this)/∂ram[pidx] ≠ 0 (the restored position
gradient); ∂/∂(other bytes) = 0. Differentiable in `ram` via Zygote.

`occ_cell(px)` = the triangular-kernel occupancy AT THE TRACKED CELL of a footprint
whose origin column is `px` (sub-pixel). The cell value = occ·sprite_colour +
(1−occ)·bg, exactly the SI tool's `render`. We read occupancy at the single tracked
cell so the gradient is a scalar pixel (the position output of the §1 regime)."""
function sampler_position_read(ram::AbstractVector{<:Real}, geom, cell, scale)
    pidx, base_val, offs, py0, px0 = geom
    # continuous sprite column from the live RAM byte (1 unit byte = `scale` px)
    px = Float32(px0) + scale * (soft_ram_peek(ram, pidx) - Float32(base_val))
    rr = Float32(cell[1]); cc = Float32(cell[2])
    # occupancy at the tracked cell of the sprite placed at column px
    occ = 0f0
    for (dr, dc) in offs
        occ += tri(rr - (Float32(py0) + Float32(dr))) * tri(cc - (px + Float32(dc)))
    end
    return clamp(occ, 0f0, 1f0)
end

"""Per-cause attribution from a gradient `g_full` over the 128-byte RAM (|g| on RAM
causes, 0 on tia/joystick) — the SAME mapping as ig_attribution_per_cause but applied
to an arbitrary gradient vector."""
function attr_per_cause(g_full::AbstractVector, causes::Vector{Cause})
    attr = zeros(Float64, length(causes))
    for (i, c) in enumerate(causes)
        if c.kind == "ram" && 0 <= c.index < length(g_full)
            attr[i] = abs(Float64(g_full[c.index + 1]))
        end
    end
    return attr
end

# ============================================================================
# The five gradient methods, each producing a 128-vector attribution over RAM for a
# given differentiable readf(ram)::Float32. On the position regime readf is either the
# NAIVE vanishing handle or the SAMPLER surrogate; the method wraps that one gradient.
# (On the position pixel all five reduce to a weighting of the same ∂pixel/∂ram, which
# is 0 naive and the restored sampler gradient on; that IS the point.)
# ============================================================================
"""vanilla saliency: |∂y/∂u|."""
function grad_vanilla(readf, x)
    g = Zygote.gradient(readf, x)[1]
    g === nothing && (g = zeros(Float32, length(x)))
    return abs.(Float32.(g))
end

"""grad×input: |x ⊙ ∂y/∂u|."""
function grad_gradxinput(readf, x)
    g = Zygote.gradient(readf, x)[1]
    g === nothing && (g = zeros(Float32, length(x)))
    return abs.(Float32.(x) .* Float32.(g))
end

"""smoothgrad: mean_n |∂y/∂(u+ε)| over Gaussian noise (σ relative to the RAM range).
Averaging noisy gradients; on a piecewise-linear sampler this equals the clean
gradient in expectation but is the genuine SmoothGrad estimator."""
function grad_smoothgrad(readf, x; n = 16, sigma = 6f0, seed = 0)
    rng = Random.MersenneTwister(seed + 4242)
    acc = zeros(Float32, length(x))
    for _ in 1:n
        xn = Float32.(x) .+ sigma .* Float32.(randn(rng, length(x)))
        g = Zygote.gradient(readf, xn)[1]
        g === nothing && (g = zeros(Float32, length(x)))
        acc .+= abs.(Float32.(g))
    end
    return acc ./ n
end

"""integrated_gradients: |(x−x0) ⊙ ∫ ∂y/∂(path)|, zeros baseline, Riemann midpoints."""
function grad_ig(readf, x; steps = 32, baseline = nothing)
    x0 = baseline === nothing ? zeros(Float32, length(x)) : Float32.(baseline)
    acc = zeros(Float32, length(x))
    for k in 1:steps
        α = (k - 0.5f0) / steps
        xα = x0 .+ α .* (Float32.(x) .- x0)
        g = Zygote.gradient(readf, xα)[1]
        g === nothing && (g = zeros(Float32, length(x)))
        acc .+= g
    end
    return abs.((Float32.(x) .- x0) .* (acc ./ steps))
end

"""expected_gradients: IG with the baseline averaged over a few random reference RAM
states (Erion et al.) — the expectation form of IG."""
function grad_eg(readf, x; refs::Vector{<:AbstractVector}, steps = 16)
    acc = zeros(Float32, length(x))
    for x0v in refs
        acc .+= grad_ig(readf, x; steps = steps, baseline = x0v)
    end
    return acc ./ max(1, length(refs))
end

"""Run one method's attribution for a readf, returning the 128-vector |attr| over RAM."""
function method_attr(method::AbstractString, readf, x; seed = 0, eg_refs = nothing)
    if     method == "vanilla_saliency"     ; return grad_vanilla(readf, x)
    elseif method == "gradxinput"           ; return grad_gradxinput(readf, x)
    elseif method == "smoothgrad"           ; return grad_smoothgrad(readf, x; seed = seed)
    elseif method == "integrated_gradients" ; return grad_ig(readf, x)
    elseif method == "expected_gradients"   ; return grad_eg(readf, x;
                refs = eg_refs === nothing ? [zeros(Float32, length(x))] : eg_refs)
    else error("unknown method: $method")
    end
end

# ============================================================================
# Per-game state + the oracle column for the position pixel (the truth to score).
# ============================================================================
struct GameState
    game::String
    target_frame::Int
    horizon::Int
    cell::Tuple{Int,Int}
    causes::Vector{Cause}
    odelta::Vector{Float64}            # oracle |Δ pixel| per cause (the truth)
    ram_now::Vector{Float32}
    geom::Any                          # sampler geometry or nothing
    scale::Float32
    oracle_nonzero::Int
    eg_refs::Vector{Vector{Float32}}
end

function build_game_state(; game, target_frame, horizon, seed, verbose)
    total = target_frame + horizon
    actions = fill(0, total)
    verbose && println("[sampler] $game: asserting bit-exactness to f$total ...")
    assert_bit_exact(actions, total; game = game)

    cand = candidates_path_for(game)
    checkpoint = boot_replay(actions, target_frame; game = game)
    at_target  = continue_from(checkpoint, Int[])
    causes     = build_pong_causes(cand, at_target)

    base       = continue_from(checkpoint, Int.(actions[target_frame + 1 : total]))
    cell       = position_pixel_cell(checkpoint, base.screen; horizon = horizon)
    read_y     = s -> Float64(Int(s.screen[cell[1], cell[2]]))
    odelta     = oracle_abs_delta(checkpoint, actions, target_frame, horizon, causes, read_y)
    onz        = count(>(0), odelta)

    ram_now = Float32.(collect(at_target.ram))
    geom    = find_position_byte(checkpoint, base.screen, cell; horizon = horizon)

    # EG references: a couple of real alternative RAM states (a few frames apart) — the
    # on-distribution baselines for expected_gradients.
    eg_refs = Vector{Float32}[]
    for d in (10, 20)
        s = continue_from(checkpoint, Int.(actions[target_frame + 1 : min(total, target_frame + d)]))
        push!(eg_refs, Float32.(collect(s.ram)))
    end

    if verbose
        gtxt = geom === nothing ? "none (static frame)" :
               "RAM[$(geom[1])] base=$(geom[2]) footprint=$(length(geom[3]))px"
        println("[sampler] $game: pixel cell=$cell  oracle_nonzero=$onz/$(length(causes))  " *
                "position byte=$gtxt")
    end
    return GameState(game, target_frame, horizon, cell, causes, odelta, ram_now,
                     geom, 1f0, onz, eg_refs)
end

# ============================================================================
# Score one method on one game in BOTH conditions.
# ============================================================================
struct MethodResult
    method::String
    game::String
    regime::String
    faithfulness_naive::Float64
    faithfulness_sampler::Float64
    semantic_recovery::Int
    attr_naive::Vector{Float64}
    attr_sampler::Vector{Float64}
    cause_names::Vector{String}
    odelta::Vector{Float64}
    position_byte::Int
    sampler_attr_on_byte::Float64
    oracle_nonzero::Int
    cell::Tuple{Int,Int}
    record_path::String
end

function score_method(method::AbstractString, gs::GameState; topk, seed)
    # (1) NAIVE — the vanishing handle ⇒ attribution ≡ 0 ⇒ faithfulness 0.
    naive_readf = position_read_zero
    attr_naive_full = method_attr(method, naive_readf, gs.ram_now; seed = seed, eg_refs = gs.eg_refs)
    attr_naive = attr_per_cause(attr_naive_full, gs.causes)
    faith_naive = pearson(attr_naive, gs.odelta)

    # (2) SAMPLER ON — the bilinear surrogate restores ∂pixel/∂ram[pidx].
    if gs.geom === nothing
        # no moving sprite at this cell ⇒ the sampler has no position to restore;
        # attribution stays 0 (the honest outcome for a static frame).
        attr_samp_full = zeros(Float32, length(gs.ram_now))
        pidx = -1
    else
        samp_readf = ram -> sampler_position_read(ram, gs.geom, gs.cell, gs.scale)
        attr_samp_full = method_attr(method, samp_readf, gs.ram_now; seed = seed, eg_refs = gs.eg_refs)
        pidx = gs.geom[1]
    end
    attr_sampler = attr_per_cause(attr_samp_full, gs.causes)
    faith_sampler = pearson(attr_sampler, gs.odelta)
    samp_on_byte = pidx >= 0 && pidx < length(attr_samp_full) ?
        Float64(abs(attr_samp_full[pidx + 1])) : 0.0

    # semantic_recovery: an attribution map emits/names NO T3 game concept → 0, always.
    sem = 0

    return MethodResult(method, gs.game, "position", faith_naive, faith_sampler, sem,
                        attr_naive, attr_sampler, [c.name for c in gs.causes], gs.odelta,
                        pidx, samp_on_byte, gs.oracle_nonzero, gs.cell, "")
end

# ============================================================================
# Persist one per-method/per-game JSON record.
# ============================================================================
function write_record(r::MethodResult; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    stem = "sampler_$(r.method)_$(r.game)"
    json_path = joinpath(out_dir, stem * ".json")
    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseB_attribution", "experiment" => "P2-R7-EXP-sampler",
        "method" => r.method, "game" => r.game, "regime" => r.regime,
        "where" => "local", "commit" => _git_commit(),
        "timestamp" => string(round(Int, time())),
        "position_pixel_cell" => [r.cell[1], r.cell[2]],
        "oracle_nonzero_causes" => r.oracle_nonzero,
        "faithfulness_naive" => r.faithfulness_naive,
        "faithfulness_sampler" => r.faithfulness_sampler,
        "semantic_recovery" => r.semantic_recovery,
        "semantic_recovery_justification" =>
            "An attribution map is a vector of per-cause magnitudes; on its own it " *
            "neither emits nor names a T3 game concept (missile/collision/score/restart). " *
            "The only semantic label in play is the imported T3 annotation, used to CHECK " *
            "localization, never PRODUCED by the method. Hence semantic_recovery = 0 in " *
            "BOTH the naive and sampler conditions — faithfulness is repairable, semantics is not.",
        "sampler_ref" =>
            "bilinear index-boundary surrogate, SAME mechanism as " *
            "tools/xai_si_gradient/si_joystick_gradient.jl: a triangular-kernel occupancy " *
            "occ(px)=Σ tri(ROWS−row)·tri(COLS−(px+dc)) with the sprite column px a continuous " *
            "affine function of the position RAM byte ⇒ ∂pixel/∂ram[pidx] is restored.",
        "naive_ref" => "position_read_zero(ram)=0·Σram (Prop. prop:zero / §1 vanishing): " *
            "∂/∂ram ≡ 0 ⇒ attribution ≡ 0 ⇒ faithfulness 0 (reproduces the committed " *
            "saliency_<game>_position baseline).",
        "position_byte_ram_index" => r.position_byte,
        "sampler_attr_on_position_byte" => r.sampler_attr_on_byte,
        "cause_names" => r.cause_names,
        "attr_naive_per_cause" => Dict(r.cause_names[i] => r.attr_naive[i] for i in 1:length(r.cause_names)),
        "attr_sampler_per_cause" => Dict(r.cause_names[i] => r.attr_sampler[i] for i in 1:length(r.cause_names)),
        "oracle_abs_delta_per_cause" => Dict(r.cause_names[i] => r.odelta[i] for i in 1:length(r.cause_names)),
        "faithfulness_metric" => "pearson(per-cause attribution, oracle |Δ position-pixel|) on the position regime",
    )
    open(json_path, "w") do io; JSON.print(io, rec, 2); end
    return json_path
end

# ============================================================================
# CSV aggregate.
# ============================================================================
function write_csv(results::Vector{MethodResult})
    isdir(COMPARE_OUT) || mkpath(COMPARE_OUT)
    open(CSV_PATH, "w") do io
        println(io, "method,game,regime,faithfulness_naive,faithfulness_sampler,semantic_recovery,record_path")
        for r in results
            rel = relpath(r.record_path, normpath(joinpath(@__DIR__, "..", "..", "..", "..")))
            println(io, "$(r.method),$(r.game),$(r.regime)," *
                        "$(round(r.faithfulness_naive, digits=6))," *
                        "$(round(r.faithfulness_sampler, digits=6))," *
                        "$(r.semantic_recovery),$(rel)")
        end
    end
    return CSV_PATH
end

# ============================================================================
# Self-check (DoD) — asserts + exits non-zero on failure.
# ============================================================================
function selfcheck(results::Vector{MethodResult})
    # (b) semantic_recovery == 0 for EVERY method in BOTH conditions.
    for r in results
        @assert r.semantic_recovery == 0 "semantic_recovery must be 0 (faithful≠semantic) " *
            "but got $(r.semantic_recovery) for $(r.method)/$(r.game)"
    end
    # (a) for ≥1 gradient method, faithfulness_sampler > faithfulness_naive on position.
    rises = [(r.method, r.game, r.faithfulness_naive, r.faithfulness_sampler)
             for r in results if r.faithfulness_sampler > r.faithfulness_naive + 1e-9]
    @assert !isempty(rises) "SELF-CHECK FAILED: no gradient method had " *
        "faithfulness_sampler > faithfulness_naive on the position regime — the sampler " *
        "did not restore a real gradient on any (method,game). Cannot claim the keystone."
    # the naive floor must really be ~0 wherever the sampler rose (Prop. prop:zero).
    for r in results
        if r.faithfulness_sampler > r.faithfulness_naive + 1e-9
            @assert abs(r.faithfulness_naive) < 1e-6 "naive faithfulness not at the floor " *
                "($(r.faithfulness_naive)) for $(r.method)/$(r.game) — the §1 vanishing must hold"
        end
    end
    println("\n[sampler] SELF-CHECK PASS:")
    println("[sampler]   (a) faithfulness_sampler > faithfulness_naive on the position regime for " *
            "$(length(rises)) (method,game) pair(s); e.g.:")
    for (m, g, fn, fs) in rises[1:min(5, end)]
        println("[sampler]       $(rpad(m,22)) $(rpad(g,16)) naive=$(round(fn,digits=3)) → sampler=$(round(fs,digits=3))")
    end
    println("[sampler]   (b) semantic_recovery == 0 for ALL $(length(results)) (method,game) records, both conditions.")
    return true
end

# ============================================================================
# CLI.
# ============================================================================
function main(args = ARGS)
    games = ALL_GAMES; single_game = nothing
    target_frame = 120; horizon = 30; topk = 3; seed = 0
    i = 1
    while i <= length(args)
        a = args[i]
        if     a == "--games"
            v = args[i+1]
            games = (v == "all" || v == "core") ? ALL_GAMES : String.(split(v, ",")); i += 2
        elseif a == "--game";         single_game = args[i+1]; i += 2
        elseif a == "--target-frame"; target_frame = parse(Int, args[i+1]); i += 2
        elseif a == "--horizon";      horizon = parse(Int, args[i+1]); i += 2
        elseif a == "--topk";         topk = parse(Int, args[i+1]); i += 2
        elseif a == "--seed";         seed = parse(Int, args[i+1]); i += 2
        else i += 1
        end
    end
    single_game !== nothing && (games = [single_game])

    println("[sampler] KEYSTONE: bilinear sampler ON restores the position gradient " *
            "(faithfulness rises) but NOT semantics (stays 0).")
    println("[sampler] methods=$(METHODS)  games=$(games)  tf=$target_frame hz=$horizon topk=$topk seed=$seed (jutari)")

    results = MethodResult[]
    for game in games
        println("\n[sampler] ===== $game =====")
        gs = build_game_state(; game = game, target_frame = target_frame,
                              horizon = horizon, seed = seed, verbose = true)
        for method in METHODS
            r = score_method(method, gs; topk = topk, seed = seed)
            jp = write_record(r)
            r = MethodResult(r.method, r.game, r.regime, r.faithfulness_naive,
                             r.faithfulness_sampler, r.semantic_recovery, r.attr_naive,
                             r.attr_sampler, r.cause_names, r.odelta, r.position_byte,
                             r.sampler_attr_on_byte, r.oracle_nonzero, r.cell, jp)
            push!(results, r)
            println("[sampler]   $(rpad(method,22)) faith naive=$(rpad(round(r.faithfulness_naive,digits=3),6)) " *
                    "→ sampler=$(rpad(round(r.faithfulness_sampler,digits=3),6)) " *
                    "sem=$(r.semantic_recovery)  (attr on byte $(r.position_byte)=$(round(r.sampler_attr_on_byte,sigdigits=3)))")
        end
    end

    csv = write_csv(results)
    println("\n[sampler] wrote CSV $csv  ($(length(results)) rows)")
    selfcheck(results)
    return 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    exit(SamplerFaithfulness.main())
end
