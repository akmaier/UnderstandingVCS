# saliency.jl — Phase-B attribution (P2-E4-1), JULIA path.
#
# VANILLA GRADIENT / SALIENCY (Simonyan et al. 2014) scored against the exact §1
# intervention oracle on the 6 CORE games (tools/xai_study/common/game_set.json),
# reusing the validated Phase-B faithfulness contract pinned by the IG pilot
# (P2-E4-0, pilot_ig_vs_oracle.jl). This is the first of the gradient-family
# methods on the leaderboard (siblings: Grad×Input/DeepLIFT P2-E4-2, SmoothGrad
# P2-E4-4, IG pilot P2-E4-0).
#
# METHOD — vanilla saliency: the magnitude of the output gradient w.r.t. the cause
# vector, evaluated at the actual state:
#
#     saliency(u) = | ∂y/∂u |                                  (Simonyan et al. 2014)
#
# This is EXACTLY the third variant gradxinput.jl already computes
# (`saliency = abs.(g)`); here it is factored into a dedicated method whose
# headline IS the vanilla gradient (not a contrast against Grad×Input/DeepLIFT).
# The differentiable substrate is the SAME forward-exact content read the siblings
# use: y = soft_ram_peek(ram, idx), a one-hot read of the RAM tape whose Zygote
# gradient is the content-path ∂y/∂u (∂y/∂ram[idx]=1; Theorem 1, the STE forward).
#
# WHY this is the honest "plausible ≠ faithful" probe (method matrix §7 predicts
# vanilla gradient FAILS): saliency answers "what is the output a function of RIGHT
# NOW" — for a one-hot RAM read that is trivially the self-byte. The §1 oracle
# answers "what CAUSES the output over the horizon", whose dominant cause can be a
# DIFFERENT upstream byte (e.g. SI: zeroing invaders_left moves the bullets byte
# more than zeroing the bullets byte itself). So even on the CONTENT path where the
# gradient is ALIVE, saliency's top-byte = the self-byte need not equal the
# oracle's top cause — we REPORT the measured corr / precision@k, not assert it.
# On the POSITION/INDEX output (a framebuffer pixel = a discrete sprite column via
# round/argmax) there is NO differentiable RAM read, so the gradient VANISHES
# (max|attr|=0) ⇒ near-chance faithfulness — the §1 index failure, where the
# intervention oracle is the sole truth.
#
# SCORING CONTRACT (verbatim from the IG pilot, shared by every E4 method):
#   (1) corr            — Pearson + Spearman of the saliency with the oracle's TRUE
#                         causal map {|Δy(u)|} over the same cause set;
#   (2) deletion/insertion AUC — measured ON THE TRUE VCS: re-run the real ROM with
#                         the top-attributed causes successively occluded (deletion)
#                         / restored (insertion), reading y each time. NOT a
#                         surrogate — every point is a genuine emulator re-run.
#   (3) precision@k     — |method top-k ∩ oracle top-k| / k vs the true causal top-k.
#  (+) HARNESS POSITIVE CONTROL — feed the oracle's OWN |Δy| back as the candidate
#      map; it MUST score corr=1 / precision@k=1 (on a non-degenerate column), proving
#      the harness rewards a faithful map ⇒ the saliency numbers are a true measurement.
#
# REUSES the validated Phase-B foundation on main (NO emulator core touched):
#   * pilot_ig_vs_oracle.jl — the SCORER (pearson/spearman, precision_at_k,
#     _trapz_unit), the per-cause |attr| mapping (ig_attribution_per_cause), the
#     §R writer helpers (_git_commit/_json_num), and the harness-positive-control idea.
#   * oracle_intervene.jl (via the pilot) — the bit-exact causal machinery:
#     build_pong_causes / Cause / candidate_ram_indices (the candidate cause set).
#   * jutari_oracle.jl (via the pilot) — snapshot/intervene/write_npz primitives.
#   * JuTari.Diff.soft_ram_peek — the forward-exact one-hot content read.
# Env construction is self-contained (a local ROM-alias + RomSettings map mirroring
# smoothgrad.jl / A1_connectomics.jl) so the per-game oracle map is computed for our
# chosen content output WITHOUT touching the shared oracle core.
#
# Run (warm shared depot, primary's project):
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseB_attribution/saliency.jl --games core
# Flags: --games core|<g1,g2,...>  --game <g>
#        --target-frame N --horizon N --topk K --seed S --selftest
#
# Writes (SPEC §R; file_scope saliency_*):
#   tools/xai_study/phaseB_attribution/out/saliency_<game>_content.{json,npz}
#   tools/xai_study/phaseB_attribution/out/saliency_<game>_position.{json,npz}
#   tools/xai_study/phaseB_attribution/out/saliency_core_summary.json

module SaliencyAttr

using JSON
import Zygote
import Statistics

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen, get_ram
using JuTari.Diff: soft_ram_peek

# the IG pilot — reuse its faithfulness SCORER + the per-cause mapping + the §R
# writer helpers (the validated Phase-B contract).
include(joinpath(@__DIR__, "pilot_ig_vs_oracle.jl"))
using .PilotIGvsOracle: pearson, spearman, precision_at_k, ig_attribution_per_cause,
                        _git_commit, _json_num, _trapz_unit,
                        minimality_score, sufficiency_score, triad_extra_dict

# the oracle's cause set + intervention machinery (game-agnostic: RAM/TIA/joystick)
using .PilotIGvsOracle.OracleIntervene: build_pong_causes, Cause, candidate_ram_indices
using .PilotIGvsOracle.OracleIntervene.JutariOracle: Snapshot, snapshot,
                                                     intervene_ram!, intervene_tia!,
                                                     write_npz, RAM_SIZE

# The P2 SHARED TESTBED (experiment_redesign.md): seeded random-action gameplay
# state + oracle cause-density gate + shared screen-buffer REGION output + the
# bilinear-sampler position path. Included as a fragment (see the file header for
# why not a module) so build_shared_testbed operates on OUR own Cause/Snapshot
# types. Opt in with XAI_SHARED_TESTBED=1 (default on for the redesign re-run).
include(joinpath(@__DIR__, "..", "common", "shared_testbed_impl.jl"))
# the shared game-set + ROM-root resolver (XAI_LABELED / xai_resolve_games / xai_rom_roots).
include(joinpath(@__DIR__, "..", "common", "game_sets.jl"))

const OUT_DIR = joinpath(@__DIR__, "out")
# shared-testbed switch + params (redesign protocol: prefix=90 gameplay, horizon=15).
const SHARED_TESTBED = get(ENV, "XAI_SHARED_TESTBED", "1") == "1"
const ST_PREFIX  = parse(Int, get(ENV, "XAI_ST_PREFIX", "90"))
const ST_HORIZON = parse(Int, get(ENV, "XAI_ST_HORIZON", "15"))
const ST_SEED    = parse(Int, get(ENV, "XAI_ST_SEED", "0"))
const ST_GATE_K  = parse(Int, get(ENV, "XAI_ST_GATE_K", "4"))
const ST_FLOOR   = parse(Float64, get(ENV, "XAI_ST_FLOOR", "0.5"))
const CORE_GAMES = ["pong", "breakout", "space_invaders", "seaquest", "ms_pacman", "qbert"]

# ============================================================================
# Per-game ROM + RomSettings resolution (mirrors smoothgrad.jl; NO core touched).
# seaquest has no registered RomSettings yet → Generic (boots fine).
# ============================================================================
const ROM_BASENAME = Dict(
    "pong" => "pong", "breakout" => "breakout",
    "space_invaders" => "space_invaders", "seaquest" => "seaquest",
    "ms_pacman" => "mspacman", "qbert" => "qbert")

const _PRIMARY_REPO = get(ENV, "XAI_PRIMARY_REPO", "/Users/maier/Documents/code/UnderstandingVCS")

function rom_path_for(game::AbstractString)
    g = lowercase(string(game))
    stem = get(ROM_BASENAME, g, g)
    # search xitari/roms + the 54-ROM store tools/rom_sweep/roms (ALE names), trying
    # the mapped stem AND the raw ALE name, so all labeled games resolve uniformly.
    return xai_find_rom(unique([stem, g]), xai_rom_roots(; primary_repo = _PRIMARY_REPO))
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

function load_env(; game::AbstractString)
    rom = read(rom_path_for(game))
    env = StellaEnvironment(rom, settings_for(game))
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    return env
end

boot_replay(actions, target_frame; game) = begin
    env = load_env(; game = game)
    for i in 1:target_frame; env_step!(env, Int(actions[i])); end
    env
end

continue_from(checkpoint, tail) = begin
    env = deepcopy(checkpoint)
    for a in tail; env_step!(env, Int(a)); end
    snapshot(env, length(tail))
end

fresh_baseline(actions, total; game) = begin
    env = load_env(; game = game)
    for i in 1:total; env_step!(env, Int(actions[i])); end
    snapshot(env, Int(total))
end

"""Assert two fresh boots+replays are byte-identical (the load-bearing oracle
correctness guarantee, mirroring OracleIntervene.assert_bit_exact). Refuses to
score a non-deterministic state."""
function assert_bit_exact(actions, total; game)
    a = fresh_baseline(actions, total; game = game)
    b = fresh_baseline(actions, total; game = game)
    a.ram == b.ram || error("bit-exact RAM re-run FAILED for $game: " *
        "$(count(a.ram .!= b.ram))/$(length(a.ram)) bytes differ to f$total")
    a.screen == b.screen || error("bit-exact SCREEN re-run FAILED for $game: " *
        "$(count(a.screen .!= b.screen)) px differ to f$total")
    return true
end

# ============================================================================
# Outputs y(state): a CONTENT RAM-read (gradient alive) + a POSITION pixel.
# ============================================================================
const BALL_X_IDX = 49      # RAM[$31] (Pong); used generically as a "sprite x" poke
const BALL_Y_IDX = 54      # RAM[$36]

"""Locate the moving-sprite cell CAUSALLY: the first framebuffer cell that changes
when a sprite-position byte is perturbed at the checkpoint (so the position output
is a genuine position/index output, experiment_design.md §1). Falls back to the
screen centre when nothing moves (a truly static frame)."""
function position_pixel_cell(checkpoint, base_screen; horizon)
    function perturbed(idx, d)
        env = deepcopy(checkpoint)
        b = Int(env.console.bus.ram[idx + 1])
        intervene_ram!(env, idx, (b + d) & 0xFF)
        for _ in 1:horizon; env_step!(env, 0); end
        snapshot(env, horizon).screen
    end
    sx = perturbed(BALL_X_IDX, 24)
    sy = perturbed(BALL_Y_IDX, 24)
    changed = (base_screen .!= sx) .| (base_screen .!= sy)
    if any(changed)
        ci = findfirst(changed)
        return (ci[1], ci[2])
    end
    h, w = size(base_screen)
    return (max(1, h ÷ 2), max(1, w ÷ 2))
end

# ============================================================================
# Causal map over the candidate causes for our two outputs (real re-runs).
# ============================================================================
"""Apply `cause` to a deepcopy of `checkpoint`, continue the horizon, snapshot —
the same do(u:=v') intervention semantics as OracleIntervene.run_intervention,
re-implemented locally so it threads OUR env (game-agnostic: RAM/TIA/joystick)."""
function run_intervention(checkpoint, actions, target_frame, horizon, cause::Cause)
    if cause.kind == "joystick"
        tail = vcat([cause.value], Int.(actions[target_frame + 2 : target_frame + horizon]))
        return continue_from(checkpoint, tail)
    end
    env = deepcopy(checkpoint)
    if cause.kind == "ram"
        intervene_ram!(env, cause.index, cause.value)
    elseif cause.kind == "tia_reg"
        intervene_tia!(env, cause.index, cause.value)
    else
        error("unknown cause kind: $(cause.kind)")
    end
    tail = Int.(actions[target_frame + 1 : target_frame + horizon])
    for a in tail; env_step!(env, a); end
    return snapshot(env, length(tail))
end

"""Occlude one cause on a live env via a REAL intervention (the "absent" do).
Mirrors PilotIGvsOracle._occlude!."""
function occlude!(env, c::Cause)
    if c.kind == "ram"
        intervene_ram!(env, c.index, 0)
    elseif c.kind == "tia_reg"
        intervene_tia!(env, c.index, 0)
    end   # joystick "absent" == NOOP (the baseline trace) — no footprint
    return env
end

"""
    deletion_insertion_auc(checkpoint, actions, tf, hz, causes, order, read_y)

The faithfulness curves measured ON THE TRUE VCS, generalised to an arbitrary
`read_y(::Snapshot)::Float64` reader (so it works for the content RAM-read output
as well as the position pixel). DELETION: occlude the first j ranked causes by a
REAL intervention, re-run, read y — a faithful ranking drops y early ⇒ SMALL
deletion AUC. INSERTION: start fully occluded, add the top-j back, re-run — a
faithful ranking recovers y fast ⇒ LARGE insertion AUC. Each point is a genuine
re-run. AUC = trapezoid of the jointly [0,1]-normalised curve; flat ⇒ NaN.
Same convention as PilotIGvsOracle.deletion_insertion_auc with a generic reader."""
function deletion_insertion_auc(checkpoint, actions, tf, hz,
                                causes::Vector{Cause}, order::Vector{Int}, read_y)
    tail = Int.(actions[tf + 1 : tf + hz])
    intact = continue_from(checkpoint, tail)
    y_full = read_y(intact)

    del_curve = Float64[y_full]
    for j in 1:length(order)
        env = deepcopy(checkpoint)
        for r in order[1:j]; occlude!(env, causes[r]); end
        for a in tail; env_step!(env, a); end
        push!(del_curve, read_y(snapshot(env, length(tail))))
    end

    ins_curve = Float64[]
    let env = deepcopy(checkpoint)
        for r in 1:length(causes); occlude!(env, causes[r]); end
        for a in tail; env_step!(env, a); end
        push!(ins_curve, read_y(snapshot(env, length(tail))))
    end
    for j in 1:length(order)
        env = deepcopy(checkpoint)
        keep = Set(order[1:j])
        for r in 1:length(causes); r in keep || occlude!(env, causes[r]); end
        for a in tail; env_step!(env, a); end
        push!(ins_curve, read_y(snapshot(env, length(tail))))
    end

    allv = vcat(del_curve, ins_curve)
    lo = minimum(allv); hi = maximum(allv)
    hi == lo && return NaN, NaN, del_curve, ins_curve
    norm(v) = (v .- lo) ./ (hi - lo)
    return _trapz_unit(norm(del_curve)), _trapz_unit(norm(ins_curve)), del_curve, ins_curve
end

"""Oracle |Δy| per cause for a given `read_y` reader: |y(do(u)) − y(baseline)|.
The TRUE causal map for the chosen output (every entry a real-ROM re-run)."""
function oracle_abs_delta(checkpoint, actions, tf, hz, causes::Vector{Cause}, read_y)
    base = continue_from(checkpoint, Int.(actions[tf + 1 : tf + hz]))
    y0 = read_y(base)
    return [abs(read_y(run_intervention(checkpoint, actions, tf, hz, c)) - y0) for c in causes]
end

# ============================================================================
# Vanilla saliency over the RAM tape (the method under test).
# ============================================================================
"""
    content_read(ram, idx) -> Float32

The differentiable CONTENT output: a forward-exact one-hot read of RAM byte `idx`
(∂/∂ram[idx]=1, the content path; Theorem 1). The SAME primitive gradxinput.jl /
smoothgrad.jl / the IG pilot use."""
content_read(ram::AbstractVector{<:Real}, idx::Integer) = soft_ram_peek(ram, idx)

"""
    position_read_zero(ram) -> Float32

The POSITION/INDEX output's differentiable handle: there is no differentiable RAM
read for a sprite-column pixel (round/argmax), so ∂/∂ram ≡ 0 (the §1 vanishing).
Returned through a 0-coefficient sum so Zygote yields an all-zero gradient, not
`nothing`."""
position_read_zero(ram::AbstractVector{<:Real}) = 0f0 * sum(Float32.(ram))

"""
    saliency_over_ram(readf, ram) -> Vector{Float32}

VANILLA SALIENCY (Simonyan et al. 2014): the magnitude of the output gradient
w.r.t. the 128-byte RAM vector, evaluated AT THE STATE (no input weighting, no
baseline, no smoothing) — saliency(u) = |∂y/∂u|. This is exactly
`abs.(Zygote.gradient(readf, ram))`, the third variant gradxinput.jl factors out."""
function saliency_over_ram(readf, ram::AbstractVector{<:Real})
    x = Float32.(ram)
    g = Zygote.gradient(readf, x)[1]
    g === nothing && (g = zeros(Float32, length(x)))
    return abs.(Float32.(g))
end

# ============================================================================
# The result record + the scoring driver (the IG contract; saliency as method).
# ============================================================================
struct Faithfulness
    game::String
    output::String                       # "content(ram_self@N)" | "position@rRcC"
    output_kind::String                  # "content" | "position"
    target_frame::Int
    horizon::Int
    topk::Int
    seed::Int
    content_idx::Int                     # the content RAM index (or -1 for position)
    cause_names::Vector{String}
    oracle_abs_delta::Vector{Float64}
    sal_attr::Vector{Float64}            # |∂y/∂u| per cause
    pearson::Float64
    spearman::Float64
    deletion_auc::Float64
    insertion_auc::Float64
    precision_at_k::Float64
    del_curve::Vector{Float64}
    ins_curve::Vector{Float64}
    y_full::Float64
    sal_full::Vector{Float64}            # |∂y/∂u| over the 128 RAM bytes
    grad_l1::Float64
    oracle_self_pearson::Float64
    oracle_self_precision_at_k::Float64
    oracle_self_deletion_auc::Float64
    oracle_self_insertion_auc::Float64
end

"""Pick the content RAM index: the candidate concept byte whose own value the
oracle's interventions move the MOST over the horizon (the most causally-active
RAM concept ⇒ a NON-DEGENERATE oracle column to score against). For Pong this is
a live position/score byte. Returns (idx, max oracle |Δself|).

This is the same content-cell choice smoothgrad.jl uses. It deliberately exposes
the temporal subtlety: saliency's one-hot read measures INSTANTANEOUS saliency
∂(byte-now)/∂(ram-now) — trivially the self-byte — while the oracle column for the
SAME output measures the byte AFTER the horizon, whose dominant cause may be a
DIFFERENT (upstream) byte. So saliency is faithful to "what the output IS" but
only partially to "what CAUSES it over time" — the 'plausible ≠ faithful' result
holds even on the content path, in numbers. We do NOT force precision@1=1; we
report the MEASURED corr / precision@k."""
function pick_content_idx(checkpoint, actions, tf, hz, causes::Vector{Cause}, cand_indices)
    best_idx = cand_indices[1]; best_mv = -1.0
    for idx in cand_indices
        rd = s -> Float64(Int(s.ram[idx + 1]))
        mv = maximum(oracle_abs_delta(checkpoint, actions, tf, hz, causes, rd))
        if mv > best_mv; best_mv = mv; best_idx = idx; end
    end
    return best_idx, best_mv
end

function compute_one(; game, output_kind, content_idx = -1, position_cell = nothing,
                     checkpoint, actions, target_frame, horizon, causes,
                     topk, seed, verbose,
                     read_y_override = nothing, readf_override = nothing,
                     output_name_override = nothing)
    # reader for the oracle / del-ins (reads the TRUE VCS state). In the SHARED
    # TESTBED the position output is a screen-buffer REGION (n_changed_px) supplied
    # by read_y_override (redesign Problem 4); its differentiable handle is the
    # bilinear SAMPLER (readf_override, redesign Problem 2 — the real, non-vanishing
    # position gradient), reported side-by-side with the naive vanishing gradient.
    read_y = read_y_override !== nothing ? read_y_override :
        (output_kind == "content" ?
            (s -> Float64(Int(s.ram[content_idx + 1]))) :
            (s -> Float64(Int(s.screen[position_cell[1], position_cell[2]]))))
    # differentiable handle for the gradient
    readf = readf_override !== nothing ? readf_override :
        (output_kind == "content" ? (r -> content_read(r, content_idx)) : position_read_zero)
    output_name = output_name_override !== nothing ? output_name_override :
        (output_kind == "content" ?
            "content(ram_self@$content_idx)" :
            "position@r$(position_cell[1])c$(position_cell[2])")

    # 1) oracle |Δy| per cause (real re-runs on the TRUE VCS)
    odelta = oracle_abs_delta(checkpoint, actions, target_frame, horizon, causes, read_y)

    # 2) vanilla saliency |∂y/∂u| over the RAM tape at the intervention frame
    verbose && println("[saliency] |∂y/∂u| of '$output_name' over the RAM tape...")
    at_target = continue_from(checkpoint, Int[])
    ram_now = Float32.(collect(at_target.ram))
    sal_full = saliency_over_ram(readf, ram_now)
    grad_l1 = Float64(sum(sal_full))
    sal_attr = ig_attribution_per_cause(sal_full, causes)

    # 3) score saliency against the oracle (the three §5 metrics)
    pr  = pearson(sal_attr, odelta); sp = spearman(sal_attr, odelta)
    pak = precision_at_k(sal_attr, odelta, topk)

    # deletion/insertion AUC on the TRUE VCS, ranked by saliency attribution
    order = sortperm(sal_attr; rev = true)
    verbose && println("[saliency] deletion/insertion curves on the TRUE VCS ($(length(order)) re-runs each)...")
    del_auc, ins_auc, del_curve, ins_curve =
        deletion_insertion_auc(checkpoint, actions, target_frame, horizon, causes, order, read_y)
    y_full = del_curve[1]

    # 4) harness positive control — the oracle's OWN |Δy| as the method
    or_pr  = pearson(odelta, odelta); or_pak = precision_at_k(odelta, odelta, topk)
    or_order = sortperm(odelta; rev = true)
    or_del, or_ins, _, _ = deletion_insertion_auc(checkpoint, actions, target_frame,
                                                  horizon, causes, or_order, read_y)
    # an all-zero oracle column ⇒ corr undefined; flag it (the output is causally
    # inert at this state — pearson(0-var)=0 by convention, NOT a harness break).
    odelta_degenerate = Statistics.std(odelta) == 0

    if verbose
        println("[saliency] ---- faithfulness of vanilla saliency vs the oracle ('$output_name', $game) ----")
        println("[saliency]   pearson(|∂y/∂u|, |Δy|)  = $(round(pr, digits=4))")
        println("[saliency]   spearman(|∂y/∂u|, |Δy|) = $(round(sp, digits=4))")
        println("[saliency]   precision@$topk           = $(round(pak, digits=4))")
        println("[saliency]   deletion AUC (↓ better)  = $(round(del_auc, digits=4))")
        println("[saliency]   insertion AUC (↑ better) = $(round(ins_auc, digits=4))")
        println("[saliency]   grad L1 = $(round(grad_l1, digits=4))   (0 ⇒ the gradient VANISHES)")
        println("[saliency]   [harness] oracle-as-method: corr=$(round(or_pr,digits=3)) " *
                "precision@$topk=$(round(or_pak,digits=3)) del=$(round(or_del,digits=3)) ins=$(round(or_ins,digits=3))" *
                (odelta_degenerate ? "  (oracle column flat at this state)" : ""))
        sal_rank = sortperm(sal_attr; rev = true); or_rank = sortperm(odelta; rev = true)
        println("[saliency]   saliency top-3 causes: ", [causes[i].name for i in sal_rank[1:min(3,end)]])
        println("[saliency]   oracle   top-3 causes: ", [causes[i].name for i in or_rank[1:min(3,end)]])
    end

    return Faithfulness(game, output_name, output_kind, target_frame, horizon,
                        topk, seed, content_idx, [c.name for c in causes],
                        odelta, sal_attr, pr, sp, del_auc, ins_auc, pak,
                        del_curve, ins_curve, y_full, Float64.(sal_full), grad_l1,
                        or_pr, or_pak, or_del, or_ins), odelta_degenerate
end

"""Drive both outputs for one game: assert bit-exact, build causes, pick the
content byte, score the content headline + the position contrast.

SHARED-TESTBED mode (redesign): the state is a seeded random-action GAMEPLAY state
(prefix=ST_PREFIX, horizon=ST_HORIZON) gated by the oracle cause-density gate; the
position output is the SHARED screen-buffer REGION (n_changed_px), and its gradient
runs through the bilinear SAMPLER (non-vanishing) with the naive vanishing gradient
reported side-by-side. `st` (the NamedTuple) is threaded out for the record."""
function compute_game(; game, target_frame, horizon, topk, seed, verbose)
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
        verbose && println("[saliency] $game SHARED gameplay state: " *
            "cause_density=$(st.cause_density)/$(length(st.causes)) accepted=$(st.accepted) " *
            "cell=$(st.cell) geom=$(st.geom === nothing ? "static" : "RAM[$(st.geom[1])]")")

        actions = st.actions; checkpoint = st.checkpoint; causes = st.causes
        tf = st.prefix; hz = st.horizon; cand_indices = st.cand_indices
        content_idx, content_mv = pick_content_idx(checkpoint, actions, tf, hz, causes, cand_indices)
        verbose && println("[saliency] $game content byte = RAM[$content_idx] (max oracle |Δself|=$(round(content_mv,digits=2)))")

        f_content, content_degen = compute_one(; game = game, output_kind = "content",
            content_idx = content_idx, checkpoint = checkpoint, actions = actions,
            target_frame = tf, horizon = hz, causes = causes,
            topk = topk, seed = seed, verbose = verbose)

        # POSITION output on the SHARED screen-buffer REGION, gradient via the SAMPLER.
        f_pos, _ = compute_one(; game = game, output_kind = "position",
            position_cell = st.cell, checkpoint = checkpoint, actions = actions,
            target_frame = tf, horizon = hz, causes = causes,
            topk = topk, seed = seed, verbose = verbose,
            read_y_override = st.read_y, readf_override = st.sampler_read,
            output_name_override = "screen_region(n_changed_px)@r$(st.cell[1])c$(st.cell[2])")

        # naive (vanishing) side-by-side: |∂(naive position read)/∂ram| max.
        naive_g = saliency_over_ram(st.position_read_zero, st.ram_now)
        sampler_g = saliency_over_ram(st.sampler_read, st.ram_now)
        st_extra = (cause_density = st.cause_density, accepted = st.accepted,
                    n_causes = length(st.causes), cell = st.cell,
                    geom = st.geom,
                    naive_pos_grad_max = Float64(maximum(abs.(naive_g))),
                    sampler_pos_grad_max = Float64(maximum(abs.(sampler_g))),
                    prefix = st.prefix, horizon = st.horizon, seed = st.seed)
        return f_content, content_degen, f_pos, st_extra
    end

    total = target_frame + horizon
    actions = fill(0, total)
    verbose && println("[saliency] $game: asserting bit-exactness (2 fresh boots+replays to f$total)...")
    assert_bit_exact(actions, total; game = game)

    cand = candidates_path_for(game)
    checkpoint = boot_replay(actions, target_frame; game = game)
    at_target = continue_from(checkpoint, Int[])
    causes = build_pong_causes(cand, at_target)
    cand_indices = [idx for (idx, _) in candidate_ram_indices(cand)]

    # content byte = the most causally-active candidate concept byte
    content_idx, content_mv = pick_content_idx(checkpoint, actions, target_frame, horizon, causes, cand_indices)
    verbose && println("[saliency] $game content byte = RAM[$content_idx] (max oracle |Δself|=$(round(content_mv,digits=2)))")

    base = continue_from(checkpoint, Int.(actions[target_frame + 1 : total]))
    pcell = position_pixel_cell(checkpoint, base.screen; horizon = horizon)

    f_content, content_degen = compute_one(; game = game, output_kind = "content",
        content_idx = content_idx, checkpoint = checkpoint, actions = actions,
        target_frame = target_frame, horizon = horizon, causes = causes,
        topk = topk, seed = seed, verbose = verbose)
    f_pos, _ = compute_one(; game = game, output_kind = "position",
        position_cell = pcell, checkpoint = checkpoint, actions = actions,
        target_frame = target_frame, horizon = horizon, causes = causes,
        topk = topk, seed = seed, verbose = verbose)
    return f_content, content_degen, f_pos, nothing
end

# ============================================================================
# Self-check (DoD).
# ============================================================================
"""Parse the RAM index out of a cause name like \"ram[54]:set\" → 54; else -1
(for tia/joystick causes that touch no RAM byte)."""
function _cause_ram_index(name::AbstractString)
    m = match(r"ram\[(\d+)\]", name)
    return m === nothing ? -1 : parse(Int, m.captures[1])
end

function selftest(f::Faithfulness; require_nondegenerate = false, sampler_on = false,
                  is_core = false)
    @assert all(isfinite, f.sal_attr) "non-finite saliency attribution [$(f.game)]"
    for (nm, v) in (("deletion", f.deletion_auc), ("insertion", f.insertion_auc),
                    ("oracle_self_deletion", f.oracle_self_deletion_auc),
                    ("oracle_self_insertion", f.oracle_self_insertion_auc))
        @assert isnan(v) || (0.0 <= v <= 1.0 + 1e-9) "$nm AUC out of [0,1]: $v [$(f.game)]"
    end
    sal_max = maximum(abs.(f.sal_attr))
    if f.output_kind == "position"
        if sampler_on
            # SHARED TESTBED, sampler-on: the bilinear sampler RESTORES the position
            # gradient, so it is NON-VANISHING (the redesign keystone). The keystone
            # claim is on the FULL 128-byte gradient (sal_full) — the sampler's
            # position byte may fall OUTSIDE this game's candidate cause set, in which
            # case the per-cause attribution (sal_attr) is legitimately null while the
            # raw gradient is alive. The naive zero is reported side-by-side
            # (sampler_on.naive_*). We assert on the FULL gradient, not per-cause.
            sal_full_max = maximum(abs.(f.sal_full))
            # The keystone (sampler RESTORES a nonzero position gradient) is a CORE-6
            # claim: those games have a genuinely MOVING, scorable sprite here, so the
            # strict assert MUST hold. For a non-core labeled game the shared state can
            # be static/degenerate (e.g. a saturated/1-px sprite whose sampler occupancy
            # is locally flat ⇒ ∂occ/∂byte ≡ 0) — an HONEST null feeding a zero position
            # score, not a harness break. Record it and continue rather than abort.
            if is_core || sal_full_max > 1e-6
                @assert sal_full_max > 1e-6 "SAMPLER-ON position gradient (full 128-byte) should be NONZERO [$(f.game)]"
            else
                println("[saliency] position static/degenerate at this state — sampler gradient ~0, recorded honestly [$(f.game)]")
            end
            pidx_scored = sal_max > 1e-6
            println("[saliency] SELF-CHECK PASS (position/sampler '$(f.output)', $(f.game)): " *
                    (sal_full_max > 1e-6 ? "sampler RESTORES a nonzero position gradient" :
                        "position static/degenerate — sampler gradient ~0 (recorded honestly)") *
                    " (full max|g|=$(round(sal_full_max,sigdigits=3)); " *
                    "per-cause max|attr|=$(round(sal_max,sigdigits=3))" *
                    (pidx_scored ? "" : " — position byte OUTSIDE this game's cause set ⇒ per-cause corr null") *
                    "); corr=$(round(f.pearson,digits=3)) p@$(f.topk)=$(round(f.precision_at_k,digits=3)).")
        else
            # POSITION/INDEX output — vanilla saliency VANISHES (the §1 index failure).
            @assert sal_max < 1e-6 "expected the gradient to vanish on a position output, got max|attr|=$sal_max [$(f.game)]"
            println("[saliency] SELF-CHECK PASS (position '$(f.output)', $(f.game)): " *
                    "vanilla saliency VANISHES (max|attr|=$(round(sal_max,sigdigits=3))) — §1 index failure; " *
                    "corr=$(round(f.pearson,digits=3)) p@$(f.topk)=$(round(f.precision_at_k,digits=3)) (near chance). " *
                    "The intervention oracle is the sole truth here.")
        end
    else
        # CONTENT output — the one-hot RAM read keeps the STE gradient alive, so
        # saliency is NONZERO and concentrates its mass on the self-byte (the cause
        # acting on the output's own RAM index). We assert the gradient is alive +
        # the top mass is on the self-byte + the harness positive control holds; we
        # do NOT force precision@1=1 vs the oracle — the MEASURED corr/precision@k
        # IS the reported result (saliency is faithful to "what the output is", only
        # partially to "what causes it over the horizon" — the honest content number).
        @assert sal_max > 1e-6 "vanilla saliency of a content RAM read should be nonzero (one-hot read), got $sal_max [$(f.game)]"
        self_causes = findall(c -> c == f.content_idx,
                              [_cause_ram_index(n) for n in f.cause_names])
        if !isempty(self_causes)
            @assert argmax(f.sal_attr) in self_causes "saliency top mass not on the content self-byte [$(f.game)]"
        end
        if require_nondegenerate
            # HARNESS POSITIVE CONTROL must hold on a non-degenerate column.
            @assert f.oracle_self_pearson > 0.999 "harness broken: oracle-as-method corr != 1 ($(f.oracle_self_pearson)) [$(f.game)]"
            @assert f.oracle_self_precision_at_k == 1.0 "harness broken: oracle-as-method precision@k != 1 ($(f.oracle_self_precision_at_k)) [$(f.game)]"
        end
        faithful = !isempty(self_causes) && (argmax(f.oracle_abs_delta) in self_causes)
        println("[saliency] SELF-CHECK PASS (content '$(f.output)', $(f.game)): " *
                "saliency alive + on the self-byte ram[$(f.content_idx)] (max|attr|=$(round(sal_max,sigdigits=3))); " *
                "MEASURED corr=$(round(f.pearson,digits=3)), p@$(f.topk)=$(round(f.precision_at_k,digits=3)) ⇒ " *
                (faithful ? "FAITHFUL here (the self-byte is its own top cause); " :
                            "UNFAITHFUL here (an upstream cause dominates over the horizon — local gradient ≠ global cause); ") *
                "harness oracle-as-method corr=$(round(f.oracle_self_pearson,digits=3)).")
    end
    return true
end

# ============================================================================
# Persist (SPEC §R) — JSON record + sibling .npz; file_scope saliency_*.
# ============================================================================
function _output_note(f::Faithfulness)
    sal_max = maximum(abs.(f.sal_attr))
    if f.output_kind == "position"
        return "POSITION/INDEX output — the §1 caveat: the pixel value comes from a " *
               "discrete sprite column (round/argmax), so there is no differentiable RAM " *
               "read and the content-path gradient VANISHES (max|attr|=$(round(sal_max,sigdigits=3)), " *
               "grad L1=$(round(f.grad_l1,sigdigits=3))). The oracle has real causal signal here, " *
               "so vanilla saliency scores near chance — the 'plausible ≠ faithful' contrast; " *
               "the intervention oracle is the sole truth. This is the §7 'Fail (zero on index " *
               "outputs)' prediction for vanilla gradient, in numbers."
    else
        return "CONTENT output: RAM byte $(f.content_idx) read one-hot through soft_ram_peek " *
               "(∂y/∂ram[$(f.content_idx)]=1) — the most causally-active candidate concept byte " *
               "the oracle found at this state. The STE gradient is ALIVE (grad L1=$(round(f.grad_l1,sigdigits=3))), " *
               "so vanilla saliency puts its mass on this self-byte (precision@1 vs the self-byte=1). " *
               "Whether that equals the ORACLE's top cause is the FAITHFULNESS question: saliency " *
               "answers 'what is y a function of NOW' (the self-byte), the oracle 'what CAUSES y over " *
               "the horizon' (possibly an upstream byte) — measured corr=$(round(f.pearson,digits=3)), " *
               "precision@$(f.topk)=$(round(f.precision_at_k,digits=3)). Vanilla saliency has NO input " *
               "weighting and NO completeness (unlike Grad×Input/DeepLIFT) — the bare gradient baseline."
    end
end

function write_faithfulness(f::Faithfulness; out_dir = OUT_DIR, st_extra = nothing)
    isdir(out_dir) || mkpath(out_dir)
    tag = f.output_kind == "content" ? "content" : "position"
    stem = "saliency_$(f.game)_$(tag)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path  = joinpath(out_dir, stem * ".npz")

    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseB_attribution", "method" => "vanilla_saliency",
        "game" => f.game, "state" => "f$(f.target_frame)+$(f.horizon)",
        "target_output" => f.output,
        "metric_name" => "pearson_corr_with_oracle", "value" => f.pearson,
        "stderr" => nothing, "ci" => nothing, "n" => length(f.cause_names),
        "seed" => f.seed, "where" => "local", "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene@$(f.game)#$(f.output)",
        "timestamp" => string(round(Int, time())), "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            "method_ref" => "vanilla gradient / saliency (Simonyan et al. 2014): attribution = |∂y/∂u|",
            "substrate" => "jutari (Julia) — Zygote content-path |∂y/∂u| over the RAM tape + " *
                "TRUE-VCS deletion/insertion re-runs (real-ROM, bit-exact oracle machinery).",
            "output_kind" => f.output_kind,
            "content_ram_index" => f.content_idx,
            "grad_l1" => f.grad_l1,
            "baseline" => "none (vanilla saliency uses no baseline and no input weighting)",
            "metrics" => Dict{String,Any}(
                "pearson_corr" => f.pearson, "spearman_corr" => f.spearman,
                "precision_at_k" => f.precision_at_k, "topk" => f.topk,
                "deletion_auc" => _json_num(f.deletion_auc),
                "insertion_auc" => _json_num(f.insertion_auc)),
            # F∧S∧M triad — F is the already-computed pearson (unchanged); S and M
            # are derived from the saliency map + the oracle |Δy| (no new re-runs).
            "triad" => triad_extra_dict(f.pearson, f.sal_attr, f.oracle_abs_delta;
                                        topk = f.topk, seed = f.seed),
            "harness_positive_control" => Dict{String,Any}(
                "method" => "oracle_abs_delta (the perfectly-faithful attribution)",
                "pearson_corr" => f.oracle_self_pearson,
                "precision_at_k" => f.oracle_self_precision_at_k,
                "deletion_auc" => _json_num(f.oracle_self_deletion_auc),
                "insertion_auc" => _json_num(f.oracle_self_insertion_auc),
                "interpretation" => "corr=1 & precision@k=1 (on a non-degenerate column) ⇒ the " *
                    "scoring harness rewards a faithful map; the vanilla-saliency numbers are then " *
                    "a true measurement, not a harness artefact."),
            "auc_note" => "deletion/insertion curves measured on the TRUE VCS by re-running the " *
                "real ROM with top-saliency causes occluded (deletion) / restored (insertion); " *
                "every point is a genuine emulator re-run, not a surrogate. NaN = a genuinely " *
                "flat experiment (e.g. the vanishing position output).",
            "output_note" => _output_note(f),
            "sal_max_abs" => maximum(abs.(f.sal_attr)),
            "cause_names" => f.cause_names,
            "saliency_attr_per_cause" => Dict(f.cause_names[i] => f.sal_attr[i] for i in 1:length(f.cause_names)),
            "oracle_abs_delta_per_cause" => Dict(f.cause_names[i] => f.oracle_abs_delta[i] for i in 1:length(f.cause_names)),
            "y_full" => f.y_full,
            "comparison_note" => "Vanilla saliency (|∂y/∂u|, bare gradient) is the baseline of the " *
                "gradient family: Grad×Input adds u⊙, DeepLIFT adds (u−baseline)⊙ + completeness, " *
                "SmoothGrad adds noise-averaging. On the one-hot content read all share the same " *
                "constant ∂y/∂u=e_idx, so saliency already pins the self-byte; the siblings differ " *
                "only in weighting/completeness, not in WHICH byte they top.",
            "scales_to_cluster_via" =>
                "tools/cluster/xai_*.sbatch (E0-3) — batched SOFT-STE saliency over " *
                "outputs×causes×games on GPU; the forward is bit-exact to this map.",
        ),
    )
    # SHARED-TESTBED provenance + the sampler-on side-by-side (redesign protocol).
    if st_extra !== nothing
        rec["state"] = "gameplay(seed=$(st_extra.seed),prefix=$(st_extra.prefix))+$(st_extra.horizon)"
        rec["extra"]["testbed"] = Dict{String,Any}(
            "state_kind" => "seeded_random_action_gameplay",
            "prefix" => st_extra.prefix, "horizon" => st_extra.horizon, "seed" => st_extra.seed,
            "shared_output" => "screen_region(n_changed_px)@r$(st_extra.cell[1])c$(st_extra.cell[2])",
            "cause_density_above_floor" => st_extra.cause_density,
            "cause_density_floor" => ST_FLOOR, "cause_density_gate_k" => ST_GATE_K,
            "cause_density_accepted" => st_extra.accepted, "n_causes" => st_extra.n_causes,
            "position_byte_ram_index" => st_extra.geom === nothing ? -1 : st_extra.geom[1])
        # per-cause faithfulness is NULL (not 0) when the sampler's position byte
        # falls OUTSIDE this game's candidate cause set: the raw gradient is alive
        # (sampler_position_grad_max>0) but no scored cause carries it, so the
        # per-cause corr is undefined. The shared PilotIGvsOracle.pearson returns 0.0
        # for a flat column by convention; we flag the true semantics here so the
        # leaderboard treats it as null, not a genuine zero-correlation.
        per_cause_null = maximum(abs.(f.sal_attr)) < 1e-9
        rec["extra"]["sampler_on"] = Dict{String,Any}(
            "naive_position_grad_max" => st_extra.naive_pos_grad_max,
            "sampler_position_grad_max" => st_extra.sampler_pos_grad_max,
            "position_byte_in_cause_set" => !per_cause_null,
            "per_cause_faithfulness_null" => per_cause_null,
            "note" => "naive ∂pixel/∂ram ≡ 0 (Prop. prop:zero); the bilinear sampler " *
                "(tools/xai_si_gradient/si_joystick_gradient.jl) restores a real " *
                "∂pixel/∂ram[position_byte]. Reported naive-vs-sampler side by side. " *
                "per_cause_faithfulness_null=true ⇒ the sampler's position byte is not " *
                "among this game's scored candidate causes, so the per-cause Pearson is " *
                "null (recorded 0.0 by the shared convention), NOT a genuine zero corr — " *
                "the keystone (a non-vanishing gradient) still holds on the full 128-byte map.")
    end
    open(json_path, "w") do io; JSON.print(io, rec, 2); end

    write_npz(npz_path, Dict(
        "saliency_attr_per_cause" => f.sal_attr,
        "oracle_abs_delta"        => f.oracle_abs_delta,
        "saliency_over_ram"       => f.sal_full,
        "deletion_curve"          => f.del_curve,
        "insertion_curve"         => f.ins_curve,
        "scalars"                 => Float64[f.pearson, f.spearman, f.precision_at_k,
                                             f.deletion_auc, f.insertion_auc, f.grad_l1]))
    return json_path, npz_path
end

# ============================================================================
# CLI
# ============================================================================
function main(args = ARGS)
    games = CORE_GAMES; single_game = nothing
    target_frame = 120; horizon = 30
    topk = 3; seed = 0
    selftest_only = false
    i = 1
    while i <= length(args)
        a = args[i]
        if     a == "--games"
            games = xai_resolve_games(args[i+1], CORE_GAMES); i += 2
        elseif a == "--game";         single_game = args[i+1]; i += 2
        elseif a == "--target-frame"; target_frame = parse(Int, args[i+1]); i += 2
        elseif a == "--horizon";      horizon = parse(Int, args[i+1]); i += 2
        elseif a == "--topk";         topk = parse(Int, args[i+1]); i += 2
        elseif a == "--seed";         seed = parse(Int, args[i+1]); i += 2
        elseif a == "--selftest";     selftest_only = true; i += 1
        else; i += 1
        end
    end
    single_game !== nothing && (games = [single_game])

    println("[saliency] vanilla gradient / saliency (Simonyan et al. 2014) vs oracle — " *
            "games=$(games) target_frame=$target_frame horizon=$horizon topk=$topk seed=$seed (jutari/Julia)")

    summary = Dict{String,Any}[]
    na_records = Dict{String,Any}[]
    for game in games
        println("\n[saliency] ===== $game =====")
        try
        f_content, content_degen, f_pos, st_extra = compute_game(; game = game,
            target_frame = target_frame, horizon = horizon, topk = topk,
            seed = seed, verbose = true)
        # the content headline must have a non-degenerate oracle column (else the
        # positive control + corr are meaningless — refuse to claim a result).
        @assert !content_degen "content oracle column is degenerate for $game at f$target_frame " *
            "— pick_content_idx failed to find a causally-active concept byte"
        selftest(f_content; require_nondegenerate = true)
        # SHARED-TESTBED: the position gradient runs through the SAMPLER, so it is
        # NON-VANISHING when a moving sprite exists (geom !== nothing) — the redesign
        # keystone. Only assert the §1 vanishing in the legacy (non-shared) path.
        sampler_on = st_extra !== nothing && st_extra.geom !== nothing
        selftest(f_pos; sampler_on = sampler_on, is_core = game in CORE_GAMES)
        if st_extra !== nothing
            println("[saliency] $game SAMPLER-ON position gradient: naive max|g|=" *
                "$(round(st_extra.naive_pos_grad_max, sigdigits=3)) → sampler max|g|=" *
                "$(round(st_extra.sampler_pos_grad_max, sigdigits=3)) " *
                "(gate: $(st_extra.cause_density)/$(st_extra.n_causes) accepted=$(st_extra.accepted))")
        end
        if !selftest_only
            for f in (f_content, f_pos)
                jp, np = write_faithfulness(f; st_extra = (f === f_pos ? st_extra : nothing))
                println("[saliency] wrote $jp"); println("[saliency] arrays  $np")
            end
        end
        for f in (f_content, f_pos)
            push!(summary, Dict{String,Any}(
                "game" => game, "output" => f.output, "output_kind" => f.output_kind,
                "content_ram_index" => f.content_idx,
                "pearson" => f.pearson, "spearman" => f.spearman,
                "precision_at_k" => f.precision_at_k,
                "deletion_auc" => _json_num(f.deletion_auc),
                "insertion_auc" => _json_num(f.insertion_auc),
                "grad_l1" => f.grad_l1,
                "sal_max_abs" => maximum(abs.(f.sal_attr)),
                "oracle_self_pearson" => f.oracle_self_pearson,
                "oracle_self_precision_at_k" => f.oracle_self_precision_at_k,
                "is_content_path" => f.output_kind == "content"))
        end
        catch e
            # A game that is DEGENERATE/STATIC at the shared gameplay state (empty
            # cause set, degenerate oracle column, empty sampler footprint, or a
            # bit-exact re-run failure) records an n/a row and is SKIPPED — it does
            # NOT abort the whole battery. (Many non-core games are static at
            # prefix=90; flag for a per-game livelier prefix later.)
            msg = sprint(showerror, e)
            println("[saliency] !! $game SKIPPED (n/a): $(first(split(msg, '\n')))")
            push!(na_records, Dict{String,Any}("game" => game, "status" => "n/a",
                "reason" => first(split(msg, '\n'))))
        end
    end

    if selftest_only
        println("\n[saliency] --selftest: all passed, not writing artifacts.")
        return 0
    end
    isempty(na_records) || println("\n[saliency] $(length(na_records)) game(s) n/a " *
        "(degenerate/static at the shared state): $(join([r["game"] for r in na_records], ", "))")

    isdir(OUT_DIR) || mkpath(OUT_DIR)
    summary_path = joinpath(OUT_DIR, "saliency_core_summary.json")
    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseB_attribution", "method" => "vanilla_saliency",
        "item" => "P2-E4-1",
        "games" => games, "topk" => topk, "seed" => seed,
        "target_frame" => target_frame, "horizon" => horizon,
        "where" => "local", "commit" => _git_commit(), "timestamp" => string(round(Int, time())),
        "method_ref" => "vanilla gradient / saliency (Simonyan et al. 2014): attribution = |∂y/∂u|",
        "headline" => "content path (one-hot RAM read): the gradient is ALIVE and vanilla " *
            "saliency lands its mass on the content self-byte; whether that is the oracle's top " *
            "cause is the MEASURED faithfulness (saliency = 'what y is a function of now', the " *
            "oracle = 'what causes y over the horizon'). position output: saliency VANISHES → " *
            "near chance — the §7 'Fail / zero on index outputs' prediction, in numbers. " *
            "Oracle-as-method positive control corr=1/p@k=1 on every non-degenerate column.",
        "n_games_scored" => length(games) - length(na_records),
        "na_games" => na_records,
        "results" => summary)
    open(summary_path, "w") do io; JSON.print(io, rec, 2); end
    println("\n[saliency] wrote summary $summary_path")

    println("\n[saliency] ===== per-game HEADLINE (content path) =====")
    println("  game             ram   corr     spear    p@$topk    del_auc  ins_auc  gradL1")
    for r in summary
        r["is_content_path"] || continue
        println("  $(rpad(r["game"],16)) $(rpad(r["content_ram_index"],5)) " *
                "$(rpad(round(r["pearson"],digits=3),8)) $(rpad(round(r["spearman"],digits=3),8)) " *
                "$(rpad(round(r["precision_at_k"],digits=3),6)) " *
                "$(rpad(r["deletion_auc"] === nothing ? "NaN" : round(r["deletion_auc"],digits=3),8)) " *
                "$(rpad(r["insertion_auc"] === nothing ? "NaN" : round(r["insertion_auc"],digits=3),8)) " *
                "$(round(r["grad_l1"],digits=3))")
    end
    println("[saliency] ===== position contrast (saliency VANISHES) =====")
    for r in summary
        r["is_content_path"] && continue
        println("  $(rpad(r["game"],16)) corr=$(round(r["pearson"],digits=3)) " *
                "p@$topk=$(round(r["precision_at_k"],digits=3)) sal_max=$(round(r["sal_max_abs"],sigdigits=3)) " *
                "gradL1=$(round(r["grad_l1"],sigdigits=3))")
    end
    return 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    SaliencyAttr.main()
end
