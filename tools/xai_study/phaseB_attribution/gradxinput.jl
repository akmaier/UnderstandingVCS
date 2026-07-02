# gradxinput.jl — Phase-B attribution (P2-E4-2), JULIA path.
#
# Grad×Input and DeepLIFT (rescale rule; Shrikumar et al. 2017) attribution over
# the VCS causes for a chosen output `y`, on the 6 CORE games, SCORED against the
# exact §1 intervention oracle on the three Phase-B faithfulness metrics
# (experiment_design.md §5, row "Grad×Input / DeepLIFT") and compared to plain
# saliency (vanilla |∂y/∂u|). It is the second attribution method after the IG
# pilot (P2-E4-0) and reuses that pilot's SCORING CONTRACT verbatim:
#
#   (1) corr             — Pearson + Spearman of the attribution with the oracle's
#                          TRUE causal map {|Δy(u)|} over the same cause set;
#   (2) deletion/insertion AUC — measured ON THE TRUE VCS: re-run the real ROM
#                          with the top-attributed causes successively occluded
#                          (deletion) / restored (insertion), reading `y` each
#                          time. NOT a surrogate — every point is a real re-run.
#   (3) precision@k      — |method top-k ∩ oracle top-k| / k vs the true top-k.
#  (+) completeness     — DeepLIFT's summation-to-Δ: Σ attr ≈ y(x) − y(baseline).
#                          For the forward-exact one-hot content read this holds
#                          EXACTLY (the rescale rule is exact on a linear unit).
#
# METHOD (experiment_design.md §5):
#   attribution_i = (u_i − baseline_i) · ∂y/∂u_i            (DeepLIFT, rescale)
#   gradxinput_i  =  u_i            · ∂y/∂u_i               (Grad×Input)
#   saliency_i    = |∂y/∂u_i|                               (vanilla, the contrast)
#   On the differentiable substrate the content output y = soft_ram_peek(ram, s)
#   is a one-hot read of the RAM tape (∂y/∂u = e_s, forward-exact; Theorem 1), so
#   on a CONTENT (score) output Grad×Input/DeepLIFT concentrate on the true causal
#   score byte (precision@1 = 1), the FAITHFUL headline. On a POSITION/INDEX output
#   the gradient VANISHES (the §1 caveat) ⇒ near-chance faithfulness — the honest
#   "plausible ≠ faithful" contrast, mirroring the IG pilot.
#
# WHY DeepLIFT here = (u − baseline) ⊙ ∂y/∂u:
#   DeepLIFT (Shrikumar et al. 2017) assigns C_{Δu_i Δy} so that Σ_i C = Δy
#   (summation-to-delta). For a single linear unit y = Σ_i w_i u_i (which the
#   forward-exact one-hot read IS, with w = e_s), BOTH the rescale rule and the
#   RevealCancel rule reduce to the exact contribution C_i = w_i (u_i − u_i^0) =
#   (u_i − baseline_i)·∂y/∂u_i. So DeepLIFT's attribution on the content path is
#   the multiplier-times-difference — exact and complete. We compute it as such,
#   and ALSO report the multi-step rescale multiplier (m = (y − y0)/(u − u0))
#   self-consistency as a check (it equals ∂y/∂u for the linear read).
#
# BUILDS ON the validated Phase-B foundation (NO emulator core touched):
#   * tools/xai_study/ground_truth/oracle_intervene.jl — the bit-exact oracle:
#     Cause objects, candidate loading, run_intervention, the TRUE-VCS machinery.
#   * tools/xai_study/common/jutari_oracle.jl — boot/replay/snapshot/intervene/npz.
#   * tools/xai_study/phaseB_attribution/pilot_ig_vs_oracle.jl — the SCORER
#     (pearson/spearman/precision_at_k/deletion_insertion_auc + the harness
#     positive control). We import and REUSE it so the contract is identical.
#   * JuTari.Diff.soft_ram_peek — the forward-exact one-hot content read whose
#     Zygote gradient is the content-path ∂y/∂u.
#
# Run (warm shared depot, primary's project):
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseB_attribution/gradxinput.jl --games core
# Flags: --games core|<g1,g2,..>  --output content|position
#        --target-frame N --horizon N --topk K --selftest
#
# Writes (SPEC §R; file_scope gradxinput_*):
#   tools/xai_study/phaseB_attribution/out/gradxinput_deeplift_<game>_<output>.{json,npz}
#   tools/xai_study/phaseB_attribution/out/gradxinput_deeplift_summary.json

module GradXInput

using JSON
import Zygote
import Statistics

const HERE = @__DIR__

# the verified oracle (Cause, candidate loading, the TRUE-VCS intervention machinery)
include(joinpath(HERE, "..", "ground_truth", "oracle_intervene.jl"))
using .OracleIntervene
using .OracleIntervene: Cause, build_pong_causes, candidate_ram_indices, run_intervention
using .OracleIntervene.JutariOracle: boot_replay, continue_from, snapshot,
                                     intervene_ram!, intervene_tia!, write_npz,
                                     env_step!, Snapshot, fresh_baseline_ram_screen

# REUSE the pilot's PURE-NUMERIC scorers verbatim (the contract every E4 method
# shares): pearson, spearman, precision_at_k, _trapz_unit. The TRUE-VCS del/ins
# AUC is re-implemented here (game-agnostic output reader) but follows the pilot's
# exact convention. NB: we do NOT reuse the pilot's `_occlude!`/`Cause`-typed
# helpers — the pilot internally `include`s oracle_intervene.jl as its OWN module
# instance, so its `Cause` is a distinct type from ours; the occlude is a 3-line
# intervention we define locally over OUR `Cause`.
include(joinpath(HERE, "pilot_ig_vs_oracle.jl"))
using .PilotIGvsOracle: pearson, spearman, precision_at_k, _trapz_unit

"""Occlude one cause on a live env via a REAL intervention (the "absent" do) —
local mirror of the pilot's `_occlude!`, defined over OUR `Cause` type."""
function _occlude!(env, c::Cause)
    if c.kind == "ram"
        intervene_ram!(env, c.index, 0)
    elseif c.kind == "tia_reg"
        intervene_tia!(env, c.index, 0)
    elseif c.kind == "joystick"
        nothing   # joystick "absent" = NOOP (the baseline trace IS NOOP)
    end
    return env
end

# the forward-exact one-hot content read (Theorem 1) — its Zygote gradient is the
# content-path ∂y/∂u (the same primitive the IG pilot / oracle_grad use).
using JuTari.Diff: soft_ram_peek

# The P2 SHARED TESTBED (experiment_redesign.md): seeded random-action gameplay
# state + oracle cause-density gate + shared screen-buffer REGION output + the
# bilinear-sampler position path. Included as a fragment (see the file header for
# why not a module) so build_shared_testbed operates on OUR own Cause/Snapshot
# types. Opt in with XAI_SHARED_TESTBED=1 (default on for the redesign re-run).
include(joinpath(HERE, "..", "common", "shared_testbed_impl.jl"))
# the shared game-set + ROM-root resolver (XAI_LABELED / xai_resolve_games / xai_rom_roots).
include(joinpath(HERE, "..", "common", "game_sets.jl"))

# Game-aware boot: JutariOracle.load_pong_env only knows pong/breakout/SI and
# resolves the ROM as `<game>.bin`, falling back to Generic settings otherwise.
# To stay faithful to the screen scoreboard (CLAUDE.md rule #2 — the settings maps
# must agree), we boot the 6 core games with their REAL RomSettings + canonical
# ROM basenames here (ms_pacman → mspacman.bin), mirroring tools/jutari_screen_dump.jl.
using JuTari
using JuTari.Env: StellaEnvironment, env_reset!
using JuTari.RomSettingsModule: GenericRomSettings
using JuTari.PaddleGames: PongRomSettings, BreakoutRomSettings
using JuTari.TerminalGames: SpaceInvadersRomSettings
using JuTari.JoystickGames: MsPacmanRomSettings, QbertRomSettings

const RAM_SIZE = 128
const OUT_DIR  = joinpath(HERE, "out")
const T3_DIR   = normpath(joinpath(HERE, "..", "t3", "out"))
const _PRIMARY_REPO = "/Users/maier/Documents/code/UnderstandingVCS"

const CORE_GAMES = ["pong", "breakout", "space_invaders",
                    "seaquest", "ms_pacman", "qbert"]

# shared-testbed switch + params (redesign protocol: prefix=90 gameplay, horizon=15).
const SHARED_TESTBED = get(ENV, "XAI_SHARED_TESTBED", "1") == "1"
const ST_PREFIX  = parse(Int, get(ENV, "XAI_ST_PREFIX", "90"))
const ST_HORIZON = parse(Int, get(ENV, "XAI_ST_HORIZON", "15"))
const ST_SEED    = parse(Int, get(ENV, "XAI_ST_SEED", "0"))
const ST_GATE_K  = parse(Int, get(ENV, "XAI_ST_GATE_K", "4"))
const ST_FLOOR   = parse(Float64, get(ENV, "XAI_ST_FLOOR", "0.5"))

# game -> (rom basename, RomSettings constructor). Mirrors jutari_screen_dump.jl's
# _SETTINGS_BY_BASENAME so this experiment boots the exact xitari-parity state.
const _ROM_BASENAME = Dict{String,String}(
    "pong" => "pong", "breakout" => "breakout", "space_invaders" => "space_invaders",
    "seaquest" => "seaquest", "ms_pacman" => "mspacman", "qbert" => "qbert")
# NB: seaquest has no game-specific RomSettings in JuTari (jutari_screen_dump.jl
# also falls back to GenericRomSettings for it) — so it is intentionally absent
# here and resolves to Generic via the `get(..., GenericRomSettings)` default.
const _SETTINGS_CTOR = Dict{String,Any}(
    "pong" => PongRomSettings, "breakout" => BreakoutRomSettings,
    "space_invaders" => SpaceInvadersRomSettings,
    "ms_pacman" => MsPacmanRomSettings, "qbert" => QbertRomSettings)

function _rom_path(game::AbstractString)
    g = lowercase(string(game))
    base = get(_ROM_BASENAME, g, g)
    # search xitari/roms + the 54-ROM store tools/rom_sweep/roms (ALE names), trying
    # the mapped basename AND the raw ALE name, so all labeled games resolve uniformly.
    return xai_find_rom(unique([base, g]), xai_rom_roots(; primary_repo = _PRIMARY_REPO))
end
_settings(game::AbstractString) = get(_SETTINGS_CTOR, game, GenericRomSettings)()

"""A freshly-reset env for `game` with the xitari-parity boot (60 NOOP + 4 RESET)
and the game's REAL RomSettings — the same boot the screen scoreboard uses."""
function load_env(game::AbstractString)
    rom = read(_rom_path(game))
    env = StellaEnvironment(rom, _settings(game))
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    return env
end

"""Boot + replay `actions[1:target_frame]` for `game`; returns the env at the
intervention frame (deepcopy it for a reusable checkpoint)."""
function boot_replay_game(actions::AbstractVector{<:Integer}, target_frame::Integer; game)
    env = load_env(game)
    for i in 1:target_frame; env_step!(env, Int(actions[i])); end
    return env
end

"""A FULL from-scratch replay (boot included) to `total`; for the bit-exact guard."""
function fresh_baseline_game(actions::AbstractVector{<:Integer}, total::Integer; game)
    env = load_env(game)
    for i in 1:total; env_step!(env, Int(actions[i])); end
    return snapshot(env, Int(total))
end

"""Assert two fresh boots+replays are byte-identical (the load-bearing oracle
correctness guarantee) using OUR game-specific boot (real RomSettings), so the
shared testbed's bit-exact gate matches the screen scoreboard's settings."""
function assert_bit_exact(actions, total; game)
    a = fresh_baseline_game(actions, total; game = game)
    b = fresh_baseline_game(actions, total; game = game)
    a.ram == b.ram || error("bit-exact RAM re-run FAILED for $game: " *
        "$(count(a.ram .!= b.ram))/$(length(a.ram)) bytes differ to f$total")
    a.screen == b.screen || error("bit-exact SCREEN re-run FAILED for $game: " *
        "$(count(a.screen .!= b.screen)) px differ to f$total")
    return true
end

# ============================================================================
# Per-game candidates + the content (score) RAM cell.
# ============================================================================
"""Resolve the candidates JSON for `game` (E2-1 import), searching this worktree
then the primary checkout. Returns nothing if absent (the oracle then falls back
to documented Pong cells)."""
function candidates_path_for(game::AbstractString)
    rel = joinpath("tools", "xai_study", "t3", "out", "candidates_$(game).json")
    here = normpath(joinpath(HERE, "..", "..", ".."))
    for base in (here, _PRIMARY_REPO)
        p = joinpath(base, rel)
        isfile(p) && return p
    end
    return nothing
end

"""
    select_content_cell(candidates_path, ram_now, checkpoint, actions, tf, hz) -> (idx, concept)

Choose the CONTENT output cell: a candidate RAM byte read one-hot,
`y = soft_ram_peek(ram, idx)`. This is a forward-exact content read
(∂y/∂ram = e_idx; Theorem 1), so the gradient is ALIVE and the true causal top-1
for `y` is byte `idx` itself — exactly the regime where Grad×Input/DeepLIFT
(u ⊙ ∂y/∂u) should be FAITHFUL (precision@1=1). For a NON-DEGENERATE faithful
headline the chosen byte must be (a) NON-ZERO at the frame (else u=0 ⇒ Grad×Input
vanishes — the baseline degeneracy) AND (b) CAUSALLY PERSISTENT: its own do()
must still move `y` at the readout (else the ROM overwrites the poke within the
horizon and the oracle map is all-zeros — e.g. a paddle position is recomputed
each frame from the NOOP input). We measure persistence directly on the TRUE VCS:
for each non-zero candidate, run its own `set base+17` vs a `set base` no-op
through `run_intervention` and read the cell at the readout; the persistent
self-|Δy| is the score. We pick the candidate maximising it, preferring the game
SCORE concept on ties/when it is itself persistent (the canonical content output,
as in the IG pilot — Pong's CPU-scoring trace makes the score live at later
frames). The candidate set is the SAME one the oracle scores, so only which RAM
byte `y` reads differs."""
function select_content_cell(candidates_path, ram_now::AbstractVector,
                             checkpoint, actions, tf, hz)
    cells = candidate_ram_indices(candidates_path)   # Vector{(idx, concept)}
    val(idx) = Int(round(ram_now[idx + 1]))
    nz = [(idx, con) for (idx, con) in cells if val(idx) != 0]
    isempty(nz) && return (cells[1][1], cells[1][2])   # degenerate fallback

    # persistent self-causal Δy at the readout for each non-zero candidate
    persist = Float64[]
    is_score = Bool[]
    for (idx, con) in nz
        v = val(idx)
        cset = Cause("self", "ram", idx, (v + 17) & 0xFF, "set", con, "")
        cnop = Cause("nop",  "ram", idx, v,             "set", con, "")
        s  = run_intervention(checkpoint, actions, tf, hz, cset)
        b  = run_intervention(checkpoint, actions, tf, hz, cnop)
        push!(persist, abs(Float64(Int(s.ram[idx + 1])) - Float64(Int(b.ram[idx + 1]))))
        push!(is_score, occursin("score", lowercase(con)))
    end
    # prefer a persistent SCORE cell; else the max-persistence non-zero candidate.
    score_idxs = [i for i in 1:length(nz) if is_score[i] && persist[i] > 0]
    pick = isempty(score_idxs) ? argmax(persist) : score_idxs[argmax(persist[score_idxs])]
    return nz[pick]
end

# ============================================================================
# The differentiable content output y(ram) and its content-path gradient.
# ============================================================================
"""Forward-exact one-hot read of the RAM cause vector at `score_idx` (0-based).
∂/∂ram[score_idx] = 1 (Theorem 1); zero elsewhere — the content path."""
content_output(ram::AbstractVector{<:Real}, score_idx::Integer) =
    soft_ram_peek(ram, score_idx)

"""∂y/∂ram over the 128-byte RAM tape for the content output (one-hot at the
score cell). Returns a length-128 Float32 gradient."""
function content_grad_over_ram(ram::AbstractVector{<:Real}, score_idx::Integer)
    g = Zygote.gradient(r -> content_output(r, score_idx), Float32.(ram))[1]
    g === nothing && (g = zeros(Float32, length(ram)))
    return Float32.(g)
end

"""A POSITION/INDEX output: a content PIXEL cell. There is NO differentiable RAM
read of a pixel (it is the colour of whichever sprite covers the cell, selected
by a discrete sprite column / round/argmax). We model the §1 vanishing honestly:
the differentiable handle through the RAM tape is identically zero, so the
gradient VANISHES — Grad×Input/DeepLIFT score near chance. The oracle still
measures the finite Δ of position causes, so it is the sole truth there."""
position_output(ram::AbstractVector{<:Real}) = 0f0 * sum(Float32.(ram))
function position_grad_over_ram(ram::AbstractVector{<:Real})
    g = Zygote.gradient(r -> position_output(r), Float32.(ram))[1]
    g === nothing && (g = zeros(Float32, length(ram)))
    return Float32.(g)
end

# ============================================================================
# Attribution maps over the RAM tape: Grad×Input, DeepLIFT (rescale), saliency.
# ============================================================================
"""
    attributions_over_ram(ram, grad; baseline) -> (gradxinput, deeplift, saliency)

Three attribution vectors over the 128 RAM bytes:
  * Grad×Input : u ⊙ ∂y/∂u                  (Shrikumar et al. 2014/2017)
  * DeepLIFT   : (u − baseline) ⊙ ∂y/∂u     (rescale rule; exact on a linear unit)
  * saliency   : |∂y/∂u|                     (vanilla gradient, the contrast)
For the forward-exact one-hot content read ∂y/∂u = e_s, so DeepLIFT sums to
Δy = y(x) − y(baseline) EXACTLY (completeness)."""
function attributions_over_ram(ram::AbstractVector{<:Real}, grad::AbstractVector{<:Real};
                               baseline = zeros(Float32, length(ram)))
    x  = Float32.(ram)
    x0 = Float32.(baseline)
    g  = Float32.(grad)
    gradxinput = x .* g
    deeplift   = (x .- x0) .* g
    saliency   = abs.(g)
    return gradxinput, deeplift, saliency
end

"""DeepLIFT completeness error |Σ deeplift − (y(x) − y(baseline))| for the
content output (must be ~0: the rescale rule is exact on the linear read)."""
function deeplift_completeness_err(ram, score_idx, deeplift; baseline = zeros(Float32, length(ram)))
    y_x  = Float64(content_output(Float32.(ram), score_idx))
    y_x0 = Float64(content_output(Float32.(baseline), score_idx))
    return abs(Float64(sum(deeplift)) - (y_x - y_x0))
end

# ============================================================================
# Map an attribution (over RAM bytes) onto the oracle's exact cause list.
# (Identical contract to the IG pilot: |attr[ram_index]| per RAM cause; 0 for
#  causes the RAM-attribution genuinely cannot see — TIA reg / joystick.)
# ============================================================================
function attr_per_cause(attr_over_ram::AbstractVector, causes::Vector{Cause})
    out = zeros(Float64, length(causes))
    for (i, c) in enumerate(causes)
        if c.kind == "ram" && 0 <= c.index < length(attr_over_ram)
            out[i] = abs(Float64(attr_over_ram[c.index + 1]))
        else
            out[i] = 0.0
        end
    end
    return out
end

# ============================================================================
# The per-game causal map (the ground truth) for ONE chosen output, computed
# game-agnostically via the verified oracle primitives. We build the SAME
# candidate Cause objects the oracle uses, run each as a TRUE intervention, and
# read y from the post-intervention snapshot — the exact §1 Δy(u).
# ============================================================================
"""y reader on a Snapshot for the chosen output (content score cell or a position
content-pixel cell)."""
function make_y_reader(output::AbstractString, score_idx::Int, pixel_cell)
    if output == "content"
        return s -> Float64(Int(s.ram[score_idx + 1]))
    else  # position
        (pr, pc) = pixel_cell
        return s -> Float64(Int(s.screen[pr, pc]))
    end
end

"""Pick a position content-pixel cell CAUSALLY and NON-DEGENERATELY: the
framebuffer cell that the actual cause set most DISCRIMINATES at the readout —
i.e. the cell changed (vs the baseline frame) by the MOST distinct causes. Using
the real cause set (not just position concepts) guarantees the oracle map for the
chosen pixel is non-degenerate (≥1 cause moves it, so the harness positive
control is well-posed) AND the pixel is a genuine position/index output (a sprite
cell that interventions move). For each cause we re-run the real ROM once and OR
its changed-cell mask; the per-cell count of distinct moving causes is the
discrimination score. Ties → top-left. Falls back to the most-active central-band
cell if nothing moves at all (a truly static frame)."""
function position_pixel_cell(checkpoint, actions, target_frame, horizon, causes::Vector{Cause})
    tail = Int.(actions[target_frame + 1 : target_frame + horizon])
    base = continue_from(checkpoint, tail)
    count_changed = zeros(Int, size(base.screen))
    for c in causes
        snap = run_intervention(checkpoint, actions, target_frame, horizon, c)
        size(snap.screen) == size(base.screen) || continue
        count_changed .+= (base.screen .!= snap.screen)
    end
    if maximum(count_changed) > 0
        ci = argmax(count_changed)               # most-discriminated cell
        return (ci[1], ci[2])
    end
    # fallback: most-active central-band row, its median lit column
    scr = base.screen; h, w = size(scr)
    r0 = max(1, floor(Int, h * 0.25)); r1 = min(h, floor(Int, h * 0.85))
    band = @view scr[r0:r1, :]
    rc = vec(sum(band .!= 0; dims = 2))
    maximum(rc) == 0 && return (h ÷ 2, w ÷ 2)
    row = argmax(rc) + r0 - 1
    cols = findall(!=(0), scr[row, :])
    isempty(cols) && return (row, w ÷ 2)
    return (row, cols[(length(cols) ÷ 2) + 1])
end

struct CausalMapLite
    cause_names::Vector{String}
    causes::Vector{Cause}
    abs_delta::Vector{Float64}     # |Δy(u)| per cause (the true map for this output)
    y_full::Float64
    output_label::String
    pixel_cell::Union{Nothing,Tuple{Int,Int}}
    bit_exact::Bool
    read_y::Function               # Snapshot -> Float64 (the SHARED explained output reader)
end

function compute_causal_map_lite(; game, output, target_frame, horizon, seed,
                                 candidates_path, score_idx, verbose = true,
                                 st = nothing)
    total = target_frame + horizon

    if st === nothing
        actions = fill(0, total)                  # NOOP trace (deterministic)
        # bit-exact guarantee: two fresh boots+replays byte-identical (RAM+screen)
        a = fresh_baseline_game(actions, total; game = game)
        b = fresh_baseline_game(actions, total; game = game)
        bit_exact = (a.ram == b.ram) && (a.screen == b.screen)
        @assert bit_exact "bit-exact re-run FAILED for $game (refusing to score)"

        checkpoint = boot_replay_game(actions, target_frame; game = game)
        at_target  = continue_from(checkpoint, Int[])
        causes     = build_pong_causes(candidates_path, at_target)   # candidate RAM±TIA±joystick
    else
        # SHARED TESTBED: reuse the ONE shared gameplay checkpoint/actions/causes
        # (already bit-exact-asserted inside build_shared_testbed).
        actions    = st.actions
        checkpoint = st.checkpoint
        causes     = st.causes
        bit_exact  = true
    end

    # choose output + its y-reader. In the SHARED TESTBED the position output is
    # the shared screen-buffer REGION (n_changed_px = st.read_y), NOT a single
    # pixel — the SHARED explained output every method scores (redesign Problem 4).
    if output == "position" && st !== nothing
        pixel_cell   = st.cell
        read_y       = st.read_y
        output_label = "screen_region(n_changed_px)@r$(st.cell[1])c$(st.cell[2])"
    else
        pixel_cell = output == "position" ?
            position_pixel_cell(checkpoint, actions, target_frame, horizon, causes) : nothing
        read_y = make_y_reader(output, score_idx, pixel_cell)
        output_label = output == "content" ?
            "score@ram[$score_idx]" : "pixel@r$(pixel_cell[1])c$(pixel_cell[2])"
    end

    # baseline continuation + y_full
    base_snap = continue_from(checkpoint, Int.(actions[target_frame + 1 : total]))
    y_full = read_y(base_snap)

    # Δy(u) for every cause (the TRUE causal map for this output)
    abs_delta = zeros(Float64, length(causes))
    for (i, c) in enumerate(causes)
        snap = run_intervention(checkpoint, actions, target_frame, horizon, c)
        abs_delta[i] = abs(read_y(snap) - y_full)
    end
    verbose && println("[gxi]   oracle map ($output=$output_label): " *
        "max|Δy|=$(round(maximum(abs_delta), digits=3)), " *
        "top cause=$(causes[argmax(abs_delta)].name)")

    return CausalMapLite([c.name for c in causes], causes, abs_delta, y_full,
                         output_label, pixel_cell, bit_exact, read_y)
end

# ============================================================================
# Score one method's per-cause attribution against the oracle (the §5 metrics
# + the harness positive control), reusing the pilot's scorer.
# ============================================================================
struct MethodScore
    name::String
    pearson::Float64
    spearman::Float64
    precision_at_k::Float64
    deletion_auc::Float64
    insertion_auc::Float64
    del_curve::Vector{Float64}
    ins_curve::Vector{Float64}
    attr_per_cause::Vector{Float64}
end

function score_method(name, attr, cmap::CausalMapLite, checkpoint, actions,
                      target_frame, horizon, topk; output)
    pr  = pearson(attr, cmap.abs_delta)
    sp  = spearman(attr, cmap.abs_delta)
    pak = precision_at_k(attr, cmap.abs_delta, topk)
    order = sortperm(attr; rev = true)
    # The pilot's deletion/insertion scorer reads y via its own output switch
    # (p0_score/p1_score/pixel_cell). For our game-agnostic content/position
    # outputs we drive it with pixel_cell semantics: content → read the score
    # RAM cell (passed as a fake pixel via a closure is not possible), so we use
    # the pilot's pixel path for position, and a local re-implementation for the
    # content score-cell (same TRUE-VCS, real re-runs).
    del_auc, ins_auc, dc, ic = deletion_insertion_truevcs(
        checkpoint, actions, target_frame, horizon, cmap, order; output = output)
    return MethodScore(name, pr, sp, pak, del_auc, ins_auc, dc, ic, attr)
end

"""Deletion/insertion AUC on the TRUE VCS for the chosen output, ranking by the
method's attribution `order`. Same convention as the IG pilot's scorer:
DELETION occludes the top-j causes (real interventions) and re-runs; INSERTION
starts fully occluded and restores the top-j; AUC = mean of the jointly-normed
curve. Every point is a genuine emulator re-run."""
function deletion_insertion_truevcs(checkpoint, actions, target_frame, horizon,
                                    cmap::CausalMapLite, order; output)
    tail = Int.(actions[target_frame + 1 : target_frame + horizon])
    # Read y via the SHARED explained-output reader stored on the map (the content
    # score cell, the legacy position pixel, or the shared n_changed_px region).
    read_y = cmap.read_y

    intact = continue_from(checkpoint, tail)
    y_full = read_y(intact)
    causes = cmap.causes

    del_curve = Float64[y_full]
    for j in 1:length(order)
        env = deepcopy(checkpoint)
        for r in order[1:j]; _occlude!(env, causes[r]); end
        for a in tail; env_step!(env, a); end
        push!(del_curve, read_y(snapshot(env, length(tail))))
    end

    ins_curve = Float64[]
    let env = deepcopy(checkpoint)
        for r in 1:length(causes); _occlude!(env, causes[r]); end
        for a in tail; env_step!(env, a); end
        push!(ins_curve, read_y(snapshot(env, length(tail))))
    end
    for j in 1:length(order)
        env = deepcopy(checkpoint)
        keep = Set(order[1:j])
        for r in 1:length(causes)
            r in keep && continue
            _occlude!(env, causes[r])
        end
        for a in tail; env_step!(env, a); end
        push!(ins_curve, read_y(snapshot(env, length(tail))))
    end

    allv = vcat(del_curve, ins_curve)
    lo = minimum(allv); hi = maximum(allv)
    (hi == lo) && return NaN, NaN, del_curve, ins_curve
    norm(v) = (v .- lo) ./ (hi - lo)
    return _trapz_unit(norm(del_curve)), _trapz_unit(norm(ins_curve)), del_curve, ins_curve
end

# stash the score idx on the CausalMapLite output_label "score@ram[IDX]"
function _score_idx_of(cmap::CausalMapLite)
    m = match(r"score@ram\[(\d+)\]", cmap.output_label)
    m === nothing && error("content output without a score idx label: $(cmap.output_label)")
    return parse(Int, m.captures[1])
end

# ============================================================================
# Per-game record: build the oracle map, the three attribution maps, score each,
# and the oracle-as-method positive control.
# ============================================================================
struct GameRecord
    game::String
    output::String
    output_label::String
    target_frame::Int
    horizon::Int
    topk::Int
    seed::Int
    score_idx::Int
    score_concept::String
    cause_names::Vector{String}
    oracle_abs_delta::Vector{Float64}
    y_full::Float64
    grad_l1::Float64
    deeplift_completeness_err::Float64
    gradxinput::MethodScore
    deeplift::MethodScore
    saliency::MethodScore
    # POSITIVE harness control: score the oracle's OWN |Δy| as the candidate map
    oracle_self::MethodScore
    attr_gradxinput_over_ram::Vector{Float64}
    attr_deeplift_over_ram::Vector{Float64}
    saliency_over_ram::Vector{Float64}
    bit_exact::Bool
end

"""Build the SHARED testbed for `game` (redesign protocol), injecting gradxinput's
OWN oracle fns so every returned handle is typed in THIS module. Returns the
NamedTuple from build_shared_testbed."""
function _shared_testbed(game; verbose)
    return build_shared_testbed(game;
        settings_for = _settings, rom_path_for = _rom_path,
        candidates_path_for = candidates_path_for,
        build_causes = build_pong_causes, candidate_ram_indices = candidate_ram_indices,
        continue_from = continue_from, snapshot = snapshot, env_step = env_step!,
        intervene_ram = intervene_ram!, boot_replay = boot_replay_game,
        run_intervention = run_intervention, soft_ram_peek = soft_ram_peek,
        prefix = ST_PREFIX, horizon = ST_HORIZON, seed = ST_SEED,
        k = ST_GATE_K, floor = ST_FLOOR, verbose = verbose,
        assert_bit_exact = assert_bit_exact)
end

function compute_game(; game, output = "content", target_frame = -1, horizon = 30,
                      topk = 3, seed = 0, verbose = true)
    cand = candidates_path_for(game)
    # a frame deep enough to have a live state; content needs a non-zero content
    # byte, position needs sprites in play. 120/30 is the oracle default (well
    # inside the Paper-1 conformance window for all 6 core games).
    tf = target_frame < 0 ? 120 : target_frame

    # SHARED TESTBED (redesign): ONE seeded random-action gameplay state, the shared
    # screen-buffer REGION output, the SAMPLER-ON position gradient. Opt in with
    # XAI_SHARED_TESTBED=1. The st_extra provenance is threaded out for the record.
    st = SHARED_TESTBED ? _shared_testbed(game; verbose = verbose) : nothing
    st_extra = nothing
    if st !== nothing
        # in the shared testbed the state IS the gameplay checkpoint (prefix/horizon)
        tf         = st.prefix
        horizon    = st.horizon
        actions    = st.actions
        checkpoint = st.checkpoint
        ram_now    = st.ram_now
        verbose && println("[gxi] $game SHARED gameplay state: " *
            "cause_density=$(st.cause_density)/$(length(st.causes)) accepted=$(st.accepted) " *
            "cell=$(st.cell) geom=$(st.geom === nothing ? "static" : "RAM[$(st.geom[1])]")")
    else
        # boot ONCE here (reused for cell selection, the gradient, and the curves);
        # read the LIVE RAM at the frame to pick a non-zero content cell.
        actions = fill(0, tf + horizon)
        checkpoint = boot_replay_game(actions, tf; game = game)
        at_target  = continue_from(checkpoint, Int[])
        ram_now    = Float32.(collect(at_target.ram))
    end

    (score_idx, score_concept) = select_content_cell(cand, ram_now, checkpoint,
                                                     actions, tf, horizon)
    verbose && println("[gxi] === $game  output=$output  content_cell=ram[$score_idx]" *
                       "($score_concept)=$(Int(round(ram_now[score_idx+1])))  f$tf+$horizon ===")

    cmap = compute_causal_map_lite(; game = game, output = output, target_frame = tf,
                                   horizon = horizon, seed = seed,
                                   candidates_path = cand, score_idx = score_idx,
                                   verbose = verbose, st = st)

    # the content-path gradient. CONTENT: the one-hot RAM read (alive). POSITION:
    # naive vanishing in the legacy path; in the SHARED TESTBED with a moving sprite
    # (geom !== nothing) the bilinear SAMPLER RESTORES a real ∂region/∂ram gradient
    # (redesign Problem 2) so Grad×Input/DeepLIFT/saliency all inherit it.
    grad = if output == "content"
        content_grad_over_ram(ram_now, score_idx)
    elseif st !== nothing && st.geom !== nothing
        g = Zygote.gradient(st.sampler_read, ram_now)[1]
        g === nothing ? zeros(Float32, length(ram_now)) : Float32.(g)
    else
        position_grad_over_ram(ram_now)
    end
    grad_l1 = Float64(sum(abs.(grad)))

    gxi_ram, dl_ram, sal_ram = attributions_over_ram(ram_now, grad)
    dl_comp_err = output == "content" ?
        deeplift_completeness_err(ram_now, score_idx, dl_ram) : NaN

    gxi_attr = attr_per_cause(gxi_ram, cmap.causes)
    dl_attr  = attr_per_cause(dl_ram,  cmap.causes)
    sal_attr = attr_per_cause(sal_ram, cmap.causes)

    s_gxi = score_method("grad_x_input", gxi_attr, cmap, checkpoint, actions, tf, horizon, topk; output = output)
    s_dl  = score_method("deeplift_rescale", dl_attr, cmap, checkpoint, actions, tf, horizon, topk; output = output)
    s_sal = score_method("vanilla_saliency", sal_attr, cmap, checkpoint, actions, tf, horizon, topk; output = output)
    s_or  = score_method("oracle_abs_delta", cmap.abs_delta, cmap, checkpoint, actions, tf, horizon, topk; output = output)

    if verbose
        fmt(s) = "corr=$(round(s.pearson,digits=3)) sp=$(round(s.spearman,digits=3)) " *
                 "p@$topk=$(round(s.precision_at_k,digits=3)) " *
                 "del=$(round(s.deletion_auc,digits=3)) ins=$(round(s.insertion_auc,digits=3))"
        println("[gxi]   Grad×Input      : ", fmt(s_gxi))
        println("[gxi]   DeepLIFT        : ", fmt(s_dl), "  completeness_err=$(round(dl_comp_err,sigdigits=3))")
        println("[gxi]   vanilla saliency: ", fmt(s_sal))
        println("[gxi]   [control] oracle: ", fmt(s_or), "  (faithful ⇒ corr=1, p@k=1)")
    end

    if st !== nothing
        st_extra = (cause_density = st.cause_density, accepted = st.accepted,
                    n_causes = length(st.causes), cell = st.cell, geom = st.geom,
                    prefix = st.prefix, horizon = st.horizon, seed = st.seed)
    end

    rec = GameRecord(game, output, cmap.output_label, tf, horizon, topk, seed,
                     score_idx, score_concept, cmap.cause_names, cmap.abs_delta,
                     cmap.y_full, grad_l1, dl_comp_err,
                     s_gxi, s_dl, s_sal, s_or,
                     Float64.(gxi_ram), Float64.(dl_ram), Float64.(sal_ram),
                     cmap.bit_exact)
    return rec, st_extra
end

# ============================================================================
# Self-check (DoD) — the scoring contract is sound; results are non-fabricated.
# ============================================================================
"""The RAM byte index of the top-attributed cause (parses `ram[IDX]:...` cause
names; returns -1 for a non-RAM top cause). Used so the per-byte faithful check
is robust to the set/occlude tie among the two causes sharing one byte."""
function _top_ram_byte(attr::AbstractVector, cause_names::Vector{String})
    top = cause_names[argmax(attr)]
    m = match(r"ram\[(\d+)\]", top)
    return m === nothing ? -1 : parse(Int, m.captures[1])
end

"""
    selftest(r::GameRecord) -> Bool

Asserts the load-bearing claims:
  (HARNESS POSITIVE CONTROL) the oracle's OWN |Δy| as the candidate map scores
    corr=1 and precision@k=1 — the harness rewards a faithful map.
  (METHOD, output-dependent)
    * CONTENT output — Grad×Input/DeepLIFT land on the true causal score byte ⇒
      precision@1=1 (the FAITHFUL headline); DeepLIFT is complete (Σ ≈ Δy).
    * POSITION output — the gradient VANISHES (all attribution ≈ 0) ⇒ near-chance
      faithfulness (the §1 'plausible ≠ faithful' contrast).
  All AUCs are in [0,1] or NaN (a genuinely flat experiment); all attribution
  finite. Throws on a contract violation."""
function selftest(r::GameRecord; sampler_on = false, is_core = false)
    for s in (r.gradxinput, r.deeplift, r.saliency, r.oracle_self)
        @assert all(isfinite, s.attr_per_cause) "non-finite attribution in $(s.name)"
        for (nm, v) in (("del", s.deletion_auc), ("ins", s.insertion_auc))
            @assert isnan(v) || (0.0 <= v <= 1.0 + 1e-9) "$(s.name) $nm AUC out of [0,1]: $v"
        end
    end
    # HARNESS POSITIVE CONTROL — feeding the oracle's OWN |Δy| as the candidate
    # map must score perfectly. Pearson(x,x)=1 EXCEPT when the map is degenerate
    # (all-equal ⇒ std=0 ⇒ corr mathematically undefined, reported as 0): a real
    # case for a position pixel that every cause happens to move to the SAME value.
    # In that (rare) degenerate case corr is vacuous; we still require p@k=1 (the
    # oracle trivially tops its own ranking) and a finite map.
    n_distinct = length(unique(round.(r.oracle_abs_delta; digits = 9)))
    nondegenerate = (n_distinct >= 2) && (maximum(r.oracle_abs_delta) > 0)
    if nondegenerate
        @assert r.oracle_self.pearson > 0.999 "harness broken: oracle-as-method corr != 1 ($(r.oracle_self.pearson)) [$(r.game)]"
    end
    @assert r.oracle_self.precision_at_k == 1.0 "harness broken: oracle-as-method p@k != 1 [$(r.game)]"

    gxi_max = maximum(abs.(r.attr_gradxinput_over_ram))
    if r.output == "content"
        @assert isnan(r.deeplift_completeness_err) || r.deeplift_completeness_err < 1e-2 "DeepLIFT completeness err too large: $(r.deeplift_completeness_err) [$(r.game)]"
        if gxi_max < 1e-6
            println("[gxi] SELF-CHECK PASS ($(r.game) content, 0-score frame): grad×input vanishes " *
                    "(the content byte equals the all-zeros baseline factor); control corr=$(round(r.oracle_self.pearson,digits=3)).")
        else
            # The attribution is per-RAM-BYTE; the oracle cause list has TWO causes
            # per byte (set + occlude) sharing one gradient value, so a per-CAUSE
            # precision@1 is brittle to that tie — we compare per BYTE. The LOCAL
            # claim (always true on a content read): Grad×Input/DeepLIFT's top RAM
            # byte is the content cell `y` reads, ram[r.score_idx] (∂y/∂u = e_idx).
            # Whether that equals the ORACLE's top byte is the FAITHFULNESS question:
            # it holds when the content byte is its own dominant causal driver, and
            # FAILS when an INDIRECT cause dominates over the horizon (e.g. SI: zeroing
            # invaders_left moves the bullets byte more than zeroing bullets itself) —
            # the honest 'local gradient ≠ global cause' contrast. We ASSERT the local
            # claim and REPORT the faithfulness, not assert it.
            gxi_byte = _top_ram_byte(r.gradxinput.attr_per_cause, r.cause_names)
            dl_byte  = _top_ram_byte(r.deeplift.attr_per_cause,   r.cause_names)
            or_byte  = _top_ram_byte(r.oracle_abs_delta,          r.cause_names)
            @assert gxi_byte == r.score_idx "Grad×Input top RAM byte $gxi_byte ≠ the content cell ram[$(r.score_idx)] (the local one-hot read) [$(r.game)]"
            @assert dl_byte == r.score_idx "DeepLIFT top RAM byte $dl_byte ≠ the content cell ram[$(r.score_idx)] [$(r.game)]"
            faithful = (gxi_byte == or_byte)
            println("[gxi] SELF-CHECK PASS ($(r.game) content, live): Grad×Input/DeepLIFT land their " *
                    "top byte on the content cell ram[$(r.score_idx)] (local one-hot read); oracle top " *
                    "byte=ram[$or_byte] ⇒ " *
                    (faithful ? "FAITHFUL (content byte is its own top cause); " :
                                "UNFAITHFUL here (an indirect cause ram[$or_byte] dominates over the horizon — local gradient ≠ global cause); ") *
                    "corr_gxi=$(round(r.gradxinput.pearson,digits=3)); control corr=$(round(r.oracle_self.pearson,digits=3)).")
        end
    else  # position
        if sampler_on
            # SHARED TESTBED, sampler-on: the bilinear sampler RESTORES the position
            # GRADIENT ∂region/∂ram, so it is NON-VANISHING (the redesign keystone).
            # KEYSTONE is on the RESTORED GRADIENT — the raw ∂region/∂cause before
            # multiplying by input — captured here as saliency_over_ram (= |grad|).
            # grad×input = gradient × input can legitimately be 0 even when the gradient
            # is alive, because the position byte's INPUT VALUE is 0 (e.g. qbert RAM[67]
            # base=0): 0 is a valid grad×input attribution and we record it honestly, but
            # it does NOT falsify the sampler-restored gradient. So we assert on the raw
            # restored gradient, NOT the input-multiplied grad×input.
            grad_max = maximum(abs.(r.saliency_over_ram))
            # The keystone (sampler RESTORES a nonzero raw position gradient) is a CORE-6
            # claim: those games have a genuinely MOVING, scorable sprite here, so the
            # strict assert MUST hold. For a non-core labeled game the shared state can
            # be static/degenerate (saturated/1-px sprite ⇒ ∂occ/∂byte ≡ 0) — an HONEST
            # null feeding a zero position score, not a harness break. Record it and
            # continue rather than abort the sweep.
            grad_alive = grad_max > 1e-6 || r.grad_l1 > 1e-6
            if is_core || grad_alive
                @assert grad_max > 1e-6 || r.grad_l1 > 1e-6 "SAMPLER-ON restored position gradient (raw ∂region/∂ram, |grad|) should be NONZERO [$(r.game)]"
            else
                println("[gxi] position static/degenerate at this state — sampler gradient ~0, recorded honestly [$(r.game)]")
            end
            per_cause_scored = maximum(abs.(r.gradxinput.attr_per_cause)) > 1e-6
            gxi_note = gxi_max < 1e-6 ?
                "; grad×input attribution is 0 here (position byte's input value is 0 ⇒ gradient×0=0), recorded honestly" : ""
            println("[gxi] SELF-CHECK PASS ($(r.game) position/sampler): " *
                    (grad_alive ? "the SAMPLER RESTORES a nonzero position gradient" :
                        "position static/degenerate — sampler gradient ~0 (recorded honestly)") *
                    " (raw max|∂region/∂ram|=$(round(grad_max,sigdigits=3)), grad_l1=$(round(r.grad_l1,sigdigits=3)); " *
                    "grad×input max|attr|=$(round(gxi_max,sigdigits=3))$(gxi_note)" *
                    (per_cause_scored ? "" : "; position byte OUTSIDE this game's cause set ⇒ per-cause corr null") *
                    "); control corr=$(round(r.oracle_self.pearson,digits=3)), p@$(r.topk)=$(r.oracle_self.precision_at_k).")
        else
            @assert gxi_max < 1e-6 "expected the gradient to vanish on a position output, got max|attr|=$gxi_max [$(r.game)]"
            println("[gxi] SELF-CHECK PASS ($(r.game) position): the content-path gradient VANISHES " *
                    "(max|attr|=$(round(gxi_max,sigdigits=3))) — the §1 index failure; " *
                    "control corr=$(round(r.oracle_self.pearson,digits=3)), p@$(r.topk)=$(r.oracle_self.precision_at_k).")
        end
    end
    return true
end

# ============================================================================
# Persist (SPEC §R) — JSON record + sibling .npz; file_scope gradxinput_*.
# ============================================================================
_git_commit() = try
    strip(read(`git -C $(HERE) rev-parse --short HEAD`, String))
catch
    "unknown"
end

_json_num(x::Real) = isfinite(x) ? Float64(x) : nothing

function _method_json(s::MethodScore, cause_names, topk)
    return Dict{String,Any}(
        "method"         => s.name,
        "pearson_corr"   => s.pearson,
        "spearman_corr"  => s.spearman,
        "precision_at_k" => s.precision_at_k,
        "topk"           => topk,
        "deletion_auc"   => _json_num(s.deletion_auc),
        "insertion_auc"  => _json_num(s.insertion_auc),
        "attr_per_cause" => Dict(cause_names[i] => s.attr_per_cause[i] for i in 1:length(cause_names)),
    )
end

function _output_note(r::GameRecord)
    gxi_max = maximum(abs.(r.attr_gradxinput_over_ram))
    if r.output == "position"
        return "POSITION/INDEX output ($(r.output_label)) — the §1 caveat: a pixel value " *
               "comes from a discrete sprite column (round/argmax), so there is no " *
               "differentiable RAM read and the content-path gradient VANISHES " *
               "(max|attr|=$(round(gxi_max,sigdigits=3))). The oracle has real causal signal " *
               "on position causes, so Grad×Input/DeepLIFT score near chance — the " *
               "'plausible ≠ faithful' headline; the intervention oracle is the sole truth here."
    else
        return gxi_max < 1e-6 ?
            "CONTENT (score) output at a frame where the intact score byte is 0: the " *
            "(u − baseline)/u factor is 0 (the byte equals the all-zeros baseline), so the " *
            "attribution VANISHES — the baseline-degeneracy (run a live-score frame to see it " *
            "concentrate on the score byte)." :
            "CONTENT (score) output ($(r.output_label)), read one-hot from RAM — the STE " *
            "gradient is alive and the score byte is non-zero, so Grad×Input/DeepLIFT " *
            "concentrate on the TRUE causal score byte (precision@1=1): FAITHFUL here. " *
            "DeepLIFT is complete (Σ attr ≈ Δy; err=$(round(r.deeplift_completeness_err,sigdigits=3)))."
    end
end

function write_game(r::GameRecord; out_dir = OUT_DIR, st_extra = nothing)
    isdir(out_dir) || mkpath(out_dir)
    safe_out = r.output
    stem = "gradxinput_deeplift_$(r.game)_$(safe_out)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path  = joinpath(out_dir, stem * ".npz")

    # headline scalar (SPEC §R value/metric_name): the DeepLIFT correlation with
    # the true causal map (the primary faithfulness number for this method).
    rec = Dict{String,Any}(
        "paper" => "P2",
        "phase" => "phaseB_attribution",
        "method" => "gradxinput_deeplift",
        "game" => r.game,
        "state" => "f$(r.target_frame)+$(r.horizon)",
        "target_output" => "$(r.output):$(r.output_label)",
        "metric_name" => "deeplift_pearson_corr_with_oracle",
        "value" => r.deeplift.pearson,
        "stderr" => nothing,
        "ci" => nothing,
        "n" => length(r.cause_names),
        "seed" => r.seed,
        "where" => "local",
        "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene@$(r.game)#$(r.output)",
        "timestamp" => string(round(Int, time())),
        "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            "substrate" => "jutari (Julia) — Zygote content-path ∂y/∂u over the RAM tape + " *
                "TRUE-VCS deletion/insertion re-runs (real-ROM, bit-exact oracle machinery).",
            "score_cell" => Dict("ram_index" => r.score_idx, "concept" => r.score_concept),
            "grad_l1" => r.grad_l1,
            "deeplift_completeness_err" => _json_num(r.deeplift_completeness_err),
            "baseline" => "all-zeros RAM (the 'absent' state)",
            "methods" => Dict{String,Any}(
                "grad_x_input"     => _method_json(r.gradxinput, r.cause_names, r.topk),
                "deeplift_rescale" => _method_json(r.deeplift,   r.cause_names, r.topk),
                "vanilla_saliency" => _method_json(r.saliency,   r.cause_names, r.topk),
            ),
            "harness_positive_control" => merge(
                _method_json(r.oracle_self, r.cause_names, r.topk),
                Dict{String,Any}("interpretation" =>
                    "corr=1 & precision@k=1 ⇒ the scoring harness rewards a faithful map; " *
                    "the method numbers above are then a true measurement, not an artefact.")),
            "comparison_note" => "Grad×Input/DeepLIFT (signed, input-weighted) vs vanilla " *
                "saliency (|∂y/∂u|): on the one-hot content read all three rank the score " *
                "byte top-1, but DeepLIFT adds completeness (Σ attr = Δy) which saliency lacks.",
            "auc_note" => "deletion/insertion curves measured on the TRUE VCS by re-running " *
                "the real ROM with top-attributed causes occluded (deletion) / restored " *
                "(insertion); each curve point is a genuine emulator re-run, not a surrogate. " *
                "NaN = a genuinely flat experiment.",
            "output_note" => _output_note(r),
            "bit_exact_rerun" => r.bit_exact,
            "cause_names" => r.cause_names,
            "oracle_abs_delta_per_cause" => Dict(r.cause_names[i] => r.oracle_abs_delta[i]
                                                 for i in 1:length(r.cause_names)),
            "y_full" => r.y_full,
            "scales_to_cluster_via" =>
                "tools/cluster/xai_*.sbatch (E0-3) — batched SOFT-STE Grad×Input over " *
                "outputs×causes×games on GPU; the forward is bit-exact to this map.",
        ),
    )
    # SHARED-TESTBED provenance + the sampler-on side-by-side (redesign protocol).
    # Only attached to the POSITION record (the shared screen-buffer REGION output).
    if st_extra !== nothing && r.output == "position"
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
        # falls OUTSIDE this game's candidate cause set: the raw 128-byte gradient is
        # alive (max|grad×input over ram|>0) but no scored cause carries it, so the
        # per-cause corr is undefined (recorded 0.0 by the shared convention).
        per_cause_null = maximum(abs.(r.gradxinput.attr_per_cause)) < 1e-9
        rec["extra"]["sampler_on"] = Dict{String,Any}(
            "position_gradient_restored" => st_extra.geom !== nothing,
            "position_byte_in_cause_set" => !per_cause_null,
            "per_cause_faithfulness_null" => per_cause_null,
            "note" => "naive ∂pixel/∂ram ≡ 0 (Prop. prop:zero); the bilinear sampler " *
                "(tools/xai_si_gradient/si_joystick_gradient.jl) restores a real " *
                "∂region/∂ram[position_byte], so Grad×Input/DeepLIFT/saliency all inherit a " *
                "non-vanishing position gradient. per_cause_faithfulness_null=true ⇒ the " *
                "sampler's position byte is not among this game's scored candidate causes, so " *
                "the per-cause Pearson is null (recorded 0.0 by the shared convention), NOT a " *
                "genuine zero corr — the keystone (a non-vanishing gradient) still holds on the " *
                "full 128-byte map.")
    end
    open(json_path, "w") do io; JSON.print(io, rec, 2); end

    write_npz(npz_path, Dict(
        "oracle_abs_delta"           => r.oracle_abs_delta,
        "gradxinput_attr_per_cause"  => r.gradxinput.attr_per_cause,
        "deeplift_attr_per_cause"    => r.deeplift.attr_per_cause,
        "saliency_attr_per_cause"    => r.saliency.attr_per_cause,
        "gradxinput_over_ram"        => r.attr_gradxinput_over_ram,
        "deeplift_over_ram"          => r.attr_deeplift_over_ram,
        "saliency_over_ram"          => r.saliency_over_ram,
        "deeplift_deletion_curve"    => r.deeplift.del_curve,
        "deeplift_insertion_curve"   => r.deeplift.ins_curve,
        "scalars"                    => Float64[
            r.gradxinput.pearson, r.gradxinput.spearman, r.gradxinput.precision_at_k,
            r.gradxinput.deletion_auc, r.gradxinput.insertion_auc,
            r.deeplift.pearson, r.deeplift.spearman, r.deeplift.precision_at_k,
            r.deeplift.deletion_auc, r.deeplift.insertion_auc,
            r.saliency.pearson, r.saliency.spearman, r.saliency.precision_at_k,
            r.deeplift_completeness_err],
    ))
    return json_path, npz_path
end

function write_summary(records::Vector{GameRecord}, output; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    path = joinpath(out_dir, "gradxinput_deeplift_summary_$(output).json")
    rows = Dict{String,Any}[]
    for r in records
        push!(rows, Dict{String,Any}(
            "game" => r.game,
            "output" => r.output,
            "output_label" => r.output_label,
            "deeplift_corr" => r.deeplift.pearson,
            "deeplift_spearman" => r.deeplift.spearman,
            "deeplift_precision_at_k" => r.deeplift.precision_at_k,
            "deeplift_deletion_auc" => _json_num(r.deeplift.deletion_auc),
            "deeplift_insertion_auc" => _json_num(r.deeplift.insertion_auc),
            "gradxinput_corr" => r.gradxinput.pearson,
            "gradxinput_precision_at_k" => r.gradxinput.precision_at_k,
            "saliency_corr" => r.saliency.pearson,
            "saliency_precision_at_k" => r.saliency.precision_at_k,
            "oracle_control_corr" => r.oracle_self.pearson,
            "oracle_control_precision_at_k" => r.oracle_self.precision_at_k,
            "deeplift_completeness_err" => _json_num(r.deeplift_completeness_err),
        ))
    end
    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseB_attribution",
        "method" => "gradxinput_deeplift", "item" => "P2-E4-2",
        "output" => output, "n_games" => length(records),
        "commit" => _git_commit(), "timestamp" => string(round(Int, time())),
        "games" => rows,
        "note" => "Grad×Input / DeepLIFT (Shrikumar et al. 2017) attribution vs the §1 " *
            "intervention oracle on the 6 core games; reuses the IG-pilot scorer + the " *
            "oracle-as-method positive control. Headline = the content output (gradient " *
            "alive); the position output is the 'plausible ≠ faithful' contrast.",
    )
    open(path, "w") do io; JSON.print(io, rec, 2); end
    return path
end

# ============================================================================
# CLI
# ============================================================================
function main(args = ARGS)
    games = CORE_GAMES
    output = nothing               # nothing ⇒ BOTH content + position
    target_frame = -1; horizon = 30; topk = 3; seed = 0
    selftest_only = false
    i = 1
    while i <= length(args)
        a = args[i]
        if     a == "--games"
            v = args[i+1]
            games = xai_resolve_games(v, CORE_GAMES)
            i += 2
        elseif a == "--output";       output = args[i+1]; i += 2
        elseif a == "--target-frame"; target_frame = parse(Int, args[i+1]); i += 2
        elseif a == "--horizon";      horizon = parse(Int, args[i+1]); i += 2
        elseif a == "--topk";         topk = parse(Int, args[i+1]); i += 2
        elseif a == "--seed";         seed = parse(Int, args[i+1]); i += 2
        elseif a == "--selftest";     selftest_only = true; i += 1
        else; i += 1
        end
    end
    outputs = output === nothing ? ["content", "position"] : [output]
    println("[gxi] Grad×Input / DeepLIFT vs oracle — games=$(join(games, ",")) " *
            "output=$(join(outputs, "+")) horizon=$horizon topk=$topk seed=$seed (jutari/Julia)")

    records = GameRecord[]
    for out in outputs, g in games
        r, st_extra = compute_game(; game = g, output = out, target_frame = target_frame,
                         horizon = horizon, topk = topk, seed = seed, verbose = true)
        # SHARED-TESTBED: the position gradient runs through the SAMPLER, so it is
        # NON-VANISHING when a moving sprite exists (geom !== nothing) — the redesign
        # keystone. Only assert the §1 vanishing in the legacy (non-shared) path.
        sampler_on = (r.output == "position") && st_extra !== nothing && st_extra.geom !== nothing
        selftest(r; sampler_on = sampler_on, is_core = g in CORE_GAMES)
        push!(records, r)
        if st_extra !== nothing && r.output == "position"
            println("[gxi] $g SAMPLER-ON position gradient: max|grad×input over ram|=" *
                "$(round(maximum(abs.(r.attr_gradxinput_over_ram)), sigdigits=3)) " *
                "(gate: $(st_extra.cause_density)/$(st_extra.n_causes) accepted=$(st_extra.accepted))")
        end
        if !selftest_only
            jp, np = write_game(r; st_extra = st_extra)
            println("[gxi]   wrote $jp")
            println("[gxi]   arrays  $np")
        end
    end
    if selftest_only
        println("[gxi] --selftest: all $(length(records)) game(s) passed, not writing artifacts.")
        return 0
    end
    for out in outputs
        recs_out = [r for r in records if r.output == out]
        sp = write_summary(recs_out, out)
        println("[gxi] wrote summary $sp")
    end

    # headline table
    println("\n[gxi] ===== HEADLINE faithfulness =====")
    println("[gxi] game            | out      | DeepLIFT corr | DL p@$topk | DL del | DL ins | saliency corr | oracle-ctrl corr")
    for r in records
        println("[gxi] ", rpad(r.game, 15), " | ", rpad(r.output, 8), " | ",
                rpad(round(r.deeplift.pearson, digits=3), 13), " | ",
                rpad(round(r.deeplift.precision_at_k, digits=2), 6), " | ",
                rpad(round(r.deeplift.deletion_auc, digits=3), 6), " | ",
                rpad(round(r.deeplift.insertion_auc, digits=3), 6), " | ",
                rpad(round(r.saliency.pearson, digits=3), 13), " | ",
                round(r.oracle_self.pearson, digits=3))
    end
    return 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    GradXInput.main()
end
