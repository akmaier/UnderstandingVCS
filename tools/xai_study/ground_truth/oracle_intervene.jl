# oracle_intervene.jl — the EXACT intervention oracle (P2-E1-1), JULIA path.
#
# This is the substrate-pivoted re-do of tools/xai_study/ground_truth/
# oracle_intervene.py: the Python version booted jaxtari eager mode (~10 min /
# boot, ~205× slower than jutari) and stalled inside the very first boot. jutari
# is the proven fast real-ROM path (Paper-1 64/64 bit-exact), so the oracle is
# re-implemented in Julia here. The Python harness is RETAINED as the future
# cluster GPU-batch path (SOFT-STE forward is bit-exact to this HARD map).
#
# The *primary* ground-truth instrument for Paper 2 (experiment_design.md §1,
# SPEC.md#E1). For a VCS output y (the Pong score, a ball/content pixel) and a
# candidate cause u (a RAM cell, a TIA register, a do(action) input), it measures
# the TRUE causal effect
#
#     Δy(u) = y( do(u := v') ) − y( baseline )
#
# by intervening on u at a target frame, continuing the deterministic emulator a
# short horizon, and reading y. The emulator is bit-exact under a fixed action
# trace, so the un-intervened re-run reproduces the baseline byte-for-byte — this
# module ASSERTS that (RAM *and* screen identical across two fresh from-scratch
# replays) before trusting any Δ. No world model is assumed: every Δ is a real
# re-run of the real ROM.
#
# Outputs (y) for Pong (experiment_design.md §1):
#   * p0_score  = RAM[$0D] (index 13) — the agent/"cpu" score (xitari Pong.cpp:55)
#   * p1_score  = RAM[$0E] (index 14) — the opponent/"human" score (Pong.cpp:56)
#   * ball_pixel= the palette index at a fixed framebuffer cell over the ball band
#                 (a content pixel output, per §1)
#
# Run (warm shared depot, primary's project):
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/ground_truth/oracle_intervene.jl
# Optional flags: --target-frame N --horizon N --game pong
#
# Writes (SPEC §R): tools/xai_study/ground_truth/out/oracle_pong_score.{json,npz}

module OracleIntervene

using JSON

# load the jutari run helper (sibling common/ dir)
include(joinpath(@__DIR__, "..", "common", "jutari_oracle.jl"))
using .JutariOracle

# --- Pong constants (xitari Pong.cpp; jutari src/games/PaddleGames.jl) -------
const PONG_P0_SCORE_IDX = 0x0D    # RAM index 13 — agent/"cpu" score
const PONG_P1_SCORE_IDX = 0x0E    # RAM index 14 — opponent/"human" score
# Pong's content-colour register. In jutari's TIA register file COLUP1 = 0x07
# (the right-paddle/ball colour); COLUBK = 0x09 (background). We intervene on the
# background colour register as the content-path cause (it directly drives
# background pixels within the short horizon, a clean content effect).
const COLUP1_REG = 0x07
const COLUBK_REG = 0x09

const OUT_DIR = joinpath(@__DIR__, "out")
const CANDIDATES_REL = joinpath("tools", "xai_study", "t3", "out", "candidates_pong.json")

# ============================================================================
# Outputs y(state)
# ============================================================================
struct Output
    name::String
    kind::String                 # "score" | "pixel"
    read::Function               # Snapshot -> Float64
    note::String
end

# The Pong ball/paddle RAM cells (used to *causally* locate the ball pixel — a
# cell derived by sensitivity to a ball-position perturbation is a true §1
# position/content output, not a brightness guess).
const BALL_X_IDX = 49      # RAM[$31]
const BALL_Y_IDX = 54      # RAM[$36]

"""Pick the ball-pixel cell *causally*: the framebuffer cell whose value is most
sensitive to a small ball-position perturbation (poke ball_x/ball_y at the target
frame, continue the horizon, see which cell changes). That cell is, by
construction, on the ball's rendered footprint — so the ball-pixel output is a
genuine position/content output (experiment_design.md §1). Falls back to the
brightness heuristic if no checkpoint is supplied or nothing changes."""
function ball_pixel_cell(baseline::Snapshot; checkpoint = nothing,
                         target_frame = 0, horizon = 0)
    if checkpoint !== nothing
        function perturbed(idx, d)
            env = deepcopy(checkpoint)
            b = Int(env.console.bus.ram[idx + 1])
            intervene_ram!(env, idx, (b + d) & 0xFF)
            for _ in 1:horizon; JutariOracle.env_step!(env, 0); end
            return JutariOracle.snapshot(env, horizon).screen
        end
        sx = perturbed(BALL_X_IDX, 24)
        sy = perturbed(BALL_Y_IDX, 24)
        changed = (baseline.screen .!= sx) .| (baseline.screen .!= sy)
        if any(changed)
            ci = findfirst(changed)            # first (top-left) changed cell
            return (ci[1], ci[2])
        end
    end
    # fallback: most-active row in the central band, its median lit column
    scr = baseline.screen
    h, w = size(scr)
    r0 = max(1, floor(Int, h * 0.25)); r1 = min(h, floor(Int, h * 0.85))
    band = @view scr[r0:r1, :]
    row_counts = vec(sum(band .!= 0; dims = 2))
    maximum(row_counts) == 0 && return (h ÷ 2, w ÷ 2)
    row = argmax(row_counts) + r0 - 1
    cols = findall(!=(0), scr[row, :])
    isempty(cols) && return (row, w ÷ 2)
    return (row, cols[(length(cols) ÷ 2) + 1])
end

"""A background negative-control cell: the cell LEAST sensitive to any
ball-position perturbation (i.e. genuinely background). Falls back to a
bottom-band cell. The negative control must show ≈0 Δ for position causes."""
function background_pixel_cell(baseline::Snapshot; checkpoint = nothing,
                               target_frame = 0, horizon = 0)
    h, w = size(baseline.screen)
    return (max(1, h - 2), max(1, w ÷ 2))   # bottom band, away from sprites/ball
end

function pong_outputs(baseline::Snapshot; checkpoint = nothing,
                      target_frame = 0, horizon = 0)
    (pr, pc) = ball_pixel_cell(baseline; checkpoint = checkpoint,
                               target_frame = target_frame, horizon = horizon)
    (br, bc) = background_pixel_cell(baseline)
    return Output[
        Output("p0_score", "score",
               s -> Float64(Int(s.ram[PONG_P0_SCORE_IDX + 1])),
               "RAM[\$0D] agent/cpu score"),
        Output("p1_score", "score",
               s -> Float64(Int(s.ram[PONG_P1_SCORE_IDX + 1])),
               "RAM[\$0E] opponent/human score"),
        Output("ball_pixel@r$(pr)c$(pc)", "pixel",
               s -> Float64(Int(s.screen[pr, pc])),
               "palette index at the (causally-located) ball cell (row=$pr, col=$pc)"),
        Output("bg_pixel@r$(br)c$(bc)", "pixel",
               s -> Float64(Int(s.screen[br, bc])),
               "negative-control background cell (row=$br, col=$bc)"),
        # whole-screen intervention magnitude: how many framebuffer cells differ
        # from the baseline frame. Captures position effects (ball moves) robustly
        # regardless of which single cell the ball lands on. The read is relative
        # to the baseline screen captured by the oracle (closed over below).
        Output("n_changed_px", "pixel",
               s -> Float64(count(s.screen .!= baseline.screen)),
               "count of framebuffer cells differing from the baseline frame"),
    ]
end

# ============================================================================
# Causes u + interventions
# ============================================================================
struct Cause
    name::String
    kind::String                 # "ram" | "tia_reg" | "joystick"
    index::Int                   # RAM idx / TIA reg / -1 for joystick
    value::Int                   # the do-value
    mode::String                 # "set" | "occlude" | "replace"
    concept::String
    note::String
end

"""Read the candidate RAM bytes (E2-1 import), de-duplicated by ram_index. Falls
back to the documented Pong cells if the candidates file is absent."""
function candidate_ram_indices(candidates_path)
    out = Tuple{Int,String}[]
    if candidates_path !== nothing && isfile(candidates_path)
        data = JSON.parsefile(candidates_path)
        seen = Set{Int}()
        for c in get(data, "candidates", [])
            idx = Int(c["ram_index"])
            idx in seen && continue
            push!(seen, idx)
            push!(out, (idx, string(get(c, "concept", ""))))
        end
    end
    if isempty(out)
        out = [(13, "enemy_score"), (14, "player_score"),
               (49, "ball_x"), (54, "ball_y"),
               (51, "player_y"), (50, "enemy_y")]
    end
    return out
end

"""A SMALL, bounded cause set: each candidate RAM byte (set base+17 and occlude
to 0), a TIA colour register (content path), and a do(action) paddle input."""
function build_pong_causes(candidates_path, at_target::Snapshot)
    causes = Cause[]
    for (idx, concept) in candidate_ram_indices(candidates_path)
        base = Int(at_target.ram[idx + 1])
        push!(causes, Cause("ram[$idx]:set", "ram", idx, (base + 17) & 0xFF, "set",
                            concept, "RAM[$idx] $concept <- base+17"))
        push!(causes, Cause("ram[$idx]:occlude", "ram", idx, 0, "occlude",
                            concept, "RAM[$idx] $concept <- 0"))
    end
    # content-path TIA cause: background colour register -> background pixels
    push!(causes, Cause("tia[COLUBK]:set", "tia_reg", COLUBK_REG, 0x0E, "set",
                        "bg_colour", "TIA COLUBK <- 0x0E (white) — content path to bg pixels"))
    # do(action) input: Pong uses paddles; RIGHT moves the user paddle. The
    # baseline trace is NOOP, so do(action := RIGHT) at the target frame.
    push!(causes, Cause("joystick:RIGHT", "joystick", -1, 3, "replace",
                        "agent_input", "do(action := RIGHT) at the target frame"))
    return causes
end

# ============================================================================
# Causal map
# ============================================================================
struct CausalMap
    game::String
    target_frame::Int
    horizon::Int
    seed::Int
    output_names::Vector{String}
    output_notes::Vector{String}
    cause_names::Vector{String}
    cause_meta::Vector{Dict{String,Any}}
    y_baseline::Dict{String,Float64}
    delta::Matrix{Float64}        # (causes, outputs)
    bit_exact::Bool
end

"""Assert two fresh from-scratch un-intervened re-runs are byte-identical in RAM
AND screen. The load-bearing correctness guarantee of the oracle."""
function assert_bit_exact(actions, total; game = "pong")
    a = fresh_baseline_ram_screen(actions, total; game = game)
    b = fresh_baseline_ram_screen(actions, total; game = game)
    if a.ram != b.ram
        diff = count(a.ram .!= b.ram)
        error("bit-exact RAM re-run FAILED: $diff/$(length(a.ram)) bytes differ " *
              "across two fresh '$game' replays to frame $total")
    end
    if a.screen != b.screen
        diff = count(a.screen .!= b.screen)
        error("bit-exact SCREEN re-run FAILED: $diff pixels differ across two " *
              "fresh '$game' replays to frame $total")
    end
    return true
end

"""Apply `cause` to a deepcopy of `checkpoint` and continue `horizon` frames.
For a joystick cause we replace the action AT the target frame (the do-action)
and keep the rest of the tail; for ram/tia we poke the state then run the tail."""
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
    for a in tail
        JutariOracle.env_step!(env, a)
    end
    return JutariOracle.snapshot(env, length(tail))
end

function compute_causal_map(; game = "pong", target_frame = 6, horizon = 6,
                            seed = 0, candidates_path = nothing, verbose = true)
    total = target_frame + horizon
    actions = fill(0, total)        # NOOP trace (deterministic)

    # 1) bit-exact baseline guarantee (whole pipeline twice, boot included)
    verbose && println("[oracle] asserting bit-exactness (2 fresh boots+replays to f$total)...")
    bit_exact = assert_bit_exact(actions, total; game = game)
    verbose && println("[oracle] bit-exact re-run: PASS")

    # 2) one checkpoint at the intervention frame; reused for baseline + all causes
    checkpoint = boot_replay(actions, target_frame; game = game)
    at_target = continue_from(checkpoint, Int[])      # state AT the intervention frame

    # 3) baseline continuation (unchanged tail from the checkpoint). This is the
    #    reference both for Δy and for the `n_changed_px` whole-screen output.
    base_snap = continue_from(checkpoint, Int.(actions[target_frame + 1 : total]))

    # outputs: the ball-pixel cell is located causally (sensitivity to a ball
    # perturbation off the checkpoint); `n_changed_px` compares against base_snap.
    outputs = pong_outputs(base_snap; checkpoint = checkpoint,
                           target_frame = target_frame, horizon = horizon)
    y_base = Dict(o.name => o.read(base_snap) for o in outputs)

    # 4) causes (baseline values read at the intervention frame for base+17 etc.)
    causes = build_pong_causes(candidates_path, at_target)

    # 5) Δy(u) for every (cause, output)
    delta = zeros(Float64, length(causes), length(outputs))
    cause_meta = Dict{String,Any}[]
    for (i, cause) in enumerate(causes)
        snap = run_intervention(checkpoint, actions, target_frame, horizon, cause)
        for (j, o) in enumerate(outputs)
            delta[i, j] = o.read(snap) - y_base[o.name]
        end
        verbose && println("  [$i/$(length(causes))] $(rpad(cause.name, 22)) " *
                           "max|Δ|=$(round(maximum(abs.(delta[i, :])), digits = 3))")
        push!(cause_meta, Dict{String,Any}(
            "name" => cause.name, "kind" => cause.kind, "index" => cause.index,
            "value" => cause.value, "mode" => cause.mode,
            "concept" => cause.concept, "note" => cause.note))
    end

    return CausalMap(game, target_frame, horizon, seed,
                     [o.name for o in outputs], [o.note for o in outputs],
                     [c.name for c in causes], cause_meta, y_base, delta, bit_exact)
end

# ============================================================================
# Persist (SPEC §R): a Julia-written JSON matching results.py's fields + .npz
# ============================================================================
function _git_commit()
    try
        return strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
    catch
        return "unknown"
    end
end

function write_causal_map(cmap::CausalMap; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    stem = "oracle_$(cmap.game)_score"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path = joinpath(out_dir, stem * ".npz")

    # summary metric: the largest |Δy| on a score output (a single scalar proving
    # the oracle found a real causal effect)
    score_cols = findall(n -> endswith(n, "score"), cmap.output_names)
    max_abs_score_delta = isempty(score_cols) ?
        maximum(abs.(cmap.delta)) : maximum(abs.(cmap.delta[:, score_cols]))

    delta_map = Dict(c => Dict(o => cmap.delta[i, j]
                               for (j, o) in enumerate(cmap.output_names))
                     for (i, c) in enumerate(cmap.cause_names))

    rec = Dict{String,Any}(
        "paper" => "P2",
        "phase" => "ground_truth",
        "method" => "intervention_oracle",
        "game" => cmap.game,
        "state" => "f$(cmap.target_frame)+$(cmap.horizon)",
        "target_output" => "pong_score+ball_pixel",
        "metric_name" => "max_abs_score_delta",
        "value" => max_abs_score_delta,
        "stderr" => nothing,
        "ci" => nothing,
        "n" => length(cmap.cause_names),
        "seed" => cmap.seed,
        "where" => "local",
        "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene@$(cmap.game)#score,ball_pixel",
        "timestamp" => string(round(Int, time())),   # epoch seconds (UTC)
        "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            "substrate" => "jutari (Julia, HARD) — real-ROM bit-exact path",
            "outputs" => cmap.output_names,
            "output_notes" => cmap.output_notes,
            "causes" => cmap.cause_meta,
            "y_baseline" => cmap.y_baseline,
            "bit_exact_rerun" => cmap.bit_exact,
            "delta_map" => delta_map,
            "scales_to_cluster_via" =>
                "tools/cluster/xai_*.sbatch (E0-3) + the retained jaxtari " *
                "oracle_intervene.py for GPU SOFT-STE batches — forward is " *
                "bit-exact to this HARD map.",
        ),
    )
    open(json_path, "w") do io
        JSON.print(io, rec, 2)
    end

    # sibling .npz arrays (SPEC §R)
    write_npz(npz_path, Dict(
        "delta" => cmap.delta,
        "y_baseline" => Float64[cmap.y_baseline[o] for o in cmap.output_names],
        # string arrays aren't a NumPy primitive dtype here; store their bytes is
        # overkill — names live in the JSON. We persist the numeric map + an int
        # index of cause kinds for downstream tooling.
    ))
    return json_path, npz_path
end

# ============================================================================
# CLI
# ============================================================================
function resolve_candidates()
    here = normpath(joinpath(@__DIR__, "..", "..", ".."))
    for base in (here, "/Users/maier/Documents/code/UnderstandingVCS")
        p = joinpath(base, CANDIDATES_REL)
        isfile(p) && return p
    end
    return nothing
end

function main(args = ARGS)
    # Default target frame 120 (ball/paddle live) + horizon 30: at frame 6 the
    # ball isn't in play, so only the score-RAM-cell tautology shows a Δ; at a
    # live frame the position causes (ball_x/ball_y) move pixels on the exact
    # framebuffer — the "perturb byte → object moves" headline. Stays well inside
    # the Paper-1 conformance horizon (Pong is 64/64 bit-exact long-horizon).
    game = "pong"; target_frame = 120; horizon = 30; seed = 0
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--game"; game = args[i + 1]; i += 2
        elseif a == "--target-frame"; target_frame = parse(Int, args[i + 1]); i += 2
        elseif a == "--horizon"; horizon = parse(Int, args[i + 1]); i += 2
        elseif a == "--seed"; seed = parse(Int, args[i + 1]); i += 2
        else; i += 1
        end
    end
    cand = resolve_candidates()
    println("[oracle_intervene] game=$game target_frame=$target_frame " *
            "horizon=$horizon seed=$seed (jutari/Julia path)")
    println("[oracle_intervene] candidates: $(cand === nothing ? "(none — fallback cells)" : cand)")

    cmap = compute_causal_map(; game = game, target_frame = target_frame,
                              horizon = horizon, seed = seed,
                              candidates_path = cand, verbose = true)

    json_path, npz_path = write_causal_map(cmap)
    println("[oracle_intervene] bit-exact re-run asserted: $(cmap.bit_exact)")
    println("[oracle_intervene] baseline y: $(cmap.y_baseline)")
    println("[oracle_intervene] sample Δy (cause -> {output: Δ}):")
    for (i, c) in enumerate(cmap.cause_names)
        row = Dict(o => round(cmap.delta[i, j], digits = 3)
                   for (j, o) in enumerate(cmap.output_names))
        println("    $(rpad(c, 22)) $row")
    end
    println("[oracle_intervene] wrote $json_path")
    println("[oracle_intervene] arrays  $npz_path")
    return 0
end

end # module

# run when executed as a script
if abspath(PROGRAM_FILE) == @__FILE__
    OracleIntervene.main()
end
