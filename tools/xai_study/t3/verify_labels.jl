# verify_labels.jl — P2-E2-2: Verify-by-intervention + offset-correct T3 labels.
#
# Upgrades the *candidate* (correlational) T3 labels from P2-E2-1
# (tools/xai_study/t3/out/candidates_<game>.json) to **verified-causal** labels
# by INTERVENTION on the bit-exact jutari substrate (experiment_design.md §2
# step (2); SPEC #E2-2). For every candidate RAM byte:
#
#   1. boot + deterministically replay to a live target frame (inside the
#      Paper-1 conformance horizon), making one reusable checkpoint;
#   2. PERTURB the candidate RAM byte by a few small symmetric deltas, re-render
#      the EXACT framebuffer for a short horizon, and ask: does the rendered
#      frame change?  A byte that changes the framebuffer under do(byte := v')
#      is **verified-causal** — strictly stronger than a probe, which only shows
#      the information is *present* (Hewitt & Liang 2019 control tasks): here we
#      show it is *used* to produce the image.
#   3. OFFSET-CORRECT: for a byte whose perturbation MOVES an object (the
#      changed-pixel region's centroid shifts monotonically along an axis with
#      the byte value), recover the empirical render-coordinate mapping
#      render_pos ≈ slope·byte + intercept by a robust (median-of-pairwise)
#      regression. `intercept` is the recovered render offset (the additive
#      constant AtariARI omits and OCAtari hard-codes); we report it next to the
#      candidate's declared offset so the (x,y) is aligned to render coords.
#
# Verification is honest: a candidate whose byte is overwritten by the CPU
# before it can affect the rendered frame (a scratch/shadow copy, not the live
# render-driving cell) shows NO framebuffer response and is recorded as
# NOT verified. The per-game **verified-rate** = (# unique candidate bytes that
# causally move the framebuffer) / (# unique candidate bytes) — a reportable
# number that quantifies the present-vs-used gap on real Atari ROMs.
#
# No JuTari/jaxtari/xitari core is modified — pure tooling under
# tools/xai_study/, building on common/jutari_oracle.jl (the bit-exact run
# helper) and reusing the same intervention primitive as the P2-E1-1 oracle.
#
# Run (warm shared depot, primary's project):
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/t3/verify_labels.jl
# Optional flags: --games pong breakout ...   --selfcheck-only
#                 --target-frame N (override the per-game default)
#
# Writes (SPEC §R): tools/xai_study/t3/out/verified_<game>.{json,npz}

module VerifyLabels

using JSON
using Statistics

# the bit-exact jutari run helper (P2-E0/E1 shared lib)
include(joinpath(@__DIR__, "..", "common", "jutari_oracle.jl"))
using .JutariOracle

const CORE_GAMES = ["pong", "breakout", "space_invaders", "seaquest", "ms_pacman", "qbert"]

# candidate game-id -> jutari ROM basename (rom_path_for uses "<name>.bin").
# ms_pacman's ROM is mspacman.bin; the others match.
const ROM_NAME = Dict(
    "pong" => "pong", "breakout" => "breakout", "space_invaders" => "space_invaders",
    "seaquest" => "seaquest", "ms_pacman" => "mspacman", "qbert" => "qbert",
)

# Per-game live target frame (inside the Paper-1 bit-exact horizon: ≈30-frame
# RAM / 60-frame screen). A frame where objects are on-screen so a position
# byte's perturbation visibly moves a sprite. Conservative defaults; overridable.
const TARGET_FRAME = Dict(
    "pong" => 120, "breakout" => 120, "space_invaders" => 90,
    "seaquest" => 90, "ms_pacman" => 90, "qbert" => 90,
)

# Verification knobs.
const DELTAS    = (-8, -4, 4, 8)   # small symmetric perturbations (avoid screen wrap)
const HORIZONS  = (1, 2, 3, 4)     # try fast (position) AND slower (HUD redraw) effects
const AXIS_MIN  = 0.5              # min centroid spread (px) to call motion along an axis
const SLOPE_H   = 1               # horizon used for the cleanest spatial offset recovery

const OUT_DIR = joinpath(@__DIR__, "out")

# --------------------------------------------------------------------------- #
# Candidate loading
# --------------------------------------------------------------------------- #
struct ByteGroup
    ram_index::Int
    concepts::Vector{String}          # all candidate concepts mapped to this byte
    sources::Vector{String}
    declared_offsets::Vector{Any}     # OCAtari offsets declared for this byte (may be nothing)
end

"""Read candidates_<game>.json, group candidates by unique RAM index (one
intervention test per byte; track every concept/source that maps to it)."""
function load_byte_groups(candidates_path::AbstractString)
    data = JSON.parsefile(candidates_path)
    by_idx = Dict{Int,ByteGroup}()
    order = Int[]
    for c in get(data, "candidates", [])
        idx = Int(c["ram_index"])
        if !haskey(by_idx, idx)
            by_idx[idx] = ByteGroup(idx, String[], String[], Any[])
            push!(order, idx)
        end
        g = by_idx[idx]
        concept = string(get(c, "concept", ""))
        concept == "nothing" && (concept = "")  # JSON null label
        push!(g.concepts, concept)
        push!(g.sources, string(get(c, "source", "")))
        push!(g.declared_offsets, get(c, "offset", nothing))
    end
    return [by_idx[i] for i in order], data
end

# --------------------------------------------------------------------------- #
# Intervention rendering
# --------------------------------------------------------------------------- #
"""Render the framebuffer `horizon` steps after poking RAM[`idx`] := `val` on a
deepcopy of the checkpoint (NOOP tail)."""
function render_poked(checkpoint, idx::Int, val::Int, horizon::Int)
    env = deepcopy(checkpoint)
    JutariOracle.intervene_ram!(env, idx, val)
    for _ in 1:horizon
        JutariOracle.env_step!(env, 0)
    end
    return JutariOracle.snapshot(env, horizon).screen
end

"""The un-intervened baseline frame at a given horizon (NOOP tail) — the
reference every perturbed render is compared against."""
baseline_screen(checkpoint, horizon::Int) =
    JutariOracle.continue_from(checkpoint, fill(0, horizon)).screen

"""Robust slope = median of all pairwise (Δpos / Δbyte). Resistant to the single
screen-wrap outlier a least-squares fit would be skewed by."""
function median_pairwise_slope(xs::Vector{Float64}, ys::Vector{Float64})
    length(xs) < 2 && return NaN
    slopes = Float64[]
    @inbounds for i in 1:length(xs), j in (i + 1):length(xs)
        xs[j] != xs[i] && push!(slopes, (ys[j] - ys[i]) / (xs[j] - xs[i]))
    end
    isempty(slopes) ? NaN : median(slopes)
end

"""Intercept for a fixed (robust) slope: median(y - slope·x). The recovered
render offset = render_pos at byte 0."""
function intercept_for_slope(xs::Vector{Float64}, ys::Vector{Float64}, slope::Float64)
    (isnan(slope) || isempty(xs)) && return NaN
    return median(ys .- slope .* xs)
end

struct ByteVerdict
    ram_index::Int
    base_value::Int
    verified::Bool
    max_changed_px::Int
    resp_per_horizon::Vector{Int}     # max |changed px| at each tested horizon
    motion_axis::String               # "x" | "y" | "none" (no spatial motion)
    moves_object::Bool
    slope::Float64                    # render-pos per byte unit (along motion_axis)
    recovered_offset::Float64         # intercept = render offset at byte 0 (along axis)
    declared_offset::Any              # OCAtari-declared offset for this byte (if any)
    reg_bytes::Vector{Float64}        # regression x (byte values) — provenance
    reg_pos::Vector{Float64}          # regression y (centroid pos along axis)
    concepts::Vector{String}
    sources::Vector{String}
end

"""Run the full intervention test on one candidate byte. Verified iff perturbing
the byte changes the framebuffer at SOME tested horizon. If it does, classify
the motion axis at horizon SLOPE_H and recover the empirical render offset."""
function verify_byte(checkpoint, g::ByteGroup, base_value::Int)
    # 1) verification + per-horizon response magnitude (max over deltas)
    resp = Int[]
    for h in HORIZONS
        base = baseline_screen(checkpoint, h)
        mx = 0
        for d in DELTAS
            v = clamp(base_value + d, 0, 255)
            v == base_value && continue
            n = count(render_poked(checkpoint, g.ram_index, v, h) .!= base)
            mx = max(mx, n)
        end
        push!(resp, mx)
    end
    verified = any(resp .> 0)
    max_changed = isempty(resp) ? 0 : maximum(resp)

    # 2) spatial classification + offset recovery (only if it moves something)
    axis = "none"; moves = false; slope = NaN; offset = NaN
    reg_bytes = Float64[]; reg_row = Float64[]; reg_col = Float64[]
    if verified
        base = baseline_screen(checkpoint, SLOPE_H)
        for d in DELTAS
            v = clamp(base_value + d, 0, 255)
            v == base_value && continue
            scr = render_poked(checkpoint, g.ram_index, v, SLOPE_H)
            ch = findall(scr .!= base)
            isempty(ch) && continue
            push!(reg_bytes, Float64(v))
            push!(reg_row, mean(c[1] for c in ch))   # row = y
            push!(reg_col, mean(c[2] for c in ch))   # col = x
        end
        if length(reg_bytes) >= 2
            rspread = maximum(reg_row) - minimum(reg_row)
            cspread = maximum(reg_col) - minimum(reg_col)
            if cspread >= rspread && cspread > AXIS_MIN
                axis = "x"; moves = true
                slope = median_pairwise_slope(reg_bytes, reg_col)
                offset = intercept_for_slope(reg_bytes, reg_col, slope)
            elseif rspread > AXIS_MIN
                axis = "y"; moves = true
                slope = median_pairwise_slope(reg_bytes, reg_row)
                offset = intercept_for_slope(reg_bytes, reg_row, slope)
            end
        end
    end

    # the OCAtari-declared offset for this byte (first non-nothing, if any)
    declared = nothing
    for o in g.declared_offsets
        if o !== nothing; declared = o; break; end
    end

    reg_pos = axis == "x" ? reg_col : (axis == "y" ? reg_row : Float64[])
    return ByteVerdict(g.ram_index, base_value, verified, max_changed, resp,
                       axis, moves, slope, offset, declared,
                       reg_bytes, reg_pos,
                       unique(g.concepts), unique(g.sources))
end

# --------------------------------------------------------------------------- #
# Per-game verification
# --------------------------------------------------------------------------- #
struct GameResult
    game::String
    target_frame::Int
    n_bytes::Int
    n_verified::Int
    verified_rate::Float64
    n_moving::Int
    bit_exact::Bool
    verdicts::Vector{ByteVerdict}
end

"""Assert the un-intervened run is bit-exact-reproducible (two fresh boots+replays
to the target frame are byte-identical in RAM AND screen) — the load-bearing
guarantee that makes every Δ a clean causal effect."""
function assert_bit_exact(game_rom::String, target_frame::Int)
    total = target_frame + maximum(HORIZONS)
    actions = fill(0, total)
    a = JutariOracle.fresh_baseline_ram_screen(actions, total; game = game_rom)
    b = JutariOracle.fresh_baseline_ram_screen(actions, total; game = game_rom)
    a.ram == b.ram || error("$game_rom: non-deterministic RAM re-run " *
                            "($(count(a.ram .!= b.ram)) bytes differ)")
    a.screen == b.screen || error("$game_rom: non-deterministic SCREEN re-run " *
                                  "($(count(a.screen .!= b.screen)) px differ)")
    return true
end

function verify_game(game::AbstractString; candidates_path::AbstractString,
                     target_frame::Union{Nothing,Int} = nothing, verbose = true)
    game = String(game)
    rom = get(ROM_NAME, game, game)
    tf = target_frame === nothing ? get(TARGET_FRAME, game, 90) : target_frame

    verbose && println("\n[verify] $game (rom=$rom) target_frame=$tf")
    bit_exact = assert_bit_exact(rom, tf)
    verbose && println("[verify]   bit-exact re-run: PASS")

    groups, _ = load_byte_groups(candidates_path)

    # ONE checkpoint at the live frame; reused for the baseline + every poke.
    actions = fill(0, tf + maximum(HORIZONS))
    checkpoint = JutariOracle.boot_replay(actions[1:tf], tf; game = rom)
    at_target = JutariOracle.continue_from(checkpoint, Int[])  # state AT the frame

    verdicts = ByteVerdict[]
    for g in groups
        base_value = Int(at_target.ram[g.ram_index + 1])
        v = verify_byte(checkpoint, g, base_value)
        push!(verdicts, v)
        if verbose
            tag = v.verified ? (v.moves_object ?
                  "VERIFIED(moves $(v.motion_axis), slope=$(round(v.slope, digits = 2)) " *
                  "offset=$(round(v.recovered_offset, digits = 1)))" :
                  "VERIFIED(non-spatial)") : "not-verified(inert)"
            println("  ram[$(lpad(g.ram_index, 3))] base=$(lpad(base_value, 3)) " *
                    "resp=$(v.resp_per_horizon) $tag  " *
                    "[$(join(filter(!isempty, v.concepts), ","))]")
        end
    end

    n_bytes = length(verdicts)
    n_ver = count(v -> v.verified, verdicts)
    n_mov = count(v -> v.moves_object, verdicts)
    rate = n_bytes == 0 ? 0.0 : n_ver / n_bytes
    verbose && println("[verify]   verified-rate = $n_ver/$n_bytes = " *
                       "$(round(100 * rate, digits = 1))%  (moving objects: $n_mov)")
    return GameResult(game, tf, n_bytes, n_ver, rate, n_mov, bit_exact, verdicts)
end

# --------------------------------------------------------------------------- #
# Persist (SPEC §R): JSON record + sibling .npz
# --------------------------------------------------------------------------- #
function _git_commit()
    try
        return strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
    catch
        return "unknown"
    end
end

function verdict_dict(v::ByteVerdict)
    Dict{String,Any}(
        "ram_index" => v.ram_index,
        "ram_addr_hex" => "0x" * uppercase(string(0x80 + v.ram_index, base = 16, pad = 2)),
        "base_value" => v.base_value,
        "verified" => v.verified,
        "max_changed_px" => v.max_changed_px,
        "resp_per_horizon" => v.resp_per_horizon,
        "horizons" => collect(HORIZONS),
        "motion_axis" => v.motion_axis,
        "moves_object" => v.moves_object,
        "render_slope_per_byte" => isnan(v.slope) ? nothing : v.slope,
        "recovered_offset" => isnan(v.recovered_offset) ? nothing : v.recovered_offset,
        "declared_offset" => v.declared_offset,
        "offset_agrees" => (v.declared_offset !== nothing && !isnan(v.recovered_offset)) ?
            (abs(Float64(v.declared_offset) - v.recovered_offset) <= 4.0) : nothing,
        "regression_bytes" => v.reg_bytes,
        "regression_render_pos" => v.reg_pos,
        "concepts" => filter(!isempty, v.concepts),
        "sources" => v.sources,
        "status" => v.verified ? "verified_causal" : "unverified_inert",
    )
end

function write_game_result(r::GameResult; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    stem = "verified_$(r.game)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path = joinpath(out_dir, stem * ".npz")

    rec = Dict{String,Any}(
        "paper" => "P2",
        "phase" => "E2-T3",
        "method" => "verify_by_intervention",
        "game" => r.game,
        "frame" => r.target_frame,
        "state" => "f$(r.target_frame)+{$(join(HORIZONS, ","))}",
        "target_output" => "verified_causal_ram_to_concept_map",
        "metric_name" => "verified_rate",
        "value" => r.verified_rate,
        "ci" => nothing,
        "stderr" => nothing,
        "n" => r.n_bytes,
        "seed" => 0,
        "where" => "local",
        "commit" => _git_commit(),
        "oracle_ref" => "verify_labels@$(r.game) (intervention on jutari, " *
                        "common/jutari_oracle.jl intervene_ram!)",
        "timestamp" => string(round(Int, time())),
        "arrays" => basename(npz_path),
        "n_verified" => r.n_verified,
        "n_unverified" => r.n_bytes - r.n_verified,
        "n_moving_objects" => r.n_moving,
        "bit_exact_rerun" => r.bit_exact,
        "deltas" => collect(DELTAS),
        "horizons" => collect(HORIZONS),
        "substrate" => "jutari (Julia, HARD) — Paper-1 64/64 bit-exact real-ROM path",
        "method_note" =>
            "perturb candidate RAM byte by small symmetric deltas, re-render the " *
            "bit-exact framebuffer over horizons $(collect(HORIZONS)); a byte that " *
            "changes the frame is verified-causal (strictly stronger than probing: " *
            "info present vs used). Offset = intercept of robust median-pairwise " *
            "regression of changed-region centroid vs byte value along the motion axis.",
        "upgrades" => "tools/xai_study/t3/out/candidates_$(r.game).json (P2-E2-1)",
        "verdicts" => [verdict_dict(v) for v in r.verdicts],
    )
    open(json_path, "w") do io
        JSON.print(io, rec, 2)
    end

    # sibling arrays (SPEC §R): per-byte numeric summary, NumPy-loadable.
    n = r.n_bytes
    idxs = Float64[Float64(v.ram_index) for v in r.verdicts]
    bases = Float64[Float64(v.base_value) for v in r.verdicts]
    ver = Float64[v.verified ? 1.0 : 0.0 for v in r.verdicts]
    moves = Float64[v.moves_object ? 1.0 : 0.0 for v in r.verdicts]
    slopes = Float64[isnan(v.slope) ? 0.0 : v.slope for v in r.verdicts]
    offs = Float64[isnan(v.recovered_offset) ? 0.0 : v.recovered_offset for v in r.verdicts]
    maxpx = Float64[Float64(v.max_changed_px) for v in r.verdicts]
    resp = zeros(Float64, max(n, 1), length(HORIZONS))
    for (i, v) in enumerate(r.verdicts), (j, _) in enumerate(HORIZONS)
        resp[i, j] = Float64(v.resp_per_horizon[j])
    end
    JutariOracle.write_npz(npz_path, Dict(
        "ram_index" => idxs,
        "base_value" => bases,
        "verified" => ver,
        "moves_object" => moves,
        "render_slope_per_byte" => slopes,
        "recovered_offset" => offs,
        "max_changed_px" => maxpx,
        "resp_per_horizon" => resp,
    ))
    return json_path, npz_path
end

# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #
"""A focused self-check on Pong (no file writes): the bit-exact guarantee holds,
the known ball-position cells (ball_x=idx49, ball_y=idx54) are verified-causal
and move the object spatially, and a clearly-non-rendering scratch cell shows the
intervention test discriminates (verified-rate strictly between 0 and 1 over the
candidate set). Asserts; returns true on success."""
function selfcheck(; verbose = true)
    candidates = resolve_candidates("pong")
    candidates === nothing && error("self-check: pong candidates file not found")
    r = verify_game("pong"; candidates_path = candidates, verbose = verbose)

    @assert r.bit_exact "self-check: bit-exact re-run failed"
    byidx = Dict(v.ram_index => v for v in r.verdicts)

    # ball cells must be verified AND move an object spatially
    for (nm, idx, want_axis) in (("ball_x", 49, "x"), ("ball_y", 54, "y"))
        haskey(byidx, idx) || error("self-check: candidate byte $idx ($nm) missing")
        v = byidx[idx]
        @assert v.verified "self-check: $nm (idx $idx) should be verified-causal"
        @assert v.moves_object "self-check: $nm (idx $idx) should move an object"
        @assert v.motion_axis == want_axis "self-check: $nm axis $(v.motion_axis) != $want_axis"
        @assert !isnan(v.recovered_offset) "self-check: $nm offset not recovered"
    end

    # the intervention test must DISCRIMINATE: not everything verifies (probing
    # would; intervention is stronger). Rate strictly inside (0, 1).
    @assert 0.0 < r.verified_rate < 1.0 (
        "self-check: verified-rate $(r.verified_rate) not in (0,1) — " *
        "intervention test failed to discriminate present-vs-used")

    verbose && println("\n[selfcheck] PASS — bit-exact, ball cells verified+moving, " *
                       "verified-rate=$(round(100 * r.verified_rate, digits = 1))% " *
                       "discriminates ($(r.n_verified)/$(r.n_bytes)).")
    return true
end

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
function resolve_candidates(game::AbstractString)
    here = normpath(joinpath(@__DIR__, "..", "..", ".."))
    rel = joinpath("tools", "xai_study", "t3", "out", "candidates_$(game).json")
    for base in (here, "/Users/maier/Documents/code/UnderstandingVCS")
        p = joinpath(base, rel)
        isfile(p) && return p
    end
    # also look directly next to this file's out/
    p = joinpath(OUT_DIR, "candidates_$(game).json")
    isfile(p) && return p
    return nothing
end

function main(args = ARGS)
    games = String[]
    target_frame = nothing
    selfcheck_only = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--games"
            i += 1
            while i <= length(args) && !startswith(args[i], "--")
                push!(games, args[i]); i += 1
            end
        elseif a == "--target-frame"
            target_frame = parse(Int, args[i + 1]); i += 2
        elseif a == "--selfcheck-only"
            selfcheck_only = true; i += 1
        else
            i += 1
        end
    end
    isempty(games) && (games = CORE_GAMES)

    if selfcheck_only
        selfcheck(; verbose = true)
        return 0
    end

    println("[verify_labels] P2-E2-2 verify-by-intervention (jutari/Julia path)")
    println("[verify_labels] games: $(join(games, ", "))")

    summary = Tuple{String,Int,Int,Float64,Int}[]
    for game in games
        cand = resolve_candidates(game)
        if cand === nothing
            println("[skip] $game: no candidates file (run P2-E2-1 first)")
            continue
        end
        r = verify_game(game; candidates_path = cand, target_frame = target_frame, verbose = true)
        jp, np = write_game_result(r)
        println("[verify_labels]   wrote $jp")
        println("[verify_labels]   arrays $np")
        push!(summary, (game, r.n_verified, r.n_bytes, r.verified_rate, r.n_moving))
    end

    println("\n[verify_labels] === per-game verified-rate (unique candidate bytes) ===")
    for (g, nv, nb, rate, nm) in summary
        println("  $(rpad(g, 16)) $(lpad(nv, 3))/$(lpad(nb, 3)) = " *
                "$(rpad(round(100 * rate, digits = 1), 5))%   (moving objects: $nm)")
    end

    # always run the focused self-check at the end (over Pong already in summary)
    println()
    selfcheck(; verbose = true)
    return 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    VerifyLabels.main()
end
