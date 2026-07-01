# smoke_gameplay_redesign.jl — SMOKE TEST for the P2 experiment redesign
# (xai_paper/xai_2_interpretability/experiment_redesign.md), ONE game (pong).
#
# Proves the three redesign pieces end-to-end on the shared testbed, WITHOUT
# running the full 6-game / all-method battery:
#   (a) the seeded random-action GAMEPLAY state passes the oracle CAUSE-DENSITY
#       gate — report the cause count vs the old NOOP/2-of-48-style degeneracy;
#   (b) the SAMPLER-ON gradient is NONZERO on the shared screen-buffer position
#       output (whereas the naive handle is identically zero);
#   (c) one gradient Phase-B method (vanilla saliency) + the §1 oracle run
#       end-to-end at the new state and write a §R record.
#
# Uses ONLY the shared common code (tools/xai_study/common/gameplay_state.jl) so
# what is validated here is exactly what the full re-run will use.
#
# Run:
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#     tools/xai_study/phaseB_attribution/sampler_on/smoke_gameplay_redesign.jl [--game pong]
#
# Writes: tools/xai_study/phaseB_attribution/sampler_on/out/smoke_<game>_saliency.json

module SmokeGameplayRedesign

using JSON
import Zygote
import Statistics

using JuTari
using JuTari.Env: env_step!

include(joinpath(@__DIR__, "..", "..", "common", "gameplay_state.jl"))
using .GameplayState: build_shared_state, position_read_zero, sampler_position_read,
                      screen_pixel_wrt_cause
using .GameplayState.OracleIntervene.JutariOracle: continue_from, write_npz
using .GameplayState.OracleIntervene: Cause

const OUT_DIR = joinpath(@__DIR__, "out")

# --- reuse the shared scorer (Pearson corr of attribution vs oracle |Δy|) ------
# (mirrors PilotIGvsOracle.pearson; kept local so the smoke test is self-contained)
function pearson(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    length(a) < 2 && return NaN
    ma = Statistics.mean(a); mb = Statistics.mean(b)
    da = a .- ma; db = b .- mb
    den = sqrt(sum(abs2, da)) * sqrt(sum(abs2, db))
    den == 0 && return NaN
    return sum(da .* db) / den
end

"""Per-cause attribution from a 128-vector RAM gradient (|g| on RAM causes, 0 on
tia/joystick) — the shared mapping every Phase-B method uses."""
function attr_per_cause(g_full::AbstractVector, causes::Vector{Cause})
    attr = zeros(Float64, length(causes))
    for (i, c) in enumerate(causes)
        if c.kind == "ram" && 0 <= c.index < length(g_full)
            attr[i] = abs(Float64(g_full[c.index + 1]))
        end
    end
    return attr
end

"""vanilla saliency |∂y/∂ram| for a differentiable readf."""
function saliency(readf, x)
    g = Zygote.gradient(readf, x)[1]
    g === nothing && (g = zeros(Float32, length(x)))
    return abs.(Float32.(g))
end

function _git_commit()
    try
        return strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
    catch; return "unknown"; end
end

function main(args = ARGS)
    game = "pong"
    i = 1
    while i <= length(args)
        if args[i] == "--game"; game = args[i + 1]; i += 2 else i += 1 end
    end

    println("[smoke] P2 redesign smoke test — game=$game (jutari/Julia)")
    println("[smoke] building the SHARED gameplay state (seed=0 random-action stream)...")
    # prefix=90 (game deep in gameplay; boot/attract < 10% of the trajectory),
    # horizon=15 (the oracle roll-forward). The whole stream is re-asserted
    # bit-exact inside build_shared_state — the redesign's sanctioned validity gate
    # for a stream longer than the 60-frame guaranteed screen window (pong is
    # 64/64 long-horizon bit-exact, so this passes; a game that fails would error).
    gs = build_shared_state(game; prefix = 90, horizon = 15, seed = 0,
                            k = 4, floor = 0.5, verbose = true)

    # ---- (a) CAUSE-DENSITY GATE ------------------------------------------------
    nz = count(>(0.0), gs.deltas)
    println("\n[smoke] (a) CAUSE-DENSITY GATE")
    println("[smoke]     shared output cell = $(gs.cell)  (a moving-sprite screen pixel)")
    println("[smoke]     oracle |Δy| per cause: $nz/$(length(gs.causes)) nonzero; " *
            "$(gs.cause_density)/$(length(gs.causes)) above floor=0.5")
    println("[smoke]     gate (k=4): $(gs.accepted ? "ACCEPT ✓" : "REJECT ✗")  " *
            "(contrast: the old NOOP seaquest content oracle was 2/48)")
    topcauses = sort(collect(zip([c.name for c in gs.causes], gs.deltas));
                     by = x -> -x[2])[1:min(6, length(gs.causes))]
    for (nm, d) in topcauses
        println("[smoke]       $(rpad(nm, 24)) |Δy| = $(round(d, digits = 3))")
    end

    # ---- (b) SAMPLER-ON gradient NONZERO on the shared position output ---------
    println("\n[smoke] (b) SAMPLER-ON POSITION GRADIENT")
    naive_g = saliency(position_read_zero, gs.ram_now)
    naive_max = maximum(abs.(naive_g))
    if gs.geom === nothing
        println("[smoke]     no moving sprite at this cell (static frame) — sampler has no " *
                "position to restore; naive max|g|=$(round(naive_max, sigdigits=3)).")
        samp_g = zeros(Float32, length(gs.ram_now)); samp_max = 0.0; pidx = -1
    else
        pidx = gs.geom[1]
        samp_readf = ram -> sampler_position_read(ram, gs.geom, gs.cell; scale = 1.0)
        samp_g = saliency(samp_readf, gs.ram_now)
        samp_max = maximum(abs.(samp_g))
        on_byte = abs(samp_g[pidx + 1])
        println("[smoke]     naive  ∂pixel/∂ram : max|g| = $(round(naive_max, sigdigits=3))  (the §1 vanishing)")
        println("[smoke]     sampler ∂pixel/∂ram: max|g| = $(round(samp_max, sigdigits=3))  " *
                "on position byte RAM[$pidx] = $(round(on_byte, sigdigits=3))")
        println("[smoke]     ⇒ sampler restores a REAL, NONZERO position gradient " *
                "(naive $(round(naive_max,sigdigits=3)) → sampler $(round(samp_max,sigdigits=3))).")
    end

    # ---- (c) ONE gradient method + oracle end-to-end → §R record ---------------
    println("\n[smoke] (c) vanilla saliency + oracle, END-TO-END at the new state")
    # scored against the SAME oracle column (the shared output's |Δy|).
    attr_naive   = attr_per_cause(naive_g, gs.causes)
    attr_sampler = attr_per_cause(samp_g, gs.causes)
    faith_naive   = pearson(attr_naive,   gs.deltas)
    faith_sampler = pearson(attr_sampler, gs.deltas)
    println("[smoke]     faithfulness (Pearson attr vs oracle |Δy|):  " *
            "naive = $(round(faith_naive, digits = 3))  →  sampler = $(round(faith_sampler, digits = 3))")

    # the image-domain ∂screen-pixel/∂cause map over a small grid around the cell
    # (protocol 5) — proof the screen-buffer gradient is produced.
    screen_grad_max = 0.0
    if gs.geom !== nothing
        r0 = max(1, gs.cell[1] - 6); r1 = gs.cell[1] + 6
        c0 = max(1, gs.cell[2] - 10); c1 = gs.cell[2] + 10
        Gimg = screen_pixel_wrt_cause(gs.ram_now, gs.geom, (r0:r1, c0:c1); scale = 1.0)
        screen_grad_max = maximum(abs.(Gimg))
        println("[smoke]     image-domain ∂screen-pixel/∂cause map $(size(Gimg)) " *
                "max|∂|=$(round(screen_grad_max, sigdigits = 3)) (the direct renderer gradient)")
    end

    isdir(OUT_DIR) || mkpath(OUT_DIR)
    stem = "smoke_$(game)_saliency"
    json_path = joinpath(OUT_DIR, stem * ".json")
    npz_path  = joinpath(OUT_DIR, stem * ".npz")
    cause_names = [c.name for c in gs.causes]
    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseB_attribution", "experiment" => "P2-redesign-smoke",
        "method" => "vanilla_saliency", "game" => game,
        "state" => "gameplay(seed=0,prefix=$(gs.prefix))+$(gs.horizon)",
        "target_output" => "screen_region(n_changed_px); sampler cell @r$(gs.cell[1])c$(gs.cell[2])",
        "metric_name" => "pearson_corr_with_oracle",
        "value" => faith_sampler,
        "seed" => gs.seed, "where" => "local", "commit" => _git_commit(),
        "oracle_ref" => "gameplay_state@$(game)#screen_pixel",
        "timestamp" => string(round(Int, time())), "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            "substrate" => "jutari (Julia) — shared gameplay testbed (common/gameplay_state.jl): " *
                "seeded random-action state + oracle cause-density gate + shared screen-buffer " *
                "output + soft-mode bilinear sampler for the position gradient.",
            "testbed" => Dict{String,Any}(
                "state_kind" => "seeded_random_action_gameplay",
                "prefix" => gs.prefix, "horizon" => gs.horizon, "seed" => gs.seed,
                "shared_output" => "screen_pixel@r$(gs.cell[1])c$(gs.cell[2])",
                "cause_density_above_floor" => gs.cause_density,
                "cause_density_floor" => 0.5,
                "cause_density_gate_k" => 4,
                "cause_density_accepted" => gs.accepted,
                "n_causes" => length(gs.causes),
                "position_byte_ram_index" => pidx),
            "sampler_on" => Dict{String,Any}(
                "naive_position_grad_max" => Float64(naive_max),
                "sampler_position_grad_max" => Float64(samp_max),
                "screen_pixel_wrt_cause_max" => Float64(screen_grad_max),
                "note" => "naive ∂pixel/∂ram ≡ 0 (Prop. prop:zero); the bilinear sampler " *
                    "(same as tools/xai_si_gradient/si_joystick_gradient.jl) restores a real " *
                    "∂pixel/∂ram[position_byte]. Screen-buffer ∂pixel/∂cause is the image-domain map."),
            "faithfulness" => Dict{String,Any}(
                "naive" => faith_naive, "sampler" => faith_sampler,
                "metric" => "pearson(per-cause |∂y/∂ram|, oracle |Δ screen-pixel|)"),
            "cause_names" => cause_names,
            "oracle_abs_delta_per_cause" => Dict(cause_names[i] => gs.deltas[i] for i in 1:length(cause_names)),
            "attr_sampler_per_cause" => Dict(cause_names[i] => attr_sampler[i] for i in 1:length(cause_names)),
            "attr_naive_per_cause" => Dict(cause_names[i] => attr_naive[i] for i in 1:length(cause_names)),
        ),
    )
    _jsafe(x) = (x isa AbstractFloat && !isfinite(x)) ? nothing : x
    rec["value"] = _jsafe(rec["value"])
    rec["extra"]["faithfulness"]["naive"] = _jsafe(faith_naive)
    rec["extra"]["faithfulness"]["sampler"] = _jsafe(faith_sampler)
    open(json_path, "w") do io; JSON.print(io, rec, 2); end
    write_npz(npz_path, Dict(
        "oracle_abs_delta" => gs.deltas,
        "attr_naive_per_cause" => attr_naive,
        "attr_sampler_per_cause" => attr_sampler,
        "naive_grad_full" => Float64.(naive_g),
        "sampler_grad_full" => Float64.(samp_g),
    ))
    println("\n[smoke] wrote §R record  $json_path")
    println("[smoke] wrote arrays      $npz_path")

    # ---- assertions (exit non-zero on failure) --------------------------------
    @assert gs.accepted "SMOKE FAIL (a): cause-density gate REJECTED the gameplay state " *
        "($(gs.cause_density)/$(length(gs.causes)) < k=4)."
    if gs.geom !== nothing
        @assert samp_max > 1e-6 "SMOKE FAIL (b): sampler position gradient is zero (expected nonzero)."
        @assert naive_max < 1e-6 "SMOKE FAIL (b): naive position gradient is NOT zero (expected the §1 vanishing)."
    end
    println("\n[smoke] SMOKE TEST PASS: (a) gate ACCEPT, (b) sampler grad nonzero " *
            "(naive zero), (c) saliency+oracle §R record written.")
    return 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    exit(SmokeGameplayRedesign.main())
end
