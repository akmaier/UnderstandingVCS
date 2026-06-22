# test_oracle.jl — self-check for the EXACT intervention oracle (P2-E1-1, Julia).
#
# Asserts the properties the DoD requires (and a position-effect check):
#   1. bit-exactness holds (two fresh from-scratch un-intervened re-runs are
#      byte-identical in RAM AND screen);
#   2. a score-cell intervention produces a NONZERO Δ on a score output (the
#      oracle finds a real causal effect on what it's supposed to);
#   3. the negative-control background cell shows EXACTLY 0 Δ across every
#      non-colour cause (the oracle doesn't manufacture spurious effects);
#   4. a ball-position RAM cause moves pixels on the exact framebuffer
#      (n_changed_px > 0) — the "perturb byte → object moves" headline.
#
# Run:
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/ground_truth/test_oracle.jl

using Test

include(joinpath(@__DIR__, "oracle_intervene.jl"))
using .OracleIntervene
using .OracleIntervene: compute_causal_map, assert_bit_exact, resolve_candidates,
                        run_intervention, pong_outputs, build_pong_causes, Cause
using .OracleIntervene.JutariOracle: boot_replay, continue_from

const GAME = "pong"
const TF = 120     # live frame: ball/paddle active (frame 6 only shows score-RAM Δ)
const HZ = 30

# the NOOP action trace used throughout (deterministic)
actions_for() = fill(0, TF + HZ)

@testset "P2-E1-1 intervention oracle (jutari)" begin

    @testset "1. bit-exact re-run (RAM + screen)" begin
        @test assert_bit_exact(actions_for(), TF + HZ; game = GAME) === true
    end

    # compute the full causal map once for the value checks
    cmap = compute_causal_map(; game = GAME, target_frame = TF, horizon = HZ,
                              candidates_path = resolve_candidates(), verbose = false)

    @test cmap.bit_exact === true

    @testset "2. score byte shows nonzero Δ on a score output" begin
        # A directed score intervention: set the P0 score cell (RAM[$0D]) high at
        # the target frame and read the score outputs after the horizon. The score
        # output reads that very RAM cell, so this MUST move (true causal effect).
        checkpoint = boot_replay(actions_for(), TF; game = GAME)
        base = continue_from(checkpoint, Int.(actions_for()[TF + 1 : TF + HZ]))
        outs = pong_outputs(base; checkpoint = checkpoint,
                            target_frame = TF, horizon = HZ)
        score_cause = Cause("ram[13]:set_high", "ram", 13, 19, "set",
                            "enemy_score", "RAM[\$0D] <- 19 (directed)")
        snap = run_intervention(checkpoint, actions_for(), TF, HZ, score_cause)
        j = findfirst(==("p0_score"), [o.name for o in outs])
        Δ = outs[j].read(snap) - outs[j].read(base)
        @info "score-cell Δ(p0_score)" Δ
        @test abs(Δ) > 0
    end

    @testset "3. negative-control bg cell == 0 Δ for non-colour causes" begin
        # The |Δ| on the bg_pixel output across every RAM/joystick cause (i.e.
        # excluding the explicit colour-register cause) must be exactly 0: poking
        # a game-variable byte does not repaint a fixed background cell.
        bg_j = findfirst(n -> startswith(n, "bg_pixel"), cmap.output_names)
        @test bg_j !== nothing
        nontia = [i for (i, m) in enumerate(cmap.cause_meta) if m["kind"] != "tia_reg"]
        bg_deltas = [abs(cmap.delta[i, bg_j]) for i in nontia]
        @info "bg negative-control max |Δ| (non-colour causes)" maxΔ=maximum(bg_deltas)
        @test maximum(bg_deltas) == 0.0
    end

    @testset "4. ball-position cause moves pixels (n_changed_px > 0)" begin
        # A ball RAM byte (ball_y = RAM[54]) perturbed at a live frame must change
        # the exact framebuffer — the headline causal demonstration.
        npx_j = findfirst(==("n_changed_px"), cmap.output_names)
        @test npx_j !== nothing
        ball_rows = [i for (i, m) in enumerate(cmap.cause_meta)
                     if m["kind"] == "ram" && m["index"] in (49, 54)]
        @test !isempty(ball_rows)
        ball_npx = [abs(cmap.delta[i, npx_j]) for i in ball_rows]
        @info "ball-position n_changed_px (max over ball causes)" maxΔ=maximum(ball_npx)
        @test maximum(ball_npx) > 0
    end

end
