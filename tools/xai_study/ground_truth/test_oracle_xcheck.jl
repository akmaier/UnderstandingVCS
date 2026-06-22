# test_oracle_xcheck.jl — self-check for the (1)↔(2) cross-check (P2-E1-3, Julia).
#
# Asserts the DoD properties of the cross-check (experiment_design.md §1):
#   1. the correlation primitives are correct (Pearson/Spearman on known data);
#   2. on the CONTENT path the gradient companion AGREES with the exact oracle
#      (small per-unit residual; correlation near +1 where well-conditioned);
#   3. the INDEX/POSITION point DISAGREES and is FLAGGED (the naive gradient
#      vanishes there while the intervention oracle sees a finite Δ) — disagreement
#      reported, not hidden;
#   4. the report DECLARES the intervention oracle the A/B/C reference.
#
# Run:
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/ground_truth/test_oracle_xcheck.jl

using Test

include(joinpath(@__DIR__, "oracle_xcheck.jl"))
using .OracleXcheck
using .OracleXcheck: compute_xcheck, selftest, pearson, spearman, _avg_ranks

const GAME = "pong"
const TF = 120     # live frame: ball/paddle active (the content+index paths are exercised)
const HZ = 30

@testset "P2-E1-3 oracle cross-check (jutari)" begin

    @testset "0. correlation primitives" begin
        # perfect positive linear relation ⇒ Pearson = Spearman = 1
        x = Float64[1, 2, 3, 4, 5]; y = 2.0 .* x .+ 1.0
        @test isapprox(pearson(x, y), 1.0; atol = 1e-12)
        @test isapprox(spearman(x, y), 1.0; atol = 1e-12)
        # perfect negative monotone (nonlinear) ⇒ Spearman = -1
        ym = Float64[100, 50, 9, 4, 1]   # strictly decreasing in x
        @test isapprox(spearman(x, ym), -1.0; atol = 1e-12)
        # tie handling: average ranks of [3,3,1] are [2.5, 2.5, 1]
        @test _avg_ranks(Float64[3, 3, 1]) == Float64[2.5, 2.5, 1.0]
    end

    # compute the full cross-check once (this RUNS the real ROM on jutari)
    x = compute_xcheck(; game = GAME, target_frame = TF, horizon = HZ, verbose = false)

    @testset "1. content path agrees with the exact oracle" begin
        # forward-exact one-hot content gradient (=1 per unit) PREDICTS the finite
        # intervention Δ on the comparable colour-register path.
        @test x.content_agrees === true
        @test isnan(x.max_abs_residual) || x.max_abs_residual < 1e-6
        # every non-saturated content gradient is +1 (the forward-exact companion)
        ok = .!x.content_saturated
        @test all(g -> isapprox(g, 1.0; atol = 1e-6), x.content_gradients[ok])
        # the variance-bearing predicted-vs-exact sample correlates near +1
        @test !isempty(x.pair_predicted)
        @test isapprox(spearman(x.pair_predicted, x.pair_actual), 1.0; atol = 1e-6)
        @test isapprox(pearson(x.pair_predicted, x.pair_actual), 1.0; atol = 1e-6)
        @info "content correlation" spearman=x.spearman pearson=x.pearson resid=x.max_abs_residual n=length(x.pair_predicted)
    end

    @testset "2. index/position point DISAGREES and is flagged (not hidden)" begin
        idx = findfirst(d -> d.reason == "index", x.disagreements)
        @test idx !== nothing
        # the naive gradient vanishes at the index point ...
        @test abs(x.disagreements[idx].gradient) < 1e-8
        # ... the disagreement is recorded with its reason + note (reported)
        @test x.disagreements[idx].reason == "index"
        @test !isempty(x.disagreements[idx].note)
        @info "index disagreement" grad=x.disagreements[idx].gradient intervention=x.disagreements[idx].intervention
    end

    @testset "3. a non-smooth (sampler-vs-naive) point is flagged" begin
        ns = findfirst(d -> d.reason == "non_smooth", x.disagreements)
        @test ns !== nothing
        # naive index flat, sampler restores a nonzero slope (the workaround)
        @test abs(x.disagreements[ns].gradient) < 1e-8
        @test abs(x.disagreements[ns].intervention) > 1e-6
    end

    @testset "4. the intervention oracle is declared the A/B/C reference" begin
        @test x.oracle_is_reference === true
        # the bundled selftest re-asserts claims (1)-(4) and must pass
        @test selftest(x) === true
    end

end
