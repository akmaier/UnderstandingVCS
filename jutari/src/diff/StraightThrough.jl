# Straight-through estimator helpers — hard forward, identity backward.
#
# `straight_through_round` rounds to nearest in the forward pass;
# `straight_through_clamp` clips to [lo, hi]. For both, the BACKWARD
# pass is the identity (modulo the inside-interval mask for clamp).
#
# Forward-behaviour implementations here. Identity-backward Jacobian
# wiring uses ChainRulesCore.@scalar_rule / @non_differentiable when
# the differentiability layer is fully wired up in P7b; for now the
# forward behaviour matches the jaxtari counterpart and the autodiff
# rrules are deferred.

"""
    straight_through_round(x) -> Float32

Round to nearest. Backward: identity (when wired into an autodiff
stack — see module docstring).
"""
@inline straight_through_round(x::Real) = Float32(round(x))

"""
    straight_through_clamp(x, lo, hi) -> Float32

Clip to [lo, hi]. Backward: 1 inside the interval, 0 outside.
"""
@inline function straight_through_clamp(x::Real, lo::Real, hi::Real)
    xf = Float32(x)
    return Float32(clamp(xf, Float32(lo), Float32(hi)))
end
