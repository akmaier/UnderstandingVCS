# Straight-through estimator helpers — hard forward, identity backward.
#
# `straight_through_round` rounds to nearest in the forward pass;
# `straight_through_clamp` clips to [lo, hi]. For both, the BACKWARD
# pass is the identity (modulo the inside-interval mask for clamp).
#
# Paper reference: the round/clamp companions of the straight-through
# estimator STE(soft, hard) = soft + sg(hard - soft) (Eq. "ste";
# supplementary fourth primitive) — "the round and clamp operations are
# treated the same way (forward exact, backward identity inside the valid
# range)" ("Hard and Soft Execution"). These keep the soft forward pass
# bit-exact to the hard one (Theorem 1) while exposing surrogate
# gradients (Corollary 1); cf. Bengio et al. 2013. (The branch STE
# itself is built in Modes._stop_gradient + SoftStep._func_do_branch.)
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
