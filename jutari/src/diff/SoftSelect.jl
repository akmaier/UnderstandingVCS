# Soft-selection primitive — softmax-weighted mixture over choices.
#
# Used for soft opcode dispatch and soft if/else relaxation.
# Saturation: with very-large logits the softmax collapses to one-hot
# and the result is bit-exact-equal to the hard-pick value. Lower
# temperature → harder pick; higher temperature → smoother mixture.
#
# Paper reference: the second relaxation in "Hard and Soft Execution" —
# opcode dispatch as a convex combination over handler outputs,
# select(l, V; T) = w' V with w = softmax(l / T) (Eq. "select";
# supplementary second primitive). The executed step uses the saturated
# (hard) form, so dispatch is forward-exact (Theorem 1); the relaxed
# form here is the one analysed in Theorem 2 ("Temperature-limit bound"),
# whose proof bounds the non-winning softmax mass by (K-1)exp(-Delta_min
# / T). Mirrors the hard opcode switch of xitari M6502Low::execute.

@inline function _softmax(logits::AbstractVector{<:Real})
    m = maximum(logits)
    e = exp.(logits .- m)
    return e ./ sum(e)
end

"""
    soft_select(logits, values; temperature=1.0)

Compute `softmax(logits / T) · values`. `logits` and `values` must
have matching leading dimension. If `values` is 2-D, the result is a
vector whose length equals `size(values, 2)`.
"""
function soft_select(logits::AbstractVector{<:Real},
                     values::AbstractVector{<:Real};
                     temperature::Real = 1.0)
    weights = _softmax(Float32.(logits) ./ Float32(temperature))
    return _dot(weights, Float32.(values))
end

function soft_select(logits::AbstractVector{<:Real},
                     values::AbstractMatrix{<:Real};
                     temperature::Real = 1.0)
    weights = _softmax(Float32.(logits) ./ Float32(temperature))
    return Float32.(values)' * weights
end
