# Soft branch gate — sigmoid-relaxed conditional PC update.
#
#   g       = sigmoid(α * flag_logit)
#   pc_next = (1 - g) * pc_no_branch + g * pc_branch
#
# With large α the gate saturates to either 0 or 1 → bit-exact-equal to
# the hard branch. With moderate α gradient flows through both PCs and
# through the flag itself.

@inline _sigmoid(x::Real) = 1.0f0 / (1.0f0 + exp(-Float32(x)))

"""
    soft_branch(flag_logit, pc_no_branch, pc_branch; alpha=10.0) -> Float32

Soft-gated PC. `flag_logit` positive → take branch, negative → don't.
Returns a Float32 — round / clamp at the call site for use as an
actual integer PC.
"""
function soft_branch(flag_logit::Real,
                     pc_no_branch::Real,
                     pc_branch::Real;
                     alpha::Real = 10.0)
    g = _sigmoid(Float32(alpha) * Float32(flag_logit))
    return (1.0f0 - g) * Float32(pc_no_branch) + g * Float32(pc_branch)
end
