# NTM-style soft memory read — differentiable access to RAM by a
# continuous-valued address.
#
#   weights = softmax(-|positions - addr| / τ)
#   value   = sum_i weights[i] * RAM[i]
#
# As τ → 0 the result collapses to ordinary indexing.
#
# Paper reference: the *relaxed read* primitive — the distance-softmax
# (temperature-T) read peek_R(r, a; T) = sum_k w_k r_{a+k}, w_k ∝
# exp(-|k| / T) (supplementary "Setup and Notation", fifth primitive,
# Eq. "s-read"). It is the differentiable counterpart of the one-hot
# exact read (RomAsWeights.peek): peek_R → r_a as T → 0 while the
# address carries a nonzero gradient for T > 0 (NTM soft addressing,
# Graves et al. 2014). The executed (SOFT-STE) path uses the exact
# one-hot read; only the fully relaxed variant of Theorem 2 uses this
# read, whose off-target weights decay unconditionally as exp(-1/T)
# (the proof's "Read term"). Mirrors xitari M6502Low::peek / System::peek.

"""
    soft_memory_read(memory, addr_continuous; temperature=1.0) -> Float32

Differentiable read of `memory` at the continuous-valued `addr_continuous`.
With `temperature → 0` the result equals `memory[round(addr) + 1]`
(Julia 1-indexed); with larger temperatures the weights spread for
smoother gradients.
"""
function soft_memory_read(memory::AbstractVector{<:Real},
                          addr_continuous::Real;
                          temperature::Real = 1.0)
    n = length(memory)
    positions = Float32.(0:(n - 1))
    distances = abs.(positions .- Float32(addr_continuous))
    logits = -distances ./ Float32(temperature)
    weights = _softmax(logits)
    return _dot(weights, Float32.(memory))
end
