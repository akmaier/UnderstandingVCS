# NTM-style soft memory read — differentiable access to RAM by a
# continuous-valued address.
#
#   weights = softmax(-|positions - addr| / τ)
#   value   = sum_i weights[i] * RAM[i]
#
# As τ → 0 the result collapses to ordinary indexing.

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
