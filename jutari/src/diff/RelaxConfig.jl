# Relaxation config for the supplementary study "Forward Bit-Exactness Under
# Full Relaxation" / "Effect of the Relaxation Parameters on the Gradient",
# run on the FULL jutari simulator.
#
# Paper reference: this switch selects between the paper's SOFT-STE mode
# and the fully relaxed FULL mode (supplementary Table "The three
# execution modes"). DEFAULT-OFF is SOFT-STE — forward bit-identical to
# HARD (Theorem 1, "Exact forward equivalence"). Turning it ON drops the
# straight-through correction, giving the FULL mode whose forward is
# bit-exact only inside the small-T / large-alpha corner (Theorem 2,
# "Temperature-limit bound"); the cast-margin model and the recommended
# operating point alpha=6, T=0.14 are in the supplementary "Forward
# Bit-Exactness Under Full Relaxation".
#
# DEFAULT-OFF. When `_relax_on[] == false` the soft execution path is
# byte-identical to the validated executed path: one-hot memory reads and the
# straight-through branch (forward = hard PC). The conformance backbone is
# unchanged.
#
# When `_relax_on[] == true` the FORWARD pass uses the *fully relaxed*
# primitives, so alpha and T genuinely perturb forward execution (the empirical
# companion to Theorem 2 / the "exact for small T and large alpha" claim):
#
#   * branch  — `soft_branch` at sharpness `_relax_alpha` with NO straight-
#               through correction. The blended (sigmoid-gated) PC is rounded
#               to the nearest instruction (`straight_through_round`) so the
#               address arithmetic stays integer. With large alpha the gate
#               saturates and the rounded PC equals the hard PC (exact); as
#               alpha shrinks the blend crosses a rounding boundary and the
#               control flow diverges.
#   * reads   — `soft_rom_peek` / `soft_ram_peek` become NTM-style distance-
#               softmax reads at temperature `_relax_temp` (rounded back to a
#               byte). With small T the weight collapses onto the addressed
#               cell (exact); as T grows neighbouring bytes bleed in and the
#               read rounds to a different value.
#
# Both knobs keep the forward integer-valued, so no downstream `Int(...)`
# conversion can throw, while the gradient path stays the soft one.

const _relax_on    = Ref{Bool}(false)
const _relax_alpha = Ref{Float64}(10.0)   # default matches the executed branch
const _relax_temp  = Ref{Float64}(0.1)

"""
    relax_config() -> (; on, alpha, temperature)

Current relaxation settings of the SOFT forward path.
"""
relax_config() = (on = _relax_on[], alpha = _relax_alpha[], temperature = _relax_temp[])

"""
    set_relax!(; on, alpha, temperature) -> nothing

Set (any subset of) the relaxation parameters. Omitted keywords are left
unchanged. `on = false` restores the bit-exact executed path.
"""
function set_relax!(; on::Bool = _relax_on[],
                    alpha::Real = _relax_alpha[],
                    temperature::Real = _relax_temp[])
    _relax_on[]    = on
    _relax_alpha[] = Float64(alpha)
    _relax_temp[]  = Float64(temperature)
    return nothing
end

"""
    using_relax(f; on=true, alpha, temperature)

Scoped relaxation switch — runs `f()` with the given settings and always
restores the previous configuration afterwards.
"""
function using_relax(f; on::Bool = true,
                     alpha::Real = _relax_alpha[],
                     temperature::Real = _relax_temp[])
    saved = (_relax_on[], _relax_alpha[], _relax_temp[])
    set_relax!(; on = on, alpha = alpha, temperature = temperature)
    try
        return f()
    finally
        _relax_on[], _relax_alpha[], _relax_temp[] = saved
    end
end
