"""
    Diff

Differentiability layer — the paper's "new methods" (SOFT execution path).

This module implements the differentiable (SOFT) execution of the VCS
from the paper "A Differentiable Atari VCS": the cartridge ROM as a
weight tensor (RomAsWeights), the RAM as a soft tape, control flow as
gates (SoftBranch / SoftSelect), and the straight-through estimator
(StraightThrough + `_stop_gradient`) that joins the soft path to the
bit-exact hard path. Together they give a soft forward pass bit-exact
equal to the hard one at any finite temperature (Theorem 1, "Exact
forward equivalence") while exposing surrogate gradients where the bit
logic has none (Corollary 1). See paper sections "Hard and Soft
Execution" and "Soft Equals Hard", and the supplementary "Setup and
Notation" for the five primitives.

Mode toggle (HARD vs SOFT, default HARD): runtime flag controlling
whether the (eventual) SOFT execution path is used in step() — see
PORTING_PLAN.md §6. The paper's three modes are HARD (conformance),
SOFT == SOFT-STE (attribution; forward == HARD), and the fully relaxed
FULL mode reached by enabling RelaxConfig's default-off forward hook
(the T -> 0 study of Theorem 2; supplementary Table "The three
execution modes").

P7 primitives:
  RomTensor + peek    differentiable ROM byte access (one-hot dot product)
  soft_select         softmax-weighted mixture over choices
  soft_memory_read    NTM-style positional read of a memory vector
  soft_branch         sigmoid-relaxed conditional PC gate
  straight_through_round / straight_through_clamp
                      hard forward, identity backward (when wired into
                      an autodiff stack — P7b)

Forward-behaviour implementations only in P7; full gradient-stack
integration (Zygote / ChainRulesCore rrules + an end-to-end soft
step() path) is the P7b follow-up.
"""
module Diff

# ChainRulesCore for `_stop_gradient` — used by `_func_do_branch` to
# build the straight-through estimator (forward = pc_hard, backward =
# ∂pc_soft). Without this the cancellation `pc_soft + (pc_hard - pc_soft)`
# collapses at the Float32 level and Zygote sees no gradient through
# the float-flag mirrors (P7c-dx gradient hook).
import ChainRulesCore

# `_dot(a, b)` — local helper used by the P7 primitives below. Avoids
# pulling in LinearAlgebra (which would need to be added to Project.toml).
@inline _dot(a::AbstractVector, b::AbstractVector) = sum(a .* b)

"""
    _stop_gradient(x) -> x

Identity on the forward pass; backward pass propagates no gradient.
Julia's equivalent of `jax.lax.stop_gradient` — the `sg(.)` operator of
the straight-through estimator STE(soft, hard) = soft + sg(hard - soft)
(paper Eq. "ste"). Used inside `_func_do_branch` so the forward PC is
bit-exact (matching the packed `P` decision, Theorem 1) while the
backward gradient runs through the sigmoid-blended soft PC (the
surrogate gradient, Corollary 1).

Implemented via a manual `rrule` because `ChainRulesCore.@ignore_derivatives`
isn't directly composable here.
"""
@inline _stop_gradient(x) = x

ChainRulesCore.rrule(::typeof(_stop_gradient), x) =
    x, _ -> (ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent())

# Extend the Bus module's `peek` (the same multi-method dispatch
# function we use for Vector{UInt8} / BusState reads) with a new method
# for RomTensor. `Bus.peek` is also `Base.peek` after the Bus-import
# step; either qualified path resolves to the same function.
import ..Bus: peek

export Mode, current_mode, set_mode!, using_mode,
       RomTensor, peek_many,
       soft_select, soft_memory_read, soft_branch,
       straight_through_round, straight_through_clamp,
       SoftCPUState, SoftBus, initial_soft_cpu_state, initial_soft_bus,
       soft_step!, soft_run!, soft_rom_peek, soft_ram_peek,
       soft_step, soft_run, update_state, update_bus,    # P7e-x — functional path
       set_relax!, using_relax, relax_config,            # relaxation study (alpha/T)
       _set_ram,                                          # P7e-x — Zygote-friendly RAM write helper
       _with_p, _float_flags_from_p,                      # P7c-dx — float-flag mirror helpers
       SOFT_SUPPORTED_OPCODES,
       soft_render_scanline, soft_render_frame, soft_collision_registers,
       SOFT_SCREEN_WIDTH

@enum Mode HARD SOFT

const _current = Ref{Mode}(HARD)

current_mode() = _current[]

set_mode!(mode::Mode) = (_current[] = mode; nothing)

"""
    using_mode(f, mode::Mode)

Scoped mode switch. Usage:

    using_mode(SOFT) do
        # ... runs in SOFT mode ...
    end
"""
function using_mode(f, mode::Mode)
    previous = _current[]
    _current[] = mode
    try
        return f()
    finally
        _current[] = previous
    end
end

# ----------------------------------------------------------------------
# P7 primitives — see PORTING_PLAN.md §6.
# ----------------------------------------------------------------------
include("RomAsWeights.jl")
include("SoftSelect.jl")
include("SoftMem.jl")
include("SoftBranch.jl")
include("StraightThrough.jl")
# Supplementary relaxation study — default-off forward-relaxation toggle so
# alpha/T affect the SOFT forward (the "effect on pixel exactness" study).
include("RelaxConfig.jl")
# P7b — parallel SOFT-mode `step!()` built on the primitives above.
include("SoftState.jl")
include("SoftStep.jl")
# P7f-a — differentiable TIA playfield renderer.
include("SoftTIA.jl")

end # module
