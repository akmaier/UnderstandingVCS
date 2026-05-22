"""
    Diff

Differentiability layer.

Mode toggle (HARD vs SOFT, default HARD): runtime flag controlling
whether the (eventual) SOFT execution path is used in step() — see
PORTING_PLAN.md §6.

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

# `_dot(a, b)` — local helper used by the P7 primitives below. Avoids
# pulling in LinearAlgebra (which would need to be added to Project.toml).
@inline _dot(a::AbstractVector, b::AbstractVector) = sum(a .* b)

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
       SOFT_SUPPORTED_OPCODES,
       soft_render_scanline, soft_render_frame, SOFT_SCREEN_WIDTH

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
# P7b — parallel SOFT-mode `step!()` built on the primitives above.
include("SoftState.jl")
include("SoftStep.jl")
# P7f-a — differentiable TIA playfield renderer.
include("SoftTIA.jl")

end # module
