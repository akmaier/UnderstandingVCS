"""
    JuTari

Differentiable Julia port of xitari. See PORTING_PLAN.md at the repo root for
scope and milestones.
"""
module JuTari

const VERSION_STRING = "0.0.1"

include("Types.jl")
include("cpu/Tables.jl")
include("cpu/M6502.jl")
include("diff/Modes.jl")

using .Types: CPUState, initial_cpu_state
using .CPU: step
using .Diff: Mode, HARD, SOFT, current_mode, set_mode!, using_mode

export CPUState, initial_cpu_state, step,
       Mode, HARD, SOFT, current_mode, set_mode!, using_mode

end # module
