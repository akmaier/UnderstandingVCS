"""
    JuTari

Differentiable Julia port of xitari. See PORTING_PLAN.md at the repo root for
scope and milestones.
"""
module JuTari

const VERSION_STRING = "0.0.1"

include("Types.jl")
# CPU module pulls in its own includes (Tables, Addressing, ALU) — keep it
# the sole entry point for the CPU subtree to avoid double-include of Tables.
include("cpu/M6502.jl")
include("diff/Modes.jl")

using .Types: CPUState, initial_cpu_state
using .CPU: step
using .CPU.CPUTables: FLAG_N, FLAG_V, FLAG_U, FLAG_B, FLAG_D, FLAG_I, FLAG_Z, FLAG_C
using .Diff: Mode, HARD, SOFT, current_mode, set_mode!, using_mode

export CPUState, initial_cpu_state, step,
       FLAG_N, FLAG_V, FLAG_U, FLAG_B, FLAG_D, FLAG_I, FLAG_Z, FLAG_C,
       Mode, HARD, SOFT, current_mode, set_mode!, using_mode

end # module
