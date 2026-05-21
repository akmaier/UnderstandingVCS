"""
    CPU

6502 fetch–decode–execute step.

P0 status: this is a *stub*. `step` reads the opcode at PC, looks up the base
cycle count and addressing mode, and advances PC by 1. It does NOT yet execute
the instruction — that lands in P1 (see PORTING_PLAN.md §5).
"""
module CPU

include("Tables.jl")
using .CPUTables: ADDRESSING_MODE_TABLE, CYCLE_TABLE
using ..Types: CPUState

export step

"""
    step(state::CPUState, memory::Vector{UInt8})

Execute one 6502 instruction.

P0 stub: advances PC by 1 and bumps `cycles` by the base cycle count of the
opcode at PC. Mutates `state` in place; returns `(state, memory)` for
convenience. Real opcode behaviour lands in P1a–P1f.
"""
function step(state::CPUState, memory::Vector{UInt8})
    opcode = memory[Int(state.PC) + 1]
    # TODO(P1): switch on opcode, perform addressing, execute, update flags,
    # add page-cross / branch-taken penalties.
    _ = ADDRESSING_MODE_TABLE[Int(opcode) + 1]
    base_cycles = UInt64(CYCLE_TABLE[Int(opcode) + 1])
    state.PC = (state.PC + 0x0001) & 0xFFFF
    state.cycles += base_cycles
    return state, memory
end

end # module
