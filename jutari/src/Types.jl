"""
    Types

Shared state types for JuTari. P0 defines only `CPUState`; `BusState`,
`TIAState`, `RIOTState`, and `ConsoleState` will be added in their respective
phases (see PORTING_PLAN.md §3.2).
"""
module Types

export CPUState, initial_cpu_state

"""
    CPUState

Mutable 6502 register file. `cycles` is `UInt64` to avoid wrap in long runs.
The status register `P` is the packed NV-BDIZC byte; bit 5 is always 1.
"""
mutable struct CPUState
    A::UInt8        # accumulator
    X::UInt8        # X index register
    Y::UInt8        # Y index register
    SP::UInt8       # stack pointer (low byte; stack lives at 0x0100 + SP)
    PC::UInt16      # program counter
    P::UInt8        # processor status register
    cycles::UInt64  # cumulative cycle counter
end

"""
    initial_cpu_state() -> CPUState

6502 state immediately after RESET (PC is later loaded from \$FFFC/\$FFFD
by the bus).
"""
initial_cpu_state() = CPUState(0x00, 0x00, 0x00, 0xFD, 0x0000, 0x34, UInt64(0))

end # module
